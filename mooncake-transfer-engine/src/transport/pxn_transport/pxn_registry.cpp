// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "transport/pxn_transport/pxn_registry.h"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <climits>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>

#if defined(__linux__)
#include <linux/fs.h>
#include <sys/random.h>
#include <sys/syscall.h>
#endif

namespace mooncake {
namespace pxn {
namespace {

constexpr std::string_view kEntryPrefix = "peer.";
constexpr std::string_view kEntrySuffix = ".reg";
constexpr std::string_view kTemporaryPrefix = "tmp.";

Status systemError(std::string_view operation, int error) {
    return Status::Memory(std::string(operation) + ": " + std::strerror(error));
}

class FileLock {
   public:
    explicit FileLock(int fd) : fd_(fd) {}

    Status lock() {
        while (flock(fd_, LOCK_EX) != 0) {
            if (errno == EINTR) continue;
            return systemError("flock LOCK_EX failed", errno);
        }
        locked_ = true;
        return Status::OK();
    }

    ~FileLock() {
        if (!locked_) return;
        while (flock(fd_, LOCK_UN) != 0 && errno == EINTR) {
        }
    }

   private:
    int fd_;
    bool locked_ = false;
};

int duplicateFd(int fd) {
    int duplicate;
    do {
        duplicate = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    } while (duplicate < 0 && errno == EINTR);
    return duplicate;
}

bool hasPrefixAndSuffix(std::string_view name, std::string_view prefix,
                        std::string_view suffix) {
    return name.size() > prefix.size() + suffix.size() &&
           name.starts_with(prefix) && name.ends_with(suffix);
}

Status validateDirectory(int fd) {
    struct stat info{};
    if (fstat(fd, &info) != 0) return systemError("fstat failed", errno);
    if (!S_ISDIR(info.st_mode) || info.st_uid != geteuid() ||
        (info.st_mode & 07777) != 0700) {
        return Status::InvalidArgument("invalid PXN registry directory");
    }
    return Status::OK();
}

Status validateRegularFile(int fd, std::optional<off_t> expected_size,
                           std::string_view error_message) {
    struct stat info{};
    if (fstat(fd, &info) != 0) return systemError("fstat failed", errno);
    if (!S_ISREG(info.st_mode) || info.st_uid != geteuid() ||
        (info.st_mode & 07777) != 0600 || info.st_nlink != 1 ||
        (expected_size.has_value() && info.st_size != *expected_size)) {
        return Status::InvalidArgument(std::string(error_message));
    }
    return Status::OK();
}

Status validateControlFile(int fd) {
    return validateRegularFile(
        fd, static_cast<off_t>(sizeof(ControlBlock)),
        "invalid PXN registry entry");
}

Status validateLockFile(int fd) {
    return validateRegularFile(fd, std::nullopt, "invalid PXN registry lock");
}

Status validateRegistryHeader(const RegistryHeader& header) {
    if (header.magic != kRegistryMagic ||
        header.abi_version != kRegistryAbiVersion ||
        header.header_bytes != sizeof(RegistryHeader) ||
        header.mapping_bytes != sizeof(ControlBlock) ||
        header.owner.uid != geteuid() || header.owner.pid <= 0 ||
        header.owner.start_ticks == 0 || header.epoch == 0 ||
        header.lane_count != kLaneCount ||
        header.slots_per_lane != kSlotsPerLane ||
        header.slot_size != kSlotSize ||
        header.arena_size != kRequiredArenaSize || header.rail_bytes == 0 ||
        header.rail_bytes > kMaxRailNameLength ||
        header.ipc_handle_bytes != kCudaIpcHandleSize || header.reserved != 0 ||
        !std::all_of(std::begin(header.padding), std::end(header.padding),
                     [](uint8_t byte) { return byte == 0; })) {
        return Status::InvalidArgument("incompatible PXN registry entry");
    }
    return Status::OK();
}

std::optional<bool> defaultProcessProbe(const ProcessIdentity& identity) {
    char path[64];
    const int length = std::snprintf(path, sizeof(path), "/proc/%d/stat",
                                     static_cast<int>(identity.pid));
    if (length <= 0 || static_cast<size_t>(length) >= sizeof(path)) {
        return std::nullopt;
    }

    int fd;
    do {
        fd = open(path, O_RDONLY | O_CLOEXEC);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) {
        if (errno == ENOENT || errno == ESRCH) return false;
        return std::nullopt;
    }

    std::string contents;
    char buffer[4096];
    while (true) {
        ssize_t count;
        do {
            count = read(fd, buffer, sizeof(buffer));
        } while (count < 0 && errno == EINTR);
        if (count < 0) {
            close(fd);
            return std::nullopt;
        }
        if (count == 0) break;
        contents.append(buffer, static_cast<size_t>(count));
        if (contents.size() > 16384) {
            close(fd);
            return std::nullopt;
        }
    }
    close(fd);

    uint64_t start_ticks = 0;
    char state = 0;
    if (!parseProcessStat(contents, start_ticks, state).ok()) {
        return std::nullopt;
    }
    if (state == 'Z' || state == 'X' || state == 'x') return false;
    return identity.uid == geteuid() && identity.start_ticks == start_ticks;
}

uint64_t generateEpoch() {
    uint64_t epoch = 0;
#if defined(__linux__)
    size_t offset = 0;
    while (offset < sizeof(epoch)) {
        ssize_t count = getrandom(reinterpret_cast<char*>(&epoch) + offset,
                                  sizeof(epoch) - offset, 0);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) break;
        offset += static_cast<size_t>(count);
    }
    if (offset == sizeof(epoch) && epoch != 0) return epoch;
#endif
    std::random_device random;
    do {
        epoch = (static_cast<uint64_t>(random()) << 32) ^ random();
    } while (epoch == 0);
    return epoch;
}

std::string makeEntryName(const ProcessIdentity& identity, uint64_t epoch,
                          bool temporary) {
    char name[128];
    std::snprintf(name, sizeof(name), "%s%d.%llu.%llu%s",
                  temporary ? "tmp." : "peer.", static_cast<int>(identity.pid),
                  static_cast<unsigned long long>(identity.start_ticks),
                  static_cast<unsigned long long>(epoch),
                  temporary ? "" : ".reg");
    return name;
}

Status mapHeader(int fd, RegistryHeader*& header) {
    void* mapped =
        mmap(nullptr, sizeof(RegistryHeader), PROT_READ, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) return systemError("mmap failed", errno);
    header = static_cast<RegistryHeader*>(mapped);
    return Status::OK();
}

Status renameNoReplace(int directory_fd, const std::string& source,
                       const std::string& destination) {
#if defined(__linux__) && defined(SYS_renameat2)
    int result;
    do {
        result = static_cast<int>(
            syscall(SYS_renameat2, directory_fd, source.c_str(), directory_fd,
                    destination.c_str(), RENAME_NOREPLACE));
    } while (result != 0 && errno == EINTR);
    if (result == 0) return Status::OK();
    return systemError("publish PXN registry entry failed", errno);
#else
    (void)directory_fd;
    (void)source;
    (void)destination;
    return Status::NotImplemented("PXN requires renameat2 RENAME_NOREPLACE");
#endif
}

Status scanEntriesLocked(int directory_fd, const ProcessProbe& process_probe,
                         std::vector<RegistryEntry>& entries) {
    entries.clear();
    int scan_fd;
    do {
        scan_fd = openat(directory_fd, ".",
                         O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    } while (scan_fd < 0 && errno == EINTR);
    if (scan_fd < 0) {
        return systemError("open registry directory scan failed", errno);
    }
    DIR* directory = fdopendir(scan_fd);
    if (directory == nullptr) {
        const int error = errno;
        close(scan_fd);
        return systemError("fdopendir failed", error);
    }

    Status result = Status::OK();
    while (true) {
        errno = 0;
        dirent* item = readdir(directory);
        if (item == nullptr) {
            if (errno != 0) {
                result =
                    systemError("read PXN registry directory failed", errno);
            }
            break;
        }
        std::string_view name(item->d_name);
        const bool final_entry =
            hasPrefixAndSuffix(name, kEntryPrefix, kEntrySuffix);
        const bool temporary_entry = name.starts_with(kTemporaryPrefix);
        if (!final_entry && !temporary_entry) continue;

        int fd;
        do {
            fd = openat(directory_fd, item->d_name,
                        O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
        } while (fd < 0 && errno == EINTR);
        if (fd < 0) {
            if (errno == ENOENT) continue;
            result = systemError("open PXN registry entry failed", errno);
            break;
        }

        auto status = validateControlFile(fd);
        if (!status.ok()) {
            close(fd);
            if (temporary_entry) continue;
            result = std::move(status);
            break;
        }

        RegistryHeader* header = nullptr;
        status = mapHeader(fd, header);
        if (!status.ok()) {
            close(fd);
            result = std::move(status);
            break;
        }

        status = validateRegistryHeader(*header);
        if (!status.ok()) {
            munmap(header, sizeof(RegistryHeader));
            close(fd);
            if (temporary_entry) continue;
            result = std::move(status);
            break;
        }

        const ProcessIdentity owner = header->owner;
        auto alive = process_probe(owner);
        if (!alive.has_value()) {
            munmap(header, sizeof(RegistryHeader));
            close(fd);
            result = Status::InvalidArgument(
                "cannot verify PXN registry process identity");
            break;
        }
        if (!*alive) {
            munmap(header, sizeof(RegistryHeader));
            close(fd);
            if (unlinkat(directory_fd, item->d_name, 0) != 0 &&
                errno != ENOENT) {
                result = systemError("unlink stale PXN entry failed", errno);
                break;
            }
            continue;
        }

        if (temporary_entry ||
            loadRegistryState(*header) !=
                static_cast<uint32_t>(RegistryState::kReady)) {
            munmap(header, sizeof(RegistryHeader));
            close(fd);
            continue;
        }

        RegistryEntry entry;
        entry.identity = owner;
        entry.epoch = header->epoch;
        entry.rail.assign(header->rail, header->rail_bytes);
        entry.arena_handle = header->arena_handle;
        entry.file_name.assign(name);
        entries.push_back(std::move(entry));
        munmap(header, sizeof(RegistryHeader));
        close(fd);
    }

    closedir(directory);
    return result;
}

void clearLane(LaneControl& lane) {
    __atomic_store_n(&lane.header.state,
                     static_cast<uint32_t>(LaneState::kClaiming),
                     __ATOMIC_RELEASE);
    lane.header.arena_attached = 0;
    lane.header.sender = {};
    lane.header.sender_epoch = 0;
    lane.header.head = 0;
    lane.header.doorbell = 0;
    lane.header.completed = 0;
    lane.header.padding = 0;
    std::memset(lane.descriptors, 0, sizeof(lane.descriptors));
    std::memset(lane.completions, 0, sizeof(lane.completions));
    __atomic_store_n(&lane.header.state,
                     static_cast<uint32_t>(LaneState::kFree), __ATOMIC_RELEASE);
}

Status validateMappedPeer(const ControlBlock& control,
                          const RegistryEntry& entry,
                          bool allow_stopping = false) {
    const uint32_t state = loadRegistryState(control.header);
    if (state != static_cast<uint32_t>(RegistryState::kReady) &&
        (!allow_stopping ||
         state != static_cast<uint32_t>(RegistryState::kStopping))) {
        return Status::InvalidArgument("PXN registry peer is not ready");
    }
    auto status = validateRegistryHeader(control.header);
    if (!status.ok()) return status;
    if (control.header.owner != entry.identity ||
        control.header.epoch != entry.epoch ||
        std::string_view(control.header.rail, control.header.rail_bytes) !=
            entry.rail ||
        control.header.arena_handle != entry.arena_handle) {
        return Status::InvalidArgument("PXN registry peer changed");
    }
    return Status::OK();
}

}  // namespace

bool isValidGroupId(std::string_view group_id) {
    if (group_id.empty() || group_id.size() > 64 || group_id == "." ||
        group_id == "..") {
        return false;
    }
    for (char value : group_id) {
        const auto character = static_cast<unsigned char>(value);
        if ((character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9') || character == '_' ||
            character == '-' || character == '.') {
            continue;
        }
        return false;
    }
    return true;
}

Status parseProcessStat(std::string_view contents, uint64_t& start_ticks,
                        char& state) {
    const size_t end = contents.rfind(')');
    if (end == std::string_view::npos || end + 2 >= contents.size() ||
        contents[end + 1] != ' ') {
        return Status::InvalidArgument("invalid /proc process stat");
    }

    std::istringstream fields(std::string(contents.substr(end + 2)));
    if (!(fields >> state)) {
        return Status::InvalidArgument("invalid /proc process state");
    }

    std::string value;
    for (int field = 4; field <= 22; ++field) {
        if (!(fields >> value)) {
            return Status::InvalidArgument("truncated /proc process stat");
        }
        if (field != 22) continue;
        uint64_t parsed = 0;
        auto [pointer, error] =
            std::from_chars(value.data(), value.data() + value.size(), parsed);
        if (error != std::errc() || pointer != value.data() + value.size() ||
            parsed == 0) {
            return Status::InvalidArgument("invalid process start ticks");
        }
        start_ticks = parsed;
    }
    return Status::OK();
}

Status readProcessIdentity(int32_t pid, ProcessIdentity& identity) {
    if (pid <= 0) return Status::InvalidArgument("invalid process id");
    char path[64];
    const int length = std::snprintf(path, sizeof(path), "/proc/%d/stat", pid);
    if (length <= 0 || static_cast<size_t>(length) >= sizeof(path)) {
        return Status::InvalidArgument("invalid process id");
    }
    int fd;
    do {
        fd = open(path, O_RDONLY | O_CLOEXEC);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) return systemError("open process stat failed", errno);

    std::string contents;
    char buffer[4096];
    while (true) {
        ssize_t count;
        do {
            count = read(fd, buffer, sizeof(buffer));
        } while (count < 0 && errno == EINTR);
        if (count < 0) {
            const int error = errno;
            close(fd);
            return systemError("read process stat failed", error);
        }
        if (count == 0) break;
        contents.append(buffer, static_cast<size_t>(count));
        if (contents.size() > 16384) {
            close(fd);
            return Status::InvalidArgument("oversized process stat");
        }
    }
    close(fd);

    uint64_t start_ticks = 0;
    char state = 0;
    auto status = parseProcessStat(contents, start_ticks, state);
    if (!status.ok()) return status;
    if (state == 'Z' || state == 'X' || state == 'x') {
        return Status::InvalidArgument("process is not live");
    }
    identity = {static_cast<uint32_t>(geteuid()), pid, start_ticks};
    return Status::OK();
}

uint32_t loadRegistryState(const RegistryHeader& header) {
    return __atomic_load_n(&header.state, __ATOMIC_ACQUIRE);
}

uint32_t loadLaneState(const LaneHeader& header) {
    return __atomic_load_n(&header.state, __ATOMIC_ACQUIRE);
}

RegistryRegistration::RegistryRegistration(
    int directory_fd, int lock_fd, int fd, std::string temporary_name,
    std::string final_name, ControlBlock* control, ProcessIdentity identity,
    ProcessProbe process_probe, std::shared_ptr<std::mutex> group_mutex)
    : directory_fd_(directory_fd),
      lock_fd_(lock_fd),
      fd_(fd),
      temporary_name_(std::move(temporary_name)),
      final_name_(std::move(final_name)),
      control_(control),
      identity_(identity),
      process_probe_(std::move(process_probe)),
      group_mutex_(std::move(group_mutex)) {}

RegistryRegistration::~RegistryRegistration() {
    (void)unpublish();
    if (control_ != nullptr) munmap(control_, sizeof(ControlBlock));
    if (fd_ >= 0) close(fd_);
    if (!temporary_name_.empty() && directory_fd_ >= 0) {
        unlinkat(directory_fd_, temporary_name_.c_str(), 0);
    }
    if (lock_fd_ >= 0) close(lock_fd_);
    if (directory_fd_ >= 0) close(directory_fd_);
}

Status RegistryRegistration::publish() {
    if (published_) return Status::OK();
    if (control_ == nullptr || directory_fd_ < 0 || lock_fd_ < 0 ||
        temporary_name_.empty()) {
        return Status::InvalidArgument("invalid PXN registration");
    }

    std::lock_guard<std::mutex> local_lock(*group_mutex_);
    FileLock lock(lock_fd_);
    auto status = lock.lock();
    if (!status.ok()) return status;

    std::vector<RegistryEntry> entries;
    status = scanEntriesLocked(directory_fd_, process_probe_, entries);
    if (!status.ok()) return status;
    for (const auto& entry : entries) {
        if (entry.identity == identity_) {
            return Status::InvalidArgument("PXN process is already published");
        }
    }
    if (entries.size() >= kLaneCount + 1) {
        return Status::TooManyRequests("PXN group has more than eight ranks");
    }

    __atomic_store_n(&control_->header.state,
                     static_cast<uint32_t>(RegistryState::kReady),
                     __ATOMIC_RELEASE);
    status = renameNoReplace(directory_fd_, temporary_name_, final_name_);
    if (!status.ok()) {
        __atomic_store_n(&control_->header.state,
                         static_cast<uint32_t>(RegistryState::kInitializing),
                         __ATOMIC_RELEASE);
        return status;
    }
    temporary_name_.clear();
    published_ = true;
    return Status::OK();
}

Status RegistryRegistration::unpublish() {
    if (!published_ || control_ == nullptr) return Status::OK();
    std::lock_guard<std::mutex> local_lock(*group_mutex_);
    FileLock group_lock(lock_fd_);
    auto status = group_lock.lock();
    if (!status.ok()) return status;
    std::lock_guard<std::mutex> entry_local_lock(entry_mutex_);
    FileLock entry_lock(fd_);
    status = entry_lock.lock();
    if (!status.ok()) return status;
    __atomic_store_n(&control_->header.state,
                     static_cast<uint32_t>(RegistryState::kStopping),
                     __ATOMIC_RELEASE);
    struct stat owned_info{};
    struct stat path_info{};
    if (fstat(fd_, &owned_info) != 0) {
        return systemError("stat owned PXN registry entry failed", errno);
    }
    if (fstatat(directory_fd_, final_name_.c_str(), &path_info,
                AT_SYMLINK_NOFOLLOW) != 0) {
        if (errno != ENOENT) {
            return systemError("stat published PXN registry entry failed",
                               errno);
        }
        published_ = false;
        return Status::OK();
    }
    if (owned_info.st_dev != path_info.st_dev ||
        owned_info.st_ino != path_info.st_ino) {
        return Status::InvalidArgument("PXN registry entry was replaced");
    }
    if (unlinkat(directory_fd_, final_name_.c_str(), 0) == 0 ||
        errno == ENOENT) {
        published_ = false;
        return Status::OK();
    }
    return systemError("unpublish PXN registry entry failed", errno);
}

Status RegistryRegistration::activeAttachments(size_t& count) {
    if (control_ == nullptr || fd_ < 0) {
        return Status::InvalidArgument("invalid PXN registration");
    }
    std::lock_guard<std::mutex> local_lock(entry_mutex_);
    FileLock lock(fd_);
    auto status = lock.lock();
    if (!status.ok()) return status;

    count = 0;
    for (auto& lane : control_->lanes) {
        if (loadLaneState(lane.header) !=
                static_cast<uint32_t>(LaneState::kReady) ||
            __atomic_load_n(&lane.header.arena_attached, __ATOMIC_ACQUIRE) ==
                0) {
            continue;
        }
        auto alive = process_probe_(lane.header.sender);
        if (alive.has_value() && !*alive) {
            clearLane(lane);
            continue;
        }
        ++count;
    }
    return Status::OK();
}

PeerMapping::PeerMapping(int fd, RegistryEntry entry, ControlBlock* control,
                         ProcessProbe process_probe)
    : fd_(fd),
      entry_(std::move(entry)),
      control_(control),
      process_probe_(std::move(process_probe)) {}

PeerMapping::~PeerMapping() {
    (void)detachArena();
    if (control_ != nullptr) munmap(control_, sizeof(ControlBlock));
    if (fd_ >= 0) close(fd_);
}

Status PeerMapping::claimLane(const ProcessIdentity& sender,
                              uint64_t sender_epoch, size_t& lane_index) {
    if (sender.uid != geteuid() || sender.pid <= 0 || sender.start_ticks == 0 ||
        sender_epoch == 0) {
        return Status::InvalidArgument("invalid PXN lane owner");
    }
    if (sender == entry_.identity) {
        return Status::InvalidArgument("PXN self lane is not allowed");
    }
    auto sender_alive = process_probe_(sender);
    if (!sender_alive.has_value() || !*sender_alive) {
        return Status::InvalidArgument("PXN lane owner is not live");
    }
    std::lock_guard<std::mutex> local_lock(mutex_);
    if (lane_index_ < kLaneCount &&
        (sender_ != sender || sender_epoch_ != sender_epoch)) {
        return Status::InvalidArgument("PXN mapping already owns a lane");
    }
    FileLock lock(fd_);
    auto status = lock.lock();
    if (!status.ok()) return status;
    status = validateMappedPeer(*control_, entry_);
    if (!status.ok()) return status;

    std::optional<size_t> available;
    for (size_t index = 0; index < kLaneCount; ++index) {
        auto& lane = control_->lanes[index];
        const uint32_t state = loadLaneState(lane.header);
        if (state == static_cast<uint32_t>(LaneState::kFree)) {
            if (!available.has_value()) available = index;
            continue;
        }
        if (state == static_cast<uint32_t>(LaneState::kClaiming)) {
            clearLane(lane);
            if (!available.has_value()) available = index;
            continue;
        }
        if (state != static_cast<uint32_t>(LaneState::kReady)) {
            return Status::InvalidArgument("PXN lane is not ready");
        }
        if (lane.header.sender == sender) {
            if (lane.header.sender_epoch == sender_epoch) {
                sender_ = sender;
                sender_epoch_ = sender_epoch;
                lane_index_ = index;
                lane_index = index;
                return Status::OK();
            }
            if (__atomic_load_n(&lane.header.arena_attached,
                                __ATOMIC_ACQUIRE) != 0) {
                return Status::BatchBusy(
                    "PXN lane from the previous sender epoch is still mapped");
            }
            clearLane(lane);
            if (!available.has_value()) available = index;
            continue;
        }
        auto alive = process_probe_(lane.header.sender);
        if (!alive.has_value()) {
            return Status::InvalidArgument(
                "cannot verify PXN lane process identity");
        }
        if (!*alive) {
            clearLane(lane);
            if (!available.has_value()) available = index;
        }
    }

    if (!available.has_value()) {
        return Status::TooManyRequests("PXN relay has no free inbound lane");
    }

    lane_index = *available;
    auto& lane = control_->lanes[lane_index];
    clearLane(lane);
    __atomic_store_n(&lane.header.state,
                     static_cast<uint32_t>(LaneState::kClaiming),
                     __ATOMIC_RELEASE);
    lane.header.sender = sender;
    lane.header.sender_epoch = sender_epoch;
    __atomic_store_n(&lane.header.state,
                     static_cast<uint32_t>(LaneState::kReady),
                     __ATOMIC_RELEASE);
    sender_ = sender;
    sender_epoch_ = sender_epoch;
    lane_index_ = lane_index;
    return Status::OK();
}

Status PeerMapping::releaseLane(const ProcessIdentity& sender,
                                uint64_t sender_epoch, size_t lane_index) {
    if (lane_index >= kLaneCount) {
        return Status::InvalidArgument("invalid PXN lane index");
    }
    std::lock_guard<std::mutex> local_lock(mutex_);
    FileLock lock(fd_);
    auto status = lock.lock();
    if (!status.ok()) return status;
    status = validateMappedPeer(*control_, entry_, true);
    if (!status.ok()) return status;

    auto& lane = control_->lanes[lane_index];
    if (loadLaneState(lane.header) !=
            static_cast<uint32_t>(LaneState::kReady) ||
        lane.header.sender != sender ||
        lane.header.sender_epoch != sender_epoch) {
        return Status::InvalidArgument("PXN lane owner does not match");
    }
    if (__atomic_load_n(&lane.header.arena_attached, __ATOMIC_ACQUIRE) != 0) {
        return Status::BatchBusy("PXN lane still has an arena mapping");
    }
    clearLane(lane);
    if (sender_ == sender && sender_epoch_ == sender_epoch &&
        lane_index_ == lane_index) {
        sender_ = {};
        sender_epoch_ = 0;
        lane_index_ = kLaneCount;
    }
    return Status::OK();
}

Status PeerMapping::attachArena() {
    std::lock_guard<std::mutex> local_lock(mutex_);
    if (arena_attached_) return Status::OK();
    if (lane_index_ >= kLaneCount || sender_epoch_ == 0) {
        return Status::InvalidArgument("PXN lane is not claimed");
    }
    FileLock lock(fd_);
    auto status = lock.lock();
    if (!status.ok()) return status;
    status = validateMappedPeer(*control_, entry_);
    if (!status.ok()) return status;

    auto& lane = control_->lanes[lane_index_];
    if (loadLaneState(lane.header) !=
            static_cast<uint32_t>(LaneState::kReady) ||
        lane.header.sender != sender_ ||
        lane.header.sender_epoch != sender_epoch_) {
        return Status::InvalidArgument("PXN lane owner changed");
    }
    if (__atomic_load_n(&lane.header.arena_attached, __ATOMIC_ACQUIRE) != 0) {
        return Status::BatchBusy("PXN arena is already mapped for this lane");
    }
    __atomic_store_n(&lane.header.arena_attached, uint32_t{1},
                     __ATOMIC_RELEASE);
    arena_attached_ = true;
    return Status::OK();
}

Status PeerMapping::detachArena() {
    std::lock_guard<std::mutex> local_lock(mutex_);
    if (!arena_attached_) return Status::OK();
    if (control_ == nullptr || fd_ < 0) {
        return Status::InvalidArgument("invalid PXN peer mapping");
    }
    FileLock lock(fd_);
    auto status = lock.lock();
    if (!status.ok()) return status;
    if (lane_index_ < kLaneCount) {
        auto& lane = control_->lanes[lane_index_];
        if (lane.header.sender == sender_ &&
            lane.header.sender_epoch == sender_epoch_) {
            __atomic_store_n(&lane.header.arena_attached, uint32_t{0},
                             __ATOMIC_RELEASE);
        }
    }
    arena_attached_ = false;
    return Status::OK();
}

Registry::Registry(int root_fd, int directory_fd, int lock_fd,
                   ProcessIdentity identity, uint64_t epoch,
                   ProcessProbe process_probe,
                   std::shared_ptr<std::mutex> group_mutex)
    : root_fd_(root_fd),
      directory_fd_(directory_fd),
      lock_fd_(lock_fd),
      identity_(identity),
      epoch_(epoch),
      process_probe_(std::move(process_probe)),
      group_mutex_(std::move(group_mutex)) {}

Registry::~Registry() {
    if (lock_fd_ >= 0) close(lock_fd_);
    if (directory_fd_ >= 0) close(directory_fd_);
    if (root_fd_ >= 0) close(root_fd_);
}

Status Registry::Open(RegistryOptions options,
                      std::unique_ptr<Registry>& registry) {
    if (!isValidGroupId(options.group_id)) {
        return Status::InvalidArgument("invalid PXN registry group");
    }

    ProcessIdentity identity{};
    if (options.identity.has_value()) {
        identity = *options.identity;
        if (identity.uid != geteuid() || identity.pid <= 0 ||
            identity.start_ticks == 0) {
            return Status::InvalidArgument("invalid PXN process identity");
        }
    } else {
        auto status =
            readProcessIdentity(static_cast<int32_t>(getpid()), identity);
        if (!status.ok()) return status;
    }

    int root_fd;
    do {
        root_fd = open(options.root_directory.c_str(),
                       O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    } while (root_fd < 0 && errno == EINTR);
    if (root_fd < 0) return systemError("open PXN registry root failed", errno);

    const std::string directory_name =
        "mooncake-pxn-" + std::to_string(geteuid()) + "-" + options.group_id;
    if (mkdirat(root_fd, directory_name.c_str(), 0700) != 0 &&
        errno != EEXIST) {
        const int error = errno;
        close(root_fd);
        return systemError("create PXN registry directory failed", error);
    }

    int directory_fd;
    do {
        directory_fd = openat(root_fd, directory_name.c_str(),
                              O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    } while (directory_fd < 0 && errno == EINTR);
    if (directory_fd < 0) {
        const int error = errno;
        close(root_fd);
        return systemError("open PXN registry directory failed", error);
    }
    auto status = validateDirectory(directory_fd);
    if (!status.ok()) {
        close(directory_fd);
        close(root_fd);
        return status;
    }

    int lock_fd;
    do {
        lock_fd = openat(directory_fd, ".lock",
                         O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
    } while (lock_fd < 0 && errno == EINTR);
    if (lock_fd < 0) {
        const int error = errno;
        close(directory_fd);
        close(root_fd);
        return systemError("open PXN registry lock failed", error);
    }

    auto lock_status = validateLockFile(lock_fd);
    if (!lock_status.ok()) {
        close(lock_fd);
        close(directory_fd);
        close(root_fd);
        return lock_status;
    }

    ProcessProbe process_probe = std::move(options.process_probe);
    if (!process_probe) process_probe = defaultProcessProbe;
    const uint64_t epoch = options.epoch == 0 ? generateEpoch() : options.epoch;
    auto group_mutex = std::make_shared<std::mutex>();
    registry.reset(new Registry(root_fd, directory_fd, lock_fd, identity, epoch,
                                std::move(process_probe),
                                std::move(group_mutex)));
    return Status::OK();
}

Status Registry::createLocal(
    std::string_view rail, const CudaIpcHandle& arena_handle,
    std::unique_ptr<RegistryRegistration>& registration) {
    if (rail.empty() || rail.size() > kMaxRailNameLength) {
        return Status::InvalidArgument("invalid PXN rail name");
    }

    const std::string temporary_name = makeEntryName(identity_, epoch_, true);
    const std::string final_name = makeEntryName(identity_, epoch_, false);
    int fd;
    do {
        fd = openat(directory_fd_, temporary_name.c_str(),
                    O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) return systemError("create PXN registry entry failed", errno);

    if (ftruncate(fd, static_cast<off_t>(sizeof(ControlBlock))) != 0) {
        const int error = errno;
        close(fd);
        unlinkat(directory_fd_, temporary_name.c_str(), 0);
        return systemError("resize PXN registry entry failed", error);
    }
    auto status = validateControlFile(fd);
    if (!status.ok()) {
        close(fd);
        unlinkat(directory_fd_, temporary_name.c_str(), 0);
        return status;
    }

    void* mapped = mmap(nullptr, sizeof(ControlBlock), PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        const int error = errno;
        close(fd);
        unlinkat(directory_fd_, temporary_name.c_str(), 0);
        return systemError("map PXN registry entry failed", error);
    }

    auto* control = static_cast<ControlBlock*>(mapped);
    std::memset(control, 0, sizeof(*control));
    control->header.magic = kRegistryMagic;
    control->header.abi_version = kRegistryAbiVersion;
    control->header.header_bytes = sizeof(RegistryHeader);
    control->header.mapping_bytes = sizeof(ControlBlock);
    control->header.owner = identity_;
    control->header.epoch = epoch_;
    control->header.lane_count = kLaneCount;
    control->header.slots_per_lane = kSlotsPerLane;
    control->header.slot_size = kSlotSize;
    control->header.arena_size = kRequiredArenaSize;
    control->header.rail_bytes = static_cast<uint32_t>(rail.size());
    control->header.ipc_handle_bytes = kCudaIpcHandleSize;
    std::memcpy(control->header.rail, rail.data(), rail.size());
    control->header.arena_handle = arena_handle;

    int registration_directory_fd = duplicateFd(directory_fd_);
    int registration_lock_fd = duplicateFd(lock_fd_);
    if (registration_directory_fd < 0 || registration_lock_fd < 0) {
        const int error = errno;
        if (registration_directory_fd >= 0) close(registration_directory_fd);
        if (registration_lock_fd >= 0) close(registration_lock_fd);
        munmap(control, sizeof(ControlBlock));
        close(fd);
        unlinkat(directory_fd_, temporary_name.c_str(), 0);
        return systemError("duplicate PXN registry fd failed", error);
    }

    registration.reset(new RegistryRegistration(
        registration_directory_fd, registration_lock_fd, fd, temporary_name,
        final_name, control, identity_, process_probe_, group_mutex_));
    return Status::OK();
}

Status Registry::discover(std::vector<RegistryEntry>& entries) {
    std::lock_guard<std::mutex> local_lock(*group_mutex_);
    FileLock lock(lock_fd_);
    auto status = lock.lock();
    if (!status.ok()) return status;

    std::vector<RegistryEntry> all_entries;
    status = scanEntriesLocked(directory_fd_, process_probe_, all_entries);
    if (!status.ok()) return status;

    std::vector<RegistryEntry> peers;
    for (auto& entry : all_entries) {
        if (entry.identity != identity_) peers.push_back(std::move(entry));
    }
    if (peers.size() > kLaneCount) {
        return Status::TooManyRequests("PXN group has more than eight ranks");
    }
    entries = std::move(peers);
    return Status::OK();
}

Status Registry::mapPeer(const RegistryEntry& entry,
                         std::unique_ptr<PeerMapping>& mapping) {
    if (entry.identity.uid != geteuid() || entry.identity.pid <= 0 ||
        entry.identity.start_ticks == 0 || entry.epoch == 0 ||
        entry.file_name != makeEntryName(entry.identity, entry.epoch, false)) {
        return Status::InvalidArgument("invalid PXN peer entry");
    }
    std::lock_guard<std::mutex> local_lock(*group_mutex_);
    FileLock lock(lock_fd_);
    auto status = lock.lock();
    if (!status.ok()) return status;

    int fd;
    do {
        fd = openat(directory_fd_, entry.file_name.c_str(),
                    O_RDWR | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) return systemError("open PXN peer entry failed", errno);
    status = validateControlFile(fd);
    if (!status.ok()) {
        close(fd);
        return status;
    }

    void* mapped = mmap(nullptr, sizeof(ControlBlock), PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        const int error = errno;
        close(fd);
        return systemError("map PXN peer entry failed", error);
    }
    auto* control = static_cast<ControlBlock*>(mapped);
    status = validateMappedPeer(*control, entry);
    if (!status.ok()) {
        munmap(control, sizeof(ControlBlock));
        close(fd);
        return status;
    }
    auto alive = process_probe_(entry.identity);
    if (!alive.has_value() || !*alive) {
        munmap(control, sizeof(ControlBlock));
        close(fd);
        return Status::InvalidArgument("PXN peer process is not live");
    }

    mapping.reset(new PeerMapping(fd, entry, control, process_probe_));
    return Status::OK();
}

}  // namespace pxn
}  // namespace mooncake
