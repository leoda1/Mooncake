#!/bin/bash
# ib_monitoring.sh — 实时监控 IB/RoCE 设备带宽，支持 bond 子口展开
#
# 用法: PERF_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7 ./ib_monitoring.sh [interval_sec]
# PERF_IB_HCA=mlx5_000,mlx5_001,mlx5_002,mlx5_003,mlx5_004,mlx5_005,mlx5_006,mlx5_007,mlx5_008,mlx5_009,mlx5_010,mlx5_011,mlx5_012,mlx5_013,mlx5_014,mlx5_015 ./ib_monitoring.sh 5
#
# 环境变量:
#   PERF_IB_HCA   逗号分隔的设备列表（不设则监控全部）
# 参数:
#   interval_sec  刷新间隔秒数，默认 1

interval=${1:-1}

# ---------- 设备列表 ----------
if [ -n "$PERF_IB_HCA" ]; then
    IFS=',' read -ra _tmp <<< "$PERF_IB_HCA"
    IB_DEVS=()
    for d in "${_tmp[@]}"; do IB_DEVS+=("${d// /}"); done
else
    mapfile -t IB_DEVS < <(ls /sys/class/infiniband/)
fi

declare -A old_recv_bytes old_xmit_bytes old_out_of_seq
declare -A old_slave_rx_bytes old_slave_tx_bytes
declare -A dev_slaves

# ---------- 工具函数 ----------

get_netdev() {
    local ib_dev=$1 nd
    if command -v ibdev2netdev &>/dev/null; then
        nd=$(ibdev2netdev 2>/dev/null | awk -v d="$ib_dev" '$1==d{print $5; exit}')
        [ -n "$nd" ] && { echo "$nd"; return; }
    fi
    local net_dir="/sys/class/infiniband/$ib_dev/device/net"
    [ -d "$net_dir" ] && ls "$net_dir" 2>/dev/null | head -1
}

# 一次 ethtool -S 调用读取 slave 字节计数器，优先 *_phy 字段
read_slave_stats() {
    local iface=$1
    eval "$(ethtool -S "$iface" 2>/dev/null | awk '
        /rx_bytes_phy:/    { rb=$2 }
        /tx_bytes_phy:/    { tb=$2 }
        /rx_bytes:/        && !rb { rb=$2 }
        /tx_bytes:/        && !tb { tb=$2 }
        END { printf "_RXB=%d _TXB=%d\n", rb+0, tb+0 }
    ')"
}

now_ms() {
    local t; t=$(date +%s%N 2>/dev/null)
    [[ "$t" == *N ]] && t=$(( $(date +%s) * 1000000000 ))
    echo $(( t / 1000000 ))
}

# ---------- 格式常量 ----------
ROW="%-28s %12s %12s %12s\n"
SLV="  %s %-23s %12s %12s\n"
SEP=$(printf '%.0s─' {1..68})

print_header() {
    printf "%s\n" "$SEP"
    printf "$ROW" "Device" "recv_Gb/s" "xmit_Gb/s" "out_of_seq"
    printf "%s\n" "$SEP"
}

# --- 纯 bash 定点运算（不依赖 bc）：返回值 = Gb/s * 100（保留 2 位小数）---
# IB counter (4-byte word) -> Gb/s*100:  delta * 4B * 8bit / 1024^3 / dt_s
ib_gbps_x100() {
    local delta=$1 dt_ms=$2
    [ "$dt_ms" -le 0 ] && { echo 0; return; }
    echo $(( delta * 3200000 / 1073741824 / dt_ms ))
}
# ethtool bytes -> Gb/s*100:  delta * 8bit / 1024^3 / dt_s
eth_gbps_x100() {
    local delta=$1 dt_ms=$2
    [ "$dt_ms" -le 0 ] && { echo 0; return; }
    echo $(( delta * 800000 / 1073741824 / dt_ms ))
}
# 把 Gb/s*100 格式化为两位小数
fmt100() { printf "%d.%02d" $(($1/100)) $(($1%100)); }

print_ib_row() {
    local label=$1 ib_dev=$2 dt_ms=$3
    local cdir="/sys/class/infiniband/$ib_dev/ports/1/counters"
    local hwdir="/sys/class/infiniband/$ib_dev/ports/1/hw_counters"
    local nr nt noos
    nr=$(cat "$cdir/port_rcv_data")
    nt=$(cat "$cdir/port_xmit_data")
    noos=$(cat "$hwdir/out_of_sequence" 2>/dev/null || echo 0)
    printf "$ROW" "$label" \
        "$(fmt100 $(ib_gbps_x100 $((nr - ${old_recv_bytes[$ib_dev]:-nr})) $dt_ms))" \
        "$(fmt100 $(ib_gbps_x100 $((nt - ${old_xmit_bytes[$ib_dev]:-nt})) $dt_ms))" \
        "$((noos - ${old_out_of_seq[$ib_dev]:-0}))"
    old_recv_bytes[$ib_dev]=$nr
    old_xmit_bytes[$ib_dev]=$nt
    old_out_of_seq[$ib_dev]=$noos
}

print_slave_row() {
    local branch=$1 slave=$2 dt_ms=$3
    read_slave_stats "$slave"   # sets _RXB _TXB
    printf "$SLV" "$branch" "$slave" \
        "$(fmt100 $(eth_gbps_x100 $((_RXB - ${old_slave_rx_bytes[$slave]:-_RXB})) $dt_ms))" \
        "$(fmt100 $(eth_gbps_x100 $((_TXB - ${old_slave_tx_bytes[$slave]:-_TXB})) $dt_ms))"
    old_slave_rx_bytes[$slave]=$_RXB
    old_slave_tx_bytes[$slave]=$_TXB
}

# ---------- 初始化 ----------
for ib_dev in "${IB_DEVS[@]}"; do
    cdir="/sys/class/infiniband/$ib_dev/ports/1/counters"
    hwdir="/sys/class/infiniband/$ib_dev/ports/1/hw_counters"
    old_recv_bytes[$ib_dev]=$(cat "$cdir/port_rcv_data")
    old_xmit_bytes[$ib_dev]=$(cat "$cdir/port_xmit_data")
    old_out_of_seq[$ib_dev]=$(cat "$hwdir/out_of_sequence" 2>/dev/null || echo 0)

    netdev=$(get_netdev "$ib_dev")
    slaves_file="/sys/class/net/$netdev/bonding/slaves"
    if [ -n "$netdev" ] && [ -f "$slaves_file" ]; then
        slaves=$(cat "$slaves_file")
        dev_slaves[$ib_dev]="$slaves"
        for slave in $slaves; do
            read_slave_stats "$slave"
            old_slave_rx_bytes[$slave]=$_RXB
            old_slave_tx_bytes[$slave]=$_TXB
        done
    fi
done
old_ts=$(now_ms)

# ---------- 循环 ----------
while true; do
    sleep "$interval"
    now_ts=$(now_ms)
    dt_ms=$((now_ts - old_ts))
    [ "$dt_ms" -le 0 ] && dt_ms=1
    old_ts=$now_ts

    clear
    printf "IB Monitor  interval=%ds  %s\n\n" "$interval" "$(date '+%Y-%m-%d %H:%M:%S')"

    # standalone 设备
    first_bond=0
    for ib_dev in "${IB_DEVS[@]}"; do
        [ -n "${dev_slaves[$ib_dev]}" ] && continue
        [ $first_bond -eq 0 ] && { print_header; first_bond=1; }
        print_ib_row "$ib_dev" "$ib_dev" "$dt_ms"
    done

    # bond 设备
    first_bond_group=0
    for ib_dev in "${IB_DEVS[@]}"; do
        [ -z "${dev_slaves[$ib_dev]}" ] && continue

        if [ $first_bond_group -eq 0 ]; then
            [ $first_bond -eq 1 ] && printf "%s\n" "$SEP" || print_header
            first_bond_group=1
        else
            printf "\n"
        fi

        print_ib_row "$ib_dev" "$ib_dev" "$dt_ms"

        slave_arr=(${dev_slaves[$ib_dev]})
        last=$(( ${#slave_arr[@]} - 1 ))
        for i in "${!slave_arr[@]}"; do
            [ $i -eq $last ] && branch="└─" || branch="├─"
            print_slave_row "$branch" "${slave_arr[$i]}" "$dt_ms"
        done
    done

    printf "%s\n" "$SEP"
done