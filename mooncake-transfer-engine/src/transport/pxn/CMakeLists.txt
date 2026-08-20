add_library(pxn_transport OBJECT pxn_core.cpp)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_sources(pxn_transport PRIVATE pxn_registry.cpp pxn_staging.cpp)
  target_link_libraries(pxn_transport PRIVATE glog::glog)

  if(USE_CUDA)
    target_sources(pxn_transport PRIVATE pxn_cuda_staging.cpp
                                         pxn_rdma_transport.cpp)
    target_include_directories(
      pxn_transport PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(pxn_transport PRIVATE CUDA::cudart)
  endif()
endif()
