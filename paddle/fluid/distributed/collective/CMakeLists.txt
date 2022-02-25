cc_library(processgroup SRCS ProcessGroup.cc DEPS pten pten_api eager_api)
cc_library(eager_reducer SRCS reducer.cc DEPS pten pten_api eager_api processgroup)

if(WITH_NCCL)
    cc_library(processgroup_nccl SRCS ProcessGroupNCCL.cc DEPS place cuda_stream enforce collective_helper device_context pten pten_api eager_api)
endif()
