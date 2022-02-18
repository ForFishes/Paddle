cc_library(nccl_utils SRCS NCCLUtils.cc DEPS pten_api enforce collective_helper device_context)
cc_library(processgroup SRCS ProcessGroup.cc DEPS pten_api eager_api)
cc_library(processgroup_nccl SRCS ProcessGroupNCCL.cc DEPS nccl_utils pten_api eager_api place cuda_stream enforce collective_helper device_context device_event_base pten)

# add_subdirectory(tests)
