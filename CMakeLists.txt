cmake_minimum_required(VERSION 3.14)

project(panopticFCNEngine)
set(CMAKE_CXX_STANDARD 17)
find_package(CUDA)
############################################################################################
# Dependencies
SET(CUDA_INC_DIR /usr/local/cuda/include)
SET(CUDNN_INC_DIR /usr/local/cuda/include)
SET(TENSORRT_INC_DIR /opt/tensorrt/include)
SET(TRT_ENGINE_INC_DIR include)
SET(PLUGINS_INC_DIR plugins)
SET(INC_DIR ${CUDA_INC_DIR} ${CUDNN_INC_DIR} ${TRT_ENGINE_INC_DIR} ${TRT_ENGINE_INC_DIR} ${PLUGINS_INC_DIR})

SET(CUDA_LIB_DIR /usr/local/cuda/lib64)
SET(CUDNN_LIB_DIR /usr/local/cuda/lib64)
SET(TENSORRT_LIB_DIR /opt/tensorrt/lib)
SET(PLUGINS_LIB_DIR plugins)
SET(LIB_DIR ${CUDA_LIB_DIR} ${CUDNN_LIB_DIR} ${TENSORRT_LIB_DIR} ${TRT_ENGINE_LIB_DIR} ${PLUGINS_LIB_DIR})

set(TRT_LIBS nvinfer nvinfer_plugin )
set(CUDNN_LIBs cudnn )
set(CUDA_LIBS cublas cudart curand cufft nppig nvinfer nvonnxparser)

set(LINK_LIBS ${TRT_LIBS} ${CUDNN_LIBs}  ${CUDA_LIBS})

set(OpenCV_INCLUDE_DIRS /usr/local/include/opencv4)
set(OpenCV_LIBS_DIR /usr/local/lib)
set(OpenCV_LIBS opencv_core opencv_dnn opencv_imgproc opencv_imgcodecs opencv_highgui)

include_directories(${INC_DIR} ${OpenCV_INCLUDE_DIRS})
link_directories(${LIB_DIR} ${OpenCV_LIBS_DIR})
link_libraries(${LINK_LIBS} ${OpenCV_LIBS})


#plugin操作
set(PLUGINS_NAME myplugins)
cuda_add_library(${PLUGINS_NAME} SHARED plugins/SeafoodPlugin.cu)
target_link_libraries(${PLUGINS_NAME} nvinfer cudart)

#添加文件
add_executable(${PROJECT_NAME} appPanopticFCNPlugin.cpp)
#dlopen
add_definitions(-O2 -pthread)
#链接库
target_link_libraries(${PROJECT_NAME} nvinfer_plugin myplugins)


