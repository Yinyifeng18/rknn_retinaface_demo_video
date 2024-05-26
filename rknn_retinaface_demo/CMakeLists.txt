cmake_minimum_required(VERSION 3.4.1)

project(rknn_retinaface_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_retinaface_demo_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")


set(LIB_ARCH aarch64)


include_directories(${CMAKE_SOURCE_DIR})

# rknn api

set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../runtime/RK356X/${CMAKE_SYSTEM_NAME}/librknn_api)

set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)

include_directories(${RKNN_API_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/../3rdparty)

# OPENCV
set(OPENCV_LIBS opencv_core opencv_flann opencv_videoio opencv_video opencv_highgui opencv_imgcodecs opencv_imgproc)
set(TOOLCHAIN_DIR /opt/atk-dlrk356x-toolchain)
set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/usr/bin/aarch64-buildroot-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/usr/bin/aarch64-buildroot-linux-gnu-c++)
set(SYSROOT ${TOOLCHAIN_DIR}/aarch64-buildroot-linux-gnu//sysroot/usr/include)
set(CMAKE_SYSROOT ${TOOLCHAIN_DIR}/aarch64-buildroot-linux-gnu/sysroot)
include_directories(${SYSROOT})
include_directories(${SYSROOT}/opencv4)

set(CMAKE_INSTALL_RPATH "lib")

#rga
set(RGA_PATH ${CMAKE_SOURCE_DIR}/../3rdparty/rga/RK356X)
set(RGA_LIB ${RGA_PATH}/lib/Linux/${LIB_ARCH}/librga.so)
include_directories( ${RGA_PATH}/include)

# drm
set(DRM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/libdrm)
set(DRM_RT_LIB ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/libdrm/lib/libdrm.so)
include_directories(${DRM_DIR}/include)
include_directories(${DRM_DIR}/include/libdrm)

# mpp
set(MPP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/mpp)
set(MPP_LIBS ${MPP_PATH}/${CMAKE_SYSTEM_NAME}/${LIB_ARCH}/librockchip_mpp.so)
include_directories(${MPP_PATH}/include)

# zlmediakit
set(ZLMEDIAKIT_PATH ${CMAKE_SOURCE_DIR}/../3rdparty/zlmediakit)
include_directories(${ZLMEDIAKIT_PATH}/include)
set(ZLMEDIAKIT_LIBS ${ZLMEDIAKIT_PATH}/${LIB_ARCH}/libmk_api.so)


set(CMAKE_INSTALL_RPATH "lib")

# rknn_retinaface_demo
include_directories( ${CMAKE_SOURCE_DIR}/include)

add_executable(rknn_retinaface_demo
        src/main.cc
        src/retinaface.cc
        src/file_utils.cc
        src/image_drawing.cc
        src/image_utils.cc
)

target_link_libraries(rknn_retinaface_demo
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  ${OPENCV_LIBS}
  dl
)


# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_retinaface_demo_${CMAKE_SYSTEM_NAME})
install(TARGETS rknn_retinaface_demo DESTINATION ./)

install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)
install(DIRECTORY model DESTINATION ./)
