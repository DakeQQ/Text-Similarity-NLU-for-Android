project("myapplication")
set(ANDROID_ABI "arm64-v8a")
cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_BUILD_TYPE Release)

include_directories(${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI})
include_directories(${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/cxx/include)
include_directories(${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/opencl/cl_kernel)

add_compile_options(
#                                -fexceptions # debug mode
        -fno-exceptions # release mode
        -flto  # release mode
        -fno-rtti  # release mode
        -Ofast
        -ffast-math
        -ffp-model=fast
        -ffp-contract=fast
        -march=armv8.7-a+fp16fml+dotprod+i8mm
        -mtune=cortex-a76
        -mcpu=cortex-a76+cortex-a55  # Replace with your chip
        -fPIC
        -fopenmp
        -fomit-frame-pointer
        -fvectorize
        -fno-stack-protector
        -finline-functions
        -ffunction-sections
        -fdata-sections
        -fstrict-aliasing
        -fvisibility=hidden
        -fvisibility-inlines-hidden
        -funroll-loops
)


add_library(${CMAKE_PROJECT_NAME} SHARED project.cpp)

add_library(
        nnadapter
        SHARED
        IMPORTED)
set_target_properties(nnadapter PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/cxx/lib/libnnadapter.so)

add_library(
        android_nnapi
        SHARED
        IMPORTED)
set_target_properties(android_nnapi PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/cxx/lib/libandroid_nnapi.so)

add_library(
        paddle_light_api_shared
        SHARED
        IMPORTED)
set_target_properties(paddle_light_api_shared PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/cxx/lib/libpaddle_light_api_shared.so)

add_library(
        omp
        SHARED
        IMPORTED)
set_target_properties(omp PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libomp.so)

find_library(
        log
        android
)

target_link_libraries(
        myapplication
        android
        omp
        paddle_light_api_shared
        nnadapter
        android_nnapi
        log)
