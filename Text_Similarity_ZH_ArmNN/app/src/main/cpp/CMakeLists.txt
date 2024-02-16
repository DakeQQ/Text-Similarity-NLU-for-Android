project("myapplication")
set(ANDROID_ABI "arm64-v8a")
cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_BUILD_TYPE Release)

include_directories(${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI})

add_compile_options(
#                -fexceptions # debug mode
        -fno-exceptions # release mode
        -flto  # release mode
        -fno-rtti  # release mode
        -Ofast
        -ffast-math
        -ffp-model=fast
        -ffp-contract=fast
        -march=armv8.7-a+fp16fml+dotprod+i8mm
        -mtune=cortex-a76
        -mcpu=cortex-a76+cortex-a55  # Replace with your chip.
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
        omp
        SHARED
        IMPORTED)
set_target_properties(omp PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libomp.so)

add_library(
        armnn
        SHARED
        IMPORTED)
set_target_properties(armnn PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libarmnn.so)

add_library(
        armnnOnnxParser
        SHARED
        IMPORTED)
set_target_properties(armnnOnnxParser PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libarmnnOnnxParser.so)

add_library(
        armnnTfLiteParser
        SHARED
        IMPORTED)
set_target_properties(armnnTfLiteParser PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libarmnnTfLiteParser.so)

add_library(
        armnnDelegate
        SHARED
        IMPORTED)
set_target_properties(armnnDelegate PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libarmnnDelegate.so)

add_library(
        armnnSerializer
        SHARED
        IMPORTED)
set_target_properties(armnnSerializer PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libarmnnSerializer.so)

add_library(
        protobuf
        SHARED
        IMPORTED)
set_target_properties(protobuf PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libprotobuf.so)

find_library(
        log
        android
)

target_link_libraries(${CMAKE_PROJECT_NAME}
        android
        armnn
        protobuf
        armnnOnnxParser
        armnnTfLiteParser
        armnnDelegate
        armnnSerializer
        omp
        log)
