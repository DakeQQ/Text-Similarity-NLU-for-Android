project("myapplication")
set(ANDROID_ABI "arm64-v8a")
cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_BUILD_TYPE Release)

include_directories(${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI})

add_compile_options(
        #   -fexceptions # debug mode
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
        MNN
        SHARED
        IMPORTED)
set_target_properties(MNN PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libMNN.so)

add_library(
        MNN_CL
        SHARED
        IMPORTED)
set_target_properties(MNN_CL PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libMNN_CL.so)

add_library(
        MNN_Express
        SHARED
        IMPORTED)
set_target_properties(MNN_Express PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libMNN_Express.so)

add_library(
        MNN_Vulkan
        SHARED
        IMPORTED)
set_target_properties(MNN_Vulkan PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libMNN_Vulkan.so)

# Setting for Mali-GPU (OpenCL), if MNN_CL has no permission for access the /system/vendor/lib64

add_library(
        OpenCL
        SHARED
        IMPORTED)
set_target_properties(OpenCL PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libOpenCL.so)

add_library(
        omp
        SHARED
        IMPORTED)
set_target_properties(omp PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libomp.so)

add_library(
        base
        SHARED
        IMPORTED)
set_target_properties(base PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libbase.so)

add_library(
        c++
        SHARED
        IMPORTED)
set_target_properties(c++ PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libc++.so)

add_library(
        c
        SHARED
        IMPORTED)
set_target_properties(c PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libc.so)

add_library(
        cutils
        SHARED
        IMPORTED)
set_target_properties(cutils PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libcutils.so)


add_library(
        dl
        SHARED
        IMPORTED)
set_target_properties(dl PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libdl.so)

add_library(
        log
        SHARED
        IMPORTED)
set_target_properties(log PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/liblog.so)

add_library(
        m
        SHARED
        IMPORTED)
set_target_properties(m PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libm.so)

add_library(
        utils
        SHARED
        IMPORTED)
set_target_properties(utils PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libutils.so)

add_library(
        ld-android
        SHARED
        IMPORTED)
set_target_properties(ld-android  PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/ld-android.so)

add_library(
        dl_android
        SHARED
        IMPORTED)
set_target_properties(dl_android  PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libdl_android.so)

add_library(
        vndksupport
        SHARED
        IMPORTED)
set_target_properties(vndksupport  PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libvndksupport.so)

add_library(
        processgroup
        SHARED
        IMPORTED)
set_target_properties(processgroup  PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libprocessgroup.so)

add_library(
        cgrouprc
        SHARED
        IMPORTED)
set_target_properties(cgrouprc  PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libcgrouprc.so)

add_library(
        GLES_mali
        SHARED
        IMPORTED)
set_target_properties(GLES_mali  PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libGLES_mali.so)

add_library(
        ion
        SHARED
        IMPORTED)
set_target_properties(ion  PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libion.so)

add_library(
        common_1_0
        SHARED
        IMPORTED)
set_target_properties(common_1_0  PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libcommon_1_0.so)

add_library(
        nativewindow
        SHARED
        IMPORTED)
set_target_properties(nativewindow  PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libnativewindow.so)

add_library(
        c_secshared
        SHARED
        IMPORTED)
set_target_properties(c_secshared  PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libc_secshared.so)

add_library(
        utilscallstack
        SHARED
        IMPORTED)
set_target_properties(utilscallstack  PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libutilscallstack.so)

add_library(
        hidlbase
        SHARED
        IMPORTED)
set_target_properties(hidlbase  PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libhidlbase.so)

add_library(
        hidltransport
        SHARED
        IMPORTED)
set_target_properties(hidltransport  PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libhidltransport.so)

add_library(
        hwbinder
        SHARED
        IMPORTED)
set_target_properties(hwbinder  PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libhwbinder.so)

add_library(
        backtrace
        SHARED
        IMPORTED)
set_target_properties(backtrace  PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libbacktrace.so)

add_library(
        common_1_1
        SHARED
        IMPORTED)
set_target_properties(common_1_1  PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libcommon_1_1.so)

add_library(
        binderthreadstate
        SHARED
        IMPORTED)
set_target_properties(binderthreadstate  PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libbinderthreadstate.so)

add_library(
        unwindstack
        SHARED
        IMPORTED)
set_target_properties(unwindstack  PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libunwindstack.so)

add_library(
        dexfile_support
        SHARED
        IMPORTED)
set_target_properties(dexfile_support  PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libdexfile_support.so)

add_library(
        lzma
        SHARED
        IMPORTED)
set_target_properties(lzma  PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/liblzma.so)

find_library(
        log
        android
)

target_link_libraries(
        myapplication
        android
        MNN
        MNN_Express
        MNN_Vulkan
        # MNN_CL  # If use it, you need the permission for accessing the /system/vendor/lib64 or use the following *.so to enable OpenCL backend.
        c_secshared
        omp
        lzma
        dexfile_support
        binderthreadstate
        unwindstack
        hidlbase
        hidltransport
        hwbinder
        backtrace
        common_1_1  # android.hardware.graphics.common@1.1.so
        utilscallstack
        c_secshared
        nativewindow
        common_1_0  # android.hardware.graphics.common@1.0.so
        ion
        GLES_mali
        processgroup
        vndksupport
        dl_android
        ld-android
        utils
        m
        dl
        cutils
        c
        c++
        base
        OpenCL
        log)
