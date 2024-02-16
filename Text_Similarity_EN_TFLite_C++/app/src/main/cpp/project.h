
#include <jni.h>
#include <iostream>
#include <fstream>
#include <android/asset_manager_jni.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_c_api.h"
#include "armnn/delegate/classic/include/armnn_delegate.hpp"
#include "armnn/delegate/common/include/DelegateOptions.hpp"

std::unique_ptr<tflite::Interpreter> interpreter_A;
std::vector<char> flatBuffer_A;  // It must declared as global variable to keep the model alive after the configurations.
const int cpu_threads = 2;
int32_t* input_0;
float* input_1;
std::string file_name_A0 = "Model_GTE.tflite";
std::string file_name_A1 = "libhexagon_nn_skel.so";
std::string cache_path = "/data/user/0/com.example.myapplication/cache/";
const std::string storage_path = "/storage/emulated/0/Android/data/com.example.myapplication/files/";







