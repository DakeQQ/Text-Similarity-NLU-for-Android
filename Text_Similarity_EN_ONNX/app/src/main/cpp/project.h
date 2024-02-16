
#include <jni.h>
#include "onnxruntime_cxx_api.h"
#include "nnapi_provider_factory.h"
#include <android/asset_manager_jni.h>
#include <iostream>
#include <fstream>

const OrtApi* ort_runtime_A;
OrtSession* session_model_A;
std::vector<const char*> input_names_A;
std::vector<const char*> output_names_A;
std::vector<std::vector<std::int64_t>> input_dims_A;
std::vector<std::vector<std::int64_t>> output_dims_A;
std::vector<ONNXTensorElementDataType> input_types_A;
std::vector<ONNXTensorElementDataType> output_types_A;
std::vector<OrtValue*> input_tensors_A;
std::vector<OrtValue*> output_tensors_A;
const std::string file_name_A = "Model_GTE.ort";
const std::string storage_path = "/storage/emulated/0/Android/data/com.example.myapplication/files/";
const char* qnn_htp_so = "/data/user/0/com.example.myapplication/cache/libQnnHtp.so";  //  If use (std::string cache_path + "libQnnHtp.so").c_str() instead, it will open failed.
const char* qnn_cpu_so = "/data/user/0/com.example.myapplication/cache/libQnnCpu.so";  //  If use (std::string cache_path + "libQnnCpu.so").c_str() instead, it will open failed.
const int max_token_limit_GTE = 25;  // Same variable in the MainActivity.java, please modify it at the same time.
const int in0_buffer_size = max_token_limit_GTE * sizeof(int32_t);
const int in1_buffer_size = max_token_limit_GTE * sizeof(float);








