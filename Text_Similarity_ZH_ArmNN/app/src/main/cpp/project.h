
#include <jni.h>
#include <android/asset_manager_jni.h>
#include "armnn/INetwork.hpp"
#include "armnn/IRuntime.hpp"
#include "armnnOnnxParser/IOnnxParser.hpp"
#include "armnnTfLiteParser/ITfLiteParser.hpp"
#include "armnn/Optional.hpp"
#include "armnn/Types.hpp"

std::string file_name_A0 = "Model_GTE.tflite";
std::string file_name_A1 = "model_cached.bin";
std::string cache_path = "/data/user/0/com.example.myapplication/cache/";
std::vector<std::string> input_names = {"input_ids", "attention_mask"};  // For ONNX Parser, If use TFLite, it is not necessary.
std::vector<std::string> output_names = {"hidden_state"};  // For ONNX Parser, If use TFLite, it is not necessary.
const int cpu_threads = 4;
const int max_token_limit_GTE = 25;
const int model_hidden_size = 512;
std::vector<int32_t> inputs_0(max_token_limit_GTE, 0);
std::vector<float> inputs_1(max_token_limit_GTE, 0.f);
std::vector<float> outputs_0(model_hidden_size, 0.f);
armnn::NetworkId network_id_A;
armnn::IRuntime::CreationOptions runtime_options;
armnn::IRuntimePtr runtime_A = armnn::IRuntime::Create(runtime_options);
armnn::InputTensors in_tensor;
armnn::OutputTensors out_tensor;
bool enable_profiling = false;  // Enable it if you need.
bool memory_protected = false;  // Enable it if you need.
