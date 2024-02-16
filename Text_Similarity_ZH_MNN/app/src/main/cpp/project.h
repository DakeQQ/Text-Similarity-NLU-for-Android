
#include <jni.h>
#include <android/asset_manager_jni.h>
#include <iostream>
#include <fstream>
#include "MNN/Interpreter.hpp"

std::shared_ptr<MNN::Interpreter> interpreter_A;
MNN::Session* session_A;
std::string file_name_A = "Model_GTE.mnn";
std::vector<const char*> input_names = {"input_ids", "attention_mask"};
std::vector<const char*> output_names = {"hidden_state"}; // If only one input or output, it can use 'nullptr' instead.
std::vector<MNN::Tensor*> inputTensor(input_names.size());
const std::string storage_path = "/storage/emulated/0/Android/data/com.example.myapplication/files/";








