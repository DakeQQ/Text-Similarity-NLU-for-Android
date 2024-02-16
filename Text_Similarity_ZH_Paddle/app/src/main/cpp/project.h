
#include <jni.h>
#include <android/asset_manager_jni.h>
#include <iostream>
#include <fstream>
#include "paddle_api.h"

std::shared_ptr<paddle::lite_api::PaddlePredictor> model_A;
const int max_token_limit_GTE = 25;
const std::string file_name_A = "Model_GTE.nb";
const std::string storage_path = "/storage/emulated/0/Android/data/com.example.myapplication/files/";
const std::string cache_path = "/data/user/0/com.example.myapplication/cache/";
std::unique_ptr<paddle::lite_api::Tensor> input_0;
std::unique_ptr<paddle::lite_api::Tensor> input_1;







