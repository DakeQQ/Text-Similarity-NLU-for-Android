
#include <jni.h>
#include <android/asset_manager_jni.h>
#include "net.h"
#include "cpu.h"

const char* file_name_A0 = "Model_GTE.param";
const char* file_name_A1 = "Model_GTE.bin";
std::vector<const char*> input_names = {"in0", "in1"};
std::vector<const char*> output_names = {"out0"};
const int max_token_limit_GTE = 25;
ncnn::Net options;
ncnn::Mat in0;
ncnn::Mat in1;
ncnn::Mat out0;








