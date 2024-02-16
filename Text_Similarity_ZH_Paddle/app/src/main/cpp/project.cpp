#include "project.h"

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_10(JNIEnv *env, jclass clazz,
                                                            jobject asset_manager,
                                                            jboolean use_gpu) {
    std::vector<char> fileBuffer;
    size_t fileSize;
    if (asset_manager != nullptr) {
        AAssetManager* mgr = AAssetManager_fromJava(env,asset_manager);
        AAsset* asset = AAssetManager_open(mgr,file_name_A.c_str(),AASSET_MODE_BUFFER);
        fileSize = AAsset_getLength(asset);
        fileBuffer.resize(fileSize);
        AAsset_read(asset,fileBuffer.data(),fileSize);
    } else {
        std::ifstream model_file(storage_path + file_name_A,std::ios::binary | std::ios::ate);
        if (!model_file.is_open()) {
            return JNI_FALSE;
        }
        fileSize = model_file.tellg();
        model_file.seekg(0,std::ios::beg);
        fileBuffer.resize(fileSize);
        if (!model_file.read(fileBuffer.data(), fileSize)) {
            return JNI_FALSE;
        }
        model_file.close();
    }
    paddle::lite_api::MobileConfig config;
    config.set_threads(2);  // If the inference takes a very long time, the issue may not lie with the demo code. There seems to be a bug in the interaction between your device and the Paddle-Lite framework. Maybe try another device.
    config.set_power_mode(paddle::lite_api::LITE_POWER_HIGH);
    config.SetArmL3CacheSize();
    config.set_model_from_buffer(fileBuffer.data(), fileSize);
    if (use_gpu) {
        config.set_opencl_binary_path_name(cache_path,"lite_opencl_kernel.bin");
        // opencl tune option
        // CL_TUNE_NONE: 0
        // CL_TUNE_RAPID: 1
        // CL_TUNE_NORMAL: 2
        // CL_TUNE_EXHAUSTIVE: 3
        config.set_opencl_tune(paddle::lite_api::CL_TUNE_EXHAUSTIVE,cache_path,"lite_opencl_tuned.bin");
        config.set_opencl_precision(paddle::lite_api::CL_PRECISION_AUTO);
        // opencl precision option
        // CL_PRECISION_AUTO: 0, first fp16 if valid, default
        // CL_PRECISION_FP32: 1, force fp32
        // CL_PRECISION_FP16: 2, force fp16
    }
    model_A = paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(config);
    input_0 = std::move(model_A->GetInput(0));
    input_0 -> Resize({1, max_token_limit_GTE});
    input_1 = std::move(model_A->GetInput(1));
    input_1 -> Resize({1, max_token_limit_GTE});
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_myapplication_MainActivity_Run_1Text_1Embedding(JNIEnv *env, jclass clazz,
                                                                jintArray token_index,
                                                                jint words_count,
                                                                jint model_hidden_size,
                                                                jint max_token_limit) {
    jint *input_ids = env->GetIntArrayElements(token_index,nullptr);
    std::vector<float> attention_masks(max_token_limit,-999999999.f);
    std::fill(attention_masks.begin(),attention_masks.begin() + words_count,1.f);
    std::move(input_ids,input_ids + max_token_limit,input_0->mutable_data<int32_t>());
    std::move(attention_masks.begin(),attention_masks.end(),input_1->mutable_data<float>());
    env->ReleaseIntArrayElements(token_index,input_ids,0);
    model_A->Run();
    std::unique_ptr<const paddle::lite_api::Tensor> output_0(std::move(model_A->GetOutput(0)));
    jfloatArray final_results = env->NewFloatArray(model_hidden_size);
    env->SetFloatArrayRegion(final_results, 0,model_hidden_size,output_0->data<float>());
    return final_results;
}
