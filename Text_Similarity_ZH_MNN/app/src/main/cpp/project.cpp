#include "project.h"

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_10(JNIEnv *env, jobject clazz,
                                                            jobject asset_manager,
                                                            jboolean use_int8, jboolean use_fp16,
                                                            jboolean use_bf16, jboolean use_gpu,
                                                            jboolean use_nnapi,
                                                            jboolean convert_from_tf) {  // If the model was converted from TensorFlow, it may encounter issues with NCHW/NHWC format conversion. Please refer to the MNN official documentation for more information.
    MNN::ScheduleConfig config;
    config.numThread = 2;
    config.type = MNN_FORWARD_CPU_EXTENSION;
    if (use_gpu) {  // It usually slower than CPU on small size model.
        config.type = MNN_FORWARD_OPENCL;  // options: MNN_FORWARD_OPENCL, MNN_FORWARD_OPENGL, MNN_FORWARD_VULKAN
        config.mode = MNN_GPU_TUNING_HEAVY | MNN_GPU_MEMORY_BUFFER;  // Qualcomm = MNN_GPU_MEMORY_IMAGE; Mali = MNN_GPU_MEMORY_BUFFER
    } else if (use_nnapi) { // It needs to add the app into /vendor/etc/nnapi_extensions_app_allowlist
        config.type = MNN_FORWARD_NN;
    }
    config.backupType = MNN_FORWARD_CPU;
    MNN::BackendConfig backendConfig;
    if (use_int8 | use_fp16 | use_bf16) {
        backendConfig.memory = MNN::BackendConfig::Memory_Low;
    } else {
        backendConfig.memory = MNN::BackendConfig::Memory_High; // It is not recommended to use Memory_High with non-FP32 model, it usually get slower.
    }
    backendConfig.power = MNN::BackendConfig::Power_Low;  // In my personal opinions, Power_High means using big-core first. Power_Low means using small-core first. Power_Normal means using both big & small cores equally.
    if (use_fp16) {
        backendConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_Low;
    }
    else if (use_bf16) {
        backendConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_Low_BF16;
    }
    else {
        backendConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_Normal;  // It is not recommended to use Precision_High unless you are going to Moon.
    }
    config.backendConfig = &backendConfig;
    auto runtime = MNN::Interpreter::createRuntime({config});
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
    interpreter_A = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(fileBuffer.data(),fileSize));
    interpreter_A->setSessionMode(MNN::Interpreter::SessionMode::Session_Codegen_Enable);
    session_A = interpreter_A->createSession(config, runtime);
    std::vector<std::vector<int>> input_dim(input_names.size());
    for (int i = 0; i < input_names.size(); i++) {
        inputTensor[i] = interpreter_A->getSessionInput(session_A,input_names[i]);
        input_dim[i] = inputTensor[i]->shape();
        interpreter_A->resizeTensor(inputTensor[i],input_dim[i]);
    }
    interpreter_A->resizeSession(session_A,1);
    interpreter_A->updateSessionToModel(session_A);
    interpreter_A->releaseModel();  // If the model does not resize the input shape anymore, it can be releaseModel() to reduce memory usage.
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
    std::move(input_ids,input_ids + max_token_limit,inputTensor[0]->host<int32_t>());
    std::move(attention_masks.begin(),attention_masks.end(),inputTensor[1]->host<float>());
    env->ReleaseIntArrayElements(token_index,input_ids,0);
    interpreter_A->runSession(session_A);
    MNN::Tensor *output = interpreter_A->getSessionOutput(session_A,output_names[0]);
    jfloatArray final_results = env->NewFloatArray(model_hidden_size);
    env->SetFloatArrayRegion(final_results, 0,model_hidden_size,output->host<float>());
    return final_results;
}
