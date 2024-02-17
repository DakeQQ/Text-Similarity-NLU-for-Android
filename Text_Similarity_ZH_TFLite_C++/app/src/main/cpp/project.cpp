#include "project.h"

bool report_error_armnn_delegate(const char* errorMessage) {  // You an edit it by yourself.
    return true;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_10(JNIEnv *env, jobject clazz,
                                                            jobject asset_manager,
                                                            jboolean use_uint8,
                                                            jboolean use_int8,
                                                            jboolean use_fp16,
                                                            jboolean use_xnnpack,
                                                            jboolean use_arm,
                                                            jboolean use_gpu,
                                                            jboolean use_nnapi,
                                                            jboolean use_dsp_npu) {
    size_t fileSize;
    if (asset_manager != nullptr) {
        AAssetManager* mgr = AAssetManager_fromJava(env,asset_manager);
        AAsset* asset = AAssetManager_open(mgr,file_name_A0.c_str(),AASSET_MODE_BUFFER);
        fileSize = AAsset_getLength(asset);
        flatBuffer_A.resize(fileSize);
        AAsset_read(asset,flatBuffer_A.data(),fileSize);
    } else {
        std::ifstream model_file(storage_path + file_name_A0,std::ios::binary | std::ios::ate);
        if (!model_file.is_open()) {
            return JNI_FALSE;
        }
        fileSize = model_file.tellg();
        model_file.seekg(0,std::ios::beg);
        flatBuffer_A.resize(fileSize);
        if (!model_file.read(flatBuffer_A.data(),fileSize)) {
            return JNI_FALSE;
        }
        model_file.close();
    }
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromBuffer(flatBuffer_A.data(),fileSize);
    if (model == nullptr) {
        return JNI_FALSE;
    }
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model,resolver)(&interpreter_A);
    if (interpreter_A == nullptr) {
        return JNI_FALSE;
    }
    interpreter_A->SetNumThreads(cpu_threads);
    interpreter_A->AllocateTensors();
    input_0 = interpreter_A->typed_input_tensor<int32_t>(0);
    input_1 = interpreter_A->typed_input_tensor<float>(1);
    if (use_xnnpack) {
        TfLiteXNNPackDelegateOptions backend_options = TfLiteXNNPackDelegateOptionsDefault();
        backend_options.num_threads = cpu_threads;
        if (use_uint8) {
            backend_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_QU8;
        } else if (use_int8) {
            backend_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_QS8;
        } else if (use_fp16) {
            backend_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;  // This GTE model can not use this option.
        }
        backend_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_DYNAMIC_FULLY_CONNECTED;
        backend_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_VARIABLE_OPERATORS;
        backend_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_TRANSIENT_INDIRECTION_BUFFER;  // May reduce memory usage in conv2D...
        backend_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;  // Enable the nightly-build ops
        interpreter_A->ModifyGraphWithDelegate(TfLiteXNNPackDelegateCreate(&backend_options));
    }
    if (use_arm) {
        std::vector<const char*> option_keys = {};
        std::vector<const char*> option_values = {};

        option_keys.push_back("number-of-threads");  // 0 for auto set.
        option_values.push_back((std::to_string(cpu_threads)).c_str());

        option_keys.push_back("reduce-fp32-to-fp16");
        if (use_fp16) {
            option_values.push_back("true");
        } else {
            option_values.push_back("false");
        }

        option_keys.push_back("save-cached-network");
        option_values.push_back("true");

        option_keys.push_back("cached-network-filepath");  // If (save-cached-network==false) {It will load the cached file from this path.}
        option_values.push_back((cache_path + "model_cached.bin").c_str());

        option_keys.push_back("memory-import");
        option_values.push_back("true");

        option_keys.push_back("enable-fast-math");
        option_values.push_back("true");

        option_keys.push_back("debug-data");
        option_values.push_back("false");

        option_keys.push_back("infer-output-shape");
        option_values.push_back("false");

        option_keys.push_back("allow-expanded-dims");
        option_values.push_back("false");

        option_keys.push_back("enable-internal-profiling");
        option_values.push_back("false");

        option_keys.push_back("internal-profiling-detail");
        option_values.push_back("1");

        option_keys.push_back("enable-external-profiling");
        option_values.push_back("false");

        option_keys.push_back("timeline-profiling");
        option_values.push_back("false");

        option_keys.push_back("file-only-external-profiling");
        option_values.push_back("false");

        option_keys.push_back("counter-capture-period");
        option_values.push_back("10000u");

        option_keys.push_back("disable-tflite-runtime-fallback");
        option_values.push_back("false");

//        option_keys.push_back("outgoing-capture-file");
//        option_values.push_back((cache_path + "outgoing_profiling.bin").c_str());  // Path for save the profiling file.
//
//        option_keys.push_back("incoming-capture-file");
//        option_values.push_back((cache_path + "incoming_profiling.bin").c_str());  // Path for save the profiling file.
//
//        option_keys.push_back("serialize-to-dot");
//        option_values.push_back((cache_path + "serialize_model.dot").c_str());  // Path for save the serialize model file.

        if (use_gpu) {
            option_keys.push_back("backends");
            option_values.push_back("GpuAcc,EthosNPU,CpuAcc,CpuRef");  // Without white space.

            option_keys.push_back("gpu-tuning-level");
            option_values.push_back("3");

            option_keys.push_back("gpu-mlgo-tuning-file");
            option_values.push_back((cache_path + "model_cached.bin").c_str());

            option_keys.push_back("gpu-enable-profiling");
            option_values.push_back("false");

            option_keys.push_back("gpu-kernel-profiling-enabled");
            option_values.push_back("false");

        } else if (use_dsp_npu) {
            option_keys.push_back("backends");
            option_values.push_back("EthosNPU,GpuAcc,CpuAcc,CpuRef");  // EthosNPU, if the chip has ARM NPU.
        } else {
            option_keys.push_back("backends");
            option_values.push_back("CpuAcc,CpuRef");
        }
        const char* add_option_keys[option_keys.size()];
        const char* add_option_values[option_keys.size()];
        std::move(option_keys.begin(),option_keys.end(),add_option_keys);
        std::move(option_values.begin(),option_values.end(),add_option_values);
        auto delegateOptions = armnnDelegate::DelegateOptions(add_option_keys,add_option_values,option_keys.size(),
                                                              reinterpret_cast<void (*)(
                                                                      const char *)>(report_error_armnn_delegate));
        std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)> arm_accelerate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),armnnDelegate::TfLiteArmnnDelegateDelete);
        interpreter_A->ModifyGraphWithDelegate(arm_accelerate.get());
    } else if (use_gpu) {
        TfLiteGpuDelegateOptionsV2 backend_options = TfLiteGpuDelegateOptionsV2Default();
        backend_options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
        backend_options.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
        backend_options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
        backend_options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
        if (use_uint8 | use_int8) {
            backend_options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
        } else {
            backend_options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;
        }
        if (use_fp16) {
            backend_options.is_precision_loss_allowed = 1;  // 0 for precision, 1 for performance
        } else {
            backend_options.is_precision_loss_allowed = 0;
        }
        backend_options.max_delegated_partitions = 16;
        interpreter_A->ModifyGraphWithDelegate(TfLiteGpuDelegateV2Create(&backend_options));
    } else if (use_nnapi) {
        TfLiteNnapiDelegateOptions backend_options = {};
        tflite::StatefulNnApiDelegate::Options options;
        backend_options.execution_preference = TfLiteNnapiDelegateOptions::ExecutionPreference::kSustainedSpeed;
        backend_options.disallow_nnapi_cpu = 1; // 1 means back to TFLite-CPU while false at NNAPI-GPU/NPU.
        backend_options.max_number_delegated_partitions = -1;  // <= 0 means no limits.
        backend_options.accelerator_name = nullptr;  // nullptr for auto, It must set a valid backend name or nnapi won't work.
        backend_options.cache_dir = nullptr;  // nullptr for turn off.
        backend_options.model_token = nullptr;  // nullptr for turn off.
        if (use_fp16) {
            backend_options.allow_fp16 = 1;
        } else {
            backend_options.allow_fp16 = 0;
        }
        interpreter_A->ModifyGraphWithDelegate(TfLiteNnapiDelegateCreate(&backend_options));
    } else if (use_dsp_npu) {
        TfLiteHexagonInitWithPath((cache_path + file_name_A1).c_str());
        TfLiteHexagonDelegateOptions backend_options = TfLiteHexagonDelegateOptionsDefault();
        backend_options.powersave_level = 0;  // 0 means max power
        backend_options.debug_level = 0;  // 0 means no debug
        backend_options.enable_dynamic_batch_size = false;
        backend_options.max_batch_size = 1;
        backend_options.max_delegated_partitions = 8;
        backend_options.min_nodes_per_partition = 2;
        interpreter_A->ModifyGraphWithDelegate(TfLiteHexagonDelegateCreate(&backend_options));
    }
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_myapplication_MainActivity_Run_1Text_1Embedding(JNIEnv *env, jclass clazz,
                                                                jintArray token_index,
                                                                jint words_count,
                                                                jint model_hidden_size,
                                                                jint max_token_limit) {
    jint* input_ids = env->GetIntArrayElements(token_index,nullptr);
    std::vector<float> attention_masks(max_token_limit,0.f);
    std::fill(attention_masks.begin(),attention_masks.begin() + words_count,1.f);
    std::move(input_ids,input_ids + max_token_limit,input_0);
    std::move(attention_masks.begin(),attention_masks.end(),input_1);
    interpreter_A->Invoke();
    jfloatArray final_results = env->NewFloatArray(model_hidden_size);
    env->SetFloatArrayRegion(final_results,0,model_hidden_size,interpreter_A->typed_output_tensor<float>(0));
    return final_results;
}
