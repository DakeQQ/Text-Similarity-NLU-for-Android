#include "project.h"

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_10(JNIEnv *env, jclass clazz,
                                                            jobject asset_manager,
                                                            jboolean use_fp16, jboolean use_bf16,
                                                            jboolean use_gpu, jboolean use_onnx) {
    std::vector<armnn::BackendOptions> backend_options = {};
    std::vector<armnn::BackendId> backendPreferences = {};
    backendPreferences.push_back(armnn::Compute::CpuAcc);  // CpuAcc = CpuRef + NEON
    armnn::BackendOptions cpu("CpuAcc",
                              {
                                      {"NumberOfThreads", cpu_threads},
                                      {"FastMathEnable", true},
                                      {"MemoryOptimizerStrategy", "ConstantMemoryStrategy"},
                              });
    backend_options.push_back(cpu);
    if (use_gpu) {
        backendPreferences.insert(backendPreferences.begin(),armnn::Compute::GpuAcc);
        armnn::BackendOptions gpu("GpuAcc",
                                  {
                                          {"TuningLevel", 3},
                                          {"TuningFile", (cache_path + file_name_A1).c_str()},
                                          {"MemoryOptimizerStrategy", "ConstantMemoryStrategy"},
                                          {"KernelProfilingEnabled", enable_profiling}
                                  });
        backend_options.insert(backend_options.begin(),gpu);
        runtime_options.m_EnableGpuProfiling = enable_profiling;
    }
    const armnn::OptimizerOptionsOpaque& optimizer_options = armnn::OptimizerOptionsOpaque(use_fp16, false, use_bf16,
                                                                                           armnn::ShapeInferenceMethod::ValidateOnly,
                                                                                           memory_protected, backend_options, false,
                                                                                           false,  false);
    runtime_options.m_BackendOptions = backend_options;
    runtime_options.m_ProtectedMode = memory_protected;
    runtime_options.m_ProfilingOptions.m_EnableProfiling = enable_profiling;
    runtime_options.m_ProfilingOptions.m_TimelineEnabled = enable_profiling;
    if (enable_profiling) {
        runtime_options.m_ProfilingOptions.m_CapturePeriod = armnn::LOWEST_CAPTURE_PERIOD;
        runtime_options.m_ProfilingOptions.m_FileFormat = "binary";
        runtime_options.m_ProfilingOptions.m_FileOnly = true;
        runtime_options.m_ProfilingOptions.m_IncomingCaptureFile = (cache_path + "in_profiling.bin").c_str();
        runtime_options.m_ProfilingOptions.m_OutgoingCaptureFile = (cache_path + "out_profiling.bin").c_str();
    }
    runtime_A = armnn::IRuntime::Create(runtime_options);
    AAssetManager* mgr = AAssetManager_fromJava(env,asset_manager);
    AAsset* asset = AAssetManager_open(mgr,file_name_A0.c_str(),AASSET_MODE_BUFFER);
    size_t fileSize = AAsset_getLength(asset);
    std::vector<uint8_t> fileBuffer(fileSize);
    AAsset_read(asset,fileBuffer.data(),fileSize);
    if (use_onnx) {
        armnnOnnxParser::IOnnxParserPtr parser(armnnOnnxParser::IOnnxParser::Create());
        armnn::INetworkPtr network = parser->CreateNetworkFromBinary(fileBuffer);
        armnn::IOptimizedNetworkPtr optimized_net = armnn::Optimize(*network, backendPreferences, runtime_A->armnn::IRuntime::GetDeviceSpec(), optimizer_options, armnn::EmptyOptional());
        runtime_A->LoadNetwork(network_id_A,std::move(optimized_net));
        {
            armnnOnnxParser::BindingPointInfo inBind = parser->GetNetworkInputBindingInfo(
                    input_names[0]);
            armnn::TensorInfo in_info = (runtime_A->GetInputTensorInfo(network_id_A, inBind.first));
            in_info.SetConstant(true);
            in_tensor.push_back({inBind.first, armnn::ConstTensor(in_info, inputs_0.data())});
        }
        {
            armnnOnnxParser::BindingPointInfo inBind = parser->GetNetworkInputBindingInfo(input_names[1]);
            armnn::TensorInfo in_info = (runtime_A->GetInputTensorInfo(network_id_A, inBind.first));
            in_info.SetConstant(true);
            in_tensor.push_back({inBind.first, armnn::ConstTensor(in_info,inputs_1.data())});
        }
        {
            armnnOnnxParser::BindingPointInfo outBind = parser->GetNetworkOutputBindingInfo(output_names[0]);
            armnn::TensorInfo out_info = (runtime_A->GetOutputTensorInfo(network_id_A, outBind.first));
            out_info.SetConstant(true);
            out_tensor.push_back({outBind.first, armnn::Tensor(out_info,outputs_0.data())});
        }
    } else {
        armnnTfLiteParser::ITfLiteParser::TfLiteParserOptions parserOptions;
        parserOptions.m_AllowExpandedDims = false;
        parserOptions.m_InferAndValidate = false;
        parserOptions.m_StandInLayerForUnsupported = true;
        armnnTfLiteParser::ITfLiteParserPtr parser(armnnTfLiteParser::ITfLiteParser::Create(parserOptions));
        armnn::INetworkPtr network = parser->CreateNetworkFromBinary(fileBuffer);
        armnn::IOptimizedNetworkPtr optimized_net = armnn::Optimize(*network, backendPreferences, runtime_A->armnn::IRuntime::GetDeviceSpec(), optimizer_options, armnn::EmptyOptional());
        runtime_A->LoadNetwork(network_id_A,std::move(optimized_net));
        input_names = parser->GetSubgraphInputTensorNames(0);
        {
            armnnTfLiteParser::BindingPointInfo inBind = parser->GetNetworkInputBindingInfo(0, input_names[0]);
            armnn::TensorInfo in_info = (runtime_A->GetInputTensorInfo(network_id_A, inBind.first));
            in_info.SetConstant(true);
            in_tensor.push_back({inBind.first, armnn::ConstTensor(in_info,inputs_0.data())});
        }
        {
            armnnTfLiteParser::BindingPointInfo inBind = parser->GetNetworkInputBindingInfo(0, input_names[1]);
            armnn::TensorInfo in_info = (runtime_A->GetInputTensorInfo(network_id_A, inBind.first));
            in_info.SetConstant(true);
            in_tensor.push_back({inBind.first, armnn::ConstTensor(in_info,inputs_1.data())});
        }
        output_names = parser->GetSubgraphOutputTensorNames(0);
        {
            armnnTfLiteParser::BindingPointInfo outBind = parser->GetNetworkOutputBindingInfo(0, output_names[0]);
            armnn::TensorInfo out_info = (runtime_A->GetOutputTensorInfo(network_id_A, outBind.first));
            out_info.SetConstant(true);
            out_tensor.push_back({outBind.first, armnn::Tensor(out_info,outputs_0.data())});
        }
    }
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_myapplication_MainActivity_Run_1Text_1Embedding(JNIEnv *env, jclass clazz,
                                                                jintArray token_index,
                                                                jint words_count) {
    jint* input_ids = env->GetIntArrayElements(token_index, nullptr);
    std::vector<float> attention_masks(max_token_limit_GTE, -999999999.f);
    std::fill(attention_masks.begin(),attention_masks.begin() + words_count,1.f);
    std::move(input_ids,input_ids + max_token_limit_GTE,inputs_0.begin());
    std::move(attention_masks.begin(),attention_masks.end(),inputs_1.begin());
    runtime_A->EnqueueWorkload(network_id_A,in_tensor,out_tensor);
    jfloatArray final_results = env->NewFloatArray(outputs_0.size());
    env->SetFloatArrayRegion(final_results,0,outputs_0.size(),outputs_0.data());
    return final_results;
}
