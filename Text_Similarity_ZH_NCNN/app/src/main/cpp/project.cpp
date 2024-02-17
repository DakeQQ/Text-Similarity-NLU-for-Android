#include "project.h"

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_10(JNIEnv *env, jobject clazz,
                                                            jobject asset_manager,
                                                            jboolean use_int8, jboolean use_fp16,
                                                            jboolean use_gpu) {
    AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
    ncnn::set_cpu_powersave(2);  // 0: All, 1: small cores, 2: big cores. It is recommended to use big-core running fp32 GTE model.
    options.opt.openmp_blocktime = 1000; // unit: ms, for multi-threads, the lower value means shutdown quickly (save power).
    options.opt.num_threads = 1; // It fast enough for using one big-core.
    options.opt.use_local_pool_allocator = false;  // If true, it usually get worse.
    options.opt.use_packing_layout = true;
    options.opt.lightmode = true;
    options.opt.use_winograd_convolution = false;
    options.opt.use_sgemm_convolution = false;
    if (use_fp16) {  // We have tried the GTE fp16 format but received NaN as outputs. This issue occurred not only with NCNN but also with MNN and ONNX.
        options.opt.use_fp16_packed = true;
        options.opt.use_fp16_storage = true;
        options.opt.use_fp16_arithmetic = true;
    } else {
        options.opt.use_fp16_packed = false;
        options.opt.use_fp16_storage = false;
        options.opt.use_fp16_arithmetic = false;
    }
    if (use_int8) { // The NCNN int8 converter is too... So, the NCNN-GTE demo only use the fp32 format.
        options.opt.use_int8_inference = true;
        options.opt.use_int8_storage = true;
        options.opt.use_int8_packed = true;
        options.opt.use_int8_arithmetic = true;
    } else {
        options.opt.use_int8_inference = false;
        options.opt.use_int8_storage = false;
        options.opt.use_int8_packed = false;
        options.opt.use_int8_arithmetic = false;
    }
    if (use_gpu) {  // It usually get slower on fp32 or small size model.
        options.opt.use_vulkan_compute = true;
        options.opt.use_shader_local_memory = true;
        options.opt.use_cooperative_matrix = true;
        options.opt.use_shader_pack8 = true;
        options.opt.use_image_storage = true; // for Adreno GPU
        options.opt.use_tensor_storage = true; // for Mali GPU
    }
    options.load_param(mgr,file_name_A0);  // It must loading the model.param first.
    options.load_model(mgr,file_name_A1);
    in0.create(max_token_limit_GTE,1,sizeof(int32_t));
    in1.create(max_token_limit_GTE,1,sizeof(float));
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_myapplication_MainActivity_Run_1Text_1Embedding(JNIEnv *env, jclass clazz,
                                                                jintArray token_index,
                                                                jint words_count,
                                                                jint model_hidden_size) {
    jint* input_ids = env->GetIntArrayElements(token_index, nullptr);
    std::vector<float> attention_masks(max_token_limit_GTE,-999999999.f);
    std::fill(attention_masks.begin(),attention_masks.begin() + words_count,1.f);
    std::move(input_ids,input_ids + max_token_limit_GTE,static_cast<int32_t*> (in0.channel(0)));
    std::move(attention_masks.begin(),attention_masks.end(),static_cast<float*> (in1.channel(0)));
    {
        ncnn::Extractor model_A = options.create_extractor();
        model_A.input(input_names[0],in0);
        model_A.input(input_names[1],in1);
        model_A.extract(output_names[0],out0);
    }
    jfloatArray final_results = env->NewFloatArray(model_hidden_size);
    env->SetFloatArrayRegion(final_results,0,model_hidden_size,out0.channel(0));
    return final_results;
}
