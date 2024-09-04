# Text-Similarity-for-Android
1. Demonstration of running text similarity with mainstream accelerators on Android device. Including:
     - ArmNN
     - MNN
     - NCNN
     - ONNX Runtime
     - Paddle-Lite
     - Pytorch Mobile (Java)
     - TFLite (C++)
     - TFLite (Java)
     - MegEngine (under develope)
     - Noah-Bolt (under develope)
     - OpenVINO Android (under develope)
     - Pytorch Mobile (C++, under develope)
     - QNN (under develope)
     - TVM (under develope)
3. The demo models were uploaded to the drive: https://drive.google.com/drive/folders/1N7kUqRUI0aE6C2Rb6IcGQzd4D5FRZ-XG?usp=drive_link
4. After downloading, place the model into the assets folder.
5. Remember to decompress the *.so zip file stored in the libs/arm64-v8a folder.
6. The demo models, named 'GTE', were converted from ModelScope and underwent code optimizations to achieve extreme execution speed.
7. Therefore, the inputs & outputs of the demo models are slightly different from the original one.
8. To better adapt to ONNX Runtime on Android, the export did not use dynamic axes. Therefore, the exported ONNX model may not be optimal for x86_64 systems.
9. To export the model on your own, please go to the 'Export_ONNX' folder, follow the comments to set the folder path and GTE_config.py, and then execute the GTE_Export.py Python script. Next, quantize / optimize the onnx model by yourself.
10. If use onnxruntime.tools.convert_onnx_models_to_ort to convert to the *.ort format, it will automatically add Cast operators that changes fp16 multiplication to fp32.
11. The quantization method for the model can be seen in the folder "Do_Quantize".
12. The q4(uint4) quantization method is not currently recommended because the "MatMulNBits" operator in ONNX Runtime is performing poorly.
13. See more projects: https://dakeqq.github.io/overview/

# 文本相似度安卓应用
1. 在Android设备上使用主流加速框架运行文本相似度应用。包含:
     - ArmNN
     - MNN
     - NCNN
     - ONNX Runtime
     - Paddle-Lite
     - Pytorch Mobile (Java)
     - TFLite (C++)
     - TFLite (Java)
     - MegEngine (开发中)
     - Noah-Bolt (开发中)
     - OpenVINO Android (开发中)
     - Pytorch Mobile (C++, 开发中)
     - QNN (开发中)
     - TVM (开发中)
3. 演示模型已上传至云端硬盘：https://drive.google.com/drive/folders/1N7kUqRUI0aE6C2Rb6IcGQzd4D5FRZ-XG?usp=drive_link
4. 百度: https://pan.baidu.com/s/1MA54Se04zBidmfVMBLk9wg?pwd=dake 提取码: dake
5. 下载后，请将模型文件放入assets文件夹。
6. 记得解压存放在libs/arm64-v8a文件夹中的*.so压缩文件。
7. 演示模型名为'GTE'，它们是从ModelScope转换来的，并经过代码优化，以实现极致执行速度。
8. 因此，演示模型的输入输出与原始模型略有不同。
9. 为了更好的适配ONNXRuntime-Android，导出时未使用dynamic-axes. 因此导出的ONNX模型对x86_64而言不一定是最优解.
10. 想自行导出模型请前往“Export_ONNX”文件夹，按照注释操作设定文件夹路径和GTE_config.py，然后执行GTE_Export.py的python脚本。下一步，自己动手量化或优化导出的ONNX模型。
11. 若使用onnxruntime.tools.convert_onnx_models_to_ort转成*.ort格式，它会自动添加Cast算子将fp16乘法转成fp32。
12. 模型的量化方法可以在文件夹 "Do_Quantize" 中查看。
13. 现在不建议使用q4(uint4)量化方法, 因为ONNX Runtime的运算符"MatMulNBits"表现不佳。
14. 看更多項目: https://dakeqq.github.io/overview/
# Demo Results 演示结果
![Demo Animation](https://github.com/DakeQQ/Text-Similarity-for-Android/blob/main/text_en.gif?raw=true)
![Demo Animation](https://github.com/DakeQQ/Text-Similarity-for-Android/blob/main/text_zh.gif?raw=true)
# 版本信息 Version Information
1. ArmNN: 23.11
2. MNN: 2.8.1
3. NCNN: 20240102
4. ONNX Runtime: 1.17.0
5. Paddle-Lite: 2.13-rc
6. Pytorch Mobile Java: 2.1.0
7. TFLite C++: 2.15.0
8. TFLite Java: 2.14.0
# 测试 Benchmark
1. 测试仅供参考。This test is for fun only.
2. 测试设备为Kirin_990_5G，选择2xA76（#1，#2）设置。Test device is Kirin_990_5G with 2xA76 (#1,#2) setting.
3. 无值表示运行失败或与CPU之间没有明显差异。No values mean the run failed or no obvious difference between the CPU backends.
4. 第一名和第二名的耗时比（1st/2nd）约为0.842。The time cost ratio between the 1st & 2nd (1st/2nd) is approximately 0.842.
![Project Logo](https://github.com/DakeQQ/Text-Similarity-for-Android/blob/main/benchmark.png?raw=true)
