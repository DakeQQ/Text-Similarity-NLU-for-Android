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
     - QNN (under develope)
     - SNPE (under develope)
     - TVM (under develope)
3. The demo models were uploaded to the drive: https://drive.google.com/drive/folders/1N7kUqRUI0aE6C2Rb6IcGQzd4D5FRZ-XG?usp=drive_link
4. After downloading, place the model into the assets folder.
5. Remember to decompress the *.so zip file stored in the libs/arm64-v8a folder.
6. The demo models, named 'GTE', were converted from ModelScope and underwent code optimizations to achieve extreme execution speed.
7. Therefore, the inputs & outputs of the demo models are slightly different from the original one.
8. To better adapt to ONNX Runtime on Android, the export did not use dynamic axes. Therefore, the exported ONNX model may not be optimal for x86_64 systems.
9. We will make the exported method public later.
10. See more projects: https://dakeqq.github.io/overview/
# Demo Results
![Demo Animation](https://github.com/DakeQQ/Text-Similarity-for-Android/blob/main/text_en.gif?raw=true)

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
     - QNN (开发中)
     - SNPE (开发中)
     - TVM (开发中)
3. 演示模型已上传至云端硬盘：https://drive.google.com/drive/folders/1N7kUqRUI0aE6C2Rb6IcGQzd4D5FRZ-XG?usp=drive_link
4. 百度: https://pan.baidu.com/s/1MA54Se04zBidmfVMBLk9wg?pwd=dake 提取码: dake
5. 下载后，请将模型文件放入assets文件夹。
6. 记得解压存放在libs/arm64-v8a文件夹中的*.so压缩文件。
7. 演示模型名为'GTE'，它们是从ModelScope转换来的，并经过代码优化，以实现极致执行速度。
8. 因此，演示模型的输入输出与原始模型略有不同。
9. 为了更好的适配ONNXRuntime-Android，导出时未使用dynamic-axes. 因此导出的ONNX模型对x86_64而言不一定是最优解.
10. 我们未来会提供转换导出的方法。
11. 看更多項目: https://dakeqq.github.io/overview/
# 演示结果
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
