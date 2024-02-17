# Text-Similarity-for-Android
1. Demonstration of running text similarity with mainstream accelerators on Android device. 
2. The demo models were uploaded to the drive: https://drive.google.com/drive/folders/1N7kUqRUI0aE6C2Rb6IcGQzd4D5FRZ-XG?usp=drive_link
3. After downloading, place the model into the assets folder.
4. Remember to decompress the *.so zip file stored in the libs/arm64-v8a folder.
5. The demo models, named 'GTE', were converted from ModelScope and underwent code optimizations to achieve extreme execution speed.
6. Therefore, the inputs & outputs of the demo models are slightly different from the original one.
7. We will make the exported method public later.
# Demo Results
![Demo Animation](https://github.com/DakeQQ/Text-Similarity-for-Android/blob/main/text_en.gif?raw=true)

# 文本相似度安卓应用
1. 在Android设备上使用主流加速框架运行文本相似度应用。
2. 演示模型已上传至云端硬盘：https://drive.google.com/drive/folders/1N7kUqRUI0aE6C2Rb6IcGQzd4D5FRZ-XG?usp=drive_link
3. 下载后，请将模型文件放入assets文件夹。
4. 记得解压存放在libs/arm64-v8a文件夹中的*.so压缩文件。
5. 演示模型名为'GTE'，它们是从ModelScope转换来的，并经过代码优化，以实现极致执行速度。
6. 因此，演示模型的输入输出与原始模型略有不同。
7. 我们未来会提供转换导出的方法。
# 演示结果
![Demo Animation](https://github.com/DakeQQ/Text-Similarity-for-Android/blob/main/text_zh.gif?raw=true)
# Version Information:  
1. ArmNN:         v23.11
2. MNN:           v2.8.1
3. NCNN:          v20240102
4. ONNX:          v1.17.0
5. Paddle-Lite:   v2.13.rc
6. Pytorch Java:  v2.1.0
7. TFLite C++:    v2.15.0
8. TFLite Java:   v2.14.0
# 测试 Benchmark
1. This test is for fun only.
![Project Logo](https://github.com/DakeQQ/Text-Similarity-for-Android/blob/main/benchmark.png?raw=true)
