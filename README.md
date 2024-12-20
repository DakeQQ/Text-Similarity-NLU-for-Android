

---

# Text Similarity for Android

This project showcases how to run text similarity applications using popular accelerators on Android devices. Supported frameworks include:

- **ArmNN**
- **MNN**
- **NCNN**
- **ONNX Runtime**
- **Paddle-Lite**
- **Pytorch Mobile (Java)**
- **TFLite (C++)**
- **TFLite (Java)**

## To-Do List

Frameworks currently under development:

- **MegEngine**
- **Noah-Bolt**
- **OpenVINO Android**
- **Pytorch Mobile (C++)**
- **QNN**
- **TVM**

## Getting Started

### 演示结果 Demo Models

You can download the demo models from:
- [Google Drive](https://drive.google.com/drive/folders/1N7kUqRUI0aE6C2Rb6IcGQzd4D5FRZ-XG?usp=drive_link)

After downloading, place the model files in the `assets` folder. Remember to extract the `*.so` files from the zip archive located in the `libs/arm64-v8a` folder.

### Model Details

The demo models, referred to as 'GTE', have been converted from ModelScope and optimized for maximum execution speed. Note that the inputs and outputs of these demo models may vary slightly from the originals.

- For the ONNX Runtime adaptation on Android, dynamic axes were not used, making the exported ONNX model potentially suboptimal for x86_64 systems.
- To export the model yourself, go to the `Export_ONNX` folder, adjust the folder path and `GTE_config.py` according to the comments, and run the `GTE_Export.py` Python script. You can then quantize or optimize the ONNX model as needed.

### Quantization

- When using `onnxruntime.tools.convert_onnx_models_to_ort` to convert to the `*.ort` format, Cast operators are automatically added, changing fp16 multiplication to fp32.
- Details on the quantization method can be found in the "Do_Quantize" folder.
- The `q4(uint4)` quantization method is not recommended due to the subpar performance of the "MatMulNBits" operator in ONNX Runtime.

## Additional Resources

For more projects, visit: [DakeQQ Projects Overview](https://dakeqq.github.io/overview/)

## Demo Results

![Demo Animation (English)](https://github.com/DakeQQ/Text-Similarity-for-Android/blob/main/text_en.gif?raw=true)
![Demo Animation (Chinese)](https://github.com/DakeQQ/Text-Similarity-for-Android/blob/main/text_zh.gif?raw=true)

## Version Information

- **ArmNN:** 23.11
- **MNN:** 2.8.1
- **NCNN:** 20240102
- **ONNX Runtime:** 1.17.0
- **Paddle-Lite:** 2.13-rc
- **Pytorch Mobile Java:** 2.1.0
- **TFLite C++:** 2.15.0
- **TFLite Java:** 2.14.0

## Benchmark

This benchmark is for informational and entertainment purposes only.

- **Test Device:** Kirin_990_5G with 2xA76 (#1, #2) configuration.
- No values indicate a failed run or no significant difference between CPU backends.
- The time cost ratio between the 1st and 2nd place is approximately 0.842.

![Benchmark Results](https://github.com/DakeQQ/Text-Similarity-for-Android/blob/main/benchmark.png?raw=true)

---

# 文本相似度安卓应用

此项目展示了在安卓设备上使用主流加速框架运行文本相似度应用。支持的框架包括：

- **ArmNN**
- **MNN**
- **NCNN**
- **ONNX Runtime**
- **Paddle-Lite**
- **Pytorch Mobile (Java)**
- **TFLite (C++)**
- **TFLite (Java)**

## 待办事项

以下框架正在开发中：

- **MegEngine**
- **Noah-Bolt**
- **OpenVINO Android**
- **Pytorch Mobile (C++)**
- **QNN**
- **TVM**

## 入门指南

### 演示模型

演示模型可在以下位置下载：
- [谷歌云盘](https://drive.google.com/drive/folders/1N7kUqRUI0aE6C2Rb6IcGQzd4D5FRZ-XG?usp=drive_link)

下载后，请将模型文件放入`assets`文件夹中。记得解压存放在`libs/arm64-v8a`文件夹中的`*.so`压缩文件。

### 模型详情

演示模型名为'GTE'，它们是从ModelScope转换来的，并经过代码优化，以实现极致执行速度。注意，演示模型的输入输出与原始模型略有不同。

- 为了更好的适配ONNX Runtime Android，导出时未使用动态轴。因此导出的ONNX模型对x86_64而言不一定是最优解。
- 想自行导出模型请前往`Export_ONNX`文件夹，按照注释操作设定文件夹路径和`GTE_config.py`，然后执行`GTE_Export.py`的Python脚本。下一步，自己动手量化或优化导出的ONNX模型。

### 量化

- 使用`onnxruntime.tools.convert_onnx_models_to_ort`转成`*.ort`格式时，会自动添加Cast算子将fp16乘法转成fp32。
- 模型的量化方法可以在"Do_Quantize"文件夹中查看。
- 现在不建议使用`q4(uint4)`量化方法，因为ONNX Runtime的运算符"MatMulNBits"表现不佳。

## 更多资源

查看更多项目，请访问：[DakeQQ 项目概览](https://dakeqq.github.io/overview/)


## 测试基准

此测试仅供参考和娱乐用途。

- **测试设备：**Kirin_990_5G，设置为2xA76（#1，#2）。
- 无值表示运行失败或与CPU后端之间没有明显差异。
- 第一名和第二名的耗时比约为0.842。
--- 

