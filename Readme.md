# Panoptic Segmentation的模型部署及TRT加速
本仓库基于NVIDIA-阿里 2021 TRT 比赛hackthon，例程及环境配置来源于[trt-samples-for-hackathon-cn](https://github.com/NVIDIA/trt-samples-for-hackathon-cn.git)
## 1、模型来源
根据文章《Fully Convolutional Networks for Panoptic Segmentation》及其[Github](https://github.com/Jia-Research-Lab/PanopticFCN)源代码得到模型的训练过程。网络模型如下图所示。该网络用于全景分割。
![模型照片](./pictures/model.png)
## 2、环境搭建
本仓库使用

    nvidia-docker pull nvcr.io/nvidia/tensorrt:21.02-py3

新建docker容器并在此基础上进行工作，具体环境配置过程见[传送门](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/blob/master/hackathon/setup.md)
## 3、训练过程
训练得[Engine/model_final.pth](Engine/model_final.pth)
## 4、pytorch -> onnx
    python3 export_onnx.py --config-file ../Engine/PanopticFCN-R50-400-3x-FAST-nodeform.yaml --opts MODEL.WEIGHTS ../Engine/model_final.pth
导出onnx文件[panoptic_fcn.onnx](Engine/panoptic_fcn.onnx)
## 5、onnx -> trt
### 5.1 trt.FP32
    trtexec --verbose --onnx=panoptic_fcn.onnx --saveEngine=panoptic_fcn.trt --explitBatch
### 5.2 trt.FP16
    trtexec --verbose --onnx=panoptic_fcn.onnx --saveEngine=panoptic_fcn_fp16.trt --explicitBatch --fp16
### 5.3 trt.INT8
#### 校准集准备
在[testpics](tespics)文件夹下准备足量照片（>batch_size*max_batch）或者修改[tools/build_int8_engine](tools/build_int8_engine)中的文件类型和路径，设定batch_size。校准过程见[calibrator.py](tools/calibrator.py)。
#### 引擎生成
    python tools/build_int8_engine
#### 引擎储存路径
文件会被储存在tools/models_save文件夹下，如没有特别标注不会生成panoptic_fcn_int.cache描述文件。为方便调用将该文件夹下的panoptic_fcn_int8.trt放于Engine文件夹下:

    mv tools/models_save/panoptic_fcn_int8.trt Engine/
不建议使用cp命令，若models_save拥有trt文件的话会首先读入该引擎文件造成混淆。
## 6、TRT引擎加速效果
运行appOnnxPanopticFCN.py文件获得

    python appOnnxPanopticFCN.py --config-file Engine/PanopticFCN-R50-400-3x-FAST-nodeform.yaml --opts MODEL.WEIGHTS Engine/model_final.pth

性能表现：因为在全景分割中每一个像素点都会被准确的分配到某个类别中，所以本程序所使用的效果对比为PQ函数（全景分割评价指标），该标准对应的模型输出即 output['scores'] 这个输出
![PQ评价标准](./pictures/PQ.png)
分别使用FP32、FP16和INT8模型进行模型部署，使用单张照片作为测试照片输入。测试结果如下：

|  | FP32 | FP16 | INT8 |
| ------ | ------ | ------ | ----|
| 运行时间 | 中等文本 | 稍微长一点的文本 | asd|
| 加速比 | 短文本 | 中等文本 | asd |
| scores减比 | 中等文本 | 稍微长一点的文本 | asd|

## 6、C++中的模型部署