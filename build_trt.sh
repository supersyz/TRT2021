python3 -m onnxsim panoptic_fcn11.onnx 11.onnx
trtexec --verbose --onnx=11.onnx --saveEngine=panoptic_fcn.trt --explicitBatch
trtexec --verbose --onnx=11.onnx --saveEngine=panoptic_fcn_fp16.trt --explicitBatch --fp16

