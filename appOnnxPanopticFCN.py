import torch
import time 
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda
import multiprocessing as mp
import numpy as np
import os
import tensorrt

from detectron2.export.api import export_onnx_model
from detectron2.engine import DefaultPredictor, DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer

import onnx
import cv2
#read the mode
import tools.pansonicFPNTrain as pansonicFPNTrain
#read args
from panopticfcn.panoptic_seg import PanopticFCN
from tools.trt_lite import TrtLite


mp.set_start_method("spawn", force=True)
args = pansonicFPNTrain.get_parser().parse_args()
setup_logger(name="fvcore")
logger = setup_logger()
logger.info("Arguments: " + str(args))

cfg = pansonicFPNTrain.setup_cfg(args)

#metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
predictor = DefaultPredictor(cfg)

model = predictor.model

#输入设置
img = cv2.imread('testpics/test.jpg')

h = 448
w = 512
img = cv2.resize(img, (512,448))
#print(img.shape)
img = img[:, :, ::-1].transpose((2, 0, 1))
#print(img.shape)
img = img[np.newaxis,:]
image = torch.from_numpy(img.copy()).to(torch.device('cuda'))
input_data = image.type(torch.float32)


# run 10 round output cuda time
nRound = 10
torch.cuda.synchronize()
t0 = time.time()
for i in range(nRound):
		classes_out_torch,pred_inst_out_torch,scores_out_torch = model(input_data)
torch.cuda.synchronize()
time_pytorch = (time.time() - t0) / nRound
print('[Pytorch] Logs_Seafood: Pytorch time:', time_pytorch)

#将结果从gpu中分离
scores_out_torch = scores_out_torch.cpu().detach().numpy()
print('[OuputDatatype] Logs_Seafood: Score Data Type: ', scores_out_torch.shape)
#================================================================================
#export onnx
#from detectron2.export.caffe2_export import  

export_flag = False
if export_flag:
	onnx_model = export_onnx_model(cfg,model,input_data)
	f = onnx_model.SerializeToString()
	file=open("pansonicFCNSeafood20210423.onnx","wb")
	file.write(f)
	print("[onnx] Logs_Seafood: export onnx success ")
#================================================================================
#tensorrt use:

#use trt engine file

#bug connot build the engine trt
tensorrt.init_libnvinfer_plugins(None, "") 

for engine_file_path in ['Engine/panoptic_fcn.trt','Engine/panoptic_fcn_fp16.trt','Engine/panoptic_fcn_int8.trt']:
	print("[Engine] Logs_Seafood: ===",engine_file_path,"===")
	if not os.path.exists(engine_file_path):
		print("[Engine] Logs_Seafood---No Engine File Found!")
	trt = TrtLite(engine_file_path=engine_file_path)
	trt.print_info()
	i2shape = {0:(1,3,h,w)}
	io_info = trt.get_io_info(i2shape)
	d_buffers = trt.allocate_io_buffers(i2shape,True)
	#io_info[1][2]:(216,)
	#申请score空间
	scores_out_trt = np.zeros(io_info[1][2],dtype=np.float32)
	cuda.memcpy_dtod(d_buffers[0],pansonicFPNTrain. PyTorchTensorHolder(input_data),input_data.nelement()*input_data.element_size())
	trt.execute(d_buffers,i2shape)
	cuda.memcpy_dtoh(scores_out_trt,d_buffers[1])
	#print('[OuputDatatype] Logs_Seafood: Score Data: ', scores_out_trt.shape)
	#tensorrt运行时间计算
	cuda.Context.synchronize()
	t0 = time.time()
	for i in range(nRound):
		trt.execute(d_buffers, i2shape)
	cuda.Context.synchronize()
	time_trt = (time.time() - t0) / nRound
	print('[TensorRT] Log_Seafood: TensorRT time of :',engine_file_path, 'with time ', time_trt)
	print('[TensorRT] Log_Seafood: Time Accelerate :',engine_file_path, 'about ', time_pytorch/time_trt)
	print('[TensorRT] Log_Seafood: Performance reduce :',engine_file_path, 'about ', np.sum(scores_out_torch-scores_out_trt)/np.sum(scores_out_torch))

