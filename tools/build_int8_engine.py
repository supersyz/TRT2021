import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import random

from calibrator import load_data, Calibrator
import sys, os

TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

# This function builds an engine from a Caffe model.
def build_int8_engine(model_file, calib, batch_size=32):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # We set the builder batch size to be the same as the calibrator's, as we use the same batches
        # during inference. Note that this is not required in general, and inference batch size is
        # independent of calibration batch size.
        builder.max_batch_size = batch_size
        config.max_workspace_size = (1 << 30) # 1GiB
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calib
        if not os.path.exists(model_file):
            print('ONNX file {} not found, please generate it.'.format(model_file))
            exit(0)
        print('Loading ONNX file from path {}...'.format(model_file))
        with open(model_file, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
        network.get_input(0).shape = [1, 3, 448, 512]
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(model_file))
        # Build engine and do int8 calibration.
        engine = builder.build_engine(network, config)
        if engine is None:
            print('Failed to create the engine')
            return None   
        print("Completed creating the engine")
        return engine

def main():
    test_set = '../testpics'
    model_file = '../Engine/11.onnx'
    # Now we create a calibrator and give it the location of our calibration data.
    # We also allow it to cache calibration data for faster engine building.
    engine_model_path = "models_save/panoptic_fcn_int8_7.2.2.trt"
    calibration_cache = 'models_save/panoptic_fcn_int8.cache'
    calib = Calibrator(test_set, cache_file=calibration_cache)

    # Inference batch size can be different from calibration batch size.
    batch_size = 1
    engine = build_int8_engine(model_file, calib, batch_size)
    with open(engine_model_path, "wb") as f:
        f.write(engine.serialize())
if __name__ == '__main__':
    main()