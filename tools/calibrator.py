# *** tensorrt校准模块  ***

import tensorrt as trt
import os

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import glob,cv2

def preprocess_v1(image_raw,height,width):
    h, w, c = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # Calculate widht and height and paddings
    r_w = width / w
    r_h = height / h
    if r_h > r_w:
        tw = width
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((height - th) / 2)
        ty2 = height - th - ty1
    else:
        tw = int(r_h * w)
        th = height
        tx1 = int((width - tw) / 2)
        tx2 = width - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
    )
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    #image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    #image = np.ascontiguousarray(image)
    return image

def load_data(filepath):
    height=448
    width=512
    img_list = glob.glob(os.path.join(filepath, "*.jpg"))
    print('found all {} images to calib.'.format(len(img_list)))
    calibration_data = np.zeros((len(img_list),3,height,width), dtype=np.uint8)
    for i in range(len(img_list)):
        img = cv2.imread(img_list[i])
        img = preprocess_v1(img,height,width)
        calibration_data[i] = img
    return np.ascontiguousarray((calibration_data/255.0).astype(np.float32).reshape(len(img_list), 3, height, width))
# calibrator
#IInt8EntropyCalibrator2
#IInt8LegacyCalibrator
#IInt8EntropyCalibrator
#IInt8MinMaxCalibrator
class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, training_data, cache_file, batch_size=64):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = load_data(training_data)
        self.batch_size = batch_size
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]


    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
