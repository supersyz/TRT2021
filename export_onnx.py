# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import torch
import sys
sys.path.append("..")
import time
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import numpy as np
from predictor import VisualizationDemo
from panopticfcn import add_panopticfcn_config, build_lr_scheduler
from detectron2.engine.defaults import DefaultPredictor

#from alfred.vis.image.mask import label2color_mask, vis_bitmasks
#from alfred.vis.image.det import visualize_det_cv2_part
import numpy as np
from detectron2.data.catalog import MetadataCatalog
import detectron2.data.transforms as T
import cv2
##
# constants
WINDOW_NAME = "COCO detections"
torch.manual_seed(1)
np.random.seed(1)
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    add_panopticfcn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.EXPORT_ONNX = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(
        description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true",
                        help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


# def vis_res_fast(res, img, meta):
#     # print(meta)
#     stuff_cls = meta.stuff_colors
#     # seg = res['sem_seg']
#     # _, seg_flatten = torch.max(seg, dim=0)
#     # seg_flatten = seg_flatten.cpu().numpy()
#
#     # override things, road, sky
#     stuff_cls[0] = [0, 0, 0]
#     stuff_cls[40] = [255, 172, 84]  # sky
#     stuff_cls[21] = [207, 61, 255]
#     # m = label2color_mask(seg_flatten, override_id_clr_map=stuff_cls)
#
#     ins = res['instances']
#     bboxes = ins.pred_boxes.tensor.cpu().numpy()
#     scores = ins.scores.cpu().numpy()
#     clss = ins.pred_classes.cpu().numpy()
#     bit_masks = ins.pred_masks.tensor
#     img = vis_bitmasks(img, bit_masks)
#     img = visualize_det_cv2_part(
#         img, scores, clss, bboxes, class_names=meta.thing_classes)
#     # img = cv2.addWeighted(img, 0.9, m, 0.6, 0.9)
#     return img


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    #metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    predictor = DefaultPredictor(cfg)

    model = predictor.model
    img = cv2.imread('000000000139.png')

    h = 448
    w = 512
    img = cv2.resize(img, (512,448))
    #print(img.shape)
    img = img[:, :, ::-1].transpose((2, 0, 1))
    #print(img.shape)
    img = img[np.newaxis,:]
    image = torch.from_numpy(img.copy()).to(torch.device('cuda'))
    image = image.type(torch.float32)
    # torch.onnx.export(model, image, 'panoptic_fcn11.onnx', output_names={
    #     #   'scores', 'classes', 'pred_inst'}, opset_version=11, do_constant_folding=True, verbose=True)
    #                   'pred_inst', 'classes', 'scores', }, opset_version=11, do_constant_folding=True, verbose=True)
    # print('Exporting done.')

#     image = torch.randn([1, 3, h, w]).to(torch.device('cuda'))
    n_round = 10
    torch.cuda.synchronize()
    begin = time.time()
    classes_out_torch,pred_inst_out_torch,scores_out_torch = model(image)
    print(pred_inst_out_torch.dtype)
    print(classes_out_torch.shape,scores_out_torch.shape,pred_inst_out_torch.shape)

    scores_out_torch = scores_out_torch.cpu().detach().numpy()
    pred_inst_out_torch = pred_inst_out_torch.cpu().detach().numpy()
    classes_out_torch = classes_out_torch.cpu().detach().numpy()
    for i in range(n_round):
        model(image)
    torch.cuda.synchronize()
    end = time.time()
    pytorch_time = (end-begin) / n_round
    print('mean_time:',pytorch_time)
#

    from trt_lite import TrtLite
    import pycuda
    import os
    import tensorrt
    import pycuda.driver as cuda
    class PyTorchTensorHolder(pycuda.driver.PointerHolderBase):
        def __init__(self,tensor):
            super(PyTorchTensorHolder,self).__init__()
            self.tensor = tensor
        def get_pointer(self):
            return self.tensor.data_ptr()

    tensorrt.init_libnvinfer_plugins(None, "")
    #engine_file_path = 'panoptic_fcn_fp16.trt'
    for engine_file_path in ['panoptic_fcn.trt','panoptic_fcn_fp16.trt']:
        if not os.path.exists(engine_file_path):
            print('bad!')
        else:
            print('=='+engine_file_path+'==')
        trt = TrtLite(engine_file_path=engine_file_path)
        trt.print_info()
        i2shape = {0:(1,3,h,w)}
        io_info = trt.get_io_info(i2shape)

       # print(io_info)
        # print(io_info[1])
        # print(io_info[1][2])
        d_buffers = trt.allocate_io_buffers(i2shape,True)
        scores_out = np.zeros(io_info[1][2],dtype=np.float32)
        pred_inst_out = np.zeros(io_info[2][2],dtype=np.int32)
        classes_out = np.zeros(io_info[3][2],dtype=np.float32)
        #print(d_buffers)
        # #output_data_trt =
        cuda.memcpy_dtod(d_buffers[0],PyTorchTensorHolder(image),image.nelement()*image.element_size())
        trt.execute(d_buffers,i2shape)
        cuda.memcpy_dtoh(scores_out,d_buffers[1])
        cuda.memcpy_dtoh(pred_inst_out, d_buffers[2])
        cuda.memcpy_dtoh(classes_out, d_buffers[3])
#
        cuda.Context.synchronize()
        begin = time.time()
        for i in range(n_round):
            trt.execute(d_buffers,i2shape)
        cuda.Context.synchronize()
        end = time.time()
        trt_time = (end-begin)/n_round
        print('trt:',trt_time)
        print('Speedup:',pytorch_time/trt_time)
        #np.seterr(divide='ignore', invalid='ignore')
        #print(pred_inst_out)
        print(scores_out.dtype)
        print(scores_out_torch.dtype)
        #print('~~~~~~~~~~~~~~')
        #print(classes_out_torch)
        #print(classes_out-classes_out_torch)
        #print('scroes:',scores_out,'scroestorch:',scores_out_torch)
        #print('inst:',pred_inst_out,'insttorch:',pred_inst_out_torch)
        #print('class',classes_out,'classtorch',classes_out_torch)

        import pandas as pd

        #scores_out_torch[scores_out_torch==0] = np.nan
        #pred_inst_out_torch[pred_inst_out_torch==0] = np.nan
        #classes_out_torch[classes_out_torch==0] = np.nan

        print(scores_out.shape,pred_inst_out.shape,classes_out.shape)
        #abs1 = np.abs((scores_out - scores_out_torch))
        #abs2 = np.abs(scores_out_torch)
        print('dtype:',classes_out_torch.dtype)
        print(np.max(np.max(scores_out - scores_out_torch)))

        print(np.max(np.max(classes_out - classes_out_torch)))
        #print(np.max(np.max()))
        #print('Avg diff percentage:', np.mean(np.abs((scores_out - scores_out_torch)) / (np.abs(scores_out_torch)+10**(-8))) )
        #print('Avg diff percentage:', np.mean(np.abs((pred_inst_out - pred_inst_out_torch) / (np.abs(pred_inst_out_torch)+1 ))))
        scores_out = scores_out.astype(np.float128)
        scores_out_torch = scores_out_torch.astype(np.float128)
        abs1 = np.abs(scores_out - scores_out_torch,dtype=np.float128)
        abs2 = np.abs(scores_out_torch,dtype=np.float128)
        print('Avg diff percentage:',np.mean(abs1/(abs2+10**(-20))))

        #print('Avg diff percentage:', np.mean(np.abs((classes_out - classes_out_torch)) / (np.abs(classes_out_torch)  ) ))
        classes_out = classes_out.astype(np.float128)
        classes_out_torch = classes_out_torch.astype(np.float128)
        abs1 = np.abs((classes_out - classes_out_torch),dtype=np.float128)
        abs2 = np.abs(classes_out_torch,dtype = np.float128)
        #print('Avg diff percentage:', np.mean(abs1 / (abs2 + 10 ** (-20))))
        scores_out_ = pd.DataFrame(scores_out)
        scores_out_.to_csv('scores_out'+engine_file_path[:-4]+'.csv')
        scores_out_torch_ = pd.DataFrame(scores_out_torch)
        scores_out_torch_.to_csv('scores_out_torch''.csv')

        pred_inst_out_ = pd.DataFrame(pred_inst_out)
        pred_inst_out_.to_csv('pred_inst_out'+engine_file_path[:-4]+'.csv')
        pred_inst_out_torch = pred_inst_out_torch.astype(np.int32)
        pred_inst_out_torch_ = pd.DataFrame(pred_inst_out_torch)
        pred_inst_out_torch_.to_csv('pred_inst_out_torch.csv')
        classes_out_ = pd.DataFrame(classes_out[0][0])
        classes_out_.to_csv('classes_out'+engine_file_path[:-4]+'.csv')
        classes_out_torch_ = pd.DataFrame(classes_out_torch[0][0])
        classes_out_torch_.to_csv('classes_out_torch.csv')


    # # TRT_LOGGER = trt.Logger()
    # logger = trt.Logger(trt.Logger.INFO)
    # with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
    #     engine = runtime.deserialize_cuda_engine(f.read())
    #     print(engine)
    #     context = engine.create_execution_context()

    # explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # with trt.Builder(TRT_LOGGER) as builder, \
    #         builder.create_network(explicit_batch) as network, \
    #         trt.OnnxParser(network, TRT_LOGGER) as parser:


    # if args.input:
    #     if len(args.input) == 1:
    #         args.input = glob.glob(os.path.expanduser(args.input[0]))
    #         assert args.input, "The input path(s) was not found"
    #     for path in tqdm.tqdm(args.input, disable=not args.output):
    #         # use PIL, to be consistent with evaluation
    #         img = read_image(path, format="BGR")
    #         start_time = time.time()
    #         predictions, visualized_output = demo.run_on_image(img)
    #         logger.info(
    #             "{}: {} in {:.2f}s".format(
    #                 path,
    #                 "detected {} instances".format(
    #                     len(predictions["instances"]))
    #                 if "instances" in predictions
    #                 else "finished",
    #                 time.time() - start_time,
    #             )
    #         )
    #
    #         if args.output:
    #             if os.path.isdir(args.output):
    #                 assert os.path.isdir(args.output), args.output
    #                 out_filename = os.path.join(
    #                     args.output, os.path.basename(path))
    #             else:
    #                 assert len(
    #                     args.input) == 1, "Please specify a directory with args.output"
    #                 out_filename = args.output
    #             visualized_output.save(out_filename)
    #         else:
    #             cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    #             cv2.imshow(
    #                 WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
    #             if cv2.waitKey(0) == 27:
    #                 break  # esc to quit
    # elif args.webcam:
    #     assert args.input is None, "Cannot have both --input and --webcam!"
    #     assert args.output is None, "output not yet supported with --webcam!"
    #     cam = cv2.VideoCapture(0)
    #     for vis in tqdm.tqdm(demo.run_on_video(cam)):
    #         cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    #         cv2.imshow(WINDOW_NAME, vis)
    #         if cv2.waitKey(1) == 27:
    #             break  # esc to quit
    #     cam.release()
    #     cv2.destroyAllWindows()
    # elif args.video_input:
    #     video = cv2.VideoCapture(args.video_input)
    #     width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     frames_per_second = video.get(cv2.CAP_PROP_FPS)
    #     num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    #     basename = os.path.basename(args.video_input)
    #
    #     while(video.isOpened()):
    #         ret, frame = video.read()
    #         print(frame.shape)
    #         image = aug.get_transform(frame).apply_image(frame)
    #         image = torch.as_tensor(image.astype(
    #             "float32").transpose(2, 0, 1)).unsqueeze(0).cuda()
    #         print(image.shape)
    #         res = model(image)
    #         print(res)
    #         res = vis_res_fast(res, frame, metadata)
    #         cv2.imshow('frame', res)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
