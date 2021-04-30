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
    image = torch.randn([1, 3, h, w]).to(torch.device('cuda'))
    torch.onnx.export(model, image, 'panoptic_fcn11.onnx', output_names={

                      'pred_inst', 'classes', 'scores', }, opset_version=11, do_constant_folding=True, verbose=True)
    print('Exporting done.')

#     image = torch.randn([1, 3, h, w]).to(torch.device('cuda'))
#     n_round = 10
#     torch.cuda.synchronize()
#     begin = time.time()
#     classes_out_torch,pred_inst_out_torch,scores_out_torch = model(image)
#     print(pred_inst_out_torch.dtype)
#     print(classes_out_torch.shape,scores_out_torch.shape,pred_inst_out_torch.shape)
#
#     scores_out_torch = scores_out_torch.cpu().detach().numpy()
#     pred_inst_out_torch = pred_inst_out_torch.cpu().detach().numpy()
#     classes_out_torch = classes_out_torch.cpu().detach().numpy()
#     for i in range(n_round):
#         model(image)
#     torch.cuda.synchronize()
#     end = time.time()
#     pytorch_time = (end-begin) / n_round
#     print('mean_time:',pytorch_time)
#

