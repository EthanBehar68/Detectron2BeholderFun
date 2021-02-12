import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random


if __name__ == '__main__':
    # Register new Dataset
    register_coco_instances(
        "beholder", 
        {}, 
        r'D:\Projects\B546 Project\BeholderImages\trainval.json', 
        r'D:\Projects\B546 Project\BeholderImages\Train'
    )
    beholderMetadata = MetadataCatalog.get("beholder")
    print(beholderMetadata)

    # Test dataset was registered correctly
    beholderDatasetDicts = DatasetCatalog.get("beholder")
    # for d in random.sample(beholderDatasetDicts, 3):
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=beholderMetadata, scale=0.5)
    #     vis = visualizer.draw_dataset_dict(d)
    #     cv2.imshow('OpenCV', vis.get_image()[:, :, ::-1])
    #     cv2.waitKey()

    cfg = get_cfg()
    cfg.merge_from_file(r'D:\Projects\B546 Project\detectron2-windows\configs\COCO-InstanceSegmentation\mask_rcnn_R_50_DC5_3x.yaml')
    cfg.DATASETS.TRAIN = ("beholder",)
    cfg.DATASETS.TEST = () # no metrics implemented for this dataSet
    cfg.DATALOADER.NUM_WORKS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    print('CFG OUTPUT DIR = ' + cfg.OUTPUT_DIR)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()