"""
Author: Kane Lindsay
This file is to train the object recognition network.
Training and Validation images are taken from content/train and content/validation
Annotated output images with bounding box & confidence level are output in annotated_results
"""

import os
from detectron2.utils.logger import setup_logger
import cv2
import random
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg, set_global_cfg
import os

DIRECTORY = os.getcwd()

# Register datasets

register_coco_instances("my_dataset_train", {}, DIRECTORY + "\\dataset\\train\\_annotations.coco.json",
                        DIRECTORY + "\\content\\train")
register_coco_instances("my_dataset_val", {}, DIRECTORY + "\\dataset\\valid\\_annotations.coco.json",
                        DIRECTORY + "\\content\\valid")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)

my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
dataset_dicts = DatasetCatalog.get("my_dataset_train")

# Visualise some samples of training data

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow("Training Data Sample - Press any key for next", vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Set up custom trainer class so evaluation can be done
class CocoTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)


# Training Configuration
cfg.DATALOADER.NUM_WORKERS = 0  # Has to be 0 to work on Windows
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Initial weights
cfg.SOLVER.IMS_PER_BATCH = 2  # Number of images to train on at once. Set this higher or lower depending on GPU power.
cfg.SOLVER.BASE_LR = 0.001  # Learning rate
cfg.SOLVER.WARMUP_ITERS = 1000  # Warmup training - not evaluated
cfg.SOLVER.MAX_ITER = 1500  # Maximum training iterations
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.TEST.EVAL_PERIOD = 500  # Validation happens every _ iterations.

# Set up trainer and output directory using configuration
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Write configuration to file
cfg.MODEL.WEIGHTS = DIRECTORY + "\\output\\model_final.pth"
with open("output_config.yaml", "w") as f:
    f.write(cfg.dump())  # save config to file
