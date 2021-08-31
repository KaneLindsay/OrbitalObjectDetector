"""
Author: Kane Lindsay
Description: This is the script to do inference on unseen images. Images from the 'Test' directory are used,
the output is placed in the 'annotated results' directory and bounding box pixel coordinates are written to
output/outputs.txt
"""
import json
import ntpath
import os
import cv2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from pyexpat import model

DIRECTORY = os.getcwd()

# Set up testing configuration
cfg = get_cfg()
cfg.merge_from_file("output_config.yaml")
cfg.DATASETS.TEST = ("my_dataset_test",)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
predictor = DefaultPredictor(cfg)

print(DIRECTORY + "\\dataset\\model_final.pth")

os.makedirs("annotated_results", exist_ok=True)  # Folder for annotated detection outputs

register_coco_instances("my_dataset_test", {}, DIRECTORY + "\\content\\test\\_annotations.coco.json",
                        DIRECTORY + "\\content\\test")

statement_metadata = MetadataCatalog.get("my_dataset_test")

# Create / clear coordinate output file
with open("annotated_results\\outputs.txt", "w") as f:
    f.truncate(0)

# For all .jpg files in testing directory
for filename in os.listdir(DIRECTORY + "\\dataset\\test"):

    if filename.endswith(".jpg"):
        # Do inference on image
        im = cv2.imread(DIRECTORY + "\\dataset\\test\\" + filename)
        outputs = predictor(im)

        print(outputs)

        with open("annotated_results\\outputs.txt", "a") as f:
            output_box = outputs['instances'][outputs['instances'].pred_classes == 1].pred_boxes
            f.write(str(output_box.tensor.cpu().numpy())+"\n")  # save output info to file

        v = Visualizer(
            im[:, :, ::-1],
            scale=0.5
        )
        instances = outputs["instances"].to("cpu")
        # Display images in window
        v = v.draw_instance_predictions(instances)
        result = v.get_image()[:, :, ::-1]
        cv2.imshow("Prediction Result - Press any key for next", result)
        cv2.waitKey(0)
        file_name = ntpath.basename(filename)
        write_res = cv2.imwrite(f'annotated_results/{file_name}', result)
