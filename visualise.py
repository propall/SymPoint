import argparse
import glob
from itertools import chain
import json
import math
import os
import re
import time
import xml.etree.ElementTree as ET # ElementTree package helps read and modify XML files
from collections import defaultdict # Same as a python dictionary but returns a default value for keys that don't exist

import mmcv
import numpy as np
import torch
from tqdm import tqdm
import yaml
from munch import Munch
from svgpathtools import parse_path

from svgnet.evaluation.point_wise_eval import InstanceEval
from svgnet.model.svgnet import SVGNet as svgnet
from svgnet.util import get_root_logger, init_dist, load_checkpoint
from svgnet.data import build_dataloader, build_dataset
from svgnet.data.svg import SVG_CATEGORIES

"""
Buggy code for visualisation, as per the github issue: https://github.com/nicehuster/SymPoint/issues/4

Working on this to visualise resultant images
"""

# Mapping of category IDs to colors for visualization
category2color = {item["id"]: item["color"] for item in SVG_CATEGORIES}

def get_args():
    """Parses command-line arguments for the SVGNet inference script.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser("svgnet")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file.",
        default="inference/svg_pointT.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the model checkpoint.",
        default="inference/best.pth",
    )
    parser.add_argument("--seed", type=int, default=2000, help="Random seed for reproducibility.")
    parser.add_argument(
        "--out",
        type=str,
        help="Directory for output results.",
        default="./visualise_outputs",
    )
    parser.add_argument("--save_lite", action="store_true", help="Save a lightweight version of the output.")
    
    args = parser.parse_args()
    return args

def reconstruct_svg(svg_file, estimated_contents, output_folder):
    """Modifies an SVG file by adding semantic and instance IDs to its elements.
    
    Args:
        svg_file (str): Path to the original SVG file.
        estimated_contents (list): List of dictionaries containing instance and semantic IDs for elements.
        output_folder (str): Directory to save the modified SVG file.
    """
    tree = ET.parse(svg_file) # Reads the SVG file into a tree structure
    root = tree.getroot() # Gets the root element of the SVG
    ns = root.tag[:-3] # Extracts XML namespace from root tag
    id = 0
    
    for g in root.iter(ns + "g"):  # Iterates through all <g> (group) elements in the SVG
        # Looks for path, circle and ellipse
        for _path in chain(
            g.iter(ns + "path"), 
            g.iter(ns + "circle"), 
            g.iter(ns + "ellipse")
        ):
            # Adds semantic and instance IDs to each element
            _path.attrib["semanticId"] = str(estimated_contents[id]["semanticId"])
            _path.attrib["instanceId"] = str(estimated_contents[id]["instanceId"])
            
            # Colors the element based on its semantic ID
            if estimated_contents[id]["semanticId"] == 0:
                _path.attrib["stroke"] = "rgb(0,0,0)"  # Default to black, ID 0
            else:
                color = category2color[estimated_contents[id]["semanticId"]]
                _path.attrib["stroke"] = f'rgb({color[0]},{color[1]},{color[2]})'
            
            id += 1
    
    tree.write(os.path.join(output_folder, os.path.basename(svg_file))) # Saves the modified SVG

def process():
    """Main function to perform inference on SVG files using the SVGNet model.
    
    Loads the model, processes test data, and generates modified SVG outputs with semantic and instance IDs.
    """
    # Gets command line arguments and read configuration file
    args = get_args()
    cfg_txt = open(args.config, "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    logger = get_root_logger()
    
    # Load model
    model = svgnet(cfg.model).cuda()
    logger.info(f"Load state dict from {args.checkpoint}")
    load_checkpoint(args.checkpoint, logger, model)
    
    # Load dataset and dataloader
    val_set = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(
        args, val_set, training=False, dist=False, visualise=True, **cfg.dataloader.test
    )
    print("Exited dataloader in visualise.py")
    
    # Sets up evaluation metrics
    instance_eval = InstanceEval(
        num_classes=cfg.model.semantic_classes, ignore_label=35, gpu_num=1
    )
    
    with torch.no_grad(): # Disable gradient calculation for inference
        model.eval() # Sets model to evaluation mode
        for i, batch in enumerate(tqdm(dataloader)):
            print("Entered dataloader loop.....................")
            json_file = batch[-1][0]  # Extracts JSON file name from batch
            # print(f"Length of batch: {len(batch)}")
            # print(f"batch0: {batch[0]}, {len(batch[0])}")
            # print(f"json_file: {batch}, {json_file}")

            # Ensure json_file is a valid path before using it
            if not isinstance(json_file, str):
                raise TypeError(f"Expected json_file to be a string, but got {type(json_file)}: {json_file}")
            
            svg_file = os.path.join(
                "./dataset/test/test/svg_gt",
                os.path.basename(json_file).replace("json", "svg"),
            )
            
            batch = batch[:-1]
            torch.cuda.empty_cache()
            
            # Runs model inference
            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                res = model(batch, return_loss=False)  # Model inference
            
            # Processes results and saves modified SVG
            if args.out:
                os.makedirs(args.out, exist_ok=True)
                instances = [len(instance["masks"]) for instance in res["instances"]]
                if len(instances) == 0:
                    continue
                
                # Prepares estimated contents for each element
                num = max(instances)
                estimated_contents = [
                    {
                        "instanceId": 0,
                        "semanticId": 0,
                    }
                ] * num
                
                # Processes each detected instance
                for index, instance in enumerate(instances):
                    src_label = instance["labels"]
                    src_score = instance["scores"]
                    
                    # Skips ignored labels and low confidence predictions
                    if src_label == instance_eval.ignore_label:
                        continue
                    if src_score < instance_eval.min_obj_score:
                        continue
                    
                    # Updates estimated contents with instance information
                    for i in np.where(instance["masks"])[0]:
                        estimated_contents[i] = {
                            "instanceId": index + 1,
                            "semanticId": instance["labels"],
                        }
                
                # Calls reconstruct_svg to save modified SVG
                reconstruct_svg(svg_file, estimated_contents, args.out)

if __name__ == "__main__":
    process()
