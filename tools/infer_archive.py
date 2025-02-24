"""
This is not related to inference, just saving as an archive file for codesnippets

This script plots the midpoints of the graphical primitives and creates one svg per each batch(22 testimages distributed inference over 4 GPUs create 6 batches of inference, so 6 svgs created.Note that this batch is not batchsize=4, just 4 GPUs inferencing simultaneously)
saved images are stored at ./visualization_outputs_batchedoutputs/
"""

import argparse # For parsing command line arguments
import glob # For file pattern matching
from itertools import chain # For flattening iterables
import json, os, re, time  # Standard utilities
import os.path as osp
import math
import xml.etree.ElementTree as ET # For XML/SVG parsing
from collections import defaultdict

import mmcv
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from tqdm import tqdm # For progress bars
import yaml
from munch import Munch # For config handling
from svgpathtools import parse_path # SVG path parsing utility

from functools import partial
from svgnet.data import build_dataloader, build_dataset
from svgnet.evaluation import PointWiseEval,InstanceEval
from svgnet.model.svgnet import SVGNet as svgnet
from svgnet.data.svg import SVG_CATEGORIES
from svgnet.util import get_root_logger, init_dist, load_checkpoint

# Creates a dictionary mapping category IDs to their colors
category2color = {item["id"]: item["color"] for item in SVG_CATEGORIES}

def get_args():
    parser = argparse.ArgumentParser("svgnet")
    parser.add_argument("config",type=str,help="path to config file",default="./inference/svg_pointTsample.yaml",)
    parser.add_argument("checkpoint",type=str,help="path to checkpoint",default="./inference/best.pth")
    parser.add_argument("--sync_bn", action="store_true", help="run with sync_bn")
    parser.add_argument("--dist", action="store_true", help="run with distributed parallel")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--out",type=str,help="directory for output results",default="./visualization_outputs",)
    parser.add_argument("--save_lite", action="store_true")
    args = parser.parse_args()
    return args

def print_instance_classes(res):
    # Print ground truth instances
    print("\nGround Truth:")
    targets = res["targets"] # This is a dictionary with labels and masks as the keys
    gt_labels = targets["labels"]  # Ground truth class labels
    for idx, label in enumerate(gt_labels):
        class_name = SVG_CATEGORIES[label]["name"]  # Get class name from the categories
        print(f"Instance {idx}: Class {label} ({class_name})")
    
    # Print predicted instances
    print("\nPredictions:")
    for idx, instance in enumerate(res["instances"]):
        pred_class = instance["labels"]  # Predicted class
        pred_score = instance["scores"]  # Confidence score
        class_name = SVG_CATEGORIES[pred_class]["name"]
        print(f"Instance {idx}: Class {pred_class} ({class_name}), Confidence: {pred_score:.2f}")
    
def reconstruct_svg(svg_file, estimated_contents, output_folder):
    # Parse the SVG file
    tree = ET.parse(svg_file)
    root = tree.getroot()
    ns = root.tag[:-3] # Get XML namespace
    
    id = 0
    # Iterate through all path, circle, and ellipse elements in groups
    for g in root.iter(ns + "g"):
        for _path in chain(
            g.iter(ns + "path"), g.iter(ns + "circle"), g.iter(ns + "ellipse")
        ):
            # Update element attributes with predicted values
            _path.attrib["semanticId"] = str(estimated_contents[id]["semanticId"])
            _path.attrib["instanceId"] = str(estimated_contents[id]["instanceId"])
            
            if estimated_contents[id]["semanticId"] == 0:
                _path.attrib["stroke"] = "rgb(0,0,0)"
            else:
                _path.attrib["stroke"] = (
                    f'rgb({category2color[estimated_contents[id]["semanticId"]][0]},{category2color[estimated_contents[id]["semanticId"]][1]},{category2color[estimated_contents[id]["semanticId"]][2]})'
                )
            id += 1
            
    output_path = os.path.join(output_folder, os.path.basename(svg_file))
    tree.write(output_path)
    print(f"Saved colored SVG to: {output_path}")


def main():
    # Setup
    args = get_args()
    cfg_txt = open(args.config, "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    if args.dist:
        init_dist()
    logger = get_root_logger()
    
    # Initialize model
    model = svgnet(cfg.model).cuda()
    if args.sync_bn:
        nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    gpu_num = dist.get_world_size()
    
    logger.info(f"Load state dict from {args.checkpoint}")
    load_checkpoint(args.checkpoint, logger, model)
    
    # Setup dataset and dataloader
    val_set = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(args, val_set, training=False, dist=args.dist, **cfg.dataloader.test)
    
    time_arr = []
    # Classes that perform Evaluation
    sem_point_eval = PointWiseEval(num_classes=cfg.model.semantic_classes, ignore_label=35, gpu_num=gpu_num) # Evaluates Mean IoU (miou), Pixel Accuracy (pacc)
    instance_eval = InstanceEval(num_classes=cfg.model.semantic_classes, ignore_label=35, gpu_num=gpu_num) # Evaluates PQ (Panoptic Quality), RQ (Recognition Quality) and SQ (Segmentation Quality)
    
    # Create output directory
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    
    # Main processing loop
    with torch.no_grad():
        model.eval()
        for i, batch_data in enumerate(tqdm(dataloader)):
            print(f"\n======== Processing batch {i} ========")
            
            # Unpack batch data - each batch contains (coord, feat, label, offset, lengths)
            coord, feat, label, offset, lengths = batch_data
            
            t1 = time.time()
            if i % 10 == 0:
                step = int(len(val_set)/gpu_num)
                logger.info(f"Infer {i+1}/{step}")
            
            # Run model inference
            torch.cuda.empty_cache()
            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                # Pass tuple of tensors to model
                res = model((coord, feat, label, offset, lengths), return_loss=False)
                
                # Print classes for each instance
                print(f"\nBatch {i}:")
                print_instance_classes(res)
            
            t2 = time.time()
            time_arr.append(t2 - t1)
            
            # Get semantic segmentation predictions
            sem_preds = torch.argmax(res["semantic_scores"], dim=1).cpu().numpy()
            sem_gts = res["semantic_labels"].cpu().numpy()
            
            # update() fn is used to return the metrics mentioned previously
            sem_point_eval.update(sem_preds, sem_gts)
            """
            res["lengths"]: These are the lengths of primitives/points in the input
            res["targets"]: This contains the ground truth instance information from the dataset(has labels and masks as 2 keys which denote actual class labels and boolean masks indicating pixels which belong to ground truth as True)
            res["instances"]: This contains the model's instance predictions (has labels, scores and masks as 3 keys which denote predicted class, confidence score for the prediction and boolean mask showing which points belong to this instance) 
            """
            instance_eval.update(
                res["instances"],
                res["targets"],
                res["lengths"]
            )
            
            # Generate colored SVG visualization if output directory is specified
            if args.out and "instances" in res and len(res["instances"]) > 0:
                instances = res["instances"]
                
                # Prepare estimated contents for SVG reconstruction
                num_elements = max(len(instance["masks"]) for instance in instances)
                estimated_contents = [{"instanceId": 0, "semanticId": 0}] * num_elements
                
                for idx, instance in enumerate(instances):
                    label = instance["labels"]
                    score = instance["scores"]
                    
                    # Skip ignored or low confidence predictions
                    if label == instance_eval.ignore_label or score < instance_eval.min_obj_score:
                        continue
                    
                    # Update contents for each element in the instance
                    for element_idx in np.where(instance["masks"])[0]:
                        estimated_contents[element_idx] = {
                            "instanceId": idx + 1,
                            "semanticId": label
                        }
                
                # Generate output filename based on batch index
                output_filename = f"prediction_batch_{i}.svg"
                output_path = os.path.join(args.out, output_filename)
                
                # Create a basic SVG structure for visualization
                svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
                    <g>
                        <!-- Add predicted elements here -->
                        {generate_svg_elements(coord, estimated_contents)}
                    </g>
                </svg>'''
                
                # Save the SVG file
                with open(output_path, 'w') as f:
                    f.write(svg_content)
                print(f"Saved visualization to: {output_path}")

def generate_svg_elements(coord, estimated_contents):
    """
    Generate SVG elements based on coordinates and predictions
    """
    elements = []
    for i, (point, content) in enumerate(zip(coord, estimated_contents)):
        x, y = point[0].item(), point[1].item()
        semantic_id = content["semanticId"]
        
        # Get color for this semantic class
        if semantic_id == 0:
            color = "rgb(0,0,0)"
        else:
            color_values = category2color[semantic_id]
            color = f"rgb({color_values[0]},{color_values[1]},{color_values[2]})"
        
        # Create a circle element for each point
        elements.append(
            f'<circle cx="{x*500 + 500}" cy="{y*500 + 500}" r="5" '
            f'fill="{color}" stroke="none" '
            f'data-semantic-id="{semantic_id}" '
            f'data-instance-id="{content["instanceId"]}"/>'
        )
    
    return "\n".join(elements)

if __name__ == "__main__":
    main()
