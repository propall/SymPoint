"""
This script does the panoptic symbol spotting in both single-gpu/multi-gpu distributed inference.
saved images are stored at ./visualization_outputs/

Usage: python tools/inference.py &> output.txt      (single GPU inference)
       bash tools/inference_dist.sh &> output.txt   (multi GPU inference)
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse # For parsing command line arguments
import glob # For file pattern matching
from itertools import chain # For flattening iterables
import json, os, re, time  # Standard utilities
import os.path as osp
import math
import xml.etree.ElementTree as ET # For XML/SVG parsing
from collections import defaultdict
import copy

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
    parser.add_argument("--config",type=str,help="path to config file",default="./inference/svg_pointTsample.yaml",)
    parser.add_argument("--checkpoint",type=str,help="path to checkpoint",default="./inference/best.pth")
    parser.add_argument("--sync_bn", action="store_true", help="run with sync_bn")
    parser.add_argument("--dist", action="store_true", help="run with distributed parallel")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--out",type=str,help="directory for output results",default="./visualization_outputs",)
    parser.add_argument("--save_lite", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID for single GPU mode")
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
    
def generate_svg_elements(coord, estimated_contents):
    """
    Never used this fn
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

## New changes

def process_single_image(model, coord, feat, label, offset, lengths, svg_file, args, cfg):
    """Process a single image and generate visualization"""
    with torch.cuda.amp.autocast(enabled=cfg.fp16):
        res = model((coord, feat, label, offset, lengths), return_loss=False)
        print_instance_classes(res)
    
    if args.out and "instances" in res and len(res["instances"]) > 0:
        instances = res["instances"]
        
        # Prepare estimated contents for SVG reconstruction
        num_elements = max(len(instance["masks"]) for instance in instances)
        estimated_contents = [{"instanceId": 0, "semanticId": 0}] * num_elements
        
        for idx, instance in enumerate(instances):
            label = instance["labels"]
            score = instance["scores"]
            
            if label == 35 or score < 0.5:  # Assuming min_obj_score is 0.5
                continue
            
            for element_idx in np.where(instance["masks"])[0]:
                estimated_contents[element_idx] = {
                    "instanceId": idx + 1,
                    "semanticId": label
                }
        
        try:
            reconstruct_svg(svg_file, estimated_contents, args.out)
        except Exception as e:
            print(f"Error processing {svg_file}: {e}")

def setup_model(args, cfg, logger):
    """Setup model for either single or multi-GPU inference"""
    if torch.cuda.is_available():
        if args.dist:  # Multi-GPU setup
            init_dist()
            device = torch.cuda.current_device()
        else:  # Single GPU setup
            device = args.gpu_id
            torch.cuda.set_device(device)
    else:
        raise RuntimeError("CUDA is not available. This script requires GPU support.")
    
    # Initialize model
    model = svgnet(cfg.model).cuda(device)
    
    # Handle batch normalization
    if args.sync_bn and args.dist:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # Wrap model for distributed training if needed
    if args.dist:
        model = DistributedDataParallel(
            model, 
            device_ids=[device],
        ) # find_unused_parameters=True
    
    # Load checkpoint
    logger.info(f"Load state dict from {args.checkpoint}")
    load_checkpoint(args.checkpoint, logger, model)
    model.eval()
    
    return model, device



def main():
    # Setup basic configuration
    args = get_args()
    cfg_txt = open(args.config, "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt)) # Convert config to easy-to-use object
    
    # Create output directory
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    
    # Setup logger
    logger = get_root_logger()
    
    # Setup model for either single or distributed mode
    model, device = setup_model(args, cfg, logger)
    
    # Setup dataset
    val_set = build_dataset(cfg.data.test, logger)
    
    # Get rank and world size (1 for single GPU)
    rank = dist.get_rank() if args.dist else 0
    world_size = dist.get_world_size() if args.dist else 1
    
    # Create dataloader according to mode
    if args.dist:
        # Distributed mode - each GPU gets its portion
        dataloader = build_dataloader(
            args, 
            val_set, 
            training=False, 
            dist=True,  # Always use distributed sampler in dist mode
            **cfg.dataloader.test
        )
    else:
        # Single GPU mode - process all data
        dataloader = build_dataloader(
            args, 
            val_set, 
            training=False, 
            dist=False,  # No distributed sampler needed
            **cfg.dataloader.test
        )
    
    # Only rank 0 prints total progress
    if rank == 0:
        print(f"\nProcessing {len(val_set)} images across {world_size} GPU(s)...")
    
    # Process images
    with torch.no_grad(): # Don't track gradients (saves memory)
        for i, batch_data in enumerate(tqdm(dataloader, disable=rank != 0)):
            # Move data to appropriate device
            coord, feat, label, offset, lengths = [
                x.cuda(device) if isinstance(x, torch.Tensor) else x 
                for x in batch_data
            ]
            
            # Get corresponding SVG file
            if args.dist:
                # In distributed mode, each GPU processes different images
                # Calculate global index based on rank and local index
                global_idx = i * world_size + rank
                if global_idx >= len(val_set.data_list):
                    continue
                json_file = val_set.data_list[global_idx]
            else:
                # In single GPU mode, process sequentially
                json_file = val_set.data_list[i]
            
            svg_file = os.path.join(
                "./dataset/verysmalltestset/test/svg_gt",
                os.path.basename(json_file).replace("json", "svg")
            )
            
            # Only print progress from rank 0 in distributed mode
            if rank == 0:
                if args.dist:
                    print(f"\nGPU {rank} processing image {global_idx + 1}/{len(val_set)}: {os.path.basename(svg_file)}")
                else:
                    print(f"\nProcessing image {i + 1}/{len(val_set)}: {os.path.basename(svg_file)}")
            
            process_single_image(model, coord, feat, label, offset, lengths, svg_file, args, cfg)
    
    # Synchronize all processes if in distributed mode
    if args.dist:
        # dist.barrier() # Synchronization point, this is not needed if only one process/distributed training is going on like in this case, but if 2 different model trainings are going on a shared GPU, better to have it
        dist.destroy_process_group() # Basic cleanup

if __name__ == "__main__":
    main()