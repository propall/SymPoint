import numpy as np
import torch
import torch.nn as nn
import yaml
from munch import Munch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import argparse
import multiprocessing as mp
import os
import os.path as osp
import time
from functools import partial
from svgnet.data import build_dataloader, build_dataset
from svgnet.evaluation import PointWiseEval, InstanceEval
from svgnet.model.svgnet import SVGNet as svgnet
from svgnet.util import get_root_logger, init_dist, load_checkpoint
from svgnet.data.svg import SVG_CATEGORIES
import xml.etree.ElementTree as ET
from itertools import chain

# Add category color mapping
category2color = {item["id"]: item["color"] for item in SVG_CATEGORIES}

def get_args():
    parser = argparse.ArgumentParser("svgnet")
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("checkpoint", type=str, help="path to checkpoint")
    parser.add_argument("--sync_bn", action="store_true", help="run with sync_bn")
    parser.add_argument("--dist", action="store_true", help="run with distributed parallel")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--out", type=str, help="directory for output results")
    parser.add_argument("--save_lite", action="store_true")
    args = parser.parse_args()
    return args

def print_instance_classes(res):
    print("\nGround Truth:")
    targets = res["targets"]
    gt_labels = targets["labels"]
    for idx, label in enumerate(gt_labels):
        class_name = SVG_CATEGORIES[label]["name"]
        print(f"Instance {idx}: Class {label} ({class_name})")
    
    print("\nPredictions:")
    for idx, instance in enumerate(res["instances"]):
        pred_class = instance["labels"]
        pred_score = instance["scores"]
        class_name = SVG_CATEGORIES[pred_class]["name"]
        print(f"Instance {idx}: Class {pred_class} ({class_name}), Confidence: {pred_score:.2f}")

def reconstruct_svg(svg_file, estimated_contents, output_folder):
    """Modifies an SVG file by adding semantic and instance IDs to its elements."""
    tree = ET.parse(svg_file)
    root = tree.getroot()
    ns = root.tag[:-3]
    id = 0
    
    for g in root.iter(ns + "g"):
        for _path in chain(
            g.iter(ns + "path"), 
            g.iter(ns + "circle"), 
            g.iter(ns + "ellipse")
        ):
            _path.attrib["semanticId"] = str(estimated_contents[id]["semanticId"])
            _path.attrib["instanceId"] = str(estimated_contents[id]["instanceId"])
            
            if estimated_contents[id]["semanticId"] == 0:
                _path.attrib["stroke"] = "rgb(0,0,0)"
            else:
                color = category2color[estimated_contents[id]["semanticId"]]
                _path.attrib["stroke"] = f'rgb({color[0]},{color[1]},{color[2]})'
            
            id += 1
    
    tree.write(os.path.join(output_folder, os.path.basename(svg_file)))

def main():
    args = get_args()
    cfg_txt = open(args.config, "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    if args.dist:
        init_dist()
    logger = get_root_logger()

    model = svgnet(cfg.model).cuda()
    if args.sync_bn:
        nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    gpu_num = dist.get_world_size()

    logger.info(f"Load state dict from {args.checkpoint}")
    load_checkpoint(args.checkpoint, logger, model)

    val_set = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(args, val_set, training=False, dist=args.dist, **cfg.dataloader.test)

    time_arr = []
    sem_point_eval = PointWiseEval(num_classes=cfg.model.semantic_classes, ignore_label=35, gpu_num=dist.get_world_size())
    instance_eval = InstanceEval(num_classes=cfg.model.semantic_classes, ignore_label=35, gpu_num=dist.get_world_size())

    # Create output directory if visualization is enabled
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        logger.info(f"Created output directory: {args.out}")

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dataloader):
            t1 = time.time()

            if i % 10 == 0:
                step = int(len(val_set)/gpu_num)
                logger.info(f"Infer  {i+1}/{step}")

            # Get filename if it exists in batch
            json_file = batch[-1][0] if len(batch) > 5 else None
            batch_data = batch[:-1] if json_file else batch

            torch.cuda.empty_cache()
            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                res = model(batch_data, return_loss=False)
                
                print(f"\nBatch {i}:")
                print_instance_classes(res)

            # Get semantic segmentation predictions
            sem_preds = torch.argmax(res["semantic_scores"], dim=1).cpu().numpy()
            sem_gts = res["semantic_labels"].cpu().numpy()
            sem_point_eval.update(sem_preds, sem_gts)
            instance_eval.update(res["instances"], res["targets"], res["lengths"])

            # Handle visualization if output directory is specified
            if args.out and json_file:
                svg_file = os.path.join(
                    "./dataset/test/test/svg_gt",
                    os.path.basename(json_file).replace("json", "svg"),
                )

                instances = res["instances"]
                if len(instances) > 0:
                    # Get maximum number of masks
                    num = max(len(instance["masks"]) for instance in instances)
                    estimated_contents = [{"instanceId": 0, "semanticId": 0}] * num

                    # Process each detected instance
                    for index, instance in enumerate(instances):
                        if instance["labels"] == 35 or instance["scores"] < 0.1:
                            continue
                        
                        for i in np.where(instance["masks"])[0]:
                            estimated_contents[i] = {
                                "instanceId": index + 1,
                                "semanticId": instance["labels"],
                            }

                    # Save the visualized SVG
                    reconstruct_svg(svg_file, estimated_contents, args.out)
                    logger.info(f"Saved visualization for {os.path.basename(svg_file)}")
            
            t2 = time.time()
            time_arr.append(t2 - t1)

    logger.info("Evaluate semantic segmentation")
    sem_point_eval.get_eval(logger)
    logger.info("Evaluate panoptic segmentation")
    instance_eval.get_eval(logger)
    
    mean_time = np.array(time_arr).mean()
    logger.info(f"Average run time: {mean_time:.4f}")

    if args.out:
        logger.info(f"All visualization results saved to {args.out}")

if __name__ == "__main__":
    main()