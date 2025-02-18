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
from svgnet.evaluation import PointWiseEval,InstanceEval
from svgnet.model.svgnet import SVGNet as svgnet
from svgnet.util  import get_root_logger, init_dist, load_checkpoint
from svgnet.data.svg import SVG_CATEGORIES

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
    #logger.info(model)
    
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    gpu_num = dist.get_world_size()

    logger.info(f"Load state dict from {args.checkpoint}")
    load_checkpoint(args.checkpoint, logger, model)

    val_set = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(args,val_set, training=False, dist=args.dist, **cfg.dataloader.test)

    time_arr = []
    sem_point_eval = PointWiseEval(num_classes=cfg.model.semantic_classes,ignore_label=35,gpu_num=dist.get_world_size())
    instance_eval = InstanceEval(num_classes=cfg.model.semantic_classes,ignore_label=35,gpu_num=dist.get_world_size())
    
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dataloader):
            t1 = time.time()

            if i % 10 == 0:
                step = int(len(val_set)/gpu_num)
                logger.info(f"Infer  {i+1}/{step}")
            torch.cuda.empty_cache()
            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                res = model(batch,return_loss=False)
                
                # Print classes for each instance
                print(f"\nBatch {i}:")
                print_instance_classes(res)
                
            # print("================================")
            # print(f"All the keys in res dict in test.py: {res.keys()}") # dict_keys(['semantic_scores', 'semantic_labels', 'instances', 'targets', 'lengths'])
            # print(f"Shape of res[semantic_scores] in test.py: {res['semantic_scores'].shape}") # torch.Size([N, 35])
            # print(f"Shape of res[semantic_labels] in test.py: {res['semantic_labels'].shape}") # torch.Size([N])
            # print(f"Shape of res[instances] in test.py: {len(res['instances'])}") # 11
            # print(f"Shape of res[targets] in test.py: {res['targets'].keys()}") # dict_keys(['labels', 'masks'])
            # print(f"Shape of res[lengths] in test.py: {res['lengths'].shape}") # torch.Size([N])
            
            # # Model returns a dictionary 'res' containing:
            # # - semantic_scores: class probabilities for each point
            # # - semantic_labels: ground truth labels
            # # - instances: list of detected instances
            # # - targets: ground truth instance information
            # # - lengths: lengths of primitives
            
            # print("================================")
            
            t2 = time.time()
            time_arr.append(t2 - t1)
            
            # Get semantic segmentation predictions
            sem_preds = torch.argmax(res["semantic_scores"],dim=1).cpu().numpy()
            sem_gts = res["semantic_labels"].cpu().numpy()
            sem_point_eval.update(sem_preds, sem_gts)
            instance_eval.update(
                res["instances"],
                res["targets"],
                res["lengths"],
            )
           
    logger.info("Evaluate semantic segmentation")
    sem_point_eval.get_eval(logger)
    logger.info("Evaluate panoptic segmentation")
    instance_eval.get_eval(logger)
    
    mean_time = np.array(time_arr).mean()
    logger.info(f"Average run time: {mean_time:.4f}")

    # If visualisation of outputs is false
    if not args.out:
        return

    #logger.info("Save results")
    


if __name__ == "__main__":
    main()
