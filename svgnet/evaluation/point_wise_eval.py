import numpy as np
from svgnet.data.svg import SVG_CATEGORIES
import torch
import torch.distributed as dist

"""
This code which implements evaluation metrics for semantic and instance segmentation tasks
"""


class PointWiseEval(object):
    """Class for evaluating point-wise segmentation performance.
    
    This class computes confusion matrices and evaluates metrics such as accuracy (ACC) and 
    mean Intersection over Union (mIoU) for semantic segmentation.
    
    Attributes:
        ignore_label (int): Label to ignore in evaluation.
        _num_classes (int): Number of semantic classes.
        _conf_matrix (np.ndarray): Confusion matrix for evaluation.
        _b_conf_matrix (np.ndarray): Confusion matrix for binary segmentation.
        _class_names (list): Names of the semantic classes.
        gpu_num (int): Number of GPUs used for distributed evaluation.
    """
    def __init__(self, num_classes=35, ignore_label=35,gpu_num=1) -> None:
        """
        - Takes number of classes (default 35), an ignore label, and number of GPUs
        - Creates a confusion matrix of size (num_classes + 1) × (num_classes + 1) for evaluation
        - Gets class names from SVG_CATEGORIES (excluding the last one)
        """
        self.ignore_label = ignore_label
        self._num_classes = num_classes
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.float32)
        # self._b_conf_matrix = np.zeros(
        #     (self._num_classes + 1, self._num_classes + 1), dtype=np.int64
        # ) # Not used in entire repo
        self._class_names = [x["name"] for x in SVG_CATEGORIES[:-1]]
        self.gpu_num = gpu_num
        
    def update(self, pred_sem, gt_sem):
        """
           - Creates a mask where class 35 isnt present and calculates predictions, groundtruths at those locations in matrix
           - Update the confusion matrix with predictions and ground truth labels.
        
        Args:
            pred_sem (np.ndarray): Predicted semantic labels.
            gt_sem (np.ndarray): Ground truth semantic labels.
        """
        pos_inds = gt_sem != self.ignore_label # Boolean mask which True at places where gt_sem is not ignore_label
        pred = pred_sem[pos_inds]
        gt = gt_sem[pos_inds]

        self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape) # efficiently count matches and update the confusion matrix

        

    def get_eval(self, logger):
        """Computes and logs evaluation metrics.
        
        Args:
            logger (Logger): Logger for logging evaluation metrics.
        
        Returns:
            tuple: Mean IoU (mIoU), Pixel accuracy (pACC)
        """
        if self.gpu_num>1: # Check if multiGPU training/evaluation
            t =  torch.from_numpy(self._conf_matrix).to("cuda") # Convert numpy confusion matrix into a tensor and send it to GPU
            conf_matrix_list = [torch.full_like(t,0) for _ in range(self.gpu_num)] # Create list of empty tensors, one for each GPU
            dist.barrier() # Synchronisation point, makes sure all GPUs are ready before gathering data
            dist.all_gather(conf_matrix_list,t)  # Gathers various local confusion matrices from different GPUs
            self._conf_matrix = torch.full_like(t,0) # Create a new confusion matrix to compile all local confusion matrices
            # Accumulates results from all GPUs into a single confusion matrix
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix
            self._conf_matrix = self._conf_matrix.cpu().numpy() # Moves the final combined confusion matrix back to CPU and convert it to numpy array
        
        # mIoU
        acc = np.full(self._num_classes, np.nan, dtype=np.float64) # Accuracy (acc): tp / pos_gt
        iou = np.full(self._num_classes, np.nan, dtype=np.float64) # IoU: tp / (pos_gt + pos_pred - tp)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float64) # True Positives (tp): Diagonal elements of confusion matrix
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float64) # Ground Truth Positives (pos_gt): Sum along columns
        class_weights = pos_gt / (np.sum(pos_gt)+1e-8)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float64) # Predicted Positives (pos_pred): Sum along rows
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / (pos_gt[acc_valid]+1e-8)
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / (union[iou_valid]+1e-8)
        macc = np.sum(acc[acc_valid]) / (np.sum(acc_valid)+1e-8) # Mean Accuracy (macc): Average of valid accuracies
        miou = np.sum(iou[iou_valid]) / (np.sum(iou_valid)+1e-8) # Mean IoU (miou): Average of valid IoUs
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid]) # Frequency weighted IoU (fiou): IoU weighted by class frequency
        pacc = np.sum(tp) / (np.sum(pos_gt)+1e-8) # Pixel Accuracy (pacc): Total correct pixels / total pixels

        miou, fiou, pACC = 100 * miou, 100 * fiou, 100 * pacc
        for i, name in enumerate(self._class_names):
            logger.info('Class_{}  IoU: {:.3f}'.format(name,iou[i]*100))
        
        logger.info('mIoU / fwIoU / pACC : {:.3f} / {:.3f} / {:.3f}'.format(miou, fiou, pACC))
        
        return miou, pACC

class InstanceEval(object):
    """Class for evaluating instance-wise segmentation performance.
    
    This class computes Precision, Recall, and IoU for instance segmentation 
    based on detected object instances.
    """
    def __init__(self, num_classes=35,
                 ignore_label=35,
                 gpu_num=8) -> None:

        self.ignore_label = ignore_label
        self._num_classes = num_classes
        self._class_names = [x["name"] for x in SVG_CATEGORIES[:-1]]
        self.gpu_num = gpu_num
        self.min_obj_score = 0.1
        self.IoU_thres = 0.5

        self.tp_classes = np.zeros(num_classes)
        self.tp_classes_values = np.zeros(num_classes)
        self.fp_classes = np.zeros(num_classes)
        self.fn_classes = np.zeros(num_classes)
        self.thing_class = [i for i in range(30)]
        self.stuff_class = [30,31,32,33,34]

    def update(self, instances, target, lengths):
        """Updates evaluation metrics based on detected instances and ground truth targets.
        
        Args:
            instances (list): List of detected instances.
            target (dict): Dictionary containing ground truth labels and masks.
            lengths (torch.Tensor): Length of each detected object instance.
        """
        lengths = np.round( np.log(1 + lengths.cpu().numpy()) , 3)
        tgt_labels = target["labels"].cpu().numpy().tolist()
        tgt_masks = target["masks"].transpose(0,1).cpu().numpy()
        for tgt_label, tgt_mask in zip(tgt_labels, tgt_masks):
            if tgt_label==self.ignore_label: continue

            flag = False
            for instance in instances:
                src_label = instance["labels"]
                src_score = instance["scores"]
                if src_label==self.ignore_label: continue
                if src_score< self.min_obj_score: continue
                src_mask = instance["masks"]
                
                interArea = sum(lengths[np.logical_and(src_mask,tgt_mask)])
                unionArea = sum(lengths[np.logical_or(src_mask,tgt_mask)])
                iou = interArea / (unionArea + 1e-6)
                if iou>=self.IoU_thres:
                    flag = True
                    if tgt_label==src_label:
                        self.tp_classes[tgt_label] += 1
                        self.tp_classes_values[tgt_label] += iou
                    else:
                        self.fp_classes[src_label] += 1
            if not flag: self.fn_classes[tgt_label] += 1
    
    def get_eval(self, logger):
        """Computes and logs instance segmentation evaluation metrics.
        
        Args:
            logger (Logger): Logger for logging evaluation metrics.
        
        Returns:
            tuple: Segmentation PQ, RQ, and SQ metrics.
        """
        if self.gpu_num>1:
            _tensor = np.stack([self.tp_classes,
                               self.tp_classes_values,
                               self.fp_classes,
                               self.fn_classes])
            _tensor = torch.from_numpy(_tensor).to("cuda")
            _tensor_list = [torch.full_like(_tensor,0) for _ in range(self.gpu_num)]
            dist.barrier()
            dist.all_gather(_tensor_list,_tensor)
            all_tensor = torch.full_like(_tensor,0)
            for tensor_ in _tensor_list:
                all_tensor += tensor_

            all_tensor = all_tensor.cpu().numpy()
            self.tp_classes, self.tp_classes_values, \
                self.fp_classes, self.fn_classes= all_tensor

       # each class
        RQ = self.tp_classes / (self.tp_classes + 0.5* self.fp_classes + 0.5* self.fn_classes + 1e-6)
        SQ = self.tp_classes_values / (self.tp_classes + 1e-6)
        PQ = RQ * SQ
        
        # thing
        thing_RQ = sum(self.tp_classes[self.thing_class]) / (sum(self.tp_classes[self.thing_class]) + 0.5* sum(self.fp_classes[self.thing_class]) + 0.5* sum(self.fn_classes[self.thing_class]) + 1e-6)
        thing_SQ = sum(self.tp_classes_values[self.thing_class]) / (sum(self.tp_classes[self.thing_class]) + 1e-6)
        thing_PQ = thing_RQ * thing_SQ
        
        # stuff
        stuff_RQ = sum(self.tp_classes[self.stuff_class]) / (sum(self.tp_classes[self.stuff_class]) + 0.5* sum(self.fp_classes[self.stuff_class]) + 0.5* sum(self.fn_classes[self.stuff_class]) + 1e-6)
        stuff_SQ = sum(self.tp_classes_values[self.stuff_class]) / (sum(self.tp_classes[self.stuff_class]) + 1e-6)
        stuff_PQ = stuff_RQ * stuff_SQ
        
        #total
        sRQ = sum(self.tp_classes) / (sum(self.tp_classes) + 0.5* sum(self.fp_classes) + 0.5* sum(self.fn_classes) + 1e-6)
        sSQ = sum(self.tp_classes_values) / (sum(self.tp_classes) + 1e-6)
        sPQ = sRQ * sSQ
        
        for i, name in enumerate(self._class_names):
            logger.info('Class_{}  PQ: {:.3f}'.format(name,PQ[i]*100))

        logger.info('PQ / RQ / SQ : {:.3f} / {:.3f} / {:.3f}'.format(sPQ*100, sRQ*100, sSQ*100))
        logger.info('thing PQ / RQ / SQ : {:.3f} / {:.3f} / {:.3f}'.format(thing_PQ*100, thing_RQ*100, thing_SQ*100))
        logger.info('stuff PQ / RQ / SQ : {:.3f} / {:.3f} / {:.3f}'.format(stuff_PQ*100, stuff_RQ*100, stuff_SQ*100))
        return sPQ*100, sRQ*100, sSQ*100