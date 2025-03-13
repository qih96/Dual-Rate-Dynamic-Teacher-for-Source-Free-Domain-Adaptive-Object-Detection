import time
import datetime
import json

import copy

import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader

from datasets.coco_style_dataset import DataPreFetcher
from datasets.coco_eval import CocoEvaluator

from models.criterion import post_process, get_pseudo_labels
from utils.distributed_utils import is_main_process, all_gather, is_dist_avail_and_initialized
from utils.box_utils import box_cxcywh_to_xyxy, convert_to_xywh
from collections import defaultdict
from typing import List

from datasets.masking import Masking
from scipy.optimize import linear_sum_assignment
from utils.box_utils import box_cxcywh_to_xyxy, generalized_box_iou
import random
import torchvision
from utils.box_utils import box_cxcywh_to_xyxy
from utils.visualizer import COCOVisualizer
import torch.nn.functional as F


def merge_pseudo_labels(base, aux):
    mix = []
    extra = []
    for bi, ai in zip(base, aux):
        iou = generalized_box_iou(box_cxcywh_to_xyxy(bi['boxes']), box_cxcywh_to_xyxy(ai['boxes']))
        if iou.shape[0] != 0 and iou.shape[1] != 0:
            val, idx = iou.max(dim=0)
            valid_mask1 = torch.logical_and(val < 0.5, bi['labels'][idx] != ai['labels'])
            valid_mask2 = val < 0.2
            valid_mask = torch.logical_or(valid_mask1, valid_mask2)

            scores = torch.cat([bi['scores'], ai['scores'][valid_mask]], dim=0).clone().detach()
            boxes = torch.cat([bi['boxes'], ai['boxes'][valid_mask]], dim=0).clone().detach()
            labels = torch.cat([bi['labels'], ai['labels'][valid_mask]], dim=0).clone().detach()
            new = {"scores":scores, "boxes":boxes, "labels":labels}
            mix.append(new)
            ext = {"scores":ai['scores'][valid_mask], "boxes":ai['boxes'][valid_mask], "labels":ai['labels'][valid_mask]}
            extra.append(ext)
        elif iou.shape[1] != 0:
            mix.append(ai)
        else:
            mix.append(bi)
            valid_mask = torch.ones(bi['scores'].shape[0]) == 0
    if extra == []:
        extra = None
    return mix, extra


def vis_function(vslzr, category_info, pseudo_label, image, fig_dir, tag):
    box_label = [category_info[k] for k in pseudo_label['labels'].cpu().numpy().tolist()]
    pred_dict = {
        'boxes': pseudo_label['boxes'],
        'scores':pseudo_label['scores'],
        'size': torch.Tensor([image.shape[1], image.shape[2]]),
        'box_label': box_label,
        'image_id': tag
    }
    vslzr.visualize(image.cpu(), pred_dict, savedir=fig_dir, dpi=300, show_in_console=False)

def train_one_epoch_standard(model: torch.nn.Module,
                             criterion: torch.nn.Module,
                             data_loader: DataLoader,
                             optimizer: torch.optim.Optimizer,
                             device: torch.device,
                             epoch: int,
                             clip_max_norm: float = 0.0,
                             print_freq: int = 20,
                             flush: bool = True):
    """
    Train the standard detection model, using only labelled training set source.
    """
    start_time = time.time()
    model.train()
    criterion.train()
    fetcher = DataPreFetcher(data_loader, device=device)
    images, masks, annotations = fetcher.next()
    # Training statistics
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)
    epoch_loss_dict = defaultdict(float)
    for i in range(len(data_loader)):
        # Forward
        out = model(images, masks)
        # Loss
        loss, loss_dict = criterion(out, annotations)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        # Record loss
        epoch_loss += loss.detach()
        for k, v in loss_dict.items():
            epoch_loss_dict[k] += v.detach().cpu().item()
        # Data pre-fetch
        images, masks, annotations = fetcher.next()
        # Log
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Training epoch ' + str(epoch) + ' : [ ' + str(i + 1) + '/' + str(len(data_loader)) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
    # Final process of training statistic
    epoch_loss /= len(data_loader)
    for k, v in epoch_loss_dict.items():
        epoch_loss_dict[k] /= len(data_loader)
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Training epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_loss_dict


def train_one_epoch_teaching_standard(student_model: torch.nn.Module,
                                      teacher_model: torch.nn.Module,
                                      criterion_pseudo: torch.nn.Module,
                                      target_loader: DataLoader,
                                      optimizer: torch.optim.Optimizer,
                                      thresholds: List[float],
                                      alpha_ema: float,
                                      device: torch.device,
                                      epoch: int,
                                      clip_max_norm: float = 0.0,
                                      print_freq: int = 20,
                                      flush: bool = True,
                                      fix_update_iter: int = 1):
    """
    Train the student model with the teacher model, using only unlabeled training set target .
    """
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    criterion_pseudo.train()
    target_fetcher = DataPreFetcher(target_loader, device=device)
    target_images, target_masks, _ = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)

    # Training data statistics
    epoch_target_loss_dict = defaultdict(float)
    total_iters = len(target_loader)

    for iter in range(total_iters):
        # Target teacher forward
        with torch.no_grad():
            teacher_out = teacher_model(target_teacher_images, target_masks)
            pseudo_labels = get_pseudo_labels(teacher_out['logit_all'][-1], teacher_out['boxes_all'][-1], thresholds)

        # Target student forward
        target_student_out = student_model(target_student_images, target_masks)
        target_loss, target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels)

        loss = target_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        # Record epoch losses
        epoch_loss += loss.detach()

        # update loss_dict
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()

        if iter % fix_update_iter == 0:
            with torch.no_grad():
                state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                for key, value in state_dict.items():
                    state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                teacher_model.load_state_dict(state_dict)

        # Data pre-fetch
        target_images, target_masks, _ = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]

        # Log
        if is_main_process() and (iter + 1) % print_freq == 0:
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(iter + 1) + '/' + str(total_iters) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)

    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_target_loss_dict



def train_one_epoch_teaching_mask(student_model: torch.nn.Module,
                                  teacher_model: torch.nn.Module,
                                  or_student_model: torch.nn.Module,
                                  criterion_pseudo: torch.nn.Module,
                                  criterion_pseudo_weak: torch.nn.Module,
                                  target_loader: DataLoader,
                                  optimizer: torch.optim.Optimizer,
                                  thresholds: List[float],
                                  thresholds_s: List[float],
                                  coef_masked_img: float,
                                  alpha_ema: float,
                                  device: torch.device,
                                  epoch: int,
                                  clip_max_norm: float = 0.0,
                                  print_freq: int = 20,
                                  masking: Masking = None,
                                  flush: bool = True,
                                  fix_update_iter: int = 1,
                                  amp=False,
                                  data_loader_val=None,
                                  alpha_aema=None):
    """
    Train the student model with the teacher model, using only unlabeled training set target (plus masked target image)
    """
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    or_student_model.train()
    criterion_pseudo.train()
    criterion_pseudo_weak.train()
    target_fetcher = DataPreFetcher(target_loader, device=device)
    target_images, target_masks, _ = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)
    
    state_list = {}

    # Training data statistics
    epoch_target_loss_dict = defaultdict(float)
    total_iters = len(target_loader)

    vslzr = COCOVisualizer()
    if teacher_model.module.num_classes == 9:
        category_info = ['none', 'person', 'car', 'train', 'rider', 'truck', 'moto', 'bic', 'bus']
    else:
        category_info = ['none',  'car', 'none', 'none']

    count= 0.0
    training_count = 0.0
    
    for iter in range(total_iters):
        current_count = 0.0
        with torch.cuda.amp.autocast(enabled=amp):
            # Target teacher forward
            teacher_out = teacher_model(target_teacher_images, target_masks)
            with torch.no_grad():
                
                pseudo_labels = get_pseudo_labels(teacher_out['logit_all'][-1], teacher_out['boxes_all'][-1], thresholds, is_nms=False)
                or_student_model.load_state_dict(student_model.state_dict())
                student_out = or_student_model(target_teacher_images, target_masks)
                # student_out = student_model(target_teacher_images, target_masks)
                
                pseudo_labels_aux = get_pseudo_labels(student_out['logit_all'][-1], student_out['boxes_all'][-1],
                                                      thresholds_s, is_nms=False)

                mix_pseudo_labels, extra_labels = merge_pseudo_labels(pseudo_labels, pseudo_labels_aux)
                if extra_labels is not None:
                    for p in extra_labels:
                        midd = p['scores'].shape[0]
                        current_count += midd
                
                current_count = torch.as_tensor([current_count], dtype=torch.float, device=device)
                if is_dist_avail_and_initialized():
                    torch.distributed.all_reduce(current_count)
                current_count = current_count.item()
                count += current_count
            
            training_labels = mix_pseudo_labels
            # training_labels = pseudo_labels
            for i, p1 in enumerate(training_labels):
                midd = p1['scores'].shape[0]
                training_count += midd

            target_student_out = student_model(target_student_images, target_masks)
            # loss from pseudo labels of current teacher
            target_loss, target_loss_dict = criterion_pseudo(target_student_out, training_labels)

            masked_target_images = masking(target_student_images)
            masked_target_student_out = student_model(masked_target_images, target_masks)
            masked_target_loss, masked_target_loss_dict = criterion_pseudo(masked_target_student_out, training_labels)

            loss = target_loss + coef_masked_img * masked_target_loss
            if current_count >= (float(count) / ((iter + 1))):
                # teacher_out = teacher_model(target_teacher_images, target_masks)
                tea_target_loss, _ = criterion_pseudo_weak(teacher_out, mix_pseudo_labels)

        # Backward

        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        if current_count >= (float(count) / ((iter + 1))):
            teacher_model.zero_grad()
            tea_target_loss.backward()

            for n, p in teacher_model.named_parameters():
                if p.grad is not None:
                    grad = torch.as_tensor(p.grad, dtype=torch.float, device=loss.device)
                    if is_dist_avail_and_initialized():
                        torch.distributed.all_reduce(grad)
                    if n not in state_list.keys():
                        state_list[n] = torch.abs(grad)
                    else:
                        state_list[n] += torch.abs(grad)
            
        
        # Record epoch losses
        epoch_loss += loss.detach()

        # update loss_dict
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()


        # EMA update teacher after fix iteration
        if iter % fix_update_iter == 0:
            with torch.no_grad():
                state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                if current_count >= (float(count) / ((iter + 1))):
                # if current_count >= 2:
                    state_dict = update_ema_v1(student_state_dict, state_dict, state_list, alpha_aema, alpha_ema)
                else:
                    for key, value in state_dict.items():
                        state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                teacher_model.load_state_dict(state_dict)
            state_list = {}

        # Data pre-fetch
        target_images, target_masks, _ = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]

        # Log
        if is_main_process() and (iter + 1) % print_freq == 0:
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(iter + 1) + '/' + str(total_iters) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)

        if (iter + 1) % 1000 == 0 and (iter + 1) != 4000:
            ap50_per_class_student, loss_val_student = evaluate(
                model=teacher_model,
                criterion=criterion_pseudo,
                data_loader_val=data_loader_val,
                device=device,
                print_freq=print_freq,
                flush=flush
            )
    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    training_count = torch.as_tensor([training_count], dtype=torch.float, device=loss.device)
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(training_count)
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()) + ', extra: ' + str(count) + ',training: ' + str(training_count.item()), flush=flush)
    return epoch_loss, epoch_target_loss_dict


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader_val: DataLoader,
             device: torch.device,
             print_freq: int,
             output_result_labels: bool = False,
             flush: bool = False):
    start_time = time.time()
    model.eval()
    criterion.eval()
    if hasattr(data_loader_val.dataset, 'coco') or hasattr(data_loader_val.dataset, 'anno_file'):
        evaluator = CocoEvaluator(data_loader_val.dataset.coco)
        coco_data = json.load(open(data_loader_val.dataset.anno_file, 'r'))
        dataset_annotations = defaultdict(list)
    else:
        raise ValueError('Unsupported dataset type.')
    epoch_loss = 0.0
    for i, (images, masks, annotations) in enumerate(data_loader_val):
        # To CUDA
        images = images.to(device)
        masks = masks.to(device)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        # Forward
        out = model(images, masks)
        logit_all, boxes_all = out['logit_all'], out['boxes_all']
        # Get pseudo labels
        if output_result_labels:
            results = get_pseudo_labels(logit_all[-1], boxes_all[-1], [0.4 for _ in range(9)])
            for anno, res in zip(annotations, results):
                image_id = anno['image_id'].item()
                orig_image_size = anno['orig_size']
                img_h, img_w = orig_image_size.unbind(0)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h])
                converted_boxes = convert_to_xywh(box_cxcywh_to_xyxy(res['boxes'] * scale_fct))
                converted_boxes = converted_boxes.detach().cpu().numpy().tolist()
                for label, box in zip(res['labels'].detach().cpu().numpy().tolist(), converted_boxes):
                    pseudo_anno = {
                        'id': 0,
                        'image_id': image_id,
                        'category_id': label,
                        'iscrowd': 0,
                        'area': box[-2] * box[-1],
                        'bbox': box
                    }
                    # dataset_annotations[image_id].append(pseudo_anno)
                    dataset_annotations[image_id].append(pseudo_anno)
        # Loss
        loss, loss_dict = criterion(out, annotations)
        epoch_loss += loss
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Evaluation : [ ' + str(i + 1) + '/' + str(len(data_loader_val)) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
        # mAP
        orig_image_sizes = torch.stack([anno['orig_size'] for anno in annotations], dim=0)
        results = post_process(logit_all[-1], boxes_all[-1], orig_image_sizes, 100)
        results = {anno['image_id'].item(): res for anno, res in zip(annotations, results)}
        evaluator.update(results)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    aps = evaluator.summarize()
    epoch_loss /= len(data_loader_val)
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Evaluation finished. Time cost: ' + total_time_str, flush=flush)
    # Save results
    if output_result_labels:
        dataset_annotations_return = []
        id_cnt = 0
        # for image_anno in dataset_annotations:
        for image_anno in dataset_annotations.values():
            for box_anno in image_anno:
                box_anno['id'] = id_cnt
                id_cnt += 1
                dataset_annotations_return.append(box_anno)
        coco_data['annotations'] = dataset_annotations_return
        return aps, epoch_loss / len(data_loader_val), coco_data
    return aps, epoch_loss / len(data_loader_val)



def update_ema_v1(stu_state_dict, tea_state_dict, weight, alpha_aema, alpha_ema=0.999):

    total_p = []
    for k, v in weight.items():
        total_p.append(v.view(-1))
    total_p = torch.cat(total_p, dim=0)
    shape = total_p.shape[0]
    mid = torch.topk(total_p, k=int(shape/10))[0][-1]
    for key, value in tea_state_dict.items():
        if key in weight.keys():
            w = torch.ones_like(weight[key]) * alpha_ema
            mask = weight[key] > mid
            w[mask] = alpha_aema
            tea_state_dict[key] = w * value + (1 - w) * stu_state_dict[key].detach()
        else:
            tea_state_dict[key] = alpha_ema * value + (1 - alpha_ema) * stu_state_dict[key].detach()
    return tea_state_dict
