
from utils import *
from IMD_dataloader import *
from custom_loss import IsolatingLossFunction, load_center_radius
from tqdm import tqdm
from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_cfg_defaults
from models.SALocation import SALLoc

from sklearn import metrics

import torch
import torch.nn as nn
import argparse
import numpy as np

device = torch.device('cuda:0')
device_ids = [0]

def config(args):
    args.crop_size = [args.crop_size, args.crop_size]
    global device 
    device = torch.device('cuda:0')
    args.save_dir    = 'lr_' + str(args.learning_rate)+'_loc'
    FENet_dir, SegNet_dir = args.save_dir+'/HRNet', args.save_dir+'/SALocal'
    FENet_cfg = get_cfg_defaults()
    FENet  = get_seg_model(FENet_cfg).to(device) # load the pre-trained model
    SegNet = SALLoc().to(device)

    FENet  = nn.DataParallel(FENet, device_ids=device_ids)
    SegNet = nn.DataParallel(SegNet, device_ids=device_ids)

    writer = None

    return args, writer, FENet, SegNet, FENet_dir, SegNet_dir

def Inference(
                args, FENet, SegNet, LOSS_MAP, tb_writer, 
                iter_num=None, 
                save_tag=False, 
                localization=True
                ):

    for val_tag in [0,1,2,3,4]:

        val_data_loader, data_label = eval_dataset_loader_init(args, val_tag)
        print(f"working on the dataset: {data_label}.")
        F1_lst, auc_lst = [], []
        with torch.no_grad():
            FENet.eval()
            SegNet.eval()
            for step, val_data in enumerate(tqdm(val_data_loader)):
                image, mask, cls, image_names = val_data
                image, mask = image.to(device), mask.to(device)
                mask = torch.squeeze(mask, axis=1)
                output = FENet(image)
                mask1_fea, mask_binary, out0, out1, out2, out3 = SegNet(output, image)
                if args.loss_type == 'D':
                    pred_mask = LOSS_MAP.dis_curBatch.squeeze(dim=1)
                    pred_mask_score = LOSS_MAP.dist.squeeze(dim=1)
                elif args.loss_type == 'B':
                    pred_mask_score = mask_binary
                    pred_mask = torch.zeros_like(mask_binary)
                    pred_mask[mask_binary > 0.5] = 1
                    pred_mask[mask_binary <= 0.5] = 0
                viz_log(args, mask, pred_mask, image, iter_num, f"{step}_{val_tag}", mode='eval')

                mask = torch.unsqueeze(mask, axis=1)
                for img_idx, cur_img_name in enumerate(image_names):
                    mask_ = torch.unsqueeze(mask[img_idx,0], 0)
                    pred_mask_ = torch.unsqueeze(pred_mask[img_idx], 0)
                    pred_mask_score_ = torch.unsqueeze(pred_mask_score[img_idx], 0)
                    mask_ = mask_.cpu().clone().cpu().numpy().reshape(-1)
                    pred_mask_ = pred_mask_.cpu().clone().cpu().numpy().reshape(-1)
                    pred_mask_score_ = pred_mask_score_.cpu().clone().cpu().numpy().reshape(-1)
                    F1_a  = metrics.f1_score(mask_, pred_mask_, average='macro')
                    auc_a = metrics.roc_auc_score(mask_, pred_mask_score_)
                    pred_mask_[np.where(pred_mask_ == 0)] = 1
                    pred_mask_[np.where(pred_mask_ == 1)] = 0
                    F1_b  = metrics.f1_score(mask_, pred_mask_, average='macro')
                    if F1_a > F1_b:
                        F1 = F1_a
                    else:
                        F1 = F1_b
                    F1_lst.append(F1)
                    AUC_score = auc_a if auc_a > 0.5 else 1-auc_a
                    auc_lst.append(AUC_score)
        print("F1: ", np.mean(F1_lst))
        print("AUC: ", np.mean(auc_lst))

def main(args):
    args, writer, FENet, SegNet, FENet_dir, SegNet_dir = config(args)
    FENet  = restore_weight_helper(FENet,  "weights/HRNet",  350000)
    SegNet = restore_weight_helper(SegNet, "weights/SALocal", 350000)
    center, radius = load_center_radius(args, FENet, SegNet, 
                                        train_data_loader=False,
                                        center_radius_dir='./center_loc')
    Dice_loss  = nn.CrossEntropyLoss().to(device)
    BCE_loss = nn.BCELoss(reduction='none').to(device)
    LOSS_MAP = IsolatingLossFunction(center,radius).to(device)
    Inference(
                args, 
                FENet, 
                SegNet,
                LOSS_MAP,
                tb_writer=writer, 
                iter_num=99999, 
                save_tag=True, 
                localization=True
                )
    print("after saving the points...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=55)
    parser.add_argument('--step_factor', type=float, default=0.95)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--val_tag', type=int, default=0)
    parser.add_argument('--path', type=str, default="", help='deprecated')
    parser.add_argument('--percent', type=float, default=1.0, help='label dataset.')
    parser.add_argument('--loss_type', type=str, default='B', choices=['B', 'D'], help='ce or deep metric.')
    parser.add_argument('--initial_epoch', type=int, default=350000)
    args = parser.parse_args()
    main(args)
