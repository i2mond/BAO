import numpy as np
import torch
import os
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils
import argparse
import importlib
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import time
from PIL import Image
from torch.autograd import Variable
import imageio
from tqdm import tqdm

def balanced_cross_entropy(logits, labels, one_hot_labels, ep, w):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """

    N, C, H, W = logits.shape

    assert one_hot_labels.size(0) == N and one_hot_labels.size(1) == C, f'label tensor shape is {one_hot_labels.shape}, while logits tensor shape is {logits.shape}'
    
    log_logits = F.log_softmax(logits, dim=1)# log_logits, one_hot_labels : 8, 21, 32, 32
    loss_structure = -torch.sum(log_logits * one_hot_labels, dim=1)  # (N) loss_structure : 8, 32, 32, 

    ignore_mask_bg = torch.zeros_like(labels)# labels, ignore_mask_bg : 8, 32, 32
    ignore_mask_fg = torch.zeros_like(labels)
    
    ignore_mask_bg[labels == 0] = 1
    ignore_mask_fg[(labels != 0) & (labels != 255)] = 1
    
    if ep == 0:
        ignore_mask_fg = ignore_mask_fg * w
    
    loss_bg = (loss_structure * ignore_mask_bg).sum() / (ignore_mask_bg.sum())
    loss_fg = (loss_structure * ignore_mask_fg).sum() / (ignore_mask_fg.sum())

    return (loss_bg+loss_fg)/2

def share_model_weights(model):
    return model.state_dict()

def apply_shared_weights(model, shared_state_dict):
    model.load_state_dict(shared_state_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=10, type=int)
    parser.add_argument("--network", default="network.conformer_CAM", type=str)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--arch", default='sm', type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--session_name", default="log_pth/..", type=str)
    parser.add_argument("--crop_size", default=512, type=int)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--voc12_root", default='VOCdevkit/VOC2012', type=str)
    parser.add_argument("--tblog_dir", default='./tblog', type=str)
    parser.add_argument("--save_dir", default='./', type=str)
    parser.add_argument("--method", default='bao', type=str)
    parser.add_argument("--eps", default=0.4, type=float)
    parser.add_argument("--momentum", type=float, default=0.9995)
    parser.add_argument("--l_pcl", type=float, default=1.0)
    parser.add_argument("--w_pcl", type=float, default=0.5)
    torch.autograd.set_detect_anomaly(True)
    args = parser.parse_args()

    pyutils.Logger(args.session_name + '.log')
    print(vars(args))
    model_t = getattr(importlib.import_module(args.network), 'Net_' + args.arch)()
    model_s = getattr(importlib.import_module(args.network), 'Net_' + args.arch)()
    print(model_s)
    tblogger = SummaryWriter(args.tblog_dir)

    train_dataset = voc12.data.VOC12ClsDataset(args.train_list, voc12_root=args.voc12_root,
                                               transform=transforms.Compose([
                                                   imutils.RandomResizeLong(320, 640),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
                                                                          hue=0.1),
                                                   np.asarray,
                                                   imutils.Normalize(),
                                                   imutils.RandomCrop(args.crop_size),
                                                   imutils.HWC_to_CHW,
                                                   torch.from_numpy
                                               ]))

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    optimizer = optim.AdamW(model_s.parameters(), lr=args.lr, weight_decay=args.wt_dec, eps=1e-8)

    model_t.load_state_dict(torch.load(args.weights), strict=False)###
    model_t = torch.nn.DataParallel(model_t).cuda()
    for p in model_t.parameters():
        p.requires_grad = False
    
    model_s.load_state_dict(torch.load(args.weights), strict=False)###
    model_s = torch.nn.DataParallel(model_s).cuda()
    model_s.train()

    momentum_schedule = pyutils.cosine_scheduler(args.momentum, 1.0, 1322)
    _schedule = np.repeat(1.0, 0)
    momentum_schedule = np.concatenate([momentum_schedule, _schedule], axis=0)
    avg_meter = pyutils.AverageMeter('loss')

    timer = pyutils.Timer('train_bao_beginning:')
    start_time = time.time()

    for ep in range(args.max_epoches):
        loader_iter = iter(train_data_loader)

        pbar = tqdm(
            range(1, len(train_data_loader) + 1),
            total=len(train_data_loader),
            dynamic_ncols=True,
        )

        for iteration, _ in enumerate(pbar):
            try:
                pack = next(loader_iter)
            except:
                loader_iter = iter(train_data_loader)
                pack = next(loader_iter)

            img_name, img, label_cls = pack

            label_cls = label_cls.cuda()
            img_name = img_name[0]; 
            img_path = voc12.data.get_img_path(img_name, args.voc12_root)
            orig_img = np.asarray(Image.open(img_path))
            orig_img_size = orig_img.shape[:2]

            if args.method == 'bao' :
                logits_conv_t, logits_trans_t, am_t, tcams_t, _ = model_t(args.method, img, label_cls)
                logits_conv_s, logits_trans_s, am_s, tcams_s, _ = model_s(args.method, img, label_cls)
                loss  = F.multilabel_soft_margin_loss((logits_conv_s + logits_trans_s).unsqueeze(2).unsqueeze(3)[:, 1:, :, :], label_cls.unsqueeze(2).unsqueeze(3))
                
                cam = tcams_t[:, 1:, :, :].detach().cpu().numpy() * label_cls.clone().view(8, 20, 1, 1).cpu().numpy()
                cams_fg = cam
                norm_cam = np.maximum(cams_fg, 0)
                norm_cam = norm_cam / (np.max(norm_cam, axis=(2, 3), keepdims=True) + 1e-5)
                cams_bg = 1 - np.max(norm_cam, axis=1, keepdims=True)
                norm_cam = np.concatenate([cams_bg, norm_cam], axis=1)
                label_cam = np.argmax(norm_cam, 1)
    
                label_cam = torch.from_numpy(label_cam)
             
                B, C, H, W = am_s.shape

                label_ = label_cam.clone()
                label_[label_cam == 255] = 0
                
                given_labels = torch.full(size=(B, C, H, W), fill_value=args.eps/(C-1)).cuda()
                label_ = label_.cuda()
                given_labels.scatter_(dim=1, index=torch.unsqueeze(label_, dim=1), value=1-args.eps)

                device = am_s.device
                label_cam = label_cam.to(device)
                given_labels = given_labels.to(device)

                losspcl = balanced_cross_entropy(am_s, label_cam, given_labels, ep, args.w_pcl)
                
                loss = loss + args.l_pcl*losspcl
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if ep == 0:
                    shared_state_dict = share_model_weights(model_s)
                    apply_shared_weights(model_t, shared_state_dict)

                avg_meter.add({'loss': loss.item()})
                #iteration = iteration + ep*1322
                if ep > 0:
                    ##EMA update
                    with torch.no_grad():
                        m = momentum_schedule[iteration]  # momentum parameter
                        for param_q, param_k in zip(model_s.module.parameters(), model_t.parameters()):
                            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        else:
            print('epoch: %5d' % ep,
                'loss: %.4f' % avg_meter.get('loss'), flush=True
                )
            avg_meter.pop()
            ep_num = str(ep)
            torch.save(model_t.module.state_dict(), os.path.join(args.save_dir, args.session_name + '_' + ep_num + 'epochs'+ '_' + "modelt" + '.pth'))   
            torch.save(model_s.module.state_dict(), os.path.join(args.save_dir, args.session_name + '_' + ep_num + 'epochs'+ '_' + "models"+ '.pth'))

    timer = pyutils.Timer('train_bao_over!!!')
    print('run_time:{} h'.format(round((time.time() - start_time) / 60 / 60, 4)))