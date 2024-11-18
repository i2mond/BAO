import numpy as np
import torch
import os
import voc12.data
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils
import argparse
from PIL import Image
import torch.nn.functional as F
from tool.imutils import crf_inference_label
import imageio
from tool.visualization import VOClabel2colormap

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--network", default="network.conformer_CAM", type=str)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--voc12_root", default='VOCdevkit/VOC2012', type=str)
    #parser.add_argument("--out_npy", default='datas/out_npy/..', type=str)##npy
    parser.add_argument("--out_crf", default='datas/out_crfs/..', type=str)
    parser.add_argument("--out_pred", default='datas/out_pred/..', type=str)
    parser.add_argument("--arch", default='sm', type=str)
    parser.add_argument("--method", default='bao', type=str)
    args = parser.parse_args()

    ##npy
    # if not os.path.exists(args.out_npy):
    #     os.makedirs(args.out_npy)
    if not os.path.exists(args.out_pred):
        os.makedirs(args.out_pred)
    if not os.path.exists(args.out_crf):
        os.makedirs(args.out_crf)

    model = getattr(importlib.import_module(args.network), 'Net_' + args.arch)()
    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()
    
    infer_dataset = voc12.data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                  inter_transform=torchvision.transforms.Compose(
                                                      [
                                                       np.asarray,
                                                       imutils.Normalize(),
                                                       imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):

        label_cls = label.cuda()

        img_name = img_name[0]; label = label[0]

        img_path = voc12.data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]

        bg_score = torch.ones((1))
        label = torch.cat((bg_score, label), dim=0)

        cam_list = []
        with torch.no_grad():
            for i, img in enumerate(img_list):

                if args.method == 'bao':
                    _, _, _, _, cam = model(args.method, img.cuda(), label_cls)

                cam = F.interpolate(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                cam = cam.cpu().numpy() * label.clone().view(21, 1, 1).numpy()

                ##npy
                # cam = F.interpolate(cam[:, 1:, :, :], orig_img_size, mode='bilinear', align_corners=False)[0]
                # cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
        
                if i % 2 == 1:
                    cam = np.flip(cam, axis=-1)
                cam_list.append(cam)
        
        sum_cam = np.sum(cam_list, axis=0)

        sum_cam[sum_cam < 0] = 0
        cam_max = np.max(sum_cam, (1,2), keepdims=True)
        cam_min = np.min(sum_cam, (1,2), keepdims=True)
        sum_cam[sum_cam < cam_min+1e-5] = 0
        norm_cam = (sum_cam-cam_min-1e-5) / (cam_max - cam_min + 1e-5)

        ##npy
        # cam_dict = {}
        # for i in range(20):
        #     if label[i] > 1e-5:
        #         cam_dict[i] = norm_cam[i]

        # if args.out_cam is not None:
        #    np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)

        if args.out_cam_pred is not None:
            pred = np.argmax(norm_cam, 0)
            pred = VOClabel2colormap(pred)
            imageio.imwrite(os.path.join(args.out_cam_pred, img_name + '.png'), pred.astype(np.uint8))
        
        if args.out_crf is not None:
            pred = np.argmax(norm_cam, 0)
            crf = crf_inference_label(orig_img, pred)
            folder = args.out_crf
            if not os.path.exists(folder):
                os.makedirs(folder)
            Image.fromarray(crf.astype(np.uint8)).save(os.path.join(folder, img_name + '.png'))

        print(iter)
