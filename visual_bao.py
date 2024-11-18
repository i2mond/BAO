import numpy as np
import torch
import cv2
import importlib
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from tool import imutils
import voc12.data
from PIL import Image
from torch.utils.data import DataLoader
import argparse

def visualize(normalized_heatmap ,original=None,):
    map_img = np.uint8(normalized_heatmap * 255)
    heatmap_img = cv2.applyColorMap(map_img, cv2.COLORMAP_JET)
    if original is not None:
        original_img = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        img = cv2.addWeighted(heatmap_img, 0.6, original_img, 0.4, 0)
    else:
        img = heatmap_img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    axes.imshow(img)
    label= str(img_label)

    ##Folder settings
    plt.imsave('../' + img_name + '_' + label + '.png', img, cmap='viridis')
    plt.close()

def visualize_img(name, label, label_20):
    img_path = voc12.data.get_img_path(name, voc12_root)
    orig_img = np.asarray(Image.open(img_path))
    orig_img_size = orig_img.shape[:2]
    img = transform(Image.open(img_path))
    _,_,_,_,cams = model('bao', img.unsqueeze(0).cuda(),label_20.cuda())
    cams = F.interpolate(cams, orig_img_size, mode='bilinear', align_corners=False).detach()
    cams = cams.cpu().numpy()[0][0:]
    cams[cams < 0] = 0
    cam_max = np.max(cams, (1, 2), keepdims=True)
    cam_min = np.min(cams, (1, 2), keepdims=True)
    norm_cam = (cams - cam_min) / (cam_max - cam_min + 1e-5)
    cam = norm_cam[label]
    visualize(cam, orig_img)

model = getattr(importlib.import_module('network.conformer_CAM'), 'Net_sm')()

##Weights settings
model.load_state_dict(torch.load('../..'), strict=False)
model = model.cuda()
model.eval()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    np.asarray,
    imutils.Normalize(),
    imutils.HWC_to_CHW,
    torch.from_numpy
])
voc12_root = 'VOCdevkit/VOC2012'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="network.conformer_CAM", type=str)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--voc12_root", default='VOCdevkit/VOC2012', type=str)
    parser.add_argument("--arch", default='sm', type=str)
    parser.add_argument("--method", default='bao', type=str)
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        np.asarray,
        imutils.Normalize(),
        imutils.HWC_to_CHW,
        torch.from_numpy
    ])
    infer_dataset = voc12.data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                    inter_transform=torchvision.transforms.Compose(
                                                        [
                                                        np.asarray,
                                                        imutils.Normalize(),
                                                        imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    with open('VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt') as f:
        for iter, ((img_name, img_list, label), img_name) in enumerate(zip(infer_data_loader, f)):
            new_label = label
            label = label[0]
            img_name = img_name.rstrip()
            with torch.no_grad():
                for img_label in range(21):
                    visualize_img(img_name, img_label,new_label)
                    
       