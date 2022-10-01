# -*- coding: UTF-8 -*-
import os
import cv2
import sys
import copy
import numpy as np

import numpy as np
import torch
import torch.nn as nn

from models.common import Conv
import torch

CUR_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(CUR_ROOT)
from utils.general import non_max_suppression_face, scale_coords, letterbox


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output
    
    
def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        model.append(torch.load(w, map_location=map_location)['model'].float().fuse().eval())  # load FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble
    
    
def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


class Yolo5FaceDetector(object):
    def __init__(self, weight=None, img_size=(640, 640), device='cuda:0', conf_thres = 0.6, iou_thres = 0.5) -> None:
        self.ckpt       = weight if weight is not None else f'{CUR_ROOT}/ckpts/yolov5m-face.pt'
        self.img_size   = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.device     = device
        self.model      = self._load_model()
        self.conf_thres = conf_thres
        self.iou_thres  = iou_thres
    
    def _load_model(self):
        return attempt_load(self.ckpt, map_location=self.device)

    def get_face(self, img0_mat, face_box, landmark):
        align_landmark = landmark.copy()
        x1, y1, x2, y2 = face_box
        face_mat = img0_mat[y1:y2, x1:x2, :]
        align_landmark[:, 0] -= x1
        align_landmark[:, 1] -= y1
        return face_mat, align_landmark
    
    def face_detect(self, img_file):
        img0 = cv2.imread(img_file)  # BGR
        assert img0 is not None, 'Image Not Found ' + img_file
        # print(f'image {self.count}/{self.nf} {img_file}: ', end='')
        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        orgimg = img.transpose(1, 2, 0)
        orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
        _img0 = copy.deepcopy(orgimg)
        img = letterbox(_img0, new_shape=self.img_size[0])[0]
        # Convert from w,h,c to c,w,h
        img = img.transpose(2, 0, 1).copy()

        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img)[0]
        
        # Apply NMS
        pred = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)
        # print(len(pred[0]), 'face' if len(pred[0]) == 1 else 'faces')

        # Process detections
        im0 = img0.copy()
        det = pred[0]    ## 单图结果
        
        faces_box_lst, landmarks_lst = [], []
        if len(det):       
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], im0.shape).round()

            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                # conf = det[j, 4].cpu().numpy()
                # class_num = det[j, 15].cpu().numpy()
                # landmarks = det[j, 5:15].view(-1).tolist() ## list
                landmarks_2d = det[j, 5:15].view(-1, 2).cpu().numpy()
            
                x1 = int(xyxy[0])
                y1 = int(xyxy[1])
                x2 = int(xyxy[2])
                y2 = int(xyxy[3])
                
                faces_box_lst.append([x1, y1, x2, y2])
                # face_mat = im0[y1:y2, x1:x2, :]
                
                ## 此处不做处理，后续处理                
                # landmarks_2d[:, 0] -= x1
                # landmarks_2d[:, 1] -= y1
                landmarks_lst.append(landmarks_2d)
        return img0, faces_box_lst, landmarks_lst


if __name__ == '__main__':
    pass