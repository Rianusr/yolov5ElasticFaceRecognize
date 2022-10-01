import os
import cv2
import sys

import torch

CUR_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(CUR_ROOT)
from models.elasticFace.iresnet import iresnet100, iresnet50

BACKBONE_ARCHS = ['ir100', 'ir50']


def get_transform():
    from torchvision import transforms
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
    return transform


def get_input_image(cvmat, with_flip):
    transform = get_transform()
    # img = cv2.imread(img_file)
    # cvmat = cv2.resize(img, (112, 112))
    sample = cv2.cvtColor(cvmat, cv2.COLOR_BGR2RGB)
    flip_sample = None
    if with_flip:
        flip_img = cv2.flip(sample, 1)
        flip_sample = transform(flip_img).unsqueeze(0)
    sample = transform(sample).unsqueeze(0)
    return sample, flip_sample


class ElasticFaceEncoding(object):
    def __init__(self, weight=None, arch='ir100', device='cuda:0', with_flip=True) -> None:
        self.ckpt       = weight if weight is not None else f'{CUR_ROOT}/ckpts/arc+295672backbone.pth'
        self.arch       = arch
        self.device     = device
        self.with_flip  = with_flip
        self.model      = self._load_model()
        
    def _load_model(self):
        if self.arch == 'ir100':
            backbone = iresnet100(num_features=512).to(self.device)
        elif self.arch == 'ir50':
            backbone = iresnet50(num_features=512).to(self.device)
        else:
            raise ValueError(f'Only {BACKBONE_ARCHS} are accepted!')
        backbone.load_state_dict(torch.load(self.ckpt))
        backbone.eval()
        return backbone
    
    def face_encoding(self, cvmat):
        sample, flip_sample = get_input_image(cvmat, self.with_flip)
        sample_out: torch.Tensor = self.model(sample.to(self.device))
        embeddings = sample_out.detach().cpu().numpy()
        if flip_sample is not None:
            flip_sample_out: torch.Tensor = self.model(flip_sample.to(self.device))
            embeddings += flip_sample_out.detach().cpu().numpy()
        return embeddings