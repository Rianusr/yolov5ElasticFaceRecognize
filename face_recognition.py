import os
import cv2
import sys
import numpy as np

## 导入人脸检测器、人脸对齐、人脸特征提取
from face_detector import Yolo5FaceDetector
from face_encoding import ElasticFaceEncoding
from face_align import insightface_alignmnet

CUR_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(CUR_ROOT)
from utils.face_utils import get_face_embeddings_db, cal_similarity_OvN, draw_face, save_face
from utils.general import getAllImgs


def face_recognition(face_detector, face_encoder, input_imgs, dist_thres=0.8, cosine_dist=True, face_db='./data/face_db', output='./data/recog_results', save_aligned=False):
    print('Loading face database ……')
    db_face_names, db_face_embeds = get_face_embeddings_db(face_detector, face_encoder, face_db)
    print('Start face recognition ……')
    for img_file in input_imgs:
        img0_mat, faces_box_lst, landmarks_lst = face_detector.face_detect(img_file)
        print(f'\tFind {len(faces_box_lst)} faces in: {img_file}')
        if not faces_box_lst:
            continue
        for face_box, landmark in zip(faces_box_lst, landmarks_lst):
            face_mat, landmark_align = face_detector.get_face(img0_mat, face_box, landmark)
            aligned_face = insightface_alignmnet(face_mat, landmark_align)
            if save_aligned:
                save_face(os.path.basename(img_file).rsplit('.', 1)[0], face_mat, aligned_face)
            face_embed = face_encoder.face_encoding(aligned_face)
            dist_ij = cal_similarity_OvN(face_embed, db_face_embeds, cosine_dist)
            recog_idx = np.argmin(dist_ij)
            if dist_ij[recog_idx] < dist_thres:
                face_name = db_face_names[recog_idx]
            else:
                face_name = '@Anonymous'
            img0_mat = draw_face(img0_mat, face_box, landmark, face_name)
        os.makedirs(output, exist_ok=True)
        cv2.imwrite(f"{output}/{os.path.basename(img_file)}", img0_mat)
    print('Face recognize done!')


'''
人脸识别流程：
    step1: 检测到人脸
    step2: 根据检测到的人脸的关键点进行人脸对齐
    step3: 使用对齐后的人脸，获取其特征表达
    step4: 与人脸库进行对比
距离阈值：需要经过大量的数据，才能对现有的模型做出更好的设定
dist_thres: 欧式距离: 1.0
            余弦距离: 0.2
'''

if __name__ == '__main__':
    yolov5_weight      = 'ckpts/yolov5m-face.pt'
    elasticface_weight = 'ckpts/arc+295672backbone.pth'
    face_detector = Yolo5FaceDetector(weight=yolov5_weight)
    face_encoder  = ElasticFaceEncoding(weight=elasticface_weight, arch='ir100')        ## ir100 or ir50
    all_imgs = getAllImgs('./data/images')
    face_recognition(face_detector, face_encoder, all_imgs, dist_thres=1.0, cosine_dist=False)