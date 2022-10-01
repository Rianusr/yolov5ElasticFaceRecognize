import os
import cv2
import sys
import numpy as np
from sklearn import preprocessing

CUR_ROOT = os.path.abspath(os.path.dirname(__file__).replace('/utils', ''))
sys.path.append(CUR_ROOT)
from utils.general import getAllImgs
from face_align import insightface_alignmnet


def getCosSimilarVecVec(vec1, vec2):
    num = float(np.dot(vec1, vec2))  # 向量点乘
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


# ! 向量与矩阵之间的余弦相似度
def getCosSimilarVecMat(vec: list, mat: list):
    num = np.dot([vec], np.array(mat).T)  # 向量点乘
    denom = np.linalg.norm(vec) * np.linalg.norm(mat, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res


# ! 矩阵与矩阵质检的余弦相似度
def getCosSimilarMatMat(mat1, mat2):
    num = np.dot(mat1, np.array(mat2).T)  # 向量点乘
    denom = np.linalg.norm(mat1, axis=1).reshape(-1, 1) * \
        np.linalg.norm(mat2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res


def cal_similarity_OvO(embeddings_i, embeddings_j, cosine_dist=False):
    embeddings_ni = preprocessing.normalize(embeddings_i)
    embeddings_nj = preprocessing.normalize(embeddings_j)
    ## 欧式距离
    dist_ij = np.sum(np.square(np.subtract(embeddings_i, embeddings_j)), 1)[0]       ## without norm
    dist_nij = np.sum(np.square(np.subtract(embeddings_ni, embeddings_nj)), 1)[0]
    if cosine_dist:
        ## 余弦距离
        dist_ij = getCosSimilarVecVec(embeddings_i.flatten(), embeddings_j.flatten())   ## without norm
        dist_nij = getCosSimilarVecVec(embeddings_ni.flatten(), embeddings_nj.flatten())
    return dist_ij, dist_nij


def cal_similarity_OvN(face_embed, db_embeddings, cosine_dist=True):
    if cosine_dist:
        ##  余弦距离无需归一化
        simlaritys  = getCosSimilarVecMat(face_embed.flatten(), db_embeddings)
        dist_iN = 1 - simlaritys[0]
    else:
        face_embed_n = preprocessing.normalize(face_embed)
        db_embeddings_n = preprocessing.normalize(db_embeddings)
        ## 欧式距离
        dist_iN = np.sum(np.square(np.subtract(face_embed_n, db_embeddings_n)), 1)
    return dist_iN


def get_face_embeddings_db(face_detector, face_encoder, face_db):
    all_imgs = getAllImgs(face_db)
    face_names, face_embeds = [], []
    for img_f in all_imgs:
        img_name = os.path.basename(img_f).rsplit('.', 1)[0]
        img0_mat, faces_box_lst, landmarks_lst = face_detector.face_detect(img_f)
        if not faces_box_lst:
            print(f'Not find face in: {img_f}, Please check it out!')
            continue
        face_box = faces_box_lst[0]     ## 默认只有一张脸
        landmark = landmarks_lst[0]
        face_mat, landmark_align = face_detector.get_face(img0_mat, face_box, landmark)
        aligned_face_i = insightface_alignmnet(face_mat, landmark_align)#, dst_shape=face_mat.shape[:2][::-1])
        embeddings = face_encoder.face_encoding(aligned_face_i)
        face_names.append(img_name)
        face_embeds.append(embeddings)
    return face_names, np.vstack(face_embeds)


def save_face(face_name, face_mat, aligned_face, output='./data/aligned_results', interval=5):
    if not os.path.exists(output):
        os.makedirs(output)
    
    face_size  = face_mat.shape[:2]         
    align_size = aligned_face.shape[:2]     
    ratio = max(face_size[0] / align_size[0], face_size[1] / align_size[1])     ## 1.196
    r_h, r_w = int(face_size[0] / ratio), int(face_size[1] / ratio)
    face_mat = cv2.resize(face_mat, (r_w, r_h))

    output_mask = np.zeros((align_size[0], align_size[0] * 2 + interval, 3))
    output_mask[0:r_h, 0:r_w, :] = face_mat
    output_mask[:, align_size[0]+interval:, :] = aligned_face
    cv2.imwrite(f'data/aligned_results/{face_name}.jpg', output_mask)


def draw_face(cvmat, face_box, landmark, face_name, color=(255, 0, 0), line_thickness=None):
    tl = line_thickness or round(0.002 * (cvmat.shape[0] + cvmat.shape[1]) / 2) + 1  # line/font thickness
    
    x1, y1, x2, y2 = face_box
    c1, c2 =(x1, y1), (x2, y2)
    ## 画框和关键点
    cv2.rectangle(cvmat, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    for point in landmark:
        point = tuple(map(int, point))
        cv2.circle(cvmat, point, 1, (0, 255, 0), 2)
        
    ## 标记名称
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(face_name, 0, fontScale=tl / 3, thickness=tf)[0]
    if c1[1] >= t_size[1] + 5:
        c1 = (c1[0], c1[1] - t_size[1] - 5)
    c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 5
    ## 背景框
    cv2.rectangle(cvmat, c1, c2, color, -1, cv2.LINE_AA)  # filled
    pos = (c1[0], c1[1] + t_size[1])
    cv2.putText(cvmat, face_name, pos, 0, tl / 3, (225, 255, 255), thickness=tf, lineType=cv2.LINE_AA)
    return cvmat
