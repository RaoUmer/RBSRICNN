import torch
import cv2
import numpy as np

def decompress(x):
    r""" Decompress an image.

         Input:
            x with shape [b, 4, H/2, W/2]
         Return:
            decompressed_image of shape [b, 3, H, W]
     """
    size = x.shape
    if x.is_cuda:
        decompressed_image = torch.cuda.FloatTensor(size[0],  # number of batches
                                                    3,  # number of bayer channels
                                                    int(size[2] * 2),  # H
                                                    int(size[3] * 2)).fill_(0)
    else:
        decompressed_image = torch.FloatTensor(size[0],  # number of batches
                                               3,  # number of bayer channels
                                               int(size[2]* 2),  # H
                                               int(size[3]* 2)).fill_(0)

    decompressed_image[:, 0, ::2, ::2] = x[:, 0]  # G
    decompressed_image[:, 1, ::2, 1::2] = x[:, 1]  # R
    decompressed_image[:, 1, 1::2, ::2] = x[:, 3]  # G
    decompressed_image[:, 2, 1::2, 1::2] = x[:, 2]  # B
    return decompressed_image

def calculate_affine_matrices(burst):
    burst = decompress(burst)
    burst = burst.permute(0,2,3,1).numpy()
    warp_matrices = np.zeros((burst.shape[0], 2, 3))
    warp_matrices[0] = np.eye(2, 3, dtype=np.float32)  # identity for reference frame
    i=0
    for _, b in enumerate(burst[1:]):
        i = i+1
        warp_matrix = calculate_ECC((burst[0]).astype(np.float32), (b).astype(np.float32), nol=2)
        if warp_matrix is None:
            return None
        
        warp_matrices[i] = warp_matrix
    return warp_matrices


def calculate_ECC(img_ref, img, nol=4):
    img_ref = bayer_to_gray(img_ref)
    img = bayer_to_gray(img)
    
    # ECC params
    init_warp = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    n_iters = 3000 # 3000
    e_thresh = 1e-6
    warp_mode = cv2.MOTION_EUCLIDEAN
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iters, e_thresh)
    warp = init_warp
    
    # construct grayscale pyramid
    gray1_pyr = [img_ref]
    gray2_pyr = [img]

    for level in range(nol - 1):
        gray1_pyr.insert(0, cv2.resize(gray1_pyr[0], None, fx=1 / 2, fy=1 / 2,
                                       interpolation=cv2.INTER_AREA))
        gray2_pyr.insert(0, cv2.resize(gray2_pyr[0], None, fx=1 / 2, fy=1 / 2,
                                       interpolation=cv2.INTER_AREA))
    
    # run pyramid ECC
    error_cnt = 0
    for level in range(nol):
        try:
            cc, warp_ = cv2.findTransformECC(gray1_pyr[level], gray2_pyr[level],
                                             warp, warp_mode, criteria, None, 1)
            warp = warp_
            if level != nol - 1:  # scale up for the next pyramid level
                warp = warp * np.array([[1, 1, 2], [1, 1, 2]], dtype=np.float32)
        except Exception as e:
            error_cnt += 1
            pass

    if error_cnt == nol:
        warp = np.eye(2, 3, dtype=np.float32)
        return warp
    else:
        return warp


def bayer_to_gray(img):
    g1 = img[1::2, ::2, 1]
    g2 = img[::2, 1::2, 1]
    g = (g1 + g2) / 2
    r = img[::2, ::2, 0]
    b = img[1::2, 1::2, 2]
    y = 0.2125 * r + 0.7154 * g + 0.0721 * b
    return y
