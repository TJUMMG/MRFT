import random
import numpy as np
import skimage.color as sc
import torch




def get_patch_test(img_in, img_tar, scale, patch_size_factor=8, multi_scale=False):
    ih, iw = img_in.shape[:2]
    p = scale if multi_scale else 1

    ix = (iw // patch_size_factor) * patch_size_factor
    iy = (ih // patch_size_factor) * patch_size_factor
    tx, ty = p * ix, p * iy

    img_in = img_in[0:iy, 0:ix, :]
    img_tar = img_tar[0:ty, 0:tx, :]
    return img_in, img_tar

def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]

def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        type = img.dtype
        if type == 'uint16':
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            tensor = torch.from_numpy(np_transpose / 1.0).float()
            tensor.mul_(1 / 65535.0)
        elif type == 'uint8':
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            tensor = torch.from_numpy(np_transpose / 1.0).float()
            tensor.mul_(1 / 255.0)
        else:
            print('Please input correct data！')
        return tensor

    return [_np2Tensor(_l) for _l in l]


def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(_l) for _l in l]


def cut_test_image(image, window_size=(256,256), overlap=0.05):
    C, H, W = image.shape
    patch_h, patch_w = window_size[0], window_size[1]
    index_list, patch_list = [], []
    h_index, w_index = 0, 0
    h_step, w_step = int(patch_h * (1-overlap)), int(patch_w * (1-overlap))
    while True:
        if h_index + patch_h < H:
            start_h, end_h = h_index, h_index + patch_h
        else:
            start_h, end_h = H - patch_h, H
        while True:
            if w_index + patch_w < W:
                start_w, end_w = w_index, w_index + patch_w
            else:
                start_w, end_w = W - patch_w, W
            index_list.append([start_h, end_h, start_w, end_w])
            patch_list.append(image[:, start_h:end_h, start_w:end_w])
            w_index += w_step
            if end_w == W:
                w_index = 0
                break
        h_index += h_step
        if end_h == H:
            h_index = 0
            break
    return patch_list, index_list