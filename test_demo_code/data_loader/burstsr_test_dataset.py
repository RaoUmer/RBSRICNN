import os
import numpy as np
import torch
import torch.nn.functional as F
import random
from data_loader.burstsr_val_dataset import SamsungRAWImage, flatten_raw_image, pack_raw_image
from torch.utils.data import DataLoader
import torchvision
from data_loader.dataset_utils import calculate_affine_matrices
import matplotlib.pyplot as plt


class BurstSRTestDataset(torch.utils.data.Dataset):
    """ Real-world burst super-resolution dataset. """
    def __init__(self, root, burst_size=8, crop_sz=80, center_crop=False, random_flip=False, split='test'):
        """
        args:
            root : path of the root directory
            burst_size : Burst size. Maximum allowed burst size is 14.
            crop_sz: Size of the extracted crop. Maximum allowed crop size is 80
            center_crop: Whether to extract a random crop, or a centered crop.
            random_flip: Whether to apply random horizontal and vertical flip
            split: Can be 'train' or 'val'
        """
        assert burst_size <= 14, 'burst_sz must be less than or equal to 14'
        assert crop_sz <= 80, 'crop_sz must be less than or equal to 80'
        assert split in ['test']

        root = root + '/' + split
        super().__init__()

        self.burst_size = burst_size
        self.crop_sz = crop_sz
        self.split = split
        self.center_crop = center_crop
        self.random_flip = random_flip

        self.root = root

        self.substract_black_level = True
        self.white_balance = False

        self.burst_list = self._get_burst_list()

    def _get_burst_list(self):
        burst_list = sorted(os.listdir('{}'.format(self.root)))

        return burst_list

    def get_burst_info(self, burst_id):
        burst_info = {'burst_size': 14, 'burst_name': self.burst_list[burst_id]}
        return burst_info

    def _get_raw_image(self, burst_id, im_id):
        raw_image = SamsungRAWImage.load('{}/{}/samsung_{:02d}'.format(self.root, self.burst_list[burst_id], im_id))
        return raw_image

    def get_burst(self, burst_id, im_ids, info=None):
        frames = [self._get_raw_image(burst_id, i) for i in im_ids]

        if info is None:
            info = self.get_burst_info(burst_id)

        return frames, info

    def _sample_images(self):
        burst_size = 14

        ids = random.sample(range(1, burst_size), k=self.burst_size - 1)
        ids = [0, ] + ids
        return ids
    
    def compute_mask(self, im_shape, mode='rggb'):
        """
        The function is to compute a mask accordying to CFA.
        """
        m_shape = (im_shape[0], 1, im_shape[2], im_shape[3])
    
        if mode == 'rggb':
            # compute mask
            r_mask = torch.zeros(m_shape)
            r_mask[:, :, 0::2, 0::2] = 1
    
            g_r_mask = torch.zeros(m_shape)
            g_r_mask[:, :, 0::2, 1::2] = 1
            
            g_b_mask = torch.zeros(m_shape)
            g_b_mask[:, :, 1::2, 0::2] = 1
            
            b_mask = torch.zeros(m_shape)
            b_mask[:, :, 1::2, 1::2] = 1
            
            mask = torch.stack((r_mask, g_r_mask, g_b_mask, b_mask), dim=1).squeeze(2) 
            
        elif mode == 'grbg':
            # compute mask
            r_mask = torch.zeros(m_shape)
            r_mask[:, 0, 0::2, 1::2] = 1
    
            g_r_mask = torch.zeros(m_shape)
            g_r_mask[:, 1, 0::2, 0::2] = 1
            
            g_b_mask = torch.zeros(m_shape)
            g_b_mask[:, 1, 1::2, 1::2] = 1
    
            b_mask = torch.zeros(m_shape)
            b_mask[:, 2, 0::2, 1::2] = 1
            
            mask = torch.stack((g_r_mask, r_mask, b_mask, g_b_mask), dim=1).squeeze(2)
    
        if len(im_shape) == 3:
            return mask.view((4, m_shape[-2], m_shape[-1]))
        else:
            return mask.view((-1, 4, m_shape[-2], m_shape[-1]))
        
    def __len__(self):
        return len(self.burst_list)

    def __getitem__(self, index):
        # Sample the images in the burst, in case a burst_size < 14 is used.
        im_ids = self._sample_images()

        # Read the burst images along with HR ground truth
        frames, meta_info = self.get_burst(index, im_ids)

        # Extract crop if needed
        if frames[0].shape()[-1] != self.crop_sz:
            if getattr(self, 'center_crop', False):
                r1 = (frames[0].shape()[-2] - self.crop_sz) // 2
                c1 = (frames[0].shape()[-1] - self.crop_sz) // 2
            else:
                r1 = random.randint(0, frames[0].shape()[-2] - self.crop_sz)
                c1 = random.randint(0, frames[0].shape()[-1] - self.crop_sz)
            r2 = r1 + self.crop_sz
            c2 = c1 + self.crop_sz

            frames = [im.get_crop(r1, r2, c1, c2) for im in frames]

        # Load the RAW image data
        burst_image_data = [im.get_image_data(normalize=True, substract_black_level=self.substract_black_level,
                                              white_balance=self.white_balance) for im in frames]

        if self.random_flip:
            burst_image_data = [flatten_raw_image(im) for im in burst_image_data]

            pad = [0, 0, 0, 0]
            if random.random() > 0.5:
                burst_image_data = [im.flip([1, ])[:, 1:-1].contiguous() for im in burst_image_data]
                pad[1] = 1

            if random.random() > 0.5:
                burst_image_data = [im.flip([0, ])[1:-1, :].contiguous() for im in burst_image_data]
                pad[3] = 1

            burst_image_data = [pack_raw_image(im) for im in burst_image_data]
            burst_image_data = [F.pad(im.unsqueeze(0), pad, mode='replicate').squeeze(0) for im in burst_image_data]

        burst_image_meta_info = frames[0].get_all_meta_data()

        burst_image_meta_info['black_level_subtracted'] = self.substract_black_level
        burst_image_meta_info['while_balance_applied'] = self.white_balance
        burst_image_meta_info['norm_factor'] = frames[0].norm_factor

        burst = torch.stack(burst_image_data, dim=0)

        burst_exposure = frames[0].get_exposure_time()

        burst_f_number = frames[0].get_f_number()

        burst_iso = frames[0].get_iso()

        burst_image_meta_info['exposure'] = burst_exposure
        burst_image_meta_info['f_number'] = burst_f_number
        burst_image_meta_info['iso'] = burst_iso

        burst = burst.float()
        
        # compute the mask 
        mask = self.compute_mask(burst.shape)
        
        # calculate wrap matrix (A) using ECC
        warp_matrix = calculate_affine_matrices(burst)
        #print('warp_matrix:', warp_matrix.shape, warp_matrix.min(), warp_matrix.max())
        if warp_matrix.ndim == 3:
            assert np.array_equal(warp_matrix[0], np.array([[1, 0, 0], [0, 1, 0]]))

        meta_info_burst = burst_image_meta_info
        meta_info_burst['burst_name'] = meta_info['burst_name']
        for k, v in meta_info_burst.items():
            if isinstance(v, (list, tuple)):
                meta_info_burst[k] = torch.tensor(v)

        return {'LR_burst': burst, 'mask': mask, 'warp_matrix': warp_matrix, 'meta_info_burst': meta_info_burst}

def imshow(img):
    npimg = img.numpy()
    #print('npimg:', npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
if __name__ == "__main__":
    root = 'track2_test_set'
    burst_size=8 
    crop_sz=80
    center_crop=False
    random_flip=False 
    split='test'
    batch_size = 1
    
    train_dataset = BurstSRTestDataset(root=root, 
                                   burst_size=burst_size, 
                                   crop_sz=crop_sz, 
                                   center_crop=center_crop, 
                                   random_flip=random_flip, 
                                   split=split)
    print('trainset samples:', len(train_dataset))
      
    trainset_loader = DataLoader(train_dataset,
                                 shuffle=True,
                                 batch_size=batch_size,
                                 pin_memory=True,
                                 num_workers=1
                                 )
    print('#train_loader:', len(trainset_loader))
          
    for epoch in range(1):
        print('epoch:', epoch)
        for i, data in enumerate(trainset_loader):
            y, mask, warp_matrix, meta_info_y  = data['LR_burst'], data['mask'], data['warp_matrix'], data['meta_info_burst']

            print('burst input:', y.shape, y.min(), y.max())
            print('warp matrix:', warp_matrix.shape, warp_matrix.min(), warp_matrix.max())
            print('mask:', mask.shape, mask.min(), mask.max())
            #print('A:', A.shape, A.min(), A.max())
            #print('meta_info burst:', meta_info_y)
            #print('img_path:', img_path)
        
            # show images
            plt.figure()
            imshow(torchvision.utils.make_grid(y[0][0]))
            #imshow(x[0])
            #plt.figure()
            #plot_batch_burst(y)
            break
