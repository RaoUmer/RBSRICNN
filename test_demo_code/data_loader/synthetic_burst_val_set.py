import torch
import cv2
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from data_loader.dataset_utils import calculate_affine_matrices

class SyntheticBurstVal(Dataset):
    """ Synthetic burst validation set. The validation burst have been generated using the same synthetic pipeline as
    employed in SyntheticBurst dataset.
    """
    def __init__(self, root):
        self.root = root
        self.burst_list = list(range(300))
        self.burst_size = 14

    def __len__(self):
        return len(self.burst_list)

    def _read_burst_image(self, index, image_id):
        im = cv2.imread('{}/{:04d}/im_raw_{:02d}.png'.format(self.root, index, image_id), cv2.IMREAD_UNCHANGED)
        im_t = torch.from_numpy(im.astype(np.float32)).permute(2, 0, 1).float() / (2**14)
        return im_t

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
    
            b_mask = np.zeros(m_shape)
            b_mask[:, 2, 0::2, 1::2] = 1
            
            mask = torch.stack((g_r_mask, r_mask, b_mask, g_b_mask), dim=1).squeeze(2)
    
        if len(im_shape) == 3:
            return mask.view((4, m_shape[-2], m_shape[-1]))
        else:
            return mask.view((-1, 4, m_shape[-2], m_shape[-1]))
            
    def __getitem__(self, index):
        """ Generates a synthetic burst
                args:
                    index: Index of the burst

                returns:
                    burst: LR RAW burst, a torch tensor of shape
                           [14, 4, 48, 48]
                           The 4 channels correspond to 'R', 'G', 'G', and 'B' values in the RGGB bayer mosaick.
                    seq_name: Name of the burst sequence
                """
        burst_name = '{:04d}'.format(index)
        burst = [self._read_burst_image(index, i) for i in range(self.burst_size)]
        burst = torch.stack(burst, 0)
        
        # compute the mask 
        mask = self.compute_mask(burst.shape)
        #print('mask:', mask.shape, mask.min(), mask.max())
        
        # calculate wrap matrix (A) using ECC
        warp_matrix = calculate_affine_matrices(burst)
        #print('warp_matrix:', warp_matrix.shape, warp_matrix.min(), warp_matrix.max())
        if warp_matrix.ndim == 3:
            assert np.array_equal(warp_matrix[0], np.array([[1, 0, 0], [0, 1, 0]]))

        return {'LR_burst': burst, 'mask': mask, 'warp_matrix': warp_matrix, 'LR_burst_name': burst_name}

def imshow(img):
    npimg = img.numpy()
    #print('npimg:', npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
if __name__ == "__main__":
    root = 'LR_burst/syn_burst_val/'
    
    train_dataset = SyntheticBurstVal(root=root)
    print('valset samples:', len(train_dataset))
    
    trainset_loader = DataLoader(train_dataset,
                                 shuffle=False,
                                 batch_size=1,
                                 pin_memory=True,
                                 num_workers=0
                                 )
    print('#train_loader:', len(trainset_loader))
          
    for epoch in range(1):
        print('epoch:', epoch)
        for i, data in enumerate(trainset_loader):
            burst, mask, warp_matrix, burst_name  = data['LR_burst'], data['mask'], data['warp_matrix'], data['LR_burst_name']
            
            print('Input:', burst.shape, burst.min(), burst.max())
            print('warp matrix:', warp_matrix.shape, warp_matrix.min(), warp_matrix.max())
            print('mask:', mask.shape, mask.min(), mask.max())
            print('Input_name:', burst_name)
            
#            burst_rgb = burst[:, 0, [0, 1, 3]]
#            print('burst_rgb:', burst_rgb.shape, burst_rgb.min(), burst_rgb.max())
#            burst_rgb = burst_rgb.view(-1, *burst_rgb.shape[-3:])
#            print('burst_rgb:', burst_rgb.shape, burst_rgb.min(), burst_rgb.max())
        
            # show images
            #plt.figure()
            #imshow(torchvision.utils.make_grid(burst[0]))
            #plt.figure()
            #imshow(torchvision.utils.make_grid(burst_rgb))
            break
