import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from modules.wmad_estimator import Wmad_estimator

def downsampling(x, size=None, scale_factor=None, mode='bilinear'):
    # define size if user has specified scale_factor
    if size is None: size = (int(scale_factor*x.size(2)), int(scale_factor*x.size(3)))
    # create coordinates
    h = torch.arange(0,size[0]) / (size[0]-1) * 2 - 1
    w = torch.arange(0,size[1]) / (size[1]-1) * 2 - 1
    # create grid
    grid = torch.zeros(size[0],size[1],2)
    grid[:,:,0] = w.unsqueeze(0).repeat(size[0],1)
    grid[:,:,1] = h.unsqueeze(0).repeat(size[1],1).transpose(0,1)
    # expand to match batch size
    grid = grid.unsqueeze(0).repeat(x.size(0),1,1,1)
    if x.is_cuda: grid = grid.cuda()
    # do sampling
    return F.grid_sample(x, grid, mode=mode)

def bicubic_interp_nd(input_, size, scale_factor=None, endpoint=False):
    """
    Args :
    input_ : Input tensor. Its shape should be
    [batch_size, height, width, channel].
    In this implementation, the shape should be fixed for speed.
    new_size : The output size [new_height, new_width]
    ref :
    http://blog.demofox.org/2015/08/15/resizing-images-with-bicubic-interpolation/
    """

    shape = input_.shape
    batch_size, channel, height, width = shape

    def _hermite(A, B, C, D, t):
        a = A * (-0.5) + B * 1.5 + C * (-1.5) + D * 0.5
        b = A + B * (-2.5) + C * 2.0 + D * (-0.5)
        c = A * (-0.5) + C * 0.5
        d = B
        return a*(t**3) + b*(t**2) + c*t + d

    def tile(a, dim, n_tile):
        " Code from https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/2"
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(a, dim, order_index)


    def meshgrid_4d(n_i, c_i, x_i, y_i):
        r""" Return a 5d meshgrid created using the combinations of the input
        Only works for 4d tensors
        """
        # tested and it is the same
        nn = n_i[:,None,None,None].expand((n_i.shape[0], c_i.shape[0], x_i.shape[0], y_i.shape[0]))
        cc = c_i[None,:,None,None].expand((n_i.shape[0], c_i.shape[0], x_i.shape[0], y_i.shape[0]))
        xx = x_i.view(-1,1).expand((n_i.shape[0], c_i.shape[0], x_i.shape[0], y_i.shape[0]))
        yy = y_i.expand((n_i.shape[0], c_i.shape[0], x_i.shape[0], y_i.shape[0]))
        return torch.cat([nn[..., None], cc[..., None], xx[..., None], yy[..., None]], dim=4)

    def get_frac_array_4d(x_d, y_d, n, c):
        # tested and it is the same
        x = x_d.shape[0]
        y = y_d.shape[0]
        x_t = x_d[None,None,:,None]
        y_t = y_d[None,None,None,:]
        # tile tensor in each dimension
        x_t = tile(x_t, 0, n)
        x_t = tile(x_t, 1, c)
        x_t = tile(x_t, 3, y)
        # x_t.transpose_(2,3)
        y_t = tile(y_t, 0, n)
        y_t = tile(y_t, 1, c)
        y_t = tile(y_t, 2, x)
        # y_t.transpose_(2,3)
        return x_t, y_t

    def roll(tensor, shift_x, shift_y):
        # calculate the shifts of the input tensor
        shift_x = shift_x.clamp(0,height-1)
        shift_y = shift_y.clamp(0,width-1)
        p_matrix = tensor[:,:,shift_x]
        p_matrix = p_matrix[..., -tensor.shape[3]:] [...,shift_y]
        return p_matrix

    if size is None: size = (int(scale_factor*height), int(scale_factor*width))
    new_height = size[0]
    new_width  = size[1]
    n_i = torch.arange(batch_size, dtype=torch.int64)
    c_i = torch.arange(channel, dtype=torch.int64)

    if endpoint:
        x_f = torch.linspace(0., height, new_height)
    else:
        x_f = torch.linspace(0., height, new_height+1)[:-1]
    x_i = torch.floor(x_f).type(torch.int64)
    x_d = x_f - torch.floor(x_f)

    if endpoint:
        y_f = torch.linspace(0., width, new_width)
    else:
        y_f = torch.linspace(0., width, new_width+1)[:-1]

    y_i = torch.floor(y_f).type(torch.int64)
    y_d = y_f - torch.floor(y_f)

    grid = meshgrid_4d(n_i, c_i, x_i, y_i)
    x_t, y_t = get_frac_array_4d(x_d, y_d, batch_size, channel)

    if input_.is_cuda:
        x_t = x_t.cuda()
        y_t = y_t.cuda()
    # calculate f-1, f0, f+1, f+2 for y axis
    p_00 = roll(input_, x_i-1, y_i-1)
    p_10 = roll(input_, x_i-1, y_i+0)
    p_20 = roll(input_, x_i-1, y_i+1)
    p_30 = roll(input_, x_i-1, y_i+2)

    p_01 = roll(input_, x_i, y_i-1)
    p_11 = roll(input_, x_i, y_i+0)
    p_21 = roll(input_, x_i, y_i+1)
    p_31 = roll(input_, x_i, y_i+2)

    p_02 = roll(input_, x_i+1, y_i-1)
    p_12 = roll(input_, x_i+1, y_i+0)
    p_22 = roll(input_, x_i+1, y_i+1)
    p_32 = roll(input_, x_i+1, y_i+2)

    p_03 = roll(input_, x_i+2, y_i-1)
    p_13 = roll(input_, x_i+2, y_i+0)
    p_23 = roll(input_, x_i+2, y_i+1)
    p_33 = roll(input_, x_i+2, y_i+2)

    col0 = _hermite(p_00, p_10, p_20, p_30, x_t)
    col1 = _hermite(p_01, p_11, p_21, p_31, x_t)
    col2 = _hermite(p_02, p_12, p_22, p_32, x_t)
    col3 = _hermite(p_03, p_13, p_23, p_33, x_t)
    value = _hermite(col0, col1, col2, col3, y_t)

    return value

class Problem(nn.Module):
    r""" An abstract Problem Class """

    def __init__(self, task_name):
        super().__init__()
        self.task_name = task_name
        self.L = torch.FloatTensor(1).fill_(1)

    def task(self):
        return self.task_name

    def energy_grad(self, x):
        pass

    def initialize(self):
        pass

    def cuda_(self):
        pass

class Burst_SR_Demosaick_Denoise(Problem):
    def __init__(self, y, M, A, scale, k=None, estimate_noise=False, mode='bilinear', task_name='burst_sr_demosaick_denoise'):
        r""" Joint Super-Resolution, Demosaick, and Denoise Problem class
                            y_{i} = MHS_{i}(x) + n_{i}
        y is the observed sequence of frames with shape [b, B, C, H, W].
        S is the alignment/warping matrix (estimated by	the	ECC	method) of each frame to reference. The respective shape is [b, B, 2, 3] and it is
        assumed the affine transformation is the same for each channel.
        M is the masking operator that defines the Bayer Pattern with shape [b, B, C, H, W].
        H is the down-sampling operator by the scale factor s.

        Abbreviations for Shape:
        b: batch size
        B: Burst size - Number of Frames in a burst
        C: number of Channels of frames
        H: Height of frames
        W : Width of frames
        """
        Problem.__init__(self, task_name)
        assert(mode in ['bilinear','bicubic','kernel', 'normal']), "Mode can be either 'bilinear','bicubic' or 'kernel'"
        self.y = y
        self.scale = scale
        self.A = A  # affine transformation matrix
        self.M = M
        self.grid = self.generate_grid(A, self.y.shape)  # generate the sampling grid based on the warping matrices
        self.mask = self.generate_pixel_mask(self.grid)
        self.A_inv = self.inverse_affine(self.A)  # calculate the inverse of the warping/affine matrices
        self.grid_inv = self.generate_grid(self.A_inv, self.y.shape)  # generate the inverse sampling grid
        if estimate_noise:
            self.estimate_noise()
        self.mode = mode
        self.k = k
        self.grad_var = None
    
    @staticmethod
    def generate_pixel_mask(grid):
        r""" Calculate a binary mask that indicates boundaries
             During warping certain portions of the warped image will be padded to maintain the original shape.
             These padded areas should be taken under consideration during the calculations of energy term gradient,
             because artificial boundaries ('zero', 'reflect') impede with the reconstruction process.
            Return:
                 mask of shape [b, B, 1, H, W]
        """
        # Calculate boundaries across H axis
        # Zero value indicates pixel of synthetic boundary area
        mask_low_x = 1 - (grid[..., 0] < -1).short()
        mask_high_x = 1 - (grid[..., 0] > 1).short()
        # Calculate boundaries across W axis
        mask_low_y = 1 - (grid[..., 1] < -1).short()
        mask_high_y = 1 - (grid[..., 1] > 1).short()

        # Generate masks for the whole image by a
        mask_x = mask_low_x * mask_high_x
        mask_y = mask_low_y * mask_high_y
        mask = mask_x * mask_y
        mask = mask[:, :, None].float()  # expand dimension for broadcasting
        return mask

    @staticmethod
    def inverse_affine(A):
        r""" Calculate the inverse of an affine / warping matrix. """
        A_inv = []
        for i in range(A.shape[1]):
            r_b = A[:, i, :, :2]
            batch_inv = []
            for j, r in enumerate(r_b):
                r_inv = r.inverse()
                t_inv = -1 * r_inv.mv(A[j, i, :, 2])
                batch_inv.append(torch.cat([r_inv, t_inv[:, None]], dim=1))
            batch_inv = torch.stack(batch_inv)
            A_inv.append(batch_inv[:, None])
        return torch.cat(A_inv, dim=1)

    @staticmethod
    def generate_grid(warp_matrix, shape):
        r""" Generate grid using warping matrices.
             Note: OpenCV warping matrices are defined using an integer intexed grid, while Pytorch uses a grid defined
             by [-1, 1]. This function performs the necessary fix.

             Input:
             warp_matrix with shape [b, B, 2, 3]
             Return:
                 grid of shape [b, B, H, W]
         """

        batch, B, C, H, W = shape
        H_lin = torch.arange(H).float()
        W_lin = torch.arange(W).float()
        H_ = H_lin.expand(H, -1).t()
        W_ = W_lin.expand(W, -1)
        Z = torch.ones(H_.shape).float()
        grid = torch.stack([W_, H_, Z])
        grid = torch.stack([grid] * batch)
        aug_warp_matrix = torch.cat([warp_matrix, torch.Tensor([[[[0, 0, 1]]] * B] * batch)], dim=2)
        affine_grids = []
        for i in range(B):
            affine_grid = torch.bmm(aug_warp_matrix[:, i], grid.reshape(batch, 3, -1)).reshape(grid.shape)
            affine_grid[:, 0] = 2 * affine_grid[:, 0] / H - 1
            affine_grid[:, 1] = 2 * affine_grid[:, 1] / W - 1
            affine_grid = affine_grid[:, :2]
            affine_grids.append(affine_grid.permute(0, 2, 3, 1)[:, None])
        return torch.cat(affine_grids, dim=1)

    def warp(self, y, grid, compress=False):
        r""" Warp frames of y according to grid.
             If y is a mosaicked image then we compress the spatial dimension in order to account for missing
             information.
        """
        x = []
        B = y.shape[1] if len(y.shape) > 4 else 0

        if B > 1:
            for i in range(B):
                if compress:
                    y_ = y[:, i]
                    y_ = self.compress(y_)
                    x.append(F.grid_sample(y_, grid[:, i], padding_mode='zeros')[:, None])
                else:
                    x.append(F.grid_sample(y[:, i], grid[:, i], padding_mode='zeros')[:, None])
            return torch.cat(x, dim=1)
        else:
            if y.ndimension() == 5 and  y.shape[1] == 1:  # case of number of frames is 1
                return y
            else:
                return F.grid_sample(y, grid[:, 0], padding_mode='zeros')

    def get_warped_burst(self, y, grid, compress=True):
        if compress:
            batch, B, C, H, W = y.shape
            res = self.warp(y, grid, True)
            res = res.reshape(batch * B, res.shape[2], res.shape[3], res.shape[4])
            res = self.decompress(res)
            res = res.reshape(batch, B, C, res.shape[2], res.shape[3])
            return res
        else:
            return self.warp(y, grid)

    @staticmethod
    def compress(x):
        r""" Compress a mosaicked image. The sequence according to channels is: R, G1, G2, B

             Input:
                x with shape [b, 3, H, W]
             Return:
                compressed_image of shape [b, 4, H/2, W/2]
         """
        size = x.shape
        if x.is_cuda:
            compressed_image = torch.cuda.FloatTensor(size[0],  # number of batches
                                                      4,  # number of bayer channels
                                                      int(size[2] / 2),  # H
                                                      int(size[3] / 2)).fill_(0)
        else:
            compressed_image = torch.FloatTensor(size[0],  # number of batches
                                                 4,  # number of bayer channels
                                                 int(size[2] / 2),  # H
                                                 int(size[3] / 2)).fill_(0)

        compressed_image[:, 0, :, :] = x[:, 0, ::2, ::2]  # R
        compressed_image[:, 1, :, :] = x[:, 1, ::2, 1::2]  # G
        compressed_image[:, 3, :, :] = x[:, 1, 1::2, ::2]  # G
        compressed_image[:, 2, :, :] = x[:, 2, 1::2, 1::2]  # B
        return compressed_image

    @staticmethod
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
    
    def energy_grad(self, x):
        r""" Returns the gradient of the data fidelity term: 1/2||y-Ax||^2, where A = MHS
        grad: A^T(-y+Ax) ---> (MHS)^T(-y+(MHS)x) ---> S^T H^T M^T(-y+MHSx) ---> S^T H^T M^T M H S x - S^T H^T M^T y
        X is given as input
        """
        batch, B, C, H, W = self.y.shape
        if self.mode == 'bilinear':
            if self.grad_var is None:
                # applying adjoint wrapping operator
                self.grad_var = self.warp(self.y, self.grid, compress=False)
                # applying masking operator
                self.grad_var = self.mask * self.grad_var
                self.grad_var = self.grad_var.reshape(batch * B, self.grad_var.shape[2], self.grad_var.shape[3],
                                                  self.grad_var.shape[4])
                self.grad_var = self.decompress(self.grad_var)
                # applying upsampling operator 
                self.grad_var = downsampling(self.grad_var, size=None, scale_factor=self.scale, mode='bilinear')
                self.grad_var = self.grad_var.reshape(batch, B, self.grad_var.shape[1], self.grad_var.shape[2], self.grad_var.shape[3])
            
            # calculate forward operation on the restored image
            # downsampling
            x = downsampling(x, size=None, scale_factor=1/self.scale, mode='bilinear')
            x = self.compress(x)
            # warpping
            x = self.warp(torch.cat([x[:, None]] * self.y.shape[1], dim=1), self.grid_inv)
            # masking
            x = self.M * x
            # calculate adjoint operation on the mosaicked restored image
            x = self.warp(x, self.grid, True)
            x = self.mask * x 
            x = x.reshape(batch * B, x.shape[2], x.shape[3], x.shape[4])
            x = self.decompress(x)
            x = downsampling(x, size=None, scale_factor=self.scale, mode='bilinear')
            x = x.reshape(batch, B, x.shape[1], x.shape[2], x.shape[3])
            
        elif self.mode == 'bicubic':
            if self.grad_var is None:
                # applying adjoint wrapping operator
                self.grad_var = self.warp(self.y, self.grid, compress=False)
                # applying masking operator
                self.grad_var = self.mask * self.grad_var
                self.grad_var = self.grad_var.reshape(batch * B, self.grad_var.shape[2], self.grad_var.shape[3],
                                                  self.grad_var.shape[4])
                self.grad_var = self.decompress(self.grad_var)
                # applying upsampling operator 
                self.grad_var = bicubic_interp_nd(self.grad_var, size=None, scale_factor=self.scale)
                self.grad_var = self.grad_var.reshape(batch, B, self.grad_var.shape[1], self.grad_var.shape[2], self.grad_var.shape[3])
            
            # calculate forward operation on the restored image
            # downsampling
            x = bicubic_interp_nd(x, size=None, scale_factor=1/self.scale)
            x = self.compress(x)
            # warpping
            x = self.warp(torch.cat([x[:, None]] * self.y.shape[1], dim=1), self.grid_inv)
            # masking
            x = self.M * x 
            # calculate adjoint operation on the mosaicked restored image
            x = self.warp(x, self.grid, True)
            x = self.mask * x
            x = x.reshape(batch * B, x.shape[2], x.shape[3], x.shape[4])
            x = self.decompress(x)
            x = bicubic_interp_nd(x, size=None, scale_factor=self.scale)
            x = x.reshape(batch, B, x.shape[1], x.shape[2], x.shape[3])

        return (x - self.grad_var).sum(dim=1) / self.y.shape[1]
    
    def initialize(self):
        r""" Initialize with bilinear interpolation of the reference frame"""

        y = self.y[:, 0]  # reference frame is assumed to be always the first
        y = self.decompress(y)
        res = downsampling(y, size=None, scale_factor=self.scale, mode='bilinear')

        return res

    def estimate_noise(self):
        r""" Estimate noise using the reference frame
             Noise is assumed to have the same characteristics across all frames, therefore a single estimation
             is enough.
        """
        y = self.y[:, 0]
        if y.max() > 1:
            y = y / 255
        y = y.sum(dim=1).detach()
        L = Wmad_estimator()(y[:, None])
        self.L = L
        if self.y.max() > 1:
            self.L *= 255  # scale back to uint8 representation
            
    def cuda_(self):
        self.y = self.y.cuda()
        self.L = self.L.cuda()
        if  self.k is not None:
            self.k = self.k.cuda()
        self.mask = self.mask.cuda()
        self.grid_inv = self.grid_inv.cuda()
        self.grid = self.grid.cuda()
        self.M = self.M.cuda()
    
    def cpu_(self):
        self.y = self.y.cpu()
        self.L = self.L.cpu()
        if  self.k is not None:
            self.k = self.k.cpu()
        self.mask = self.mask.cpu()
        self.grid_inv = self.grid_inv.cpu()
        self.grid = self.grid.cpu()
        self.M = self.M.cpu()

if __name__ == "__main__":
    M =  torch.rand(5, 8, 4, 48, 48)
    A =  torch.rand(5,8,2,3)
    x = torch.rand(5, 3, 384, 384)
    y = torch.rand(5, 8, 4, 48, 48)
    #k = torch.rand(3,1,3,3)
    p = Burst_SR_Demosaick_Denoise(y, M, A, scale=4, k=None, estimate_noise=True, mode='bilinear')
    #eng_grad = p.energy_grad(x)
    x_init = p.initialize()
    print('x_init:', x_init.shape)
    eng_grad =x_init - p.energy_grad(x_init)
    print('eng_grad:', eng_grad.shape)
    print('noise_estimation:',p.L)