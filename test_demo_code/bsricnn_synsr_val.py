import os
import cv2
import numpy as np
import torch
from models.ResDNet import ResDNet, BasicBlock
from models.MMNet import MMNet
from modules.problems import Burst_SR_Demosaick_Denoise
from collections import OrderedDict
from torch.utils.data import DataLoader
from data_loader.synthetic_burst_val_set import SyntheticBurstVal

# select the particular GPU device to run the code
os.environ['CUDA_VISIBLE_DEVICES'] =  "0" # GPU id: 0,1,2, etc.

# test settings
model_path = 'trained_nets/model_x4.pth'  # path of trained RBSRICNN model
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
#device = torch.device('cpu')
test_img_folder = 'track1_val_set/' # testset LR images path
out_dir = 'sr_results_track1_val_set/' # save images path
os.makedirs(out_dir, exist_ok=True)
upscale_factor = 4 # upscaling factor
num_iter_steps = 10 # number of iterative steps

# loading Network
model = ResDNet(BasicBlock, layer_size=5, weightnorm=True)
model = MMNet(model, max_iter=10, sigma_max=2, sigma_min=1)
states = torch.load(model_path)
model.load_state_dict(states['model_state_dict'])
model.eval()
model = model.to(device)

# loading test data
val_dataset = SyntheticBurstVal(root=test_img_folder)
print('#valset samples:', len(val_dataset))
  
valset_loader = DataLoader(val_dataset,
                             shuffle=False,
                             batch_size=1,
                             pin_memory=True,
                             num_workers=1
                             )
print('#valset_loader:', len(valset_loader))
          
print('Model path {:s}. \nStart Testing...'.format(model_path))
test_results = OrderedDict()
test_results['time'] = []
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
for i, data in enumerate(valset_loader):
    burst, mask, warp_matrix, burst_name  = data['LR_burst'], data['mask'], data['warp_matrix'], data['LR_burst_name']
    burst_name = burst_name[0]
    print('Img:', i+1, burst_name)
    
    LR_burst = burst.to(device)
    LR_burst = (LR_burst*255).float()
    mask = mask.float()
    warp_matrix = warp_matrix.float()
    
    p = Burst_SR_Demosaick_Denoise(y=LR_burst, M=mask, A=warp_matrix, 
                                   scale=upscale_factor, 
                                   estimate_noise=True, 
                                   mode='bilinear')
    
    if device.type == 'cuda':
        p.cuda_()  # run on GPU
    else:
        p.cpu_()    # run on CPU
    
    # testing
    start.record()
    with torch.no_grad():
        output_SR = model.forward_all_iter(p, init=True, noise_estimation=True, max_iter=num_iter_steps)
        output_SR = output_SR.clamp_(0, 255)
        output_SR = output_SR/255.     
    end.record()
    torch.cuda.synchronize()
    end_time = start.elapsed_time(end)
    
    # Normalize to 0  2^14 range and convert to numpy array
    output_sr = (output_SR.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).cpu().numpy().astype(np.uint16)
    
    test_results['time'].append(end_time)
    print('{:->4d}--> {:>10s}, time: {:.4f} miliseconds.'.format(i+1, burst_name, end_time))
    
    # Save images as png
    cv2.imwrite('{}/{}.png'.format(out_dir, burst_name), output_sr)
    
    del burst, LR_burst, mask, warp_matrix, burst_name
    del  output_SR, output_sr
    torch.cuda.empty_cache()

avg_time = sum(test_results['time']) / len(test_results['time']) / 1000.0
print('Avg. Time:{:.4f} seconds'.format(avg_time))
print('End Testing!!!')