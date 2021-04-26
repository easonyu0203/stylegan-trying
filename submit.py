import torch

from model import StyledGenerator

import sys
import argparse

from matplotlib import pyplot as plt

#check cuda is availible
can_cuda = torch.cuda.is_available()

#get args
parser = argparse.ArgumentParser(description='make sample and submit file')
parser.add_argument('path', type=str, help='path to G checkpoint')
args = parser.parse_args()
ckpt = torch.load(args.path)

#load generator(G)
G = StyledGenerator(code_size).cuda()
G.load_state_dict(ckpt['g_running'])
if can_cuda: G.cuda()
code_size = 512

# Generate 1000 images and make a grid to save them.
n_output = 1000
with torch.no_grad():
    z_sample = torch.randn(n_output, code_size)
    if can_cuda: z_sample.cuda()
    imgs_sample = G(z_sample, step=4, alpha=-1).data
    torchvision.utils.save_image(
        imgs_sample,
        './result.jpg', 
        nrow=10, 
        normalize=True, 
        range=(-1, 1),
    )

# Show 64 of the images.
grid_img = torchvision.utils.make_grid(imgs_sample[:64].cpu(), nrow=10)
plt.figure(figsize=(20,20))
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()