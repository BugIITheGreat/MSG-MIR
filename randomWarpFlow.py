import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random as rn
import math
from PIL import Image
from torchvision import transforms
import os


'''
class Warper2d(nn.Module):
    def __init__(self, img_size):
        super(Warper2d, self).__init__()
        self.img_size = img_size
        H, W = img_size, img_size
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,H,W)
        yy = yy.view(1,H,W)
        self.grid = torch.cat((xx,yy),0).float() # [2, H, W]
            
    def forward(self, flow, img):
        grid = self.grid.repeat(flow.shape[0],1,1,1)#[bs, 2, H, W]
        if img.is_cuda:
            grid = grid.cuda()
#        if flow.shape[2:]!=img.shape[2:]:
#            pad = int((img.shape[2] - flow.shape[2]) / 2)
#            flow = F.pad(flow, [pad]*4, 'replicate')#max_disp=6, 32->44
        vgrid = Variable(grid, requires_grad = False) + flow
 
        # scale grid to [-1,1] 
#        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/(W-1)-1.0 #max(W-1,1)
#        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/(H-1)-1.0 #max(H-1,1)
        vgrid = 2.0*vgrid/(self.img_size-1)-1.0 #max(W-1,1)
 
        vgrid = vgrid.permute(0,2,3,1)        
        output = F.grid_sample(img, vgrid)
#        mask = Variable(torch.ones(img.size())).cuda()
#        mask = F.grid_sample(mask, vgrid)
#        
#        mask[mask<0.9999] = 0
#        mask[mask>0] = 1
        
        return output#*mask
'''

if __name__ == "__main__":
	token = './vid1_IR'
	savePath = './vid1_IR_new'
	annoPath = './gt.txt'
	os.mkdir(savePath) if not os.path.isdir(savePath) else None
	listi = os.listdir(token)
	listi.sort()
	loader = transforms.ToTensor()
	unloader = transforms.ToPILImage()	
	with open(annoPath, 'w') as fp:
		for i in range(len(listi)):
			img_path = os.path.join(token, (str(i+1) + '.tiff'))
			print(img_path + 'Start.')
			img_torch = loader(Image.open(img_path))

			default = torch.tensor([0,0,1], dtype=torch.float).unsqueeze(0)

			offsetx = rn.uniform(-0.5, 0.5)
			offsety = rn.uniform(-0.5, 0.5)
			print(str(offsetx) + " Offsetx")
			print(str(offsety) + " Offsety")
			thetaOffset = torch.tensor([
			[1, 0, offsetx],
			[0, 1, offsety]
			], dtype=torch.float)
			thetaOffset = torch.cat((thetaOffset, default), 0)

			dAngle = rn.uniform(-35, 35)
			print(str(dAngle) + " dAngle")
			rAngle = dAngle * math.pi / 180.0
			thetaRotate = torch.tensor([
			[math.cos(rAngle), math.sin(-rAngle), 0],
			[math.sin(rAngle), math.cos(rAngle), 0]
			], dtype=torch.float)
			thetaRotate = torch.cat((thetaRotate, default), 0)
	
			scale = rn.uniform(0.6, 1/0.6)
			print(str(scale) + " scale")
			thetaScale = torch.tensor([
			[scale, 0, 0],
			[0, scale, 0]
			], dtype=torch.float)
			thetaScale = torch.cat((thetaScale, default), 0)

			#H = torch.matmul(thetaScale, torch.matmul(thetaOffset, thetaRotate))
			H = thetaScale @ thetaOffset @ thetaRotate
			#print(H)

			grid = F.affine_grid(H[0:2, :].unsqueeze(0), img_torch.unsqueeze(0).size())
			
			output = F.grid_sample(img_torch.unsqueeze(0), grid)
			newImgTorch = output[0]

			newImgTorch = newImgTorch.cpu().clone()
			newImgTorch = newImgTorch.squeeze(0)
			imageDone = unloader(newImgTorch)

		
			fp.write(str(i+1) + '.tiff' + '\n')
			fp.write(str(H.numpy()) + '\n')

			imageDone.save(os.path.join(savePath, (str(i+1) + '.tiff')))
			print(os.path.join(savePath, (str(i+1) + '.tiff')) + " Done.")

