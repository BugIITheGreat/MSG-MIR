from PIL import Image
import os
import re
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
import random


class SenDataset(BaseDataset):
	def __init__(self, opt):
		BaseDataset.__init__(self, opt)
		self.SenDataPathA = os.path.join(opt.dataroot, '1/')
		self.SenDataPathB = os.path.join(opt.dataroot, '2/')
		self.loader = transforms.Compose([transforms.ToTensor()])

	def __getitem__(self, item):
		if item == 0:
			item = item + random.randint(1, self.__len__())
		#loader = transforms.Compose([transforms.ToTensor()])
		self.tokenA = '(.*)' + '_' + str(item) + '_opt_.tif'
		self.tokenB = '(.*)' + '_' + str(item) + '_sar_.tif'

		def meowA(stra):
			is_got = re.search(self.tokenA, stra, re.M | re.I)
			if is_got:
				return is_got.group()

		def meowB(stra):
			is_got = re.search(self.tokenB, stra, re.M | re.I)
			if is_got:
				return is_got.group()

		def kickoff(list):
			for i,str in enumerate(list):
				if str is not None:
					return str

		# nameA = self.SenDataPathA + kickoff(list(map(meowA, os.listdir(self.SenDataPathA))))
		# nameB = self.SenDataPathB + kickoff(list(map(meowB, os.listdir(self.SenDataPathB))))
		#print(nameA, nameB)
		nameA = self.SenDataPathA + str(item) + '.jpg'
		nameB = self.SenDataPathB + str(item) + '.jpg'
		# image_a = Image.open(nameA).convert('L')
		image_a = Image.open(nameA)
		image_a = self.loader(image_a)
		# image_b = Image.open(nameB).convert('L')
		image_b = Image.open(nameB)
		image_b = self.loader(image_b)
		return {'A': image_a, 'B': image_b, 'A_paths': self.SenDataPathA, 'B_paths': self.SenDataPathB}

	def __len__(self):
		return len(os.listdir(self.SenDataPathA))
