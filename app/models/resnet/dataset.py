import os
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as FT
import torchvision.transforms as transforms
import numpy as np

# label 0 for MARY
# label 1 for other class

class ArtDLDataset(Dataset):
  def __init__(self, data_dir = None, transform = None, labels_path = None, set_type = 'train'):

    # Setting the inital_dir to take images from
    self.data_dir = data_dir

    # Setting up the transforms
    self.transform = transform

    res = []
    for path in os.listdir('./data/images/'):
    # check if current path is a file
      if os.path.isfile(os.path.join('./data/images/', path)):
          res.append(path)
    self.img_names = res 

  def __getitem__(self, idx):

    # Getting the filename based on idx
    filename = self.img_names[idx]

    # Reading using PIL
    image = Image.open(self.data_dir + "/" + filename)

    # Applying transforms if any
    if(self.transform!=None):
      image = self.transform(image)
    
    
    
    
    image_label = 0   # image_label is irrelevant in this case

    return (image, image_label, filename)

  def __len__(self):
    return len(self.img_names)


# Util class to apply padding to all the images
class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return FT.pad(image, padding, 0, 'constant')

transform=transforms.Compose([
    SquarePad(),

    transforms.Resize((224,224)),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomHorizontalFlip(p=0.5)
])


val_transform = transforms.Compose([

	  SquarePad(),
		transforms.Resize(224),
	  transforms.CenterCrop(224),
		transforms.ToTensor()
		
])
