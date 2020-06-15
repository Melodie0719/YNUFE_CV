import torch
import torchvision.transforms as transforms
from PIL import Image
from model2 import LeNet
import cv2

transform = transforms.ToTensor()


classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')

net = LeNet()
net.load_state_dict(torch.load('Lenet2.pth'))

im = cv2.imread('9.jpg')
im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
im = transform(im)  # [C, H, W]
im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

with torch.no_grad():
    outputs = net(im)
    predict = torch.max(outputs, dim=1)[1].data.numpy()
print(classes[int(predict)])
