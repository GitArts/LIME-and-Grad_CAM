import torch
import torch.nn as nn
from torch.utils import data
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torchvision import datasets

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

"""
This is modified code from https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82 article
"""

# use the ImageNet transformation
transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# define a 1 image dataset
dataset = datasets.ImageFolder(root='./data/', transform=transform)

# define the dataloader to load that single image
dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # get the pretrained VGG19 network
        # |=== model ===|
        self.weights = ResNet50_Weights.IMAGENET1K_V1
        self.model = resnet50(weights=self.weights)

        # disect the network to access its last convolutional layer
        self.features_conv = nn.ModuleList(self.model.children())[:-2]
        self.features_conv = nn.Sequential(*self.features_conv)
        # get the max pool of the features stem
        self.max_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        # get the classifier of the vgg19
        self.classifier = self.model.fc

        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)

# initialize the ResNet model
resnet = Net()

# set the evaluation mode
resnet.eval()

# get the image from the dataloader
for i, data in enumerate(dataloader):
  img, _ = data
  if i==1: break

# Viz image
if 1:
  plt.imshow(img.reshape(224,224,3))
  plt.show()

# get the most likely prediction of the model
pred = resnet.forward(img) # 282

# get the gradient of the output with respect to the parameters of the model
pred[:, 282].backward()

# pull the gradients out of the model
gradients = resnet.get_activations_gradient()

# pool the gradients across the channels
# |=== mean beigu gradienti ===|
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

# get the activations of the last convolutional layer
# |=== pēdējie conv gradienti ===|
activations = resnet.get_activations(img).detach()
# weight the channels by corresponding gradients
for i in range(512):
    activations[:, i, :, :] *= pooled_gradients[i]
breakpoint()

# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()

# relu on top of the heatmap
# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
heatmap = np.maximum(heatmap, 0)

# normalize the heatmap
heatmap /= torch.max(heatmap)

# draw the heatmap
plt.matshow(heatmap.squeeze())
plt.show()

img = np.array(Image.open('./data/Dog/dog.jpg'))
heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)

plt.imshow(img)
plt.imshow(heatmap,alpha=.5)
plt.subplots()
plt.imshow(img)
plt.show()
