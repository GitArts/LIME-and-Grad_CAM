import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import gray2rgb
from sklearn.utils import check_random_state
from tqdm import tqdm
from torchvision import transforms
import torch
import cv2
import torch.nn.functional as F

from lime import lime_image

from torchvision.models import resnet50, ResNet50_Weights

from wrappers.scikit_image import SegmentationAlgorithm

"""
This script replicates part of the LIME tool.
Input is one image.
"""

'''
1. It's possible to run LIME as it is.
2. It's possible to analyse LIME part by part.
  2.1. SegImgs() function creates images with randomized black regions.
  2.2. NNcheck() function uses SegImgs() output to create Hot picels
							for original image.
  Note. ( 2.2.) is not how LIME creates Hot pixels. ( 2.2.) is just an idea.
  To understand how LIME craetes please refer to lime_base.py --> LimeBase() class -->
							explain_instance_with_data() method.
if __name__=='__main__': <-- line at the end is start of the code.
'''

# |=== Images and masks are generated using one input image ===|
def GetImgs(image, fudged_image, segments, 
		classifier_fn, numImgs, batch_size=10):
  '''
  image - original image.
  fudged_image - Copy of the image.
  segments - Created regions for image. Shape of the 'segments' is image size.
  classifier_fn - This is a function to activate neural network and test
			all generated images. Original function name is batch_predict().
  numImgs - define how many images to create.
  batch_size - Batch size for neural network (NN).
  classifier_fn, batch_size, fudged_image, batch_size - is needed for NN activation.
								Not availible in this code.
  '''
  # |=== How many regions for image ===|
  n_features = np.unique(segments).shape[0]
  # |=== Data structure info ===|
  data = random_state.randint(0, 2, numImgs * n_features).reshape((numImgs, n_features))
  # |=== Lists ===|
  labels = []
  data[0, :] = 1
  imgs = []
  masks = []
  # |=== TODO: Do better coding here ===|
  for row in tqdm(data):
    temp = image.copy()
    zeros = np.where(row == 0)[0]
    mask = np.zeros(segments.shape).astype(bool)
    for z in zeros:
      mask[segments == z] = True
    temp[mask] = 0
    masks.append(mask)
    imgs.append(temp)
  return imgs, masks

def batch_predict(images):
  model.eval()
  batch = torch.stack(tuple(preprocess_transform(img) for img in images), dim=0)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model.to(device)
  batch = batch.to(device)
  logits = model(batch)

  probs = F.softmax(logits, dim=1)
  return probs.detach().cpu().numpy()


def get_preprocess_transform():
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
  transf = transforms.Compose([
      transforms.ToTensor(),
      normalize
  ])
  return transf


def SegImgs(im):
  # |=== Make img RGB if is gray ===|
  NpIm = np.array(im)
  if len(NpIm.shape) == 2: NpIm = gray2rgb(NpIm)
  # |=== Random seed for algorithm ===|
  random_seed = random_state.randint(0, high=1000)
  # |=== Define segmentation function ===|
  # NOTE: wrappers.scikit_image SegmentationAlgorithm --> segm algorithms have non-readable files - coded.
  segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=random_seed)
  # |=== Apply transform for masks with shape (224,224) ==|
  imForMask = cv2.resize( NpIm.astype('uint8'), (224,224) )
  # |=== Apply segmantaion funkction for PIL image ==|
  segments = segmentation_fn(imForMask)
  segments = cv2.resize(segments.astype('uint8'), NpIm[:,:,0].T.shape)
  # |=== numpy img copy ===|
  imgCpy = NpIm.copy()
  # |=== Covered regions based on segments ===|
  imgs,masks = GetImgs(NpIm, imgCpy, segments, None, 1000)
  return imgs,masks, segments

# |======= Analysing lime =======|
def LIME(img):
  explainer = lime_image.LimeImageExplainer()
  explanation = explainer.explain_instance(np.array(pill_transf(img)),
                                         batch_predict, # classification function
                                         top_labels=5,
                                         hide_color=0,
                                         num_samples=1000) # number of images that will be sent to classification function

  from skimage.segmentation import mark_boundaries

  temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
  plt.imshow(temp)
  plt.imshow(mask, alpha=.3)
  plt.show()

  img_boundry1 = mark_boundaries(temp/255.0, mask)
  plt.imshow(img_boundry1)
  plt.show()


def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])

    return transf

# |=== Personal Function for NN analysation. Not so good ===|
def NNcheck():
  # |=== Preprocess images ===|
  ImSh = imgs[0].shape
  HotScore = np.zeros((imgs[0].shape[:2]))

  for im, mask in zip(imgs, tqdm(masks) ):
    # |=== complicated but nessessory way to rehsape im to (3, w, h) ===|
    im = np.vstack((im[:,:,0], im[:,:,1], im[:,:,2])).reshape(3, ImSh[0], ImSh[1])
    imB = torch.tensor(im)
    # |=== Weights are trasformed based on image shape ===|
    preprocess = weights.transforms()
    batch = preprocess(imB).unsqueeze(0)
    # |=== Test the model ===|
    model.eval()
    pred = model(batch).squeeze(0).softmax(0)
    class_id = pred.argmax().item()
    # |== Dog Samoyed class_id = 258 ==|
    category_name = weights.meta["categories"][258]
    # |=== Vis images ===|
    if 0:
      plt.imshow(im[0],cmap='gray')
      plt.show()
    # |=== Hot pixels ===|
    #print (category_name)
    score = round(100*pred[258].item(),2)
    #print ("score:", score,"%")
    mask = ~mask
    tmpScore = HotScore * mask
    HotScore[mask] = ((tmpScore + score)/2)[mask]

  HotScore[HotScore < ( HotScore.max()+HotScore.min() )/2] = 0

  plt.imshow(img)
  plt.imshow(HotScore,cmap='gray', alpha=.5)
  plt.subplots()
  plt.imshow(DEBUGseg)
  plt.show()

  breakpoint()

if __name__=='__main__':
  # |=== Load image ===|
  img = Image.open("data/Dog/dog.jpg")
  # |=== model ===|
  weights = ResNet50_Weights.IMAGENET1K_V1
  model = resnet50(weights=weights)
  pill_transf = get_pil_transform()
  preprocess_transform = get_preprocess_transform()
  img = pill_transf(img)
  # |=== Orig Lime ===|
  # |=== Run LIME as it is ===|
  if 0: LIME(img)
  # |=== .. Or analyse LIME part by part ===|
  # |=== Define random state ===|
  random_state = check_random_state(None)
  # |=== Segment images ===|
  imgs, masks, DEBUGseg = SegImgs(img)
  if 1:
    # TODO: fix masks. Cause of cv2.resize(segments) in func SegImgs()
	# between segments are lines
    plt.imshow(imgs[1])
    plt.subplots()
    plt.imshow(imgs[2])
    plt.subplots()
    plt.imshow(imgs[3])
    plt.show()
  # |=== Vis all the images ===|
  if 0:
    for i, im in enumerate(imgs):
      if i!=0: plt.subplots()
      plt.imshow(im)
      if i > 10: break
    plt.show()
  # |=== NNcheck() is a try to create Hot pixels easy way ===|
  # Better results can be achieved, running this script several times
  NNcheck()
  # LIME uses more difficult way to create Hot pixels. Ridge regression in use.

