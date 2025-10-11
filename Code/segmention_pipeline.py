
import numpy as np
import cv2 
import os 
import matplotlib.pyplot as plt
import regionGrowingCells
from datset_segmentation import MV_segment_dataset

# %% Access example data 

img_path = ""

img = cv2.imread(img_path) 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# %% Automatic segmentation (intermediate steps)

im2, BW2, BW3, BW4, BW6, finalCells = regionGrowingCells(img)

plt.imshow(img)
plt.title("Source image")
plt.axis("off")
plt.show()

plt.imshow(im2)
plt.title("Image corrected")
plt.axis("off")
plt.show()

plt.imshow(BW2)
plt.title("Mask with edge erosion")
plt.axis("off")
plt.show()

plt.imshow(BW3)
plt.title("Mask with joined objects")
plt.axis("off")
plt.show()

plt.imshow(BW4)
plt.title("Mask with closed objects")
plt.axis("off")
plt.show()

plt.imshow(BW6)
plt.title("Mask small objects filtered")
plt.axis("off")
plt.show()

plt.imshow(finalCells)
plt.title("Final mask cleaned objects")
plt.axis("off")
plt.show()


# %% Dataset segmentation  

dataset_path = ""
dataset_segmentation_path = ""

MV_segment_dataset(dataset_path, dataset_segmentation_path)