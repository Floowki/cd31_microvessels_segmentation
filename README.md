# CD31_microvessels_segmentation
Automatic blood vessels segmentation on CD31-stained Whole-Slide Images 

An automatic Python algorithm for the segmentation of microvessels in CD31-immunostained histological tumour sections. \
Adapted from Matlab to Python: 

C.C. Reyes-Aldasoro, L Williams, S Akerman, C Kanthou and G. M. Tozer, "An automatic algorithm for the segmentation and morphological analysis of microvessels in immunostained histological tumour sections.", Journal of Microscopy, Volume 242, Issue 3, pages 262–278, June 2011. \
🔗 Article: https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1365-2818.2010.03464.x \
🔗 GitHub: https://github.com/reyesaldasoro/Microvessel-Segmentation.git

# 🏁 Introduction 

The CD31 marker is primarily used to demonstrate the presence of endothelial cells, to measure vessels density or quantify angiogenesis. CD31 is expressed in the vast majority of all types of vascular neoplasms. On immunostained histological sections, it appears in brown, stained with Diaminobenzidine, with a good contrast against the blue background and cell nuclei, counterstained with Heamtoxylin. 
The algorithms is based on pre-processing steps that provide the seeds for a region-growing algorithm in the 3D Hue, Saturation, Value (HSV) colour model. The objects resulting from this process are further refined through morphological operations and splitted. This algorithm concerns patches extracted from tumour Whole-Slide Images. 

# ✨ Subfunctions 

➜ bwmorph \
<br>| \
   ShadingCorrection ➜ simple_pad_2d \
    | \
   gaussF \
    | \
   expandu \
    | \ 
   imfilter \ 
    | \ 
   BackBlueBrown ➜ colourHist2 ➜ quanti_r \ 
    |                | \
    |               bwmorph_spur \ 
    |                | \ 
    |               bwmorph_majority \ 
    | \ 
  JoinObjects ➜ BranchPoints ➜ padData 
    |                             | \
    |                            bwhitmiss \
    | CloseOpenObjects \
    | \
  bwlabel \
    | \
  SplitObjects ➜ regionGrowing \ 

    
# 🔰 Automatic segmentation 

## Importations 

```python
import numpy as np
import cv2 
import os 
import matplotlib.pyplot as plt
import regionGrowingCells
from datset_segmentation import MV_segment_dataset
```

## Access data 

```python
img_path = ""

img = cv2.imread(img_path) 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

<img src='Images/Intro kidney.jpg' width='100%'> 

## Successive steps 

```python
im2, BW2, BW3, BW4, BW6, finalCells = regionGrowingCells(img)
```

```python
plt.imshow(img)
plt.title("Source image")
plt.axis("off")
plt.show()
```

<img src='Images/Intro kidney.jpg' width='100%'> 

```
plt.imshow(im2)
plt.title("Image corrected")
plt.axis("off")
plt.show()
```

<img src='Images/Intro kidney.jpg' width='100%'> 

```python
plt.imshow(BW2)
plt.title("Mask with edge erosion")
plt.axis("off")
plt.show()
```

<img src='Images/Intro kidney.jpg' width='100%'> 

```python
plt.imshow(BW3)
plt.title("Mask with joined objects")
plt.axis("off")
plt.show()
```

<img src='Images/Intro kidney.jpg' width='100%'> 

```python
plt.imshow(BW4)
plt.title("Mask with closed objects")
plt.axis("off")
plt.show()
```

<img src='Images/Intro kidney.jpg' width='100%'> 

```python
plt.imshow(BW6)
plt.title("Mask small objects filtered")
plt.axis("off")
plt.show()
```

<img src='Images/Intro kidney.jpg' width='100%'> 

```python
plt.imshow(finalCells)
plt.title("Final mask cleaned objects")
plt.axis("off")
plt.show()
```

<img src='Images/Intro kidney.jpg' width='100%'> 

# 🚩 Dataset segmentation 

```python
dataset_path = ""
dataset_segmentation_path = ""

MV_segment_dataset(dataset_path, dataset_segmentation_path)
```


