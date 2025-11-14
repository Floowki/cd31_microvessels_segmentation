# Code for generating figures illustrating the pipeline 

# %% 
import cv2
import matplotlib.pyplot as plt
import regionGrowingCells as RGC

# %%
img_path = "C:/Users/augus/Desktop/PORTFOLIO GITHUB/Figures CD31/P1_T98304_50688_100.png"
save_path = "C:/Users/augus/Desktop/PORTFOLIO GITHUB/Figures CD31/"

img = cv2.imread(img_path) 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_AREA)

plt.imshow(img)
plt.title("Image source")
plt.axis("off")
plt.show()

# %%
im1, im2, BW2, BW3, BW4, BW6, finalCells = RGC.regionGrowingCells(img)

plt.imshow(im1)
plt.title("Image corrected")
plt.axis("off")
plt.show()
plt.imsave(save_path + "im1" + ".png", im1, dpi=300)

plt.imshow(im2)
plt.title("Image equalized")
plt.axis("off")
plt.show()
plt.imsave(save_path + "im2" + ".png", im2, dpi=300)

plt.imshow(BW2, cmap="gray")
plt.title("Mask with edge erosion")
plt.axis("off")
plt.show()
plt.imsave(save_path + "BW2" + ".png", BW2, cmap="gray", dpi=300)

plt.imshow(BW3, cmap="gray")
plt.title("Mask with joined objects")
plt.axis("off")
plt.show()
plt.imsave(save_path + "BW3" + ".png", BW3, cmap="gray", dpi=300)

plt.imshow(BW4, cmap="gray")
plt.title("Mask with closed objects")
plt.axis("off")
plt.show()
plt.imsave(save_path + "BW4" + ".png", BW4, cmap="gray", dpi=300)

plt.imshow(BW6, cmap="gray")
plt.title("Mask small objects filtered")
plt.axis("off")
plt.show()
plt.imsave(save_path + "BW6" + ".png", BW6, cmap="gray", dpi=300)

plt.imshow(finalCells, cmap="gray")
plt.title("Final mask cleaned objects")
plt.axis("off")
plt.show()
plt.imsave(save_path + "finalCells" + ".png", finalCells, cmap="gray", dpi=300)



