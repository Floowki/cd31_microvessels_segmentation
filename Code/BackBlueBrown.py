
import numpy as np 
from scipy.signal import convolve2d
from scipy.ndimage import label
from skimage.measure import regionprops
import skimage
from scipy.ndimage import binary_fill_holes
import colourHist2

# Helper function to remove spur pixels.
def bwmorph_spur(binary_img, iterations):
    # Remove spur pixels (endpoints) iteratively.
    img = binary_img.copy().astype(np.uint8)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    for _ in range(iterations):
        neighbors = convolve2d(img, kernel, mode='same', boundary='fill', fillvalue=0)
        endpoints = (img == 1) & (neighbors == 1)
        img[endpoints] = 0
    return img.astype(bool)

# Helper function to perform majority filtering.
def bwmorph_majority(binary_img):
    # For each pixel, if at least 5 of the 9 pixels in the 3x3 neighborhood are 1, set pixel to 1.
    img = binary_img.copy().astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    conv = convolve2d(img, kernel, mode='same', boundary='symm')
    majority = conv >= 5
    return majority


def BackBlueBrown(dataRGB):
    #| Args : 
    #     #|   # dataRGB : normalized immunochemistry image stained with CD31 / Hematoxylin 
        
    #     #| Outputs :
    #     #|   # totMask : color-based segmentation mask (Brown=1, Black=2, Blue=3)
    #     #|   # totMask0 :  smoothed version of the BBB_mask
    #     #|   # totMask1 :  smoothed with Blue pixels assigned to Brown by connectivity 
    #     #|   # dataHue : the Hue component of the image in the HSV space 
    
    # Parse Input
    # No input data is received, error 
    if dataRGB is None:
        print(help(BackBlueBrown))
        totMask = np.array([])
        totMask1 = np.array([])
        totMask0 = np.array([])
        dataHue = np.array([])
        return totMask, totMask0, totMask1, dataHue

    # Pre-processing
    rows, cols, levs = dataRGB.shape  
    dataHSV = skimage.color.rgb2hsv(dataRGB)
    #----- colourHist2 will return the chromaticity histogram and will set H,S,V in different matrices
    hs_im1, chrom3D, dataHue, dataSaturation, dataValue = colourHist2(dataHSV, 32, 32, 32)
    sizeSaturation, sizeHue, sizeValue = chrom3D.shape  
    

    #  Maximum Saturation Profile P_max_S (not used at the moment ...), 99% Profile P_99_S and hue histogram
    im_H_S = np.sum(chrom3D, axis=2)
    im_H_S_cum = np.cumsum(im_H_S, axis=0)
    cummulativeLevel_99 = np.ceil(np.tile(0.99 * im_H_S_cum[-1, :], (sizeSaturation, 1)))

    
    # saturationSpread99: cumulative values less than 99% level.
    saturationSpread99 = im_H_S_cum < cummulativeLevel_99
    
    
    # P_99_S: maximum saturation index for which the cumulative histogram is below the threshold.
    P_99_S = np.max(saturationSpread99 * np.tile(np.arange(1, sizeSaturation + 1).reshape(-1, 1), (1, sizeHue)), axis=0)

    
    # Find hues biases
    hueTendsToBrown = np.sum(np.sum((np.isin(dataHue, np.arange(1, 7))) & (dataSaturation < 8))) / (rows * cols)
                                  
    # kernels for the region growing
    kernelDil1 = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]])
    kernelDil3 = np.ones((3, 3), dtype=np.uint8)
    kernelDil5 = np.ones((5, 5), dtype=np.uint8)
    kernelDil7 = np.ones((7, 7), dtype=np.uint8)

    # Define INITIAL Ranges of hues, will serve as seed to be expanded  it depends on the characteristics (blueish, redish) of the image
    # Brown:        red to ocre
    # Background:   yellow to cyan
    if hueTendsToBrown > 0.55:
        brownRange = np.arange(2, 5)       
        backRange = np.arange(5, 17)         
        blueRange = np.arange(17, 32)        
        purpleRange = np.array([1, 32])
    elif (hueTendsToBrown <= 0.55) and (hueTendsToBrown > 0.4):
        brownRange = np.arange(3, 5)         
        backRange = np.arange(5, 17)
        blueRange = np.arange(17, 33)        
        purpleRange = np.array([1, 2])
    elif (hueTendsToBrown <= 0.4) and (hueTendsToBrown > 0.2):
        brownRange = np.arange(1, 6)         
        backRange = np.arange(6, 16)         
        blueRange = np.arange(16, 31)        
        purpleRange = np.array([31, 32])
    elif (hueTendsToBrown <= 0.2) and (hueTendsToBrown > 0.16):
        brownRange = np.concatenate((np.arange(1, 5), np.array([31, 32])))
        backRange = np.arange(5, 20)         
        blueRange = np.arange(20, 27)        
        purpleRange = np.arange(27, 30)      
    elif (hueTendsToBrown <= 0.16) and (hueTendsToBrown > 0.11):
        brownRange = np.concatenate((np.arange(1, 5), np.array([30, 31, 32]))) 
        backRange = np.arange(5, 19)         
        blueRange = np.arange(19, 24)        
        purpleRange = np.arange(27, 30)      
    elif (hueTendsToBrown <= 0.11) and (hueTendsToBrown > 0.06):
        brownRange = np.concatenate((np.arange(1, 7), np.array([30, 31, 32])))
        backRange = np.arange(7, 20)        
        blueRange = np.arange(20, 26)        
        purpleRange = np.arange(26, 30)      
    elif (hueTendsToBrown <= 0.06) and (hueTendsToBrown > 0.02):
        brownRange = np.concatenate((np.arange(1, 9), np.array([30, 31, 32])))
        backRange = np.arange(9, 18)         
        blueRange = np.arange(18, 27)        
        purpleRange = np.arange(27, 30)      
    elif hueTendsToBrown <= 0.02:
        brownRange = np.concatenate((np.arange(1, 16), np.array([30, 31, 32])))
        backRange = np.arange(16, 19)        
        blueRange = np.arange(19, 25)        
        purpleRange = np.arange(26, 30)
            

    #The huesTendTo* will help define how to set the threshold for saturation
    #define the Saturation Threshold as the average of P_max_S along the hues of the background
    satThresholdBack = np.ceil(np.mean(P_99_S[backRange - 1]))
    satThresholdBrown = 0.6 * np.mean(P_99_S[brownRange - 1]) # 0.5 normally 
    satThresholdBlue = 0.6 * np.mean(P_99_S[blueRange - 1]) # 0.5 normally 
    satThresholdPurple = 0.6 * np.mean(P_99_S[purpleRange - 1]) # 0.5 normally 
    
    
    #----------------------- Saturation  ------------------------------------------
    highSatBrown = dataSaturation >= satThresholdBrown
    highSatBlue = dataSaturation >= satThresholdBlue
    highSatPurple = dataSaturation >= satThresholdPurple
    lowSatBack = dataSaturation <= max(2, satThresholdBack)
    verylowSat = dataSaturation <= max(2, (0.7 * np.min(P_99_S)))
    
    meanDataValue = np.mean(dataValue)

    if meanDataValue < 16:
        darkValue = dataValue <= (0.80 * meanDataValue)
        brightValue = dataValue > (1.05 * meanDataValue)
    elif meanDataValue > 19:
        darkValue = dataValue <= (0.95 * meanDataValue)
        brightValue = dataValue > (0.99 * meanDataValue)
    else:
        darkValue = dataValue <= (0.499 * sizeValue)
        brightValue = dataValue > (0.59 * sizeValue)

        
    medSatBrown = dataSaturation >= (max(2, satThresholdBrown / 2))
    medSatBlue = dataSaturation >= (max(2, satThresholdBlue / 2))
    medSatPurple = dataSaturation >= (max(2, satThresholdPurple / 2))
    
    #----------------Define seeds of regions --------------------------------------
    # Above the satThreshold will define initial seeds for brown and blue
    initBrown = (darkValue) & (highSatBrown) & (np.isin(dataHue, brownRange))
    initBlue = (darkValue) & (highSatBlue) & (np.isin(dataHue, blueRange))
    initPurple = (darkValue) & (highSatPurple) & (np.isin(dataHue, purpleRange))
    # Below the satThreshold and restricted to hues yellow and green will define background
    initBack = skimage.morphology.dilation((brightValue) & (lowSatBack) & (np.isin(dataHue, backRange)), footprint=kernelDil3)
    initBack0 = initBack.copy()
    boundaryRegionBack = skimage.morphology.dilation(initBack, footprint=kernelDil7)
    initBack = initBack | ((boundaryRegionBack & verylowSat) & (~(initBlue | initBrown)))
    initAreas = np.sum(np.stack([initBrown, initBack, initBlue], axis=-1), axis=(0, 1)) / (rows * cols)
    # Brown is smaller.
    initBrown = initBrown | initPurple
    kernelBrown = kernelDil5
    kernelBlue = kernelDil1
    

    totMask = initBrown.astype(np.uint8) + 2 * initBack.astype(np.uint8) + 3 * initBlue.astype(np.uint8)
    
    
    counterGrow = 1
    changeFromPrevious = 11
    numGrowthCycles = 5 # 9 before 
    
    # Region growing two regions blue and brown, restrict to unassigned pixels and saturation levels
    while (counterGrow < numGrowthCycles) and (changeFromPrevious > 50):
        
        if initAreas[0] < 0.2:
            #-----------first extend the brown  ---------------
            boundaryRegionBrown = (convolve2d(initBrown.astype(np.float64), kernelBrown, mode='same') > 0)
            brightBrown = (darkValue) & (medSatBrown)
            darkBrownPurple = (darkValue) & (medSatPurple)
            combinedRegion = (brightBrown | darkBrownPurple) & (~(initBack | initBlue))
            initBrown = initBrown | (combinedRegion & boundaryRegionBrown)
            
            
        if initAreas[2] < 0.2:
            #-----------second extend the blue, restrict it to areas not assigned before ---------------
            boundaryRegionBlue = skimage.morphology.dilation(initBlue, footprint=kernelBlue)
            brightBlue = medSatBlue & (np.isin(dataHue, np.arange(blueRange[0], blueRange[-1] + 1)))
            darkBluePurple = medSatPurple & (np.isin(dataHue, purpleRange))
            combinedRegion = (brightBlue | darkBluePurple) & (~(initBack | initBrown))
            initBlue = initBlue | (combinedRegion & boundaryRegionBlue)
            
            
        if initAreas[1] < 0.2:
            # Convert the convolution result to boolean by comparing with 0
            boundaryRegionBack = (convolve2d(initBack.astype(np.float64), kernelDil1, mode='same') > 0)
            boundaryRegionLowSat = (convolve2d(brightValue.astype(np.float64), kernelDil1, mode='same') > 0) & (convolve2d(lowSatBack.astype(np.float64), kernelDil3, mode='same') > 0)
            combinedRegion = (boundaryRegionBack & boundaryRegionLowSat) & (~(initBlue | initBrown))
            initBack = initBack | combinedRegion
            

        newMask = initBrown.astype(np.uint8) + 2 * initBack.astype(np.uint8) + 3 * initBlue.astype(np.uint8)
        changeFromPrevious = np.sum(totMask != newMask)
        totMask = newMask.copy()
        counterGrow = counterGrow + 1


    totMask0 = totMask.copy()

    #  Dilate only the brown area, only restriction is the hue
    counterGrow = 1
    changeFromPrevious = 11
    numGrowthCycles = 2
    while (counterGrow < numGrowthCycles) and (changeFromPrevious > 10):
        boundaryRegionBrown = skimage.morphology.dilation(initBrown, footprint=kernelDil1)
        brightBrown = np.isin(dataHue, np.unique(np.concatenate((brownRange - 1, brownRange + 1))))
        combinedRegion = brightBrown & (~(initBack | initBlue))
        initBrown = initBrown | (combinedRegion & boundaryRegionBrown)
        newMask = initBrown.astype(np.uint8) + 2 * initBack.astype(np.uint8) + 3 * initBlue.astype(np.uint8)
        changeFromPrevious = np.sum(totMask != newMask)
        totMask = newMask.copy()
        counterGrow = counterGrow + 1
        

    # Assignment of isolated mediumly saturated regions
    initAreas = np.sum(np.stack([initBrown, initBack, initBlue], axis=-1), axis=(0, 1)) / (rows * cols)
    
    
    # Removal of noise (isolated pixels or pixels in pairs)
    initBrownL = skimage.measure.label(initBrown, connectivity=1)  # change here 
    initBackL = skimage.measure.label(initBack, connectivity=1)
    initBlueL = skimage.measure.label(initBlue, connectivity=1)
    
    
    smallBrown = regionprops(np.squeeze(initBrownL))
    smallBack = regionprops(np.squeeze(initBackL))
    smallBlue = regionprops(np.squeeze(initBlueL))
     
    Brown_labels = {prop.label for prop in smallBrown if prop.area >= 3}
    Back_labels = {prop.label for prop in smallBack if prop.area >= 3}  
    Blue_labels = {prop.label for prop in smallBlue if prop.area >= 3}  

    
    initBrown = np.isin(initBrownL, list(Brown_labels))
    initBack = np.isin(initBackL, list(Back_labels))
    initBlue = np.isin(initBlueL, list(Blue_labels))
    
    

    # Region growing all regions blue and brown, restrict to unassigned pixels only
    # add those blue and dark pixels In contact with brown, to brown  
    totMask = initBrown.astype(np.uint8) + 2 * initBack.astype(np.uint8) + 3 * initBlue.astype(np.uint8)
    totMask[totMask == 0] = 2
    totMask1 = totMask.copy()
    darkBlueNuclei = ((totMask == 3) & (dataValue < 17))
    

    # exclude now pixels that do not touch brown cells
    darkBlueNucleiL = label(darkBlueNuclei, connectivity=1)
    dilated_totMask0 = skimage.morphology.dilation((totMask0 == 1), footprint=kernelDil1)
    keepElem = np.nonzero(darkBlueNucleiL * dilated_totMask0)
    darkNucleiNextBrown = np.isin(darkBlueNucleiL, darkBlueNucleiL[keepElem])
    totMask1[darkNucleiNextBrown] = 1

    # Smoothing of the final mask
    initBrown = (totMask1 == 1)
    initBack = (totMask1 == 2)
    initBlue = (totMask1 == 3)

    # the combination of a closing operator and a majority smooth very nicely, BUT it also fills in the
    # holes, which will be needed later on for shape analysis, therefore, keep all the holes larger than 15
    # Pixels in area.
    HolesBrown = binary_fill_holes(initBrown)
    HolesBrownL = label(HolesBrown.astype(np.int32) - initBrown.astype(np.int32), connectivity=1)
    HolesBrownR = regionprops(HolesBrownL)
    HolesBrownK = np.isin(HolesBrownL, [prop.area for prop in HolesBrownR if prop.area > 10])
                             
    # Smooth with an imclose, majority and removal of small objects
    initBrown = skimage.morphology.closing(bwmorph_spur(initBrown, 2), footprint=np.array([[0, 1, 1, 0],
                                                                                        [1, 1, 1, 1],
                                                                                        [0, 1, 1, 0]]))
    initBrown[HolesBrownK] = 0
    initBrown[initBack0] = 0

    
    initBrownL = label(initBrown, connectivity=2)
    smallBrown = regionprops(initBrownL)
    initBrown = np.isin(initBrownL, [prop.label for prop in smallBrown if prop.area >= 25])
    initBluePlus = np.isin(initBrownL, [prop.label for prop in smallBrown if prop.area < 25])
    
    initBlue[initBluePlus] = 1
    initBlue = bwmorph_majority(initBlue)
    initBlue[initBrown] = 0

    initBack[initBrown] = 0
    initBack[initBlue] = 0

    totMask1 = initBrown.astype(np.uint8) + 2 * initBack.astype(np.uint8) + 3 * initBlue.astype(np.uint8)
    totMask1[totMask1 == 0] = 2

    # Final cleaning procedure, remove objects that are only partly brown
    # Label to identify objects uniquely
    totMaskLabeled, numBrownObjects = label(totMask1 == 1, connectivity=2, return_num=True)
    # Obtain area of objects
    statsTotMaskLab = regionprops(totMaskLabeled)

    if numBrownObjects > 0:
        relFactor = np.zeros(numBrownObjects)
        for counterBrObjs in range(1, numBrownObjects + 1):
            qqqq = dataHue[totMaskLabeled == counterBrObjs]
            if qqqq.size > 0:
                relFactor[counterBrObjs - 1] = np.sum(np.isin(qqqq, brownRange)) / qqqq.size
            else:
                relFactor[counterBrObjs - 1] = 0
        
        # discard ALL OBJECTS whose relFactor <0.2
        totMask1[np.isin(totMaskLabeled, np.where(relFactor < 0.2)[0] + 1)] = 3
        # discard ALL OBJECTS whose area <20
        totMask1[np.isin(totMaskLabeled, [prop.area for prop in statsTotMaskLab if prop.area < 20])] = 3
        # discard objects whose relFactor <0.75 AND its area is less than 50 pixels
        mask1 = np.isin(totMaskLabeled, np.intersect1d(np.where(relFactor < 0.5)[0] + 1,
                                                         np.array([prop.label for prop in statsTotMaskLab if prop.area < 50]))
        )
        totMask1[mask1] = 3
        # discard objects whose relFactor <0.7 AND its area is less than 100 pixels
        mask2 = np.isin(totMaskLabeled, np.intersect1d(np.where(relFactor < 0.4)[0] + 1,
                                                         np.array([prop.label for prop in statsTotMaskLab if prop.area < 100]))
        )
        totMask1[mask2] = 3
        # discard objects whose relFactor <0.55 AND its area is less than 200 pixels
        mask3 = np.isin(totMaskLabeled, np.intersect1d(np.where(relFactor < 0.3)[0] + 1,
                                                         np.array([prop.label for prop in statsTotMaskLab if prop.area < 200]))
        )
        totMask1[mask3] = 3

    return totMask, totMask0, totMask1, dataHue