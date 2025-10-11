
import numpy as np
from scipy.ndimage import binary_dilation, binary_opening, binary_erosion, zoom
from scipy.ndimage import label
from skimage.measure import regionprops
import ndimage
import cv2
import sk_morphology
from skimage import measure 
from skimage import color

import JoinObjects
import SplitObjects
import expandu
import gaussF
import ShadingCorrection
import BackBlueBrown
import CloseOpenObjects


def regionGrowingCells(dataIn) : 
    #| Args : 
    #|   # dataIn = image to be analysed -- image as an array 
    
    #| Outputs :
    #|   # finalCells : final labeled image of segmented cells,
    #|   # statsObjects3 : statistical properties of the objects in finalCells
    #|   # finalCellsIm : visualization of cell boundaries highlighted on the original image
    #|   # statsObjects2 : more detailed statistical properties of objects at an earlier processing stage
    #|   # BW2 : binary image after edge erosion
    #|   # BW3 : binary image after joining objects that are close to each other
    #|   # BW4 : binary image after morphological closing and opening operations
    #|   # BW6 : labeled image of objects with areas greater than 25 pixels
    #|   # im2 : pre-processed input image with equalized channel levels : segmentation basis 
        
    
    # Intermediate functions #
    def imfilter(image, kernel, mode='same', boundary='fill', fillvalue=0):
        # Handle boundary conditions
        if boundary == 'fill':
            mode_scipy = 'constant'
            cval = fillvalue
        elif boundary == 'circular':
            mode_scipy = 'wrap'
            cval = 0
        elif boundary == 'replicate':
            mode_scipy = 'nearest'
            cval = 0
        elif boundary == 'symmetric':
            mode_scipy = 'reflect'
            cval = 0
        else:
            raise ValueError(f"Unknown boundary condition: {boundary}")
        
        filtered = ndimage.convolve(image, kernel, mode=mode_scipy, cval=cval)
        
        if mode == 'full':
            pad_width = [(k//2, k//2) for k in kernel.shape]
            filtered = np.pad(filtered, pad_width, mode=mode_scipy, constant_values=cval)
        
        return filtered

    def bwmorph(image, operation, iterations=1):
        # Perform morphological operations on a binary image
        out = image.astype(bool).copy()
        if operation == 'clean':
            out = binary_opening(out, structure=np.ones((3,3)))
        elif operation == 'bridge':
            # 'bridge' can be approximated with a dilation to connect close pixels
            out = binary_dilation(out, structure=np.ones((3,3)))
        elif operation == 'spur':
            for _ in range(iterations):
                out = binary_erosion(out, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        return out.astype(np.uint8)

    def bwlabel(binary):
        return measure.label(binary, connectivity=1)
    
    # Function main core #  
    
    rows, cols, levs = dataIn.shape

    RCL = rows * cols * levs
    if RCL < 1e6:
        subSampL = 1
    elif RCL < 3e6:
        subSampL = 2
    elif RCL < 12e6:
        subSampL = 4
    else:
        subSampL = 8

    # call shadingCorrection on a subsampled version of dataIn
    dataIn_sub = dataIn[0:rows:subSampL, 0:cols:subSampL, :]
    dataOutS, errSurfaceS, errSurfaceMin, errSurfaceMax = ShadingCorrection(dataIn_sub, numScales=None)
    dataOutS = dataOutS.astype("uint8")
    
    if subSampL == 1:
        errSurfaceS2 = errSurfaceS
    else:
        # create a 3x3 Gaussian filter 
        filtG = gaussF(3, 3)
        
        errSurfaceS2 = np.zeros((rows, cols, levs))
        
        for counterL in range(levs):
            subsampled = errSurfaceS[:,:,counterL]
            
            # calculate needed expansions to reach at least target size
            current_rows, current_cols = subsampled.shape
            expansions_needed = 0
            while (current_rows * 2 <= rows) and (current_cols * 2 <= cols):
                current_rows *= 2
                current_cols *= 2
                expansions_needed += 1
            
            # apply expansions
            expanded = subsampled.copy()
            if expansions_needed > 0:
                expanded = expandu(subsampled, expansions_needed)
            
            # Now handle residual scaling with interpolation
            if expanded.shape[0] < rows or expanded.shape[1] < cols:

                scale_factor = min(rows/expanded.shape[0], cols/expanded.shape[1])
                expanded = zoom(expanded, (scale_factor, scale_factor), order=3)
                
                if expanded.shape[0] > rows or expanded.shape[1] > cols:
                    expanded = expanded[:rows, :cols]
                elif expanded.shape[0] < rows or expanded.shape[1] < cols:
                    # pad with edge values
                    pad_rows = rows - expanded.shape[0]
                    pad_cols = cols - expanded.shape[1]
                    expanded = np.pad(expanded, ((0, pad_rows), (0, pad_cols)), mode='edge')
            
            # apply Gaussian smoothing
            smoothed = imfilter(expanded, filtG, mode='replicate')
            
            errSurfaceS2[:smoothed.shape[0], :smoothed.shape[1], counterL] = smoothed
    
    errSurfaceS2 = errSurfaceS2[:rows, :cols, :]
    
    # subtract shading correction error surface
    dataOut = dataIn.astype(float) - errSurfaceS2
    avChannels = np.mean(np.mean(dataOut, axis=0), axis=0)
    maxAvChannel = np.max(avChannels)
    for counterL in range(levs):
        dataOut[:,:,counterL] = maxAvChannel * (dataOut[:,:,counterL] / avChannels[counterL])
    dataOut[dataOut > 255] = 255
    dataOut[dataOut < 0] = 0

    im1 = dataOut.copy()
    del dataOut

    # equalize channel levels for whitish background
    meanLevChannels = np.mean(np.mean(im1, axis=0), axis=0)
    maxMeanLev = np.max(meanLevChannels)
    if maxMeanLev < 140:
        maxMeanLev = 150
    if maxMeanLev > 160:
        maxMeanLev = 150
    im2 = np.empty_like(im1)
    im2[:,:,0] = maxMeanLev * im1[:,:,0] / meanLevChannels[0]
    im2[:,:,1] = maxMeanLev * im1[:,:,1] / meanLevChannels[1]
    im2[:,:,2] = maxMeanLev * im1[:,:,2] / meanLevChannels[2]
    im2[im2 > 255] = 255
    
    # get seeds of background, blue cells and brown cells
    totMask, totMask0, totMask1, dataHue = BackBlueBrown(im2)

    # edge erosion
    BW2 = bwmorph(bwmorph((totMask1==1).astype(np.uint8), 'bridge'), 'spur', 5)
    
    # join objects
    BW3 = JoinObjects(BW2, (totMask1==2))
    
    # closing of objects
    BW4 = CloseOpenObjects(BW3)
    
    # clean and label objects
    BW5 = bwlabel(BW4)
    statsObjects1 = regionprops(BW5)
    
    # filter objects by area  (to tweak)
    min_area = 30
    areas = np.array([prop.area for prop in statsObjects1]) if statsObjects1 else np.array([])
    indices = np.where(areas > min_area)[0] + 1 if areas.size else np.array([]) 
    BW6 = bwlabel(np.isin(BW5, indices))
    
    # remove small holes 
    BW6 = BW6 > 0                                               
    BW6 = sk_morphology.remove_small_holes(BW6, area_threshold=150)           
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))           
    BW6 = cv2.morphologyEx(BW6.astype(np.uint8), cv2.MORPH_CLOSE, kernel) 
    BW6 = bwlabel(BW6).astype(np.uint8)
    statsObjects2 = regionprops(BW6) 

    # splitting of objects
    if not statsObjects2:
        finalCells = np.zeros_like(BW6)
        statsObjects3 = []
    else:
        NosplittedCells, splittedCells, NoHoleCells = SplitObjects(BW6, statsObjects2, rows, cols)
        combined = (NosplittedCells + splittedCells + NoHoleCells) > 0
        finalCells = bwlabel(combined)
        statsObjects3 = regionprops(finalCells)
        

    # final cleaning: remove objects that are only partly brown
    numBrownObjects = np.max(finalCells) if finalCells.size > 0 else 0
    dataHSV = color.rgb2hsv(im2.astype(np.float64)/255)
    dataSaturation = dataHSV[:,:,1]
    hueTendsToBrown = np.sum((np.isin(dataHue, np.arange(1,7))) & (dataSaturation < 8)) / (rows * cols)
    
    if hueTendsToBrown > 0.11:
        brownRange = np.array([1,2,3,4,30,31,32])
    elif (hueTendsToBrown <= 0.11) and (hueTendsToBrown > 0.069):
        brownRange = np.array([1,2,3,4,5,30,31,32])
    elif hueTendsToBrown <= 0.069:
        brownRange = np.array([1,2,3,4,5,6,30,31,32])
    
    if numBrownObjects > 0:
        relFactor = np.zeros((2, numBrownObjects))
        for counterBrObjs in range(1, numBrownObjects+1):
            qqqq = dataHue[finalCells==counterBrObjs]
            relFactor[0, counterBrObjs-1] = np.sum(np.isin(qqqq, brownRange)) / qqqq.size if qqqq.size>0 else 0
            if statsObjects3 and (counterBrObjs-1 < len(statsObjects3)):
                relFactor[1, counterBrObjs-1] = statsObjects3[counterBrObjs-1].area
        
        # discard objects with relFactor < 0.25
        remove_indices = np.where(relFactor[0,:] < 0.25)[0] + 1
        if remove_indices.size > 0:
            finalCells[np.isin(finalCells, remove_indices)] = 0
        
        # discard objects with area < 30
        remove_indices_area = [i+1 for i, prop in enumerate(statsObjects3) if prop.area < 30]
        if remove_indices_area:
            finalCells[np.isin(finalCells, remove_indices_area)] = 0
        
        
        finalCells = ndimage.binary_fill_holes(finalCells).astype(np.uint8)
        finalCells = bwlabel(finalCells > 0).astype(np.uint8)
    statsObjects3 = regionprops(finalCells)

    
    BW2 = (finalCells > 0) & BW2                                              
    BW2 = sk_morphology.remove_small_holes(BW2.astype(bool), area_threshold=150)           
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))          
    BW2 = cv2.morphologyEx(BW2.astype(np.uint8), cv2.MORPH_CLOSE, kernel)   

    return im2, BW2, BW3, BW4, BW6, finalCells