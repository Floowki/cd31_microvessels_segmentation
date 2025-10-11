
import numpy as np
from scipy.ndimage import binary_opening, zoom
from scipy import ndimage
from scipy.ndimage import label
from skimage.measure import regionprops
from scipy.ndimage import convolve, binary_dilation
from skimage.morphology import closing, binary_erosion, square 
from skimage import measure, morphology
import skimage.morphology as sk_morphology
import cv2
from skimage import color
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import ShadingCorrection
import gaussF
import expandu 
import JoinObjects
import CloseOpenObjects
import SplitObjects
import BackBlueBrown

def imfilter(image, kernel, mode='same', boundary='fill', fillvalue=0):
    #| Args : 
    #|   # image : the image to be filtered  
    #|   # kernel : the kernel used for filtering 
    #|   # mode : mode
    #|   # boundary : mode for dealing with the boundaries 
    #|   # fillValue : if mode='constant', value to fill past edges 
        
    #| Outputs :
    #|   # filtered : resulting filtered image 
    
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
    
    # Apply convolution
    filtered = ndimage.convolve(image, kernel, mode=mode_scipy, cval=cval)
    
    # Handle 'full' mode if needed
    if mode == 'full':
        pad_width = [(k//2, k//2) for k in kernel.shape]
        filtered = np.pad(filtered, pad_width, mode=mode_scipy, constant_values=cval)
    
    return filtered

def bwmorph(image, operation, iterations=1):
    #| Args : 
    #|   # image : image to perform an operation on 
    #|   # operation : the type of operation that is to be performed 
    #|   # iterations : the number of iterations to perform 
    
    #| Outputs :
    #|   # out : resulting image 
    
    # Perform morphological operations on a binary image.
    out = image.astype(bool).copy()
    if operation == 'clean':
        out = binary_opening(out, structure=np.ones((3,3)))
    elif operation == 'bridge':
        # 'bridge' can be approximated with a dilation to connect close pixels.
        out = binary_dilation(out, structure=np.ones((3,3)))
    elif operation == 'spur':
        for _ in range(iterations):
            out = binary_erosion(out, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
            
    out = out.astype(np.uint8)
            
    return out

def bwlabel(binary):
    # Label connected components.
    return measure.label(binary, connectivity=1)

def single_ShadingCoorection(dataIn) : 
    #| Args : 
    #|   # dataIn : raw image from stained WSI
    
    #| Outputs :
    #|   # im2 : image corrected for shading and reflects 
    
    rows, cols, levs = dataIn.shape
    
    # Remove Shading
    RCL = rows * cols * levs
    if RCL < 1e6:
        subSampL = 1
    elif RCL < 3e6:
        subSampL = 2
    elif RCL < 12e6:
        subSampL = 4
    else:
        subSampL = 8
    
    # Call shadingCorrection on a subsampled version of dataIn
    dataIn_sub = dataIn[0:rows:subSampL, 0:cols:subSampL, :]
    dataOutS, errSurfaceS, errSurfaceMin, errSurfaceMax = ShadingCorrection(dataIn_sub, numScales=None)
    
    dataOutS = dataOutS.astype("uint8")
    
    if subSampL == 1:
        errSurfaceS2 = errSurfaceS
    else:
        # Create Gaussian filter (3x3 as in original MATLAB code)
        filtG = gaussF(3, 3)
        
        # Initialize output at full size
        errSurfaceS2 = np.zeros((rows, cols, levs))
        
        for counterL in range(levs):
            # Get subsampled data for this channel
            subsampled = errSurfaceS[:,:,counterL]
            
            # Calculate needed expansions to reach at least target size
            current_rows, current_cols = subsampled.shape
            expansions_needed = 0
            while (current_rows * 2 <= rows) and (current_cols * 2 <= cols):
                current_rows *= 2
                current_cols *= 2
                expansions_needed += 1
            
            # Apply expansions
            expanded = subsampled.copy()
            if expansions_needed > 0:
                expanded = expandu(subsampled, expansions_needed)
            
            # Now handle residual scaling with interpolation
            if expanded.shape[0] < rows or expanded.shape[1] < cols:
                # Use MATLAB-like imresize behavior
                scale_factor = min(rows/expanded.shape[0], cols/expanded.shape[1])
                expanded = zoom(expanded, (scale_factor, scale_factor), order=3)
                
                # Handle any rounding differences
                if expanded.shape[0] > rows or expanded.shape[1] > cols:
                    expanded = expanded[:rows, :cols]
                elif expanded.shape[0] < rows or expanded.shape[1] < cols:
                    # Pad with edge values
                    pad_rows = rows - expanded.shape[0]
                    pad_cols = cols - expanded.shape[1]
                    expanded = np.pad(expanded, ((0, pad_rows), (0, pad_cols)), mode='edge')
            
            # Apply Gaussian smoothing (MATLAB-style convolution)
            smoothed = imfilter(expanded, filtG, mode='replicate')
            
            # Store result
            errSurfaceS2[:smoothed.shape[0], :smoothed.shape[1], counterL] = smoothed
    
    # Ensure errSurfaceS2 matches dataIn size
    errSurfaceS2 = errSurfaceS2[:rows, :cols, :]
    
    # Subtract shading correction error surface
    dataOut = dataIn.astype(float) - errSurfaceS2
    avChannels = np.mean(np.mean(dataOut, axis=0), axis=0)
    maxAvChannel = np.max(avChannels)
    for counterL in range(levs):
        dataOut[:,:,counterL] = maxAvChannel * (dataOut[:,:,counterL] / avChannels[counterL])
    dataOut[dataOut > 255] = 255
    dataOut[dataOut < 0] = 0
    
    im1 = dataOut.copy()
    del dataOut
    
    # Equalize channel levels for whitish background
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
    
    return im2


def seeds_to_splittedObjects(im2) :
    #| Args : 
    #|   # im2 : binary mask identifying objects 
    
    #| Outputs :
    #|   # final_Cells : binary mask with splitted vascular objects
    
    rows, cols, levs = im2.shape
    
    # Get seeds of background, blue cells and brown cells
    totMask, totMask0, totMask1, dataHue = BackBlueBrown(im2)

    # Edge erosion
    BW2 = bwmorph(bwmorph((totMask1==1).astype(np.uint8), 'bridge'), 'spur', 5)
    
    # Join objects
    BW3 = JoinObjects(BW2, (totMask1==2))
    
    # Closing of objects
    BW4 = CloseOpenObjects(BW3) 
    
    # Clean and label objects
    BW5 = bwlabel(BW4)
    statsObjects1 = regionprops(BW5)
    
    # Filter objects by area  (tweak the min area here )
    areas = np.array([prop.area for prop in statsObjects1]) if statsObjects1 else np.array([])
    indices = np.where(areas > 1200)[0] + 1 if areas.size else np.array([]) ### 1000 arbitrary 
    BW6 = bwlabel(np.isin(BW5, indices))
    #
    BW6 = BW6 > 0                                               
    BW6 = sk_morphology.remove_small_holes(BW6, area_threshold=150)           
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))           
    BW6 = cv2.morphologyEx(BW6.astype(np.uint8), cv2.MORPH_CLOSE, kernel) 
    
    BW6 = bwlabel(BW6).astype(np.uint8)
    statsObjects2 = regionprops(BW6) 
    
    # Splitting of objects
    if not statsObjects2:
        finalCells = np.zeros_like(BW6)
        statsObjects3 = []
    else:
        NosplittedCells, splittedCells, NoHoleCells = SplitObjects(BW6, statsObjects2, rows, cols)
        combined = (NosplittedCells + splittedCells + NoHoleCells) > 0
        finalCells = bwlabel(combined)
        statsObjects3 = regionprops(finalCells)
        

    # Final cleaning - remove objects that are only partly brown
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
        
        # Discard objects with relFactor < 0.25
        remove_indices = np.where(relFactor[0,:] < 0.25)[0] + 1
        if remove_indices.size > 0:
            finalCells[np.isin(finalCells, remove_indices)] = 0
        
        # Discard objects with area < 30
        remove_indices_area = [i+1 for i, prop in enumerate(statsObjects3) if prop.area < 500]
        if remove_indices_area:
            finalCells[np.isin(finalCells, remove_indices_area)] = 0
        
        
        finalCells = ndimage.binary_fill_holes(finalCells).astype(np.uint8) # binary mask 
        #finalCells = bwlabel(finalCells > 0).astype(np.uint8)
    #statsObjects3 = regionprops(finalCells)
    
    return finalCells


def parallel_vessels_segmentation(image) : 
    #| Args : 
    #|   # image : image to be segmented (16 quadrants automatically segmented and stitched back together) 
    
    #| Outputs :
    #|   # BW2 : segmentation mask
    #|   # finalCells : segmentation mask with filled holes 
     
    im2 = single_ShadingCoorection(image)
    
    # Split image, segment each part in parallel, then recombine
    height, width = image.shape[:2]
    hq = height // 4 
    overlap = 12
    
    
    quadrants = [
        (im2[0:hq + overlap, 0:hq + overlap, :]), 
        (im2[0:hq + overlap, hq - overlap:2*hq + overlap, :]),
        (im2[0:hq + overlap, 2*hq - overlap:3*hq + overlap, :]),
        (im2[0:hq + overlap, 3*hq - overlap:height, :]),  
        
        (im2[hq - overlap:2*hq + overlap, 0:hq + overlap, :]),  
        (im2[hq - overlap:2*hq + overlap, hq - overlap:2*hq + overlap, :]),
        (im2[hq - overlap:2*hq + overlap, 2*hq - overlap:3*hq + overlap, :]),
        (im2[hq - overlap:2*hq + overlap, 3*hq - overlap:height, :]),
        
        (im2[2*hq - overlap:3*hq + overlap, 0:hq + overlap, :]),
        (im2[2*hq - overlap:3*hq + overlap, hq - overlap:2*hq + overlap, :]),  
        (im2[2*hq - overlap:3*hq + overlap, 2*hq - overlap:3*hq + overlap, :]),  
        (im2[2*hq - overlap:3*hq + overlap, 3*hq - overlap:height, :]),  
        
        (im2[3*hq - overlap:height, 0:hq + overlap, :]),
        (im2[3*hq - overlap:height, hq - overlap:2*hq + overlap, :]),
        (im2[3*hq - overlap:height, 2*hq - overlap:3*hq + overlap, :]), 
        (im2[3*hq - overlap:height, 3*hq - overlap:height, :]) 
    ]
    
    positions =  [
        (0,hq + overlap, 0,hq + overlap),  
        (0,hq + overlap, hq - overlap,2*hq + overlap),
        (0,hq + overlap, 2*hq - overlap,3*hq + overlap),
        (0,hq + overlap, 3*hq - overlap,height),
        
        (hq - overlap,2*hq + overlap, 0,hq + overlap), 
        (hq - overlap,2*hq + overlap, hq - overlap,2*hq + overlap),
        (hq - overlap,2*hq + overlap, 2*hq - overlap,3*hq + overlap),
        (hq - overlap,2*hq + overlap, 3*hq - overlap,height),
        
        (2*hq - overlap,3*hq + overlap, 0,hq + overlap),
        (2*hq - overlap,3*hq + overlap, hq - overlap,2*hq + overlap),
        (2*hq - overlap,3*hq + overlap, 2*hq - overlap,3*hq + overlap),
        (2*hq - overlap,3*hq + overlap, 3*hq - overlap,height),
        
        (3*hq - overlap,height, 0,hq + overlap),
        (3*hq - overlap,height, hq - overlap,2*hq + overlap),
        (3*hq - overlap,height, 2*hq - overlap,3*hq + overlap),
        (3*hq - overlap,height, 3*hq - overlap,height)
    ]
    
    num_cores = 18
    use_threads = True
    
    results = []
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    with executor_class(max_workers=num_cores) as executor:
        for finalCells in executor.map(seeds_to_splittedObjects, quadrants):
            results.append(finalCells)
    
    # Initialize the final segmentation mask
    #final_mask = np.zeros((height, width), dtype=np.uint8)
    final_filled_mask = np.zeros((height, width), dtype=np.uint8)
    
    for q, quadrant in enumerate(quadrants) : 
        
        row_start = positions[q][0]
        row_end = positions[q][1]
        col_start = positions[q][2]
        col_end = positions[q][3]
        
        final_filled_mask[row_start:row_end, col_start:col_end] = results[q]
        
    
    return final_filled_mask