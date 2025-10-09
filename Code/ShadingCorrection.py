
import numpy as np 
from scipy.signal import convolve2d
import math
import gaussF

def simple_pad_2d(data_2d, pad_size):
    
    return np.pad(data_2d, int(pad_size), mode='edge')



def ShadingCorrection(dataIn, numScales = None):
    #| Args : 
    #|   # dataIn :  an image with cells vessels or any other kind of objects
    #|   # numScales : number ... 
    
    #| Outputs :
    #|   # dataOut : image with a uniform background
    #|   # errSurface : the shading surface
    #|   # errSurfaceMin : the lower envelope
    #|   # errSurfaceMax : the higher envelope
    

    ## Parse input data
    if dataIn is None:
        print(ShadingCorrection.__doc__)
        return [], None, None, None

    # Convert input to numpy array
    dataIn = np.asarray(dataIn, dtype=np.float64)
    
    # Handle dimensionality
    orig_shape = dataIn.shape
    if dataIn.ndim == 2:
        dataIn = dataIn[:, :, np.newaxis]  # Add channel dimension
    
    # Get dimensions
    rows, cols, levs = dataIn.shape
    reconvertTo255 = 1 if dataIn.dtype != np.float64 else 0

    # Set default parameters
    if numScales is None:
        numScales = 55
    stopCriterion = 0.05


    # Multi-channel case
    if levs > 1:
        dataOut = np.zeros((rows, cols, levs))
        subSampl = 1
        
        # Process each channel
        dataOut_S = np.zeros((rows, cols, levs))
        errSurface_S = np.zeros((rows, cols, levs))
        errSurfaceMin_S = np.zeros((rows, cols, levs))
        errSurfaceMax_S = np.zeros((rows, cols, levs))
        
        for currLev in range(levs):
            # Get current channel and ensure it's 2D
            sliced = dataIn[0::subSampl, 0::subSampl, currLev]
            
            # Process single channel
            dOut, errSurf, errSurfMin, errSurfMax = ShadingCorrection(sliced, numScales)
            
            # Store results, handling potential size differences
            r_limit = min(rows, dOut.shape[0] if dOut.ndim == 2 else dOut.shape[0])
            c_limit = min(cols, dOut.shape[1] if dOut.ndim == 2 else dOut.shape[1])
            
            dataOut_S[:r_limit, :c_limit, currLev] = dOut[:r_limit, :c_limit] if dOut.ndim == 2 else dOut[:r_limit, :c_limit, 0]
            errSurface_S[:r_limit, :c_limit, currLev] = errSurf[:r_limit, :c_limit]
            errSurfaceMin_S[:r_limit, :c_limit, currLev] = errSurfMin[:r_limit, :c_limit]
            errSurfaceMax_S[:r_limit, :c_limit, currLev] = errSurfMax[:r_limit, :c_limit]
        
        # Combine results
        dataOut = dataOut_S
        errSurface = errSurface_S
        errSurfaceMax = errSurfaceMax_S
        errSurfaceMin = errSurfaceMin_S
        
        # Clip and normalize
        dataOut[dataOut > 255] = 255
        dataOut[dataOut < 0] = 0
        dataOut = dataOut - np.min(dataOut)
        if np.max(dataOut) > 0:
            dataOut = 255 * (dataOut / np.max(dataOut))
    
    # Single-channel case
    else:
        # Get 2D image data
        img_data = np.squeeze(dataIn[:, :, 0])  # Ensure 2D by removing singleton dimensions
        
        # Initialize arrays
        y_scaleMax = np.zeros((rows, cols, int(np.ceil(numScales/2))))
        y_scaleMin = np.zeros((rows, cols, int(np.ceil(numScales/2))))
        dataOut = np.zeros((rows, cols, 1))
        
        # Create Gaussian filter
        sizeFR_S = 3
        sizeFC_S = sizeFR_S
        #filtG_small = gaussF(sizeFR_S, sizeFC_S, 1, 0)
        filtG_small = gaussF(sizeFR_S, sizeFC_S)
        
        # Ensure filter is 2D for convolve2d
        if filtG_small.ndim > 2:
            filtG_small = filtG_small[:, :, 0]
        elif filtG_small.ndim == 1:
            filtG_small = np.outer(filtG_small, np.ones(1))
        
        
        # Create padded data for convolution
        padded_data = simple_pad_2d(img_data, math.ceil(sizeFR_S/2))
        #print(f"Padded data shape: {padded_data.shape}, ndim: {padded_data.ndim}")
        
        # Apply low-pass filter
        y1 = convolve2d(padded_data, filtG_small, mode='full')
        
        # Fix: Resize y_LPF to match the original image size
        # First extract the valid region from the convolution
        y_LPF_full = y1[sizeFR_S:y1.shape[0]-sizeFR_S, sizeFC_S:y1.shape[1]-sizeFC_S]
        
        # Resize to match original image dimensions
        y_LPF = np.zeros((rows, cols))
        min_rows = min(rows, y_LPF_full.shape[0])
        min_cols = min(cols, y_LPF_full.shape[1])
        y_LPF[:min_rows, :min_cols] = y_LPF_full[:min_rows, :min_cols]
        

        # Process at different scales
        tot_grad_max1 = []
        tot_grad_min1 = []
        
        for cStep in range(1, numScales+1, 2):
            cStep2 = int(np.ceil(cStep/2))
            y_scaleMax[:, :, cStep2-1] = y_LPF
            y_scaleMin[:, :, cStep2-1] = y_LPF

            # Calculate neighbors
            if rows > 2*cStep and cols > 2*cStep:
                tempNW = y_LPF[0:rows-2*cStep, 0:cols-2*cStep]
                tempSW = y_LPF[2*cStep:rows, 0:cols-2*cStep]
                tempSE = y_LPF[2*cStep:rows, 2*cStep:cols]
                tempNE = y_LPF[0:rows-2*cStep, 2*cStep:cols]
                
                tempN = y_LPF[0:rows-2*cStep, cStep:cols-cStep]
                tempW = y_LPF[cStep:rows-cStep, 0:cols-2*cStep]
                tempS = y_LPF[2*cStep:rows, cStep:cols-cStep]
                tempE = y_LPF[cStep:rows-cStep, 2*cStep:cols]
                
                # Calculate averages
                tempAv1 = (tempNE + tempSW) / 2
                tempAv2 = (tempNW + tempSE) / 2
                tempAv3 = (tempN + tempS) / 2
                tempAv4 = (tempW + tempE) / 2
                
                # Stack and find min/max
                tempAv = np.stack((tempAv1, tempAv2, tempAv3, tempAv4), axis=2)
                tempAvMax = np.max(tempAv, axis=2)
                tempAvMin = np.min(tempAv, axis=2)
                
                # Update scales
                y_scaleMax[cStep:rows-cStep, cStep:cols-cStep, cStep2-1] = np.maximum(
                    y_scaleMax[cStep:rows-cStep, cStep:cols-cStep, cStep2-1], tempAvMax)
                y_scaleMin[cStep:rows-cStep, cStep:cols-cStep, cStep2-1] = np.minimum(
                    y_scaleMin[cStep:rows-cStep, cStep:cols-cStep, cStep2-1], tempAvMin)

                # Find absolute min/max at all scales
                yMin = np.min(y_scaleMin[:, :, 0:cStep2], axis=2)
                yMax = np.max(y_scaleMax[:, :, 0:cStep2], axis=2)
                
                # Create new filter
                sizeFR_S = cStep
                sizeFC_S = sizeFR_S
                
                # Generate filter with correct parameters
                filtG_small = gaussF(sizeFR_S, sizeFC_S)
                
                # Ensure filter is 2D
                if filtG_small.ndim > 2:
                    filtG_small = filtG_small[:, :, 0]
                elif filtG_small.ndim == 1:
                    filtG_small = np.outer(filtG_small, np.ones(1))
                
                # Skip if too few points for downsampling
                if yMin.shape[0] < 3 or yMin.shape[1] < 3:
                    continue
                
                # Downsample and filter
                yMin_ds = yMin[::2, ::2]
                yMax_ds = yMax[::2, ::2]
                
                # Use simple pad for 2D data
                yMin0 = simple_pad_2d(yMin_ds, math.ceil(sizeFR_S/2))
                yMax0 = simple_pad_2d(yMax_ds, math.ceil(sizeFR_S/2))
                
                # Apply convolution
                try:
                    yMin1 = convolve2d(yMin0, filtG_small, mode='full')
                    
                    # Fix: Get the valid region and resize
                    yMin1_valid = yMin1[sizeFR_S:yMin1.shape[0]-sizeFR_S, sizeFC_S:yMin1.shape[1]-sizeFC_S]
                    yMin2 = np.zeros((yMin_ds.shape[0], yMin_ds.shape[1]))
                    min_r = min(yMin2.shape[0], yMin1_valid.shape[0])
                    min_c = min(yMin2.shape[1], yMin1_valid.shape[1])
                    yMin2[:min_r, :min_c] = yMin1_valid[:min_r, :min_c]
                    
                    yMax1 = convolve2d(yMax0, filtG_small, mode='full')
                    
                    # Fix: Get the valid region and resize
                    yMax1_valid = yMax1[sizeFR_S:yMax1.shape[0]-sizeFR_S, sizeFC_S:yMax1.shape[1]-sizeFC_S]
                    yMax2 = np.zeros((yMax_ds.shape[0], yMax_ds.shape[1]))
                    min_r = min(yMax2.shape[0], yMax1_valid.shape[0])
                    min_c = min(yMax2.shape[1], yMax1_valid.shape[1])
                    yMax2[:min_r, :min_c] = yMax1_valid[:min_r, :min_c]
                    
                except Exception as e:
                    print(f"Error in convolution: {e}")
                    print(f"Filter shape: {filtG_small.shape}, yMin0 shape: {yMin0.shape}")
                    continue
                
                # Calculate derivatives
                if yMin2.shape[0] > 1 and yMin2.shape[1] > 1:
                    y_rderiv_Max = np.diff(yMax2, axis=0)
                    y_cderiv_Max = np.diff(yMax2, axis=1)
                    y_rderiv_Min = np.diff(yMin2, axis=0)
                    y_cderiv_Min = np.diff(yMin2, axis=1)
                    
                    # Calculate gradient magnitudes
                    if y_rderiv_Max.shape[0] > 0 and y_cderiv_Max.shape[1] > 0:
                        y_magGrad_Max = np.sqrt((y_rderiv_Max[:, 1:]**2) + (y_cderiv_Max[1:, :]**2))
                        y_magGrad_Min = np.sqrt((y_rderiv_Min[:, 1:]**2) + (y_cderiv_Min[1:, :]**2))
                        
                        # Calculate totals
                        tot_grad_max1.append(np.sum(y_magGrad_Max))
                        tot_grad_min1.append(np.sum(y_magGrad_Min))
                        
                        # Check stop criterion
                        if cStep > 1 and len(tot_grad_max1) >= 3:
                            if tot_grad_max1[-3] != 0 and tot_grad_min1[-3] != 0:
                                diffGradMax = abs((tot_grad_max1[-1]-tot_grad_max1[-3]) / tot_grad_max1[-3])
                                diffGradMin = abs((tot_grad_min1[-1]-tot_grad_min1[-3]) / tot_grad_min1[-3])
                                
                                if (diffGradMax < stopCriterion) or (diffGradMin < stopCriterion):
                                    break
            else:
                # Skip this step if dimensions are too small
                continue
        
        # Recalculate final surfaces if we have gradients
        if tot_grad_max1 and tot_grad_min1:
            tot_grad_max = tot_grad_max1[-1]
            tot_grad_min = tot_grad_min1[-1]
            
            # Create large filters for final calculation
            sizeFR_S = cStep
            sizeFC_S = sizeFR_S
            
            # Generate filter with correct parameters
            filtG_small = gaussF(sizeFR_S, sizeFC_S)
            
            # Ensure filter is 2D
            if filtG_small.ndim > 2:
                filtG_small = filtG_small[:, :, 0]
            elif filtG_small.ndim == 1:
                filtG_small = np.outer(filtG_small, np.ones(1))
            
            # Apply final filtering using simple padding
            yMin0 = simple_pad_2d(yMin, math.ceil(sizeFR_S/2))
            yMax0 = simple_pad_2d(yMax, math.ceil(sizeFR_S/2))
            
            try:
                yMin1 = convolve2d(yMin0, filtG_small, mode='full')
                
                # Fix: Get the valid region and resize
                yMin1_valid = yMin1[sizeFR_S:yMin1.shape[0]-sizeFR_S, sizeFC_S:yMin1.shape[1]-sizeFC_S]
                yMin2 = np.zeros((yMin.shape[0], yMin.shape[1]))
                min_r = min(yMin2.shape[0], yMin1_valid.shape[0])
                min_c = min(yMin2.shape[1], yMin1_valid.shape[1])
                yMin2[:min_r, :min_c] = yMin1_valid[:min_r, :min_c]
                
                yMax1 = convolve2d(yMax0, filtG_small, mode='full')
                
                # Fix: Get the valid region and resize
                yMax1_valid = yMax1[sizeFR_S:yMax1.shape[0]-sizeFR_S, sizeFC_S:yMax1.shape[1]-sizeFC_S]
                yMax2 = np.zeros((yMax.shape[0], yMax.shape[1]))
                min_r = min(yMax2.shape[0], yMax1_valid.shape[0])
                min_c = min(yMax2.shape[1], yMax1_valid.shape[1])
                yMax2[:min_r, :min_c] = yMax1_valid[:min_r, :min_c]
                
            except Exception as e:
                print(f"Error in final convolution: {e}")
                print(f"Filter shape: {filtG_small.shape}, yMin0 shape: {yMin0.shape}")
                # Use original data as fallback
                dataOut[:, :, 0] = img_data
                errSurface = np.zeros_like(img_data)
                errSurfaceMax = np.zeros_like(img_data)
                errSurfaceMin = np.zeros_like(img_data)
                return dataOut, errSurface, errSurfaceMin, errSurfaceMax
            
            # Choose surface based on gradients
            if tot_grad_max != 0 and abs((tot_grad_max-tot_grad_min)/tot_grad_max) < 0.05:
                yProm = 0.5*yMin2 + 0.5*yMax2
                errSurface = yProm - np.mean(yProm)
            else:
                if tot_grad_max > tot_grad_min:
                    errSurface = yMin2 - np.mean(yMin2)
                else:
                    errSurface = yMax2 - np.mean(yMax2)
            
            # Apply correction
            dataOut[:, :, 0] = img_data - errSurface
            
            # Set error surfaces
            errSurfaceMax = yMax2
            errSurfaceMin = yMin2
        else:
            # Default if we couldn't calculate gradients
            dataOut[:, :, 0] = img_data
            errSurface = np.zeros_like(img_data)
            errSurfaceMax = np.zeros_like(img_data)
            errSurfaceMin = np.zeros_like(img_data)
            
        # Clip values if needed
        if reconvertTo255 == 1:
            dataOut[dataOut > 255] = 255
            dataOut[dataOut < 0] = 0
    
    # Reshape output to match input
    if len(orig_shape) == 2:
        dataOut = dataOut[:, :, 0]
    
    return dataOut, errSurface, errSurfaceMin, errSurfaceMax