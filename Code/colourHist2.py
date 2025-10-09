
import numpy as np 
import quanti_r 

def colourHist2(dataHSV,sizeHue,sizeSaturation=None,sizeValue=None):
    #| Args : 
    #|   # dataHSV : img in the HSV space 
    #|   # sizeHue : None
    #|   # size_Saturation : None  
    #|   # sizeValue : None 
    
    #| Outputs :
    #|   # [hs_im1,chrom3D,dataHue,dataSaturation,dataValue] : corresponding Joint Histograms 
    

    if sizeSaturation is None:
        sizeSaturation = np.ceil(sizeHue / 2).astype(int)
        
    if sizeValue is None:
        sizeValue = np.ceil(sizeHue / 2).astype(int)

    # Extract HSV channels
    dataHue_raw = dataHSV[:, :, 0].copy()
    dataSaturation_raw = dataHSV[:, :, 1].copy()
    dataValue_raw = dataHSV[:, :, 2].copy()

    # Quantize each channel
    dataHue, _ = quanti_r(dataHue_raw, bitsq=int(np.log2(sizeHue)))
    dataHue = 1 + (sizeHue - 1) * dataHue
    
    dataSaturation, _ = quanti_r(dataSaturation_raw, bitsq=int(np.log2(sizeSaturation)))
    dataSaturation = 1 + (sizeSaturation - 1) * dataSaturation
    
    # Normalize Value if necessary
    if np.max(dataValue_raw) > 1:
        dataValue_raw = dataValue_raw / 255
        
    dataValue, _ = quanti_r(dataValue_raw, bitsq=int(np.log2(sizeValue)))
    dataValue = 1 + (sizeValue - 1) * dataValue
    
    # Make copies for return values - IMPORTANT: these must remain 2D
    dataHue2 = dataHue.copy()
    dataSaturation2 = dataSaturation.copy()
    dataValue2 = dataValue.copy()
    
    # Initialize the 3D histogram array
    chromaticity3D = np.zeros((sizeSaturation, sizeHue, sizeValue))
    
    # Convert to integers for indexing
    dataHue_int = np.round(dataHue).astype(int)
    dataSaturation_int = np.round(dataSaturation).astype(int)
    dataValue_int = np.round(dataValue).astype(int)
    
    # Define the bin edges for Value
    #kk3 = np.arange(1, sizeValue + 1)
    
    # Flatten arrays for processing (only for the loop, not for returns)
    dataHue_flat = dataHue_int.flatten()
    dataSaturation_flat = dataSaturation_int.flatten()
    dataValue_flat = dataValue_int.flatten()
    
    # Loop over Hue levels
    for k in range(1, sizeHue + 1):
        # Find indices where dataHue equals k
        tempHue_idx = np.where(dataHue_flat == k)[0]
        
        if len(tempHue_idx) > 0:
            # Get values for this hue level
            temp_sat = dataSaturation_flat[tempHue_idx]
            temp_val = dataValue_flat[tempHue_idx]
            
            # Loop over Saturation levels
            for k2 in range(1, sizeSaturation + 1):
                # Find indices where saturation equals k2
                tempSat_idx = np.where(temp_sat == k2)[0]
                
                if len(tempSat_idx) > 0:
                    # Get value data for this saturation level
                    tempSat_values = temp_val[tempSat_idx]
                    
                    # More efficient version of the histogram calculation:
                    values, counts = np.unique(tempSat_values, return_counts=True)
                    hist = np.zeros(sizeValue)
                    for val, count in zip(values, counts):
                        if 1 <= val <= sizeValue:  # make sure the value is within our range
                            hist[int(val)-1] = count

            
            # Remove processed indices
            mask = np.ones(len(dataHue_flat), dtype=bool)
            mask[tempHue_idx] = False
            dataHue_flat = dataHue_flat[mask]
            dataSaturation_flat = dataSaturation_flat[mask]
            dataValue_flat = dataValue_flat[mask]
    
    # Sum over Value axis to get Hue-Saturation histogram
    h_hue_sat = np.sum(chromaticity3D, axis=2)
    
    return h_hue_sat, chromaticity3D, dataHue2, dataSaturation2, dataValue2