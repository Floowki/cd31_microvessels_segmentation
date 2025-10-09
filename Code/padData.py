
import numpy as np

def padData(qtdata,numPadPixels, dimsToPad, padWith):
    #| Args : 
    #|   # qtdata : data to be padded
    #|   # numPadPixels : defines the padding area 
    #|   # dimsToPad : the dimension of the data to pad, can be empty 
    #|   # padWith : the value to pad with 
    
    #| Outputs :
    #|   # data_padded : the padded data 
    
    qtdata = np.expand_dims(qtdata, axis=(2, 3))  # Converts (rows, cols) → (rows, cols, 1, 1)
    
    if padWith == None:
        padWith = 1
    
    if ( dimsToPad == None ) or ( len(dimsToPad) ) == 0:
        rows, cols, levs, num_feats = qtdata.shape

    else:
        dimsToPad = list(dimsToPad) + [1] * (4 - len(dimsToPad)) # concatenation 
        rows, cols, levs, num_feats = dimsToPad[:4]
    
    # if needed: pad rows 
    if rows > 1:
        top_padding = padWith * np.tile(qtdata[0:1, :, :, :], (numPadPixels, 1, 1, 1))
        bottom_padding = padWith * np.tile(qtdata[-1:, :, :, :], (numPadPixels, 1, 1, 1))
        qtdata = np.concatenate([top_padding, qtdata, bottom_padding], axis=0)
    
    # if needed : pad columns 
    if cols > 1:
        left_padding = padWith * np.tile(qtdata[:, 0:1, :, :], (1, numPadPixels, 1, 1))
        right_padding = padWith * np.tile(qtdata[:, -1:, :, :], (1, numPadPixels, 1, 1))
        qtdata = np.concatenate([left_padding, qtdata, right_padding], axis=1)
    
    data_padded = qtdata
    
    # if needed : pad levels 
    if levs > 1:
        new_shape = list(qtdata.shape)
        new_shape[2] += 2 * numPadPixels
        qtdata3 = np.zeros(new_shape, dtype=qtdata.dtype)
        qtdata3[:, :, numPadPixels:numPadPixels + qtdata.shape[2], :] = qtdata
        
        front_padding = padWith * np.tile(qtdata[:, :, 0:1, :], (1, 1, numPadPixels, 1))
        back_padding = padWith * np.tile(qtdata[:, :, -1:, :], (1, 1, numPadPixels, 1))
        
        qtdata3[:, :, :numPadPixels, :] = front_padding
        qtdata3[:, :, numPadPixels + qtdata.shape[2]:, :] = back_padding
        
        data_padded = qtdata3
    
    return np.squeeze(data_padded)