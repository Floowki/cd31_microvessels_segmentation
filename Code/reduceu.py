
import numpy as np
from scipy.ndimage import convolve

def reduceu(data, numReductions, reducein3D):
    #| Args : 
    #|   # data : data to be reduced 
    #|   # numReductions : number of reductions (1 : 1000x1000 -> 500X500, 2: 1000x1000 -> 250x250, etc)
    #|   # reducein3D : if 1, will combine levels; 0, performs reduction in a per-slice basis
    
    #| Outputs :
    #|   # redData : reduced data 
    
    if numReductions == None : 
        numReductions = 1
        
    if reducein3D == None : 
        reducein3D = 0

    if not isinstance(data, np.float64) :
        data = data.astype(np.float64)
        
    
    if numReductions == 0 :
        redData = data
        
    else : 
    
        if numReductions > 1 : 
            data = reduceu(data,numReductions-1,reducein3D)
    

        # perform the reduction depending on the data's dimensions
        if data.ndim == 2 or reducein3D == 0:
            redData = convolve(data, np.array([[1, 1], [1, 1]]))
            redData = redData[1::2, 1::2] / 4 
            
        else:
            redData = convolve(data, np.ones((2, 2, 2)))  
            redData = redData[1::2, 1::2, 1::2] / 8 


    return redData