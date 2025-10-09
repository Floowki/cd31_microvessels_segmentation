
import numpy as np 


# Expand data in uniform levels 
def expandu(data,numExpansions):
    #| Args : 
    #|   # data : data as an image 
    #|   # numExpansions : how many times the data should be expanded 
    
    #| Outputs :
    #|   # expData : expanded data 
 
    if numExpansions == None :
         numExpansions = 1     
 
    # Check if data is empty or None
    if data is None or data.size == 0:
        return np.array([])

    # Default numExpansions to 1 if not provided
    if numExpansions > 0:
        if numExpansions > 1:
            data = expandu(data, numExpansions - 1)

        # Get the dimensions of the input array
        shape = data.shape
        if len(shape) == 2:  # 2D case
            rows, cols = shape
            levels = 1
        elif len(shape) == 3:  # 3D case
            rows, cols, levels = shape
        else:
            raise ValueError("Input data must be either 2D or 3D")

        # Create an expanded array with doubled dimensions
        if levels == 1:
            expData = np.zeros((rows * 2, cols * 2), dtype=data.dtype)
        else:
            expData = np.zeros((rows * 2, cols * 2, levels * 2), dtype=data.dtype)

        # Fill the expanded array
        expData[::2, ::2, ...] = data
        expData[::2, 1::2, ...] = data
        expData[1::2, ::2, ...] = data
        expData[1::2, 1::2, ...] = data

        # If 3D, duplicate the third dimension as well
        if levels > 1:
            expData[:, :, ::2] = expData[:, :, :levels]
            expData[:, :, 1::2] = expData[:, :, ::2]

        return expData

    return data