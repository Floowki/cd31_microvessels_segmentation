import numpy as np 

# Discretize the distinct values that can be taken 
def quanti_r(data, bitsq=1, zeroLevel=None):
    #| Args : 
    #|   # data : data to be quantized 
    #|   # bitsq : 
    #|   # zeroLevel : 
    
    #| Outputs :
    #|   # lq_data : linearly quantized data
    #|   # data : original data 
    
    if data is None:
        return [], []

    maxdata = np.max(data)
    if zeroLevel is not None:
        mindata = 0
    else:
        mindata = np.min(data)

    # case of all zeros, return from here
    if maxdata == 0 and mindata == 0:
        return data, data

    # case of constant intensity, return from here
    if mindata == maxdata:
        return data, data

    # case of values between 0 and 1, enlarge because the 'round' does not work for small numbers
    if maxdata <= 1:
        lq_data = (np.round(data * (2**bitsq - 1))) / (2**bitsq - 1)
        nl_qdata = data
        return lq_data, nl_qdata

    maxbits = np.ceil(np.log2(maxdata - mindata)).astype(int)  # number of bits that describe the data

    # makes sure that the levels remains possible
    if bitsq >= maxbits or bitsq == 0:
        return data, data

    step = (maxdata - mindata) / (2**bitsq - 1)

    # linear quantising
    lq_data = np.round(mindata + step * np.round((data - mindata) / step))
    
    return lq_data, data