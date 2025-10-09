
import numpy as np
import math


def gaussF(rowDim, colDim=None, levDim=None, rowSigma=None, colSigma=None, levSigma=None,
           rowMiu=None, colMiu=None, levMiu=None, rho=None):
    #| Args : 
    #|   # rowDim : dimension x  
    #|   # colDim : dimension y
    #|   # levDim : dimension z
    #|   # rowSigma, colSigma, levSigma : sigma value x, y, z 
    #|   # rowmiu, colmiu, levmiu : Miu value x, y, z   ]-inf, inf[
    #|   # rho : oblique distributions angle control ]-1, 1[ default : 0 
    
    #| Outputs :
    #|   # gauss : n-dimensional gaussian function
    
    
    if rowDim is None:
        print(gaussF.__doc__)
        gauss = []
        return gauss

    if rho is None:
        rho = 0

    # Determine the dimensions of the gaussian function
    # can be input vectors as in (size(a))

    if colDim is None:
        rowDim_arr = np.atleast_2d(rowDim)
        wRow, wCol = rowDim_arr.shape
  
        if wCol == 3:
            # 3D case 
            levDim = rowDim_arr[0,2]
            colDim = rowDim_arr[0,1]
            rowDim = rowDim_arr[0,0]
        elif wCol == 2:
            # case 2D
            colDim = rowDim_arr[0,1]
            rowDim = rowDim_arr[0,0]
            levDim = 1
        elif wCol == 1:
            # case 1D 
            rowDim = rowDim_arr[0,0]
            colDim = 1
            levDim = 1
    elif levDim is None:
        levDim = 1

    # x, y, z dimensions of the filter

    # Create a dictionary
    filter = {}
    filter['x'] = np.arange(1, math.ceil(rowDim) + 1)
    filter['y'] = np.arange(1, math.ceil(colDim) + 1)
    filter['z'] = np.arange(1, math.ceil(levDim) + 1)
    filter['data'] = np.zeros((math.ceil(rowDim), math.ceil(colDim), math.ceil(levDim)))
    rr = filter['x'][:, None, None] 
    cc = filter['y'][None, :, None]  
    dd = filter['z'][None, None, :] 


    # determine mius and sigmas in case not provided 
    if rowMiu is None or colMiu is None or levMiu is None:
        rowMiu = np.sum(filter['x']) / len(filter['x'])
        colMiu = np.sum(filter['y']) / len(filter['y'])
        levMiu = np.sum(filter['z']) / len(filter['z'])
    sigmVal = 1.1774

    if rowSigma is None or colSigma is None or levSigma is None:
        rowSigma = (rowMiu - 1) / sigmVal
        colSigma = (colMiu - 1) / sigmVal
        levSigma = (levMiu - 1) / sigmVal

    # sigma must > 0
    rowSigma = max(rowSigma, 0.000001)
    colSigma = max(colSigma, 0.000001)
    levSigma = max(levSigma, 0.000001)

    # Check if rho is not a scalar
    rho_arr = np.array(rho)
    if np.prod(rho_arr.shape) != 1:
        # rho is the covariance matrix
        if rho_arr.shape[0] == 2:
            invSigma = np.linalg.inv(rho_arr)
            Srr = invSigma[0, 0]
            Scc = invSigma[1, 1]
            Src = 2 * invSigma[1, 0]
            Srd = 0
            Scd = 0
            Sdd = 0
        else:
            invSigma = np.linalg.inv(rho_arr)
            Srr = invSigma[0, 0]
            Scc = invSigma[1, 1]
            Src = 2 * invSigma[1, 0]
            Srd = 2 * invSigma[0, 2]
            Scd = 2 * invSigma[1, 2]
            Sdd = invSigma[2, 2]
        exp_r = (1/(rowSigma**2)) * (rr - rowMiu)**2
        exp_c = (1/(colSigma**2)) * (cc - colMiu)**2
        exp_d = (1/(levSigma**2)) * (dd - levMiu)**2
        exp_rc = (1/(rowSigma * colSigma)) * (rr - rowMiu) * (cc - colMiu)
        exp_rd = (1/(rowSigma * levSigma)) * (rr - rowMiu) * (dd - levMiu)
        exp_cd = (1/(levSigma * colSigma)) * (dd - levMiu) * (cc - colMiu)
        gauss = np.exp(-(Srr * exp_r + Scc * exp_c + Sdd * exp_d + Src * exp_rc + Srd * exp_rd + Scd * exp_cd))
    else:
        rho = min(rho, 0.999999)
        rho = max(rho, -0.999999)

        # in each dimension, calculate exponential functions in each dimension
        filter['x2'] = (1/(np.sqrt(2*np.pi)*rowSigma)) * np.exp(-((filter['x'] - rowMiu)**2) / (2 * rowSigma**2))
        filter['y2'] = (1/(np.sqrt(2*np.pi)*colSigma)) * np.exp(-((filter['y'] - colMiu)**2) / (2 * colSigma**2))
        filter['z2'] = (1/(np.sqrt(2*np.pi)*levSigma)) * np.exp(-((filter['z'] - levMiu)**2) / (2 * levSigma**2))

        filter['x2'] = filter['x2'] / np.sum(filter['x2'])
        filter['y2'] = filter['y2'] / np.sum(filter['y2'])
        filter['z2'] = filter['z2'] / np.sum(filter['z2'])

        rhoExponent = - (rho / (rowSigma * colSigma)) * np.outer((filter['x'] - rowMiu), (filter['y'] - colMiu))

        filter['rho'] = (1/np.sqrt(1 - rho**2)) * np.exp(rhoExponent)

        # get the 2D function
        if colDim > 1 and rowDim > 1:
            twoDFilter = np.outer(filter['x2'], filter['y2']) * filter['rho']
            # get the 3D function 
            if math.ceil(levDim) > 1:
                threeDFilter = np.empty((len(filter['x2']), len(filter['y2']), math.ceil(levDim)))
                for ii in range(math.ceil(levDim)):
                    threeDFilter[:, :, ii] = twoDFilter * filter['z2'][ii]
                gauss = threeDFilter
            else:
                gauss = twoDFilter
        else:
            # covers the 1D cases both row and column
            if len(filter['x2']) > len(filter['y2']):
                gauss = filter['x2']
            else:
                gauss = filter['y2']
        # set any NaN values in gauss to 0
        gauss = np.nan_to_num(gauss, nan=0)
        
    return gauss