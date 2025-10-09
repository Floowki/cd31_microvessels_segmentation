
import numpy as np

def zerocross(img): 
    #| Args : 
    #|   # img : the image to extract the edges from ( indeed the zero-crossings higlighht the edges)
    
    #| Outputs :
    #|   # bor : detected borders / edges in the image 
        
    lins, cols = img.shape[:2]
    if len(img.shape) == 2 : 
        levels = 1 
    else :
        levels = img.shape[2]
        
    delta = 0.00001
    
    if not np.issubdtype(img.dtype, np.bool_):
        
        img = np.sign(img)
    
    if ((cols == 1 or lins == 1) and levels == 1) :  # 1D case
        yy = np.concatenate(([0], img[:-1]))
        bor = np.abs(np.sign(img + delta) - np.sign(yy + delta))
    
    elif (cols == 1 and lins == 1 and levels != 1) :  # 1D over another plane
        img = np.transpose(img, (1, 2, 0))
        yy = np.concatenate(([0], img[:-1]))
        bor = np.sign(img + delta) - np.sign(yy + delta)
    
    elif (lins != 1 and cols != 1 and levels == 1):  # 2D case
        diffVer = np.diff(img, axis=0)
        zerCols = np.zeros((1, cols))
        diffHor = np.diff(img, axis=1)
        zerRows = np.zeros((lins, 1))
        
        qq1 = np.vstack((zerCols, diffVer > 0))
        qq2 = np.vstack((diffVer < 0, zerCols))
        qq3 = np.hstack((diffHor < 0, zerRows))
        qq4 = np.hstack((zerRows, diffHor > 0))
        
        bor = (qq1.astype(bool) | qq2.astype(bool) | qq3.astype(bool) | qq4.astype(bool)).astype(np.uint8)
    
    elif (lins != 1 and cols!= 1 and levels != 1):  # 3D case
        yy5 = np.vstack((np.zeros((1, cols, levels)), img[:-1, :, :]))
        yy6 = np.vstack((img[1:, :, :], np.zeros((1, cols, levels))))
        yy7 = np.hstack((np.zeros((lins, 1, levels)), img[:, :-1, :]))
        yy8 = np.hstack((img[:, 1:, :], np.zeros((lins, 1, levels))))
        
        bor5 = np.fix(delta + (np.sign(img) - np.sign(yy5)) / 2)
        bor6 = np.fix(delta + (np.sign(img) - np.sign(yy6)) / 2)
        bor7 = np.fix(delta + (np.sign(img) - np.sign(yy7)) / 2)
        bor8 = np.fix(delta + (np.sign(img) - np.sign(yy8)) / 2)
        
        bor = np.sign(bor5 + bor6 + bor7 + bor8)
    
    return bor