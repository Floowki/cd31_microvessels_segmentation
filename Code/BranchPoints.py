
import numpy as np 
from skimage.morphology import closing, binary_erosion, square 
from scipy.ndimage import convolve, binary_dilation
import padData


def bwhitmiss(image, kernel):
    #| Args : 
    #|   # image : image from which the branching points should be detected 
    #|   # kernel : the kernel to perform the hit-or-miss operation 
    
    #| Outputs :
    #|          # foreground_erosion : eroded foreground (=ROI)
    #|          # background_dilation : dilated background 
    #|   # = foreground_erosion & background_dilation (intersection)
    
    # Erosion of the foreground (white areas in the image)
    foreground_erosion = binary_erosion(image, structure=kernel)
    
    # Dilation of the background (black areas in the image)
    background_dilation = binary_dilation(~image, structure=kernel)
    
    return foreground_erosion & background_dilation
    
# Find the branching points in a binary image with a skeletonized structure 
def BranchPoints(dataIn):
    #| Args : 
    #|   # dataIn : binary image with skeletal structures identified 
    
    #| Outputs :
    #|   # branch_points1 : detected branch points 
    #|   # branch_points2 : spread detected branch points across neighbouring pixels 
    #|   # num_branch_points : total number of branch points 
    
    
    # Define structuring elements (kernels)
    branch_kernel1 = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]])  # Corner
    branch_kernel2 = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 1]])  # Y-junction
    branch_kernel3 = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]])  # Diagonal
    
    # Pad input data
    dataIn = padData(dataIn, 1, [], 0)
    
    # Initialize output array
    branch_points1 = np.zeros_like(dataIn, dtype=bool)
    
    # Apply hit-or-miss transformation for all rotations (0, 90, 180, 270 degrees)
    for k in range(4):
        rotated_k1 = np.rot90(branch_kernel1, k)
        rotated_k2 = np.rot90(branch_kernel2, k)
        rotated_k3 = np.rot90(branch_kernel3, k)
        
        branch_points1 = np.logical_or(branch_points1,  bwhitmiss(dataIn, rotated_k1))
        branch_points1 = np.logical_or(branch_points1,  bwhitmiss(dataIn, rotated_k2))
        branch_points1 = np.logical_or(branch_points1,  bwhitmiss(dataIn, rotated_k3))
    
    # Remove padding
    branch_points1 = branch_points1[1:-1, 1:-1]

    # If additional outputs are required
    branch_points2 = branch_points1.copy()
    
    branch_points2[:-1, :] |= branch_points1[1:, :]
    branch_points2[1:, :] |= branch_points1[:-1, :]
    branch_points2[:, :-1] |= branch_points1[:, 1:]
    branch_points2[:, 1:] |= branch_points1[:, :-1]

    num_branch_points = np.sum(branch_points1 > 0)

    return branch_points1, branch_points2, num_branch_points 