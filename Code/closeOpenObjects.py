
import numpy as np 
import ndimage
from scipy.ndimage import label
from skimage import measure, morphology
from scipy.ndimage import distance_transform_edt


def CloseOpenObjects(BW1):
    #| Args : 
    #|   # BW1 : image with identifyed objects - binary image
    
    #| Outputs :
    #|   # BW3 : Processed binary image with close objects connected


    # Make sure input is binary
    BW1 = BW1.astype(bool)
    
    # Label connected components
    BW2, num_objects = ndimage.label(BW1)
    
    # Initialize output image
    BW3 = np.zeros_like(BW1, dtype=bool)
    
    # Process each object
    for obj_id in range(1, num_objects + 1):
        # Extract the object mask
        obj_mask = (BW2 == obj_id)
        
        # Dilate slightly to find nearby regions
        dilated = morphology.binary_dilation(obj_mask, morphology.disk(2))
        
        # Find objects that are close to the current object
        nearby_objs = np.unique(BW2[dilated & ~obj_mask])
        
        # Remove 0 (background) if it's in the list
        nearby_objs = nearby_objs[nearby_objs > 0]
        
        # If there are nearby objects, check if they should be connected
        if len(nearby_objs) > 0:
            # For each nearby object
            for near_id in nearby_objs:
                near_mask = (BW2 == near_id)
                
                # Calculate the shortest distance between objects
                dist_obj1 = ndimage.distance_transform_edt(~obj_mask)
                dist_obj2 = ndimage.distance_transform_edt(~near_mask)
                
                # Find minimum distance between objects
                min_dist = np.min(dist_obj1[near_mask])
                
                # If objects are close enough, connect them
                if min_dist < 5:  # Threshold distance for connection
                    # Create a line between the closest points
                    point1 = np.unravel_index(np.argmin(dist_obj1[near_mask]), dist_obj1.shape)
                    point2 = np.unravel_index(np.argmin(dist_obj2[obj_mask]), dist_obj2.shape)
                    
                    # Draw a line to connect the objects
                    rr, cc = np.linspace(point1[0], point2[0], 10).astype(int), np.linspace(point1[1], point2[1], 10).astype(int)
                    for r, c in zip(rr, cc):
                        if 0 <= r < BW1.shape[0] and 0 <= c < BW1.shape[1]:
                            BW3[r, c] = True
        
        # Add the original object to the output
        BW3 = BW3 | obj_mask
    
    return BW3