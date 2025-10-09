
import numpy as np
from scipy.ndimage import label
from skimage.measure import regionprops
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed


def splitCells(dataIn,minAreaObject = 17):
    #| Args : 
    #|   # dataIn : image identifying the cells with more than one inner region
    #|   # minAreaObject : minimum area of an object after splitting it (default arbitrary value = 17 pixels)
    
    #| Outputs :
    #|   # splittedCellsLab : image identifying the splitted cells 
    #|   # numNewObjects : count of new objects after splitting the cells 
    
    if len(dataIn.shape) == 2:
        rows, cols = dataIn.shape
        levs = 1  # Set levs to 1 if there's no third dimension
    elif len(dataIn.shape) == 3:
        rows, cols, levs = dataIn.shape
    
    splitted_cells = np.zeros((rows, cols), dtype=int)
    max_data = np.max(dataIn)
    min_data = np.min(dataIn)

    if (min_data == 0) and (max_data == 1) :
        
        data_labelled, num_objects = label(dataIn, structure=np.ones((3, 3)))
    else:
     
        num_objects = max_data

    for counter_objs in range(1, num_objects + 1):
        current_object = (data_labelled == counter_objs)
        row_proj = np.where(np.max(current_object, axis=1))[0]
        col_proj = np.where(np.max(current_object, axis=0))[0]
        min_row = max(row_proj[0] - 1, 0)
        max_row = min(row_proj[-1] + 1, rows)
        min_col = max(col_proj[0] - 1, 0)
        max_col = min(col_proj[-1] + 1, cols)
        current_object2 = current_object[min_row:max_row, min_col:max_col]
        data_filled = binary_fill_holes(current_object2)
        holes_in_data, num_holes_in_data = label(data_filled ^ current_object2)

        if num_holes_in_data > 1:
            area_of_holes = regionprops(holes_in_data)
            index_to_objects = [i for i, prop in enumerate(area_of_holes) if prop.area >= minAreaObject]

            big_holes_only = np.isin(holes_in_data, index_to_objects)

            if len(index_to_objects) > 1:
                dist_to_holes = distance_transform_edt(big_holes_only)
                boundaries_between_holes = watershed(dist_to_holes)
                splitted_objects, num_splitted_objects = label(boundaries_between_holes > 0, structure=np.ones((3, 3)))
                area_splitted_objects = regionprops(splitted_objects)
                index_to_objects = [i for i, prop in enumerate(area_splitted_objects) if prop.area >= minAreaObject]
                new_object = np.isin(splitted_objects, index_to_objects)
                splitted_cells[min_row:max_row, min_col:max_col] = new_object
            else:
                splitted_cells[min_row:max_row, min_col:max_col] = current_object2

        else:
            splitted_cells[min_row:max_row, min_col:max_col] = current_object2

    splittedCellsLab, numNewObjects = label(splitted_cells)
    
    return splittedCellsLab, numNewObjects