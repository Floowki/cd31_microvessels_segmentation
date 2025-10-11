
import os 
from tqdm import tqdm
import cv2
import numpy as np
from parallel_segmentation import parallel_vessels_segmentation


def MV_segment_dataset(dataset_path, dataset_segmentation_path) :
    #| Args : 
    #|   # dataset_path : path of the dataset, containing the extracted tiles 
    #|   # dataset_segmentation_path :  path of the dataset to put the tiles in 
    
    #| Outputs :
    #|   # BW2 : original segmented microvessels mask 
    #|   # finalCells : mask with filled segmented microvessels 
    
    filenames = os.listdir(dataset_path)

    for filename in tqdm(filenames, desc="Microvessels segmentation", unit="tile", total=len(filenames)):
        
        path_direction = os.path.join(dataset_segmentation_path + "/finalCells", filename)
        
        if os.path.exists(path_direction):
            #print("Already segmented")
            continue  # Skip and continue 
        
        #print("Tile :", filename)
        img_path = os.path.join(dataset_path, filename)
        img = cv2.imread(img_path) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        

        if img is not None :
            
            
            finalCells = parallel_vessels_segmentation(img)
            finalCells = 255 * finalCells.astype(np.uint8)
            
            output_path_finalCells = os.path.join(dataset_segmentation_path, filename)
            
            cv2.imwrite(output_path_finalCells, finalCells)

    return 



