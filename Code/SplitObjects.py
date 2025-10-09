
import numpy as np
from skimage.measure import regionprops 
from scipy.ndimage import label
from skimage.segmentation import watershed
from skimage.morphology import closing, square
import expandu
import reduceu
import regionGrowing

def SplitObjects(largeEndoCells,statsObjects = None,rows = None, cols = None):
    #| Args : 
    #|   # largeEndoCells : the image with the labelled Objects
    #|   # statsObjects : the region properties of the objects
    #|   # rows, cols : dimensions of the image 
    
    #| Outputs :
    #|   # NosplittedCells : those objects that were not splitted (small holes or one hole)
    #|   # splittedCells : objects that were split
    #|   # NoHoleCells : solid objects that did not contain holes
    
    
    # If rows (and cols) are not provided, compute from largeEndoCells shape
    if rows is None or cols is None:
        rows, cols = largeEndoCells.shape

    # If statsObjects is not provided, compute regionprops on largeEndoCells.
    if statsObjects is None:
        # ------- extracts the characteristics of the Remaining objects
        statsObjects = regionprops(largeEndoCells)
    
    # ------- cycle over the large cells to determine their equivalente ellipse but first determine whether
    # ------- A - they have several holes and thus need to be split into several cells or
    # ------- B - they can be combined into a single new cell that was split before
    splittedCells = np.zeros((rows, cols), dtype=np.uint8)
    NosplittedCells = np.zeros((rows, cols), dtype=np.uint8)
    NoHoleCells = np.zeros((rows, cols), dtype=np.uint8)
    minAreaObject = 25
    numLargeCells = len(statsObjects)

    for counterObjs in range(numLargeCells):
        # ------- determine bounding box indices
        # In MATLAB, BoundingBox = [x y width height].
        # In skimage, bbox = (min_row, min_col, max_row, max_col) with max indices exclusive.
        stat = statsObjects[counterObjs]
        # Use ceil/floor as in MATLAB
        minRow = max(0, int(np.ceil(stat.bbox[0])))
        minCol = max(0, int(np.ceil(stat.bbox[1])))
        maxRow = min(rows, int(np.floor(stat.bbox[2])))
        maxCol = min(cols, int(np.floor(stat.bbox[3])))
        # figure(3); surfdat(largeEndoCells==counterObjs); drawnow;
        # ----- process only objects with holes
        if stat.euler_number < 0:
            # ----- areas are different if there is at least one hole in the object
            # ----- label difference between image and the filled image, i.e. the HOLES
            diff_image = stat.filled_image ^ stat.image
            holesInData, numHolesInData = label(diff_image, return_num=True)
            if numHolesInData > 1:
                # ------ more than one hole implies  linked objects that need to be splitted
                areaOfHoles = regionprops(holesInData)
                # ------- remove holes that are smaller than a minimum Area **** 17 **** pixels
                indexToObjects = np.where(np.array([prop.area for prop in areaOfHoles]) >= minAreaObject)[0]
                # Create binary image including only big holes
                bigHolesOnly = np.isin(holesInData, indexToObjects + 1)
                if len(indexToObjects) > 1:
                    #            [distToHoles, correspondingHoles] =  (bwdist(holesInData));
                    # [distToHoles, correspondingHoles] =  (bwdist(bigHolesOnly));
                    distToHoles, distBetHoles = regionGrowing(bigHolesOnly, 1 - stat.filled_image)
                    #print("distToHoles shape:", distToHoles.shape)
                    if np.any(distBetHoles[:,0] > 3):
                        # ------ avoid splitting between very close holes
                        distToHoles[distToHoles == 0] = np.max(distToHoles)
                        boundariesBetweenHoles = watershed(distToHoles)
                        # ------ close regions where distance between holes is less than 10
                        if np.any(distBetHoles[:,0] < 10):
                            for k in range(distBetHoles.shape[0]):
                                if (distBetHoles[k,0] < 10) and (np.sum(distBetHoles[k,3:5]) < 199):
                                    mask = np.isin(boundariesBetweenHoles, [int(distBetHoles[k,1]), int(distBetHoles[k,2])])
                                    closed_mask = closing(mask, square(3))
                                    boundariesBetweenHoles = boundariesBetweenHoles + closed_mask.astype(np.uint8)
                        rBound, cBound = boundariesBetweenHoles.shape
                        # ----- the boundaries may be too thin to separate the objects due to 4- 8- connectivity
                        # ----- thicken the boundary
                        boundaryThick = expandu(reduceu(boundariesBetweenHoles == 0, numReductions = None, reducein3D = None) > 0, numExpansions=2) # modified after bug
                        # [SplittedObjects, numSplitedObjects] =  bwlabel((1-boundaryThick(1:rBound,1:cBound)).*stat.image);
                        #
                        #print("boundaryThick shape:", boundaryThick.shape)
                        #print("stat.image shape:", stat.image.shape)
                        #
                        #prod_temp = stat.image[0:rBound-1, 0:cBound] # modified by me 
                        new_img = (1 - boundaryThick[0:rBound, 0:cBound]).astype(np.uint8) * stat.image # modified
                        SplittedObjects, numSplitedObjects = label(new_img, return_num=True)
                        areaSplitedObjects = [prop.area for prop in regionprops(SplittedObjects)]
                        indexToObjects_inner = np.where(np.array(areaSplitedObjects) >= minAreaObject)[0]
                        newObject = np.isin(SplittedObjects, indexToObjects_inner + 1).astype(np.uint8)
                        #print("newObject shape:", newObject.shape)
                        splittedCells[minRow:maxRow, minCol:maxCol] = newObject + splittedCells[minRow:maxRow, minCol:maxCol]
                        # splittedCells[minRow:maxRow, minCol:maxCol] = SplittedObjects;
                        # sC = sC + 1;
                    else:
                        # ------- holes are close to each other, leave as it is
                        NosplittedCells[minRow:maxRow, minCol:maxCol] = stat.image + NosplittedCells[minRow:maxRow, minCol:maxCol]
                else:
                    # ------- several small holes, leave as it is
                    NosplittedCells[minRow:maxRow, minCol:maxCol] = stat.image + NosplittedCells[minRow:maxRow, minCol:maxCol]
                    # nSC = nSC + 1;
            else:
                # ------- just one hole, leave as it is
                NosplittedCells[minRow:maxRow, minCol:maxCol] = stat.image + NosplittedCells[minRow:maxRow, minCol:maxCol]
                # nSC = nSC + 1;
        else:
            # -------- no holes, try to link with other neighbours in case is was a splitted cell
            NoHoleCells[minRow:maxRow, minCol:maxCol] = NoHoleCells[minRow:maxRow, minCol:maxCol] + stat.image
            # nHC = nHC + 1;
    return NosplittedCells, splittedCells, NoHoleCells