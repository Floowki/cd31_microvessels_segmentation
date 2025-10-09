
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
    
    
    if rows is None or cols is None:
        rows, cols = largeEndoCells.shape

    if statsObjects is None:

        statsObjects = regionprops(largeEndoCells)
    
    # Loop over the large cells to determine the equivalent ellipse 
    # case1 : several holes -- need to be split into several cells 
    # case2 : can be combined into a single new cell that was split before
    splittedCells = np.zeros((rows, cols), dtype=np.uint8)
    NosplittedCells = np.zeros((rows, cols), dtype=np.uint8)
    NoHoleCells = np.zeros((rows, cols), dtype=np.uint8)
    minAreaObject = 25
    numLargeCells = len(statsObjects)

    for counterObjs in range(numLargeCells):
        # bounding box indices
        stat = statsObjects[counterObjs]
        # Use ceil/floor as in MATLAB
        minRow = max(0, int(np.ceil(stat.bbox[0])))
        minCol = max(0, int(np.ceil(stat.bbox[1])))
        maxRow = min(rows, int(np.floor(stat.bbox[2])))
        maxCol = min(cols, int(np.floor(stat.bbox[3])))

        # know whether there are holes: euler number 
        if stat.euler_number < 0:
            #  areas different if at least one hole 
            # label difference between image and the filled image, i.e. the HOLES
            diff_image = stat.filled_image ^ stat.image
            holesInData, numHolesInData = label(diff_image, return_num=True)
            if numHolesInData > 1:
                # more than one hole: linked objects to be splitted
                areaOfHoles = regionprops(holesInData)
                # remove holes smaller than 17 pixels
                indexToObjects = np.where(np.array([prop.area for prop in areaOfHoles]) >= minAreaObject)[0]
                # create binary image with big holes
                bigHolesOnly = np.isin(holesInData, indexToObjects + 1)
                if len(indexToObjects) > 1:
                  
                    distToHoles, distBetHoles = regionGrowing(bigHolesOnly, 1 - stat.filled_image)
                   
                    if np.any(distBetHoles[:,0] > 3):
                        # not splitting between very close holes
                        distToHoles[distToHoles == 0] = np.max(distToHoles)
                        boundariesBetweenHoles = watershed(distToHoles)
                        # close regions where distance between holes is less than 10
                        if np.any(distBetHoles[:,0] < 10):
                            for k in range(distBetHoles.shape[0]):
                                if (distBetHoles[k,0] < 10) and (np.sum(distBetHoles[k,3:5]) < 199):
                                    mask = np.isin(boundariesBetweenHoles, [int(distBetHoles[k,1]), int(distBetHoles[k,2])])
                                    closed_mask = closing(mask, square(3))
                                    boundariesBetweenHoles = boundariesBetweenHoles + closed_mask.astype(np.uint8)
                        rBound, cBound = boundariesBetweenHoles.shape
                        # thicken the boundary that bìmay be too thin to separate objects under 4- 8- connectivity
                        boundaryThick = expandu(reduceu(boundariesBetweenHoles == 0, numReductions = None, reducein3D = None) > 0, numExpansions=2) # modified after bug
        
                        new_img = (1 - boundaryThick[0:rBound, 0:cBound]).astype(np.uint8) * stat.image 
                        SplittedObjects, numSplitedObjects = label(new_img, return_num=True)
                        areaSplitedObjects = [prop.area for prop in regionprops(SplittedObjects)]
                        indexToObjects_inner = np.where(np.array(areaSplitedObjects) >= minAreaObject)[0]
                        newObject = np.isin(SplittedObjects, indexToObjects_inner + 1).astype(np.uint8)
                        splittedCells[minRow:maxRow, minCol:maxCol] = newObject + splittedCells[minRow:maxRow, minCol:maxCol]
                    else:
                        # holes close to each other: leave as it is
                        NosplittedCells[minRow:maxRow, minCol:maxCol] = stat.image + NosplittedCells[minRow:maxRow, minCol:maxCol]
                else:
                    # several small holes: leave as it is
                    NosplittedCells[minRow:maxRow, minCol:maxCol] = stat.image + NosplittedCells[minRow:maxRow, minCol:maxCol]
            else:
                # one hole: leave as it is
                NosplittedCells[minRow:maxRow, minCol:maxCol] = stat.image + NosplittedCells[minRow:maxRow, minCol:maxCol]
        else:
            # no holes, try to link with other neighbours 
            NoHoleCells[minRow:maxRow, minCol:maxCol] = NoHoleCells[minRow:maxRow, minCol:maxCol] + stat.image
    return NosplittedCells, splittedCells, NoHoleCells