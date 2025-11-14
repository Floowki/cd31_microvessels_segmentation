

import numpy as np
from skimage.measure import regionprops
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
import math
from scipy.ndimage import binary_fill_holes
from skimage.morphology import skeletonize
import BranchPoints

def JoinObjects(BW1,backgroundMask = None):
    #| Args : 
    #|   # BW1 : binary image identifying objects
    #|   # backgroundMask : binary mask identifying the background 
    
    #| Outputs :
    #|   # BW2 : binary mask with joined objects 

    rows, cols = BW1.shape
    if backgroundMask is None:
        backgroundMask = np.zeros(BW1.shape, dtype=BW1.dtype)
    
    # Label input data to process objects by pairs
    BW2 = label(BW1, connectivity=2)
    numInitialObjs = BW2.max()
    
    if numInitialObjs > 1:
        
        props = regionprops(BW2)
       
        statsObjects0 = []
        for prop in props:

            bbox = [prop.bbox[1], prop.bbox[0], prop.bbox[3] - prop.bbox[1], prop.bbox[2] - prop.bbox[0]]
            statsObjects0.append({
                'Area': prop.area,
                'MinorAxisLength': prop.minor_axis_length if hasattr(prop, 'minor_axis_length') else 1,
                'MajorAxisLength': prop.major_axis_length if hasattr(prop, 'major_axis_length') else 1,
                'EulerNumber': prop.euler_number,
                'BoundingBox': bbox
            })
    
        for countObjs in range(1, numInitialObjs + 1):
            # Process of joining close objects, analyse each against its closest object
            currentObject = (BW2 == countObjs)
            # the object could be empty if joined to another previously
            if np.any(currentObject):
                # generate distance map to find distance to other objects
                distanceMap = distance_transform_edt(~currentObject)
                # exclude current object
                BWT = np.logical_and(BW1, BW2 != countObjs)
                # superimpose all objects except current in the distance map --> minimum value
                if np.any(BWT):
                    distToClosest = np.min(distanceMap[BWT])
                    
                    mask_points = BW2[np.logical_and(distanceMap == distToClosest, BWT)]
                    if mask_points.size > 0:
                        numClosestObj = int(np.max(mask_points))
                        
                        # create a separate object to which comparisons will be made
                        closestObject = (BW2 == numClosestObj)
                        
                        # calculate the maximum distance accepted between objects
                        dist1 = 1 + math.log(max(1, statsObjects0[numClosestObj - 1]['MajorAxisLength']))
                        dist2 = 1 + math.log(max(1, statsObjects0[countObjs - 1]['MajorAxisLength']))
                        maxDistanceAccepted = dist1 + dist2
                        
                        if (math.floor(distToClosest) <= math.ceil(maxDistanceAccepted)):
                            # join regions, first is turned into second to allow second to be analysed as well
                            distanceMap2 = distance_transform_edt(~closestObject) + distanceMap
                            # Ccheck the region in between is NOT background region slightly enlarged
                            regionOfConnection = (distanceMap2 <= math.ceil(2 + distToClosest))
                            ratioOfBackground = np.mean(backgroundMask[regionOfConnection])
                            
                            if (ratioOfBackground < 0.6):
                                # if the region in between is NOT predominantly background then check shape
                                if (statsObjects0[numClosestObj - 1]['EulerNumber'] + statsObjects0[countObjs - 1]['EulerNumber']) > 1:
                                    # Accept a merge only between objects that do not contain holes
                                    regionOfConnection = binary_fill_holes(distanceMap2 <= round(1 + distToClosest + 1e-5))
                                    
                                    currentBBox = statsObjects0[countObjs - 1]['BoundingBox']
                                    minCol = max(0, -2 + int(math.floor(currentBBox[0])))
                                    maxCol = min(cols, minCol + 4 + int(currentBBox[2]))
                                    minRow = max(0, -2 + int(math.floor(currentBBox[1])))
                                    maxRow = min(rows, minRow + 4 + int(currentBBox[3]))
                                    
                                    closestBBox = statsObjects0[numClosestObj - 1]['BoundingBox']
                                    minColClose = max(0, -2 + int(math.floor(closestBBox[0])))
                                    maxColClose = min(cols, minColClose + 4 + int(closestBBox[2]))
                                    minRowClose = max(0, -2 + int(math.floor(closestBBox[1])))
                                    maxRowClose = min(rows, minRowClose + 4 + int(closestBBox[3]))
                                    
                                    commonRegion = np.logical_or(regionOfConnection, np.logical_or(closestObject, currentObject))
                                    minCommonRow = min(minRow, minRowClose)
                                    maxCommonRow = max(maxRow, maxRowClose)
                                    minCommonCol = min(minCol, minColClose)
                                    maxCommonCol = max(maxCol, maxColClose)
                                    
                                    commonRegionRed = commonRegion[minCommonRow:maxCommonRow, minCommonCol:maxCommonCol]
                                    
                                    # calculate skeletons and branch points
                                    skelCurrent = skeletonize(currentObject[minRow:maxRow, minCol:maxCol])
                                    BPoints1, BPoints12, numPoints1 = BranchPoints.BranchPoints(skelCurrent)
                                    
                                    skelJoint = skeletonize(commonRegionRed)
                                    BPoints3, BPoints32, numPoints3 = BranchPoints.BranchPoints(skelJoint)
                                    
                                    skelClosest = skeletonize(closestObject[minRowClose:maxRowClose, minColClose:maxColClose])
                                    BPoints2, BPoints22, numPoints2 = BranchPoints.BranchPoints(skelClosest)
                                    
                                    if (numPoints1 + numPoints2) >= (numPoints3):
                                        # if there are less branching points in the joined region, then join
                                        BW2[currentObject] = numClosestObj
                                        # join region in between
                                        BW2[regionOfConnection] = numClosestObj
                                        
                                        labeledClosest = label(closestObject, connectivity=2)
                                        newProps = regionprops(labeledClosest)
                                        if newProps:
                                            newAxis = newProps[0].major_axis_length
                                            statsObjects0[numClosestObj - 1]['MinorAxisLength'] = newAxis
    
        BW2 = (BW2 > 0)
        BW2 = binary_fill_holes(BW2)
    else:
        BW2 = BW1
    return BW2