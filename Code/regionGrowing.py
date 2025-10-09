
import numpy as np
from scipy.ndimage import label
import math
from skimage.morphology import dilation

def regionGrowing(initialSeeds,backgroundMask):
    #| Args : 
    #|   # initialSeeds : labeled image representing distinct regions starting point for region growing
    #|   # backgroundMask : specifies areas that should be excluded from the region growing process
    
    #| Outputs :
    #|   # dataOut : contains the result of the region growing process
    #|   # distBetweenHole : each row represents [distance, Hole1, Hole2, SizeHole1, SizeHole2]
    
    rows, cols = initialSeeds.shape

    if backgroundMask is None : 
        backgroundMask = np.zeros(rows, cols)
        
    
    if isinstance(initialSeeds, np.ndarray) and initialSeeds.dtype == bool:
        initialSeeds = label(initialSeeds)
    
    numObjects = np.max(initialSeeds) 

    if numObjects > 1:
        if numObjects > 100:
            try:
                dataOut2 = np.zeros((rows, cols, 100))
            except:
                qqq = 1
            numIterations = int(math.ceil(numObjects / 100))
            dataOut3 = np.zeros((rows, cols, numIterations))
            for countIteration in range(1, numIterations + 1):
                print('numiterations:' + str(countIteration))
                counterLowLevel = 100 * (countIteration - 1)
                for counterObjs in range(counterLowLevel + 1, min(100, numObjects - counterLowLevel) + 1):
                    # Recursively call regionGrowing for each object
                    dataOut2[:, :, counterObjs - 1] = regionGrowing((initialSeeds == counterObjs).astype(initialSeeds.dtype), backgroundMask)[0]
                if True:  # Equivalent to: if nargout>1
                    distBetweenHoles = np.empty((0, 5))
                    counterLowLevel = 100 * (countIteration - 1)
                    for counterHoles in range(counterLowLevel + 1, min(100, numObjects - counterLowLevel)):
                        for counterHoles2 in range(counterHoles + 1, min(100, numObjects - counterLowLevel) + 1):
                            tempDist = dataOut2[:, :, counterHoles - 1] * ((initialSeeds == counterHoles2).astype(initialSeeds.dtype))
                            try:
                                indices = np.nonzero(tempDist)
                                if indices[0].size > 0:
                                    minVal = np.min(tempDist[indices])
                                else:
                                    raise ValueError
                                newRow = [minVal, counterHoles, counterHoles2,
                                          np.sum(initialSeeds == counterHoles),
                                          np.sum(initialSeeds == counterHoles2)]
                                distBetweenHoles = np.vstack((distBetweenHoles, newRow))
                            except:
                                ttt = 1
                dataOut3[:, :, countIteration - 1] = np.min(dataOut2, axis=2)
            dataOut = np.min(dataOut3, axis=2)
        else:
            dataOut = np.zeros((rows, cols, numObjects))
            for counterObjs in range(1, numObjects + 1):
                dataOut[:, :, counterObjs - 1] = regionGrowing((initialSeeds == counterObjs).astype(initialSeeds.dtype), backgroundMask)[0]
            if True:  # Equivalent to: if nargout>1
                distBetweenHoles = np.empty((0, 5))
                for counterHoles in range(1, numObjects):
                    for counterHoles2 in range(counterHoles + 1, numObjects + 1):
                        tempDist = dataOut[:, :, counterHoles - 1] * ((initialSeeds == counterHoles2).astype(initialSeeds.dtype))
                        try:
                            indices = np.nonzero(tempDist)
                            if indices[0].size > 0:
                                minVal = np.min(tempDist[indices])
                            else:
                                raise ValueError
                            newRow = [minVal, counterHoles, counterHoles2,
                                      np.sum(initialSeeds == counterHoles),
                                      np.sum(initialSeeds == counterHoles2)]
                            distBetweenHoles = np.vstack((distBetweenHoles, newRow))
                        except:
                            ttt = 1
            dataOut = np.min(dataOut, axis=2)
    else:
        fourConnectedKernel = np.array([[0, 1, 0],
                                        [1, 1, 1],
                                        [0, 1, 0]])
        outerBoundary = -initialSeeds + dilation(initialSeeds, footprint=fourConnectedKernel)
        seedsProps = np.flatnonzero(initialSeeds)
        outerB_Props = np.flatnonzero(outerBoundary)
        #seedsProps = regionprops(initialSeeds, 'area','pixellist','pixelidxlist')
        #outerB_Props = regionprops(outerBoundary,'area','pixellist','pixelidxlist')
        #------- define outer Boundary
        outerBLocation = outerB_Props.copy()
        seedLocation = np.flatnonzero(initialSeeds)
    
        k = 0
        while outerBLocation.size != 0:
            k = k + 1
            #----- some pixels are outside the image
            outerBLocation = outerBLocation[outerBLocation >= 0]
            outerBLocation = outerBLocation[outerBLocation < (rows * cols)]
            # First, discard all pixels that belong to the background Mask
            try:
                outerBLocation = outerBLocation[backgroundMask.flat[outerBLocation] <= 0]
            except:
                pass
            if outerBLocation.size != 0:
                #------- growing process until it stops
                # update regions include new pixel on initialSeeds,  join the pixels that is closest to the current Seed
                initialSeeds.flat[outerBLocation] = k
#                # figure(2);surfdat(initialSeeds);axis off;drawnow;
                #jj=length(FRA_1)+1
                #FRA_1(jj)=getframe;
                seedsProps = np.concatenate((seedsProps, outerBLocation))
                #----- add pixels added "outerBlocation" into the seedLocation
                seedLocation = np.concatenate((seedLocation, outerBLocation))
                #----- find new neighbours, up, down, left and right
                newNeighboursUp = outerBLocation - 1
                newNeighboursDown = outerBLocation + 1
                newNeighboursRight = outerBLocation + rows
                newNeighboursLeft = outerBLocation - rows
                #----- find new neighbours, NE, NW, SE and SW
                newNeighboursNE = newNeighboursRight - 1
                newNeighboursSE = newNeighboursRight + 1
                newNeighboursNW = newNeighboursLeft - 1
                newNeighboursSW = newNeighboursLeft + 1

                newNeighbours4Conn = np.concatenate((newNeighboursRight, newNeighboursLeft, newNeighboursUp, newNeighboursDown))
                newNeighbours8Conn = np.concatenate((newNeighboursNE, newNeighboursSE, newNeighboursNW, newNeighboursSW))
                newNeighbours = np.unique(np.concatenate((newNeighbours4Conn, newNeighbours8Conn)))
                newNeighbours = newNeighbours[newNeighbours >= 0]
                newNeighbours = newNeighbours[newNeighbours < (rows * cols)]
                newNeighbours = newNeighbours[((newNeighbours + 1) % rows != 0)]
                newNeighbours = newNeighbours[((newNeighbours + 1) % rows != 1)]
                #----- add new neighbours into the outerBLocation
                outerBLocation = np.unique(np.concatenate((outerBLocation, newNeighbours)))
                #----- remove from OuterBLocation all the pixels that belong to seedLocation
                outerBLocation = np.array([val for val in outerBLocation if val not in seedLocation])
        dataOut = initialSeeds

    distBetweenHole = locals().get('distBetweenHoles', None)
    
    return dataOut, distBetweenHole