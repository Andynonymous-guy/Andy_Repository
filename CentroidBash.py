import numpy as np


square = [([0, 0], [1, 0], [0, 1]), ([1, 0], [0, 1], [1, 1])]

##Use this to find the centroid of a triangle D in any coordinate space by taking the simple arithmetic mean. Outputs floats not int!
def TriangleCentroidFinder (D):
    return (((D[0][0] + D[1][0] + D[2][0])/3), (D[0][1] + D[1][1] + D[2][1])/3)

##Use this to find the area of a polygon D in any coordinate space by using the shoelance formula. 
def AreaFinder(D):
    area = 0
    for i in range(len(D)):
        x1, y1 = D[i]
        x2, y2 = D[(i + 1) % len(D)]
        area += (x1 * y2 - x2 * y1)
    return abs(area / 2)

##Use this to find the centroid of any polygon given its triangulation list L, Output in the same coordinate system. 
def CentroidCollector(L):
    Finalx = 0
    Finaly = 0
    FinalArea = 0

    for i in L:
        x, y = TriangleCentroidFinder(i)
        IArea = AreaFinder(i)
        Finalx += (x * IArea)
        Finaly += (y * IArea)
        FinalArea += IArea

    return(Finalx/FinalArea, Finaly/FinalArea)


print(CentroidCollector(square))