import math

## Potential Use for Cleaning up Polygons
## Chase angle ABC using the dot product of vectors. 
def chase(A, B, C, format = None):
    vu = ((B[0] - A[0]), (B[1]-A[1]))
    vv = ((C[0] - B[0]), (C[1]-B[1]))
    angle = int(math.degrees(math.acos((vu[0]*vv[0] + vu[1]*vv[1])/((math.sqrt(vu[0]**2 + vu[1]**2))*(math.sqrt(vv[0]**2 + vv[1]**2))))))
    if format == 'FirstQuad':
        match angle:
            case _ if angle < 90:
                return('Acute')
            case _ if angle == 90:
                return('Right')
            case _ if angle > 90:
                return('Obtuse')
    elif format == 'Concavity':
        match angle:
                case _ if angle < 180:
                    return('Convex')
                case _ if angle == 180:
                    return ('Linear')
                case _ if angle > 180: 
                    return ('Concave')
    return(angle)

## Takes out three collinear sides by using the angle sweeper.  
def cleaner(P):
    i = 0
    while i <= len(P) - 1:
        if chase(P[i-1], P[i], P[(i+1) % len(P)]) != 180:
            P.append(P[i])
        i = i + 1
    return(P)
