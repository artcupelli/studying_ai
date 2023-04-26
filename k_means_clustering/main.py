import math

def euclidian_distance(x1, x2):
    if(len(x1) == 0 or len(x2) == 0):
        raise Exception("Both vectors must have at least 01 dimention")

    if(len(x1) != len(x2)):
        raise Exception("Vectors have different dimensions")

    somatory = 0
    dimension = 0
    num_all_dimensions = len(x1)

    while dimension < num_all_dimensions:
        somatory += math.pow((x1[dimension] - x2[dimension]), 2)
        dimension += 1
    
    return math.sqrt(somatory)
