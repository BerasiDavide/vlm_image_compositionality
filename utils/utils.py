import numpy as np
import torch
import random

def chunks(l, n):
    '''
    Yield successive n-sized chunks from l.
    '''
    for i in range(0, len(l), n):
        yield l[i:i + n]


def set_seed(seed):
    """function sets the seed value
    Args:
        seed (int): seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # if you are suing GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_fibers(array, axis):
    '''
    Returns a list containing the fibers of array along the given axis.
    '''
    proj_shape = list(array.shape)
    proj_shape.pop(axis)
    fibers = []
    for fiber_index in np.ndindex(tuple(proj_shape)):
        fiber_index = list(fiber_index)
        fiber_index.insert(axis, slice(None))

        fibers.append(array[tuple(fiber_index)])
    return fibers


def avg_distance(vectors, sample_size=10000):
    '''
    Approximates average square distance between vectors.
    '''
    n = vectors.shape[0]
    i = np.random.randint(0, n, (sample_size,))
    j = np.random.randint(0, n, (sample_size,))
    I, J = vectors[i[i!=j]], vectors[j[i!=j]]
    
    squared_distances = np.sum((I - J) ** 2, axis=1)

    avg = squared_distances.mean()
    std = squared_distances.std()
    return avg, std