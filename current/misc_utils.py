import numpy as np

def fixup_scipy_ndimage_result(whatever_it_returned):
    """Convert a result from scipy.ndimage to a numpy array
    
    scipy.ndimage has the annoying habit of returning a single, bare
    value instead of an array if the indexes passed in are of length 1.
    For instance:
    scind.maximum(image, labels, [1]) returns a float
    but
    scind.maximum(image, labels, [1,2]) returns a list
    """
    if getattr(whatever_it_returned, "__getitem__", False):
        return np.array(whatever_it_returned)
    else:
        return np.array([whatever_it_returned])

def strel_disk(radius):
    """Create a disk structuring element for morphological operations
    
    radius - radius of the disk
    """
    iradius = int(radius)
    x, y = np.mgrid[-iradius : iradius + 1, -iradius : iradius + 1]
    radius2 = radius * radius
    strel = np.zeros(x.shape)
    strel[x * x + y * y <= radius2] = 1
    return strel