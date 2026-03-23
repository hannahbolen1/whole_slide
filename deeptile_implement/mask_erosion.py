import numpy as np
import matplotlib.pyplot as plt

import scipy.ndimage as ndi
from skimage.filters import gaussian
from skimage.morphology import (
    binary_closing, binary_dilation, binary_erosion,
    remove_small_objects, remove_small_holes, disk
)
from skimage.measure import label, regionprops
from skimage.transform import resize
import skimage as ski

def detect_coverslip_from_rim(
    img,
    downscale=32,
):
    """
    Detect coverslip interior by finding the bright rim and filling it.

    img: 2D dask array or numpy array
    returns:
        full_mask      boolean mask at full resolution
        thumb          low-res thumbnail used for detection
        rim_mask       thresholded rim mask on thumbnail
        filled_mask    filled coverslip mask on thumbnail
    """

    h, w = img.shape

    edges = ski.feature.canny(img, sigma=1, low_threshold=5)
    edges = ski.morphology.dilation(edges, footprint=np.ones((10, 10)))
    mask = edges>ski.filters.threshold_otsu(edges)
    mask = ndi.binary_fill_holes(ski.util.invert(mask))
    labels = ski.measure.label(mask)

    # keep largest object only
    props = ski.measure.regionprops(labels)
    if not props:
        raise ValueError("No coverslip region detected from rim.")
    largest = max(props, key=lambda r: r.area).label
    filled = labels == largest

    # erode inward to remove bright edge region
    erosion_small = max(1, erosion_px_fullres // downscale)
    filled = binary_erosion(filled, disk(erosion_small))

    # back to full resolution
    full_mask = resize(
        filled.astype(float),
        (h, w),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ) > 0.5

    return img, mask, filled
