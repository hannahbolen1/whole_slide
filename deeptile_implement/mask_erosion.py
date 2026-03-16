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
    erosion_px_fullres=200,
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

    # # cheap thumbnail from dask / large array
    # if hasattr(img, "compute"):
    #     thumb = img[::downscale, ::downscale].compute()
    # else:
    #     thumb = img[::downscale, ::downscale]

    # thumb = np.asarray(img, dtype=np.float32)
    # thumb = np.nan_to_num(thumb, nan=0.0, posinf=0.0, neginf=0.0)

    # # normalize to 0-1
    # p1, p99 = np.percentile(thumb, [0, 100])
    # if p99 <= p1:
    #     raise ValueError("Thumbnail has no usable intensity range.")
    # thumb = np.clip((thumb - p1) / (p99 - p1), 0, 1)

    # # # light blur to stabilize rim
    # # blur = gaussian(ski.morphology.dilation(thumb, footprint=ski.morphology.disk(10)), sigma=blur_sigma)

    # # # detect only the brightest pixels = rim
    # # thr = ski.filters.threshold_otsu(blur)
    # # rim_mask_raw = blur >= thr
    # canny = ski.feature.canny(thumb, sigma=1, low_threshold=0.05)
    # canny_dilation = ski.morphology.dilation(canny, footprint=np.ones((10, 10)))
    # rim_mask_raw = canny_dilation>ski.filters.threshold_otsu(canny_dilation)
    # # connect broken rim fragments
    # rim_mask = binary_closing(rim_mask_raw, disk(close_radius))
    # rim_mask = binary_dilation(rim_mask, disk(dilate_radius))
    # rim_mask = remove_small_objects(rim_mask, min_size=500)

    # # fill enclosed region
    # filled = binary_fill_holes(rim_mask)
    # filled = remove_small_holes(filled, area_threshold=5000)

    edges = ski.feature.canny(img, sigma=1, low_threshold=5)
    edges = ski.morphology.dilation(edges, footprint=np.ones((10, 10)))
    mask = edges>ski.filters.threshold_otsu(edges)
    mask = ndi.binary_fill_holes(ski.util.invert(mask))
    labels = ski.measure.label(mask)

    # keep largest object only
    props = regionprops(labels)
    if not props:
        raise ValueError("No coverslip region detected from rim.")
    largest = max(props, key=lambda r: r.area).label
    filled = labels == largest

    # erode inward to remove bright edge region
    erosion_small = max(1, erosion_px_fullres // downscale)
    filled = binary_erosion(filled, disk(erosion_small))

    # # back to full resolution
    # full_mask = resize(
    #     filled.astype(float),
    #     (h, w),
    #     order=0,
    #     preserve_range=True,
    #     anti_aliasing=False,
    # ) > 0.5

    return img, mask, filled



# # import numpy as np
# # import tifffile as tiff

# # from skimage.transform import resize
# # from skimage.filters import gaussian, threshold_otsu
# # from skimage.morphology import (
# #     binary_closing, binary_opening, binary_erosion,
# #     remove_small_holes, remove_small_objects, disk
# # )
# # from skimage.measure import label, regionprops
# # from scipy.ndimage import binary_fill_holes


# # def detect_coverslip_mask(
# #     img,
# #     downscale=16,
# #     blur_sigma=2,
# #     min_size_fraction=0.05,
# #     erosion_px_fullres=150
# # ):
# #     """
# #     img: 2D numpy array
# #     returns: boolean full-resolution mask for coverslip interior
# #     """

# #     h, w = img.shape

# #     # downsample
# #     small = resize(
# #         img,
# #         (h // downscale, w // downscale),
# #         order=1,
# #         preserve_range=True,
# #         anti_aliasing=True
# #     ).astype(np.float16)

# #     # normalize
# #     small = small - small.min()
# #     if small.max() > 0:
# #         small = small / small.max()

# #     # smooth
# #     small_blur = gaussian(small, sigma=blur_sigma)

# #     # threshold
# #     thr = threshold_otsu(small_blur)
# #     mask = small_blur > thr

# #     # morphology cleanup
# #     mask = binary_closing(mask, disk(5))
# #     mask = binary_opening(mask, disk(3))
# #     mask = remove_small_holes(mask, area_threshold=5000)
# #     mask = remove_small_objects(mask, min_size=int(mask.size * min_size_fraction))

# #     # keep largest connected component
# #     lab = label(mask)
# #     props = regionprops(lab)

# #     if len(props) == 0:
# #         raise ValueError("No coverslip region detected.")

# #     largest = max(props, key=lambda r: r.area).label
# #     mask = (lab == largest)

# #     # fill holes
# #     mask = binary_fill_holes(mask)

# #     # erode inward to remove rim
# #     erosion_small = max(1, erosion_px_fullres // downscale)
# #     mask = binary_erosion(mask, disk(erosion_small))

# #     # resize back to full res
# #     full_mask = resize(
# #         mask.astype(float),
# #         img.shape,
# #         order=0,
# #         preserve_range=True,
# #         anti_aliasing=False
# #     ) > 0.5

# #     return full_mask


# import numpy as np

# from skimage.filters import gaussian, threshold_otsu
# from skimage.morphology import (
#     binary_closing, binary_opening, binary_erosion,
#     remove_small_holes, remove_small_objects, disk
# )
# from skimage.measure import label, regionprops
# from scipy.ndimage import binary_fill_holes
# from skimage.transform import resize


# def detect_coverslip_mask_dask(
#     img,
#     downscale=32,
#     blur_sigma=6,
#     min_size_fraction=0.05,
#     erosion_px_fullres=150
# ):
#     """
#     img: 2D dask array or numpy array
#     returns: full-resolution boolean mask
#     """

#     h, w = img.shape

#     # make a cheap low-res thumbnail directly by striding, then compute
#     small = img[::downscale, ::downscale].compute().astype(np.float16)
#     #small = np.asarray(small, dtype=np.float32)

#     # normalize
#     small_min = small.min()
#     small_max = small.max()
#     if small_max <= small_min:
#         raise ValueError("Thumbnail is constant; cannot threshold coverslip.")
#     small = (small - small_min) / (small_max - small_min)

#     # blur so the coverslip becomes one broad object
#     small_blur = gaussian(small, sigma=blur_sigma)
#     #small_blur = np.nan_to_num(small_blur, nan=0.0, posinf=0.0, neginf=0.0)

#     # threshold
#     thr = threshold_otsu(small_blur)
#     mask = small_blur > thr

#     # cleanup
#     mask = binary_closing(mask, disk(5))
#     mask = binary_opening(mask, disk(3))
#     mask = remove_small_holes(mask, area_threshold=5000)
#     mask = remove_small_objects(mask, min_size=max(1000, int(mask.size * min_size_fraction)))

#     # keep largest object
#     lab = label(mask)
#     props = regionprops(lab)
#     if not props:
#         raise ValueError("No coverslip region detected.")
#     largest = max(props, key=lambda r: r.area).label
#     mask = (lab == largest)

#     # fill interior
#     mask = binary_fill_holes(mask)

#     # erode inward to remove bright rim
#     erosion_small = max(1, erosion_px_fullres // downscale)
#     mask = binary_erosion(mask, disk(erosion_small))

#     # resize back to full image size
#     full_mask = resize(
#         mask.astype(float),
#         (h, w),
#         order=0,
#         preserve_range=True,
#         anti_aliasing=False
#     ) > 0.5

#     return full_mask