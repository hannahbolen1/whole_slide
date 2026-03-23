from deeptile import lift
import skimage as ski
from scipy import ndimage as ndi
import numpy as np
from deeptile.core.data import Output
from deeptile.core.lift import lift
from deeptile.core.utils import compute_dask
from functools import partial

def coverslip_mask(
    img,
    ds=32,
):
    """
    Mask area outside coverslip.
    Args:
        img: 2D dask array or numpy array
        downscale (int, optional): Downsample factor for mask processing. Default is 32.
    Returns:
        full_mask (Bool): boolean mask at full resolution
    """
    if ds:
        img_ds = img[::ds,::ds]
    else:
        img_ds = img

    edges = ski.feature.canny(img_ds, sigma=1, low_threshold=5)
    edges = ski.morphology.dilation(edges, footprint=np.ones((10, 10)))
    mask = edges>ski.filters.threshold_otsu(edges)
    mask = ndi.binary_fill_holes(ski.util.invert(mask))
    labels = ski.measure.label(mask)

    # keep largest object only
    props = ski.measure.regionprops(labels)
    if not props:
        raise ValueError("No coverslip region detected from rim.")
    filled = labels == max(props, key=lambda r: r.area).label
    
    filled = ski.filters.gaussian(ski.morphology.isotropic_erosion(filled, radius=20), sigma=3, preserve_range=True)>0
    # # erode inward to remove bright edge region
    # if ds:
    #     erosion_small = max(1, 150 // ds)
    # else:
    #     erosion_small = 150
    # filled = ski.morphology.binary_erosion(filled, ski.morphology.disk(erosion_small))

    # back to full resolution
    if ds:
        # full_mask = ski.transform.rescale(filled, ds)
        full_mask = ski.transform.resize(
            filled.astype(float),
            (img.shape[0], img.shape[1]),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ) > 0.5
    else:
        full_mask = filled
    
    return full_mask

def cellpose_segmentation(model_parameters, eval_parameters, output_format='masks'):

    """Generate lifted function for the Cellpose segmentation algorithm.

    Parameters
    ----------
    model_parameters : dict
        Dictionary of model parameters.
    eval_parameters : dict
        Dictionary of evaluation parameters.
    output_format : str, optional
        Format of the output. Supported formats are 'masks' and 'polygons'. Default is 'masks'.

    Returns
    -------
    func_segment : Callable
        Lifted function for the Cellpose segmentation algorithm.
    """

    from cellpose.models import CellposeModel
    from cellpose.io import logger_setup
    logger_setup()

    model = CellposeModel(**model_parameters)

    @lift
    def _func_segment(tile, index, tile_index, stitch_index, tiling):

        tile = compute_dask(tile)
        if tile.max() > 1000:
            mask = model.eval(tile, **eval_parameters)[0]
            return mask
        else:
            return np.zeros(tile.shape, dtype="uint16")

    def func_segment(tiles):

        return _func_segment(tiles, tiles.index_iterator, tiles.tile_indices_iterator, tiles.stitch_indices_iterator,
                             tiles.profile.tiling)

    return func_segment

@lift
def segment_foci_tiled(tiles, **kwargs):
    """Detect foci in the given image using a white tophat filter and other processing steps.
    
    Args:
        data (numpy.ndarray): Input image data.
        radius (int, optional): Radius of the disk used in the white tophat filter. Default is 3.
        threshold (float, optional): Threshold value for identifying foci in the processed image. Default is 10.
        remove_border_foci (bool, optional): Flag to remove foci touching the image border. Default is False.
        regions (numpy.ndarray, optional): Labeled segmentation mask of nuclei to find foci in. Default is none.
    
    """
    return find_foci(tiles, **kwargs)

# from https://github.com/cheeseman-lab/brieflow/blob/main/workflow/lib/phenotype/extract_phenotype_cp_emulator.py#L361
def find_foci(data, radius=3, threshold=10, min_distance=1, remove_border_foci=False, regions=None):
    """Detect foci in the given image using a white tophat filter and other processing steps.
    
    Args:
        data (numpy.ndarray): Input image data.
        radius (int, optional): Radius of the disk used in the white tophat filter. Default is 3.
        threshold (float, optional): Threshold value for identifying foci in the processed image. Default is 10.
        remove_border_foci (bool, optional): Flag to remove foci touching the image border. Default is False.
        regions (numpy.ndarray, optional): Labeled segmentation mask of nuclei to find foci in. Default is none.

    Returns:
        labeled (numpy.ndarray): Labeled segmentation mask of foci.
    """
    # Apply white tophat filter to highlight foci
    tophat = ski.morphology.white_tophat(
        data, footprint=ski.morphology.disk(radius)
    )

    # Apply Laplacian of Gaussian to the filtered image
    tophat_log = log_ndi(tophat, sigma=radius)

    # Threshold the image to create a binary mask
    mask = tophat_log > threshold
    # Remove small objects from the mask
    mask = ski.morphology.remove_small_objects(mask, min_size=(radius**2))
    # Label connected components in the mask
    labeled = ski.measure.label(mask)
    # Apply watershed algorithm to refine segmentation
    labeled = apply_watershed(labeled, regions=regions, smooth=1, min_distance=min_distance)

    if remove_border_foci:
        # Remove foci touching the border
        labeled = remove_border(labeled, regions)

    return labeled

def log_ndi(data, sigma=1, *args, **kwargs):
    """Apply Laplacian of Gaussian to each image in a stack of shape (..., I, J).

    Args:
        data (numpy.ndarray): Input data.
        sigma (float, optional): Standard deviation of the Gaussian kernel. Default is 1.
        *args: Additional positional arguments passed to scipy.ndimage.filters.gaussian_laplace.
        **kwargs: Additional keyword arguments passed to scipy.ndimage.filters.gaussian_laplace.

    Returns:
        numpy.ndarray: Resulting images after applying Laplacian of Gaussian.
    """
    # Define the Laplacian of Gaussian filter function
    f = ndi.filters.gaussian_laplace

    # Apply the filter to the data and invert the output
    arr_ = -1 * f(data.astype(float), sigma, *args, **kwargs)

    # Clip values to ensure they are within the valid range [0, 65535] and convert back to uint16
    arr_ = np.clip(arr_, 0, 65535) / 65535

    # Suppress precision warning from skimage
    return ski.img_as_uint(arr_)

def apply_watershed(img, regions = None, smooth=4, min_distance=1):
    """Apply the watershed algorithm to the given image to refine segmentation.

    Args:
        img (numpy.ndarray): Input binary image.
        smooth (float, optional): Size of Gaussian kernel used to smooth the distance map. Default is 4.

    Returns:
        result (numpy.ndarray): Labeled image after watershed segmentation.
    """
    # Compute the distance transform of the image
    distance = ndi.distance_transform_edt(img)

    if smooth > 0:
        # Apply Gaussian smoothing to the distance transform
        distance = ski.filters.gaussian(distance, sigma=smooth)

    # Identify local maxima in the distance transform
    local_max_coords = ski.feature.peak_local_max(
        distance, footprint=np.ones((3, 3)), labels=regions, exclude_border=False, min_distance=1 ## if need to adjust -- start with min_distance=1
    )

    # Create a boolean mask for peaks
    local_max = np.zeros_like(distance, dtype=bool)
    local_max[tuple(local_max_coords.T)] = True  # Convert coordinates to a boolean mask

    # Label the local maxima
    markers = ndi.label(local_max)[0]

    # Apply watershed algorithm to the distance transform
    result = ski.segmentation.watershed(-distance, markers, mask=img)

    return result.astype(np.uint16)

def remove_border(labels, mask, dilate=3):
    """Remove labeled regions that touch the border of the given mask.

    Args:
        labels (numpy.ndarray): Labeled image.
        mask (numpy.ndarray): Mask indicating the border regions.
        dilate (int, optional): Number of dilation iterations to apply to the mask. Default is 5.

    Returns:
        labels (numpy.ndarray): Labeled image with border regions removed.
    """
    # Dilate the mask to ensure regions touching the border are included
    mask = ski.segmentation.find_boundaries(mask, mode="outer")
    mask = ski.morphology.binary_dilation(mask, np.ones((dilate, dilate)))
    # Identify labels that need to be removed
    remove = np.unique(labels[mask])

    # Remove the identified labels from the labeled image
    labels = labels.copy()
    labels.flat[np.in1d(labels, remove)] = 0

    return labels

def filter_by_region(labeled, threshold, score=lambda r: r.mean_intensity, intensity_image=None, relabel=True):
    """Apply a filter to label image. The `score` function takes a single region 
    as input and returns a score. 
    If scores are boolean, regions where the score is false are removed.
    Otherwise, the function `threshold` is applied to the list of scores to 
    determine the minimum score at which a region is kept.
    If `relabel` is true, the regions are relabeled starting from 1.
    """
    # make copy of labeled image
    labeled = labeled.copy().astype(int)
    # calculate region properties
    regions = ski.measure.regionprops(labeled, intensity_image=intensity_image)
    # caluclate scores for each region
    scores = np.array([score(r) for r in regions])

    if all([s in (True, False) for s in scores]):
        # identify regions to cut based on boolean scores
        cut = [r.label for r, s in zip(regions, scores) if not s]
    else:
        # determine threshold value for scores
        if callable(threshold):
            t = threshold(scores)
        else:
            t = threshold
        cut = [r.label for r, s in zip(regions, scores) if s < t]

    # remove identified regions from the labeled image
    labeled.flat[np.isin(labeled.flat[:], cut)] = 0
    
    if relabel:
        labeled, _, _ = ski.segmentation.relabel_sequential(labeled)

    return labeled