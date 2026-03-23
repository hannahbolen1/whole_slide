# from https://github.com/cheeseman-lab/brieflow/blob/main/workflow/lib/phenotype/extract_phenotype_cp_multichannel.py#L247
"""Utility functions for working with feature tables."""

from collections import defaultdict
from collections.abc import Iterable

import pandas as pd
import numpy as np
import skimage.measure
import dask_image.ndfilters

def find_foci(data, radius=5, threshold=50, remove_border_foci=False):
    """Detect foci in the given image using a white tophat filter and other processing steps.

    Args:
        data (numpy.ndarray): Input image data.
        radius (int, optional): Radius of the disk used in the white tophat filter. Default is 3.
        threshold (float, optional): Threshold value for identifying foci in the processed image. Default is 10.
        remove_border_foci (bool, optional): Flag to remove foci touching the image border. Default is False.

    Returns:
        labeled (numpy.ndarray): Labeled segmentation mask of foci.
    """
    # Apply white tophat filter to highlight foci
    tophat = skimage.morphology.white_tophat(
        data, footprint=skimage.morphology.disk(radius)
    )

    # Apply Laplacian of Gaussian to the filtered image
    tophat_log = log_ndi(tophat, sigma=radius)

    # Threshold the image to create a binary mask
    mask = tophat_log > threshold

    # Remove small objects from the mask
    mask = skimage.morphology.remove_small_objects(mask, min_size=(radius**2))

    # Label connected components in the mask
    labeled = skimage.measure.label(mask)

    # Apply watershed algorithm to refine segmentation
    labeled = apply_watershed(labeled, smooth=1)

    if remove_border_foci:
        # Remove foci touching the border
        border_mask = data > 0
        labeled = remove_border(labeled, ~border_mask)

    return labeled

def feature_table(data, labels, features, global_features=None):
    """Apply functions in feature dictionary to regions in data specified by integer labels.

    If provided, the global feature dictionary is applied to the full input data and labels.
    Results are combined in a dataframe with one row per label and one column per feature.

    Args:
        data (np.ndarray): Image data.
        labels (np.ndarray): Labeled segmentation mask defining objects to extract features from.
        features (dict): Dictionary of feature names and their corresponding functions.
        global_features (dict, optional): Dictionary of global feature names and their corresponding functions.

    Returns:
        pd.DataFrame: DataFrame containing extracted features with one row per label and one column per feature.
    """
    # Extract regions from the labeled segmentation mask
    regions = regionprops(labels, intensity_image=data)

    # Initialize a defaultdict to store feature values
    results = defaultdict(list)

    # Loop through each region and compute features
    for region in regions:
        for feature, func in features.items():
            # Apply the feature function to the region and append the result to the corresponding feature list
            results[feature].append(fix_uint16(func(region)))

    # If global features are provided, compute them and add them to the results
    if global_features:
        for feature, func in global_features.items():
            # Apply the global feature function to the full input data and labels
            results[feature] = fix_uint16(func(data, labels))

    # Convert the results dictionary to a DataFrame
    return pd.DataFrame(results)


def feature_table_multichannel(data, labels, features, global_features=None):
    """Apply functions in feature dictionary to regions in data specified by integer labels.

    If provided, the global feature dictionary is applied to the full input data and labels.
    Results are combined in a dataframe with one row per label and one column per feature.

    Args:
        data (np.ndarray): Image data.
        labels (np.ndarray): Labeled segmentation mask defining objects to extract features from.
        features (dict): Dictionary of feature names and their corresponding functions.
        global_features (dict, optional): Dictionary of global feature names and their corresponding functions.

    Returns:
        pd.DataFrame: DataFrame containing extracted features with one row per label and one column per feature.
    """
    # Extract regions from the labeled segmentation mask
    regions = regionprops_multichannel(labels, intensity_image=data)

    # Initialize a defaultdict to store feature values
    results = defaultdict(list)

    # Loop through each feature and compute features for each region
    for feature, func in features.items():
        # Check if the result of applying the function to the first region is iterable
        result_0 = func(regions[0])
        if isinstance(result_0, Iterable):
            if len(result_0) == 1:
                # If the result is a single value, apply the function to each region and append the result to the corresponding feature list
                results[feature] = [func(region)[0] for region in regions]
            else:
                # If the result is a sequence, apply the function to each region and append each element of the result to the corresponding feature list
                for result in map(func, regions):
                    for index, value in enumerate(result):
                        results[f"{feature}_{index}"].append(value)
        else:
            # If the result is not iterable, apply the function to each region and append the result to the corresponding feature list
            results[feature] = list(map(func, regions))

    # If global features are provided, compute them and add them to the results
    if global_features:
        for feature, func in global_features.items():
            # Apply the global feature function to the full input data and labels
            results[feature] = func(data, labels)

    # Convert the results dictionary to a DataFrame
    return pd.DataFrame(results)


def regionprops(labeled, intensity_image):
    """Supplement skimage.measure.regionprops with additional field `intensity_image_full` containing multi-dimensional intensity image.

    Args:
        labeled (np.ndarray): Labeled segmentation mask defining objects.
        intensity_image (np.ndarray): Intensity image.

    Returns:
        list: List of region properties objects.
    """
    # If intensity image has more than 2 dimensions, consider only the first channel
    if intensity_image.ndim == 2:
        base_image = intensity_image
    else:
        base_image = intensity_image[..., 0, :, :]

    # Compute region properties using skimage.measure.regionprops
    regions = skimage.measure.regionprops(labeled, intensity_image=base_image)

    # Iterate over regions and add the 'intensity_image_full' attribute
    for region in regions:
        b = region.bbox  # Get bounding box coordinates
        # Extract the corresponding sub-image from the intensity image and assign it to the 'intensity_image_full' attribute
        region.intensity_image_full = intensity_image[..., b[0] : b[2], b[1] : b[3]]

    return regions


def regionprops_multichannel(labeled, intensity_image):
    """Format intensity image axes for compatibility with updated skimage.measure.regionprops that allows multichannel images.

    Some operations are faster than regionprops, others are slower.

    Args:
        labeled (np.ndarray): Labeled segmentation mask defining objects.
        intensity_image (np.ndarray): Multichannel intensity image.

    Returns:
        list: List of region properties objects.
    """
    import skimage.measure

    # If intensity image has only 2 dimensions, consider it as a single-channel image
    if intensity_image.ndim == 2:
        base_image = intensity_image
    else:
        # Move the channel axis to the last position for compatibility with skimage.measure.regionprops
        base_image = np.moveaxis(
            intensity_image,
            range(intensity_image.ndim - 2),
            range(-1, -(intensity_image.ndim - 1), -1),
        )

    # Compute region properties using skimage.measure.regionprops
    regions = skimage.measure.regionprops(labeled, intensity_image=base_image)

    return regions


def fix_uint16(x):
    """Pandas bug converts np.uint16 to np.int16!!!

    Args:
        x (Union[np.uint16, int]): Value to fix.

    Returns:
        Union[int, np.uint16]: Fixed value.
    """
    if isinstance(x, np.uint16):
        return int(x)
    return x

"""Utility functions for extracting features from image data."""

# Basic features added to all feature extractions
features_basic = {
    "area": lambda r: r.area,
    "i": lambda r: r.centroid[0],
    "j": lambda r: r.centroid[1],
    "label": lambda r: r.label,
    "bounds": lambda r: r.bbox,
}


def extract_features(data, labels, wildcards, features=None, multichannel=False):
    """Extract features from the provided image data within labeled segmentation masks.

    Args:
        data (numpy.ndarray): Image data of dimensions (CHANNEL, I, J).
        labels (numpy.ndarray): Labeled segmentation mask defining objects to extract features from.
        wildcards (dict): Metadata to include in the output table, e.g., well, tile, etc.
        features (dict or None): Features to extract and their defining functions. Default is None.
        multichannel (bool): Flag indicating whether the data has multiple channels.

    Returns:
        pandas.DataFrame: Table of labeled regions in labels with corresponding feature measurements.
    """
    features = features.copy() if features else dict()
    features.update(features_basic)

    # Choose appropriate feature table based on multichannel flag
    # Extract features using the feature table function
    if multichannel:
        df = feature_table_multichannel(data, labels, features)
    else:
        df = feature_table(data, labels, features)

    # Add wildcard metadata to the DataFrame
    for k, v in sorted(wildcards.items()):
        df[k] = v

    return df


def extract_features_bare(
    data, labels, features=None, wildcards=None, multichannel=False
):
    """Extract features in dictionary and combine with generic region features.

    Args:
        data (numpy.ndarray): Image data of dimensions (CHANNEL, I, J).
        labels (numpy.ndarray): Labeled segmentation mask defining objects to extract features from.
        features (dict or None): Features to extract and their defining functions. Default is None.
        wildcards (dict or None): Metadata to include in the output table, e.g., well, tile, etc. Default is None.
        multichannel (bool): Flag indicating whether the data has multiple channels.

    Returns:
        pandas.DataFrame: Table of labeled regions in labels with corresponding feature measurements.
    """
    features = features.copy() if features else dict()
    features.update({"label": lambda r: r.label})

    # Choose appropriate feature table based on multichannel flag
    # Extract features using the feature table function
    if multichannel:
        df = feature_table_multichannel(data, labels, features)
    else:
        df = feature_table(data, labels, features)

    # Add wildcard metadata to the DataFrame if provided
    if wildcards is not None:
        for k, v in sorted(wildcards.items()):
            df[k] = v

    return df



def log_filter(aligned_image_data, sigma=1, skip_index=None):
    """Apply Laplacian-of-Gaussian filter from scipy.ndimage to the input data.

    Args:
        aligned_image_data (numpy.ndarray): Aligned SBS image data with expected dimensions of (CYCLE, CHANNEL, I, J).
        sigma (float, optional): Size of the Gaussian kernel used in the Laplacian-of-Gaussian filter. Default is 1.
        skip_index (None or int, optional): If an integer, skips transforming a specific channel (e.g., DAPI with skip_index=0).

    Returns:
        loged (numpy.ndarray): LoG-ed `data`.
    """
    # Convert input data to a numpy array
    aligned_image_data = np.array(aligned_image_data)

    # Apply Laplacian-of-Gaussian filter
    loged = log_ndi(aligned_image_data, sigma=sigma)

    # If skip_index is specified, keep the original values for the corresponding channel
    if skip_index is not None:
        loged[..., skip_index, :, :] = aligned_image_data[..., skip_index, :, :]

    return loged


# @applyIJ
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
    f = dask_image.ndfilters.gaussian_laplace

    # Apply the filter to the data and invert the output
    arr_ = -1 * f(data.astype(float), sigma, *args, **kwargs)

    # Clip values to ensure they are within the valid range [0, 65535] and convert back to uint16
    arr_ = np.clip(arr_, 0, 65535) / 65535

    # Suppress precision warning from skimage
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return skimage.img_as_uint(arr_)