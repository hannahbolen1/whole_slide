from functools import partial
import numpy as np
import skimage as ski
from collections import defaultdict
import pandas as pd


# adapted from brieflow cp_emulator -- need to go through and rework places using featuretable etc

# constants they define im not sure why (with comments)
EDGE_CONNECTIVITY = 2 # cellprofiler uses edge connectivity of 1, which exlucdes pixels catty-corner to a boundary

def neighbor_measurements(labeled, distances=[1, 10], n_cpu=1):
    from pandas import concat

    dfs = [
        object_neighbors(labeled, distance=distance).rename(
            columns=lambda x: x + "_" + str(distance)
        )
        for distance in distances
    ]

    dfs.append(
        closest_objects(labeled, n_cpu=n_cpu).drop(
            columns=["first_neighbor", "second_neighbor"]
        )
    )

    return concat(dfs, axis=1, join="outer").reset_index()

def closest_objects(labeled, n_cpu=1):
    from myutils import feature_table
    from scipy.spatial import cKDTree

    features = {
        "i": lambda r: r.centroid[0],
        "j": lambda r: r.centroid[1],
        "label": lambda r: r.label,
    }

    df = feature_table(labeled, labeled, features)

    # Handle cases with fewer than 3 objects
    if len(df) < 3:
        result_df = df.copy()
        result_df["first_neighbor"] = np.nan
        result_df["first_neighbor_distance"] = np.nan
        result_df["second_neighbor"] = np.nan
        result_df["second_neighbor_distance"] = np.nan
        result_df["angle_between_neighbors"] = np.nan

        # If we have exactly 2 objects, we can fill in the first neighbor info
        if len(df) == 2:
            # Each object's first neighbor is the other object
            result_df["first_neighbor"] = result_df.index[::-1].values
            # Calculate distance between the two objects
            points = result_df[["i", "j"]].values
            distance = np.sqrt(((points[0] - points[1]) ** 2).sum())
            result_df["first_neighbor_distance"] = distance
            # No second neighbor, angle remains NaN

        return result_df.drop(columns=["i", "j"]).set_index("label")

    kdt = cKDTree(df[["i", "j"]])

    distances, indexes = kdt.query(df[["i", "j"]], 3, workers=n_cpu)

    df["first_neighbor"], df["first_neighbor_distance"] = indexes[:, 1], distances[:, 1]
    df["second_neighbor"], df["second_neighbor_distance"] = (
        indexes[:, 2],
        distances[:, 2],
    )

    first_neighbors = df[["i", "j"]].values[df["first_neighbor"].values]
    second_neighbors = df[["i", "j"]].values[df["second_neighbor"].values]

    angles = [
        angle(v, p0, p1)
        for v, p0, p1 in zip(df[["i", "j"]].values, first_neighbors, second_neighbors)
    ]

    df["angle_between_neighbors"] = np.array(angles) * (180 / np.pi)

    return df.drop(columns=["i", "j"]).set_index("label")


def object_neighbors(labeled, distance=1):
    from skimage.measure import regionprops
    from pandas import DataFrame

    outlined = (
        boundaries(labeled, connectivity=EDGE_CONNECTIVITY, mode="inner") * labeled
    )

    regions = regionprops(labeled)

    bboxes = [r.bbox for r in regions]

    labels = [r.label for r in regions]

    neighbors_disk = skimage.morphology.disk(distance)

    perimeter_disk = cp_disk(distance + 0.5)

    info_dicts = [
        neighbor_info(
            labeled, outlined, label, bbox, distance, neighbors_disk, perimeter_disk
        )
        for label, bbox in zip(labels, bboxes)
    ]

    return DataFrame(info_dicts).set_index("label")


def neighbor_info(
    labeled, outlined, label, bbox, distance, neighbors_disk=None, perimeter_disk=None
):
    if neighbors_disk is None:
        neighbors_disk = skimage.morphology.disk(distance)
    if perimeter_disk is None:
        perimeter_disk = cp_disk(distance + 0.5)

    label_mask = subimage(labeled, bbox, pad=distance)
    outline_mask = subimage(outlined, bbox, pad=distance) == label

    dilated = skimage.morphology.binary_dilation(
        label_mask == label, footprint=neighbors_disk
    )
    neighbors = np.unique(label_mask[dilated])
    neighbors = neighbors[(neighbors != 0) & (neighbors != label)]
    n_neighbors = len(neighbors)

    dilated_neighbors = skimage.morphology.binary_dilation(
        (label_mask != label) & (label_mask != 0), footprint=perimeter_disk
    )
    percent_touching = (outline_mask & dilated_neighbors).sum() / outline_mask.sum()

    return {
        "label": label,
        "number_neighbors": n_neighbors,
        "percent_touching": percent_touching,
    }

def subimage(stack, bbox, pad=0):
    """
    Extract a rectangular region from a stack of images with optional padding.

    Args:
        stack (np.ndarray): Input stack of images [...xYxX].
        bbox (np.ndarray or list): Bounding box coordinates (min_row, min_col, max_row, max_col).
        pad (int, optional): Padding width. Defaults to 0.

    Returns:
        np.ndarray: Extracted subimage.

    Notes:
        - If boundary lies outside stack, raises error.
        - If padded rectangle extends outside stack, fills with zeros.
    """
    i0, j0, i1, j1 = bbox + np.array([-pad, -pad, pad, pad])

    sub = np.zeros(stack.shape[:-2] + (i1 - i0, j1 - j0), dtype=stack.dtype)

    i0_, j0_ = max(i0, 0), max(j0, 0)
    i1_, j1_ = min(i1, stack.shape[-2]), min(j1, stack.shape[-1])
    s = (
        Ellipsis,
        slice(i0_ - i0, (i0_ - i0) + i1_ - i0_),
        slice(j0_ - j0, (j0_ - j0) + j1_ - j0_),
    )

    sub[s] = stack[..., i0_:i1_, j0_:j1_]
    return sub

def boundaries(labeled, connectivity=1, mode="inner", background=0):
    """Supplement skimage.segmentation.find_boundaries to include image edge pixels of
    labeled regions as boundary
    """
    from skimage.segmentation import find_boundaries

    kwargs = dict(connectivity=connectivity, mode=mode, background=background)
    # if mode == 'inner':
    pad_width = 1
    # else:
    #     pad_width = connectivity

    padded = np.pad(
        labeled, pad_width=pad_width, mode="constant", constant_values=background
    )
    return find_boundaries(padded, **kwargs)[
        ..., pad_width:-pad_width, pad_width:-pad_width
    ]

def angle(vertex, p0, p1):
    v0 = p0 - vertex
    v1 = p1 - vertex

    cosine_angle = np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))
    return np.arccos(cosine_angle)

def cp_disk(radius):
    """Create a disk structuring element for morphological operations

    radius - radius of the disk
    """
    iradius = int(radius)
    x, y = np.mgrid[-iradius : iradius + 1, -iradius : iradius + 1]
    radius2 = radius * radius
    strel = np.zeros(x.shape)
    strel[x * x + y * y <= radius2] = 1
    return strel

DEFAULT_METADATA_COLS = [
    "day",
    "condition",
    "image",
    "nucleus_i",
    "nucleus_j",
    "nucleus_bounds_0",
    "nucleus_bounds_1",
    "nucleus_bounds_2",
    "nucleus_bounds_3",
]


def feature_table(labels, features, data=None):
    """Apply functions in feature dictionary to regions in data specified by integer labels.

    If provided, the global feature dictionary is applied to the full input data and labels.
    Results are combined in a dataframe with one row per label and one column per feature.

    Args:
        labels (np.ndarray): Labeled segmentation mask defining objects to extract features from.
        features (dict): Dictionary of feature names and their corresponding functions.
        data (np.ndarray, optional): Image data. Default is None.

    Returns:
        pd.DataFrame: DataFrame containing extracted features with one row per label and one column per feature.
    """
    # Extract regions from the labeled segmentation mask
    regions = ski.measure.regionprops(labels, intensity_image=data)

    # Initialize a defaultdict to store feature values
    results = defaultdict(list)

    # Loop through each region and compute features
    for region in regions:
        for feature, func in features.items():
            # Apply the feature function to the region and append the result to the corresponding feature list
            results[feature].append(func(region))

    # Convert the results dictionary to a DataFrame
    return pd.DataFrame(results)


# features to describe foci
foci_features = {
    "label": lambda r: r.label,
    "foci_count": lambda r: count_labels(r.intensity_image),
    "foci_area": lambda r: (r.intensity_image > 0).sum(),
}

# Basic features to describe nuclei
features_basic = {
    "area": lambda r: r.area,
    "i": lambda r: r.centroid[0],
    "j": lambda r: r.centroid[1],
    "label": lambda r: r.label,
    "bounds": lambda r: r.bbox,
}

def count_labels(labels, return_list=False):
    """Count the unique non-zero labels in a labeled segmentation mask.

    Args:
        labels (numpy array): Labeled segmentation mask.
        return_list (bool): Flag indicating whether to return the list of unique labels along with the count.

    Returns:
        int or tuple: Number of unique non-zero labels. If return_list is True, returns a tuple containing the count
      and the list of unique labels.
    """
    # Get unique labels in the segmentation mask
    uniques = np.unique(labels)
    # Remove the background label (0)
    ls = np.delete(uniques, np.where(uniques == 0))
    # Count the unique non-zero labels
    num_labels = len(ls)
    # Return the count or both count and list of unique labels based on return_list flag
    if return_list:
        return num_labels, ls
    return num_labels