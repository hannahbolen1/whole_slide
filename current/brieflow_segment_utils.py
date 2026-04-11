"""Shared segmentation utilities for all segmentation methods.

This module provides common functions used across different segmentation methods:
- image_log_scale: Apply log scaling to images for preprocessing
- reconcile_nuclei_cells: Reconcile nuclei and cell labels based on overlap
- center_pixels: Assign labels to center pixels of regions
- relabel_array: Map values in an array based on a label dictionary

These utilities are extracted from individual segmentation modules to avoid code duplication
and ensure consistent behavior across different segmentation methods.
"""

import numpy as np
from collections import defaultdict
from skimage.measure import regionprops
import dask.array as da
from deeptile.core.lift import lift
from deeptile.core.utils import compute_dask


def image_log_scale(data, bottom_percentile=10, floor_threshold=50, ignore_zero=True):
    """Apply log scaling to an image.

    Args:
        data (numpy.ndarray): Input image data.
        bottom_percentile (int, optional): Percentile value for determining the bottom threshold. Default is 10.
        floor_threshold (int, optional): Floor threshold for cutting out noisy bits. Default is 50.
        ignore_zero (bool, optional): Whether to ignore zero values in the data. Default is True.

    Returns:
        numpy.ndarray: Scaled image data after log scaling.
    """
    is_dask = isinstance(data, da.Array)

    # Handle empty arrays early if shape is known
    if data.size == 0:
        return data.astype(float)

    # Convert to float
    data = data.astype(float)

    if is_dask:
        # Dask-safe percentile calculation
        if ignore_zero:
            # Replace zeros/non-positives with NaN so they are ignored
            positive = data[data > 0]
            positive = positive.compute_chunk_sizes()
            bottom = da.percentile(positive, bottom_percentile)
        else:
            bottom = da.percentile(data.ravel(), bottom_percentile)

        # Clip values below bottom without in-place assignment
        clipped = da.maximum(data, bottom)

        # Apply log scaling
        scaled = da.log10(clipped - bottom + 1)

        # Apply floor
        floor = np.log10(floor_threshold)
        scaled = da.maximum(scaled, floor)

        return scaled - floor
    else:
        # Select data based on whether to ignore zero values or not
        if ignore_zero:
            data_perc = data[data > 0]
        else:
            data_perc = data

        # Determine the bottom percentile value
        bottom = np.percentile(data_perc, bottom_percentile)

        # Set values below the bottom percentile to the bottom value
        data[data < bottom] = bottom

        # Apply log scaling with floor threshold
        scaled = np.log10(data - bottom + 1)

        # Cut out noisy bits based on the floor threshold
        floor = np.log10(floor_threshold)
        scaled[scaled < floor] = floor

        # Subtract the floor value
        return scaled - floor


def center_pixels(label_image):
    """Assign labels to center pixels of regions in a labeled image.

    Args:
        label_image (numpy.ndarray): Labeled image.

    Returns:
        numpy.ndarray: Image with labels assigned to center pixels of regions.
    """
    ultimate = np.zeros_like(label_image)  # Initialize an array to store the result
    for r in regionprops(label_image):  # Iterate over regions in the labeled image
        # Calculate the mean coordinates of the bounding box of the region
        i, j = np.array(r.bbox).reshape(2, 2).mean(axis=0).astype(int)
        # Assign the label of the region to the center pixel
        ultimate[i, j] = r.label
    return ultimate  # Return the image with labels assigned to center pixels



def dask_image_log_scale(data, bottom_percentile=10, floor_threshold=50, ignore_zero=True):
    """Apply log scaling to an image.

    Args:
        data (numpy.ndarray | dask.array.Array): Input image data.
        bottom_percentile (int, optional): Percentile value for determining the bottom threshold. Default is 10.
        floor_threshold (int, optional): Floor threshold for cutting out noisy bits. Default is 50.
        ignore_zero (bool, optional): Whether to ignore zero values in the data. Default is True.

    Returns:
        numpy.ndarray | dask.array.Array: Scaled image data after log scaling.
            Returns a dask array if the input was a dask array, otherwise a numpy array.
    """

    is_dask = isinstance(data, da.Array)


    xp = da if is_dask else np

    # Safety check: return early for empty or all-zero data
    if data.size == 0:
        return data

    all_zero = xp.all(data == 0)
    if is_dask:
        all_zero = all_zero.compute()
    if all_zero:
        return data

    # Convert input data to float
    data = data.astype(float)

    # Select data based on whether to ignore zero values or not
    # Note: dask supports boolean indexing but produces unknown-length chunks;
    # we only use it transiently here for the percentile calculation.
    if ignore_zero:
        data_perc = data[data > 0]
    else:
        data_perc = data

    # Determine the bottom percentile value (always resolve to a Python scalar)
    if is_dask:
       bottom = float(da.percentile(data_perc.ravel(), bottom_percentile).compute().item())
    else:
        bottom = np.percentile(data_perc, bottom_percentile)

    # Set values below the bottom percentile to the bottom value.
    # da.Array does not support boolean index assignment, so use where() for both backends.
    data = xp.where(data < bottom, bottom, data)

    # Apply log scaling with floor threshold
    scaled = xp.log10(data - bottom + 1)

    # Cut out noisy bits based on the floor threshold
    floor = float(np.log10(floor_threshold))  # scalar – fine for both backends
    scaled = xp.where(scaled < floor, floor, scaled)

    # Subtract the floor value
    return scaled - floor

def relabel_array(arr, new_label_dict):
    """Map values in an integer array based on `new_label_dict`, a dictionary from old to new values.

    Args:
        arr (numpy.ndarray): The input integer array to be relabeled.
        new_label_dict (dict): A dictionary mapping old values to new values.

    Returns:
        numpy.ndarray: The relabeled integer array.

    Notes:
    - The function iterates through the items in `new_label_dict` and maps old values to new values in the array.
    - Values in the array that do not have a corresponding mapping in `new_label_dict` remain unchanged.
    """
    n = arr.max()  # Find the maximum value in the array
    arr_ = np.zeros(n + 1)  # Initialize an array to store the relabeled values
    for old_val, new_val in new_label_dict.items():
        if old_val <= n:  # Check if the old value is within the range of the array
            arr_[old_val] = (
                new_val  # Map the old value to the new value in the relabeling array
            )
    return arr_[arr]  # Return the relabeled array





def reconcile_nuclei_cells(nuclei, cells, how="consensus"):
    """Reconcile nuclei and cells labels based on their overlap.

    Args:
        nuclei (memmap array): Nuclei mask.
        cells (memmap array): Cell mask.
        how (str, optional): Method to reconcile labels.
            - 'consensus': Only keep nucleus-cell pairs where label matches are unique.
            - 'contained_in_cells': Keep multiple nuclei for a single cell but merge them.

    Returns:
        tuple: Tuple containing the reconciled nuclei and cells masks.
    """

    def get_unique_label_map(regions, keep_multiple=False):
        """Get unique label map from regions.

        Args:
            regions (list): List of regions.
            keep_multiple (bool, optional): Whether to keep multiple labels for each region.

        Returns:
            dict: Dictionary containing the label map.
        """
        label_map = {}
        for region in regions:
            intensity_image = region.intensity_image[region.intensity_image > 0]
            labels = np.unique(intensity_image)
            if keep_multiple:
                label_map[region.label] = labels
            elif len(labels) == 1:
                label_map[region.label] = labels[0]
        return label_map

    # Erode nuclei to prevent overlapping with cells
    nuclei_eroded = center_pixels(nuclei)

    # Get unique label maps for nuclei and cells
    nucleus_map = get_unique_label_map(
        regionprops(nuclei_eroded, intensity_image=cells)
    )

    # Always get the multiple nuclei mapping for analysis
    cell_map_multiple = get_unique_label_map(
        regionprops(cells, intensity_image=nuclei_eroded), keep_multiple=True
    )

    # Count cells with multiple nuclei
    nuclei_per_cell = defaultdict(int)
    for cell_label, nuclei_labels in cell_map_multiple.items():
        nuclei_per_cell[len(nuclei_labels)] += 1

    # Print statistics
    print("\nNuclei per cell statistics:")
    print("--------------------------")
    for num_nuclei, count in sorted(nuclei_per_cell.items()):
        print(f"Cells with {num_nuclei} nuclei: {count}")
    print("--------------------------\n")

    if how == "contained_in_cells":
        cell_map = get_unique_label_map(
            regionprops(cells, intensity_image=nuclei_eroded), keep_multiple=True
        )
    else:
        cell_map = get_unique_label_map(
            regionprops(cells, intensity_image=nuclei_eroded)
        )

    # Keep only nucleus-cell pairs with matching labels
    keep = []
    for nucleus in nucleus_map:
        try:
            if how == "contained_in_cells":
                if nucleus in cell_map[nucleus_map[nucleus]]:
                    keep.append([nucleus, nucleus_map[nucleus]])
            else:
                if cell_map[nucleus_map[nucleus]] == nucleus:
                    keep.append([nucleus, nucleus_map[nucleus]])
        except KeyError:
            pass

    # If no matches found, return zero arrays
    if len(keep) == 0:
        return np.zeros_like(nuclei), np.zeros_like(cells), nuclei_per_cell

    # Extract nuclei and cells to keep
    keep_nuclei, keep_cells = zip(*keep)

    # Reassign labels based on the reconciliation method
    if how == "contained_in_cells":
        nuclei = relabel_array(
            nuclei, {nuclei_label: cell_label for nuclei_label, cell_label in keep}
        )
        cells[~np.isin(cells, keep_cells)] = 0
        labels, cell_indices = np.unique(cells, return_inverse=True)
        _, nuclei_indices = np.unique(nuclei, return_inverse=True)
        cells = np.arange(0, labels.shape[0])[cell_indices.reshape(*cells.shape)]
        nuclei = np.arange(0, labels.shape[0])[nuclei_indices.reshape(*nuclei.shape)]
    else:
        nuclei = relabel_array(
            nuclei, {label: i + 1 for i, label in enumerate(keep_nuclei)}
        )
        cells = relabel_array(
            cells, {label: i + 1 for i, label in enumerate(keep_cells)}
        )

    # Convert arrays to integers
    nuclei, cells = nuclei.astype(int), cells.astype(int)
    
    return nuclei, cells, nuclei_per_cell


def tiled_reconcile_nuclei_cells(nuclei, cells, howT="consensus"):
    
    totals = defaultdict(int)
    @lift
    def _func_reconc(nuclei_tile, nuclei_index, nuclei_tile_index, nuclei_stitch_index, nuclei_tiling, 
                      cell_tile, cell_index, cell_tile_index, cell_stitch_index, cell_tiling):

        nuclei_tile = compute_dask(nuclei_tile)
        cell_tile = compute_dask(cell_tile)

        nuclei, cells, nuclei_per_cell = reconcile_nuclei_cells(nuclei_tile, cell_tile, how=howT)

        for num_nuclei, count in nuclei_per_cell.items():
            totals[num_nuclei] += count

        return nuclei, cells

    def func_reconc(nuclei_tiles, cells_tiles):

        return _func_reconc(nuclei_tiles, nuclei_tiles.index_iterator, nuclei_tiles.tile_indices_iterator, nuclei_tiles.stitch_indices_iterator,
                             nuclei_tiles.profile.tiling, cells_tiles, cells_tiles.index_iterator, cells_tiles.tile_indices_iterator, cells_tiles.stitch_indices_iterator,
                             cells_tiles.profile.tiling)
    
    nuclei, cells = func_reconc(nuclei, cells)
    # Print statistics
    print("\nNuclei per cell statistics:")
    print("--------------------------")
    for num_nuclei, count in sorted(totals.items()):
        print(f"Cells with {num_nuclei} nuclei: {count}")
    print("--------------------------\n")

    return nuclei, cells