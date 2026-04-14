import numpy as np
from misc_utils import strel_disk
import scipy.ndimage as ndi
import scipy.signal
import tifffile
from misc_utils import strel_disk

def neighbors_for_object(object_number, min_i, max_i, min_j, max_j, labels_path, distance):
    img = tifffile.memmap(labels_path, mode='r')

    patch = img[
        min_i : max_i,
        min_j : max_j,
    ].copy()

    #
    # Find the neighbors
    #
    
    patch_mask = patch == (object_number)

    strel = strel_disk(distance)

    if distance <= 5:
        extended = ndi.binary_dilation(patch_mask, strel)
    else:
        extended = (
            scipy.signal.fftconvolve(patch_mask, strel, mode="same") > 0.5
        )

    neighbors = np.unique(patch[extended])

    neighbors = neighbors[(neighbors != 0) & (neighbors != object_number)]

    if hasattr(neighbors, "compute"):
        neighbors = neighbors.compute()

    return [(int(n), int(object_number)) for n in neighbors]



def objects_bounds(label_image, distance):
    nobjects = np.max(label_image)
    # object_indexes = np.arange(nobjects, dtype=np.int32) + 1
    # strel = strel_disk(distance)

    objs = ndi.find_objects(label_image, max_label=nobjects)
    # objs is a list of slice-tuples, one per label 1..nobjects
    # missing labels give None

    minimums_i = np.empty(nobjects, dtype=np.int64)
    maximums_i = np.empty(nobjects, dtype=np.int64)
    minimums_j = np.empty(nobjects, dtype=np.int64)
    maximums_j = np.empty(nobjects, dtype=np.int64)

    for k, slc in enumerate(objs):
        if type(distance) == list:
            d = int(distance[k])
        else:
            d = distance

        if slc is None:
            minimums_i[k] = 0
            maximums_i[k] = 0
            minimums_j[k] = 0
            maximums_j[k] = 0
            continue

        si, sj = slc
        minimums_i[k] = max(si.start - d, 0)
        maximums_i[k] = min(si.stop  + d, label_image.shape[0])
        minimums_j[k] = max(sj.start - d, 0)
        maximums_j[k] = min(sj.stop  + d, label_image.shape[1])

    return minimums_i, maximums_i, minimums_j, maximums_j