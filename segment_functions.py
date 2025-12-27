# adapting from from https://github.com/feldman4/OpticalPooledScreens/blob/master/ops/process.py

import time
from contextlib import contextmanager
import skimage
import skimage.segmentation
import skimage.morphology
import numpy as np
from scipy import ndimage as ndi

OPS_PROFILE = False
OPS_PROFILE_VERBOSE = False
OPS_PROFILE_TIMES = {}

@contextmanager
def ops_timer(name):
    if not OPS_PROFILE:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        OPS_PROFILE_TIMES[name] = OPS_PROFILE_TIMES.get(name, 0.0) + elapsed
        if OPS_PROFILE_VERBOSE:
            print(f"[ops_timer] {name}: {elapsed:.3f}s", flush=True)

def ops_timing_reset():
    OPS_PROFILE_TIMES.clear()

def ops_timing_report():
    return sorted(OPS_PROFILE_TIMES.items(), key=lambda kv: kv[1], reverse=True)

def ops_timing_summary():
    total = sum(OPS_PROFILE_TIMES.values())
    rows = ops_timing_report()
    lines = []
    for name, secs in rows:
        pct = (secs / total * 100.0) if total > 0 else 0.0
        lines.append(f"{name:24s} {secs:8.3f}s  {pct:5.1f}%")
    lines.append(f"{'total':24s} {total:8.3f}s  100.0%")
    return "\n".join(lines)


# SEGMENT
def find_nuclei(dapi, threshold, radius=15, area_min=50, area_max=500,
                score=lambda r: r.mean_intensity,
                smooth=1.35):
    """
    """

    with ops_timer("binarize"):
        mask = binarize(dapi, radius, area_min)
    with ops_timer("label"):
        labeled = skimage.measure.label(mask)
    with ops_timer("filter_by_region_initial"):
        labeled = filter_by_region(
            labeled, score, threshold, intensity_image=dapi
        ) > 0

    # only fill holes below minimum area
    with ops_timer("fill_holes"):
        filled = ndi.binary_fill_holes(labeled)
    with ops_timer("label_hole_diff"):
        difference = skimage.measure.label(filled != labeled)

    with ops_timer("filter_by_region_holes"):
        change = filter_by_region(difference, lambda r: r.area < area_min, 0) > 0
    with ops_timer("apply_hole_fill"):
        labeled[change] = filled[change]

    with ops_timer("watershed"):
        nuclei = apply_watershed(labeled, smooth=smooth)

    with ops_timer("filter_by_region_final"):
        result = filter_by_region(
            nuclei, lambda r: area_min < r.area < area_max, threshold
        )

    return result


def binarize(image, radius, min_size):
    """Apply local mean threshold to find outlines. Filter out
    background shapes. Otsu threshold on list of region mean intensities will remove a few
    dark cells. Could use shape to improve the filtering.
    """
    dapi = skimage.img_as_ubyte(image)
    # slower than optimized disk in ImageJ
    # scipy.ndimage.uniform_filter with square is fast but crappy
    selem = skimage.morphology.disk(radius)
    mean_filtered = skimage.filters.rank.mean(dapi, footprint=selem)
    mask = dapi > mean_filtered
    mask = skimage.morphology.remove_small_objects(mask, min_size=min_size)

    return mask

def filter_by_region(labeled, threshold, score=lambda r: r.mean_intensity, intensity_image=None, relabel=True):
    """Apply a filter to label image. The `score` function takes a single region 
    as input and returns a score. 
    If scores are boolean, regions where the score is false are removed.
    Otherwise, the function `threshold` is applied to the list of scores to 
    determine the minimum score at which a region is kept.
    If `relabel` is true, the regions are relabeled starting from 1.
    """
    labeled = labeled.copy().astype(int)
    regions = skimage.measure.regionprops(labeled, intensity_image=intensity_image)
    scores = np.array([score(r) for r in regions])

    if all([s in (True, False) for s in scores]):
        cut = [r.label for r, s in zip(regions, scores) if not s]
    else:
        t = threshold(scores)
        cut = [r.label for r, s in zip(regions, scores) if s < t]

    labeled.flat[np.isin(labeled.flat[:], cut)] = 0
    
    if relabel:
        labeled, _, _ = skimage.segmentation.relabel_sequential(labeled)

    return labeled


def apply_watershed(img, smooth=4):
    distance = ndi.distance_transform_edt(img)
    if smooth > 0:
        distance = skimage.filters.gaussian(distance, sigma=smooth)
    local_max = skimage.feature.peak_local_max(
                    distance, footprint=np.ones((4, 4)), 
                    exclude_border=False, labels=img)

    # Newer skimage returns coordinates; convert to mask for labeling.
    if local_max.shape != distance.shape:
        mask = np.zeros_like(distance, dtype=bool)
        if local_max.size:
            mask[tuple(local_max.T)] = True
        local_max = mask

    markers = ndi.label(local_max)[0]
    result = skimage.segmentation.watershed(-distance, markers, mask=img)
    return result.astype(np.uint16)