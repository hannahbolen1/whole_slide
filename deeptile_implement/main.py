from extract_features import features_basic, foci_features, feature_table, neighbor_measurements
import os
import deeptile
import numpy as np
from deeptile.extensions import stitch
import tifffile
import dask.array as da
import utils
import skimage as ski
import extract_features
import pandas as pd

root = "/Users/hannahbolen/Desktop/image_analysis/"
# img_name = 'o8p_day18_s22.tif'
# img_name = 'o8p_day18.vsi'
img_name = "o8p_day24_s12.ome.tif"
img_path = os.path.join(root, img_name)
img = da.from_array(tifffile.imread(img_path))[:, 29500:29500+5120, 10700:10700+5120]
dt_nuclei = deeptile.load(img[0])
dt_foci = deeptile.load(img[1])

# Configure
tile_size = (512, 512)
overlap = (0.1, 0.1)
# Get nuceli tiles
tiles_nuclei = dt_nuclei.get_tiles(tile_size, overlap)
tiles_nuclei = tiles_nuclei.pad()
# Get foci tiles
tiles_foci = dt_foci.get_tiles(tile_size, overlap)
tiles_foci = tiles_foci.pad()
# Individual tile
tiles_nuclei[0, 0]
tiles_foci[0, 0]


# Segment tiles and stitch
model_parameters = {'gpu': True, 'model_type': 'nuclei'}
eval_parameters = {'diameter': 60}
cellpose = utils.cellpose_segmentation(model_parameters, eval_parameters)

masks_nuclei = cellpose(tiles_nuclei)
mask_nuclei = stitch.stitch_masks(masks_nuclei)


## find foci
## make tiled nuclei mask with same profile as tiled foci
import_masks_nuclei = tiles_foci.import_data(mask_nuclei, "image").unpad().pad() # need to unpad, pad bc bug in package code
# segment foci and stitch
kwargs = {"radius":2, "threshold":25, "min_distance":1, "regions":import_masks_nuclei, "remove_border_foci":True}
masks_foci = utils.segment_foci_tiled(tiles_foci, **kwargs)
mask_foci = stitch.stitch_masks(masks_foci)


dfs = []
dfs.append(
    feature_table(mask_nuclei, features_basic)
    .set_index("label")
    .add_prefix("nuclei_")
)

dfs.append(
    feature_table(mask_nuclei, foci_features, mask_foci)
    .set_index("label")
)

dfs.append(
    neighbor_measurements(mask_nuclei, distances=[1])
    .set_index("label")
    .add_prefix("nucleus_")
)

results = pd.concat(dfs, axis=1, join="outer", sort=False).reset_index().set_index("label", drop=False)
results.to_csv("/Users/hannahbolen/Desktop/image_analysis/whole_slide/results.csv")