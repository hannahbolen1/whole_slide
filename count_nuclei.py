import os
import matplotlib.pyplot as plt
import dask.array as da
from skimage.measure import label
import tifffile
import skimage as ski
import pandas as pd

root = "/yoren62/users/hannahbolen/immunofluorescence/slidescan_20250723_o8_gH2AX"
images_path = "/yoren62/users/hannahbolen/immunofluorescence/slidescan_20250723_o8_gH2AX/converted"
masks_folder = "masks"
masks_path = "/yoren62/users/hannahbolen/immunofluorescence/slidescan_20250723_o8_gH2AX/masks/"
gfp_intensity = (256, 5000)
ds = 10

#counts = pd.DataFrame({"file":[], "count":[]})
#counts.to_csv("/yoren62/users/hannahbolen/immunofluorescence/slidescan_20250723_o8_gH2AX/counts.csv", index=False)
counts = pd.read_csv("/yoren62/users/hannahbolen/immunofluorescence/slidescan_20250723_o8_gH2AX/counts.csv")
w = 5000
y0 = 10000
x0 = 10000


for file in os.listdir(images_path):
    img_name = file.split("/")[-1]
    img = tifffile.imread(os.path.join(images_path, file), aszarr = True)
    img = da.from_zarr(img)
    nuclei = img[0]

    # downsample to calculate threshold
    nuclei_ds = nuclei[::ds,::ds].compute()

    # calculate threshold
    thresholdOtsu = ski.filters.threshold_otsu(nuclei_ds)

    # calculate mask, label nuclei
    mask_nuclei = (nuclei >= thresholdOtsu).compute()
    lbl_nuclei = label(mask_nuclei, connectivity=2)
    
    # add number of nuclei to dataframe
    counts.loc[len(counts)] = [file.split("/")[-1], lbl_nuclei.max()]

    # save mask
    mask_file = "".join([masks_path, img_name.split(".")[0], "_mask.tif"])
    img_dtype = tifffile.TiffFile(os.path.join(images_path, file)).pages[0].dtype
    tifffile.imwrite(mask_file, mask_nuclei.astype(img_dtype))

    # save zoomed in images to check
    nuclei_rescale = ski.exposure.rescale_intensity(nuclei, in_range = gfp_intensity)
    mask_crop = "".join([masks_path, file.split(".")[0], "_mask_cropped.tif"])
    gfp_crop = "".join([masks_path, file.split(".")[0], "_gfp_cropped.tif"])
    tifffile.imwrite(gfp_crop, nuclei_rescale[y0:y0+w, x0:x0+w].astype(img_dtype))
    tifffile.imwrite(mask_crop, mask_nuclei[y0:y0+w, x0:x0+w].astype(img_dtype))

counts.to_csv("/yoren62/users/hannahbolen/immunofluorescence/slidescan_20250723_o8_gH2AX/counts.csv", index=False)