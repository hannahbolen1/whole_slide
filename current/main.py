import os
import tifffile
import dask.array as da
import utils


root = "/Users/hannahbolen/Desktop/image_analysis/images"
input_folder = "input"
coverslip_folder =  "coverslip"
memmap_folder = "memmap_slide"
inputs = os.listdir(os.path.join(root, input_folder))

for slide in inputs:
    img_path = os.path.join(root, input_folder, slide)

    gfp_name = "".join([slide.split(".")[0],"_GFP.tif"])
    cy5_name = "".join([slide.split(".")[0],"_Cy5.tif"])
    coverslip_name = "".join([slide.split(".")[0],"_coverslip.tif"])
    
    with tifffile.TiffFile(img_path) as tif:
        img_dtype = tif.pages[0].dtype
        gfp = da.from_zarr(tif.pages[0].aszarr())
        cy5 = da.from_zarr(tif.pages[1].aszarr())
    


    # save channels as separate contiguous images
    tifffile.imwrite(os.path.join(root, memmap_folder, gfp_name), gfp.astype(img_dtype), dtype=img_dtype, contiguous=True)
    tifffile.imwrite(os.path.join(root, memmap_folder, cy5_name), cy5.astype(img_dtype), dtype=img_dtype, contiguous=True)

    # create coverslip mask from cy5 channel
    coverslip = utils.coverslip_mask(cy5)
    coverslip = coverslip > 0
    tifffile.imwrite(os.path.join(root, coverslip_folder, coverslip_name), coverslip.astype('bool'), dtype='bool', contiguous=True)