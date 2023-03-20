#### Scripts

These scripts are used to create Orthomosaics using the georeferenced images produced by `georeference_image.py`. The 
`sfm_workflow.py` contains the function `run_sfm_workflow()` that takes in a folder directory (`image_folder`) 
containing georeferenced `jpg` images as input. The function will glob all files ending with `jpg`, and also search for
a `csv` file called `image_centroid.csv`, a product of `georefence_images.py`. The two will be used with the 
Metashape API to perform the following:

- Setting the coordinate system (`EPSG 32655`)
- Image importing
- Image quality testing (will disable images with quality lower than threshold)
- Image matching
- Image alignment (highest)
- Optimize image alignment
- Depth maps creation (highest)
- Mesh creation (highest)
- Orthomosaic creation 
- Exporting of georeferenced Orthomosaic (saved in provided folder, `./image_folder/Orthomosaic.tif`)

The script `create_tif.py` is used to convert the original `jpg` image into a georeferenced GeoTIFF (`tif`) using the 
georeference side-car file (`.jpg.aux.xml`). Currently, this function is not used, but may be useful to have.

The script `create_reference.py` is used to convert the `image_centroids.gpkg` output by `georeference_images.py` into 
a `csv` file in the format that Metashape expects. This `csv` is created in the `run_sfm_workflow()` function, and 
passed to the Metashape before image alignment.