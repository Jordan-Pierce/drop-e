import os
import sys
import glob
import shutil

import math
import numpy as np
import pandas as pd
import geopandas as gpd
from affine import Affine
from osgeo import gdal, osr
from sklearn.linear_model import LinearRegression


def create_reference_csv(input_folder):
    """Takes in an input folder, tries to find the image_centroids.gpkg,
    opens it using geopandas, creates a new version containing only a few
    columns needed for metashape."""

    files = [file for file in glob.glob(input_folder + "*.gpkg") if "image_centroid" in file]

    if len(files) == 1:
        file = files[0]
    else:
        # Error identifying the image_centroids.gpkg
        sys.exit()

    # Load GeoPackage file into a GeoDataFrame
    gdf = gpd.read_file(file)

    # Extract "img_name" and "GPS_Altitude" columns and rename them
    gdf_ = gdf[["img_name"]].rename(columns={"img_name": "Label"})

    # Split point geometry into separate columns for easting and northing
    gdf_['Easting'] = gdf['geometry'].apply(lambda p: p.x)
    gdf_['Northing'] = gdf['geometry'].apply(lambda p: p.y)
    gdf_['Altitude'] = gdf['CaAltCor_m']

    # Output file
    out_file = file.replace(".gpkg", ".csv")

    # Save the csv
    gdf_.to_csv(out_file, index=False)

    if os.path.exists(out_file):
        print("Created Reference: ", out_file)
    else:
        print("ERROR: Could not create Reference: ", out_file)
        out_file = None

    return gdf_, out_file


def georeference_orthomosaic(src_path, dst_path, georeference_path):
    """Takes in the path to the orthomosaic, and the geo-reference
    information. Applies the geo-reference information to the orthomosaic
    and saves it as the output path."""

    # Copy the orthomosaic to the output path
    shutil.copyfile(src_path, dst_path)

    # Path the georeferencing information
    df = pd.read_csv(georeference_path, index_col=0)
    gcps = df[['Easting', 'Northing', 'x_pixels', 'y_pixels']].dropna()

    # Define the independent and dependent variables
    X = gcps[["x_pixels", "y_pixels"]]
    y = gcps[["Easting", "Northing"]]

    # Create the linear regression model
    model = LinearRegression().fit(X, y)

    # Calculate estimated origin of orthomosaic
    easting_origin, northing_origin = model.predict(np.array([0, 0]).reshape(1, -1)).T

    # Load the image to be georeference
    ds = gdal.Open(dst_path, gdal.GA_Update)

    # Default pixel size
    pixel_size = 0.0003

    # Set the geotransform using the estimated origin point.
    # Reversing the sign of the pixel size is necessary because
    # the y-axis is inverted in the image.
    geotransform = (easting_origin[0],
                    -pixel_size,
                    0,
                    northing_origin[0],
                    0,
                    pixel_size)

    print(geotransform)

    # Set the projection using the UTM coordinate system
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32655)

    # Set the geotransform and projection to the image
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(srs.ExportToWkt())
    # Save the image
    ds = None