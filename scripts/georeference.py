import os
import sys
import glob
import shutil
from tqdm import tqdm

import cv2
import math
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from sklearn.linear_model import LinearRegression

import geopandas as gpd
from osgeo import gdal, osr


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


def get_rotation_angle(gdf, output_folder):
    """Takes in a geodataframe with the camera locations and whether they are aligned,
    returns the rotation angle of the orthomosaic."""

    # Calculate the rotation angle of the orthomosaic given the camera locations that are aligned.
    gcps = gdf[gdf['aligned'] == True][['Easting', 'Northing']]

    # From first point in order to last point
    X = gcps['Easting'].values.reshape(-1, 1)
    Y = gcps['Northing'].values.reshape(-1, 1)
    # Calculate the rotation angle of just those cameras
    model = LinearRegression().fit(X, Y)
    # Get the rotation angle
    r = math.atan(model.coef_[0][0])
    print("Rotation Angle: ", np.rad2deg(r))
    # Plot the UTM points and the slope of the model
    fig, ax = plt.subplots()
    ax.scatter(gcps['Easting'].values, gcps['Northing'].values, color='blue')
    ax.plot(X, model.predict(X), color='red')
    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')
    ax.set_title(f"Rotation Angle: {np.rad2deg(r)} degrees")

    plt.show()

    return r


def georeference_orthomosaic(src_path, georeference_path):
    """Takes in the path to the orthomosaic, and the geo-reference
    information. Applies the geo-reference information to the orthomosaic
    and saves it as the output path."""

    # Path the georeferencing information, drop not aligned
    df = pd.read_csv(georeference_path, index_col=0)
    df = df[df['aligned'] == True]
    # Get the rotation angle
    r = get_rotation_angle(df, os.path.dirname(src_path))

    # Load the image to be georeference
    ds = gdal.Open(src_path, gdal.GA_Update)

    # Get the information for the GCPs
    gcps = df[['Easting', 'Northing', 'x_pixels', 'y_pixels']].dropna()

    # Define the independent and dependent variables
    Xmin = gcps[["Easting"]].min()
    Xmax = gcps[["Easting"]].max()
    Ymin = gcps[["Northing"]].min()
    Ymax = gcps[["Northing"]].max()

    Xmid = (Xmin + Xmax) / 2
    Ymid = (Ymin + Ymax) / 2

    # Default pixel size
    pixel_size = 0.00015

    # Set the geotransform using the estimated origin point.
    # Reversing the sign of the pixel size is necessary because
    # the y-axis is inverted in the image.
    geotransform = (Xmid, pixel_size, 0, Ymid, 0, pixel_size)

    # Set the projection using the UTM coordinate system
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32655)

    # Set the geotransform and projection to the image
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(srs.ExportToWkt())
    # Save the image
    ds = None