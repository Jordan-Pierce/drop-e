import shutil
import numpy as np
import pandas as pd
from osgeo import gdal, osr


def georeference_ortho(src_data, dst_data, geo_path):
    """Takes in the path to the orthomosaic, the path to the output
    georeferenced orthomosaic, and the path to the image_centroids.csv file.
    Applies the GCPs to the orthomosaic and saves it as the output path."""

    # Path the georeferencing information
    df = pd.read_csv(geo_path, index_col=0)

    # Create a copy of the original file and save it as the output filename:
    shutil.copy(src_data, dst_data)

    # Open the output file for writing for writing:
    ds = gdal.Open(dst_data, gdal.GA_Update)

    # Set spatial reference:
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(32655)

    gcps = df[['Easting', 'Northing', 'Altitude', 'x_pixels', 'y_pixels']]
    gcps = gcps.dropna().drop_duplicates().values

    gdal_gcps = []
    for cp in gcps:
        try:
            e, n = np.around(cp[0:2], 0)
            x, y = int(cp[3]), int(cp[4])
            gdal_gcps.append(gdal.GCP(e, n, 0, x, y))
        except:
            pass

    # Select a subset of the GCPs
    select_list = [0, len(gdal_gcps)//2, -1]
    gdal_gcps = [gdal_gcps[i] for i in select_list]

    print("Number of control points: ", len(gdal_gcps))

    # Apply the GCPs to the open output file
    ds.SetGCPs(gdal_gcps, sr.ExportToWkt())

    ds = None


if __name__ == "__main__":

    src_path = "C://Users/jordan.pierce/Downloads/Orthomosaic.tif"
    dst_path = "C://Users/jordan.pierce/Downloads/Geo_Orthomosaic_3.tif"
    geo_path = "C://Users/jordan.pierce/Downloads/image_centroids.csv"

    georeference_ortho(src_path, dst_path, geo_path)