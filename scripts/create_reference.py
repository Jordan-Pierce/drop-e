import os.path
import sys
import glob
import geopandas as gpd


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
    gdf_['Altitude'] = 0

    # Output file
    out_file = file.replace(".gpkg", ".csv")

    # Save the csv
    gdf_.to_csv(out_file, index=False)

    if os.path.exists(out_file):
        print("Created Reference: ", out_file)
    else:
        print("ERROR: Could not create Reference: ", out_file)
        out_file = None

    return out_file

