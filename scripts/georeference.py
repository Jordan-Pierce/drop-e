import os
import sys
import glob
import shutil

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from osgeo import gdal, osr
gdal.SetConfigOption('GDAL_TIFF_INTERNAL_MASK', 'YES')
gdal.SetConfigOption('GDAL_TIFF_OVR_BLOCKSIZE', '512')

import geopandas as gpd
from shapely.geometry import LineString
import movingpandas as mpd
from movingpandas.trajectory_smoother import KalmanSmootherCV

import warnings
warnings.filterwarnings("ignore", message="CRS not set for some of the concatenation inputs.")


# ----------------------------------------------------------------------------------------------------------------------
# Code for calculating Rotation Angle (Radians) using subset of aligned, georeferenced images
# ----------------------------------------------------------------------------------------------------------------------

def _build_usbl_gdf(site_gpkg, datetime_field='DateTime', pdop_field=None, filter_quartile=0.95) -> gpd.GeoDataFrame:
    """
    Given a path to a geospatial vector point file (e.g. shapefile, geopackage, etc),
    read the file into a GeoPandas GeoDataFrame and optionally filter outliers based
    on a precision (PDOP) field.
    """
    # read USBL data into GeoPandas
    usbl_gdf = site_gpkg.copy()

    # filter outliers by keeping lower quartile of PDOP values
    if pdop_field is not None:
        max_pdop = usbl_gdf[pdop_field].quantile(filter_quartile)
        print(type(max_pdop), max_pdop)

        if not math.isnan(max_pdop):
            count = len(usbl_gdf) - len(usbl_gdf[usbl_gdf[pdop_field] < max_pdop])
            usbl_gdf = usbl_gdf[usbl_gdf[pdop_field] < max_pdop]

            print(f"--filter_quartile of {filter_quartile} allows a max PDOP of {max_pdop}.")
            print(f"{count} USBL pings were filtered based on their {pdop_field} field.")
        else:
            print(f"WARNING: No valid PDOP values found in column {pdop_field}. No filtering will be performed.")
    else:
        print("WARNING: No PDOP field specified. No filtering will be performed.")

    # standardize the datetime field, and set that as the index
    # usbl_gdf["datetime_idx"] = usbl_gdf[datetime_field].apply(
    #     lambda x: datetime.strptime(re.sub('[/.:]', '-', x), '%Y-%m-%d %H-%M-%S')
    # )
    # usbl_gdf.index = usbl_gdf['datetime_idx']
    usbl_gdf.index = usbl_gdf['DateTime']

    return usbl_gdf


def _calc_trajectory(pt_gdf, process_noise_std=1.0, measurement_noise_std=0.25):
    """
    Given a GeoPandas GeoDataFrame of points, calculate the trajectory with
    MovingPandas. Optionally smooth the trajectory with a Kalman filter.
    """
    # calculate trajectory information with MovingPandas
    traj = mpd.Trajectory(pt_gdf, 1, t="datetime_idx", crs=pt_gdf.crs)

    # Smooth trajectories... smoothing is skipped if either std is equal to 0.0
    if process_noise_std != 0.0 or measurement_noise_std != 0.0:
        traj = KalmanSmootherCV(traj).smooth(
            process_noise_std=process_noise_std,
            measurement_noise_std=measurement_noise_std)
    else:
        print(f"Either process_noise_std or measurement_noise_std is set to 0.0, \
            therefore no smoothing will be applied to trackline.")

    traj.add_direction(overwrite=True, name="Direction")
    traj.add_speed(overwrite=True, name="Speed")
    # s_traj.add_acceleration(overwrite=True, name="Acceleration")
    traj.add_distance(overwrite=True, name="Distance")
    # s_traj.add_timedelta(overwrite=True, name="TimeDelta")

    return traj


def global_rotation(site_gpkg, output_folder=None, datetime_field="DateTime",
                    pdop_field=None, filter_quartile=0.95, process_noise_std=1.0,
                    measurement_noise_std=0.25, reversed=False) -> float:
    """
    Given a GeoPandas GeoDataFrame of points, use the XY coordinates and
    datetime stamps to calculate a trajectory and derive the global rotation
    angle (direction). The direction is given in degrees, clockwise from North.
    """
    # Build the geodataframe from usbl
    pt_gdf = _build_usbl_gdf(
        site_gpkg,
        datetime_field=datetime_field,
        pdop_field=pdop_field,
        filter_quartile=filter_quartile,
    )
    # Get the trajectory
    trajectory = _calc_trajectory(
        pt_gdf,
        process_noise_std=process_noise_std,
        measurement_noise_std=measurement_noise_std,
    )

    # Get the direction
    direction = trajectory.get_direction()
    start = trajectory.get_start_location()
    end = trajectory.get_end_location()

    print(f"Start: {start}\nEnd: {end}")

    # Reverse if needed
    if reversed:
        start, end = end, start
        direction -= 180

    # Create a direction line, given the start and end points
    direction_line = LineString([start, end])
    dirline_gdf = gpd.GeoDataFrame(geometry=[direction_line], crs=pt_gdf.crs)

    # plot the outputs
    f, ax = plt.subplots()
    trajectory.to_line_gdf().plot(ax=ax)
    dirline_gdf.plot(ax=ax, color="red")
    f.legend(["Trajectory", "Direction"])
    plt.scatter(start.x, start.y, c='green', s=20)
    plt.scatter(end.x, end.y, c='red', s=20)
    plt.title(f"Trajectory / Direction: {str(direction)}")

    # Save if needed
    if output_folder:
        plt.savefig(f"{output_folder}Direction.png")

    plt.show()

    return np.deg2rad(direction)


# Ignore this, just testing
def progress_callback(complete, message, data):
    sys.stdout.write('\rGeorectification Progress: {:.1%}  {}'.format(complete, message))
    sys.stdout.flush()


# Ignore this, just testing
def calculate_scale(src_path, georeference_path):

    # Path the georeferencing information, drop not aligned
    df = pd.read_csv(georeference_path, index_col=0)
    df = df[['Easting', 'Northing', 'x_pixels', 'y_pixels']].dropna()

    # Extract the features (pixel values) and targets (Easting, Northing coordinates)
    X = df[['x_pixels', 'y_pixels']].values
    y = df[['Easting', 'Northing']].values

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(X, y)

    # Create temporary files
    tmp_1_path = src_path.replace(".", "_temp_1.")
    tmp_2_path = src_path.replace(".", "_temp_2.")
    shutil.copyfile(src_path, tmp_1_path)

    # Open the source dataset and add GCPs to it
    src_ds = gdal.OpenShared(str(tmp_1_path), gdal.GA_Update)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32655)
    src_ds.SetProjection(srs.ExportToWkt())

    # Get the height and width of the image
    width = src_ds.RasterYSize
    height = src_ds.RasterXSize

    # Calculate the coordinates of the four corners and the center
    top_left = (0, 0)
    bottom_left = (0, height)
    bottom_right = (width, height)
    top_right = (width, 0)
    center = (width // 2, height // 2)

    pixels = np.array([center,
                       top_left,
                       bottom_left,
                       bottom_right,
                       top_right])

    # Convert pixel coordinates to Easting and Northing using the linear regression model
    easting, northing = model.predict(pixels).T

    gcps = []

    # Iterate over each row in the DataFrame
    for i, pixel_coord in enumerate(pixels):
        # Create a new GCP object
        gcp = gdal.GCP()
        gcp.GCPX = easting[i]
        gcp.GCPY = northing[i]
        gcp.GCPZ = 0.0
        gcp.GCPPixel = int(pixel_coord[0])
        gcp.GCPLine = int(pixel_coord[1])

        # Add the GCP to the list
        gcps.append(gcp)

    gcp_srs = osr.SpatialReference()
    gcp_srs.ImportFromEPSG(32655)
    gcp_crs_wkt = gcp_srs.ExportToWkt()
    src_ds.SetGCPs(gcps, gcp_crs_wkt)

    # Define target SRS
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(32655)
    dst_wkt = dst_srs.ExportToWkt()

    error_threshold = 0.5  # error threshold --> use same value as in gdalwarp
    resampling = gdal.GRA_NearestNeighbour

    # Call AutoCreateWarpedVRT() to fetch default values for target raster dimensions and geotransform
    tmp_ds = gdal.AutoCreateWarpedVRT(src_ds,
                                      None,
                                      dst_wkt,
                                      resampling,
                                      error_threshold)

    dst_xsize = tmp_ds.RasterXSize
    dst_ysize = tmp_ds.RasterYSize
    dst_gt = tmp_ds.GetGeoTransform()
    tmp_ds = None

    # Now create the true target dataset
    dst_ds = gdal.GetDriverByName('GTiff').Create(tmp_2_path,
                                                  dst_xsize,
                                                  dst_ysize,
                                                  src_ds.RasterCount)

    dst_ds.SetProjection(dst_wkt)
    dst_ds.SetGeoTransform(dst_gt)
    dst_ds.GetRasterBand(1).SetNoDataValue(0)

    # And run the reprojection
    gdal.ReprojectImage(src_ds,
                        dst_ds,
                        None,  # src_wkt : left to default value --> will use the one from source
                        None,  # dst_wkt : left to default value --> will use the one from destination
                        resampling,
                        0,  # WarpMemoryLimit : left to default value
                        error_threshold,
                        progress_callback,  # Progress callback : function that outputs progress
                        None)  # Progress callback user data
    dst_ds = None
    src_ds = None

    tmp_ds = gdal.OpenShared(str(tmp_2_path), gdal.GA_ReadOnly)
    x_origin, x_res, _, y_origin, _, y_res = tmp_ds.GetGeoTransform()
    tmp_ds = None

    # Delete these, not needed
    os.remove(tmp_1_path)
    os.remove(tmp_2_path)

    return np.abs(x_res)

# ----------------------------------------------------------------------------------------------------------------------
# Working code for taking in orthomosaic, subsetted gpkg, and georeferencing orthomosaic (copy)
# ----------------------------------------------------------------------------------------------------------------------

def georeference_orthomosaic(src_path, dst_path, georeference_path):

    # Get the pixel size by creating a temp georeference tif with GCPs
    pixel_size = 0.0002 # calculate_scale(src_path, georeference_path)

    # Path the georeferencing information, drop not aligned
    df = pd.read_csv(georeference_path, index_col=0)
    df = df[['Easting', 'Northing', 'x_pixels', 'y_pixels']].dropna()

    # Check if horizontal flip is needed
    min_x_pixel_row = df[df['x_pixels'] == df['x_pixels'].min()]
    max_x_pixel_row = df[df['x_pixels'] == df['x_pixels'].max()]

    # Determine the x resolution sign
    if min_x_pixel_row['Easting'].item() > max_x_pixel_row['Easting'].item():
        print("Horizontal flip is needed")
        x_origin = df['Easting'].max()
        x_res = pixel_size * -1
    else:
        print("No horizontal flip is needed")
        x_origin = df['Easting'].min()
        x_res = pixel_size * 1

    # Check if vertical flip is needed
    min_y_pixel_row = df[df['y_pixels'] == df['y_pixels'].min()]
    max_y_pixel_row = df[df['y_pixels'] == df['y_pixels'].max()]

    # Determine the y resolution sign
    if min_y_pixel_row['Northing'].item() < max_y_pixel_row['Northing'].item():
        print("Vertical flip is needed")
        y_origin = df['Northing'].min()
        y_res = pixel_size * 1
    else:
        print("No vertical flip is needed")
        y_res = pixel_size * -1
        y_origin = df['Northing'].max()

    print("x_origin:", x_origin, "\tx_res:", x_res)
    print("y_origin:", y_origin, "\ty_res:", y_res)

    # Geotransform to be added to header
    transform = [
        x_origin,
        x_res,
        0.0,
        y_origin,
        0.0,
        y_res,
    ]

    print(transform)

    # Make copy of the original orthomosaic
    shutil.copyfile(src_path, dst_path)

    # Open the source dataset and geotransform
    dst_ds = gdal.OpenShared(str(dst_path), gdal.GA_Update)
    dst_ds.SetGeoTransform(transform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32655)
    dst_ds.SetProjection(srs.ExportToWkt())

    dst_ds = None

    print("Done.")
