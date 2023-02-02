import os
import math
import warnings
import pint

from datetime import datetime
import dateutil

from PIL import Image as Image
from exif import Image as ExifImage

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

import pandas as pd
import geopandas as gpd
import movingpandas as mpd
from movingpandas.trajectory_smoother import KalmanSmootherCV

from shapely.geometry import Point, LineString
import folium
import utm

import rasterio
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint


# this is needed to supress a redundant warning from movingpandas
warnings.filterwarnings("ignore", message="CRS not set for some of the concatenation inputs.")

allowed_img_ext = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

res_unit_lookup = {"1": "None", 2: "Inch", 3: "Centimeter"}

ureg = pint.UnitRegistry()


def utm_epsg_from_latlot(lat, lon):
    # source: https://github.com/Turbo87/utm/issues/51
    zone = utm.from_latlon(lat, lon)[2]
    return f"326{zone:02d}" if lat >= 0 else f"327{zone:02d}"


def dms_to_dd(dms, direction, verbose=False) -> float:
    """
    Converts a lat or long in Degree/Minutes/Seconds format to Decimal Degrees.

    Parameters:
        - dms (list): A list of 3 floats, representing the Degree, Minutes, and Seconds.
        - direction (str): Either 'N' or 'S' for latitude, or 'E' or 'W' for longitude.
        - verbose (bool): If True, prints the inputs and resulting outputs to the
            console. If false (default), this function does not print anything.

    Returns:
        - dd (float): The lat or long in Decimal Degrees.
    """

    degrees, minutes, seconds = dms

    dd = float(degrees) + float(minutes)/60 + float(seconds)/(3600)

    if direction == 'S' or direction == 'W':
        dd *= -1

    if verbose:
        print(
            f"DMS_TO_DD VERBOSE | Input: {degrees}° {minutes}' {seconds}'' {direction} "
            f"| Output: {dd} of type: {type(dd)}"
        )

    return dd


def rotate_point_3d(point, angle, axis):
    # RADIANS ONLY!
    x, y, z = point
    c, s = math.cos(angle), math.sin(angle)

    if axis == 'x':
        return x, y*c - z*s, y*s + z*c
    elif axis == 'y':
        return x*c + z*s, y, -x*s + z*c
    elif axis == 'z':
        return x*c - y*s, x*s + y*c, z
    else:
        raise ValueError("Invalid axis")


class TowLine:
    def __init__(
        self, img_dir, out_dir, usbl_path="None", datetime_field="DateTime",
        process_noise_std=1.0, measurement_noise_std=0.25
    ):

        self.img_dir = img_dir
        self.out_dir = out_dir
        self.usbl_path = usbl_path

        self.datetime_field = datetime_field
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std

        self.img_gdf = None  # the default primary gdf for imagery
        self.fit_img_gdf = None  # if USBL is available, the primary gdf for imagery

        # only one will be populated...
        self.raw_usbl_df = None
        self.smooth_usbl_df = None

        # only one will be populated...
        self.raw_usbl_traj = None
        self.smooth_usbl_traj = None

        self.max_gsd = 0.0

        # Step 1: Read all images in img_dir, extract relevant EXIF data (geoexif),
        # and build a GeoPandasDataFrame (gdf) from this information...
        self.build_img_gdf()

        # Step 2: If a higher-order source of location data is available (usbl), then
        # we will read that as a separate gdf, perform some optional filtering, extract
        # trajectory info, and fit the img_gdf to this new trajectory...
        if self.usbl_path is not None:
            self.build_usbl_gdf()

            self.calc_trajectory(self.usbl_gdf, process_noise_std, measurement_noise_std)

            self.fit_to_usbl()

        # ... otherwise, just calculate the trajectory from the img_gdf and proceed
        else:
            self.calc_trajectory(self.img_gdf, process_noise_std, measurement_noise_std)

        # Step 3: Round out the internal / external orientation parameters needed for
        # georeferencing OR orthorectification, and perform the operation.

        self.apply_gsd(self.fit_gdf)

        self.max_gsd = self.fit_gdf.GSD.max()
        print(f"Average GSD: {self.fit_gdf.GSD.mean()}")
        print(f"Max GSD: {self.max_gsd}")
        # self.calc_affine()

        # self.georeference()
        # self.orthorectify()

        # TODO: plotting / metadata
        # self.write_external_params()
        # self.write_internal_params()
        # self.preview_images()
        # self.plot_traj_smoothing()
        # self.plot_usbl_fit()
        # self.traj_plot_3d()

        # TODO: writing images / deltas
        # if write_images == True:
        #    self.write_georef()
        #    self.write_ortho()
        #    self.write_gdfs()

    """DATA INGEST FCNS"""
    def build_img_gdf(self):
        geoexifs = []
        for img in os.listdir(self.img_dir):
            if img.endswith(allowed_img_ext):
                geoexif = self._extract_img_exif(img)

                geoexifs.append(geoexif)

        epsg = utm_epsg_from_latlot(geoexifs[0]['GPS_Latitude_DD'], geoexifs[0]['GPS_Longitude_DD'])

        img_df = pd.DataFrame(geoexifs)

        # convert img_df to geodataframe, index by timestamp
        img_gdf = gpd.GeoDataFrame(
            img_df,
            geometry=gpd.points_from_xy(img_df.UTM_Easting, img_df.UTM_Northing),
            crs=epsg
        )

        img_gdf.index = img_gdf['DateTime']

        # TODO: how to reliably filter the "whiteboard images" w/o USBL timestamps?
        # IDEA: timedelta from movingpandas trajectory, reverse sort, and drop under
        # a certain threshold (or percentile). This assumes whiteboards are always first
        # in the sequence, which is a safe assumption for now, but maybe add reverse too?

        self.img_gdf = img_gdf

        #print(self.img_gdf.dtypes)
        #print(self.img_gdf.head(5))
        self._write_gdf(self.img_gdf, "img_gdf")

    def _extract_img_exif(self, img):
        print(f"Reading GeoExif for {img}")

        img_path = os.path.join(self.img_dir, img)
        exif_dict = {}

        # get image dimensions
        with open(img_path, 'rb') as f:
            img_pil = Image.open(f)
            img_w, img_h = img_pil.size
            exif_dict['Pixel_X_Dimension'] = img_w
            exif_dict['Pixel_Y_Dimension'] = img_h

        # get exif data
        with open(img_path, 'rb') as f:
            img_exif = ExifImage(f)

            exif_dict["img_path"] = img_path
            exif_dict["img_name"] = img

            exif_dict['Focal_Length_mm'] = img_exif.get('Focal_Length')

            exif_dict['Focal_Plane_X_Resolution'] = img_exif.get('focal_plane_x_resolution')
            exif_dict['Focal_Plane_Y_Resolution'] = img_exif.get('focal_plane_y_resolution')
            exif_dict['Focal_Plane_Resolution_Unit'] = str.lower(
                res_unit_lookup[img_exif.get('focal_plane_resolution_unit')]
            )

            # Estimate the Sensor Width, Height, and Pixel Pitch
            if exif_dict['Focal_Plane_X_Resolution'] is not None and exif_dict['Focal_Plane_Y_Resolution'] is not None:
                exif_dict['Est_Sensor_Width'] = exif_dict['Pixel_X_Dimension'] / exif_dict['Focal_Plane_X_Resolution']
                exif_dict['Est_Sensor_Height'] = exif_dict['Pixel_Y_Dimension'] / exif_dict['Focal_Plane_Y_Resolution']
                exif_dict['Est_Sensor_HW_Unit'] = str.lower(exif_dict['Focal_Plane_Resolution_Unit'])

                exif_dict['Pixel_X_Pitch'] = 1 / exif_dict['Focal_Plane_X_Resolution']
                exif_dict['Pixel_Y_Pitch'] = 1 / exif_dict['Focal_Plane_Y_Resolution']
                exif_dict['Pixel_Pitch_Unit'] = exif_dict['Focal_Plane_Resolution_Unit']

            # Infer DateTime from Exif
            exif_dict['DateTime_Original'] = img_exif.get('datetime_original')
            exif_dict["DateTime"] = dateutil.parser.parse(exif_dict['DateTime_Original'])
            exif_dict["Minute"] = exif_dict["DateTime"].minute
            exif_dict["Second"] = exif_dict["DateTime"].second

            # Pull GPS tags in Lat/Long
            exif_dict['GPS_Latitude_DMS'] = img_exif.get('GPS_Latitude')
            exif_dict['GPS_Latitude_Ref'] = img_exif.get('GPS_Latitude_Ref')
            exif_dict['GPS_Longitude_DMS'] = img_exif.get('GPS_Longitude')
            exif_dict['GPS_Longitude_Ref'] = img_exif.get('GPS_Longitude_Ref')

            # Convert Lat/Long to Decimal Degrees, infer UTM Coordinates and Zone
            if exif_dict['GPS_Latitude_DMS'] is not None and exif_dict['GPS_Longitude_DMS'] is not None:
                exif_dict['GPS_Latitude_DD'] = dms_to_dd(
                    exif_dict['GPS_Latitude_DMS'], exif_dict['GPS_Latitude_Ref']
                )

                exif_dict['GPS_Longitude_DD'] = dms_to_dd(
                    exif_dict['GPS_Longitude_DMS'], exif_dict['GPS_Longitude_Ref']
                )

                east1, north1, zone, zoneLetter = utm.from_latlon(
                    exif_dict['GPS_Latitude_DD'], exif_dict['GPS_Longitude_DD']
                )

                exif_dict['UTM_Easting'] = east1
                exif_dict['UTM_Northing'] = north1
                exif_dict['UTM_Zone_Estimated'] = str(zone) + zoneLetter
                exif_dict['UTM_EPSG_Estimated'] = utm_epsg_from_latlot(
                exif_dict['GPS_Latitude_DD'], exif_dict['GPS_Longitude_DD']
                )
            else:
                print(f"WARNING: No GPS data found in {img}.")

            # pull GPS Altitude data
            exif_dict['GPS_Altitude'] = img_exif.get('GPS_Altitude')
            exif_dict['GPS_Altitude_Ref'] = img_exif.get('GPS_Altitude_Ref')

        return exif_dict

    def build_usbl_gdf(
        self, pdop_field="Max_PDOP", filter_quartile=0.95
    ):
        # read USBL data into GeoPandas
        usbl_gdf = gpd.read_file(self.usbl_path)

        # filter outliers by keeping lower quartile of PDOP values
        max_pdop = usbl_gdf[pdop_field].quantile(filter_quartile)
        usbl_gdf = usbl_gdf[usbl_gdf[pdop_field] < max_pdop]

        # TODO: metadata on how many were filtered...
        print(f"filter_quartile set to {filter_quartile}, this will filter all PDOP \
            values below {max_pdop}")

        # parse datetime, set as index
        usbl_gdf["datetime_field"] = usbl_gdf[self.datetime_field].apply(
            lambda x: dateutil.parser.parse(x)
        )
        usbl_gdf.index = usbl_gdf["datetime_field"]

        self.usbl_gdf = usbl_gdf

    """TRAJECTORY FCNS"""
    def calc_trajectory(self, pt_gdf, process_noise_std=1.0, measurement_noise_std=0.25):
        # calculate trajectory information with MovingPandas
        traj = mpd.Trajectory(pt_gdf, 1, t="datetime_field", crs=pt_gdf.crs)

        traj.add_direction()
        traj.add_speed()
        traj.add_distance(name="mpd_distance")
        # traj.add_timedelta()

        self.raw_usbl_traj = traj
        self.raw_usbl_df = traj.to_point_gdf()

        # Smooth trajectories...
        # TODO: experiment with best methods, make smoothing optional.
        # if process_noise_std is not equal to 0.0, smoothing is applied
        if process_noise_std != 0.0 or measurement_noise_std != 0.0:
            s_traj = KalmanSmootherCV(traj).smooth(
                process_noise_std=process_noise_std,
                measurement_noise_std=measurement_noise_std)

            s_traj.add_direction(overwrite=True)
            s_traj.add_direction(overwrite=True)
            s_traj.add_speed(overwrite=True)
            s_traj.add_distance(overwrite=True, name="mpd_distance")
            # traj.add_timedelta(overwrite=True, name="mpd_timedelta")

            self.smooth_usbl_traj = s_traj
            self.smooth_usbl_df = s_traj.to_point_gdf()

        else:
            print(f"Either process_noise_std or measurement_noise_std is set to 0.0, \
                therefore no smoothing will be applied to trackline.")

    def _zLookup(self, img_gdf, usbl_gdf, datetime_field='DateTime', z_field='CameraZ'):
        # get a list of datetimes every second between start and end:
        start = usbl_gdf[datetime_field].min()
        stop = usbl_gdf[datetime_field].max()
        dt_list = pd.date_range(start, stop, freq='S')

        # merge dt_list with usbl_gdf, fill in missing values with NaN, keep only CaAltCor_m field
        usbl_dt_gdf = pd.merge(dt_list.to_series(name='time_range'), usbl_gdf, how='left', left_index=True, right_index=True)
        usbl_dt_gdf = usbl_dt_gdf[['time_range', z_field, datetime_field]]

        # perform linear interpolation to fill in missing Z values
        usbl_dt_gdf[z_field].interpolate(method='linear', inplace=True)

        # use usbl_dt_gdf as a lookup table to add CaAltCor_m to img_gdf
        img_gdf[z_field] = img_gdf['DateTime'].map(usbl_dt_gdf.set_index('time_range')[z_field])

        return img_gdf

    def fit_to_usbl(self):
        # create a copy of the original gdf to work with...
        new_gdf = self.img_gdf.copy()
        print(new_gdf.head())
        # TODO: Smooth or raw?
        # TODO: filter Time outside of USBL range...

        # TODO: Fit Images to USBL using DateTime
        new_gdf['Improved_Position'] = new_gdf.apply(
            lambda row: self.smooth_usbl_traj.get_position_at(
                row.DateTime, method='interpolated'), axis=1)

        # store the vector lines represenintg the  "delta" between each EXIF point and
        # the USBL location at that datetime. These are strictly used for plotting.
        delta_gdf = new_gdf[new_gdf.geometry.is_empty == False].copy()

        delta_gdf["Delta_LineString"] = delta_gdf.apply(
            lambda row: LineString([row.geometry, row.Improved_Position]), axis=1)

        delta_gdf["Delta_LineLength"] = delta_gdf.apply(
            lambda row: row.Delta_LineString.length, axis=1)

        print(f"Len filt / new: {len(delta_gdf)}, {len(new_gdf)}")
        print(delta_gdf['Delta_LineLength'].describe())

        new_gdf.geometry = new_gdf.Improved_Position
        new_gdf.drop(columns=['Improved_Position'], inplace=True)

        # Set the delta_gdf geometry to linestring
        delta_gdf.geometry = delta_gdf.Delta_LineString  # TODO: clean this gdf up?
        delta_gdf.drop(columns=['Delta_LineString', 'Improved_Position'], inplace=True)

        # TODO: log metadata about shift distance, etc. to flag potential issues.

        # TODO: fit Z info too
        if self.smooth_usbl_df is not None:
            z_gdf = self._zLookup(new_gdf, self.smooth_usbl_df, z_field='CamAlt', datetime_field='TimeCor') # TODO: fix hardcodes
        else:
            z_gdf = self._zLookup(new_gdf, self.raw_usbl_df, z_field='CamAlt', datetime_field='TimeCor')

        self.delta_gdf = delta_gdf
        self.fit_gdf = z_gdf

    """GEOREF FCNS"""
    def apply_gsd(self, in_gdf):
        # Extract the ground spacing distance from each row of the fit_gdf
        in_gdf["GSD"] = in_gdf.apply(
            lambda row: self._calc_gsd(row), axis=1)


    def _calc_gsd(self, row, z_field='CamAlt'):  #TODO: hardcode CamAlt
        # calculate the ground spacing distance (GSD) for each image
        H = row[z_field]
        F = row.Focal_Length_mm
        img_w = row.Pixel_X_Dimension
        img_h = row.Pixel_Y_Dimension
        sensor_w = ureg.convert(row.Est_Sensor_Width, row.Est_Sensor_HW_Unit, "mm")
        sensor_h = ureg.convert(row.Est_Sensor_Height, row.Est_Sensor_HW_Unit, "mm")

        gsd_w = (H * sensor_w) / (F * img_w)
        gsd_h = (H * sensor_h) / (F * img_h)

        print(f"gsd_w: {gsd_w}, gsd_h: {gsd_h}, returning: {max(gsd_w, gsd_h)}")

        img_width = (H * sensor_w) / F
        img_height = (H * sensor_h) / F
        print(f"img_w/h: {img_width} / {img_height} meters")

        # return the greater (coarser) value of gsd_w and gsd_h as float
        return max(gsd_w, gsd_h)


    """PLOTTING + WRITING FCNS"""
    def _write_gdf(self, target_gdf, basename, format="GPKG", index=False):
        # TODO: this is a patch because writing tuples is a no-no. Need long-term fix...
        target_gdf.drop(['GPS_Latitude_DMS', 'GPS_Latitude_Ref', 'GPS_Longitude_DMS', 'GPS_Longitude_Ref'], axis=1, inplace=True, errors='ignore')

        if format == "GPKG":
            out_path = os.path.join(self.out_dir, f"{basename}.gpkg")
        elif format == "ESRI Shapefile":
            out_path = os.path.join(self.out_dir, f"{basename}.shp")
        else:
            raise ValueError(f"Invalid geospatial format: {format}. Must be GPKG or ESRI Shapefile.")

        target_gdf.to_file(
            out_path, driver=format, index=index
        )

    def dump_gdfs(self):
        if self.fit_gdf is not None:
            self._write_gdf(self.fit_gdf, "fit_gdf", format="GPKG", index=False)

        if self.img_gdf is not None:
            self._write_gdf(self.img_gdf, "img_gdf", format="GPKG", index=False)

        if self.delta_gdf is not None:
            self._write_gdf(self.delta_gdf, "delta_gdf", format="GPKG", index=False)

        if self.raw_usbl_df is not None:
            self._write_gdf(self.raw_usbl_df, "raw_usbl_df", format="GPKG", index=False)

        if self.smooth_usbl_df is not None:
            self._write_gdf(self.smooth_usbl_df, "smooth_usbl_df", format="GPKG", index=False)

    def plot_smoothing_operation(self, save_fig=False):
        f, ax = plt.subplots()
        if self.raw_usbl_traj is not None:
            self.raw_usbl_traj.plot(ax=ax, color='red', label="Raw Trajectory", legend=True)
        else:
            print("No raw USBL trajectory to plot. Skipping...")
        if self.smooth_usbl_traj is not None:
            self.smooth_usbl_traj.plot(ax=ax, column="speed", cmap="viridis", label="Smoothed Trajectory", legend=True)

        plt.title("Smoothed USBL Trajectory")

        plt.xlabel("UTM Easting (m)")
        plt.ylabel("UTM Northing (m)")

        plt.legend()
        plt.text(1.0, 1.0, f"Process Noise: {self.process_noise_std} \n Measurement Noise: {self.measurement_noise_std}")

        # plt.show()

        if save_fig is True:
            plt.savefig(os.path.join(self.out_dir, "smooth_op_plot.png"))

    def plot_usbl_fit(self, save_fig=False):
        f, ax = plt.subplots()

        # plot the fit points...
        if self.fit_gdf is None:
            print(f"fit_gdf not found. Please run fit_usbl_to_exif() first.")

        else:
            # plot the fit points...
            self.fit_gdf.plot(ax=ax, color='green', label="Img. USBL Locations", legend=True)

            # plot the original points...
            self.img_gdf.plot(ax=ax, color='red', label="Img. Exif Locations", legend=True)

            # plot the deltas...
            self.delta_gdf.plot(ax=ax, color='blue', label="Shifts", legend=True)

            # only plot the best usbl trackline...
            if self.smooth_usbl_traj is not None:
                self.smooth_usbl_traj.plot(ax=ax, column="speed", cmap="viridis", label='Smoothed Trackline', legend=True)
            else:
                self.raw_usbl_traj.plot(ax=ax, color='red', label="Raw Trackline", legend=True)

        plt.text(0, 0, f"Average shift: {self.delta_gdf['Delta_LineLength'].mean()} meters")
        plt.title("EXIF Points to USBL Trajectory Fit")

        plt.xlabel("UTM Easting (m)")
        plt.ylabel("UTM Northing (m)")

        plt.legend()

        # TODO: label deltas, print delta metrics.

        # plt.show()
        if save_fig is True:
            plt.savefig(os.path.join(self.out_dir, "plot_usbl_fit.png"))