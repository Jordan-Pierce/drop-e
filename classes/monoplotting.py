import os
import re
import math
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

from shapely.geometry import Point, LineString, Polygon
import folium
import utm

import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint as GCP

import warnings
warnings.filterwarnings("ignore", message="CRS not set for some of the concatenation inputs.")

allowed_img_ext = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

res_unit_lookup = {1: "None", 2: "Inch", 3: "Centimeter", None: "None" }

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


class TowLine:
    def __init__(
        self, img_dir, out_dir, usbl_path="None", datetime_field="DateTime",
        pdop_field=None, alt_field="CaAltDepth", filter_quartile=0.95,
        process_noise_std=1.0, measurement_noise_std=0.25, preview_mode=False
    ):
        self.img_dir = img_dir
        self.out_dir = out_dir
        self.usbl_path = usbl_path

        self.datetime_field = datetime_field
        self.pdop_field = pdop_field
        self.alt_field = alt_field

        self.filter_quartile = filter_quartile
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.preview_mode = preview_mode

        self.img_gdf = None  # the default primary gdf for imagery
        self.fit_gdf = None  # if USBL is available, the primary gdf for imagery
        self.transforms = {}

        # only one will be populated...
        self.raw_usbl_df = None
        self.smooth_usbl_df = None

        self.delta_gdf = None

        # only one will be populated...
        self.raw_usbl_traj = None
        self.smooth_usbl_traj = None
        self.smooth_usbl_traj_line =  None

        self.epsg_str = None
        self.max_gsd_mode = 0.0
        self.max_gsd = 0.0
        self.min_gsd = 0.0
        self.datetime_min = None
        self.datetime_max = None

        # Step 1: Read all images in img_dir, extract relevant EXIF data (geoexif),
        # and build a GeoPandasDataFrame (gdf) from this information...
        self.build_img_gdf()

        # Step 2: If a higher-order source of location data is available (usbl), then
        # we will read that as a separate gdf, perform some optional filtering, extract
        # trajectory info, and fit the img_gdf to this new trajectory...
        if self.usbl_path is not None:
            self.build_usbl_gdf(
                pdop_field=self.pdop_field, filter_quartile=self.filter_quartile
            )

            self.calc_trajectory(
                self.usbl_gdf, process_noise_std, measurement_noise_std
            )

            self.fit_to_usbl()

            self.apply_gsd(self.fit_gdf)

        # ... otherwise, just calculate the trajectory from the img_gdf and proceed
        else:
            self.calc_trajectory(self.img_gdf, process_noise_std, measurement_noise_std)

        # Step 3: Round out the internal / external orientation parameters needed for
        # georeferencing OR orthorectification, and perform the operation.

        if self.fit_gdf is not None:
            self.orient_images(self.fit_gdf)
            if not self.preview_mode:
                self.georeference_images(self.fit_gdf)

        else:
            self.orient_images(self.img_gdf)
            if not self.preview_mode:
                self.georeference_images(self.img_gdf)

    """DATA INGEST FCNS"""
    def build_img_gdf(self):
        # TODO: how to reliably filter the "whiteboard images" w/o USBL timestamps?
        # IDEA: timedelta from movingpandas trajectory, reverse sort, and drop under
        # a certain threshold (or percentile). This assumes whiteboards are always first
        # in the sequence, which is a safe assumption for now, but maybe add reverse too?

        imgs = [i for i in os.listdir(self.img_dir) if i.endswith(allowed_img_ext)]
        print(f"Found {len(imgs)} images in {self.img_dir}...")

        geoexifs = []
        for img in imgs:
            geoexif = self._extract_img_exif(img)
            geoexifs.append(geoexif)


        if geoexifs[0]['GPS_Latitude_DMS'] is not None and geoexifs[0]['GPS_Longitude_DMS'] is not None:
            epsg = utm_epsg_from_latlot(geoexifs[0]['GPS_Latitude_DD'], geoexifs[0]['GPS_Longitude_DD'])
            self.epsg_str = f"epsg:{epsg}"

            img_df = pd.DataFrame(geoexifs)

            # convert img_df to geodataframe, index by timestamp, use EXIF as geometry
            img_gdf = gpd.GeoDataFrame(
                img_df,
                geometry=gpd.points_from_xy(img_df.UTM_Easting, img_df.UTM_Northing),
                crs=epsg
            )
            img_gdf.index = img_gdf['DateTime']

            self.img_gdf = img_gdf

        else:
            print(f"EPSG can not be inferred from image EXIF data...")

            img_df = pd.DataFrame(geoexifs)
            img_df.index = img_df['DateTime']

            self.img_gdf = img_df

    def _extract_img_exif(self, img):
        #print(f"Reading GeoExif for {img}")

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

            exif_dict['Focal_Length'] = img_exif.get('Focal_Length')
            exif_dict['Focal_Length_Unit'] = "millimeters"

            exif_dict['Focal_Plane_X_Resolution'] = img_exif.get('focal_plane_x_resolution')
            exif_dict['Focal_Plane_Y_Resolution'] = img_exif.get('focal_plane_y_resolution')
            print(img_exif.get('focal_plane_resolution_unit'))
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
            #exif_dict["DateTime"] = dateutil.parser.parse(exif_dict['DateTime_Original'])
            exif_dict['DateTime'] = datetime.strptime(re.sub('[/.:]', '-', exif_dict['DateTime_Original']), '%Y-%m-%d %H-%M-%S')
            print(exif_dict['DateTime'], exif_dict['DateTime_Original'])

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
                exif_dict['Estimated_UTM_Zone'] = str(zone) + zoneLetter
                exif_dict['Estimated_UTM_EPSG'] = utm_epsg_from_latlot(
                exif_dict['GPS_Latitude_DD'], exif_dict['GPS_Longitude_DD']
                )
            else:
                print(f"WARNING: No GPS data found in {img}.")

            # pull GPS Altitude data
            exif_dict['GPS_Altitude'] = img_exif.get('GPS_Altitude')
            exif_dict['GPS_Altitude_Ref'] = img_exif.get('GPS_Altitude_Ref')

        return exif_dict

    def build_usbl_gdf(
        self, pdop_field=None, filter_quartile=0.95
    ):
        # read USBL data into GeoPandas
        usbl_gdf = gpd.read_file(self.usbl_path)

        # filter outliers by keeping lower quartile of PDOP values
        if self.pdop_field is not None:
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
            print ("WARNING: No PDOP field specified. No filtering will be performed.")

        # TODO: Consider how to handle dates... this parser has already failed once,
        # and has some heavy overhead. It may be better to just prompt user for a
        # date format string and use strptime instead...

        # parse datetime, set as index
        #usbl_gdf["datetime_field"] = usbl_gdf[self.datetime_field].apply(
        #    lambda x: dateutil.parser.parse(x)
        #)
        usbl_gdf["datetime_idx"] = usbl_gdf[self.datetime_field].apply(
            lambda x: datetime.strptime(re.sub('[/.:]', '-', x), '%Y-%m-%d %H-%M-%S')
        )
        usbl_gdf.index = usbl_gdf['datetime_idx']

        # these fill be used to filter out the images that don't have USBL data during
        # the fit procedure...
        self.datetime_min = usbl_gdf["datetime_idx"].min()
        self.datetime_max = usbl_gdf["datetime_idx"].max()

        self.usbl_gdf = usbl_gdf

        if self.epsg_str is None:
            print(self.epsg_str)
            self.epsg_str = str(usbl_gdf.crs)


    """TRAJECTORY FCNS"""
    def calc_trajectory(self, pt_gdf, process_noise_std=1.0, measurement_noise_std=0.25):
        # calculate trajectory information with MovingPandas
        print(f"EPSG: {self.epsg_str}")
        traj = mpd.Trajectory(pt_gdf, 1, t="datetime_idx", crs=pt_gdf.crs)

        traj.add_direction(overwrite=True, name="Direction")
        traj.add_speed(overwrite=True, name="Speed")
        # traj.add_acceleration(overwrite=True, name="Acceleration")
        traj.add_distance(overwrite=True, name="Distance")
        # traj.add_timedelta(overwrite=True, name="TimeDelta")

        self.raw_usbl_traj = traj
        self.raw_usbl_df = traj.to_point_gdf()

        # Smooth trajectories...
        # TODO: experiment with best methods, make smoothing optional.
        # if process_noise_std is not equal to 0.0, smoothing is applied
        if process_noise_std != 0.0 or measurement_noise_std != 0.0:
            s_traj = KalmanSmootherCV(traj).smooth(
                process_noise_std=process_noise_std,
                measurement_noise_std=measurement_noise_std)

            s_traj.add_direction(overwrite=True, name="Direction")
            s_traj.add_speed(overwrite=True, name="Speed")
            # s_traj.add_acceleration(overwrite=True, name="Acceleration")
            s_traj.add_distance(overwrite=True, name="Distance")
            # s_traj.add_timedelta(overwrite=True, name="TimeDelta")

            self.smooth_usbl_traj = s_traj
            self.smooth_usbl_df = s_traj.to_point_gdf()
            self.smooth_usbl_df.crs = self.epsg_str
            self.smooth_usbl_traj_line = s_traj.to_line_gdf()
            self.smooth_usbl_traj_line.crs = self.epsg_str

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

        # filter new_gdf based on DateTime field and self.datetime_min/max
        new_gdf = new_gdf[
            (new_gdf.DateTime >= self.datetime_min) & (new_gdf.DateTime <= self.datetime_max)
        ]
        # new_gdf = new_gdf.loc[self.datetime_min:self.datetime_max]
        count = len(self.img_gdf) - len(new_gdf)
        print(f"{count} images were filtered based on their DateTime field.")

        print(new_gdf.head())

        # Fit Images to USBL using DateTime
        new_gdf['Improved_Position'] = new_gdf.apply(lambda row: self.smooth_usbl_traj.get_position_at(row.DateTime, method='interpolated'), axis=1)

        print(new_gdf['Improved_Position'])



        # store the vector lines represenintg the  "delta" between each EXIF point and
        # the USBL location at that datetime. These are strictly used for plotting.
        if 'geometry' in new_gdf.columns:
            delta_gdf = new_gdf[new_gdf.geometry.is_empty == False].copy()

            delta_gdf["Delta_LineString"] = delta_gdf.apply(
                lambda row: LineString([row.geometry, row.Improved_Position]), axis=1)

            delta_gdf["Delta_LineLength"] = delta_gdf.apply(
                lambda row: row.Delta_LineString.length, axis=1)

            # TODO: log metadata about shift distance, etc. to flag potential issues.
            #print(delta_gdf['Delta_LineLength'].describe())
            print(f"Len filt / new: {len(delta_gdf)}, {len(new_gdf)}")

            # Set the delta_gdf geometry to linestring
            delta_gdf.geometry = delta_gdf.Delta_LineString  # TODO: clean this gdf up?
            delta_gdf.drop(columns=['Delta_LineString', 'Improved_Position'], inplace=True)

            self.delta_gdf = delta_gdf

            # TODO: supersede geom, store geoms as txt?
            new_gdf.geometry = new_gdf.Improved_Position
            new_gdf.drop(columns=['Improved_Position'], inplace=True)
            new_gdf.sort_index(inplace=True)
        else:
            new_gdf = gpd.GeoDataFrame(new_gdf, geometry=new_gdf.Improved_Position, crs=self.epsg_str)
            new_gdf.drop(columns=['Improved_Position'], inplace=True)
            new_gdf.sort_index(inplace=True)

        # TODO: Clean up this code, a lot of redundancy... (for testing)
        # fit the direction (degrees) and Z (height) values from USBL
        if self.smooth_usbl_df is not None:
            new_gdf2 = pd.merge_asof(new_gdf, self.smooth_usbl_df[['Direction']], left_index=True, right_index=True)
            z_gdf = self._zLookup(new_gdf2, self.smooth_usbl_df, z_field=self.alt_field, datetime_field=self.datetime_field)
        else:
            new_gdf2 = pd.merge_asof(new_gdf, self.raw_usbl_df[['Direction']], left_index=True, right_index=True)
            z_gdf = self._zLookup(new_gdf2, self.raw_usbl_df, z_field=self.alt_field, datetime_field=self.datetime_field)

        self.fit_gdf = z_gdf
        print(type(self.fit_gdf))

    """GEOREF FCNS"""
    def apply_gsd(self, in_gdf):
        # Extract the ground spacing distance from each row of the fit_gdf
        in_gdf["GSD_W"] = in_gdf.apply(
            lambda row: self._calc_gsd(row, z_field=self.alt_field), axis=1)

        in_gdf["GSD_H"] = in_gdf.apply(
            lambda row: self._calc_gsd(row, z_field=self.alt_field, height=True), axis=1)

        in_gdf["GSD_Unit"] = "meters"

        in_gdf['GSD_MAX'] = in_gdf[['GSD_W', 'GSD_H']].max(axis=1)

        in_gdf['GSD_MIN'] = in_gdf[['GSD_W', 'GSD_H']].min(axis=1)

    def _calc_gsd(self, row, height=False, z_field='CamAltCor'):
        # calculate the ground spacing distance (GSD) for each image in meters
        H = row[z_field]
        F = row.Focal_Length
        img_w = row.Pixel_X_Dimension
        img_h = row.Pixel_Y_Dimension

        # TODO: handle other RES UNITs. This assumes CM.
        sensor_w = ureg.convert(row.Est_Sensor_Width, row.Est_Sensor_HW_Unit, "mm")
        sensor_h = ureg.convert(row.Est_Sensor_Height, row.Est_Sensor_HW_Unit, "mm")

        if height:
            gsd_h = (H * sensor_h) / (F * img_h)
            return gsd_h
        else:
            gsd_w = (H * sensor_w) / (F * img_w)
            return gsd_w

    def _apply_upscale_factor(self, in_gdf):
        # Extract the ground spacing distance from each row of the fit_gdf
        in_gdf["Upscale_Factor"] = in_gdf.apply(
            lambda row: self.max_gsd_mode / row.GSD_MAX, axis=1)

    def _apply_corner_gcps(self, in_gdf):
        # Extract the ground spacing distance from each row of the fit_gdf

        in_gdf['bbox'] = in_gdf.apply(lambda row: self._rotate_corner_gcps(row), axis=1)

    def _rotate_corner_gcps(self, row):
        # calculate the size of each image in meters
        center_x = row.geometry.x
        center_y = row.geometry.y

        left = center_x - ((row.Pixel_X_Dimension * row.GSD_W) / 2)
        right = center_x + ((row.Pixel_X_Dimension * row.GSD_W) / 2)
        bottom = center_y - ((row.Pixel_Y_Dimension * row.GSD_H) / 2)
        top = center_y + ((row.Pixel_Y_Dimension * row.GSD_H) / 2)

        bl = (left, bottom, 0)
        br = (right, bottom, 0)
        tl = (left, top, 0)
        tr = (right, top, 0)
        cols, rows = row.Pixel_X_Dimension, row.Pixel_Y_Dimension
        origin = (center_x, center_y, 0)

        rot_tl = self._rotate_point_3d(tl, math.radians(row.Direction), 'z', origin=origin)  # NOTE: may want to cut the custom code here in favor of a shapely poly rotation...
        rot_bl = self._rotate_point_3d(bl, math.radians(row.Direction), 'z', origin=origin)
        rot_tr = self._rotate_point_3d(tr, math.radians(row.Direction), 'z', origin=origin)
        rot_br = self._rotate_point_3d(br, math.radians(row.Direction), 'z', origin=origin)

        corners = [Point(rot_bl), Point(rot_br), Point(rot_tr), Point(rot_tl)]

        # create shapely polygon from the corners
        rot_bbox = Polygon([[p.x, p.y] for p in corners])

        return rot_bbox

    def _rotate_point_3d(self, point, angle, axis, origin=(0, 0, 0)):
        # RADIANS ONLY!
        x, y, z = point
        xo, yo, zo = origin
        c, s = math.cos(angle), math.sin(angle)

        if axis == 'x':
            return x, yo + (y-yo)*c - (z-zo)*s, zo + (y-yo)*s + (z-zo)*c
        elif axis == 'y':
            return xo + (x-xo)*c + (z-zo)*s, y, zo - (x-xo)*s + (z-zo)*c
        elif axis == 'z':
            return xo + (x-xo)*c - (y-yo)*s, yo + (x-xo)*s + (y-yo)*c, z
        else:
            raise ValueError("Invalid axis")

    def _apply_transform(self, in_gdf):
        # Create an affine transform for each image
        in_gdf.apply(lambda row: self._calc_transform(row), axis=1)

    def _calc_transform(self, row):
        # print(len(row.bbox.exterior.coords[0:4]))
        tl, tr, br, bl = row.bbox.exterior.coords[0:4]
        cols, rows = row.Pixel_X_Dimension, row.Pixel_Y_Dimension

        # TODO: figure out if 1/2 pixel shift is needed??
        gcps = [
            GCP(0, 0, *tl),
            GCP(0, cols, *tr),
            GCP(rows, 0, *bl),
            GCP(rows, cols, *br)
        ]

        transform = from_gcps(gcps)
        # print(transform, type(transform))

        self.transforms[row.img_name] = transform

    def orient_images(self, in_gdf):
        self.apply_gsd(in_gdf)

        self.gsd_mode_max = in_gdf.GSD_MAX.mode().max()
        print(f"Mode of Max GSD: {self.gsd_mode_max}")

        self._apply_upscale_factor(in_gdf)

        self._apply_corner_gcps(in_gdf)

        self._apply_transform(in_gdf)

        self.bbox_gdf = in_gdf[['img_path', 'img_name', 'bbox']].copy()
        self.bbox_gdf.geometry = self.bbox_gdf.bbox
        self.bbox_gdf.crs = self.epsg_str


    def georeference_images(self, in_gdf):
        # Extract the ground spacing distance from each row of the fit_gdf
        in_gdf.apply(lambda row: self._scale_and_write_image(row), axis=1)


    """PLOTTING + WRITING FCNS"""
    def _write_gdf(self, target_gdf, basename, format="GPKG", index=False):
        # TODO: this is a patch because writing tuples is a no-no. Need long-term fix...
        target_gdf.drop(['GPS_Latitude_DMS', 'GPS_Latitude_Ref', 'GPS_Longitude_DMS', 'GPS_Longitude_Ref', 'bbox'], axis=1, inplace=True, errors='ignore')

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
            self._write_gdf(self.fit_gdf, "image_centroids", format="GPKG", index=False)

        #if self.img_gdf is not None:
        #    self._write_gdf(self.img_gdf, "img_gdf", format="GPKG", index=False)

        if self.delta_gdf is not None:
            self._write_gdf(self.delta_gdf, "image_to_traj_fit", format="GPKG", index=False)

        #if self.raw_usbl_df is not None:
        #    self._write_gdf(self.raw_usbl_df, "raw_usbl_df", format="GPKG", index=False)

        #if self.smooth_usbl_df is not None:
        #    self._write_gdf(self.smooth_usbl_df, "smooth_usbl_df", format="GPKG", index=False)

        if self.smooth_usbl_traj_line is not None:
            self._write_gdf(self.smooth_usbl_traj_line, "calculated_trajectory", format="GPKG", index=False)

        if self.bbox_gdf is not None:
            self._write_gdf(self.bbox_gdf, "image_bboxes", format="GPKG", index=False)

    def plot_smoothing_operation(self, save_fig=False):
        f, ax = plt.subplots()
        if self.raw_usbl_traj is not None:
            self.raw_usbl_traj.plot(ax=ax, color='red', label="Raw Trajectory", legend=True)
        else:
            print("No raw USBL trajectory to plot. Skipping...")
        if self.smooth_usbl_traj is not None:
            self.smooth_usbl_traj.plot(ax=ax, column="Speed", cmap="viridis", label="Smoothed Trajectory", legend=True)

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
            if self.delta_gdf is not None:
                # plot the deltas...
                self.delta_gdf.plot(ax=ax, color='blue', label="Shifts", legend=True)

            # only plot the best usbl trackline...
            if self.smooth_usbl_traj is not None:
                self.smooth_usbl_traj.plot(ax=ax, column="Speed", cmap="viridis", label='Smoothed Trackline', legend=True)
            else:
                self.raw_usbl_traj.plot(ax=ax, color='red', label="Raw Trackline", legend=True)
        if self.delta_gdf is not None:
            plt.text(0, 0, f"Average shift: {self.delta_gdf['Delta_LineLength'].mean()} meters")
        plt.title("EXIF Points to USBL Trajectory Fit")

        plt.xlabel("UTM Easting (m)")
        plt.ylabel("UTM Northing (m)")

        plt.legend()

        # TODO: label deltas, print delta metrics.

        # plt.show()
        if save_fig is True:
            plt.savefig(os.path.join(self.out_dir, "plot_usbl_fit.png"))

    def plot_rotate(self, pts1, pts2, save_fig=False):
        f, ax = plt.subplots()

        # plot pts1 and pts2 as points...
        plt.plot(pts1, 'o', color='red', label="OG")
        plt.plot(pts2, 'o', color='blue', label="Rot")

        plt.show()

    def _scale_and_write_image(self, row):
        # use rasterio to write image with crs and transform
        img_transform = self.transforms[row.img_name]
        output_file = os.path.join(self.out_dir, row.img_name)

        with rasterio.open(row.img_path, 'r') as src:
            data=src.read()
            with rasterio.open(output_file, 'w', **src.profile) as dst:
                out_data = src.read(
                    out_shape=(
                        src.count,
                        int(src.height * row.Upscale_Factor),
                        int(src.width * row.Upscale_Factor)
                    ),
                    resampling=Resampling.bilinear
                )
                dst.crs = self.epsg_str
                dst.nodata = 0
                dst.transform = img_transform * img_transform.scale(
                    (src.width / data.shape[-1]),
                    (src.height / data.shape[-2])
                )

                dst.write(out_data)
        print(f"Finished writing {row.img_name} to {output_file}")
