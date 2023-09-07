import os
import re
import math
import pint

from datetime import datetime

from PIL import Image as Image
from exif import Image as ExifImage

from matplotlib import pyplot as plt
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

from rasterio import logging
log = logging.getLogger()
log.setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", message="CRS not set for some of the concatenation inputs.")

allowed_img_ext = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

res_unit_lookup = {1: "None", 2: "Inch", 3: "Centimeter", None: "None" }

ureg = pint.UnitRegistry()


class TowLine:
    def __init__(
        self, img_dir, out_dir, usbl_path="None", datetime_field="DateTime",
        pdop_field=None, alt_field="CaAltDepth", filter_quartile=0.95,
        process_noise_std=1.0, measurement_noise_std=0.25, preview_mode=False
    ):
        """ Initialize the TowLine class, and either preview or receive images."""

        self.img_dir = img_dir
        self.out_dir = out_dir

        # Make the output directory if it doesn't exist..
        os.makedirs(self.out_dir, exist_ok=True)

        self.usbl_path = usbl_path

        self.datetime_field = datetime_field
        self.pdop_field = pdop_field
        self.alt_field = alt_field

        self.filter_quartile = filter_quartile
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.preview_mode = preview_mode

        self.img_gdf = None  # the imagery database
        self.usbl_pts = None  # the USBL database (Point)

        self.bbox_gdf = None  # the bounding box database

        self.usbl_traj = None  # the movinpandas trajectory
        self.usbl_traj_lines = None # the movingpandas trajectory as lines
        self.usbl_traj_pts = None  # the movingpandas trajectory as points
        
        self.epsg_str = None
        self.max_gsd_mode = 0.0
        self.max_gsd = 0.0
        self.min_gsd = 0.0

        self.transforms = {}

        self.imgs_datetime_min = None
        self.imgs_datetime_max = None

        self.usbl_datetime_min = None
        self.usbl_datetime_max = None

        # Step 1: Read all images in img_dir, extract relevant EXIF data (geoexif),
        # and build a GeoPandasDataFrame (gdf) from this information...
        print("BUILD IMG DF")
        self.build_img_df()

        # Step 2: Read the USBL GPS data.
        print("BUILD USBL DF")
        self.build_usbl_gdf(
            pdop_field=self.pdop_field, filter_quartile=self.filter_quartile
        )

        # Step 3: Filter the USBL and image dataframes by the DateTime intersection
        # of the two data sets. This will remove any images or USBL pings that were
        # not synchronized during the survey.
        print("FILTER BY DATETIME")
        self._filter_by_datetime()

        # Step 4: Interpolate a trackline from the filtered USBL pings using MovingPandas.
        # Optionally filter the USBL pings again by speed information. 
        # Optionally smooth the trackline with a Kalman filter.
        print("CALC TRAJ")
        self.calc_trajectory()

        print("FIT IMGS TO USBL")
        self.fit_to_usbl()

        print("ESTIMATE GSD")
        self.apply_gsd()

        # Step 5: Round out the internal / external orientation parameters needed for
        # georeferencing OR orthorectification, and perform the operation.
        print("ORIENT IMAGES")
        self.orient_images()

    """DATA INGEST FCNS"""
    def build_img_df(self):
        """ Given a directory of images, extract the relevant EXIF data and build a
        GeoPandasDataFrame (gdf) from this information. """

        imgs = [i for i in os.listdir(self.img_dir) if i.endswith(allowed_img_ext)]
        print(f"Found {len(imgs)} images in {self.img_dir}...")

        geoexifs = []
        for img in imgs:
            geoexif = self._extract_img_exif(img)
            geoexifs.append(geoexif)

        img_df = pd.DataFrame(geoexifs)
        img_df['datetime_idx'] = img_df['DateTime'].astype('datetime64[ns]')
        img_df.index = img_df['datetime_idx']

        self.img_gdf = img_df

        self.imgs_datetime_min = img_df['datetime_idx'].min()
        self.imgs_datetime_max = img_df['datetime_idx'].max()

    def _extract_img_exif(self, img):
        """ Given an image path, extract the relevant EXIF data and return as a
        dictionary."""

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
            exif_dict["Label"] = img

            exif_dict['Focal_Length'] = img_exif.get('Focal_Length')
            exif_dict['Focal_Length_Unit'] = "millimeters"

            exif_dict['Focal_Plane_X_Resolution'] = img_exif.get('focal_plane_x_resolution')
            exif_dict['Focal_Plane_Y_Resolution'] = img_exif.get('focal_plane_y_resolution')
            #print(img_exif.get('focal_plane_resolution_unit'))
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
            #print(exif_dict['DateTime'], exif_dict['DateTime_Original'])

        return exif_dict

    def build_usbl_gdf(self, pdop_field=None, filter_quartile=0.95):
        """ Given a path to a geospatial vector point file
        (e.g. shapefile, geopackage, etc), read the file into a GeoPandas GeoDataFrame
        and optionally filter outliers based on a precision (PDOP) field.
        """
        # read USBL data into GeoPandas
        usbl_gdf = gpd.read_file(self.usbl_path)

        # NOTE: Ross is commenting out the PDOP filter for now because Jordan pointed out flaws in the approach. This may
        #        not really be needed, if we are able to just filter outliers using MovingPandas' built-in fcns.

        # filter outliers by keeping lower quartile of PDOP values
        #if self.pdop_field is not None:
        #    max_pdop = usbl_gdf[pdop_field].quantile(filter_quartile)
        #    print(type(max_pdop), max_pdop)
#
        #    if not math.isnan(max_pdop):
        #        count = len(usbl_gdf) - len(usbl_gdf[usbl_gdf[pdop_field] < max_pdop])
        #        usbl_gdf = usbl_gdf[usbl_gdf[pdop_field] < max_pdop]
#
        #        print(f"--filter_quartile of {filter_quartile} allows a max PDOP of {max_pdop}.")
        #        print(f"{count} USBL pings were filtered based on their {pdop_field} field.")
        #    else:
        #        print(f"WARNING: No valid PDOP values found in column {pdop_field}. No filtering will be performed.")
        #else:
        #    print ("WARNING: No PDOP field specified. No filtering will be performed.")

        # standardize the datetime field, and set that as the index
        usbl_gdf["datetime_idx"] = usbl_gdf[self.datetime_field].apply(
            lambda x: datetime.strptime(re.sub('[/.:]', '-', x), '%Y-%m-%d %H-%M-%S')
        )
        usbl_gdf.index = usbl_gdf['datetime_idx']

        # these will be used to filter out the images that don't have USBL data during
        # the fit procedure...
        self.usbl_datetime_min = usbl_gdf["datetime_idx"].min()
        self.usbl_datetime_max = usbl_gdf["datetime_idx"].max()

        self.usbl_pts = usbl_gdf
        self.epsg_str = str(usbl_gdf.crs)

    def _filter_by_datetime(self):
        """Given the a GeoDataFrame of USBL pings and DataFrame of image EXIF metadata, determine
        the intersection of time stamps between the two data sets (valid time range), and drop
        any USBL pings or images collected outside of the valid time range."""
        print(self.imgs_datetime_min, self.usbl_datetime_min)

        valid_min = max(self.imgs_datetime_min, self.usbl_datetime_min)
        valid_max = min(self.imgs_datetime_max, self.usbl_datetime_max)
        print(f"Valid time range: {valid_min} to {valid_max}")

        og_usbl_count = len(self.usbl_pts)
        og_img_count = len(self.img_gdf)

        # filter USBL pings by valid time range
        self.usbl_pts = self.usbl_pts[self.usbl_pts['datetime_idx'] > valid_min]
        self.usbl_pts = self.usbl_pts[self.usbl_pts['datetime_idx'] < valid_max]

        # filter images by valid time range
        self.img_gdf = self.img_gdf[self.img_gdf['datetime_idx'] > valid_min]
        self.img_gdf = self.img_gdf[self.img_gdf['datetime_idx'] < valid_max]

        filter_usbl_count = len(self.usbl_pts)
        filter_img_count = len(self.img_gdf)
        print(f"Filtered {og_usbl_count - filter_usbl_count} USBL pings and {og_img_count - filter_img_count} images based on valid time range of {valid_min} to {valid_max}.")

    """TRAJECTORY FCNS"""
    def calc_trajectory(self):
        """ Given a GeoPandas GeoDataFrame of points, calculate the trajectory with
        MovingPandas. Optionally smooth the trajectory with a Kalman filter."""

        # calculate trajectory information with MovingPandas
        print(f"EPSG: {self.epsg_str}")
        traj = mpd.Trajectory(self.usbl_pts, 1, t="datetime_idx", crs=self.usbl_pts.crs)

        traj.add_direction(overwrite=True, name="Direction")
        traj.add_speed(overwrite=True, name="Speed")
        # traj.add_acceleration(overwrite=True, name="Acceleration")
        traj.add_distance(overwrite=True, name="Distance")
        # traj.add_timedelta(overwrite=True, name="TimeDelta")

        #speed_thresh = traj.df['Speed'].mean() * 3
        #print(f"Mean Speed * 3: {speed_thresh}")

        #mpd.OutlierCleaner(traj).clean(v_max=speed_thresh, units='m')
        mpd.IqrCleaner(traj).clean(columns={'Speed': 3})

        self.usbl_traj = traj
        self.usbl_traj_pts = traj.to_point_gdf()
        self.usbl_traj_lines = traj.to_line_gdf()

        # Smooth trajectories...
        # TODO: experiment with best methods, make smoothing optional.
        # if process_noise_std is not equal to 0.0, smoothing is applied
        if self.process_noise_std != 0.0 or self.measurement_noise_std != 0.0:
            s_traj = KalmanSmootherCV(traj).smooth(
                process_noise_std=self.process_noise_std,
                measurement_noise_std=self.measurement_noise_std)

            s_traj.add_direction(overwrite=True, name="Direction")
            s_traj.add_speed(overwrite=True, name="Speed")
            # s_traj.add_acceleration(overwrite=True, name="Acceleration")
            s_traj.add_distance(overwrite=True, name="Distance")
            # s_traj.add_timedelta(overwrite=True, name="TimeDelta")

            self.usbl_traj = s_traj
            self.usbl_traj_pts = s_traj.to_point_gdf()
            self.usbl_traj_lines = s_traj.to_line_gdf()
        else:
            print(f"Either process_noise_std or measurement_noise_std is set to 0.0, \
                therefore no smoothing will be applied to trackline.")

    def _zLookup(self, datetime_field='DateTime', z_field='CameraZ'):
        """ Given a a USBL dataframe with precise Z data, correlate and interpolate
        Z values for all images in the image GeoDataFrame.
        """
        # get a list of datetimes every second between start and end:
        start = self.usbl_datetime_min
        stop = self.usbl_datetime_max
        dt_list = pd.date_range(start, stop, freq='S')

        # merge dt_list with usbl_gdf, fill in missing values with NaN, keep only CaAltCor_m field
        self.usbl_traj_pts = pd.merge(dt_list.to_series(name='time_range'), self.usbl_traj_pts, how='left', left_index=True, right_index=True)
        self.usbl_traj_pts = self.usbl_traj_pts[['time_range', z_field, datetime_field]]

        # perform linear interpolation to fill in missing Z values
        self.usbl_traj_pts[z_field].interpolate(method='linear', inplace=True)

        # use usbl_dt_gdf as a lookup table to add CaAltCor_m to img_df
        self.img_gdf[z_field] = self.img_gdf['DateTime'].map(self.usbl_traj_pts.set_index('time_range')[z_field])

    def fit_to_usbl(self):
        """ Given a point GeoDataFrame of images, fit the images to the USBL trajectory
        line (a MovingPandas Trajectory object) using the DateTime field as the key.

        TODO: this function is doing alot... maybe split it up?
        - This function also filters images by DateTime (if an image is outside trackline).
        - This function also computes the "delta' between each image and USBL position, and
        stores this info as a GeoDataFrame of linestrings.
        """

        # Fit Images to USBL using DateTime
        self.img_gdf['Improved_Position'] = self.img_gdf.apply(
            lambda row: self.usbl_traj.interpolate_position_at(row.DateTime), axis=1
        )

        self.img_gdf = gpd.GeoDataFrame(self.img_gdf, geometry=self.img_gdf.Improved_Position, crs=self.epsg_str)
        self.img_gdf.drop(columns=['Improved_Position'], inplace=True)
        self.img_gdf.sort_index(inplace=True)

        # fit the direction (degrees) and Z (height) values from USBL
        self.img_gdf = pd.merge_asof(self.img_gdf, self.usbl_traj_pts[['Direction']], left_index=True, right_index=True)

        self._zLookup(z_field=self.alt_field, datetime_field=self.datetime_field)

    """GEOREF FCNS"""
    def orient_images(self):
        """ Given a GeoDataFrame of images, run a series of pandas apply functions to
        calculate image GSD, corner GCPs, rotated corner GCPs, and produce a dataframe
        with the appropriate geometry and CRS information.
        """
        self._apply_gsd()

        self.gsd_mode_max = self.img_gdf.GSD_MAX.mode().max()
        print(f"Mode of Max GSD: {self.gsd_mode_max}")

        self._apply_upscale_factor()

        self._apply_corner_gcps()

        self._apply_transform()

        self.bbox_gdf = self.img_gdf[['img_path', 'Label', 'bbox']].copy()
        self.bbox_gdf.geometry = self.bbox_gdf.bbox
        self.bbox_gdf.crs = self.epsg_str

    def _apply_gsd(self):
        """ Given a GeoDataFrame of images, run a pandas apply function to estimate
        the ground spacing distance (GSD) for each image's X and Y dimensions, and
        sets new columns specifying which is the min and max GSD.
        """

        # Extract the ground spacing distance from each row of the.img_gdf
        self.img_gdf["GSD_W"] = self.img_gdf.apply(
            lambda row: self.__calc_gsd(row, height=False), axis=1)

        self.img_gdf["GSD_H"] = self.img_gdf.apply(
            lambda row: self.__calc_gsd(row, height=True), axis=1)

        # metadata column allows users to ascertain the units of GSD
        self.img_gdf["GSD_Unit"] = "meters"

        # set the min and max GSD columns
        self.img_gdf['GSD_MAX'] = self.img_gdf[['GSD_W', 'GSD_H']].max(axis=1)

        self.img_gdf['GSD_MIN'] = self.img_gdf[['GSD_W', 'GSD_H']].min(axis=1)

    def __calc_gsd(self, row, height=False):
        """ This function contains the mathematics for estimating the ground spacing
        distance (GSD) of an image (for either height or width).

        If height is True, the GSD is calculated for the height of the image.
        """
        # calculate the ground spacing distance (GSD) for each image in meters
        H = row[self.alt_field]
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

    def _apply_upscale_factor(self):
        """ Given a GeoDataFrame of images, run a pandas apply function to estimate
        the amount of upscaling (or downscaling) that needs to be applied to each
        image to match the max GSD of the GeoDataFrame. The upscale factor is
        only estimated here, the actual upscaling operation happens when the image
        data is actually read and warped by rasterio.
        """
        # Extract the ground spacing distance from each row of the.img_gdf
        self.img_gdf["Upscale_Factor"] = self.img_gdf.apply(
            lambda row: self.max_gsd_mode / row.GSD_MAX, axis=1)

    def _apply_corner_gcps(self):
        """ Given a GeoDataFrame of images, run a pandas apply function that rotates
        the corner gcps of each image to match the direction (bearing) of the image."""
        # Extract the ground spacing distance from each row of the.img_gdf
        self.img_gdf['bbox'] = self.img_gdf.apply(lambda row: self.__rotate_corner_gcps(row), axis=1)

    def __rotate_corner_gcps(self, row):
        """ This function contains the mathematics for rotating the corner gcps of
        an image to match the direction (bearing) of the image.

        This function is called by the _apply_corner_gcps function.
        """
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

        rot_tl = self.___rotate_point_3d(tl, math.radians(row.Direction), 'z', origin=origin)  # NOTE: may want to cut the custom code here in favor of a shapely poly rotation...
        rot_bl = self.___rotate_point_3d(bl, math.radians(row.Direction), 'z', origin=origin)
        rot_tr = self.___rotate_point_3d(tr, math.radians(row.Direction), 'z', origin=origin)
        rot_br = self.___rotate_point_3d(br, math.radians(row.Direction), 'z', origin=origin)

        corners = [Point(rot_bl), Point(rot_br), Point(rot_tr), Point(rot_tl)]

        # create shapely polygon from the corners
        rot_bbox = Polygon([[p.x, p.y] for p in corners])

        return rot_bbox

    def ___rotate_point_3d(self, point, angle, axis, origin=(0, 0, 0)):
        """ Contains the mathematics for rotating a point around an arbitrary axis and
        origin. All angles in radians.

        NOTE: the current workflow only requires rotating about the Z axis (to adjust image
        orientation to match direction/bearing of travel). However, the ability to rotate about
        X/Y would be useful for future development (i.e. roll/pitch adjustments).
        """
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

    def _apply_transform(self):
        """ Given a GeoDataFrame of images, run a pandas apply function to calculate
        the corner GCPs for each image, and generate/return an affine tranformation
        matrix for each image. This affine transform is used to reference or warp the
        image data to a common coordinate system.
        """
        # Create an affine transform for each image
        self.img_gdf.apply(lambda row: self.__calc_transform(row), axis=1)

    def __calc_transform(self, row):
        """ This function contains the operations required to extract image GCPs and
        generate an affine transformation matrix (rasterio does the heavy lifting here).
        """
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

        self.transforms[row.Label] = transform

    def write_georeferenced_images(self):
        """ Given a GeoDataFrame of images (rows), run a pandas apply function to open
        the original image, resample to a new GSD, and write to the output directory.
        """

        # Extract the ground spacing distance from each row of the.img_gdf
        self.img_gdf.apply(lambda row: self._scale_and_write_image(row), axis=1)
    
    def _scale_and_write_image(self, row):
        """ Contains the operations to take an input image and affine transform, open
        the source image, retrieve the pixel data, resample this data using the upscale
        factor, and write this resampled image data to disk with the new geometry and CRS
        information. The output images are effectively georeferenced.
        """
        # use rasterio to write image with crs and transform
        img_transform = self.transforms[row.Label]
        output_file = os.path.join(self.out_dir, row.Label)

        with rasterio.open(row.img_path, 'r') as src:
            data = src.read()
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
        print(f"Finished writing {row.Label} to {output_file}")



    """PLOTTING + WRITING FCNS"""
    def write_metashape_csv(self):
        self.img_gdf['Easting'] = self.img_gdf.geometry.x
        self.img_gdf['Northing'] = self.img_gdf.geometry.y
        self.img_gdf['Altitude'] = self.img_gdf[self.alt_field]

        out_gdf = self.img_gdf[['Easting', 'Northing', 'Altitude', 'DateTime']]
        # out_gdf['Label'] = self.img_gdf['Label']
        out_gdf.to_csv(os.path.join(self.out_dir, "metashape.csv"), index=False)

    def _write_gdf(self, in_gdf, basename, format="GPKG", index=False):
        # TODO: this is a patch because writing tuples is a no-no. Need long-term fix...
        in_gdf.drop(['GPS_Latitude_DMS', 'GPS_Latitude_Ref', 'GPS_Longitude_DMS', 'GPS_Longitude_Ref', 'bbox'],
                        axis=1, inplace=True, errors='ignore')

        if format == "GPKG":
            out_path = os.path.join(self.out_dir, f"{basename}.gpkg")
        elif format == "ESRI Shapefile":
            out_path = os.path.join(self.out_dir, f"{basename}.shp")
        else:
            raise ValueError(f"Invalid geospatial format: {format}. Must be GPKG or ESRI Shapefile.")

        in_gdf.to_file(
            out_path, driver=format, index=index
        )

    def dump_gdfs(self):
        # TODO: carry output format through script (gpkg vs shp)
        if self.img_gdf is not None:
            self._write_gdf(self.img_gdf, "image_centroids", format="GPKG", index=False)

        #if self.delta_gdf is not None:
        #    self._write_gdf(self.delta_gdf, "image_to_traj_fit", format="GPKG", index=False)

        if self.usbl_traj_lines is not None:
            self._write_gdf(self.usbl_traj_lines, "calculated_trajectory", format="GPKG", index=False)
        
        #if self.usbl_traj_pts is not None:
        #    self._write_gdf(self.usbl_traj_pts, "calculated_trajectory", format="GPKG", index=False)

        if self.bbox_gdf is not None:
            self._write_gdf(self.bbox_gdf, "image_bboxes", format="GPKG", index=False)
        f, ax = plt.subplots()