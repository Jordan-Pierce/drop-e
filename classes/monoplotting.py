import os
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

from shapely.geometry import Point, LineString
import folium
import utm

import rasterio
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint


class TowLine:
    def __init__(self, img_dir, out_dir, usbl_path="None"):
        self.img_dir = img_dir
        self.out_dir = out_dir
        self.usbl_path = usbl_path

        self.raw_traj = None
        self.smooth_traj = None
        self.usbl_traj = None

        self.max_gsd = 0.5

        self.usbl_min_timestamp = None
        self.usbl_max_timestamp = None


    def calc_usbl_traj(
        self, datetime_field="DateTime", pdop_field="Max_PDOP", filter_quartile=0.95,
        process_noise_std=1.0, measurement_noise_std=0.25,
    ):
        # read USBL data into GeoPandas
        usbl_gdf = gpd.read_file(self.usbl_path)

        # parse datetime
        usbl_gdf["datetime_field"] = usbl_gdf[datetime_field].apply(
            lambda x: dateutil.parser.parse(x)
        )

        # filter outliers by keeping lower quartile of PDOP values
        max_pdop = usbl_gdf[pdop_field].quantile(filter_quartile)
        print(f"filter_quartile set to {filter_quartile}, this will filter all PDOP \
            values below {max_pdop}")

        usbl_gdf = usbl_gdf[usbl_gdf[pdop_field] < max_pdop]

        # calculate trajectory information with MovingPandas
        usbl_traj = mpd.Trajectory(usbl_gdf, 1, t="datetime_field", crs=usbl_gdf.crs)

        self.raw_traj = usbl_traj

        # Smooth trajectories...
        smoothed_traj = KalmanSmootherCV(usbl_traj).smooth(
            process_noise_std=process_noise_std, measurement_noise_std=measurement_noise_std
        )
        print(f"Smoothed Traj: {smoothed_traj}")

        smoothed_traj.add_direction()
        smoothed_traj.add_speed()
        smoothed_traj.add_distance()
        smoothed_traj.add_timedelta()

        smooth_df = smoothed_traj.df

        self.usbl_traj = smoothed_traj
        self.usbl_traj_df = smooth_df


    def plot_traj(self):
        f, ax = plt.subplots()
        if self.raw_traj is not None:
            self.raw_traj.plot(ax=ax, color='red', legend=True)
        if self.usbl_traj is not None:
            self.usbl_traj.plot(ax=ax, column="speed", cmap="viridis", legend=True)

        #plt.show()
        plt.savefig(os.path.join(self.out_dir, "trajectory.png"))