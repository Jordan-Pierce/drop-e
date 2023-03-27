import shutil
import pandas as pd
from osgeo import gdal, osr

geo_data = 'B://Drop_e/Not_Geo/GV027/image_centroids.csv'
df = pd.read_csv(geo_data, index_col=0)

src_data = 'B://Drop_e/Not_Geo/GV027/Orthomosaic.tif'
dst_data = 'B://Drop_e/Not_Geo/GV027/Geo_Orthomosaic.tif'

# Create a copy of the original file and save it as the output filename:
shutil.copy(src_data, dst_data)

# Open the output file for writing for writing:
ds = gdal.Open(dst_data, gdal.GA_Update)

# Set spatial reference:
sr = osr.SpatialReference()
sr.ImportFromEPSG(32655)

# Enter the GCPs
#   Format: [map x-coordinate(longitude)], [map y-coordinate (latitude)], [elevation],
#   [image column index(x)], [image row index (y)]

gcps = df[['Easting', 'Northing', 'Altitude', 'x_pixels', 'y_pixels']].dropna().values
gdal_gcps = [gdal.GCP(gcp[0], gcp[1], 0, int(gcp[3]), int(gcp[4])) for gcp in gcps]
gdal_gcps = [gdal_gcps[0], gdal_gcps[-1]]

# Apply the GCPs to the open output file:
ds.SetGCPs(gdal_gcps, sr.ExportToWkt())

# Close the output file in order to be able to work with it in other programs:
ds = None