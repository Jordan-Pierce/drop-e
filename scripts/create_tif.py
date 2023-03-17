import sys
import glob
import os.path
import rasterio


def create_tifs(input_folder):
    """Takes in a directory path, globs all jpg files, and converts
    them into georeferenced geotiff files."""

    # Glob all the image files
    image_files = glob.glob(input_folder + "*.jpg")

    for image_file in image_files:
        # Open the JPEG image
        with rasterio.open(image_file) as src:
            # Read the image data
            data = src.read()

        # Create a new profile for the output GeoTIFF
        profile = {
            'driver': 'GTiff',
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'dtype': src.dtypes[0],
            'crs': src.crs,
            'transform': src.transform,
            'compress': 'jpeg',
            'photometric': 'rgb'
        }

        output_file = image_file.replace(".jpg", ".tif")

        # Save the output GeoTIFF
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(data)


if __name__ == "__main__":
    # Pass in the path to the directory
    input_folder = sys.argv[1]

    assert os.path.exists(input_folder), print("Path does not exists: ", input_folder)

    create_tifs(input_folder)
