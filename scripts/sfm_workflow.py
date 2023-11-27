# TODO
# Sort based cameras based on time before passing to Metashape instead of Easting or Northing
# Look into methods (topologies tools) for identifying loops instead of linear track lines

import os
import gc
import sys
import time
from tqdm import tqdm

import Metashape

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from georeference import *

# Star the timer
t0 = time.time()

# Get the Metashape License stored in the environmental variable
Metashape.License().activate(os.getenv("METASHAPE_LICENSE"))

# Check that the Metashape version is compatible with this script
compatible_major_version = "2.0"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


def print_progress(p):
    """Prints the progress of a proccess to console"""

    elapsed = float(time.time() - t0)
    if p:
        sec = elapsed / p * 100
        print('Current task progress: {:.2f}%, estimated time left: {:.0f} seconds'.format(p, sec))
    else:
        print('Current task progress: {:.2f}%, estimated time left: unknown'.format(p))


def find_files(folder, types):
    """Takes in a folder and a list of file types, returns a list of file paths
    that end with any of the specified extensions."""
    return [entry.path for entry in os.scandir(folder) if
            (entry.is_file() and os.path.splitext(entry.name)[1].lower() in types)]


def run_sfm_workflow(input_folder, output_folder):
    """Takes in an input folder, runs SfM Workflow on all images in it,
    outputs the results in the same folder."""

    # Check that input folder exists
    if not os.path.exists(input_folder):
        print("ERROR: Input folder doesn't exist ", input_folder)
        return

    try:
        # Get the gpkg and metashape files from previous scripts
        site_gpkg_path = input_folder + "image_centroids.gpkg"
        site_reference = input_folder + "metashape.csv"
        new_site_reference = output_folder + "/aligned_metashape.csv"

        # If it doesn't exist, that's an issue
        if not os.path.exists(site_gpkg_path):
            raise Exception("ERROR: Could not find image_centroids.gpkg for site")

        # Read and adjust it
        site_gpkg = gpd.read_file(site_gpkg_path)
        site_gpkg['Easting'] = site_gpkg.geometry.x
        site_gpkg['Northing'] = site_gpkg.geometry.y
        site_gpkg.rename({'img_name': 'Label'}, inplace=True, axis=1)
        site_gpkg.rename({'GPS_Altitude': 'Altitude'}, inplace=True, axis=1)
        # Fill any Nulls (i.e., Altitude)
        site_gpkg.fillna(0, inplace=True)

        # Get the names of files (no path, or file format)
        names = [l.split(".")[0] for l in site_gpkg['Label'].values]
        site_gpkg['Name'] = names

        # Create paths to files
        paths = [f"{input_folder}{n}" for n in site_gpkg['Label'].values]
        site_gpkg['Path'] = paths

        # Finally, sort based on max easting first
        # TODO, look at sorting based on Time rather than Easting or Northing
        site_gpkg.sort_values('Easting', ascending=False, inplace=True)

        reference = site_gpkg[["Label", "Easting", "Northing", "Altitude"]]
        reference.to_csv(site_reference)

    except Exception as e:
        print(f"ERROR: Could not read GPKG or CSV\n{e}")
        return

    # Create the output folder if it doesn't already exist
    os.makedirs(output_folder, exist_ok=True)
    output_orthomosaic = output_folder + "/Orthomosaic.tif"
    output_geo_orthomosaic = output_folder + "Geo_Orthomosaic.tif"

    # Call the "find_files" function to get a list of photo file paths
    # with specified extensions from the image folder. ".jpeg", ".tif", ".tiff"
    photos = find_files(input_folder, [".jpg"])

    # Create a metashape doc object
    doc = Metashape.Document()

    if not os.path.exists(output_folder + "/project.psx"):
        # Create a new Metashape document and save it as a project file in the output folder.
        doc.save(output_folder + '/project.psx')
    else:
        # Else open the existing one
        doc.open(output_folder + '/project.psx')

    # Create a new chunk (3D model) in the Metashape document.
    if doc.chunk is None:
        doc.addChunk()
        doc.save()

    # Assign the chunk
    chunk = doc.chunk

    # Add the photos to the chunk.
    if not chunk.cameras:
        chunk.addPhotos(photos, progress=print_progress)
        print(str(len(chunk.cameras)) + " images loaded")
        doc.save()

    # Match the photos by finding common features and establishing correspondences.
    if not chunk.tie_points:
        chunk.matchPhotos(keypoint_limit=40000,
                          tiepoint_limit=10000,
                          generic_preselection=True,
                          reference_preselection=True,
                          downscale=1,
                          progress=print_progress)

        # Align the cameras to estimate their relative positions in space.
        chunk.alignCameras()
        doc.save()

    # Build depth maps (2.5D representations of the scene) from the aligned photos.
    if chunk.tie_points and not chunk.depth_maps:
        chunk.buildDepthMaps(filter_mode=Metashape.MildFiltering,
                             progress=print_progress)
        doc.save()

    # Build a 3D model from the depth maps.
    if chunk.depth_maps and not chunk.model:
        chunk.buildModel(source_data=Metashape.DepthMapsData,
                         progress=print_progress)
        doc.save()

    # Find which cameras were aligned
    aligned_cameras = []

    for camera in chunk.cameras:
        # Get the center of the camera
        width, height = camera.sensor.width // 2, camera.sensor.height // 2

        try:
            # If the camera is aligned, this will work
            p = chunk.model.pickPoint(camera.center, camera.unproject(Metashape.Vector((width, height, 1))))
            aligned_cameras.append(camera.label)
        except Exception as e:
           pass

    # Subset to get only those that are aligned
    aligned_site_gpkg = site_gpkg[site_gpkg['Name'].isin(aligned_cameras)]

    # Get the rotation angle of just those aligned cameras
    r = global_rotation(aligned_site_gpkg)

    # Build the orthomosaic from the model
    if chunk.model and not chunk.orthomosaic:

        # Local coordinate system transformation matrix for planar XY Top-down view
        R = Metashape.Matrix([[math.cos(r), -math.sin(r), 0.0],
                              [math.sin(r), math.cos(r),  0.0],
                              [0.0,         0.0,          1.0]])

        # Set the projection object
        # This defaults to the CRS of the chunk
        # If local CRS, rotation matrix is topXY view
        projection = Metashape.OrthoProjection()
        projection.crs = chunk.crs
        projection.matrix = Metashape.Matrix().Rotation(R)
        projection.type = Metashape.OrthoProjection.Type.Planar

        # Create the orthomosaic
        chunk.buildOrthomosaic(surface_data=Metashape.ModelData,
                               blending_mode=Metashape.BlendingMode.MosaicBlending,
                               projection=projection,
                               progress=print_progress)
        # Save the document
        doc.save()

    # Export the orthomosaic as a GeoTIFF file if it exists in the chunk.
    # Then georeference the orthomosaic using the updated reference csv.
    if chunk.orthomosaic:

        # Identifying a camera's middle projected onto the mesh, orthomosaic
        x_pixels = []
        y_pixels = []
        image_names = []

        # For each camera, project the camera center onto the mesh, orthomosaic
        # Then get the pixel coordinates of the point on the orthomosaic
        # Then add the pixel coordinates and image name to the lists
        # Then add the lists to the reference csv, output to a new csv.

        # For each camera, project the camera center onto the mesh, to
        # determine if the camera is aligned
        for camera in chunk.cameras:
            # Get the center of the camera
            width, height = camera.sensor.width // 2, camera.sensor.height // 2

            try:
                # If the camera is aligned, this will work
                p = chunk.model.pickPoint(camera.center, camera.unproject(Metashape.Vector((width, height, 1))))
                P = chunk.orthomosaic.crs.project(chunk.transform.matrix.mulp(p))
                x = int((P.x - chunk.orthomosaic.left) / chunk.orthomosaic.resolution)
                y = int((chunk.orthomosaic.top + P.y) / chunk.orthomosaic.resolution)

                # Add a marker to the chunk
                chunk.addMarker(point=p)

                x_pixels.append(x)
                y_pixels.append(y)
                image_names.append(camera.label)

            except Exception as e:
               pass

        # Add the lists to the reference csv
        site_gpkg_aligned = site_gpkg[site_gpkg['Name'].isin(image_names)]
        site_gpkg_aligned['x_pixels'] = x_pixels
        site_gpkg_aligned['y_pixels'] = y_pixels
        site_gpkg_aligned = site_gpkg_aligned[["Label",
                                               "Easting",
                                               "Northing",
                                               "Altitude",
                                               "DateTime",
                                               "x_pixels",
                                               "y_pixels"]]
        # Fill any NA with 0 (altitude)
        site_gpkg_aligned.fillna(0, inplace=True)
        # Save reference file
        site_gpkg_aligned.to_csv(new_site_reference)
        # Save the chunk with markers
        doc.save()

        # Export the orthomosaic as a GeoTIFF file if it doesn't already exist
        if not os.path.exists(output_orthomosaic):

            try:
                # Set compression parameters
                compression = Metashape.ImageCompression()
                compression.tiff_big = True
                # Export Orthomosaic
                chunk.exportRaster(output_orthomosaic,
                                   source_data=Metashape.OrthomosaicData,
                                   image_compression=compression,
                                   progress=print_progress)

            except Exception as e:
                print(f"ERROR: BigTIFF Error\n{e}")

            doc.save()
            gc.collect()

        # If the orthomosaic and the reference csv exist, georeference the orthomosaic
        if os.path.exists(output_orthomosaic) and os.path.exists(new_site_reference):
            # Create the georeferenced orthomosaic using the updated reference csv
            georeference_orthomosaic(output_orthomosaic, output_geo_orthomosaic, new_site_reference)

            if os.path.exists(output_geo_orthomosaic):
                # After creating it, import it into metashape
                chunk.importRaster(path=output_geo_orthomosaic,
                                       raster_type=Metashape.DataSource.OrthomosaicData,
                                       progress=print_progress)

                doc.save()

                # Provide reference data to the chunk.
                chunk.importReference(site_reference,
                                      format=Metashape.ReferenceFormatCSV,
                                      columns='nxyz[XYZ]',
                                      delimiter=',')
                doc.save()

    # Print a message indicating that the processing has finished and the results have been saved.
    print('Processing finished, results saved to ' + output_folder + '.')
    print("Time: ", (time.time() - t0) / 60)
    print("Done.")


if __name__ == '__main__':

    for input_folder in glob.glob("B:\\Drop_e\\Testing\\*"):

        if not "GV229" in input_folder:
            continue

        input_folder += "\\"
        output_folder = input_folder + "Project\\"
        print(input_folder, output_folder)

        try:
            run_sfm_workflow(input_folder, output_folder)
        except Exception as e:
            print(e)
