import Metashape
import os, sys, time

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from create_tif import *
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
        print("ERROR: Folder doesn't exist ", input_folder)
        return

    # Create the output folder if it doesn't already exist
    os.makedirs(output_folder, exist_ok=True)
    output_orthomosaic = output_folder + "/Orthomosaic.tif"

    # Create the reference csv, get the path.
    gdf, reference_path = create_reference_csv(input_folder)
    new_reference_path = output_folder + "/new_image_centroids.csv"

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
        chunk.addPhotos(photos)
        print(str(len(chunk.cameras)) + " images loaded")
        doc.save()

    # Match the photos by finding common features and establishing correspondences.
    if not chunk.tie_points:
        chunk.matchPhotos(keypoint_limit=40000,
                          tiepoint_limit=10000,
                          generic_preselection=True,
                          reference_preselection=True,
                          downscale=1)

        # Align the cameras to estimate their relative positions in space.
        chunk.alignCameras()
        doc.save()

    # Build depth maps (2.5D representations of the scene) from the aligned photos.
    if chunk.tie_points and not chunk.depth_maps:
        chunk.buildDepthMaps(filter_mode=Metashape.MildFiltering)
        doc.save()

    # Build a 3D model from the depth maps.
    if chunk.depth_maps and not chunk.model:
        chunk.buildModel(source_data=Metashape.DepthMapsData)
        doc.save()

    # For each camera, project the camera center onto the mesh, to
    # determine if the camera is aligned
    camera_aligned = []
    for camera in chunk.cameras:
        try:
            # Get the camera center, project it onto the mesh
            width, height = camera.sensor.width // 2, camera.sensor.height // 2
            p = chunk.model.pickPoint(camera.center, camera.unproject(Metashape.Vector((width, height, 1))))
            # Add a marker to the chunk
            chunk.addMarker(point=p)
            # Set aligned to True
            aligned = True
        except:
            print("ERROR: Could not project camera ", camera.label)
            aligned = False

        camera_aligned.append(aligned)

    # Add the lists to the reference csv
    gdf['aligned'] = camera_aligned
    gdf.to_csv(new_reference_path)
    # Save the document
    doc.save()

    if chunk.model and not chunk.orthomosaic:
        # Local coordinate system transformation matrix
        R = Metashape.Matrix(np.array([[1.0,  0.0, 0.0],
                                       [0.0, -1.0, 0.0],
                                       [0.0,  0.0, 1.0]]))

        # Set the projection object
        projection = Metashape.OrthoProjection()
        projection.crs = chunk.crs
        projection.matrix = Metashape.Matrix().Rotation(R)
        projection.type = Metashape.OrthoProjection.Type.Planar
        # Create the orthomosaic
        chunk.buildOrthomosaic(surface_data=Metashape.ModelData,
                               blending_mode=Metashape.BlendingMode.MosaicBlending,
                               projection=projection)
        # Save the document
        doc.save()

    # Export the orthomosaic as a GeoTIFF file if it exists in the chunk.
    # Then georeference the orthomosaic using the updated reference csv.
    if chunk.orthomosaic:

        # Identifying a cameras's middle projected onto the mesh, orthomosaic
        x_pixels = []
        y_pixels = []
        image_names = []

        # For each camera, project the camera center onto the mesh, orthomosaic
        # Then get the pixel coordinates of the point on the orthomosaic
        # Then add the pixel coordinates and image name to the lists
        # Then add the lists to the reference csv, output to a new csv.
        for camera in chunk.cameras:
            try:
                width, height = camera.sensor.width // 2, camera.sensor.height // 2
                p = chunk.model.pickPoint(camera.center, camera.unproject(Metashape.Vector((width, height, 1))))
                P = chunk.orthomosaic.crs.project(chunk.transform.matrix.mulp(p))
                x = int((P.x - chunk.orthomosaic.left) / chunk.orthomosaic.resolution)
                y = int((chunk.orthomosaic.top - P.y) / chunk.orthomosaic.resolution)

            except:
                print("ERROR: Could not project camera ", camera.label)
                x = None
                y = None

            x_pixels.append(x)
            y_pixels.append(y)
            image_names.append(camera.label)

        # Save the chunk with markers
        doc.save()

        # Add the lists to the reference csv
        # Then save the csv
        gdf['image_label'] = image_names
        gdf['x_pixels'] = x_pixels
        gdf['y_pixels'] = y_pixels
        gdf.to_csv(new_reference_path)

        # Export the orthomosaic as a GeoTIFF file if it doesn't already exist
        if not os.path.exists(output_orthomosaic):
            chunk.exportRaster(output_orthomosaic, source_data=Metashape.OrthomosaicData)
            doc.save()

        # If the orthomosaic and the reference csv exist, georeference the orthomosaic
        if os.path.exists(output_orthomosaic) and os.path.exists(new_reference_path):
            georeference_orthomosaic(output_orthomosaic, new_reference_path)

    # Print a message indicating that the processing has finished and the results have been saved.
    print('Processing finished, results saved to ' + output_folder + '.')
    print("Time: ", (time.time() - t0) / 60)
    print("Done.")


if __name__ == '__main__':

    # Run the workflow on all folders located within ROOT
    ROOT = "B://Drop_e/Geo/"
    image_folders = os.listdir(ROOT)

    # Loop through all the folders
    for image_folder in image_folders:

        if not image_folder in ["GV027"]:
            continue

        try:
            input_folder = ROOT + image_folder + "/"
            output_folder = input_folder.replace("Geo", "Not_Geo")
            run_sfm_workflow(input_folder, output_folder)

        except Exception as e:
            print(e, "\nIssue with: ", image_folder)
