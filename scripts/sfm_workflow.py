import Metashape
import os, sys, time
from tqdm import tqdm

from create_tif import *
from create_reference import *
from georeference_orthomosaic import *

# Star the timer
t0 = time.time()

# Get the Metashape License stored in the environmental variable
Metashape.License().activate(os.getenv("METASHAPE_LICENSE"))

# Checking compatibility
compatible_major_version = "2.0"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


# Define a function named "find_files" that takes in two arguments:
# a folder (directory) and a list of file types (extensions).
# The function returns a list of file paths that end with any of the specified extensions.
def find_files(folder, types):
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
        doc.save()

    # Print the number of images that were successfully loaded into the chunk.
    print(str(len(chunk.cameras)) + " images loaded")

    # Estimate image quality, remove those that are blurry
    if chunk.cameras:
        print("Checking image quality")
        quality_threshold = 0.2
        for camera in tqdm(chunk.cameras):
            camera_quality = Metashape.Utils.estimateImageQuality(camera.image())
            if float(camera_quality) < quality_threshold:
                print("Removing Low Quality Camera: ", camera.label)
                camera.enabled = False

        print("Remaining # Cameras: ", len([c for c in chunk.cameras if c.enabled]))
        doc.save()

    # Match the photos by finding common features and establishing correspondences.
    if not chunk.tie_points:
        chunk.matchPhotos(keypoint_limit=40000,
                          tiepoint_limit=10000,
                          generic_preselection=True,
                          reference_preselection=True,
                          downscale=1)
        doc.save()

        # Align the cameras to estimate their relative positions in space.
        chunk.alignCameras()
        doc.save()

        # Optimize camera alignment
        chunk.optimizeCameras()
        doc.save()

    # Build depth maps (2.5D representations of the scene) from the aligned photos.
    if chunk.tie_points and not chunk.depth_maps:
        chunk.buildDepthMaps(filter_mode=Metashape.MildFiltering)
        doc.save()

    # Build a 3D model from the depth maps.
    if chunk.depth_maps and not chunk.model:
        chunk.buildModel(source_data=Metashape.DepthMapsData)
        doc.save()

    # Build a orthomosaic from the 3D model.
    if chunk.model and not chunk.orthomosaic:
        chunk.buildOrthomosaic(surface_data=Metashape.ModelData,
                               fill_holes=True)
        doc.save()

    # Export the orthomosaic as a GeoTIFF file if it exists in the chunk.
    if chunk.orthomosaic:

        if not os.path.exists(output_orthomosaic):
            chunk.exportRaster(output_orthomosaic, source_data=Metashape.OrthomosaicData)

        # Identifying a cameras's origin (0, 0)
        x_pixels = []
        y_pixels = []
        image_names = []

        for camera in chunk.cameras:
            try:
                width, height = camera.sensor.width//2, camera.sensor.height//2
                p = chunk.model.pickPoint(camera.center, camera.unproject(Metashape.Vector((width, height, 1))))
                P = chunk.orthomosaic.crs.project(chunk.transform.matrix.mulp(p))
                x = int((P.x - chunk.orthomosaic.left) / chunk.orthomosaic.resolution)
                y = int((chunk.orthomosaic.top - P.y) / chunk.orthomosaic.resolution)

                # Add a marker to the chunk
                chunk.addMarker(point=p)

            except:
                print("ERROR: Could not project camera ", camera.label)
                x = None
                y = None

            x_pixels.append(x)
            y_pixels.append(y)
            image_names.append(camera.label)

        gdf['image_label'] = image_names
        gdf['x_pixels'] = x_pixels
        gdf['y_pixels'] = y_pixels
        new_reference_path = output_folder + "/new_image_centroids.csv"
        gdf.to_csv(new_reference_path)

        doc.save()

        if os.path.exists(output_orthomosaic) and os.path.exists(new_reference_path):
            # Georeference the orthomosaic
            output_geo_orthomosaic = output_folder + "/GeoOrthomosaic.tif"
            georeference_ortho(output_orthomosaic, output_geo_orthomosaic, reference_path)

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

        if not "GV027" in image_folder:
            continue

        try:
            input_folder = ROOT + image_folder + "/"
            output_folder = input_folder.replace("Geo", "Not_Geo")
            run_sfm_workflow(input_folder, output_folder)

        except Exception as e:
            print(e, "\nIssue with: ", image_folder)
