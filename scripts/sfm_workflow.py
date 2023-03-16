import os, sys, time
import Metashape

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

# Check if the number of command-line arguments is less than 3.
# If true, print usage instructions and exit the program with a status code of 1.
if len(sys.argv) < 2:
    print("Usage: general_workflow.py <image_folder>")
    sys.exit(1)

# Get the image folder and output folder paths from command-line arguments.
image_folder = output_folder = sys.argv[1]

# Call the "find_files" function to get a list of photo file paths
# with specified extensions from the image folder.
photos = find_files(image_folder, [".jpg", ".jpeg", ".tif", ".tiff"])

# Create a new Metashape document and save it as a project file in the output folder.
doc = Metashape.Document()
doc.save(output_folder + '/project.psx')

# Create a new chunk (3D model) in the Metashape document.
if doc.chunk is None:
    chunk = doc.addChunk()

# Add the photos to the chunk.
if not chunk.cameras:
    chunk.addPhotos(photos)

# Save the Metashape document.
doc.save()

# Print the number of images that were successfully loaded into the chunk.
print(str(len(chunk.cameras)) + " images loaded")

# Match the photos by finding common features and establishing correspondences.
if not chunk.tie_points:
    chunk.matchPhotos(keypoint_limit=40000, tiepoint_limit=10000, generic_preselection=True, reference_preselection=True)
    doc.save()

    # Align the cameras to estimate their relative positions in space.
    chunk.alignCameras()
    doc.save()

# Build depth maps (2.5D representations of the scene) from the aligned photos.
if chunk.tie_points and not chunk.depth_maps:
    chunk.buildDepthMaps(downscale=2, filter_mode=Metashape.MildFiltering)
    doc.save()

# Build a 3D model from the depth maps.
if chunk.depth_maps and not chunk.model:
    chunk.buildModel(source_data=Metashape.DepthMapsData)
    doc.save()

# Build a orthomosaic from the 3D model.
if chunk.model and not chunk.orthomosaic:
    chunk.buildOrthomosaic(surface_data=Metashape.ModelData)
    doc.save()

# Export the orthomosaic as a GeoTIFF file if it exists in the chunk.
if chunk.orthomosaic:
    chunk.exportRaster(output_folder + '/orthomosaic.tif', source_data=Metashape.OrthomosaicData)

# Print a message indicating that the processing has finished and the results have been saved.
print('Processing finished, results saved to ' + output_folder + '.')
print("Time: ", (time.time() - t0) / 60)

