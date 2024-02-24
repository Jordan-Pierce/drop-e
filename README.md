# Drop-E
Fast georeferencing of underwater towed-camera imagery.

## About the Project
This project facilitates the geolocation of underwater imagery and from towed-camera datasets. This codebase's primary function is to locate, align,
stretch, rotate, resample, and warp a collection of camera images using a set of higher-order accuracy GPS points, ideally
from an ultra-short baseline GPS (USBL GPS). There are several output products that can be generated:

1. A set of georeferenced images that can be viewed together in Geographic Information Software (GIS) along with vector files that show image and towline positions (see example image below).

2. A CSV file that contains per-image exterior orientation parameters, which can be used to improve 3D reconstruction and orthorectification in Structure-from-Motion (SfM) software. We primarily target [Agisoft Metashape](https://www.agisoft.com/). Additionally, the vector files for towline and image locations will also be produced for use in GIS software.

![A map showing the Drop-E example workflow](/assets/drop-e_workflow_example.jpeg)

## Project Description
The main component of this project is the **TowLine** class. TowLines are built from a series of GPS point information, which contains estimated trajectory information. The creation of TowLines can be influenced through user-specified filtering/path-smoothing parameters. 
Underwater camera images can then be affixed to the TowLine based on coincident timestamps, and the exterior orientation parameters (X/Y/Z location, heading, etc.) of each image are then extracted from the TowLine using coincident datetime information. Each image's interior orientation parameters are read from the image's EXIF metadata, and together the interior and exterior orientation parameters allow the image to be georeferenced without per-pixel ground control point information.

Some caveats to be aware of:
1. This project **does not** incorporate seafloor elevation models, which is required to perform full orthorectification. 
2. This project **does not** incorporate pitch/roll/yaw information, we assume each image is perfectly top-down, which can negatively impact pixel alignment.
3. This project **does not** perform any corrections to account for refraction or
distortion which may be introduced by the water column. This can also negatively impact pixel alignment.
4. This project has only been tested on a single combination of camera model, USBL GPS, and field data collection procedures. 

Due to the above, the exact pixel locations may not align perfectly when georeferencing. If perfect pixel-alignment is required, additional processing with SfM software is recommended.

## Installation
This codebase uses several open source Python packages that can be easily installed using [Conda](https://docs.conda.io/en/latest/). A pre-configured Conda environment file is included (conda-environment.yml). 

Once Conda is installed on your local machine, execute the following command from this repository's root directory to re-create the Python environment on your local machine and install the required packages.

```
conda env create -f conda-environment.yml
```

## Usage
All workflows are executed via a command line interface. 

1. Usage information is available from the `-h` flag of the `.py` files in this repository's root directory. Here is an example:

```
python georeference_images.py -h
```

2. Generate georeferenced images. Also generate TowLine GIS files with default TowLine smoothing and GPS PDOP filtering:

```
python georeference_images.py --image_dir /path/to/camera/imagery --usbl_path /path/to/gps/points.gpkg --out_dir /path/to/output/directory --datetime_field GUDateTime --pdop_field Max_PDOP --elevation_field GNSS_Height
```

3. Generate a CSV file of image locations to enhance SfM workflows. Also generate TowLine GIS files with custom towline smoothing and no GPS PDOP filtering:

```
python metashape_csv.py --image_dir /path/to/camera/imagery --usbl_path /path/to/gps/points.gpkg --out_dir /path/to/output/directory --datetime_field GUDateTime --pdop_field Max_PDOP --elevation_field GNSS_Height --filter_quartile 1.0 --process_noise_std 0.5 --measurement_noise_std 0.10
```

## Limitations
This software is completely experimental, and developed for use in-house at NOAA's National Centers for Coastal Ocean Science (NOAA NCCOS). No warranty is expressed or implied by release of this software. This software is not suitable for high-accuracy applications, such as those related to navigation or saftey.

Users of this software agree that NOAA, NOAA NCCOS, ORBTL AI, and their partners bear no liability or responsibility for any outcomes related to the download, installation, and usage of this software.

## Support and Contributions
This software was originally written by [ORBTL AI](orbtl.ai) with funding, technical support, and guidance from [NOAA NCCOS](https://coastalscience.noaa.gov/)]. NOAA NCCOS and ORBTL AI can not commit to providing future technical support. No updates to this codebase are planned, and this software is offered as-is. User contributions may or may not be considered in the form of pull requests, so please open an issue to discuss potential contributions prior to submission.