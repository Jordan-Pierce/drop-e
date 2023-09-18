# TODO Get database containing PDOP fields :/
# TODO Standardize the naming convention between field names, arg.parse default values, and function default values

from classes.monoplotting import TowLine
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", "-i",
                        type=str,
                        default="/Users/ross/Documents/git/drop-e_inputs/Sites_ross2023/GV241",
                        help="Path to a directory containing underwater towcam imagery.")

    parser.add_argument("--out_dir", "-o",
                        type=str,
                        default="/Users/ross/Documents/git/drop-e_outputs/Sep1/GV241_test2",
                        help="Path to a directory where output files will be saved.")

    parser.add_argument("--usbl_path", "-u",
                        default="/Users/ross/Documents/git/drop-e_inputs/Sites_ross2023/GV241/GV241.gpkg",
                        type=str,
                        help="Path to a GIS file containing USBL GPS points.")

    parser.add_argument("--datetime_field", "-df",
                        type=str,
                        default="GUDateTime",
                        help="Name of the field in the USBL attribute table that contains temporal \
                information. Default value is 'DateTime'.")

    parser.add_argument("--pdop_field", "-pf",
                        type=str,
                        default=None,
                        help="Name of the field in the USBL attribute table that contains some form \
            of positional accuracy, most often in the form of a PDOP value. Default \
            value is 'Max_PDOP'.")

    parser.add_argument("--elevation_field", "-ef",
                        type=str,
                        default="CaAltCor_m",
                        help="Name of the field in the USBL attribute table that \
                        contains elevation (depth) values. Default value is \
                        'CaCorAlt'.")

    parser.add_argument("--filter_quartile", "-f",
                        type=float,
                        default=0.95,
                        help="A value between 0.0 and 1.0 that controls the filtering of USBL points \
                based on a user-specified --pdop_field (-pf). If set to 1.0 then no \
                filtering will be applied. Default value is 0.95 (keep 95% of values).")

    parser.add_argument("--process_noise_std", "-p",
                        type=float,
                        default=1.0,
                        help="One of two parameters that controls the smoothing of tracklines. If set \
                to 0.0, no smoothing will be applied. Default value is 1.0.")

    parser.add_argument("--measurement_noise_std", "-m",
                        type=float,
                        default=0.25,
                        help="One of two parameters that controls the smoothing of tracklines. If set \
                to 0.0, no smoothing will be applied. Default value is 0.25.")
    
    parser.add_argument("--metashape_csv_sort_order", "-ms",
                        type=str,
                        default="DateTime",
                        choices={"DateTime", "Easting", "Northing"},
                        help="The field to sort the metashape csv file by. Default value is 'DateTime'.\
                                The options are 'DateTime', 'Easting', or 'Northing'")

    # TODO: add verbose

    args = parser.parse_args()

    # Create a TowLine object, which kicks off a pre-defined processing chain based on
    # the input arguments. This processing chain computes vector files (GDFs) and meta-
    # data (stored in GDF attribute tables) that pertain to georeferencing of imagery.

    towline = TowLine(args.image_dir,
                      args.out_dir,
                      args.usbl_path,
                      args.datetime_field,
                      args.pdop_field,
                      args.elevation_field,
                      args.filter_quartile,
                      args.process_noise_std,
                      args.measurement_noise_std,
                      args.metashape_csv_sort_order)

    # Currently a TowLine object has the following exposed methods:
    #  - write_georeferenced_images() - to write georeferenced images to disk
    #  - dump_gdfs() - to dump all vector GDFs to disk
    #  - plot_smoothing_operation() - generates a plot of the trackline smoothing operation
    #    which is useful for debugging and testing
    #  - plot_usbl_fit() - generates a plot showing where USBL points are falling on the
    #    trackline, which is useful for debugging and testing
    #  - plot_rotate() - generates a plot showing the rotation of points about an axis.
    #    This is useful for debugging and testing

    # TO ADD:
    #  - plot_georeferencing() - generates a plot showing the footprints of georeferenced
    #    images on a map. This is useful for debugging and testing
    #  - write_orthorectied_images() - to write orthorectified images to disk (once a
    #    DEM / depth mask is available)
    #  - dump_XXXX() - expose the commands to save individual GDFs to disk, rather than
    #    just dumping all of them at once

    # Write the metashape csv file..
    towline.write_metashape_csv()
    
    # Dump the GDFs...
    towline.dump_gdfs()

    # Write the georeferenced images...
    towline.write_georeferenced_images()
