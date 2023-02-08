from classes.monoplotting import TowLine
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", "-i",
        type=str,
        help="Path to a directory containing underwater towcam imagery.")
    parser.add_argument("--out_dir", "-o",
        type=str,
        help="Path to a directory where output files will be saved.")
    parser.add_argument("--usbl_path", "-u",
        type=str,
        help="Path to a GIS file containing USBL GPS points.")
    parser.add_argument("--datetime_field", "-df",
        type=str,
        default="DateTime",
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
                        default="CaCorAlt",
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
    parser.add_argument("--preview_mode", "-pm",
                        action="store_true",
                        help="If set, the script will run in 'preview mode', which \
                            will process the data set, generate vectors outputs of \
                            the towline trajectory and image footprints, but will \
                            not save any output images. This is useful for quickly \
                            previewing the results of a processing run.")

    # TODO: add verbose


    args = parser.parse_args()

    # Create a TowLine object, which kicks off a processing chain...

    towline = TowLine(args.image_dir, args.out_dir, args.usbl_path,
                    args.datetime_field, args.pdop_field, args.elevation_field,
                    args.filter_quartile, args.process_noise_std,
                    args.measurement_noise_std, args.preview_mode)

    # Dump the GDFs...
    towline.dump_gdfs()

    # Plot the smoothing operation...
    # TODO: make this optional... (verbose?)
    towline.plot_smoothing_operation(save_fig=True)

    # Plot the EXIF-USBL fit...
    towline.plot_usbl_fit(save_fig=True)
