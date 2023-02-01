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
    parser.add_argument("--datetime_field", "-d",
        type=str,
        default="DateTime",
        help="Name of the field in the USBL attribute table that contains temporal \
                information. Default value is 'DateTime'.")
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

    args = parser.parse_args()

    # Create a TowLine object, which kicks off a processing chain...

    towline = TowLine(args.image_dir, args.out_dir, args.usbl_path,
                    args.datetime_field,
                    args.process_noise_std, args.measurement_noise_std)


    # Plot the smoothing operation...
    # TODO: make this optional... (verbose?)
    towline.plot_smoothing_operation(save_fig=True)

    # Plot the EXIF-USBL fit...
    towline.plot_usbl_fit(save_fig=True)

    # Dump the GDFs...
    towline.dump_gdfs()