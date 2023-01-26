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
        help="Name of the field in the USBL attribute table that contains temporal information.")

    args = parser.parse_args()

    # Create a towline object
    towline = TowLine(args.image_dir, args.out_dir, args.usbl_path)

    # Calculate the trajectory
    towline.calc_usbl_traj(datetime_field=args.datetime_field)

    # Plot the trajectory
    towline.plot_traj()
