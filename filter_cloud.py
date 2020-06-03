#!python3
import open3d as o3d
import argparse


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s has to be a positive int value > 0" % value)
    return ivalue


def main():

    parser = argparse.ArgumentParser(
        description='Filtering of point cloud data (PCD) using Open3D',
    )

    parser.add_argument('-i', '--input', dest='input', required=True, action='store',
                        help='point cloud input ply file', type=str)
    parser.add_argument('-o', '--output', dest='output', required=True, action='store',
                        help='mesh output ply file', type=str)
    parser.add_argument('-f', '--filter', dest='filter', required=False, default=True, action='store_true',
                        help='remove outliers')
    parser.add_argument('--f-points', dest='filter_nb_points', required=False, default=10, action='store',
                        help='filter parameter: The minimum number of neighbour points within the filter radius.',
                        type=check_positive)
    parser.add_argument('--f-radius', dest='filter_radius', required=False, default=0.2, action='store',
                        help='filter parameter: The radius in which to count for the minimum number of points.',
                        type=float)
    parser.add_argument('-n', '--normals', dest='normals', action='store_true', required=False, default=False,
                        help='esitamte normals')
    parser.add_argument('--n-radius', dest='n_radius', action='store', required=False, default=0.3, type=float,
                        help='radius to consider for the normal estimation.')
    parser.add_argument('--n-max-nn', dest='n_max_nn', action='store', required=False, default=30, type=float,
                        help='maximum number of nearest neighbors for normal estimation.')

    args = parser.parse_args()

    cloud = o3d.io.read_point_cloud(args.input)

    print("Remove radius outlier...")
    cloud, ids = cloud.remove_radius_outlier(args.filter_nb_points, args.filter_radius)
    print("Removed", len(ids), "points.")

    if args.normals or not cloud.has_normals():
        if not cloud.has_normals():
            print("No normals in the point cloud, estimating normals...")
        else:
            print("Estimate normals...")
        cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.n_radius, max_nn=args.n_max_nn))

    print("Save filtered cloud to", args.output)
    o3d.io.write_point_cloud(args.output, cloud, print_progress=True)


if __name__ == "__main__":
    main()
