#!python3
import argparse
import open3d as o3d
import numpy as np
import glob
import os
import gc
import re


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s has to be a positive int value > 0" % value)
    return ivalue


def main():

    parser = argparse.ArgumentParser(
        description='Transform and combine several scans to one cloud using Open3D',
    )

    parser.add_argument('-i', '--input', dest='dataset', required=True, action='store',
                        help='Dataset directory, point cloud input directory', type=str)
    parser.add_argument('-o', '--output', dest='output', required=True, action='store',
                        help='cloud output ply file', type=str)
    parser.add_argument('-v', '--voxel-size', dest='voxel_size', required=False, default=0.08, action='store',
                        help='down sample voxel-size. 0 for no downsampling', type=float)
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
                        
    parser.add_argument('--no-outlier-removal', dest='no_outlier_removal', required=False, default=False,
                        action='store_true',
                        help='use remove_radius_outlier for each pointcloud')

    scan_file_pattern = "scan_*.ply"
    parser.add_argument('--scan-file-pattern', dest='scan_pattern', action='store', required=False,
                        default=scan_file_pattern, type=str,
                        help='scan file pattern with file extension. default: {}'.format(scan_file_pattern))
    trans_file_pattern = "scan_*.dat"
    parser.add_argument('--trans-file-pattern', dest='trans_pattern', action='store', required=False,
                        default=trans_file_pattern, type=str,
                        help='transformation file pattern with file extension, default: {}'.format(trans_file_pattern))

    args = parser.parse_args()
    owd = os.getcwd()
    os.chdir(args.dataset)
    sf_pattern = re.compile(args.scan_pattern.replace("*", "(.*)"))
    cloud = o3d.geometry.PointCloud()

    for scan_file in glob.glob(args.scan_pattern):
        match = sf_pattern.search(scan_file)
        id = match.group(1)
        trans_file = args.trans_pattern.replace("*", id)
        if not os.path.isfile(trans_file):
            print("Transformation file {} does not exist. Skip scan file {}.".format(trans_file, scan_file))
            continue
        trans = np.loadtxt(trans_file)
        print("read {} and transform with {}".format(scan_file, trans_file))
        scan = o3d.io.read_point_cloud(scan_file).transform(trans)
        if args.normals or not scan.has_normals():
            if not scan.has_normals():
                print("No normals in the cloud, estimating normals...")
            else:
                print("Estimate normals...")
            scan.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=args.n_radius, max_nn=args.n_max_nn))
        print("add all points to the combined cloud")
        cloud += scan
        del scan
        
    gc.collect()

    if args.voxel_size > 0:
        print("Down sample combined cloud with voxel_size", args.voxel_size)
        cloud = cloud.voxel_down_sample(voxel_size=args.voxel_size)
        print("max bound", cloud.get_max_bound(), "min_bound", cloud.get_min_bound())

    if not args.no_outlier_removal:
        print("Remove radius outlier...")
        cloud, ids = cloud.remove_radius_outlier(args.filter_nb_points, args.filter_radius)
        print("Removed", len(ids), "points.")

    os.chdir(owd)

    print("Save combined cloud to ", args.output, "...")
    o3d.io.write_point_cloud(args.output, cloud, print_progress=True)
    print("Successfully saved cloud to ", args.output)


if __name__ == "__main__":
    main()
