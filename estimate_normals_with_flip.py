#!python3
import argparse
import open3d as o3d
import numpy as np
import glob
import os
import re


def main():

    parser = argparse.ArgumentParser(
        description='Transform and combine several scans to one cloud using Open3D',
    )

    parser.add_argument('-i', '--input', dest='dataset', required=True, action='store',
                        help='Dataset directory, point cloud input directory', type=str)
    parser.add_argument('-o', '--output', dest='output', required=True, action='store',
                        help='cloud output directory (file name same as input file, different direction is needed)', type=str)
    parser.add_argument('-n', '--normals', dest='normals', action='store_true', required=False, default=False,
                        help='esitamte normals')
    parser.add_argument('--n-radius', dest='n_radius', action='store', required=False, default=0.3, type=float,
                        help='radius to consider for the normal estimation.')
    parser.add_argument('--n-max-nn', dest='n_max_nn', action='store', required=False, default=30, type=float,
                        help='maximum number of nearest neighbors for normal estimation.')
  
    scan_file_pattern = "scan_*.ply"
    parser.add_argument('--scan-file-pattern', dest='scan_pattern', action='store', required=False,
                        default=scan_file_pattern, type=str,
                        help='scan file pattern with file extension. default: {}'.format(scan_file_pattern))

    args = parser.parse_args()
    owd = os.getcwd()
    os.chdir(args.dataset)
    sf_pattern = re.compile(args.scan_pattern.replace("*", "(.*)"))

    for scan_file in glob.glob(args.scan_pattern):
        os.chdir(args.dataset)
        match = sf_pattern.search(scan_file)
        id = match.group(1)
        scan = o3d.io.read_point_cloud(scan_file)
        if not scan.has_normals():
            if not scan.has_normals():
                print("No normals in the cloud, estimating normals...")
            else:
                print("Estimate normals...")
            scan.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=args.n_radius, max_nn=args.n_max_nn))
            
            for point_id in range(len(scan.points)):
              point = scan.points[point_id]
              normal = scan.normals[point_id]
              #if direction of position and normal is not the same
              if np.dot(normal, point) > 0:
                #flip normal from point
                scan.normals[point_id] = -normal
            
        
        os.chdir(owd)
        print("Save combined cloud to ", args.output, "...")
        o3d.io.write_point_cloud(args.output+"/"+scan_file, scan, print_progress=True)
        print("Successfully saved cloud to ", args.output)


if __name__ == "__main__":
    main()
