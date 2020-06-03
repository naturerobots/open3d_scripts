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
        description='Reconstruction of point cloud data (PCD) to a reconstructed mesh using Open3D',
    )

    parser.add_argument('-i', '--input', dest='input', required=True, action='store',
                        help='point cloud input ply file', type=str)
    parser.add_argument('-o', '--output', dest='output', required=True, action='store',
                        help='mesh output ply file', type=str)
    parser.add_argument('--filtered-cloud', dest='filtered_cloud', required=False, type=str)
    parser.add_argument('-f', '--filter', dest='filter', required=False, default=True, action='store_true',
                        help='remove outliers')
    parser.add_argument('--f-points', dest='filter_nb_points', required=False, default=10, action='store',
                        help='filter parameter: The minimum number of neighbour points within the filter radius.',
                        type=check_positive)
    parser.add_argument('--f-radius', dest='filter_radius', required=False, default=0.2, action='store',
                        help='filter parameter: The radius in which to count for the minimum number of points.',
                        type=float)
    parser.add_argument('--depth', dest='depth', required=False, default=14, action='store',
                        help='Maximum depth of the tree that will be used for surface reconstruction. '
                             'Running at depth d corresponds to solving on a grid whose resolution is '
                             'no larger than 2 ^ d x 2 ^ d x 2 ^ d.' 'Note that since the reconstructor'
                             ' adapts the octree to the sampling density, the specified reconstruction '
                             'depth is only an upper bound.',
                        type=check_positive)
    parser.add_argument('-n', '--normals', dest='normals', action='store_true', required=False, default=False,
                        help='esitamte normals')
    parser.add_argument('--n-radius', dest='n_radius', action='store', required=False, default=0.3, type=float,
                        help='radius to consider for the normal estimation.')
    parser.add_argument('--n-max-nn', dest='n_max_nn', action='store', required=False, default=30, type=float,
                        help='maximum number of nearest neighbors for normal estimation.')
    parser.add_argument('--linear-fit', dest='linear_fit', required=False, default=False, action='store_true',
                        help='If true, the reconstructor use linear interpolation to estimate the positions '
                             'of iso-vertices.')
    parser.add_argument('--edge-collapse', dest='edge_collapse', action='store', default=0, type=check_positive,
                        help='Use Quadric edge collapse, specify the target number of triangles')

    args = parser.parse_args()

    cloud = o3d.io.read_point_cloud(args.input)

    if args.filter:
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

    if args.filtered_cloud:
        print("Save filtered cloud to", args.filtered_cloud)
        o3d.io.write_point_cloud(args.filtered_cloud, cloud, print_progress=True)

    print("Reconstructing the PCD to a mesh using poisson...")
    mesh, vec = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=args.depth, linear_fit=args.linear_fit)

    print("Remove degenerated, duplicated, non manifold, and unreferenced triangles, edges and vertices...")
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_non_manifold_edges()
    mesh = mesh.remove_unreferenced_vertices()

    if args.edge_collapse:
        print("Quadric edge collapse to a target triangle number %s" % args.edge_collapse)
        mesh = mesh.simplify_quadric_decimation(args.edge_collapse)

    print("Save mesh to ", args.output, "...")
    o3d.io.write_triangle_mesh(args.output, mesh, print_progress=True)
    print("Successfully saved mesh to ", args.output)
    o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":
    main()
