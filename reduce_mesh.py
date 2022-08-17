#!python3
import open3d as o3d
import argparse
import numpy as np


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Filtering of point cloud data (PCD) using Open3D',
    )

    parser.add_argument('-i', '--input', dest='input', required=True, action='store',
                        help='point cloud input ply file', type=str)
    parser.add_argument('-o', '--output', dest='output', required=True, action='store',
                        help='mesh output ply file', type=str)
    parser.add_argument('-f', '--faces', dest='faces', default=0.5,
                        help='factor [0, 1] or number ]1,n] of faces to remove')
    parser.add_argument('-q', '--quadric-edge-collapse', dest='quadric', action='store_true', required=False, default=False,
                        help='Use quadric edge collapse method to reduce the number of faces.')
    parser.add_argument('-c', '--cluster-reduction', dest='cluster_reduction', action='store_true', required=False, default=False,
                        help='Use the cluster reduction flag to reduce the non-connected cluster fragment.')
    parser.add_argument('-e', '--edge-length', dest='max_edge_length', default=0,
                        help='the maximum edge length. Edges with a longer edge will be removed')
    parser.add_argument('--input-cloud', dest='input_cloud', default=None,
                        help='Use an input cloud to remove triangles from the mesh which are to fare away')


    args = parser.parse_args()

    print(f"Loading mesh from {args.input} ...")
    mesh = o3d.io.read_triangle_mesh(args.input, print_progress=True)

    num_triangles = int(np.asarray(mesh.triangles).size / 3)
    print(f"The input mesh has {num_triangles} triangles.")

    if args.quadric:
        num_triangles_out = int(args.faces * num_triangles) if 0 < args.faces <= 1 else int(args.faces)
        print(f"Quadric Edge Collapse from {num_triangles} to {num_triangles_out} triangles...")
        mesh_out = mesh.simplify_quadric_decimation(num_triangles_out)
        print(f"Quadric Edge Collapse done")
    else:
        mesh_out = mesh

    if args.input_cloud:
        print(f"reading input cloud to remove triangles which are to far away from the cloud.")
        cloud = o3d.io.read_point_cloud(args.input_cloud, print_progress=True)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cloud, voxel_size=0.6)
        cloud_mask = ~np.asarray(voxel_grid.check_if_included(mesh_out.vertices))
        mesh_out.remove_vertices_by_mask(cloud_mask)
        mesh_out.remove_unreferenced_vertices()

    if args.max_edge_length:
        print("remove triangles with edges longer than", args.max_edge_length)
        vertices = np.asarray(mesh_out.vertices)
        triangles = np.asarray(mesh_out.triangles)
        num_triangles = int(triangles.size / 3)
        mask = np.zeros(num_triangles)

        for i, triangle in enumerate(triangles):
            if i % 100 == 0:
                printProgressBar(i, num_triangles)
            verts = vertices[triangle]
            a = verts[0]
            b = verts[1]
            c = verts[2]
            dist_c = np.linalg.norm(b - a)
            dist_b = np.linalg.norm(a - c)
            dist_a = np.linalg.norm(c - b)
            mask[i] = dist_a > args.max_edge_length or dist_b > args.max_edge_length or dist_c > args.max_edge_length

        mesh_out.remove_triangles_by_mask(mask)
        mesh_out.remove_unreferenced_vertices()

    if args.cluster_reduction:
        # Function that clusters connected triangles, i.e., triangles that are connected via edges are assigned the same
        # cluster index. This function returns an array that contains the cluster index per triangle, a second array
        # contains the number of triangles per cluster, and a third vector contains the surface area per cluster.

        print("Find connected mesh fragments to remove non-connected artefacts...")
        cluster_indices, num_triangles_per_cluster, surface_area_per_cluster = mesh_out.cluster_connected_triangles()

        print(f"Found {len(num_triangles_per_cluster)} clusters.")
        max_index = num_triangles_per_cluster.index(max(num_triangles_per_cluster))
        print(f"Cluster {max_index} has {num_triangles_per_cluster[max_index]} triangles.")
        # for i, num_triangles in enumerate(num_triangles_per_cluster):
        #    if num_triangles > 1000:
        #        print(f"Cluster {i} has {num_triangles} triangles.")

        print(f"Sum of all clusters:{sum(num_triangles_per_cluster)}.")
        indices_to_remove = np.ma.masked_equal(np.asarray(cluster_indices), max_index)

        # This function removes the triangles where triangle_mask is set to true. Call remove_unreferenced_vertices
        # to clean up vertices afterwards.
        mesh_out.remove_triangles_by_mask(indices_to_remove)

        # This function removes vertices from the triangle mesh that are not referenced in any triangle of the mesh.
        mesh_out.remove_unreferenced_vertices()

    print(f"Writing mesh to {args.input} ...")
    o3d.io.write_triangle_mesh(args.output, mesh_out, print_progress=True)


if __name__ == "__main__":
    main()
