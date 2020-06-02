'''
Posegraph:
    Directed graph with a node pointing to all its successors.
    The source node will keep its position and the target nodes will be transformed.
'''

import argparse
import open3d as o3d
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import glob
import os
import re
import csv

class color:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s has to be a positive int value > 0" % value)
    return ivalue

def load_pointcloud(file, args = None, preview = False):
    print(f'Loading pointcloud from file {file} ...')
    pointcloud = o3d.io.read_point_cloud(file)
    if preview:
        # filter cloud if it is used for preview
        print('Filtering pointcloud ...')
        pointcloud = pointcloud.voxel_down_sample(voxel_size=args.voxel_size)
        pointcloud.remove_radius_outlier(args.filter_nb_points, args.filter_radius)

    print('Estimating normals ...')
    pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pointcloud

def load_posegraph(args):
    graph = nx.DiGraph()
    pointclouds = {}
    pose_graph_adjacency = {}
    with open(args.graph, newline='') as graph_file:
        os.chdir(args.dataset)

        graph_data = csv.reader(graph_file, delimiter=';', quotechar='#')
        for edge in graph_data:
            id_from = str(edge[0]).zfill(args.num_id_digits)
            id_to = str(edge[1]).zfill(args.num_id_digits)

            file_from = args.scan_pattern.replace('*', id_from)
            file_to = args.scan_pattern.replace('*', id_to)

            # check if both files exist
            error = False
            if not os.path.isfile(file_from):
                print(f'{color.FAIL}WARNING: "{file_from}" is not a file{color.ENDC}')
                error = True
            if not os.path.isfile(file_to):
                print(f'{color.FAIL}WARNING: "{file_to}" is not a file{color.ENDC}')
                error = True
            if error:
                continue

            # load pointclouds
            if not edge[0] in pointclouds:
                pointclouds[edge[0]] = load_pointcloud(file_from, args=args, preview=True)
            if not edge[1] in pointclouds:
                pointclouds[edge[1]] = load_pointcloud(file_to, args=args, preview=True)

            # add edge to adjacency list
            if not edge[0] in pose_graph_adjacency:
                pose_graph_adjacency[edge[0]] = []
            if not edge[1] in pose_graph_adjacency[edge[0]]:
                pose_graph_adjacency[edge[0]].append(edge[1])
            # make sure to use every node as a key
            if not edge[1] in pose_graph_adjacency:
                pose_graph_adjacency[edge[1]] = []

            # add edge to networkx graph
            graph.add_edge(edge[0], edge[1])

    return pointclouds, pose_graph_adjacency, graph

def pick_points(pcd, node_from, node_to, node_current):
    print("")
    print("+--------------------------------------------------------------------------+")
    print("| 1) Please pick at least three correspondences using [shift + left click] |")
    print("|    Press [shift + right click] to undo point picking                     |")
    print("| 2) Afther picking points, press q for close the window                   |")
    print("|                                                                          |")
    print("| If you messed something up in the first pointcloud of the current pair   |")
    print("| of pointclouds, simply select no points and press q. By doing this the   |")
    print("| procedure will be restarted for the current pair of pointclouds.         |")
    print("+--------------------------------------------------------------------------+")
    print("")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(f'Pick correspondences between pointcloud {node_from} and {node_to} - pointcloud {node_current}')
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    return vis.get_picked_points()

def register_pointcloud_pair(node_from, node_to, pointclouds, run_icp=True, max_icp_distance=0.2, show_result=True):
    pointcloud_from = pointclouds[node_from]
    pointcloud_to = pointclouds[node_to]

    # pick at least three correspondences between both pointclouds
    picked_points_from = []
    picked_points_to = []
    while len(picked_points_from) < 3 or len(picked_points_to) < 3 or len(picked_points_from) != len(picked_points_to):
        picked_points_from = pick_points(pointcloud_from , node_from, node_to, node_from)
        picked_points_to = pick_points(pointcloud_to, node_from, node_to, node_to)

    correspondences = np.zeros((len(picked_points_from), 2))
    correspondences[:, 0] = picked_points_to
    correspondences[:, 1] = picked_points_from

    # estimate rough transformation using correspondences
    print(f'{color.BOLD}Compute a rough transform using the correspondences given by user ...{color.ENDC}')
    p2p = o3d.registration.TransformationEstimationPointToPoint()
    transformation = p2p.compute_transformation(
            pointcloud_to,
            pointcloud_from,
            o3d.utility.Vector2iVector(correspondences))

    icp_information = np.zeros((6, 6))

    if run_icp:
        print(f'{color.BOLD}Applying point-to-plane ICP ...{color.ENDC}')

        icp_fine = o3d.registration.registration_icp(
            pointcloud_to,
            pointcloud_from,
            max_icp_distance,
            transformation,
            o3d.registration.TransformationEstimationPointToPlane())

        transformation = icp_fine.transformation

        icp_information = o3d.registration.get_information_matrix_from_point_clouds(
            pointcloud_to,
            pointcloud_from,
            max_icp_distance,
            icp_fine.transformation)

        print(f'{color.BOLD}Finished point-to-plane ICP{color.ENDC}')

    if show_result:
        # define function to mark current registration as bad
        global bad
        bad = False
        def bad_reg(vis):
            global bad
            bad = True
            print(f'{color.WARNING}Registration has been marked as BAD. Press q to retry!{color.ENDC}')

        key_to_callback = {}
        key_to_callback[ord("B")] = bad_reg

        # transform pointcloud_to
        transformed_to = o3d.geometry.PointCloud(pointcloud_to)
        transformed_to.transform(transformation)

        # show result
        print("")
        print("+--------------------------------------------------------------------------+")
        print("| Take a look at the combined pointcloud. If the registration looks bad,   |")
        print("| press B and then q to rerun the registration of the current pointclouds. |")
        print("| If everything looks fine, just press q.                                  |")
        print("+--------------------------------------------------------------------------+")
        print("")

        o3d.visualization.draw_geometries_with_key_callbacks(
                [transformed_to, pointcloud_from],
                window_name=f'Result of registration between pointcloud {node_from} and {node_to}',
                key_to_callback=key_to_callback)

        if bad:
            print(f'{color.WARNING}Retrying registration of pointcloud {node_from} and {node_to}.{color.ENDC}')
            # rerun pair registration
            return register_pointcloud_pair(node_from, node_to, pointclouds, run_icp, max_icp_distance, show_result)

        return (transformation, icp_information)

def register_pointclouds(pointclouds, pose_graph_adjacency, run_icp=True, max_icp_distance=0.2, show_result=True):
    # because every node is a key of the adjacency list, this will give all node ids
    nodes = [node for node in pose_graph_adjacency.keys()]
    graph_id_mapping = {}
    for idx, id in enumerate(nodes):
        print("id:", id, "idx:", idx)
        graph_id_mapping[id] = idx

    pose_graph = o3d.registration.PoseGraph()

    for node_from, nodes_to in pose_graph_adjacency.items():
        for node_to in nodes_to:
            # calculate transformation for the current edge of the posegraph
            transformation, icp_information = register_pointcloud_pair(
                    node_from,
                    node_to,
                    pointclouds,
                    run_icp,
                    max_icp_distance,
                    show_result)

def main():
    parser = argparse.ArgumentParser(description='Register several scans using Open3D',)

    # IO
    parser.add_argument('-i', '--input', dest='dataset', required=True, action='store',
                        help='Dataset directory, point cloud input directory', type=str)

    parser.add_argument('-g', '--graph', dest='graph', required=False, action='store', default="graph.csv",
                        help="csv containing an edgelist describing the graph")

    scan_file_pattern = "scan_*.ply"
    parser.add_argument('--scan-file-pattern', dest='scan_pattern', action='store', required=False,
                        default=scan_file_pattern, type=str,
                        help='scan file pattern with file extension. default: {}'.format(scan_file_pattern))
    trans_file_pattern = "scan_*.dat"
    parser.add_argument('--trans-file-pattern', dest='trans_pattern', action='store', required=False,
                        default=trans_file_pattern, type=str,
                        help='transformation file pattern with file extension, default: {}'.format(trans_file_pattern))
    parser.add_argument('--id-digits', dest='num_id_digits', required=False, action='store', default=0,
                        type=int, help='Fill up id with zeros to have at least this amount of digits')

    # filters for preview to filter the rinal cloud, use the filter script
    parser.add_argument('-v', '--voxel-size', dest='voxel_size', required=False, default=0.08, action='store',
                        help='down sample voxel-size for preview')
    parser.add_argument('--f-points', dest='filter_nb_points', required=False, default=10, action='store',
                        help='preview filter parameter: The minimum number of neighbour points within the filter radius.',
                        type=check_positive)
    parser.add_argument('--f-radius', dest='filter_radius', required=False, default=0.2, action='store',
                        help='preview filter parameter: The radius in which to count for the minimum number of points.',
                        type=float)

    # miscelaneous options
    parser.add_argument('--show-graph', dest='show_graph', required=False, default=False, action='store_true',
                        help='shows the graph before starting registration')

    args = parser.parse_args()

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    print(f'{color.OKBLUE}{color.BOLD}Loading graph and pointclouds ...{color.ENDC}')
    pointclouds, pose_graph_adjacency, graph = load_posegraph(args)
    print(f'{color.OKBLUE}{color.BOLD}Finished loading graph and pointclouds{color.ENDC}')

    # show posegraph
    if args.show_graph:
        nx.draw_networkx(graph, nx.spring_layout(graph))
        plt.show()

    print(f'{color.OKBLUE}{color.BOLD}Registering pointclouds ...{color.ENDC}')
    #register_pointclouds(pointclouds, pose_graph_adjacency)
    print(f'{color.OKBLUE}{color.BOLD}Finished registration of pointclouds{color.ENDC}')


if __name__ == "__main__":
    main()
