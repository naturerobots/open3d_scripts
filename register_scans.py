#!python3
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


def load_pointcloud(cloud_id, args, preview=False):
    file = args.scan_pattern.replace('*', str(cloud_id).zfill(args.num_id_digits))

    print(f'Loading point cloud from file {file} ...')
    pointcloud = o3d.io.read_point_cloud(file)
    if preview:
        # filter cloud if it is used for preview
        print('Filtering point cloud ...')
        pointcloud = pointcloud.voxel_down_sample(voxel_size=args.voxel_size)

    if not args.no_outlier_removal:
        pointcloud, ids = pointcloud.remove_radius_outlier(args.filter_nb_points, args.filter_radius)

    if not pointcloud.has_normals():
        print('Estimating normals ...')
        pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    return pointcloud


def decomment(csvfile):
    for row in csvfile:
        raw = row.split('#')[0].strip()
        if raw:
            yield raw


def load_posegraph(args):
    graph = nx.DiGraph()
    pointclouds = {}
    with open(args.graph, newline='') as graph_file:
        os.chdir(args.dataset)

        graph_data = csv.reader(decomment(graph_file), delimiter=';', quotechar='#')
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
                pointclouds[edge[0]] = None
            if not edge[1] in pointclouds:
                pointclouds[edge[1]] = None

            # add edge to networkx graph
            graph.add_edge(edge[0], edge[1])

    return pointclouds, graph


def pick_points(pcd, node_from, node_to, node_current):
    print("")
    print("+--------------------------------------------------------------------------+")
    print("| 1) Please pick at least three correspondences using [shift + left click] |")
    print("|    Press [shift + right click] to undo point picking                     |")
    print("| 2) After picking points, press q for close the window                    |")
    print("|                                                                          |")
    print("| If you messed something up in the first point cloud of the current pair  |")
    print("| of point clouds, simply select no points and press q. By doing this the  |")
    print("| procedure will be restarted for the current pair of point clouds.        |")
    print("+--------------------------------------------------------------------------+")
    print("")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(f'Pick correspondences between point cloud {node_from} and {node_to} - point cloud {node_current}')
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    return vis.get_picked_points()


def get_cloud(pointclouds, args, node_id):
    # TODO remove point clouds afterwards if it is not needed anymore
    if not pointclouds[node_id]:
        pointclouds[node_id] = load_pointcloud(node_id, args)
    return pointclouds[node_id]

def downsample_pointcloud(pcd, voxel_size):
    print(f'{color.BOLD}Downsample with a voxel size %.3f.{color.ENDC}' % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(f'{color.BOLD}Estimate normal with search radius %.3f.{color.ENDC}' % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(f'{color.BOLD}Compute FPFH feature with search radius %.3f.{color.ENDC}' % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def register_pointcloud_pair(node_from, node_to, pointclouds, args):
    temp_dir = args.temp_dir
    run_icp = not args.no_icp
    max_icp_distance = args.icp_max_distance
    show_result = not args.hide_result

    transformation_file = f'{temp_dir}/{node_from}_to_{node_to}.trans'
    information_file = f'{temp_dir}/{node_from}_to_{node_to}.info'

    # already done
    if os.path.isfile(transformation_file) and os.path.isfile(information_file):
        print(f'{color.BOLD}Found registration of {node_from} and {node_to}{color.ENDC}')
        transformation = np.loadtxt(transformation_file)
        icp_information = np.loadtxt(information_file)

        if not args.refine_all and node_to not in args.refine_list:
            return transformation, icp_information
        else:
            print(f'{color.BOLD}Refining registration of {node_from} and {node_to}{color.ENDC}')

    elif args.mode == 'picking':
        # pick at least three correspondences between both pointclouds
        picked_points_from = []
        picked_points_to = []
        while len(picked_points_from) < 3 or len(picked_points_to) < 3 or len(picked_points_from) != len(picked_points_to):
            picked_points_from = pick_points(get_cloud(pointclouds, args, node_from), node_from, node_to, node_from)
            picked_points_to = pick_points(get_cloud(pointclouds, args, node_to), node_from, node_to, node_to)

        correspondences = np.zeros((len(picked_points_from), 2))
        correspondences[:, 0] = picked_points_to
        correspondences[:, 1] = picked_points_from

        # estimate rough transformation using correspondences
        print(f'{color.BOLD}Compute a rough transform using the correspondences given by user ...{color.ENDC}')
        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        transformation = p2p.compute_transformation(
                get_cloud(pointclouds, args, node_to),
                get_cloud(pointclouds, args, node_from),
                o3d.utility.Vector2iVector(correspondences))

        icp_information = np.zeros((6, 6))

    elif args.mode == 'fast_feature':
        distance_threshold = args.voxel_size * 0.5

        source_down, source_fpfh = downsample_pointcloud(get_cloud(pointclouds, args, node_to), args.voxel_size)
        target_down, target_fpfh = downsample_pointcloud(get_cloud(pointclouds, args, node_from), args.voxel_size)

        print(f'{color.BOLD}Apply fast global registration with distance threshold %.3f{color.ENDC}' % distance_threshold)
        result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))

        transformation = result.transformation

    if run_icp:
        print(f'{color.BOLD}Applying point-to-plane ICP ...{color.ENDC}')

        icp_fine = o3d.pipelines.registration.registration_icp(
            get_cloud(pointclouds, args, node_to),
            get_cloud(pointclouds, args, node_from),
            max_icp_distance,
            transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())

        transformation = icp_fine.transformation

        icp_information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            get_cloud(pointclouds, args, node_to),
            get_cloud(pointclouds, args, node_from),
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
        transformed_to = o3d.geometry.PointCloud(get_cloud(pointclouds, args, node_to))
        transformed_to.transform(transformation)

        # show result
        print("")
        print("+--------------------------------------------------------------------------+")
        print("| Take a look at the combined point cloud. If the registration looks bad,  |")
        print("| press B and then q to rerun the registration of the current point clouds.|")
        print("| If everything looks fine, just press q.                                  |")
        print("+--------------------------------------------------------------------------+")
        print("")

        o3d.visualization.draw_geometries_with_key_callbacks(
                [transformed_to, get_cloud(pointclouds, args, node_from)],
                window_name=f'Result of registration between point cloud {node_from} and {node_to}',
                key_to_callback=key_to_callback)

        if bad:
            print(f'{color.WARNING}Retrying registration of point cloud {node_from} and {node_to}.{color.ENDC}')
            # rerun pair registration
            return register_pointcloud_pair(node_from, node_to, pointclouds, args)

    np.savetxt(transformation_file, transformation)
    np.savetxt(information_file, icp_information)

    return transformation, icp_information


def register_pointclouds(pointclouds, nx_pose_graph, args):

    # because every node is a key of the adjacency list, this will give all node ids
    node_id_mapping = {}
    for index, node in enumerate(nx_pose_graph.nodes()):
        node_id_mapping[node] = index

    o3d_pose_graph = o3d.pipelines.registration.PoseGraph()
    for _ in range(0, len(nx_pose_graph.nodes())):
        o3d_pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.identity(4)))

    for node_from, node_to in nx.edge_dfs(nx_pose_graph):
        node_from_id = node_id_mapping[node_from]
        node_to_id = node_id_mapping[node_to]

        # calculate transformation for the current edge of the posegraph
        print(f'{color.BOLD}Starting registration between point cloud {node_from} and {node_to} ...{color.ENDC}')
        transformation, icp_information = register_pointcloud_pair(
                node_from,
                node_to,
                pointclouds,
                args
        )

        # only update pose of the node if the pose has not been set yet. The other possible case is a loop closure
        if (o3d_pose_graph.nodes[node_to_id].pose == np.identity(4)).all():
            o3d_pose_graph.nodes[node_to_id].pose = np.dot(o3d_pose_graph.nodes[node_from_id].pose, transformation)

        # add edge to pose graph
        # if we have a loop closure and didnt update the pose, we mark our edge as uncertain
        print(f'{color.BOLD}Adding new edge from point cloud {node_from} (id {node_from_id}) to point cloud {node_to} (id {node_to_id}){color.ENDC}')
        o3d_pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(node_to_id,
                                           node_from_id,
                                           transformation,
                                           icp_information,
                                           uncertain=(o3d_pose_graph.nodes[node_to_id].pose != np.identity(4)).all()))

    # optimize the pose graph
    if not args.no_optimization:
        print(f'{color.BOLD}Optimizing pose graph ...{color.ENDC}')
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=args.icp_max_distance,
            edge_prune_threshold=2.0,
            preference_loop_closure=0.1,
            reference_node=0)
        o3d.pipelines.registration.global_optimization(
            o3d_pose_graph, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option)
        print(f'{color.BOLD}Finished optimization{color.ENDC}')

    return (o3d_pose_graph, node_id_mapping)


def main():
    parser = argparse.ArgumentParser(description='Register several scans using Open3D',)

    # IO
    parser.add_argument('-i', '--input', dest='dataset', required=True, action='store',
                        help='Dataset directory, point cloud input directory', type=str)

    parser.add_argument('-g', '--graph', dest='graph', required=False, action='store', default="graph.csv",
                        help="csv containing an edgelist describing the graph")

    parser.add_argument('--temp', dest='temp_dir', required=False, default="transformation", action='store',
                        help='directory of the temporary files that store the current progress of the registration. \
                            (inside your dataset directory) default: "transformation"', type=str)

    scan_file_pattern = "scan_*.ply"
    parser.add_argument('--scan-file-pattern', dest='scan_pattern', action='store', required=False,
                        default=scan_file_pattern, type=str,
                        help='scan file pattern with file extension. default: {}'.format(scan_file_pattern))

    trans_file_pattern = "scan_*.dat"
    parser.add_argument('--trans-file-pattern', dest='trans_pattern', action='store', required=False,
                        default=trans_file_pattern, type=str,
                        help='transformation file pattern with file extension, default: {}'.format(trans_file_pattern))

    parser.add_argument('--id-digits', dest='num_id_digits', required=False, action='store', default=3,
                        type=int, help='Fill up id with zeros to have at least this amount of digits')

    # filters for preview to filter the rinal cloud, use the filter script
    parser.add_argument('--voxel-size', dest='voxel_size', required=False, default=0.05, action='store',
                        help='down sample voxel-size')
    parser.add_argument('--f-points', dest='filter_nb_points', required=False, default=10, action='store',
                        help='preview filter parameter: The minimum number of neighbour points within the filter radius.',
                        type=check_positive)
    parser.add_argument('--f-radius', dest='filter_radius', required=False, default=0.2, action='store',
                        help='preview filter parameter: The radius in which to count for the minimum number of points.',
                        type=float)
    parser.add_argument('--no-outlier-removal', dest='no_outlier_removal', required=False, default=False,
                        action='store_true',
                        help='use remove_radius_outlier for each pointcloud')

    # optimization
    parser.add_argument('--no-icp', dest='no_icp', required=False, default=False, action='store_true',
                        help='disables the optimization of the registration of two pointclouds using icp')

    parser.add_argument('--icp-max-distance', dest='icp_max_distance', required=False, default=0.2, action='store',
                        help='maximum distance allowed to satisfy icp')

    parser.add_argument('--no-optimization', dest='no_optimization', required=False, default=False, action='store_true',
                        help='disables the optimization of the resulting posegraph')

    parser.add_argument('-R', '--refine-all', dest='refine_all', required=False, default=False, action='store_true',
                        help='this does only work if "--no-icp" is not set')

    parser.add_argument('-r', '--refine-list', dest='refine_list', required=False, default=[], action='store', nargs='+',
                        help='list of nodes whose poses should be refined (this does only work if "--no-icp" is not set)', type=str)




    # miscelaneous options
    parser.add_argument('--show-graph', dest='show_graph', required=False, default=False, action='store_true',
                        help='shows the graph before starting registration')

    parser.add_argument('--hide-result', dest='hide_result', required=False, default=False, action='store_true',
                        help='disables showing the resutlt of the registation of two pointclouds')

    parser.add_argument('-v', '--verbose', dest='verbose', required=False, default=False, action='store_true',
                        help='enable open3d debug output')

    parser.add_argument('-m', '--mode', dest='mode', required=False, action='store', default='picking',
                        help="mode of global registration e.g. \'picking\', \'fast_feature\'")

    args = parser.parse_args()

    if args.verbose:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    print(f'{color.OKBLUE}{color.BOLD}Loading graph and point clouds ...{color.ENDC}')

    # point clouds are initialized as key values dict, where the values will be initialized later
    pointclouds, nx_pose_graph = load_posegraph(args)
    print(f'{color.OKBLUE}{color.BOLD}Finished loading graph and point clouds{color.ENDC}')

    # check pose graph correctness
    if len(list(nx.simple_cycles(nx_pose_graph))) > 0:
        print(f'{color.FAIL}ERROR: There are no circles allowed in the pose graph!{color.ENDC}')
        print('Tf your circle is a loop closure, flip the last edge of the loop.')
        return

    # TODO root node detection might be removed
    root_nodes = [node for node in nx_pose_graph.nodes() if len(list(nx_pose_graph.in_edges(node))) == 0]
    if len(root_nodes) != 1:
        print(f'{color.FAIL}ERROR: There is only one node with in-degree 1 (origin node) allowed in the pose graph!{color.ENDC}')
        return
    root_node = root_nodes[0]

    # show posegraph
    if args.show_graph:
        nx.draw_networkx(nx_pose_graph, nx.spring_layout(nx_pose_graph))
        plt.show()

    print(f'{color.OKBLUE}{color.BOLD}Registering point clouds ...{color.ENDC}')
    # create temp dir
    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir)

    # start registration
    o3d_pose_graph, node_id_mapping = register_pointclouds(
            pointclouds,
            nx_pose_graph,
            args)

    print(f'{color.OKBLUE}{color.BOLD}Finished registration of pointclouds{color.ENDC}')

    print(f'{color.OKBLUE}{color.BOLD}Saving results ...{color.ENDC}')
    for node in node_id_mapping.keys():
        node_id = node_id_mapping[node]
        node_text = str(node).zfill(args.num_id_digits)
        np.savetxt(args.trans_pattern.replace('*', node_text), o3d_pose_graph.nodes[node_id].pose)
    print(f'{color.OKBLUE}{color.BOLD}Finished saving results{color.ENDC}')


if __name__ == "__main__":
    main()
