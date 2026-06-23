import cv2
import networkx as nx
import numpy as np

def create_cov_graph(frames_meta, matches):
    covis_graph = nx.Graph()
    covis_graph.add_nodes_from([k for k in frames_meta.keys()])

    for edge, match_data in matches.items():
        frame_a, frame_b = edge
        match = match_data['matches'][0]
        scores = match_data['scores'][0]
        num_matches = len(match[scores > 0.95])
        covis_graph.add_edge(frame_a, frame_b, num_matches=num_matches)
    return covis_graph

def prune_graph(graph, match_thresh=512):
    edges_to_remove = [(u, v) for u, v, d in graph.edges(data=True) if d.get('num_matches') < match_thresh]
    graph.remove_edges_from(edges_to_remove)

def get_strongest_root(graph):
    node, degree = max(graph.degree, key=lambda x:x[1])
    print(nx.is_connected(graph))
    return node, degree

def calc_rel_pose(pts_a, pts_b):
    E, mask = cv2.findEssentialMat(pts_a, pts_b, np.eye(3), cv2.RANSAC)
    _, R, t, _ = cv2.recoverPose(E, pts_a, pts_b)
    return(R, t)

def create_cam_pose_graph(covis_graph, covis_root, matches, frame_feats):
    cam_pose_tree = nx.bfs_tree(covis_graph, source=covis_root)
    cam_pose_tree.nodes[covis_root]["R"] = np.eye(3)
    cam_pose_tree.nodes[covis_root]["t"] = np.zeros((3, 1))
    # Calculate poses through graph
    for src, sink in cam_pose_tree.edges():
        edge_key = tuple(sorted([src, sink]))
        match = matches[edge_key]['matches'][0]
        if src < sink:
            mask_src = match[:, 0]
            mask_sink = match[:, 1]
        else:
            mask_src = match[:, 1]
            mask_sink = match[:, 0]
        pts_src = frame_feats[src]['keypoints'][0][mask_src].cpu().numpy()
        pts_sink = frame_feats[sink]['keypoints'][0][mask_sink].cpu().numpy()
        R, t = calc_rel_pose(pts_src, pts_sink)
        src_R = cam_pose_tree.nodes[src]["R"]
        src_t = cam_pose_tree.nodes[src]["t"]

        cam_pose_tree.nodes[sink]["R"] = R @ src_R
        cam_pose_tree.nodes[sink]["t"] = R @ src_t + t
        # print(R.shape, src_t.shape, t.shape)

    return cam_pose_tree
