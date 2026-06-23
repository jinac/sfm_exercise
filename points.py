import cv2
import networkx as nx
import numpy as np

def create_keypoint_graph(matches, frame_feats_keys, threshold=0.95):
    kpt_graph = nx.Graph()
    kpt_graph.add_nodes_from(
        [
            (frame_key, idx)
            for idx in range(2048)
            for frame_key in frame_feats_keys
        ]
    )

    for edge, match_result in matches.items():
        match_data = match_result['matches'][0]
        scores = match_result['scores'][0]
        if match_data.shape[0] > 512:
            frame_a, frame_b = edge
            for idx_a, idx_b in match_data[scores > threshold]:
            # for (idx_a, idx_b), score in zip(match_data, scores):
                # if score.cpu().item() >= threshold:
                node_a = (frame_a, idx_a.cpu().item())
                node_b = (frame_b, idx_b.cpu().item())
                # kpt_graph.add_node(node_a)
                # kpt_graph.add_node(node_b)
                kpt_graph.add_edge(node_a, node_b)

    return kpt_graph

def get_filtered_groups(kpt_graph):
    comp = [x for x in nx.connected_components(kpt_graph)
            if len(x) > 2]
    new_comp = []
    for pt_group in comp:
        num_nodes = len(pt_group)
        num_edges = [len(kpt_graph.edges(x)) for x in pt_group]
        if sum([num_nodes - 1 == num_edge for num_edge in num_edges]): # filter out disjointed connected components
            # print(pt_group)
            # print(len(pt_group), [nx.cycle_basis(kpt_graph, x) for x in pt_group],[len(kpt_graph.edges(x)) for x in pt_group])
            new_comp.append(pt_group)

    return new_comp

def get_ba_data(kpt_graph, frame_feats, cams_meta, cam_pose_tree, frames_meta):
    comp = get_filtered_groups(kpt_graph)
    num_pts_3d = len(comp)
    pts_3d = np.zeros((num_pts_3d, 3))
    uv = {
        k: [np.zeros((num_pts_3d, 2)), np.zeros(num_pts_3d)]
        for k in frame_feats.keys()
    }
    counts = []
    for pts_3d_idx, pt_group in enumerate(comp):
        if len(pt_group) > 1:
            counts.append(len(pt_group))
            
            # Collect projection matrices and points for all frames in this group
            projection_matrices = []
            pts_2d_list = []
            
            for frame_key, kpt_idx in pt_group:
                # Get 2D coordinate
                uv_coord = frame_feats[frame_key]['keypoints'][0][kpt_idx].cpu().numpy()
                pts_2d_list.append(uv_coord)
                
                # Get Projection Matrix P = K [R | t]
                frame_meta = frames_meta[frame_key]
                cam = cams_meta[frame_meta.camera_id]
                K = cam.K
                
                R = cam_pose_tree.nodes[frame_key]['R']
                t = cam_pose_tree.nodes[frame_key]['t']
                # print(R.shape, t.shape)
                
                P = K @ np.hstack([R, t.reshape(3, 1)])
                projection_matrices.append(P)
                
                # Also populate uv for BA
                uv[frame_key][0][pts_3d_idx, :] = uv_coord
                uv[frame_key][1][pts_3d_idx] = 1

            # Triangulate using the first two projection matrices in the group
            if len(projection_matrices) >= 2:
                P1 = projection_matrices[0]
                P2 = projection_matrices[1]
                # pts_2d_list[0] is (2,), needs to be (1, 2) for cv2.triangulatePoints
                pts1 = np.array([pts_2d_list[0]]).T 
                pts2 = np.array([pts_2d_list[1]]).T
                
                # Wait, cv2.triangulatePoints expects (2, N) for pts1 and pts2
                pts1 = pts_2d_list[0].reshape(2, 1)
                pts2 = pts_2d_list[1].reshape(2, 1)
                
                pts_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
                pts_3d_homo = pts_4d[:3, :] / pts_4d[3, :]
                pts_3d[pts_3d_idx] = pts_3d_homo.flatten()

    return (uv, pts_3d)
