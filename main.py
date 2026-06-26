from pathlib import Path

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
from lightglue import match_pair

import cv2
import rerun as rr
import numpy as np
import networkx as nx
import torch

from load_sfm_data import SfMParser
import bundle_adjustment
import points
import frame_graph

def get_images():
    data_dir = Path(".", "courtyard", "images", "dslr_images_undistorted")
    img_paths = sorted([x for x in data_dir.glob("*.JPG")])
    # print(img_paths)
    imgs = [load_image(img_path) for img_path in img_paths]
    return imgs

def get_cams(data_dir):
    # data_dir = Path(".", "courtyard", "dslr_calibration_undistorted")
    # folder = "courtyard/dslr_calibration_undistorted"
    parser = SfMParser(data_dir)
    parser.parse_cameras("cameras.txt")
    parser.parse_images("images.txt")
    parser.parse_points3d("points3D.txt")

    return (parser.cameras, parser.images)

def calc_rel_pose(pts_a, pts_b):
    # print(pts_a.shape, pts_b.shape)
    # F, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, ransacReprojThreshold=3.0, confidence=0.99, maxIters=1000)
    # E = K_b.T @ F @ K_a

    E, mask = cv2.findEssentialMat(pts_a, pts_b, np.eye(3), cv2.RANSAC)
    _, R, t, _ = cv2.recoverPose(E, pts_a, pts_b)
    return(R.T, t)
    # num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H_norm, K_a)

class SfM():
    def __init__(self, base_dir="./courtyard"):
        self.base_dir = Path(base_dir)

        # self.frames = {}
        self.cams = {}
        self.frames_meta = {}
        # self.frame_feats = {}
        # self.matches = {}
        self.global_points = {}  # Stores { (key_a, key_b): list of correspondences } mapping 2D points across pairs to global coordinates/metadata
        # self.covis_graph = nx.Graph()

        # imgs = get_images()
        cams_meta, imgs_meta = get_cams(self.base_dir / "dslr_calibration_undistorted")
        self.cams = cams_meta
        self.frames_meta = imgs_meta
        # self.frames[]

    def get_features(self, frames_meta):
        frame_feats = {}
        with torch.no_grad():
            extractor = SuperPoint().eval().cuda() 
            for key, frame_meta in frames_meta.items():
                path = self.base_dir / "images" / frame_meta.name
                frame = load_image(path).cuda()
                data = extractor.extract(frame)
                frame_feats[key] = data
        del extractor
        torch.cuda.empty_cache()

        return(frame_feats)

    def get_matches(self, frame_feats):
        matches = {}
        with torch.no_grad():
            matcher = LightGlue(features='superpoint').eval().cuda()
            # matcher.compile(mode='reduce-overhead')
            keys = sorted([k for k in self.frames_meta.keys()])
            for idx_a, key_a in enumerate(keys[:-1]):
                feats_a = frame_feats[key_a]
                for key_b in keys[idx_a+1:]:
                    edge = (key_a, key_b)
                    feats_b = frame_feats[key_b]
                    # print(f"Processing {key_a}, {key_b}")
                    match = matcher(
                        {'image0': feats_a,
                         'image1': feats_b}
                    )
                    # print(match.keys())
                    matches[edge] = match
        del matcher
        torch.cuda.empty_cache()

        return(matches)

    def create_cov_graph(self, matches):
        covis_graph = nx.Graph()
        covis_graph.add_nodes_from([k for k in self.frames_meta.keys()])

        for edge, match_data in matches.items():
            frame_a, frame_b = edge
            match = match_data['matches'][0]
            scores = match_data['scores'][0]
            num_matches = len(match[scores > 0.95])
            covis_graph.add_edge(frame_a, frame_b, num_matches=num_matches)
            # print(edge, num_matches)
        return covis_graph
        # max_edge = max(covis_graph.edges(data=True), key=lambda x:x[2]['num_matches'])
        # print(max_edge)

    def get_init_pose(self, frame_feats, matches, covis_graph):
        max_edge = max(covis_graph.edges(data=True), key=lambda x:x[2]['num_matches'])
        frame_a, frame_b, _ = max_edge
        match = matches[(frame_a, frame_b)]['matches'][0]
        # matches = self.matches_cpu[(frame_a, frame_b)]
        # print(matches.shape)
        # print(self.frame_feats[frame_a]['keypoints'].shape)
        pts_a = frame_feats[frame_a]['keypoints'].cpu()[0][match[..., 0].cpu()].numpy()
        pts_b = frame_feats[frame_b]['keypoints'].cpu()[0][match[..., 1].cpu()].numpy()
        # print(match)
        # pts_a = self.keypoints[frame_a][match[:, 0]].numpy()
        # pts_b = self.keypoints[frame_b][match[:, 1]].numpy()
        R, t = calc_rel_pose(pts_a, pts_b)
        # print(R, t)

        cam_a = self.cams[self.frames_meta[frame_a].camera_id]
        cam_b = self.cams[self.frames_meta[frame_b].camera_id]

        K_a = cam_a.K
        K_b = cam_b.K

        P_a = np.zeros((3, 4))
        P_a[:3, :3] = K_a

        R = np.diag([1, -1, -1]) @ R
        rvec_ba,_ = cv2.Rodrigues(R)
        rvec_ba = rvec_ba.flatten()
        R, _ = cv2.Rodrigues(rvec_ba)
        P = np.hstack([R, -t])
        P_b = K_b @ P
        pts_4d = cv2.triangulatePoints(P_a, P_b, pts_a.T, pts_b.T)
        pts_3d = pts_4d[:3, :] / pts_4d[3, :]
        pts_3d = pts_3d.T  # Shape becomes Nx3

        cam_params = np.concatenate([np.array(cam_b.params), rvec_ba, t.flatten()])
        # print(R, t)
        return(R, t, pts_3d, pts_b, cam_params)

    def serialize_params(self, pts_3d, cam_pose_tree):
        num_cams = len(self.cams)
        num_intr = 4
        sorted_cams = sorted(self.cams)
        intrinsics = []
        cam_map = {}
        for idx, k in enumerate(sorted_cams):
            data = self.cams[k]
            intrinsics.extend(data.params)
            cam_map[data.camera_id] = idx
        # print()
        # print(num_cams)
        # print(num_intr)
        # print(cam_map)
        # print(len(intrinsics))
        # print()

        num_frames = len(self.frames_meta)
        num_extr = 6
        extr_to_cam = {}
        frame_map = {}
        extrinsics = []
        for idx, (k, data) in enumerate(sorted(self.frames_meta.items())):
            extr_to_cam[k] = data.camera_id
            frame_map[k] = idx

            R = cam_pose_tree.nodes[k]["R"]
            R = np.diag([1, -1, -1]) @ R
            r, _ = cv2.Rodrigues(R)
            r = r.flatten()
            t = cam_pose_tree.nodes[k]["t"].flatten()
            data = np.stack([*r, *t])

            extrinsics.extend(data)
        # print()
        # print(num_frames)
        # print(num_extr)
        # print(extr_to_cam)
        # print(len(extrinsics))
        # print()

        # num_cam_params = 10 * len(init_cams)
        num_pts = pts_3d.shape[0]
        pts = pts_3d.ravel()

        out_params = np.concatenate([intrinsics, extrinsics, pts])
        out_deserialize = (
            (num_cams, num_intr, cam_map),
            (num_frames, num_extr, extr_to_cam, frame_map),
            (num_pts, 3)
            )

        return(out_params, out_deserialize)
        # params = np

    def deserialize_params(self, params, info):
        pass

    def run(self):
        frame_feats = self.get_features(self.frames_meta)
        matches = self.get_matches(frame_feats)

        covis_graph = self.create_cov_graph(matches)
        covis_graph = nx.maximum_spanning_tree(covis_graph, weight='num_matches')
        node_weights = covis_graph.degree(weight='num_matches')
        covis_root, _ = max(node_weights, key=lambda x:x[1])
        cam_pose_tree = frame_graph.create_cam_pose_graph(covis_graph, covis_root, matches, frame_feats)

        # import matplotlib.pyplot as plt
        # nx.draw(cam_pose_tree, with_labels=True)
        # plt.show()
        kpt_graph = points.create_keypoint_graph(matches, frame_feats.keys())

        # import pickle
        # with open("kpt_graph.pkl", "wb") as f:
        #     pickle.dump(kpt_graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        uv_coords, uv_masks, p3d = points.get_ba_data(kpt_graph, frame_feats, self.cams, cam_pose_tree, self.frames_meta)
        # print(ba_data[1][0][:5])
        # print(ba_data[1][1][:5])
        # params, info = self.serialize_params(p3d, cam_pose_tree)
        # print(params.shape)
        # print(info)
        # print(len(ba_data[1][0]))
        # print(len(ba_data[1][1]))
        # print(p3d.shape) 

        import jls_ba
        pts_3d = jls_ba.ba(self.cams, cam_pose_tree, self.frames_meta, p3d, uv_coords, uv_masks)
        # R, t, pts_3d, pts_2d, cam_params = self.get_init_pose(frame_feats, matches, covis_graph)
        # r = bundle_adjustment.bundle_adjust(cam_params, pts_3d, pts_2d)
        # r = bundle_adjustment.bundle_adjustment(params, info, ba_data)
        # print(r)

sfm = SfM()
sfm.run()

"""
# rr.init("sfm_test", spawn=True)
# # rr.log("cam/image_a", rr.Image(img_a.permute(1, 2, 0)))
# # rr.log("cam/image_a/pts", rr.Points2D(positions=pts_a, colors=(255,0,0),radii=3.0))
# # rr.log("cam/image_b", rr.Image(img_b.permute(1, 2, 0)))
# # rr.log("cam/image_b/pts", rr.Points2D(positions=pts_b, colors=(255,0,0),radii=3.0))
# rr.log("world/pts_3d", rr.Points3D(pts_3d, radii=0.01))

# # rr.log("world/cam_a",
# #        rr.Transform3D(
# #            translation=np.array([0., 0., 0.]),
# #            rotation=rr.Matrix3x3(np.eye(3)),
# #            relation=rr.TransformRElation.ChildFromParent
# #        )
# # )
# img_a = img_a.permute(1, 2, 0).numpy()
# h, w = img_a.shape[:2]
# rr.log(
#     "world/cam_a",
#     rr.Pinhole(
#         image_from_camera=K_a,
#         resolution=[w, h],
#         image_plane_distance=0.5 # Scales the size of the 3D frame visualization
#     )
# )
# rr.log("world/cam_a/image", rr.Image(img_a))

# img_b = img_b.permute(1, 2, 0).numpy()
# rr.log(
#     "world/cam_b",
#     rr.Pinhole(
#         image_from_camera=K_b,
#         resolution=[w, h],
#         image_plane_distance=0.5 # Scales the size of the 3D frame visualization
#     )
# )
# rr.log("world/cam_b",
#        rr.Transform3D(
#            translation=np.squeeze(-t),
#            mat3x3=R,
#        )
# )
# rr.log("world/cam_b/image", rr.Image(img_b))
"""