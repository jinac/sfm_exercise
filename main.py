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
import points
import frame_graph
import jls_ba

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
        self.images = {}
        # self.frame_feats = {}
        # self.matches = {}
        self.global_points = {}  # Stores { (key_a, key_b): list of correspondences } mapping 2D points across pairs to global coordinates/metadata
        # self.covis_graph = nx.Graph()

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
                img_tensor = load_image(path).cuda()
                frame_feats[key] = extractor.extract(img_tensor)
                self.images[key] = img_tensor.cpu().permute(1, 2, 0).numpy()
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
        return covis_graph

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
        uv_coords, uv_masks, p3d, colors = points.get_ba_data(kpt_graph, frame_feats, self.cams, cam_pose_tree, self.frames_meta, self.images)
        pts_3d = jls_ba.ba(self.cams, cam_pose_tree, self.frames_meta, p3d, uv_coords, uv_masks)

        return pts_3d, colors

sfm = SfM(base_dir="/mnt/e/dataset/sfm/courtyard_dslr_undistorted/courtyard/")
pts_3d, colors = sfm.run()

rr.init("sfm_test", spawn=True)
# rr.log("cam/image_a", rr.Image(sfm.frames[1].permute(1, 2, 0)))
rr.log("world/pts_3d", rr.Points3D(pts_3d, colors=colors, radii=0.01))
