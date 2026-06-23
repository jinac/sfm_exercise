"""
Courtyard dataset from https://www.eth3d.net/datasets
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def cam_params_to_mat(params):
    fx, fy, cx, cy = params
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ])

@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: List[float]
    K: np.ndarray

@dataclass
class Point2D:
    x: float
    y: float
    point3d_id: int

@dataclass
class Image:
    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str
    points2d: List[Point2D] = field(default_factory=list)

@dataclass
class Point3D:
    point3d_id: int
    x: float
    y: float
    z: float
    r: int
    g: int
    b: int
    error: float
    tracks: List[Tuple[int, int]] = field(default_factory=list)

class SfMParser:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.cameras: Dict[int, Camera] = {}
        self.images: Dict[int, Image] = {}
        self.points3d: Dict[int, Point3D] = {}

    def parse_cameras(self, filename: str):
        path = os.path.join(self.base_path, filename)
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(p) for p in parts[4:]]
                k = cam_params_to_mat(params)
                self.cameras[camera_id] = Camera(camera_id, model, width, height, params, k)

    def parse_images(self, filename: str):
        path = os.path.join(self.base_path, filename)
        with open(path, 'r') as f:
            all_lines = [line.strip() for line in f if line.strip()]
            # Find where data starts (after comments)
            data_start_idx = 0
            while data_start_idx < len(all_lines) and all_lines[data_start_idx].startswith('#'):
                data_start_idx += 1
            
            current_idx = data_start_idx
            while current_idx < len(all_lines):
                line = all_lines[current_idx]
                if line.startswith('#'):
                    current_idx += 1
                    continue
                
                parts = line.split()
                # The first part of the image info line is usually the number of images if it's a header,
                # but based on our reading, the first real data line might be an ID or a count.
                # Let's check: "38 0.599829... DSC_0323.JPG" -> this is an image.
                # But there was a line: "4: # Number of images: 38..." which was skipped.
                # So we are looking at lines that start with IMAGE_ID.
                
                try:
                    image_id = int(parts[0])
                    qw = float(parts[1])
                    qx = float(parts[2])
                    qy = float(parts[3])
                    qz = float(parts[4])
                    tx = float(parts[5])
                    ty = float(parts[6])
                    tz = float(parts[7])
                    camera_id = int(parts[8])
                    name = parts[9]
                    
                    image = Image(image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name)
                    current_idx += 1
                    
                    # The next line(s) contain the points2d data
                    if current_idx < len(all_lines) and not all_lines[current_idx].startswith('#'):
                        p2d_line = all_lines[current_idx]
                        p2d_parts = p2d_line.split()
                        # Points are triples: x, y, point3d_id
                        for i in range(0, len(p2d_parts), 3):
                            if i + 2 < len(p2d_parts):
                                x = float(p2d_parts[i])
                                y = float(p2d_parts[i+1])
                                pid = int(float(p2d_parts[i+2])) # use float first to handle 123.0
                                image.points2d.append(Point2D(x, y, pid))
                        current_idx += 1
                    
                    self.images[image_id] = image
                except (ValueError, IndexError):
                    # This might be the "Number of images" line if it wasn't a comment
                    current_idx += 1

    def parse_points3d(self, filename: str):
        path = os.path.join(self.base_path, filename)
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                point3d_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                r = int(float(parts[4]))
                g = int(float(parts[5]))
                b = int(float(parts[6]))
                error = float(parts[7])
                
                tracks = []
                for i in range(8, len(parts), 2):
                    if i + 1 < len(parts):
                        img_id = int(float(parts[i]))
                        p2d_idx = int(float(parts[i+1]))
                        tracks.append((img_id, p2d_idx))
                
                self.points3d[point3d_id] = Point3D(point3d_id, x, y, z, r, g, b, error, tracks)

if __name__ == "__main__":
    import sys
    folder = "courtyard/dslr_calibration_undistorted"
    parser = SfMParser(folder)
    parser.parse_cameras("cameras.txt")
    parser.parse_images("images.txt")
    parser.parse_points3d("points3D.txt")
    
    print(f"Loaded {len(parser.cameras)} cameras")
    print(f"Loaded {len(parser.images)} images")
    print(f"Loaded {len(parser.points3d)} 3D points")
    
    if parser.images:
        first_img = next(iter(parser.images.values()))
        print(f"First image: {first_img.name}, Points2D: {len(first_img.points2d)}")
        cam_id = first_img.camera_id
        print(cam_id, parser.cameras[cam_id])
