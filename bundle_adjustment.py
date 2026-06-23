from typing import NamedTuple

import jax.numpy as jnp
import jax
from jaxopt import LevenbergMarquardt

@jax.jit
def rotvec_to_matrix(rvec):
    """
    Converts a 3D rotation vector (axis-angle) to a 3x3 rotation matrix
    using Rodrigues' formula.
    """
    angle = jnp.linalg.norm(rvec)
    
    # Handle the limit case where the angle is practically zero
    # to avoid division by zero
    is_zero = angle < 1e-8
    
    theta = jnp.where(is_zero, 1.0, angle)
    k = rvec / theta
    
    kx, ky, kz = k[0], k[1], k[2]
    K = jnp.array([
        [0.0, -kz, ky],
        [kz, 0.0, -kx],
        [-ky, kx, 0.0]
    ])
    
    # Rodrigues formula
    # R = I + sin(theta) * K + (1 - cos(theta)) * K^2
    I = jnp.eye(3)
    R = I + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * jnp.matmul(K, K)
    
    # Return identity matrix if angle is zero, else the computed R
    return jnp.where(is_zero, I, R)

@jax.jit
def project_points(cam_params, points_3d):
    fx, fy, cx, cy = cam_params[:4]
    rvec = cam_params[4:7]
    tvec = cam_params[7:]

    K = jnp.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ])
    R = rotvec_to_matrix(rvec)
    C = jnp.hstack((R, jnp.expand_dims(tvec, 1)))
    print(C.shape)

    points_3d = jnp.column_stack((points_3d, jnp.ones(points_3d.shape[0])))
    # proj_2d = jnp.dot(K, C)
    proj_2d = K @ C
    # print(proj_2d.shape)
    proj_2d = proj_2d @ points_3d.T
    # proj_2d = jnp.dot(proj_2d, points_3d)
    proj_2d = proj_2d.T
    return proj_2d[:, :2]

@jax.jit
def calc_proj(cam_intr, cam_extr):
    fx, fy, cx, cy = cam_intr
    rvec = cam_extr[:3]
    tvec = cam_extr[3:]

    K = jnp.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ])
    R = rotvec_to_matrix(rvec)
    C = jnp.hstack((R, jnp.expand_dims(tvec, 1)))
    P = K @ C

    return(P)

@jax.jit
def proj_error(cam_intr, cam_extr, pts_3d, pts_2d, mask_3d):
    P = calc_proj(cam_intr, cam_extr)

    pts_3d_masked = jnp.column_stack((jnp.where(mask_3d[:, None], pts_3d, 0), jnp.ones(mask_3d.shape[0])))

    proj_pts_2d = P @ pts_3d_masked.T
    proj_pts_2d = proj_pts_2d.T[:, :2]
    error = (pts_2d - proj_pts_2d).flatten()
    return error

@jax.jit
def residual_func(params, observations):
    # cameras, points3d = params
    cameras = params[:10]
    points3d_flat = params[10:]
    # print(cameras.shape, points3d_flat.shape)
    points3d = points3d_flat.reshape((points3d_flat.shape[0] // 3, 3))
    u_2d = observations

    # residuals = []
    # for pts_2d in observations:
        # cam_idx, _, pts_2d = pts_2d
    r = u_2d - project_points(cameras, points3d)
    # return r
    return r.flatten()
    # residuals.append(r)

    # return jnp.concatenate(residuals)

# @jax.jit
def residual_func2(params, observations, info):
    num_cams, num_intr, cam_map = info[0]
    num_intr_vals = num_cams * num_intr
    cameras = params[:num_intr_vals]
    cam_intrinsics_params = jnp.split(cameras, num_cams)

    num_frames, num_extr, extr_to_cam, frame_map = info[1]
    num_extr_vals = num_frames * num_extr
    extrinsics = params[num_intr_vals: num_intr_vals + num_extr_vals]
    frame_extrinsics_params = jnp.split(extrinsics, num_frames)

    num_pts, num_chan = info[2]
    pts_3d = params[num_intr_vals + num_extr_vals:]
    pts_3d = pts_3d.reshape((num_pts, num_chan))

    # u_2d, u_mask = observations
    # print(u_2d[0].shape)

    in_intr = []
    in_extr = []
    in_u2d = []
    in_mask = []
    for frame_key in observations.keys():
        u_2d, mask = observations[frame_key]
        cam_key = extr_to_cam[frame_key]

        in_intr.append(cam_intrinsics_params[cam_map[cam_key]])
        # print(frame_key, frame_map[frame_key], len(frame_extrinsics_params))
        in_extr.append(frame_extrinsics_params[frame_map[frame_key]])
        in_u2d.append(jnp.asarray(u_2d))
        in_mask.append(mask)

    in_intr = jnp.asarray(in_intr)
    in_extr = jnp.asarray(in_extr)
    in_u2d = jnp.asarray(in_u2d)
    in_mask = jnp.asarray(in_mask)
    # print(in_intr.shape, in_extr.shape, pts_3d.shape, in_u2d.shape, in_mask.shape)

    batched_residuals = jax.vmap(proj_error, in_axes=[0, 0, None, 0, 0])
    residuals = batched_residuals(in_intr, in_extr, pts_3d, in_u2d, in_mask)
    return jnp.concatenate(residuals)
    # residuals = []
    # for pts_2d in observations:
        # cam_idx, _, pts_2d = pts_2d
    # r = u_2d - project_points(cameras, pts_3d[u_mask])
    # return r
    # return r.flatten()
    # residuals.append(r)

    # return jnp.concatenate(residuals)

def bundle_adjust(initial_cams, initial_points3d, u_2d):
    lm = LevenbergMarquardt(residual_fun=residual_func)
    initial_params = jnp.concatenate([initial_cams, initial_points3d.ravel()])
    sol = lm.run(initial_params, observations=u_2d)
    params = sol.params
    cams = params[:10]
    pts_3d = params[10:]
    pts_3d = pts_3d.reshape((pts_3d.shape[0] // 3, 3))
    return(cams, pts_3d)

def bundle_adjustment(params, info, ba_data):
    # lm = LevenbergMarquardt(residual_fun=residual_func2)

    jnp_params = jnp.asarray(params)
    # print(jnp_params.shape)
    # print(info)

    r = residual_func2(jnp_params, observations=ba_data, info=info)

    # sol = lm.run(jnp_params, observations=ba_data, info=info)

    return r
