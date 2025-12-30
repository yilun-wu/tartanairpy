#!/usr/bin/env python3
"""
Generate backward optical flow for TartanAir dataset.

This script generates backward optical flow from forward flow, depth maps, and camera poses.
The backward flow is computed by warping the forward flow from frame t+1 to frame t.

Usage:
    python generate_backward_flow.py --data-root /path/to/tartanair \
                                      --env Downtown \
                                      --difficulty easy \
                                      --trajectory P000 \
                                      --camera lcam_front
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tartanair.reader import TartanAirImageReader
from tartanair.flow_calculation import depthmap_to_camera_coordinates
from scipy.spatial.transform import Rotation as R


class BackwardFlowGenerator:
    """Generate backward optical flow for TartanAir dataset."""

    def __init__(self, data_root):
        """
        Initialize the backward flow generator.

        Args:
            data_root: Root directory of TartanAir dataset
        """
        self.data_root = Path(data_root)
        self.reader = TartanAirImageReader()

        # Camera intrinsics for TartanAir (pinhole cameras with 90 deg FOV)
        # Image size is typically 640x640
        self.width = 640
        self.height = 640
        self.focal_length = self.width / 2.0  # For 90 degree FOV
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

    def get_camera_intrinsics(self):
        """Get camera intrinsics matrix."""
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = self.focal_length  # fx
        K[1, 1] = self.focal_length  # fy
        K[0, 2] = self.cx  # cx
        K[1, 2] = self.cy  # cy
        return K

    def load_pose(self, pose_file):
        """
        Load camera pose from file.

        Args:
            pose_file: Path to pose file

        Returns:
            Pose array with shape (N, 7) where each row is [x, y, z, qx, qy, qz, qw]
        """
        if not pose_file.exists():
            return None

        pose_data = np.loadtxt(pose_file)
        return pose_data

    def get_trajectory_path(self, env, difficulty, trajectory_id):
        """Get path to trajectory directory."""
        difficulty_dir = f"Data_{difficulty}"
        traj_path = self.data_root / env / difficulty_dir / trajectory_id
        return traj_path

    def compute_backward_flow_from_forward(self, flow_fwd, depth_t, depth_t1,
                                          pose_t, pose_t1, K):
        """
        Compute backward optical flow by warping forward flow.

        This uses a simple warping approach:
        1. Use forward flow to find correspondences from t to t+1
        2. Invert these correspondences to get backward flow from t+1 to t

        Args:
            flow_fwd: Forward optical flow (H, W, 2) from frame t to t+1
            depth_t: Depth map at frame t (H, W)
            depth_t1: Depth map at frame t+1 (H, W)
            pose_t: Camera pose at frame t (7,) - [x, y, z, qx, qy, qz, qw]
            pose_t1: Camera pose at frame t+1 (7,)
            K: Camera intrinsics (3, 3)

        Returns:
            flow_bwd: Backward optical flow (H, W, 2) from frame t+1 to t
            mask_bwd: Validity mask for backward flow (H, W)
        """
        H, W = flow_fwd.shape[:2]

        # Create meshgrid
        u, v = np.meshgrid(np.arange(W), np.arange(H))

        # Get pixel coordinates in frame t+1 using forward flow
        u_t1 = u + flow_fwd[:, :, 0]
        v_t1 = v + flow_fwd[:, :, 1]

        # Create output arrays
        flow_bwd = np.zeros((H, W, 2), dtype=np.float32)
        mask_bwd = np.zeros((H, W), dtype=np.uint8)

        # For each pixel in frame t+1, find where it came from in frame t
        # This is done by inverting the forward flow mapping

        # Round coordinates to nearest integer
        u_t1_int = np.round(u_t1).astype(np.int32)
        v_t1_int = np.round(v_t1).astype(np.int32)

        # Check bounds
        valid = (u_t1_int >= 0) & (u_t1_int < W) & (v_t1_int >= 0) & (v_t1_int < H)
        valid = valid & (depth_t > 0) & (depth_t1[v_t1_int.clip(0, H-1), u_t1_int.clip(0, W-1)] > 0)

        # Compute backward flow (from t+1 to t)
        # If pixel at (u_t1, v_t1) in frame t+1 came from (u, v) in frame t,
        # then the backward flow at (u_t1, v_t1) is (u - u_t1, v - v_t1)
        bwd_u = u - u_t1
        bwd_v = v - v_t1

        # Splat backward flow to output image
        valid_idx = np.where(valid)
        u_t1_valid = u_t1_int[valid_idx]
        v_t1_valid = v_t1_int[valid_idx]

        # Use a simple splatting approach (last write wins)
        # For better quality, could use weighted averaging
        flow_bwd[v_t1_valid, u_t1_valid, 0] = bwd_u[valid_idx]
        flow_bwd[v_t1_valid, u_t1_valid, 1] = bwd_v[valid_idx]
        mask_bwd[v_t1_valid, u_t1_valid] = 1

        # Fill holes using nearest neighbor interpolation
        if np.sum(mask_bwd) > 0:
            flow_bwd = self.fill_holes(flow_bwd, mask_bwd)

        return flow_bwd, mask_bwd

    def compute_backward_flow_geometric(self, depth_t, depth_t1, pose_t, pose_t1, K):
        """
        Compute backward optical flow using geometric reprojection.

        This method computes backward flow directly by:
        1. Unprojecting pixels in frame t+1 to 3D using depth
        2. Transforming to frame t coordinate system
        3. Projecting back to frame t image plane

        Args:
            depth_t: Depth map at frame t (H, W)
            depth_t1: Depth map at frame t+1 (H, W)
            pose_t: Camera pose at frame t (7,) - [x, y, z, qx, qy, qz, qw]
            pose_t1: Camera pose at frame t+1 (7,)
            K: Camera intrinsics (3, 3)

        Returns:
            flow_bwd: Backward optical flow (H, W, 2) from frame t+1 to t
            mask_bwd: Validity mask for backward flow (H, W)
        """
        H, W = depth_t1.shape

        # Convert poses to transformation matrices
        def pose_to_matrix(pose):
            T = np.eye(4, dtype=np.float32)
            T[:3, 3] = pose[:3]  # translation
            quat = pose[3:]  # quaternion [x, y, z, w]
            T[:3, :3] = R.from_quat(quat).as_matrix()
            return T

        T_world_t = pose_to_matrix(pose_t)
        T_world_t1 = pose_to_matrix(pose_t1)

        # Compute relative transformation from t+1 to t
        T_t_world = np.linalg.inv(T_world_t)
        T_t_t1 = T_t_world @ T_world_t1

        # Unproject pixels in frame t+1 to 3D camera coordinates
        X_cam_t1, valid_depth = depthmap_to_camera_coordinates(depth_t1, K)

        # Transform to frame t
        X_cam_t1_homo = np.concatenate([X_cam_t1, np.ones((H, W, 1))], axis=-1)
        X_cam_t = (X_cam_t1_homo @ T_t_t1.T)[:, :, :3]

        # Project to frame t image plane
        x = X_cam_t[:, :, 0]
        y = X_cam_t[:, :, 1]
        z = X_cam_t[:, :, 2]

        u_t = self.focal_length * x / z + self.cx
        v_t = self.focal_length * y / z + self.cy

        # Create meshgrid for frame t+1
        u_t1, v_t1 = np.meshgrid(np.arange(W), np.arange(H))

        # Compute backward flow
        flow_bwd = np.stack([u_t - u_t1, v_t - v_t1], axis=-1).astype(np.float32)

        # Compute validity mask
        mask_bwd = valid_depth.astype(np.uint8)
        mask_bwd &= (z > 0)
        mask_bwd &= (u_t >= 0) & (u_t < W) & (v_t >= 0) & (v_t < H)

        # Mask out invalid flow
        flow_bwd[~mask_bwd] = 0

        return flow_bwd, mask_bwd

    def fill_holes(self, flow, mask):
        """
        Fill holes in flow using nearest neighbor interpolation.

        Args:
            flow: Flow field (H, W, 2)
            mask: Validity mask (H, W)

        Returns:
            Filled flow field (H, W, 2)
        """
        if np.sum(mask) == 0:
            return flow

        # Use inpainting to fill holes
        filled_u = cv2.inpaint(flow[:, :, 0], (1 - mask).astype(np.uint8), 3, cv2.INPAINT_NS)
        filled_v = cv2.inpaint(flow[:, :, 1], (1 - mask).astype(np.uint8), 3, cv2.INPAINT_NS)

        return np.stack([filled_u, filled_v], axis=-1)

    def process_trajectory(self, env, difficulty, trajectory_id, camera_name,
                          method='geometric', overwrite=False):
        """
        Process a trajectory and generate backward flow for all frames.

        Args:
            env: Environment name (e.g., 'Downtown')
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            trajectory_id: Trajectory ID (e.g., 'P000')
            camera_name: Camera name (e.g., 'lcam_front')
            method: Method to use ('geometric' or 'forward_warp')
            overwrite: Whether to overwrite existing backward flow
        """
        traj_path = self.get_trajectory_path(env, difficulty, trajectory_id)

        if not traj_path.exists():
            print(f"Trajectory not found: {traj_path}")
            return

        # Get paths
        depth_dir = traj_path / f"depth_{camera_name}"
        flow_dir = traj_path / f"flow_{camera_name}"
        pose_file = traj_path / f"pose_{camera_name}.txt"

        if not depth_dir.exists():
            print(f"Depth directory not found: {depth_dir}")
            return

        if not flow_dir.exists():
            print(f"Flow directory not found: {flow_dir}")
            return

        # Load poses
        poses = self.load_pose(pose_file)
        if poses is None:
            print(f"Pose file not found: {pose_file}")
            return

        # Get camera intrinsics
        K = self.get_camera_intrinsics()

        # Get list of flow files
        flow_files = sorted(flow_dir.glob("*.npz"))
        if len(flow_files) == 0:
            flow_files = sorted(flow_dir.glob("*.png"))

        print(f"Processing {len(flow_files)} flow files in {traj_path}")
        print(f"Method: {method}")

        # Process each flow file
        for flow_file in tqdm(flow_files, desc=f"Generating backward flow"):
            # Parse frame indices from filename
            # Format: XXXXXX_YYYYYY_*_flow.npz or XXXXXX_YYYYYY_*_flow.png
            filename = flow_file.stem
            parts = filename.split('_')

            if len(parts) >= 2:
                frame_t = int(parts[0])
                frame_t1 = int(parts[1])
            else:
                print(f"Skipping file with unexpected format: {flow_file.name}")
                continue

            # Check if backward flow already exists
            if flow_file.suffix == '.npz':
                with np.load(flow_file) as data:
                    if 'flow_bwd' in data and not overwrite:
                        continue
                    if 'flow_fwd' in data:
                        flow_fwd = data['flow_fwd'][0].transpose(1, 2, 0)  # CHW to HWC
                    else:
                        print(f"No forward flow found in {flow_file.name}, skipping")
                        continue
            else:
                # Load PNG flow
                flow_fwd = self.reader.read_flow(str(flow_file), direction='fwd')
                if flow_fwd is None:
                    print(f"Failed to read flow from {flow_file.name}, skipping")
                    continue

            # Load depth maps
            depth_file_t = depth_dir / f"{frame_t:06d}_{camera_name}_depth.png"
            depth_file_t1 = depth_dir / f"{frame_t1:06d}_{camera_name}_depth.png"

            if not depth_file_t.exists() or not depth_file_t1.exists():
                print(f"Depth files not found for frames {frame_t}-{frame_t1}, skipping")
                continue

            depth_t = self.reader.read_depth(str(depth_file_t))
            depth_t1 = self.reader.read_depth(str(depth_file_t1))

            if depth_t is None or depth_t1 is None:
                print(f"Failed to read depth for frames {frame_t}-{frame_t1}, skipping")
                continue

            # Get poses
            if frame_t >= len(poses) or frame_t1 >= len(poses):
                print(f"Pose index out of range for frames {frame_t}-{frame_t1}, skipping")
                continue

            pose_t = poses[frame_t]
            pose_t1 = poses[frame_t1]

            # Compute backward flow
            if method == 'geometric':
                flow_bwd, mask_bwd = self.compute_backward_flow_geometric(
                    depth_t, depth_t1, pose_t, pose_t1, K
                )
            elif method == 'forward_warp':
                flow_bwd, mask_bwd = self.compute_backward_flow_from_forward(
                    flow_fwd, depth_t, depth_t1, pose_t, pose_t1, K
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            # Save backward flow
            if flow_file.suffix == '.npz':
                # Update existing npz file
                with np.load(flow_file) as data:
                    flow_data = dict(data.items())

                # Add backward flow (convert to CHW format)
                flow_data['flow_bwd'] = flow_bwd.transpose(2, 0, 1)[np.newaxis, ...]  # HWC to 1CHW
                flow_data['mask_bwd'] = mask_bwd[np.newaxis, ...]  # HW to 1HW

                np.savez_compressed(flow_file, **flow_data)
            else:
                # Save as new npz file
                output_file = flow_file.parent / f"{flow_file.stem}.npz"
                np.savez_compressed(
                    output_file,
                    flow_fwd=flow_fwd.transpose(2, 0, 1)[np.newaxis, ...],  # HWC to 1CHW
                    flow_bwd=flow_bwd.transpose(2, 0, 1)[np.newaxis, ...],  # HWC to 1CHW
                    mask_fwd=np.ones_like(mask_bwd)[np.newaxis, ...],  # Placeholder
                    mask_bwd=mask_bwd[np.newaxis, ...]
                )

        print(f"Backward flow generation complete for {env}/{difficulty}/{trajectory_id}/{camera_name}")


def main():
    parser = argparse.ArgumentParser(description='Generate backward optical flow for TartanAir dataset')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory of TartanAir dataset')
    parser.add_argument('--env', type=str, nargs='+', required=True,
                       help='Environment name(s) (e.g., Downtown Prison)')
    parser.add_argument('--difficulty', type=str, nargs='+', default=['easy'],
                       choices=['easy', 'medium', 'hard'],
                       help='Difficulty level(s)')
    parser.add_argument('--trajectory', type=str, nargs='+', default=['P000'],
                       help='Trajectory ID(s) (e.g., P000 P001)')
    parser.add_argument('--camera', type=str, nargs='+',
                       default=['lcam_front'],
                       help='Camera name(s) (e.g., lcam_front rcam_front)')
    parser.add_argument('--method', type=str, default='geometric',
                       choices=['geometric', 'forward_warp'],
                       help='Method for computing backward flow')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing backward flow')

    args = parser.parse_args()

    # Create generator
    generator = BackwardFlowGenerator(args.data_root)

    # Process each combination
    for env in args.env:
        for difficulty in args.difficulty:
            for trajectory_id in args.trajectory:
                for camera_name in args.camera:
                    print(f"\n{'='*60}")
                    print(f"Processing: {env} / {difficulty} / {trajectory_id} / {camera_name}")
                    print(f"{'='*60}")

                    generator.process_trajectory(
                        env, difficulty, trajectory_id, camera_name,
                        method=args.method, overwrite=args.overwrite
                    )

    print("\n✓ All backward flow generation complete!")


if __name__ == '__main__':
    main()
