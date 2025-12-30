#!/usr/bin/env python3
"""
Example script demonstrating backward optical flow generation and usage.

This example shows how to:
1. Generate backward flow for a trajectory
2. Load and visualize both forward and backward flow
3. Verify flow consistency
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tartanair.reader import TartanAirImageReader
from generate_backward_flow import BackwardFlowGenerator


def flow_to_color(flow, max_flow=None):
    """
    Convert optical flow to color image for visualization.

    Args:
        flow: Optical flow (H, W, 2)
        max_flow: Maximum flow magnitude for normalization

    Returns:
        Color image (H, W, 3) in RGB format
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # Compute flow magnitude and angle
    mag = np.sqrt(u**2 + v**2)
    ang = np.arctan2(v, u)

    # Normalize magnitude
    if max_flow is None:
        max_flow = np.max(mag)

    if max_flow > 0:
        mag = mag / max_flow

    # Create HSV image
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = (ang + np.pi) / (2 * np.pi) * 179  # Hue: angle
    hsv[:, :, 1] = 255  # Saturation: full
    hsv[:, :, 2] = np.clip(mag * 255, 0, 255)  # Value: magnitude

    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb


def visualize_flow_pair(flow_fwd, flow_bwd, frame_idx, save_path=None):
    """
    Visualize forward and backward flow side by side.

    Args:
        flow_fwd: Forward flow (H, W, 2)
        flow_bwd: Backward flow (H, W, 2)
        frame_idx: Frame index
        save_path: Optional path to save visualization
    """
    # Get maximum flow for consistent scaling
    max_flow = max(np.max(np.abs(flow_fwd)), np.max(np.abs(flow_bwd)))

    # Convert flows to color
    flow_fwd_color = flow_to_color(flow_fwd, max_flow)
    flow_bwd_color = flow_to_color(flow_bwd, max_flow)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(flow_fwd_color)
    axes[0].set_title(f'Forward Flow (frame {frame_idx} → {frame_idx+1})')
    axes[0].axis('off')

    axes[1].imshow(flow_bwd_color)
    axes[1].set_title(f'Backward Flow (frame {frame_idx+1} → {frame_idx})')
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def check_flow_consistency(flow_fwd, flow_bwd, threshold=1.0):
    """
    Check consistency between forward and backward flow using cycle consistency.

    Args:
        flow_fwd: Forward flow (H, W, 2)
        flow_bwd: Backward flow (H, W, 2)
        threshold: Threshold for cycle consistency error

    Returns:
        consistency_map: Consistency error map (H, W)
        percentage_consistent: Percentage of pixels that are consistent
    """
    H, W = flow_fwd.shape[:2]

    # Create meshgrid
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Forward warp: from t to t+1
    u_t1 = u + flow_fwd[:, :, 0]
    v_t1 = v + flow_fwd[:, :, 1]

    # Backward warp: from t+1 to t (should return to original position)
    # Sample backward flow at the warped position
    u_t1_int = np.round(u_t1).astype(np.int32)
    v_t1_int = np.round(v_t1).astype(np.int32)

    # Check bounds
    valid = (u_t1_int >= 0) & (u_t1_int < W) & (v_t1_int >= 0) & (v_t1_int < H)

    # Initialize consistency error
    consistency_error = np.full((H, W), np.inf, dtype=np.float32)

    # Compute cycle consistency error for valid pixels
    valid_idx = np.where(valid)
    u_back = u[valid_idx] + flow_fwd[valid_idx[0], valid_idx[1], 0] + \
             flow_bwd[v_t1_int[valid_idx], u_t1_int[valid_idx], 0]
    v_back = v[valid_idx] + flow_fwd[valid_idx[0], valid_idx[1], 1] + \
             flow_bwd[v_t1_int[valid_idx], u_t1_int[valid_idx], 1]

    # Error should be close to 0 if flows are consistent
    consistency_error[valid_idx] = np.sqrt((u_back - u[valid_idx])**2 +
                                          (v_back - v[valid_idx])**2)

    # Calculate percentage of consistent pixels
    consistent_pixels = np.sum(consistency_error < threshold)
    percentage_consistent = 100.0 * consistent_pixels / np.sum(valid)

    return consistency_error, percentage_consistent


def main():
    """Main function demonstrating backward flow usage."""

    # Configuration
    data_root = Path.home() / "tartanair_v2"  # Change this to your data root
    env = "ArchVizTinyHouseDay"
    difficulty = "easy"
    trajectory_id = "P000"
    camera_name = "lcam_front"

    print("="*60)
    print("Backward Optical Flow Example")
    print("="*60)

    # Step 1: Generate backward flow if needed
    print("\nStep 1: Generating backward flow...")
    generator = BackwardFlowGenerator(data_root)

    try:
        generator.process_trajectory(
            env, difficulty, trajectory_id, camera_name,
            method='geometric', overwrite=False
        )
    except Exception as e:
        print(f"Note: Could not generate backward flow: {e}")
        print("Continuing with existing flow files...")

    # Step 2: Load and visualize flows
    print("\nStep 2: Loading and visualizing flows...")
    reader = TartanAirImageReader()

    # Get trajectory path
    difficulty_dir = f"Data_{difficulty}"
    traj_path = data_root / env / difficulty_dir / trajectory_id
    flow_dir = traj_path / f"flow_{camera_name}"

    if not flow_dir.exists():
        print(f"Flow directory not found: {flow_dir}")
        print("Please download flow data first or check the path.")
        return

    # Load a few flow samples
    flow_files = sorted(flow_dir.glob("*.npz"))[:3]  # First 3 flows

    if len(flow_files) == 0:
        print("No .npz flow files found. Backward flow is only available in .npz format.")
        return

    for flow_file in flow_files:
        print(f"\nProcessing {flow_file.name}...")

        # Parse frame index
        frame_idx = int(flow_file.stem.split('_')[0])

        try:
            # Load forward flow
            flow_fwd = reader.read_flow(str(flow_file), direction='fwd')

            # Load backward flow
            flow_bwd = reader.read_flow(str(flow_file), direction='bwd')

            print(f"  Forward flow shape: {flow_fwd.shape}")
            print(f"  Backward flow shape: {flow_bwd.shape}")
            print(f"  Forward flow range: [{flow_fwd.min():.2f}, {flow_fwd.max():.2f}]")
            print(f"  Backward flow range: [{flow_bwd.min():.2f}, {flow_bwd.max():.2f}]")

            # Check consistency
            consistency_error, percentage_consistent = check_flow_consistency(
                flow_fwd, flow_bwd, threshold=1.0
            )
            print(f"  Cycle consistency: {percentage_consistent:.2f}% of pixels have error < 1.0 px")

            # Visualize
            output_dir = Path("./flow_visualizations")
            output_dir.mkdir(exist_ok=True)

            save_path = output_dir / f"flow_pair_{frame_idx:06d}.png"
            visualize_flow_pair(flow_fwd, flow_bwd, frame_idx, save_path)

        except KeyError as e:
            print(f"  Error: {e}")
            print(f"  This file may not have backward flow yet. Run generation first.")
            continue
        except Exception as e:
            print(f"  Error loading flow: {e}")
            continue

    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)
    print("\nUsage in your code:")
    print("  from tartanair.reader import TartanAirImageReader")
    print("  reader = TartanAirImageReader()")
    print("  flow_fwd = reader.read_flow(flow_path, direction='fwd')")
    print("  flow_bwd = reader.read_flow(flow_path, direction='bwd')")


if __name__ == '__main__':
    main()
