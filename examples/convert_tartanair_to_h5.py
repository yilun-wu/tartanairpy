'''
Author: Claude
Date: 2025-12-10

Script to convert TartanAir V2 data to H5 format.
Saves RGB, Depth, and Pose in a single H5 file per sequence.
Flow is generated on-the-fly by users from depth+pose.

Reads files directly from zip archives without extracting to disk.

Output structure:
- Combined: {env}.{difficulty}.{traj}.{camera}.h5
  - Contains: rgb, depth, pose datasets
'''

import h5py
import hdf5plugin
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from collections import defaultdict

# Add tartanairpy to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tartanair.reader import TartanAirImageReader


class TartanAirH5Converter:
    """
    Converts TartanAir V2 data from zip files to H5 format.

    Combines RGB, Depth, and Pose into a single H5 file per sequence.
    Flow can be generated on-the-fly by users from depth+pose.

    Output structure:
    - {env}.{difficulty}.{traj}.{camera}.h5
      - rgb: (N, H, W, 3) uint8
      - depth: (N, H, W) float32
      - pose: (N, 7) float64 [x, y, z, qx, qy, qz, qw]
    """

    def __init__(self, root_dir, output_dir=None):
        """
        Args:
            root_dir: Root directory containing zip files and extracted data
            output_dir: Output directory for H5 files (defaults to root_dir)
        """
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir) if output_dir else self.root_dir
        self.reader = TartanAirImageReader()

    def read_decode_depth_from_bytes(self, depth_bytes):
        """Read and decode depth from RGBA PNG bytes."""
        depth_array = np.frombuffer(depth_bytes, dtype=np.uint8)
        depth_rgba = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
        depth = depth_rgba.view("<f4")
        return np.squeeze(depth, axis=-1)

    def write_rgbd_pose_h5(self, image_zip, depth_zip, env_name, difficulty, traj_name, h5_path, camera_name):
        """
        Write RGB, Depth, and Pose to a single H5 file by reading directly from zip files.

        Args:
            image_zip: ZipFile object for RGB images
            depth_zip: ZipFile object for depth images
            env_name: Environment name (e.g., "AbandonedFactory")
            difficulty: Difficulty level (e.g., "Data_easy")
            traj_name: Trajectory name (e.g., "P000")
            h5_path: Output H5 file path
            camera_name: Camera identifier
        """
        # Find RGB files in zip for this trajectory
        rgb_prefix = f"{env_name}/{difficulty}/{traj_name}/image_{camera_name}/"
        rgb_files = sorted([name for name in image_zip.namelist()
                           if name.startswith(rgb_prefix) and name.endswith('.png') and not name.endswith('/')])

        # Find depth files in zip for this trajectory
        depth_prefix = f"{env_name}/{difficulty}/{traj_name}/depth_{camera_name}/"
        depth_files = sorted([name for name in depth_zip.namelist()
                             if name.startswith(depth_prefix) and name.endswith('.png') and not name.endswith('/')])

        # Find pose file in zip
        pose_file_path = f"{env_name}/{difficulty}/{traj_name}/pose_{camera_name}.txt"

        if not rgb_files:
            print(f"⚠ No RGB files found in zip for {traj_name}/image_{camera_name}")
            return
        if not depth_files:
            print(f"⚠ No depth files found in zip for {traj_name}/depth_{camera_name}")
            return

        if len(rgb_files) != len(depth_files):
            print(f"⚠ RGB and depth frame counts don't match: {len(rgb_files)} vs {len(depth_files)}")
            return

        num_frames = len(rgb_files)

        # Read pose file from zip
        try:
            with image_zip.open(pose_file_path) as pose_f:
                poses = np.loadtxt(pose_f, dtype=np.float64)
        except KeyError:
            # Try depth zip if not in image zip
            try:
                with depth_zip.open(pose_file_path) as pose_f:
                    poses = np.loadtxt(pose_f, dtype=np.float64)
            except KeyError:
                print(f"⚠ Pose file not found in zips: {pose_file_path}")
                return

        # Read first RGB and depth to get dimensions
        with image_zip.open(rgb_files[0]) as f:
            sample_rgb = Image.open(f)
            W, H = sample_rgb.size

        with depth_zip.open(depth_files[0]) as f:
            depth_bytes = f.read()
            sample_depth = self.read_decode_depth_from_bytes(depth_bytes)

        if sample_depth.shape != (H, W):
            print(f"⚠ RGB and depth dimensions don't match: RGB {(H, W)} vs depth {sample_depth.shape}")
            return

        with h5py.File(h5_path, "w") as h5:
            # File-level attributes
            h5.attrs["camera_id"] = camera_name
            h5.attrs["dataset_name"] = "TartanAirV2"
            h5.attrs["dataset_seq"] = f"{env_name}.{difficulty}.{traj_name}"
            h5.attrs["resolution"] = (H, W)
            h5.attrs["sensor"] = "synthetic"
            h5.attrs["window_duration_us"] = 100000

            # Create RGB dataset
            rgb_dset = h5.create_dataset(
                "rgb",
                shape=(num_frames, H, W, 3),
                dtype=np.uint8,
                chunks=(1, H, W, 3),
                compression=hdf5plugin.Blosc(cname='zstd', clevel=5, shuffle=hdf5plugin.Blosc.BITSHUFFLE)
            )

            # Create depth dataset
            depth_dset = h5.create_dataset(
                "depth",
                shape=(num_frames, H, W),
                dtype=np.float32,
                chunks=(1, H, W),
                shuffle=True,
                compression=hdf5plugin.Blosc(cname='zstd', clevel=5, shuffle=hdf5plugin.Blosc.BITSHUFFLE)
            )

            # Create pose dataset
            pose_dset = h5.create_dataset(
                "pose",
                data=poses,
                chunks=None,
                compression=None
            )

            # Write RGB and depth by reading from zip
            for i in tqdm(range(num_frames), desc=f"Writing RGB+Depth to {h5_path.name}", leave=False):
                # Read and write RGB
                with image_zip.open(rgb_files[i]) as f:
                    rgb_img = Image.open(f)
                    rgb_dset[i] = np.array(rgb_img)

                # Read and write depth
                with depth_zip.open(depth_files[i]) as f:
                    depth_bytes = f.read()
                    depth = self.read_decode_depth_from_bytes(depth_bytes)
                    depth_dset[i] = depth

        print(f"✓ Wrote {num_frames} RGB images, depths, and poses to {h5_path}")

    def process_env_sensor_pair(self, env_name, sensor_name, difficulty="Data_easy"):
        """
        Process a single environment/sensor pair by reading directly from zip files.

        Args:
            env_name: Environment name (e.g., "AbandonedFactory")
            sensor_name: Sensor/camera name (e.g., "lcam_front")
            difficulty: Difficulty level (e.g., "Data_easy")

        Returns:
            True if successful, False otherwise
        """
        print(f"\n{'='*80}")
        print(f"Processing: {env_name}/{difficulty}/{sensor_name}")
        print(f"{'='*80}")

        env_dir = self.root_dir / env_name / difficulty
        if not env_dir.exists():
            print(f"⚠ Environment directory not found: {env_dir}")
            return False

        # Find zip files for this sensor
        depth_zip_path = env_dir / f"depth_{sensor_name}.zip"
        image_zip_path = env_dir / f"image_{sensor_name}.zip"

        if not depth_zip_path.exists() or not image_zip_path.exists():
            print(f"⚠ Missing zip files for {env_name}/{sensor_name}")
            return False

        try:
            # Open zip files
            with zipfile.ZipFile(image_zip_path, 'r') as image_zip, \
                 zipfile.ZipFile(depth_zip_path, 'r') as depth_zip:

                # Discover trajectories from zip contents
                # Look for pose files in the zip to identify trajectories
                pose_pattern = f"{env_name}/{difficulty}/"
                trajectories = set()

                for name in image_zip.namelist():
                    if f"pose_{sensor_name}.txt" in name and name.startswith(pose_pattern):
                        # Extract trajectory name from path like: "AbandonedFactory/Data_easy/P000/pose_lcam_front.txt"
                        parts = Path(name).parts
                        if len(parts) >= 3:
                            traj_name = parts[2]  # P000, P001, etc.
                            if traj_name.startswith('P') and len(traj_name) == 4:
                                trajectories.add(traj_name)

                if not trajectories:
                    print(f"⚠ No trajectories found in {env_name}/{sensor_name}")
                    return False

                print(f"Found {len(trajectories)} trajectories: {sorted(trajectories)}")

                # Process each trajectory
                for traj_name in sorted(trajectories):
                    # Construct output path for combined H5 file
                    # Output directly under root output directory (e.g., /data/tartanair/)
                    h5_name = f"{env_name}.{difficulty}.{traj_name}.{sensor_name}.h5"
                    h5_path = self.output_dir / h5_name

                    try:
                        print(f"\nProcessing trajectory: {traj_name}")

                        # Create output directory
                        h5_path.parent.mkdir(parents=True, exist_ok=True)

                        # Process RGB+Depth+Pose from zip files
                        if not h5_path.exists():
                            self.write_rgbd_pose_h5(
                                image_zip, depth_zip,
                                env_name, difficulty, traj_name,
                                h5_path, sensor_name
                            )
                        else:
                            print(f"⊘ Skipping (already exists): {h5_path.name}")

                    except Exception as e:
                        print(f"✗ Failed to process trajectory {traj_name}: {e}")
                        raise  # Re-raise to allow debugger to catch

            return True

        except Exception as e:
            print(f"✗ Failed to process {env_name}/{sensor_name}: {e}")
            raise  # Re-raise to allow debugger to catch

    def find_all_env_sensor_pairs(self, difficulty="Data_easy"):
        """
        Find all environment/sensor pairs in root directory.

        Args:
            difficulty: Difficulty level to search (e.g., "Data_easy")

        Returns:
            List of (env_name, sensor_name) tuples
        """
        pairs = []
        env_sensor_count = defaultdict(int)

        # Search for all zip files
        for zip_path in self.root_dir.rglob("*.zip"):
            # Check if it's in the difficulty directory
            if difficulty not in zip_path.parts:
                continue

            # Parse zip filename
            # Expected format: depth_SENSOR.zip or image_SENSOR.zip
            filename = zip_path.stem
            if not (filename.startswith("depth_") or filename.startswith("image_")):
                continue

            # Extract environment name (parent of difficulty directory)
            parts = zip_path.parts
            try:
                diff_idx = parts.index(difficulty)
                if diff_idx > 0:
                    env_name = parts[diff_idx - 1]
                    sensor_name = "_".join(filename.split("_")[1:])  # Remove "depth_" or "image_" prefix
                    pair = (env_name, sensor_name)
                    env_sensor_count[pair] += 1
            except (ValueError, IndexError):
                continue

        # Only include pairs that have both depth and image zips (count == 2)
        pairs = [pair for pair, count in env_sensor_count.items() if count >= 2]

        print(f"\nFound {len(pairs)} environment/sensor pairs:")
        for env, sensor in sorted(pairs):
            print(f"  - {env}/{sensor}")

        return pairs

    def process_all_env_sensor_pairs(self, max_workers=4, difficulty="Data_easy"):
        """
        Process all environment/sensor pairs in parallel.

        Args:
            max_workers: Number of parallel workers
            difficulty: Difficulty level to process
        """
        pairs = self.find_all_env_sensor_pairs(difficulty=difficulty)

        if not pairs:
            print("No environment/sensor pairs found!")
            return

        print(f"\nProcessing {len(pairs)} pairs with {max_workers} workers...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_pair = {
                executor.submit(
                    self.process_env_sensor_pair,
                    env_name, sensor_name, difficulty
                ): (env_name, sensor_name)
                for env_name, sensor_name in pairs
            }

            # Wait for completion and report results
            for future in as_completed(future_to_pair):
                env_name, sensor_name = future_to_pair[future]
                try:
                    success = future.result()
                    if success:
                        print(f"✓ Completed: {env_name}/{sensor_name}")
                    else:
                        print(f"✗ Failed: {env_name}/{sensor_name}")
                except Exception as e:
                    print(f"✗ Worker error for {env_name}/{sensor_name}: {e}")
                    raise  # Re-raise to allow debugger to catch


if __name__ == "__main__":
    # Configuration
    ROOT_DIR = Path("/data/tartanair")
    OUTPUT_DIR = ROOT_DIR  # Output to same directory
    DIFFICULTY = "Data_easy"
    MAX_WORKERS = 6

    # Create converter
    converter = TartanAirH5Converter(
        root_dir=ROOT_DIR,
        output_dir=OUTPUT_DIR
    )

    # Option 1: Process a specific environment/sensor pair
    # converter.process_env_sensor_pair(
    #     env_name="AmericanDiner",
    #     sensor_name="lcam_front",
    #     difficulty=DIFFICULTY
    # )

    # Option 2: Process all environment/sensor pairs in parallel
    converter.process_all_env_sensor_pairs(
        max_workers=MAX_WORKERS,
        difficulty=DIFFICULTY
    )
