import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import cv2

class LeRobotDatasetViewer:
    """
    A comprehensive viewer for LeRobot dataset parquet files.
    Reads both the main data file and episodes metadata file.
    """

    def __init__(self, dataset_path: str):
        """
        Initialize the dataset viewer.

        Args:
            dataset_path: Path to the dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.data_file = self.dataset_path / "data" / "chunk-000" / "file-000.parquet"
        self.episodes_file = self.dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        self.info_file = self.dataset_path / "meta" / "info.json"
        self.stats_file = self.dataset_path / "meta" / "stats.json"

        # Load metadata
        self.info = self._load_json(self.info_file)
        self.stats = self._load_json(self.stats_file)

        # Load parquet data
        self.data_df = None
        self.episodes_df = None

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file safely."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            return {}

    def load_data(self):
        """Load both parquet files."""
        print("Loading dataset files...")

        # Load main data file
        if self.data_file.exists():
            self.data_df = pd.read_parquet(self.data_file)
            print(f"✓ Loaded main data: {len(self.data_df)} rows")
        else:
            print(f"✗ Main data file not found: {self.data_file}")

        # Load episodes metadata
        if self.episodes_file.exists():
            self.episodes_df = pd.read_parquet(self.episodes_file)
            print(f"✓ Loaded episodes metadata: {len(self.episodes_df)} rows")
        else:
            print(f"✗ Episodes file not found: {self.episodes_file}")

    def print_dataset_info(self):
        """Print comprehensive dataset information."""
        print("\n" + "="*60)
        print("DATASET INFORMATION")
        print("="*60)

        if self.info:
            print(f"Codebase Version: {self.info.get('codebase_version', 'N/A')}")
            print(f"Robot Type: {self.info.get('robot_type', 'N/A')}")
            print(f"Total Episodes: {self.info.get('total_episodes', 'N/A')}")
            print(f"Total Frames: {self.info.get('total_frames', 'N/A')}")
            print(f"FPS: {self.info.get('fps', 'N/A')}")
            print(f"Data Files Size: {self.info.get('data_files_size_in_mb', 'N/A')} MB")
            print(f"Video Files Size: {self.info.get('video_files_size_in_mb', 'N/A')} MB")

            print(f"\nFeatures:")
            for feature_name, feature_info in self.info.get('features', {}).items():
                print(f"  - {feature_name}: {feature_info.get('dtype', 'N/A')} {feature_info.get('shape', 'N/A')}")

    def print_data_summary(self):
        """Print summary of the loaded data."""
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)

        if self.data_df is not None:
            print(f"Main Data Shape: {self.data_df.shape}")
            print(f"Columns: {list(self.data_df.columns)}")
            print(f"\nFirst few rows:")
            print(self.data_df.head())

            print(f"\nData types:")
            print(self.data_df.dtypes)

            print(f"\nBasic statistics:")
            print(self.data_df.describe())

        if self.episodes_df is not None:
            print(f"\nEpisodes Data Shape: {self.episodes_df.shape}")
            print(f"Episodes Columns: {list(self.episodes_df.columns)}")
            print(f"\nEpisodes data:")
            print(self.episodes_df)

    def analyze_episodes(self):
        """Analyze episode structure and content."""
        if self.episodes_df is None:
            print("No episodes data loaded.")
            return

        print("\n" + "="*60)
        print("EPISODE ANALYSIS")
        print("="*60)

        print(f"Number of episodes: {len(self.episodes_df)}")

        if 'episode_index' in self.episodes_df.columns:
            print(f"Episode indices: {sorted(self.episodes_df['episode_index'].unique())}")

        # Show episode details
        for idx, row in self.episodes_df.iterrows():
            print(f"\nEpisode {idx}:")
            for col in self.episodes_df.columns:
                print(f"  {col}: {row[col]}")

    def analyze_episodes_metadata_detailed(self):
        """Analyze episodes metadata in detail to look for size indicators."""
        if self.episodes_df is None:
            print("No episodes data loaded.")
            return

        print("\n" + "="*60)
        print("DETAILED EPISODES METADATA ANALYSIS")
        print("="*60)

        print(f"Episodes DataFrame Shape: {self.episodes_df.shape}")
        print(f"All Columns: {list(self.episodes_df.columns)}")

        # Look for size-related columns
        size_related_cols = [col for col in self.episodes_df.columns if any(keyword in col.lower() for keyword in ['size', 'bytes', 'mb', 'file_size', 'chunk', 'file_index'])]
        print(f"\nSize-related columns: {size_related_cols}")

        # Show all data for the episode
        print(f"\nComplete Episode 0 Data:")
        for col in self.episodes_df.columns:
            value = self.episodes_df.iloc[0][col]
            print(f"  {col}: {value}")

        # Check if there are any file path or chunk information
        file_related_cols = [col for col in self.episodes_df.columns if any(keyword in col.lower() for keyword in ['file', 'chunk', 'path', 'index'])]
        print(f"\nFile/Chunk-related columns: {file_related_cols}")

        for col in file_related_cols:
            value = self.episodes_df.iloc[0][col]
            print(f"  {col}: {value}")

    def analyze_episode_sizes_from_metadata(self):
        """Analyze episode sizes using metadata without processing actual files."""
        if self.episodes_df is None:
            print("No episodes data loaded.")
            return

        print("\n" + "="*60)
        print("EPISODE SIZE ANALYSIS FROM METADATA")
        print("="*60)

        for idx, row in self.episodes_df.iterrows():
            episode_idx = row['episode_index']
            length = row['length']

            print(f"\nEpisode {episode_idx}:")
            print(f"  Length: {length} frames")

            # Data range information
            data_from = row['dataset_from_index']
            data_to = row['dataset_to_index']
            data_range = data_to - data_from
            print(f"  Data range: {data_from} to {data_to} ({data_range} frames)")

            # Video information
            if 'videos/observation.images.main/from_timestamp' in row:
                video_from = row['videos/observation.images.main/from_timestamp']
                video_to = row['videos/observation.images.main/to_timestamp']
                video_duration = video_to - video_from
                print(f"  Video duration: {video_from:.3f} to {video_to:.3f} seconds ({video_duration:.3f} seconds)")

                # Calculate frames per second
                if video_duration > 0:
                    fps = length / video_duration
                    print(f"  Effective FPS: {fps:.2f}")

            # Image count (actual video frames)
            if 'stats/observation.images.main/count' in row:
                image_count = row['stats/observation.images.main/count'][0]
                print(f"  Image frames: {image_count}")
                print(f"  Missing image frames: {length - image_count}")

                # Calculate percentage of frames with images
                image_percentage = (image_count / length) * 100
                print(f"  Image coverage: {image_percentage:.1f}%")

            # File organization
            data_chunk = row['data/chunk_index']
            data_file = row['data/file_index']
            print(f"  Data file: chunk-{data_chunk:03d}/file-{data_file:03d}.parquet")

            if 'videos/observation.images.main/chunk_index' in row:
                video_chunk = row['videos/observation.images.main/chunk_index']
                video_file = row['videos/observation.images.main/file_index']
                print(f"  Video file: chunk-{video_chunk:03d}/file-{video_file:03d}.mp4")

            # Estimate data size per frame (rough calculation)
            if self.info and 'features' in self.info:
                features = self.info['features']

                # Calculate approximate bytes per frame for non-video data
                bytes_per_frame = 0
                for feature_name, feature_info in features.items():
                    if feature_info.get('dtype') != 'video':  # Skip video data
                        dtype = feature_info.get('dtype', 'float32')
                        shape = feature_info.get('shape', [1])

                        # Estimate bytes based on dtype
                        if dtype == 'float32':
                            bytes_per_element = 4
                        elif dtype == 'float64':
                            bytes_per_element = 8
                        elif dtype == 'int64':
                            bytes_per_element = 8
                        elif dtype == 'int32':
                            bytes_per_element = 4
                        else:
                            bytes_per_element = 4  # Default assumption

                        # Calculate total elements
                        total_elements = 1
                        for dim in shape:
                            total_elements *= dim

                        bytes_per_frame += bytes_per_element * total_elements

                estimated_data_size_mb = (bytes_per_frame * length) / (1024**2)
                print(f"  Estimated non-video data size: {estimated_data_size_mb:.2f} MB")
                print(f"  Estimated bytes per frame (non-video): {bytes_per_frame} bytes")

    def analyze_robot_data(self):
        """Analyze robot-specific data (actions, observations)."""
        if self.data_df is None:
            print("No data loaded.")
            return

        print("\n" + "="*60)
        print("ROBOT DATA ANALYSIS")
        print("="*60)

        # Analyze actions
        if 'action' in self.data_df.columns:
            print("Action Data:")
            actions = self.data_df['action'].values
            if len(actions) > 0 and hasattr(actions[0], '__len__'):
                # Convert to numpy array if it's a list of lists
                actions_array = np.array([np.array(action) for action in actions])
                print(f"  Shape: {actions_array.shape}")
                print(f"  Min values: {actions_array.min(axis=0)}")
                print(f"  Max values: {actions_array.max(axis=0)}")
                print(f"  Mean values: {actions_array.mean(axis=0)}")

                # Show action names if available
                if self.info and 'features' in self.info and 'action' in self.info['features']:
                    action_names = self.info['features']['action'].get('names', [])
                    if action_names:
                        print(f"  Action names: {action_names}")

        # Analyze observations
        if 'observation.state' in self.data_df.columns:
            print("\nObservation State Data:")
            obs_states = self.data_df['observation.state'].values
            if len(obs_states) > 0 and hasattr(obs_states[0], '__len__'):
                obs_array = np.array([np.array(obs) for obs in obs_states])
                print(f"  Shape: {obs_array.shape}")
                print(f"  Min values: {obs_array.min(axis=0)}")
                print(f"  Max values: {obs_array.max(axis=0)}")
                print(f"  Mean values: {obs_array.mean(axis=0)}")

        # Analyze timestamps
        if 'timestamp' in self.data_df.columns:
            print(f"\nTimestamp Data:")
            timestamps = self.data_df['timestamp'].values
            if len(timestamps) > 0:
                timestamps_array = np.array(timestamps).flatten()
                print(f"  Range: {timestamps_array.min():.2f} to {timestamps_array.max():.2f}")
                print(f"  Duration: {timestamps_array.max() - timestamps_array.min():.2f} seconds")
                print(f"  Mean: {timestamps_array.mean():.2f}")

    def plot_data(self, save_plots: bool = False):
        """Create visualizations of the data."""
        if self.data_df is None:
            print("No data loaded for plotting.")
            return

        print("\n" + "="*60)
        print("CREATING PLOTS")
        print("="*60)

        # Plot actions over time
        if 'action' in self.data_df.columns and 'timestamp' in self.data_df.columns:
            actions = self.data_df['action'].values
            timestamps = self.data_df['timestamp'].values.flatten()

            if len(actions) > 0 and hasattr(actions[0], '__len__'):
                actions_array = np.array([np.array(action) for action in actions])

                plt.figure(figsize=(12, 8))
                for i in range(actions_array.shape[1]):
                    plt.subplot(2, 3, i+1)
                    plt.plot(timestamps, actions_array[:, i])
                    plt.title(f'Action {i+1}')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Value')

                plt.tight_layout()
                if save_plots:
                    plt.savefig('actions_over_time.png', dpi=150, bbox_inches='tight')
                plt.show()

        # Plot episode distribution
        if 'episode_index' in self.data_df.columns:
            plt.figure(figsize=(10, 6))
            episode_counts = self.data_df['episode_index'].value_counts().sort_index()
            plt.bar(episode_counts.index, episode_counts.values)
            plt.title('Frame Distribution Across Episodes')
            plt.xlabel('Episode Index')
            plt.ylabel('Number of Frames')
            if save_plots:
                plt.savefig('episode_distribution.png', dpi=150, bbox_inches='tight')
            plt.show()

    def export_sample_data(self, num_samples: int = 10, output_file: str = "sample_data.csv"):
        """Export a sample of the data to CSV for inspection."""
        if self.data_df is None:
            print("No data loaded.")
            return

        print(f"\nExporting {num_samples} samples to {output_file}...")

        # Take a sample
        sample_df = self.data_df.head(num_samples)

        # Flatten complex columns for CSV export
        export_df = sample_df.copy()

        # Convert list columns to strings for CSV compatibility
        for col in export_df.columns:
            if export_df[col].dtype == 'object':
                export_df[col] = export_df[col].apply(lambda x: str(x) if x is not None else '')

        export_df.to_csv(output_file, index=False)
        print(f"✓ Exported to {output_file}")

    def get_file_sizes_without_processing(self):
        """Get file sizes by examining the file system without processing content."""
        print("\n" + "="*60)
        print("FILE SIZES (WITHOUT PROCESSING CONTENT)")
        print("="*60)

        # Get parquet file sizes
        if self.data_file.exists():
            disk_size = self.data_file.stat().st_size / (1024**2)
            print(f"Main Data Parquet File: {disk_size:.2f} MB")

        if self.episodes_file.exists():
            disk_size = self.episodes_file.stat().st_size / (1024**2)
            print(f"Episodes Metadata Parquet File: {disk_size:.2f} MB")

        # Get video file sizes
        video_dir = self.dataset_path / "videos"
        if video_dir.exists():
            total_video_size = 0.0
            video_files = []

            for video_key_dir in video_dir.iterdir():
                if video_key_dir.is_dir():
                    print(f"\nVideo Key: {video_key_dir.name}")
                    for chunk_dir in video_key_dir.iterdir():
                        if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk-"):
                            for video_file in chunk_dir.glob("*.mp4"):
                                video_size = video_file.stat().st_size / (1024**2)
                                total_video_size += video_size
                                video_files.append(video_file)
                                print(f"  {video_file.name}: {video_size:.2f} MB")

            if video_files:
                print(f"\nTotal Video Files: {len(video_files)}")
                print(f"Total Video Size: {total_video_size:.2f} MB")
                print(f"Average Video File Size: {total_video_size/len(video_files):.2f} MB")

                # Calculate size per episode if we have episode info
                if self.episodes_df is not None and len(self.episodes_df) > 0:
                    total_episodes = len(self.episodes_df)
                    avg_video_size_per_episode = total_video_size / total_episodes
                    print(f"Average Video Size per Episode: {avg_video_size_per_episode:.2f} MB")

                    # Calculate size per frame
                    total_image_frames = 0
                    for idx, row in self.episodes_df.iterrows():
                        if 'stats/observation.images.main/count' in row:
                            total_image_frames += row['stats/observation.images.main/count'][0]

                    if total_image_frames > 0:
                        size_per_frame = total_video_size / total_image_frames
                        print(f"Average Size per Video Frame: {size_per_frame:.2f} MB")

    def run_full_analysis(self):
        """Run a complete analysis of the dataset."""
        print("Starting full dataset analysis...")

        self.load_data()
        self.print_dataset_info()
        self.print_data_summary()
        self.analyze_episodes()
        self.analyze_episodes_metadata_detailed()
        self.analyze_episode_sizes_from_metadata()  # Add this
        self.get_file_sizes_without_processing()    # Add this
        self.analyze_robot_data()
        self.plot_data(save_plots=True)
        self.export_sample_data()

def main():
    """Main function to run the data viewer."""
    # Dataset path
    dataset_path = r"C:\Users\Nicholas\.cache\huggingface\lerobot\so100_follower-001\dataset"

    # Create viewer instance
    viewer = LeRobotDatasetViewer(dataset_path)

    # Run full analysis
    viewer.run_full_analysis()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Check the generated plots and CSV file for detailed data inspection.")

if __name__ == "__main__":
    main()
