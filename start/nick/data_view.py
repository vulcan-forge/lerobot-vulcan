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

    def run_full_analysis(self):
        """Run a complete analysis of the dataset."""
        print("Starting full dataset analysis...")

        self.load_data()
        self.print_dataset_info()
        self.print_data_summary()
        self.analyze_episodes()
        self.analyze_robot_data()
        self.plot_data(save_plots=True)
        self.export_sample_data()

def main():
    """Main function to run the data viewer."""
    # Dataset path
    dataset_path = r"C:\Users\Nicholas\.cache\huggingface\lerobot\local\combined_dataset"

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
