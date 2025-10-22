from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load from cache
dataset = LeRobotDataset("so100-001/so100-001__green-tape-in-cup-a001-chris", root="C:/Users/Theor/.cache/huggingface/lerobot/so100-001/so100-001__green-tape-in-cup-a001-chris")

print(f"Total episodes: {dataset.num_episodes}")
print(f"Total frames: {dataset.num_frames}")
print(f"\nEpisode lengths:")
for ep_idx in range(dataset.num_episodes):
    ep_data = dataset.meta.episodes[ep_idx]
    from_idx = ep_data["dataset_from_index"]
    to_idx = ep_data["dataset_to_index"]
    print(f"  Episode {ep_idx}: {to_idx - from_idx} frames")

