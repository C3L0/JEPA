from lerobot.datasets.lerobot_dataset import LeRobotDataset
import jax.numpy as jnp
import numpy as np

def inspect_dataset():
    dataset_id = "lerobot/aloha_static_coffee_new"
    print(f"Loading dataset: {dataset_id}...")
    
    # Load the dataset
    dataset = LeRobotDataset(dataset_id)
    
    print("\n--- Dataset Info ---")
    print(f"Number of episodes: {dataset.num_episodes}")
    print(f"Number of frames: {dataset.num_frames}")
    print(f"Features: {list(dataset.features.keys())}")
    
    # Inspect a single item
    item = dataset[0]
    print("\n--- Sample Item (Frame 0) ---")
    for key, val in item.items():
        if isinstance(val, (np.ndarray, list)):
            print(f"{key}: shape {np.array(val).shape}")
        elif hasattr(val, "shape"):
            print(f"{key}: shape {val.shape}")
        else:
            print(f"{key}: {val}")

if __name__ == "__main__":
    inspect_dataset()
