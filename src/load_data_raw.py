from datasets import load_dataset
import numpy as np

def inspect_dataset_raw():
    dataset_id = "lerobot/aloha_static_coffee_new"
    print(f"Loading raw dataset using 'datasets' library: {dataset_id}...")
    
    # Load the dataset (this usually downloads the parquet files)
    # We might need to specify the split, usually 'train'
    ds = load_dataset(dataset_id, split="train")
    
    print("\n--- Raw Dataset Info ---")
    print(f"Number of frames: {len(ds)}")
    print(f"Features: {ds.features.keys()}")
    
    # Inspect a single item
    item = ds[0]
    print("\n--- Sample Item (Frame 0) ---")
    for key in ['observation.state', 'action', 'episode_index', 'frame_index', 'timestamp']:
        if key in item:
            val = item[key]
            if isinstance(val, (list, np.ndarray)):
                print(f"{key}: shape {np.array(val).shape}")
            else:
                print(f"{key}: {val}")

if __name__ == "__main__":
    inspect_dataset_raw()
