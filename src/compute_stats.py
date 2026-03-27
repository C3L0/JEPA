from datasets import load_dataset
import numpy as np
import jax
import jax.numpy as jnp
import pickle

def compute_stats():
    dataset_id = "lerobot/aloha_static_coffee_new"
    print(f"Computing stats for: {dataset_id}...")
    
    ds = load_dataset(dataset_id, split="train")
    
    # Convert all states and actions to numpy arrays for calculation
    states = np.array(ds['observation.state'])
    actions = np.array(ds['action'])
    
    stats = {
        'state': {
            'mean': np.mean(states, axis=0).tolist(),
            'std': np.std(states, axis=0).tolist(),
            'max': np.max(states, axis=0).tolist(),
            'min': np.min(states, axis=0).tolist(),
        },
        'action': {
            'mean': np.mean(actions, axis=0).tolist(),
            'std': np.std(actions, axis=0).tolist(),
            'max': np.max(actions, axis=0).tolist(),
            'min': np.min(actions, axis=0).tolist(),
        }
    }
    
    with open('data/stats.pkl', 'wb') as f:
        pickle.dump(stats, f)
    
    print("\n--- Stats Computed ---")
    print(f"State mean (first 3): {stats['state']['mean'][:3]}")
    print(f"State std (first 3): {stats['state']['std'][:3]}")
    print(f"Action mean (first 3): {stats['action']['mean'][:3]}")
    print(f"Action std (first 3): {stats['action']['std'][:3]}")
    print("\nStats saved to data/stats.pkl")

if __name__ == "__main__":
    compute_stats()
