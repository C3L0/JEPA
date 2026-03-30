from datasets import load_dataset
import numpy as np
import jax
import jax.numpy as jnp
import pickle
import random

class RobotDataLoader:
    def __init__(self, dataset_id, stats_path, window_size=5, prediction_horizon=1, batch_size=32):
        self.ds_raw = load_dataset(dataset_id, split="train")
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.batch_size = batch_size
        
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
            
        # Pre-load and normalize everything to numpy for speed
        print("Pre-loading and normalizing dataset...")
        self.all_states = self.normalize(np.array(self.ds_raw['observation.state']), 'state')
        self.all_actions = self.normalize(np.array(self.ds_raw['action']), 'action')
        self.episode_indices = np.array(self.ds_raw['episode_index'])
        
        # Group frames by episode to avoid crossing boundaries
        self.episodes = {}
        for i, ep_idx in enumerate(self.episode_indices):
            if ep_idx not in self.episodes:
                self.episodes[ep_idx] = []
            self.episodes[ep_idx].append(i)
            
        # Create all possible valid start indices
        # We need: context window [i, i+W-1], action a_cond [i+W-1], target window [i+H, i+H+W-1]
        self.valid_indices = []
        for ep_idx, frame_indices in self.episodes.items():
            num_frames = len(frame_indices)
            # Max index needed is i + prediction_horizon + window_size - 1
            if num_frames >= window_size + prediction_horizon:
                for i in range(num_frames - window_size - prediction_horizon + 1):
                    self.valid_indices.append(frame_indices[i])
        
        print(f"Total valid windows: {len(self.valid_indices)}")

    def normalize(self, val, key):
        mean = np.array(self.stats[key]['mean'])
        std = np.array(self.stats[key]['std'])
        return (val - mean) / (std + 1e-8)

    def __iter__(self):
        random.shuffle(self.valid_indices)
        for i in range(0, len(self.valid_indices), self.batch_size):
            batch_start_indices = self.valid_indices[i : i + self.batch_size]
            if len(batch_start_indices) < self.batch_size:
                continue
                
            batch_s_ctx = []
            batch_a_cond = []
            batch_s_target = []
            
            for start_idx in batch_start_indices:
                # Context Window: [start, start + window_size - 1]
                s_ctx = self.all_states[start_idx : start_idx + self.window_size]
                
                # Conditioning Action: action at the last frame of the context window
                # This action leads to the state at start + window_size
                a_cond = self.all_actions[start_idx + self.window_size - 1]
                
                # Target Window: [start + H, start + H + window_size - 1]
                # Default H=1 means target window is shifted by 1 frame from context window
                t_start = start_idx + self.prediction_horizon
                s_target = self.all_states[t_start : t_start + self.window_size]
                
                batch_s_ctx.append(s_ctx)
                batch_a_cond.append(a_cond)
                batch_s_target.append(s_target)
                
            yield {
                's_ctx': jnp.array(batch_s_ctx),    # (B, window, 14)
                'a_cond': jnp.array(batch_a_cond),  # (B, 14)
                's_target': jnp.array(batch_s_target) # (B, window, 14)
            }

if __name__ == "__main__":
    loader = RobotDataLoader("lerobot/aloha_static_coffee_new", "data/stats.pkl", batch_size=4)
    for batch in loader:
        print("Batch Shapes:")
        for k, v in batch.items():
            print(f"{k}: {v.shape}")
        break
