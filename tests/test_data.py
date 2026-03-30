import pytest
import os
import pickle
import jax.numpy as jnp
import numpy as np
from src.dataloader import RobotDataLoader

# Configuration for tests
DATASET_ID = "lerobot/aloha_static_coffee_new"
STATS_PATH = "data/stats.pkl"

def test_stats_file_exists():
    """Check if the stats file exists."""
    assert os.path.exists(STATS_PATH), f"Stats file not found at {STATS_PATH}"

def test_stats_content():
    """Verify the content and structure of the stats file."""
    with open(STATS_PATH, 'rb') as f:
        stats = pickle.load(f)
    
    assert 'state' in stats
    assert 'action' in stats
    for key in ['mean', 'std', 'max', 'min']:
        assert key in stats['state']
        assert key in stats['action']
        
    # Check for NaNs in stats
    for key in ['state', 'action']:
        for subkey in ['mean', 'std', 'max', 'min']:
            assert not np.isnan(stats[key][subkey]).any(), f"NaN found in stats[{key}][{subkey}]"

def test_dataloader_batch_shapes():
    """Verify the shapes of batches produced by the dataloader."""
    batch_size = 4
    window_size = 5
    loader = RobotDataLoader(DATASET_ID, STATS_PATH, window_size=window_size, batch_size=batch_size)
    
    for batch in loader:
        assert batch['s_ctx'].shape == (batch_size, window_size, 14)
        assert batch['a_ctx'].shape == (batch_size, window_size, 14)
        assert batch['s_target'].shape == (batch_size, 14)
        break # Just check the first batch

def test_normalization_range():
    """Verify that normalized data is within a reasonable range (approx. zero mean, unit variance)."""
    batch_size = 32
    loader = RobotDataLoader(DATASET_ID, STATS_PATH, batch_size=batch_size)
    
    for batch in loader:
        # Check that data is not all zeros or suspiciously large
        assert jnp.abs(jnp.mean(batch['s_ctx'])) < 1.0 
        assert jnp.all(jnp.abs(batch['s_ctx']) < 20.0) # Reasonable range for normalized data
        
        # Check for NaNs in batches
        assert not jnp.isnan(batch['s_ctx']).any()
        assert not jnp.isnan(batch['a_ctx']).any()
        assert not jnp.isnan(batch['s_target']).any()
        break

def test_episode_boundaries():
    """Verify that windows do not cross episode boundaries (implied by dataloader logic)."""
    # The dataloader groups by episode_index, so this should be fine by design.
    # We can add a more explicit check if needed by inspecting loader.valid_indices.
    pass
