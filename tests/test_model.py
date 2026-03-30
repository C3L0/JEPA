import pytest
import jax
import jax.numpy as jnp
import jax.random as jrandom
from src.model import MLP, Encoder, Predictor, AJEPA

def test_mlp_shapes():
    key = jrandom.PRNGKey(0)
    in_size, out_size, width, depth = 10, 5, 32, 2
    mlp = MLP(in_size, out_size, width, depth, key)
    
    x = jnp.ones((in_size,))
    out = mlp(x)
    assert out.shape == (out_size,)

def test_encoder_shapes():
    key = jrandom.PRNGKey(0)
    state_dim, window_size, latent_dim = 14, 5, 64
    encoder = Encoder(state_dim, window_size, latent_dim, key)
    
    x = jnp.ones((window_size, state_dim))
    out = encoder(x)
    assert out.shape == (latent_dim,)

def test_predictor_shapes():
    key = jrandom.PRNGKey(0)
    latent_dim, action_dim = 64, 14
    predictor = Predictor(latent_dim, action_dim, key)
    
    z = jnp.ones((latent_dim,))
    a = jnp.ones((action_dim,))
    out = predictor(z, a)
    assert out.shape == (latent_dim,)

def test_ajepa_forward_pass():
    key = jrandom.PRNGKey(0)
    state_dim, action_dim, window_size, latent_dim, proj_dim = 14, 14, 5, 64, 128
    model = AJEPA(state_dim, action_dim, window_size, latent_dim, proj_dim, key)
    
    s_ctx = jnp.ones((window_size, state_dim))
    a_cond = jnp.ones((action_dim,))
    
    z_pred = model(s_ctx, a_cond)
    assert z_pred.shape == (latent_dim,)

def test_ajepa_target_and_project():
    key = jrandom.PRNGKey(0)
    state_dim, action_dim, window_size, latent_dim, proj_dim = 14, 14, 5, 64, 128
    model = AJEPA(state_dim, action_dim, window_size, latent_dim, proj_dim, key)
    
    s_target = jnp.ones((window_size, state_dim))
    z_target = model.encode_target(s_target)
    assert z_target.shape == (latent_dim,)
    
    p_target = model.project(z_target)
    assert p_target.shape == (proj_dim,)

def test_vmap_support():
    """Ensure the model can be vectorized over a batch."""
    key = jrandom.PRNGKey(0)
    batch_size = 8
    state_dim, action_dim, window_size, latent_dim, proj_dim = 14, 14, 5, 64, 128
    model = AJEPA(state_dim, action_dim, window_size, latent_dim, proj_dim, key)
    
    s_ctx_batch = jnp.ones((batch_size, window_size, state_dim))
    a_cond_batch = jnp.ones((batch_size, action_dim))
    
    v_model = jax.vmap(model)
    z_pred_batch = v_model(s_ctx_batch, a_cond_batch)
    
    assert z_pred_batch.shape == (batch_size, latent_dim)
