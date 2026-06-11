"""Flax NNX modules for the JAX backend.

Implement one module file per algorithm family, for example:
- ``efficient_zero.py``: representation, dynamics, and prediction networks
- ``rssm.py``: recurrent state-space model for Dreamer / PlaNet
- ``td_mpc.py``: latent encoder and task-oriented heads for TD-MPC

Networks live here. World models in ``world_models/`` should call into these modules.
"""
