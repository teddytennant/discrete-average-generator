Implementation in JAX of the discrete average generator (arXiv:2609.38364). The matching loss is equation 8, including the transport term (r - t) Q U.
Install into the shared venv with `pip install -e .`.
Run `pytest` from this directory.
Does not match a full paper run: no OpenWebText, no ImageNet, no 4x256 MLP, no D=4 Potts, no claimed 67% TV.
