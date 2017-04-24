## Neural music pad

A variational autoencoder ([VAE](https://arxiv.org/abs/1606.05908)) is used to generate outputs (`X`) from a continuous latent space (`Z`) by learning a generative model `p(X | Z)`. In our case we want to generate music by controlling the position of the user's cursor. So we let `X(t)` be an 88-d binary vector that specifies which of 88 different notes to play at some time `t`, and `Z(t)` is controlled by your mouse.

## Running locally

First, install requirements with `pip install -r requirements.txt`. Then run `python app.py` and navigate in your browser to `http://0.0.0.0:8080`.
