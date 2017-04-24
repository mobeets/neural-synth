## Neural music pad

A variational autoencoder ([VAE](https://arxiv.org/abs/1606.05908)) is used to generate outputs (`X`) from a continuous latent space (`Z`) by learning a generative model `p(X | Z)`. In our case we want to generate music by controlling the position of the user's cursor. So we let `X(t)` be an 88-d binary vector that specifies which of 88 different notes to play at some time `t`, and `Z(t)` is controlled by your mouse.

## How it works

This variational autoencoder was trained on 382 of Bach's four-part chorales ([source](http://www-etud.iro.umontreal.ca/~boulanni/icml2012)), transposed to Cmaj or Amin. The model was built and trained using [keras](keras.io). Code for this process can be found [here](https://github.com/mobeets/vrnn).

In the browser, I'm using [p5.js](https://p5js.org/) to play sound and handle mouse clicks. In the backend, I use the model loaded in keras to generate notes from the position clicked on the pad. I then use [music21](http://web.mit.edu/music21/doc/index.html) to detect which chord those notes correspond to.

## Running locally

First, install requirements with `pip install -r requirements.txt`. Then run `python app.py` and navigate in your browser to `http://0.0.0.0:8080`.
