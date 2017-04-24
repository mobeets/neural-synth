import json
import numpy as np
from keras.utils import to_categorical
from app import vrnn, vae
from music21 import chord

def sample_z(args):
    Z_mean, Z_log_var = args
    eps = np.random.randn(*Z_mean.squeeze().shape)
    return Z_mean + np.exp(Z_log_var/2) * eps

def sample_x(x_mean):
    return 1.0*(np.random.rand(*x_mean.squeeze().shape) <= x_mean)

# def notes_to_freqs(notes, offset=21, base=440):
#     return [int((base/32)*np.power(2, (x - 9.) / 12)) for x in notes]

class Generator:
    def __init__(self, model_file, model_type='vrnn', x_seed=None):
        self.model_file = model_file
        self.model_type = model_type
        self.x_seed = x_seed
        self.args = json.load(open(model_file.replace('.h5', '.json')))
        self.model, self.enc_model, self.dec_model = self.load_models(self.model_type)
        self.init_models(self.model_type)

    def load_models(self, model_type):
        if model_type == 'vrnn':
            load_model_fcn = vrnn.load_models
        else:
            load_model_fcn = vae.load_models
        return load_model_fcn(self.args, self.model_file)

    def init_models(self, model_type):
        if model_type == 'vrnn':
            self.w = to_categorical(0, self.args['n_classes'])
            self.use_w = True
            if self.x_seed is not None:
                self.seed_models(self.x_seed) # [nseeds x original_dim]
            else:
                self.x_prev = np.zeros((1, 1, self.args['original_dim']))
                z_t = np.array([-2.988,-2.928])
                self.generate(z_t[None,None,:])
        else:
            self.use_w = False
            self.args['use_x_prev'] = False
            z_t = np.array([-2.988,-2.928])
            self.generate(z_t[None,:])

    def seed_models(self, x_seed, w=None):
        original_dim = x_seed.shape[-1]
        nsteps = x_seed.shape[0]
        for t in xrange(nsteps):
            self.x_prev = x_seed[t][None,None,:]
            z_mean_t, z_log_var_t, w_t = self.enc_model.predict(self.x_prev)
            w_t = w_t if w is None else w
            z_t = [sample_z((z_mean_t, z_log_var_t))]
            x_t = self.generate(z_t, w_t)

    def generate(self, z_t, w=None):
        if self.args['use_x_prev']:
            z_t = [z_t, self.x_prev]
        if self.use_w:
            w_t = w if w is not None else self.w
            z_t = [z_t, w_t]
        x_t = sample_x(self.dec_model.predict(z_t, batch_size=1))
        self.x_prev = x_t
        return x_t

    def generate_as_notes(self, z_t, w=None, offset=21):
        if type(z_t) is not np.ndarray:
            z_t = np.array(z_t)
        if len(z_t.shape) == 1 and self.use_w:
            z_t = z_t[None,None,:]
        elif len(z_t.shape) == 1:
            z_t = z_t[None,:]
        if self.use_w:
            x_t = self.generate(z_t, w)
        else:
            x_t = self.generate(z_t)
        x_t = np.squeeze(x_t).tolist()
        return [x+offset for x in np.where(x_t)[0].tolist()]

    def encode_as_notes(self, notes):
        x_t = []
        if type(x_t) is not np.ndarray:
            x_t = np.array(x_t)
        if len(x_t.shape) == 1 and self.use_w:
            x_t = x_t[None,None,:]
        elif len(x_t.shape) == 1:
            x_t = x_t[None,:]
        z_mean, z_logvar = self.enc_model.predict(x_t, batch_size=1)
        z_mean = np.squeeze(z_mean).tolist()
        return z_mean

def detect_chord(notes):
    cd = chord.Chord(notes)
    return cd.root().name + " " + cd.commonName
    # return chord.Chord(notes).pitchedCommonName

if __name__ == '__main__':
    P = Generator('static/model/vrnn.h5')
    z_t = np.array([-2.988,-2.928])
    print P.generate(z_t[None,None,:])
