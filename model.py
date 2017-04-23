import json
import numpy as np
from keras import losses
from keras import backend as K
from keras import initializers
from keras.utils import to_categorical
from keras.layers import Input, Dense, LSTM, TimeDistributed, Lambda, concatenate, RepeatVector
from keras.models import Model

def make_decoder(model, original_dim, intermediate_dim, latent_dim, n_classes, use_x_prev, seq_length=1, batch_size=1):
    # build decoder
    Z = Input(batch_shape=(batch_size, seq_length, latent_dim), name='Z')
    if use_x_prev:
        Xp = Input(batch_shape=(batch_size, seq_length, original_dim), name='history')
        XpZ = concatenate([Xp, Z], axis=-1)
    else:
        XpZ = Z
    W = Input(batch_shape=(batch_size, n_classes), name='W')
    XpZ = concatenate([XpZ, RepeatVector(seq_length)(W)], axis=-1)

    if intermediate_dim > 0:
        decoder_h = LSTM(intermediate_dim, return_sequences=True, activation='relu', stateful=True, name='decoder_h')(XpZ)
        X_mean_t = Dense(original_dim, activation='sigmoid', name='X_mean_t')
        X_decoded_mean = TimeDistributed(X_mean_t, name='X_decoded_mean')(decoder_h)
    else:
        X_decoded_mean = LSTM(original_dim, activation='sigmoid', return_sequences=True, stateful=True, name='X_decoded_mean')(XpZ)

    if use_x_prev:
        decoder = Model([Z, Xp, W], X_decoded_mean)
    else:
        decoder = Model([Z, W], X_decoded_mean)
    decoder.get_layer('X_decoded_mean').set_weights(model.get_layer('X_decoded_mean').get_weights())
    if intermediate_dim > 0:
        decoder.get_layer('decoder_h').set_weights(model.get_layer('decoder_h').get_weights())
    return decoder

def get_model(batch_size, original_dim, intermediate_dim, latent_dim, seq_length, n_classes, use_x_prev, optimizer, class_weight):
    """
    if intermediate_dim == 0, uses the output of the lstms directly
        otherwise, adds dense layers
    """
    X = Input(batch_shape=(batch_size, seq_length, original_dim), name='current')
    if use_x_prev:
        Xp = Input(batch_shape=(batch_size, seq_length, original_dim), name='history')

    # Sample w ~ logitNormal before continuing...
    encoder_w_layer = LSTM(2*(n_classes-1), return_sequences=False, name='Wargs')
    Wargs = encoder_w_layer(X)
    def get_w_mean(x):
        return x[:,:(n_classes-1)]
    def get_w_log_var(x):
        return x[:,(n_classes-1):]
    W_mean = Lambda(get_w_mean)(Wargs)
    W_log_var = Lambda(get_w_log_var)(Wargs)
    # sample latents, w
    def sampling_w(args):
        W_mean, W_log_var = args
        eps = K.random_normal(shape=(batch_size, (n_classes-1)), mean=0., stddev=1.0)
        W_samp = W_mean + K.exp(W_log_var/2) * eps
        W0 = concatenate([W_samp, K.zeros((batch_size,1))], axis=-1)
        num = K.exp(W0)
        denom = K.sum(num, axis=-1, keepdims=True)
        return num/denom
    W = Lambda(sampling_w, output_shape=(n_classes,), name='W')([W_mean, W_log_var])
    
    XW = concatenate([X, RepeatVector(seq_length)(W)], axis=-1)

    # build encoder
    if intermediate_dim > 0:
        encoder_h = LSTM(intermediate_dim, return_sequences=True, activation='relu', name='encoder_h')(XW)
        Z_mean_t = Dense(latent_dim,
            bias_initializer='zeros',
            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
            name='Z_mean_t')
        Z_log_var_t = Dense(latent_dim,
            bias_initializer='zeros',
            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
            name='Z_log_var_t')
        Z_mean = TimeDistributed(Z_mean_t, name='Z_mean')(encoder_h)
        Z_log_var = TimeDistributed(Z_log_var_t, name='Z_log_var')(encoder_h)
    else:
        # half of LSTM is the z_mean, the other half is z_log_var
        encoder_h = LSTM(2*latent_dim, return_sequences=True, name='encoder_h')(XW)
        def get_mean(x):
            return x[:,:,:latent_dim]
        def get_log_var(x):
            return x[:,:,latent_dim:]
        Z_mean = Lambda(get_mean)(encoder_h)
        Z_log_var = Lambda(get_log_var)(encoder_h)

    # sample latents, z
    def sampling(args):
        Z_mean, Z_log_var = args
        eps = K.random_normal(shape=(batch_size, seq_length, latent_dim), mean=0., stddev=1.0)
        return Z_mean + K.exp(Z_log_var/2) * eps
    Z = Lambda(sampling, output_shape=(seq_length, latent_dim,))([Z_mean, Z_log_var])

    if use_x_prev:
        XpZ = concatenate([Xp, Z], axis=-1)
    else:
        XpZ = Z
    XpZ = concatenate([XpZ, RepeatVector(seq_length)(W)], axis=-1)

    # build decoder
    if intermediate_dim > 0:
        decoder_h = LSTM(intermediate_dim, return_sequences=True, activation='relu', name='decoder_h')(XpZ)
        X_mean_t = Dense(original_dim,
            bias_initializer='zeros',
            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1), 
            activation='sigmoid',
            name='X_mean_t')
        X_decoded_mean = TimeDistributed(X_mean_t, name='X_decoded_mean')(decoder_h)
    else:
        X_decoded_mean = LSTM(original_dim, activation='sigmoid', return_sequences=True, name='X_decoded_mean')(XpZ)

    def kl_loss(z_true, z_args):
        Z_mean = Z_args[:,:,:latent_dim]
        Z_log_var = Z_args[:,:,latent_dim:]
        return -0.5*K.sum(1 + Z_log_var - K.square(Z_mean) - K.exp(Z_log_var), axis=-1)

    def vae_loss(X, X_decoded_mean):
        xent_loss = original_dim * losses.binary_crossentropy(X, X_decoded_mean)
        # kl_loss = - 0.5 * K.sum(1 + Z_log_var - K.square(Z_mean) - K.exp(Z_log_var), axis=-1)
        return xent_loss# + kl_loss

    Z_args = concatenate([Z_mean, Z_log_var], axis=-1, name='Z_args')
    if use_x_prev:
        model = Model([X, Xp], [X_decoded_mean, W, Z_args])
    else:
        model = Model(X, [X_decoded_mean, W, Z_args])
    model.compile(optimizer=optimizer,
        loss={'X_decoded_mean': vae_loss, 'W': 'categorical_crossentropy', 'Z_args': kl_loss},
        loss_weights={'X_decoded_mean': 1.0, 'W': class_weight, 'Z_args': 1.0},
        metrics={'X_decoded_mean': 'binary_crossentropy',
            'W': 'accuracy'})

    encoder = Model(X, [Z_mean, Z_log_var, W])
    return model, encoder

def sample_z(args):
    Z_mean, Z_log_var = args
    eps = np.random.randn(*Z_mean.squeeze().shape)
    return Z_mean + np.exp(Z_log_var/2) * eps

def sample_x(x_mean):
    return 1.0*(np.random.rand(*x_mean.squeeze().shape) <= x_mean)

def notes_to_freqs(notes, offset=21, base=440):
    return [int((base/32)*np.power(2, (x - 9.) / 12)) for x in notes]

class Generator:
    def __init__(self, model_file, x_seed=None):
        self.model_file = model_file
        self.args = json.load(open(model_file.replace('.h5', '.json')))
        self.model, self.enc_model, self.dec_model = self.load_models(self.args, self.model_file)
        self.w = to_categorical(0, self.args['n_classes'])
        if x_seed is not None:
            self.seed_models(x_seed) # [nseeds x original_dim]
        else:
            self.x_prev = np.zeros((1, 1, self.args['original_dim']))
            z_t = np.array([-2.988,-2.928])
            self.generate(z_t[None,None,:])

    def load_models(self, args, model_file, batch_size=1, seq_length=1, optimizer='adam'):
        model, enc_model = get_model(batch_size, args['original_dim'], args['intermediate_dim'], args['latent_dim'], seq_length, args['n_classes'], args['use_x_prev'], optimizer, args['class_weight'])
        model.load_weights(model_file)
        dec_model = make_decoder(model, args['original_dim'], args['intermediate_dim'], args['latent_dim'], args['n_classes'], args['use_x_prev'], seq_length=seq_length, batch_size=batch_size)
        return model, enc_model, dec_model

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
        w_t = w if w is not None else self.w
        z_t = [z_t, w_t]
        x_t = sample_x(self.dec_model.predict(z_t, batch_size=1))
        self.x_prev = x_t
        return x_t

    def generate_as_notes(self, z_t, w=None, offset=21):
        if type(z_t) is not np.ndarray:
            z_t = np.array(z_t)
        if len(z_t.shape) == 1:
            z_t = z_t[None,None,:]
        x_t = np.squeeze(self.generate(z_t, w)).tolist()
        return [x+offset for x in np.where(x_t)[0].tolist()]

if __name__ == '__main__':
    P = Generator('static/model/lcvrnn15.h5')
    z_t = np.array([-2.988,-2.928])
    print P.generate(z_t[None,None,:])
