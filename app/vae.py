from keras import losses
from keras import backend as K
from keras import initializers
from keras.layers import Input, Dense, Lambda
from keras.models import Model

def make_decoder(model, latent_dim, intermediate_dim, batch_size=1):
    # build a decoder that can sample from the learned distribution
    decoder_input = Input(batch_shape=(batch_size, latent_dim,))
    decoder_mean = model.get_layer('decoder_mean')
    if intermediate_dim > 0:
        decoder_h = model.get_layer('decoder_h')
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
    else:
        _x_decoded_mean = decoder_mean(decoder_input)
    decoder = Model(decoder_input, _x_decoded_mean)
    return decoder

def make_encoder(model, original_dim, intermediate_dim, batch_size=1):
    # build a model to project inputs on the latent space
    # x = model.get_layer('x')
    x = Input(batch_shape=(batch_size, original_dim), name='x')
    if intermediate_dim > 0:
        h = model.get_layer('h')(x)
        z_mean = model.get_layer('z_mean')(h)
        z_log_var = model.get_layer('z_log_var')(h)
    else:
        z_mean = model.get_layer('z_mean')(x)
        z_log_var = model.get_layer('z_log_var')(x)
    encoder = Model(x, [z_mean, z_log_var])
    # encoder = Model(x, z_mean)
    return encoder

def get_model(batch_size, original_dim, latent_dim, intermediate_dim, optimizer):

    x = Input(batch_shape=(batch_size, original_dim), name='x')

    # build encoder
    if intermediate_dim > 0:
        h = Dense(intermediate_dim, activation='relu', name='h')(x)
        z_mean = Dense(latent_dim, name='z_mean')(h)
        z_log_var = Dense(latent_dim, name='z_log_var')(h)
    else:
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # sample latents
    def sampling(args):
        z_mean, z_log_var = args
        eps = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var/2) * eps
    z = Lambda(sampling)([z_mean, z_log_var])

    # build decoder
    decoder_mean = Dense(original_dim, activation='sigmoid', name='decoder_mean')
    if intermediate_dim > 0:
        decoder_h = Dense(intermediate_dim, activation='relu', name='decoder_h')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)
    else:
        x_decoded_mean = decoder_mean(z)

    def vae_loss(x, x_decoded_mean):
        xent_loss = original_dim * losses.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    vae = Model(x, x_decoded_mean)
    vae.compile(optimizer=optimizer, loss=vae_loss, metrics=['accuracy', losses.binary_crossentropy])
    return vae

def load_models(args, model_file, batch_size=1, optimizer='adam'):
    model = get_model(batch_size, args['original_dim'], args['latent_dim'], args['intermediate_dim'], optimizer)
    model.load_weights(model_file)
    enc_model = make_encoder(model, args['original_dim'], args['intermediate_dim'], batch_size=batch_size)
    dec_model = make_decoder(model, args['latent_dim'], args['intermediate_dim'], batch_size=batch_size)
    return model, enc_model, dec_model
