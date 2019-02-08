from keras import losses
from keras import backend as K
from keras.layers import Input, Dense, Lambda, concatenate
from keras.models import Model

def make_decoder(model, (latent_dim_0, latent_dim), class_dim, batch_size=1):
    z = Input(batch_shape=(batch_size, latent_dim), name='z')
    w = Input(batch_shape=(batch_size, class_dim), name='w')
    wz = concatenate([w, z], axis=-1)

    # build x decoder
    decoder_mean = model.get_layer('x_decoded_mean')
    if latent_dim_0 > 0:
        decoder_h = model.get_layer('decoder_h')
        h_decoded = decoder_h(wz)
        x_decoded_mean = decoder_mean(h_decoded)
    else:
        x_decoded_mean = decoder_mean(wz)

    mdl = Model([z, w], x_decoded_mean)
    return mdl

def LL_frame(y, yhat, original_dim=88):
    return original_dim*losses.binary_crossentropy(y, yhat)

def get_model(batch_size, original_dim,
    (latent_dim_0, latent_dim),
    (class_dim_0, class_dim), optimizer, class_weight=0.5):

    x = Input(batch_shape=(batch_size, original_dim), name='x')

    # build label encoder
    h_w = Dense(class_dim_0, activation='relu', name='h_w')(x)
    w_mean = Dense(class_dim-1, name='w_mean')(h_w)
    w_log_var = Dense(class_dim-1, name='w_log_var')(h_w)

    # sample label
    def w_sampling(args):
        """
        sample from a logit-normal with params w_mean and w_log_var
            (n.b. this is very similar to a logistic-normal distribution)
        """
        w_mean, w_log_var = args
        eps = K.random_normal(shape=(batch_size, class_dim-1), mean=0., stddev=1.0)
        w_norm = w_mean + K.exp(w_log_var/2) * eps
        # need to add '0' so we can sum it all to 1
        w_norm = concatenate([w_norm, K.tf.zeros(batch_size, 1)[:,None]])
        return K.exp(w_norm)/K.sum(K.exp(w_norm), axis=-1)[:,None]
    w = Lambda(w_sampling, name='w')([w_mean, w_log_var])

    # build latent encoder
    xw = concatenate([x, w], axis=-1)
    if latent_dim_0 > 0:
        h = Dense(latent_dim_0, activation='relu', name='h')(xw)
        z_mean = Dense(latent_dim, name='z_mean')(h)
        z_log_var = Dense(latent_dim, name='z_log_var')(h)
    else:
        z_mean = Dense(latent_dim, name='z_mean')(xw)
        z_log_var = Dense(latent_dim, name='z_log_var')(xw)

    # sample latents
    def sampling(args):
        z_mean, z_log_var = args
        eps = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var/2) * eps
    z = Lambda(sampling, name='z')([z_mean, z_log_var])

    # build decoder
    wz = concatenate([w, z], axis=-1)
    decoder_mean = Dense(original_dim, activation='sigmoid', name='x_decoded_mean')
    if latent_dim_0 > 0:
        decoder_h = Dense(latent_dim_0, activation='relu', name='decoder_h')
        h_decoded = decoder_h(wz)
        x_decoded_mean = decoder_mean(h_decoded)
    else:
        x_decoded_mean = decoder_mean(wz)

    def kl(x, x_decoded_mean):
        return -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

    def vae_loss(x, x_decoded_mean):
        rec_loss = original_dim * losses.binary_crossentropy(x, x_decoded_mean)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return rec_loss + kl_loss

    def w_vae_loss(w_true, w):
        rec_loss = (class_dim-1) * losses.categorical_crossentropy(w_true, w)
        kl_loss = -0.5 * K.sum(1 + w_log_var - K.square(w_mean) - K.exp(w_log_var), axis=-1)
        return rec_loss + kl_loss

    model = Model(x, [x_decoded_mean, w])
    model.compile(optimizer=optimizer,
        loss={'x_decoded_mean': vae_loss, 'w': w_vae_loss},
        loss_weights={'x_decoded_mean': 1.0, 'w': class_weight},
        metrics={'x_decoded_mean': LL_frame, 'w': 'accuracy'})
    enc_model = Model(x, [z_mean, w_mean])
    return model, enc_model

def load_models(args, model_file, batch_size=1, optimizer='adam'):
    model, enc_model = get_model(batch_size, args['original_dim'], (args['intermediate_dim'], args['latent_dim']), (args['intermediate_class_dim'], args['n_classes']), optimizer, args['class_weight'])
    model.load_weights(model_file)
    dec_model = make_decoder(model, (args['intermediate_dim'], args['latent_dim']), args['n_classes'], batch_size=batch_size)
    return model, enc_model, dec_model
