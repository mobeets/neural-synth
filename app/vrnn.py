from keras import losses
from keras import backend as K
from keras import initializers
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

def load_models(args, model_file, batch_size=1, seq_length=1, optimizer='adam'):
    model, enc_model = get_model(batch_size, args['original_dim'], args['intermediate_dim'], args['latent_dim'], seq_length, args['n_classes'], args['use_x_prev'], optimizer, args['class_weight'])
    model.load_weights(model_file)
    dec_model = make_decoder(model, args['original_dim'], args['intermediate_dim'], args['latent_dim'], args['n_classes'], args['use_x_prev'], seq_length=seq_length, batch_size=batch_size)
    return model, enc_model, dec_model
