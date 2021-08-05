
def encode_layer(input_layer, filters, batch_norm=True):
    init = RandomNormal(stddev=0.02)
    encode = Conv2D(filters, kernel_size=(4, 4), padding='same', dilation_rate=4, kernel_initializer=init)(input_layer)
    encode = LeakyReLU(alpha=0.2)(encode)
    encode = Conv2D(filters, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(encode)
    if batch_norm:
        encode = BatchNormalization()(encode, training=True)
    encode = LeakyReLU(alpha=0.2)(encode)
    return encode

def decode_layer(input_layer, skip_input_layer, filters, dropout=True):
    init = RandomNormal(stddev=0.02)
    decode = Conv2DTranspose(filters, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(input_layer)
    decode = BatchNormalization()(decode, training=True)
    if dropout:
        decode = Dropout(0.5)(decode, training=True)
    decode = Concatenate()([decode, skip_input_layer])
    decode = Activation('relu')(decode)
    return decode

def bottleneck_layer(input_layer, filters):
    init = RandomNormal(stddev=0.02)
    bottleneck = Conv2D(filters, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(input_layer)
    bottleneck = Activation('relu')(bottleneck)
    return bottleneck

def define_generator(in_shape=(128, 128, 1)):
    init = RandomNormal(stddev=0.02)
    input = Input(shape=in_shape)
    
    e1 = encode_layer(input, 32, batch_norm=False)
    e2 = encode_layer(e1, 64)
    e3 = encode_layer(e2, 128)
    e4 = encode_layer(e3, 256)
    e5 = encode_layer(e4, 512)
    e6 = encode_layer(e5, 512)
   
    b = bottleneck_layer(e6, 512)

    d1 = decode_layer(b, e6, 512)
    d2 = decode_layer(d1, e5, 512)
    d3 = decode_layer(d2, e4, 256)
    d4 = decode_layer(d3, e3, 128)
    d5 = decode_layer(d4, e2, 64, dropout=False)
    d6 = decode_layer(d5, e1, 32, dropout=False)

    output = Conv2DTranspose(2, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d6)
    output = Activation('tanh')(output)

    model = keras.Model(input, output)
    return model
# model = define_generator()
# model.summary()