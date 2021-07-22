import tensorflow as tf

def CustomNet(ELAB_IMG_SIZE_X, ELAB_IMG_SIZE_Y):
    def Bundle1(inputs):
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization(momentum=0.0)(x)
        #x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ReLU(max_value=6.0)(x)
        return x

    def Bundle2(inputs, output_chan):
        x = tf.keras.layers.Conv2D(output_chan, kernel_size=(1, 1))(inputs)
        x = tf.keras.layers.BatchNormalization(momentum=0.0)(x)
        #x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ReLU(max_value=6.0)(x)
        return x

    def BundleComp(inputs, mid_chan):
        x = Bundle1(inputs)
        x = Bundle2(x, mid_chan)
        x = Bundle2(x, 10)
        return x

    inputs = tf.keras.Input(shape=[ELAB_IMG_SIZE_Y, ELAB_IMG_SIZE_X, 3], name="InputLayer")
    x = inputs
    x = Bundle1(x)
    x = Bundle2(x, 24)
    x = Bundle1(x)
    x = Bundle2(x, 48)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid')(x)
    x = Bundle1(x)
    x = Bundle2(x, 48)
    x = Bundle1(x)
    x = Bundle2(x, 24)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid')(x)
    x = Bundle1(x)
    x = Bundle2(x, 12)
    x = Bundle1(x)
    x = Bundle2(x, 6)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(10, activation='softmax')(x)
    outputs = x

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

