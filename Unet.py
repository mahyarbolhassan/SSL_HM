import tensorflow as tf
from tensorflow import keras
#Multi class U-Net

def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    x = keras.layers.Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    return x


def deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
    y = keras.layers.Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    y = keras.layers.concatenate([y, residual], axis=3)
    y = conv_block(y, nfilters)
    return y


def Unet(image_size, nclasses=4, filters=64):
# down
    input_layer = keras.layers.Input(shape=(image_size, image_size, 1), name='image_input')
    conv1 = conv_block(input_layer, nfilters=filters)
    conv1_out = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(conv1_out, nfilters=filters*2)
    conv2_out = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(conv2_out, nfilters=filters*4)
    conv3_out =keras.layers. MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(conv3_out, nfilters=filters*8)
    conv4_out = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out = keras.layers.Dropout(0.5)(conv4_out)
    conv5 = conv_block(conv4_out, nfilters=filters*16)
    conv5 = keras.layers.Dropout(0.5)(conv5)
# up
    deconv6 = deconv_block(conv5, residual=conv4, nfilters=filters*8)
    deconv6 = keras.layers.Dropout(0.5)(deconv6)
    deconv7 = deconv_block(deconv6, residual=conv3, nfilters=filters*4)
    deconv7 = keras.layers.Dropout(0.5)(deconv7) 
    deconv8 = deconv_block(deconv7, residual=conv2, nfilters=filters*2)
    deconv9 = deconv_block(deconv8, residual=conv1, nfilters=filters)
# output
    output_layer = keras.layers.Conv2D(filters=nclasses, kernel_size=(1, 1))(deconv9)
    output_layer = keras.layers.BatchNormalization()(output_layer)
    output_layer = keras.layers.Activation('softmax')(output_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer, name='Unet')
    return model
model = Unet(image_size)
# optimizer = tensorflow.keras.optimizers.Adam(lr = 0.001)
optimizer = keras.optimizers.Adam(lr = 0.001)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy' , metrics=[dice_background,dice_LVbloodpool,dice_RVbloodpool,dice_LVmyo])
model.compile(optimizer=optimizer, loss= multiclass_dice_loss , metrics=[dice_background,dice_LVbloodpool,dice_RVbloodpool,dice_LVmyo])

model.summary()
