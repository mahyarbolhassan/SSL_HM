import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt 
from tensorflow.python.keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras.callbacks import Callback

# One hot encoding 
train_mask_multi = to_categorical(train_mask, 4)

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.00001)

class Histories(Callback):

    def on_train_begin(self,logs={}):
        self.losses = []
#         self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
#         self.accuracies.append(logs.get('acc'))

histories = Histories()

tensorboard = TensorBoard(log_dir = "logs_hm/{}".format(time()))

callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

# history = model.fit(train_image , train_mask_multi, epochs=600, batch_size=32, validation_split=0.2, callbacks=[tensorboard, callback_lr])

history = model.fit(X_train_aug , y_train_aug, epochs=600, batch_size=32, validation_split=0.2, callbacks=[tensorboard, callback_lr])

# import os.path
# if os.path.isfile('multi_resUnet_0.01.h5') is False:
model.save_weights('hm.h5')
##############################################################################################
# Prediction
result = model.predict(train_image_val)
result2 = np.argmax(result, axis=-1)
# print(result2.shape)
#Visualization of the test set

for i in range(10):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    y = train_mask_val
#     print(y.shape)
    ax = fig.add_subplot(1, 3, 1)
    ax.set_title('ground truth')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(np.reshape(y[i+100], (image_size, image_size)), cmap="gray")

    ax = fig.add_subplot(1, 3, 2)
    ax.set_title('predicted')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(np.reshape(result2[i+100], (image_size, image_size)), cmap="gray")

    ax = fig.add_subplot(1, 3, 3)
    ax.set_title('original image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(np.reshape(train_image_val[i+100], (image_size, image_size)), cmap="gray")
    #############################################################################################
    #Visualization of the loss function and dice score results
    # Get training and test loss histories
training_loss = history.history['loss']
val_training_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, val_training_loss, 'b-')
plt.legend(['Training Loss', 'val_training_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();
plt.figure()
# Get training and test loss histories
val_dice_background = history.history['val_dice_background']
dice_background = history.history['dice_background']
val_dice_LVbloodpool = history.history['val_dice_LVbloodpool']
dice_LVbloodpool = history.history['dice_LVbloodpool']
dice_RVbloodpool = history.history['dice_RVbloodpool']
val_dice_RVbloodpool = history.history['val_dice_RVbloodpool']
val_dice_LVmyo = history.history['val_dice_LVmyo']
dice_LVmyo = history.history['dice_LVmyo']

# Visualize loss history
plt.plot(epoch_count, dice_LVbloodpool, 'r--')
plt.plot(epoch_count, val_dice_LVbloodpool, 'b-')
plt.legend(['DSC_LV', 'val_DSC_LV'])
plt.xlabel('Epoch')
plt.ylabel('Dice')
plt.show();
####################################################################
print(history.history.keys())
#dict_keys(['loss', 'DSC_av', 'DSC_LV', 'DSC_RV', 'DSC_MU', 'val_loss', 'val_DSC_av', 'val_DSC_LV', 'val_DSC_RV', 'val_DSC_MU'])
epochs = 600
dice_background = history.history['dice_background']
dice_background = sum(dice_background)/epochs 
print('dice_background_av = ', dice_background)
DSC_LV = history.history['dice_LVbloodpool']
DSC_LV_mean = sum(DSC_LV)/epochs 
print('DSC_LV_mean = ',DSC_LV_mean)
DSC_RV = history.history['dice_RVbloodpool']
DSC_RV_mean = sum(DSC_RV)/epochs 
print('DSC_RV_mean = ',DSC_RV_mean)
DSC_MU = history.history['dice_LVmyo']
DSC_MU_mean = sum(DSC_MU)/epochs 
print('DSC_MU_mean = ',DSC_MU_mean)
val_DSC_av = history.history['val_dice_background']
val_DSC_av_mean = sum(val_DSC_av)/epochs 
print('val_dice_background_av = ',val_DSC_av_mean)
val_DSC_LV = history.history['val_dice_LVbloodpool']
val_DSC_LV_mean = sum(val_DSC_LV)/epochs 
print('val_DSC_LV_mean = ',val_DSC_LV_mean)
val_DSC_RV = history.history['val_dice_RVbloodpool']
val_DSC_RV_mean = sum(val_DSC_RV)/epochs 
print('val_DSC_RV_mean = ',val_DSC_RV_mean)
val_DSC_MU = history.history['val_dice_LVmyo']
val_DSC_MU_mean = sum(val_DSC_MU)/epochs 
print('val_DSC_MU_mean = ',val_DSC_MU_mean)
loss = history.history['loss']
average_loss = sum(loss)/epochs
print('average_loss = ',average_loss)
val_loss = history.history['val_loss']
average_val_loss = sum(val_loss)/epochs
print('average_val_loss = ',average_val_loss)
lr = history.history['lr']
lr_mean = sum(lr)/epochs 
print('lr_mean = ',lr_mean)
mean_ss = (DSC_LV_mean + DSC_RV_mean + DSC_MU_mean)/3
print(mean_ss)
############################################################################
