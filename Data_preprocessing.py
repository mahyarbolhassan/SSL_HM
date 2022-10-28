import numpy as np
import matplotlib.pyplot as plt 

image_size = 224
index = []
for i in range((masks.shape[0])-1):
    mask11 = np.all((masks[i] == 0))
    if mask11 == True:
        index.append(i)

masks = np.delete(masks, (index), axis=0)
images = np.delete(images, (index), axis=0) 
# print(masks.shape, images.shape)
#Normalizing the unlabeled data
Uimages = img_C
for i in range(Uimages.shape[0]):
    
    min_Uimages = np.min(Uimages[i,:,:])
    Uimages_tr = Uimages[i,:,:] - min_Uimages
    max_im = np.max(Uimages_tr)
    min_im = np.min(Uimages_tr)
    Uimages[i,:,:] = Uimages_tr / max_im
print(np.max(Uimages[0]))
print(np.min(Uimages[0]))

Uimages = np.reshape(Uimages, (-1, image_size,image_size,1))
print(Uimages.shape)

# Reshaping the masks and images
train_image = np.reshape(images, (-1, image_size,image_size,1))

train_mask = np.reshape(masks, (-1, image_size,image_size,1))

# Uimages = np.reshape(Uimages, (-1, 256,256,1))


val_size = 300 #10% of the training set considered as validation set
train_image_val = train_image[:val_size,:,:,:]
train_mask_val = train_mask[:val_size,:,:,:]

train_image = train_image[val_size:,:,:,:]
train_mask = train_mask[val_size:,:,:,:]

#Normalizing the labeled data
#Normalizing validation set
images_tr = 0

for i in range(train_mask_val.shape[0]):
    
    min_images = np.min(train_image_val[i,:,:])
    images_tr = train_image_val[i,:,:] - min_images
    max_im = np.max(images_tr)
    min_im = np.min(images_tr)
    train_image_val[i,:,:] = images_tr / max_im
print(np.max(train_image_val[10]))
print(np.min(train_image_val[10]))

#Normalizing validation set
images_tr = 0
max_im = 0
min_im =0
for i in range(train_mask.shape[0]):
    
    min_images = np.min(train_image[i,:,:])
    images_tr = train_image[i,:,:] - min_images
    max_im = np.max(images_tr)
    min_im = np.min(images_tr)
    train_image[i,:,:] = images_tr / max_im
print(np.max(train_image[10]))
print(np.min(train_image[10]))

#Visualization of the original image and the correspending mask
fig = plt.figure()
fig.subplots_adjust(hspace=0.1, wspace=0.3)
y = train_mask
ax = fig.add_subplot(1, 2, 1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('ground truth')
ax.imshow(np.reshape(y[1], (image_size, image_size)), cmap="gray")

ax = fig.add_subplot(1, 2, 2)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('original image')
ax.imshow(np.reshape(train_image[1], (image_size, image_size)), cmap="gray")
