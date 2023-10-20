import math
import os
import numpy as np
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def split_data_training_validation_testing(size, training=0.7, validation=0.2):
    """Function that determines the size of training, validation and testing data set"""
    training_size = int(np.floor(training * size))
    validation_size = int(np.floor(validation * size))
    testing_size = size - training_size - validation_size
    return training_size, validation_size, testing_size


def copy_files(images_class, training_size, validation_size, testing_size):
    """Function that split set of images to training, validation and testing data set"""
    directory_path = f'images/{images_class}/'
    images = []

    for i in os.listdir(directory_path):
        images.append(i)

    for i in range(len(os.listdir(directory_path))):
        if i < training_size:
            image = images[i]
            src = f'images/{images_class}/{image}'
            dst = f'data/training/{images_class}/{image}'
            shutil.copy(src, dst)
        elif i < training_size + validation_size:
            image = images[i]
            src = f'images/{images_class}/{image}'
            dst = f'data/validation/{images_class}/{image}'
            shutil.copy(src, dst)
        else:
            image = images[i]
            src = f'images/{images_class}/{image}'
            dst = f'data/test/{images_class}/{image}'
            shutil.copy(src, dst)


classes = ['Lions', 'Cheetahs']
catalogs = ['training', 'validation', 'test']
for i in catalogs:
    for j in classes:
        path = f'data/{i}/{j}'
        if not os.path.exists(path):
            os.makedirs(path)

cheetahs_images_size = len(os.listdir('images/Cheetahs/'))
lions_images_size = len(os.listdir('images/Lions/'))

cheetahs_training, cheetahs_validation, cheetahs_testing = split_data_training_validation_testing(cheetahs_images_size)
print(cheetahs_images_size, cheetahs_training, cheetahs_validation, cheetahs_testing)

lions_training, lions_validation, lions_testing = split_data_training_validation_testing(lions_images_size)
print(lions_images_size, lions_training, lions_validation, lions_testing)

copy_files("Cheetahs", cheetahs_training, cheetahs_validation, cheetahs_testing)
copy_files("Lions", lions_training, lions_validation, lions_testing)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,  # random rotation of image
    rescale=1. / 255.,
    width_shift_range=0.15,  # vertical image transformation
    height_shift_range=0.15,  # horizontal image transformation
    shear_range=0.15, #random cropping range
    zoom_range=0.15, #random zoom range
    horizontal_flip=True, #random reflection of half of the image in the horizontal plane
    fill_mode='nearest', #filling the newly created pixels
)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255.)

train_generator = train_datagen.flow_from_directory(directory='data/training/',
                                                    target_size=(150, 150),
                                                    batch_size=4,
                                                    class_mode='binary')

valid_generator = valid_datagen.flow_from_directory(directory='data/validation/',
                                                    target_size=(150, 150),
                                                    batch_size=4,
                                                    class_mode='binary')

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=512, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
model.summary()

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

batch_size = 2

steps_per_epoch = cheetahs_training // batch_size
valid_steps = cheetahs_validation // batch_size

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=30,
                              validation_data=valid_generator,
                              validation_steps=valid_steps)


trained_model_dir = "model"
if not os.path.exists(trained_model_dir):
    os.makedirs(trained_model_dir)

model.save('model/CNN.model')
loaded_model = tf.keras.models.load_model('model/CNN.model')

def load_image(path):
    img = tf.keras.utils.load_img(path)
    img = tf.image.resize(img, [150, 150])
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img.copy() / 255.0
    img = img.reshape((150, 150, 3))
    img = np.expand_dims(img, axis=0)
    return  img

lion_image = load_image('test_images/lew.jpg')
gepard_image = load_image('test_images/gepard.jpg')
prediciton = loaded_model.predict(lion_image)
prediciton2 = loaded_model.predict(gepard_image)
print(prediciton, prediciton2)
pred = [prediciton , prediciton2]

for i in pred:
    if np.round(i, 0) == 0:
        print(i,": gepard", np.round(i, 0))
    else:
        print(i, ": lion", np.round(i, 0))

image = 'test_images/lew.jpg'
img_load = mpimg.imread(image)
plt.imshow(img_load)
plt.show()

plt.plot(history.epoch, history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()






