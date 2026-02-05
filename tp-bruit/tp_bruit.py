import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate,BatchNormalization, Activation


IMG_SIZE = 128  
FOLDER = './images'
NOISE_FACTOR = 0.3

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
    return np.array(images)


data = load_images(FOLDER)

data = data.astype('float32') / 255.
data = np.reshape(data, (len(data), IMG_SIZE, IMG_SIZE, 1))
x_train, x_test = train_test_split(data, test_size=0.2, random_state=42)


def add_noise(img_set):
    noisy = img_set + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=img_set.shape)
    return np.clip(noisy, 0., 1.)

x_train_noisy = add_noise(x_train)
x_test_noisy = add_noise(x_test)



def build_unet(img_size):
    inputs = Input(shape=(img_size, img_size, 1))

    # encodeur
    
    c1 = Conv2D(64, (3, 3), padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

   
    c2 = Conv2D(128, (3, 3), padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # bottleneck
    b = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)

    #decodeur
    
    u1 = UpSampling2D((2, 2))(b)
    m1 = concatenate([u1, c2])
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(m1)

    u2 = UpSampling2D((2, 2))(c3)
    m2 = concatenate([u2, c1])
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(m2)
   
    last = Conv2D(32, (3, 3), activation='relu', padding='same')(c4)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(last)
    return models.Model(inputs, decoded)

autoencoder = build_unet(IMG_SIZE)
autoencoder.compile(optimizer='adam', loss='mse')

# train
autoencoder.fit(x_train_noisy, x_train,
                epochs=100, 
                batch_size=8,
                validation_data=(x_test_noisy, x_test))


decoded_imgs = autoencoder.predict(x_test_noisy)
n = min(len(x_test), 5)
plt.figure(figsize=(15, 8))
for i in range(n):
    # Normale
    plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title("Originale")
    plt.axis('off')
    
    # Bruit
    plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title("Bruit")
    plt.axis('off')

    #Débruitée
    plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title("Auto-encodeur")
    plt.axis('off')
plt.show()