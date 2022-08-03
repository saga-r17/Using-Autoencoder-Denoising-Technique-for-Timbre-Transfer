
import os
from autoencoder import Autoencoder
import numpy as np
import pickle
from tensorflow.keras.datasets import mnist


LEARNING_RATE = 0.0005
BATCH_SIZE = 50  
EPOCHS = 5   
 


SEPCTROGRAM_PATH = 'spectrograms/'
F0_SPECTROGRAM_PATH = 'f0_spectrograms/'



def load_specs(spectorgram_path, f0_spectorgram_path):

    x_train = []
    y_train = []
    file_paths = []

    for root, _, file_names in os.walk(f0_spectorgram_path):

        for file_name in file_names:

            file_path  = os.path.join(root, file_name)

            spectrogram = np.load(file_path)
            x_train.append(spectrogram)

    #x_train = np.append(np.array(x_train[:][:][0]),np.array(x_train[:][:][1]))



    for root, _, file_names in os.walk(spectorgram_path):

        for file_name in file_names:

            file_path  = os.path.join(root, file_name).replace("\\","/")

            file_paths.append(file_path)

            spectrogram = np.load(file_path)

            y_train.append(spectrogram)

    #y_train = np.append(np.array(y_train[:][:][0]),np.array(y_train[:][:][1]))

    return x_train, y_train, file_paths




def train(x_train,y_train, learning_rate, batch_size, epochs):
    autoencoder = Autoencoder(
        input_shape=(512, 128, 2),
        conv_filters=(512, 256),
        conv_kernels=(3, 3 ),
        conv_strides=(2, (2, 1)),
        latent_space_dim = 128
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, y_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    x_train, y_train = load_specs(SEPCTROGRAM_PATH, F0_SPECTROGRAM_PATH)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    autoencoder = train(x_train[:310], y_train[:310], LEARNING_RATE, BATCH_SIZE, EPOCHS)

    autoencoder.save("model")