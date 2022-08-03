import librosa
from autoencoder import Autoencoder
from preprocess import MinMaxNormaliser
import train as T
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
from scipy.signal import savgol_filter
import librosa.display


class SoundGenerator:
    """SoundGenerator is responsible for generating audios from
    spectrograms.
    """

    def __init__(self, autoencoder, frame_size, hop_length):
        self.autoencoder = autoencoder
        self.hop_length = hop_length
        self.frame_size = frame_size
        self._min_max_normaliser = MinMaxNormaliser(0, 1)

    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):

        log_spectrogram = spectrograms[0,:,:,:]

        denorm_log_spec = self._min_max_normaliser.denormalise(
            log_spectrogram, min_max_values["min"], min_max_values["max"])

        mag = denorm_log_spec[:,:,0]
        angle = denorm_log_spec[:,:,1]

        spec =   librosa.db_to_amplitude(mag)

        spectrogram_plus = spec*(np.cos(angle)+1j*np.sin(angle))

        signal = librosa.istft(spectrogram_plus, hop_length= self.hop_length, n_fft=self.frame_size)
            # append signal to "signals"
        return signal, mag

def plot_spectrogram(spectrogram):


    spectrogram = spectrogram[0,:,:,0]

    fig, ax = plt.subplots()
    img = librosa.display.specshow(spectrogram , x_axis='time', y_axis = 'log', ax = ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()


HOP_LENGTH = 512
FRAME_SIZE = 1024

MIN_MAX_VALUES_PATH = "min_max_values.pkl"
SPECTROGRAM_PATH = "spectrograms"
F0_SPECTROGRAM_PATH = "f0_spectrograms"
SAVE_DIR = "output_audio"

if __name__ == "__main__":
    # initialise sound generator
    autoencoder = Autoencoder.load("model")

    sound_generator = SoundGenerator(autoencoder,FRAME_SIZE, HOP_LENGTH)

    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)

    y = 319  #choosing a sample for test

    x_train, y_train, file_paths = T.load_specs(SPECTROGRAM_PATH, F0_SPECTROGRAM_PATH)

    original_spectrogram  = y_train[y]

    original_spectrogram = original_spectrogram[np.newaxis,...]

    sampled_min_max_values = min_max_values[file_paths[y]]

    original_signals, spectro = sound_generator.convert_spectrograms_to_audio( original_spectrogram, sampled_min_max_values)

  

    f0_spectrogram = x_train[y]
    f0_spectrogram = f0_spectrogram[np.newaxis,...]

    reconstructed_spectrogram  = autoencoder.reconstruct(f0_spectrogram)

    reconstructed_spectrogram[0,:,:,1] = f0_spectrogram[0,:,:,1] 

    plot_spectrogram(f0_spectrogram)


    reconstructed_signal, respectro = sound_generator.convert_spectrograms_to_audio( reconstructed_spectrogram, sampled_min_max_values)

    spectro = spectro[np.newaxis,:,:,np.newaxis]
    plot_spectrogram(spectro)
    respectro = respectro[np.newaxis,:,:,np.newaxis]
    plot_spectrogram(respectro)

    sample_rate = 22050

    save_path = os.path.join(SAVE_DIR, "original.wav")
    sf.write(save_path, original_signals, sample_rate)


    save_path = os.path.join(SAVE_DIR, "generated.wav")
    sf.write(save_path, reconstructed_signal, sample_rate)
    




