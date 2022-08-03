
import khai as k
import numpy as np
import matplotlib.pyplot as plt
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import os
import soundfile as sf
from playsound import playsound


def amplitude_envelope(signal, frame_size, hop_length):
    """Fancier Python code to calculate the amplitude envelope of a signal with a given frame size."""
    return np.array([max(signal[i:i+frame_size]) for i in range(0, len(signal), hop_length)])



def interp1d(array: np.ndarray, new_len: int) -> np.ndarray:
    la = len(array)
    return np.interp(np.linspace(0, la - 1, num=new_len), np.arange(la), array)



def f0_extraction(y, sr):

	f0, voiced_flag, voiced_probs = librosa.pyin(y,
                                             fmin=librosa.note_to_hz('C2'),
                                             fmax=librosa.note_to_hz('C7'))
	times = librosa.times_like(f0)

	f0[np.isnan(f0)] = 1
	#f0 = np.nan_to_num(f0)

	h = int(len(y)/len(f0))


	a = np.full(h, 1)

	y_gen = k.sinewaveSynth(f0, a, h, sr)

	pad = len(y)-len(y_gen)

	y_gen = np.pad(y_gen, (0, pad), mode = 'constant')






	frame_ae = 10024

	hop_ae = int(frame_ae/2)

	ae = amplitude_envelope(y, frame_ae, hop_ae)

	print(ae)

	ae_x = interp1d(ae, new_len=len(y_gen))




	y_f0 = np.multiply(y_gen, ae_x)

	return y_f0


