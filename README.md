# Using-Autoencoder-Denoising-Technique-for-Timbre-Transfer



# Autoencoder
An autoencoder is a type of artificial neural network used to learn data encodings in an unsupervised manner.
The encoder can then be used as a data preparation technique to perform feature extraction on raw data that can be used to train a different machine learning model.
In general, autoencoders work on the premise of reconstructing their inputs.

# Denoising Autoencoder
To achieve this equilibrium of matching target outputs to inputs, denoising autoencoders accomplish this goal in a specific way – the program takes in a corrupted version of some model, and tries to reconstruct a clean model through the use of denoising techniques.

# Timbre of a Sound
Unlike other characteristics of a sound, it's hard to define timbre in terms of features. So instead of extracting the features to represent the timbre, we can use autoencoder to learn timbre representation from raw data. Which we can then use for timbre transfer purpose.


# Project Pipeline


* Extracting fundamental frequency from given input audio
* Preprocessing audio datasets for training ( Audio to Spectrogram )
* Training model with fundamental frequency
* Training ends when output resembles the characteristics of input data
* Testing model with unseen data


# Result


