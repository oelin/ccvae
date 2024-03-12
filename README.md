# CCVAE

Conditional VAEs are often implemented by concatenating the class label to the encoder and decoder activations within an MLP-based architecture. However, convolutional architectures are often superior. To this end, we present a fully convolutional conditional VAE. Conditioning is implemented via DiT-style modulation. Preliminary experiments on MNIST demonstrate the model's ability to perform analogical inference and other behaviours. We find that `layers=3, regularization_weight=0.5` are suitable hyperparameters for this task.
