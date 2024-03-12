# CoCoVAE

Conditional VAEs are typically implemented by concatenating the class label to activations within encoder and/or decoder MLP layers. However, fully convolutional architectures are generally preferable for image related tasks. To this end, we present CoCoVAE, a fully convolutional conditional VAE that features DiT-style affine conditioning as apposed to concatenation. Preliminary experiments on MNIST show that the model is capable of analogical inference and related conditional tasks.
