# DiffusionExperiments
Experimenting with alternative architectures for image generation. We have code for both imagenet 256 and cifar10 class conditional image generation models. For cifar10, the low sequence length allows us to use a small llama architecture on the continuous real and imaginary values of the fft of images. We autoregress completely in the frequency domain with no need for ifft within the architecture;we predict quantized frequencies (real and imaginary values separately) causally from low to high using a custom flatten (and unflatten) method of the 2d fft, since transformers require 1d sequences of information. Theoretically this type of method should explicitly represent the implicit frequency autoregression capabilities of diffusion models with N examples rather than N*T examples (where T is denoising steps). This should allow for faster training.

For imagenet, to prevent extreme costs, we use this frequency autoregression architecture in latent space. For the continuous latents, we use COSMOS tokenizer CI 16x16 from Nvidia. See here:
https://huggingface.co/nvidia/Cosmos-Tokenizer-CI16x16


More Information:

<img width="649" alt="image" src="https://github.com/user-attachments/assets/a906ed6c-5c44-4d92-9694-fc5e9039768a">




