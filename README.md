# DiffusionExperiments
Experimenting with alternative architectures for image generation. Model.py is currently a transformer architecture in the frequency domain, via a 2d fft. We then train with a causal mask to support single forward pass batched loss across the entire sequence. If this does not work, we will likely patch the frequency domain and autoregress on patches. As of right now we start our testing on the simplest of cases -unconditional cifar10 image generation.


More Information:

<img width="649" alt="image" src="https://github.com/user-attachments/assets/a906ed6c-5c44-4d92-9694-fc5e9039768a">


Diffusion models have been well studied to implicitly predict low frequency coarse features before high frequency fine features via the repeated denoising of an image. This is very nice, but it has not been well studied whether the same implicit operations can be explicitly guided with reconstructing the frequency space of an image. We employ some of the tactics used in Fourier Neural Operators, and take care to ensure convolution theorem does not break information leakage in sequence to sequence testing. 



