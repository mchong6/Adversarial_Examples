## Adversarial example generator

This script generates adversarial examples for convolutional neural networks
by minimizing the correct classification label of the original image. Adversarial
examples are trained with multiple versions of the same image with added Gaussian
noise.


### Dependency

This script requires trained [OverFeat](https://github.com/sermanet/OverFeat) network 
or VGG19 network. Also requires torch and CUDA.


### Example

The example script predicts the output category of original and its adversarial examples.

```bash
th example.lua
```

![](example.png)
