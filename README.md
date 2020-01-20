# Structured Noise Injection (Official TensorFlow implementation)
A TensorFlow implementation of structured noise injection as described in the CVPR2020 submission. We adapt the original StyleGAN architecture from https://github.com/NVlabs/stylegan.

The code allows:
1.  Disentangled editing of generated images (local features, mid-scale features, pose, and overall style)
1.  Training a model with structured noise injection on any dataset
1.  Modifying the paper's choices of grid dimensions, local code length, shared code length, and global code length 


# Examining a pretrained network
We follow the same approach as the original StyleGAN code.

In order to randomly generate a few images, and preview the changes possible by our method:
```
python pretrained_SNI.py
```
This will generate two unique faces, and multiple figure showing specific modifications while maintaining the face identity.
Any cell of the noise grid can be changed individually by providing a 1/0 mask to the function ``` randomize_specific_local_codes ``` as demonstrated in the example file.

![GitHub Logo](/images/logo.png)
