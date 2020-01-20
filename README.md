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

Changing the globally-shared code entry (affects pose)
![GlobalCodeExamples](/example_fakes_global.png)
Changing the codes that are shared by region (affects mid-level features such as age and accessories)
![SharedCodeExamples](/example_fakes_shared.png)
Changing all local codes (changing the fine details of the face)
![localCodeExamples](/example_fakes_alllocal.png)
Changing specific local codes (2x2 cells around the mouth)
![mouthCodeExamples](/example_fakes_mouth.png)
Changing specific local codes (3x7 cells covering the top of the head)
![hairCodeExamples](/example_fakes_hair.png)
