# Structured Noise Injection (Official TensorFlow implementation)
A TensorFlow implementation of structured noise injection as described in the paper. We adapt the original StyleGAN architecture code from https://github.com/NVlabs/stylegan.

The code allows:
-  Disentangled editing of generated images (local features, mid-scale features, pose, and overall style)
-  Training a model with structured noise injection on any dataset
-  Modifying the paper's choices of grid dimensions, local code length, shared code length, and global code length 


# Examining a pretrained network
We follow the same approach as the original StyleGAN code.

In order to randomly generate a few images, and preview the changes possible by our method:
```
python pretrained_SNI.py
```
This will generate two unique faces, and multiple figure showing specific modifications while maintaining the face identity.
Any cell of the noise grid can be changed individually by providing an 8x8 binary to the function ``` randomize_specific_local_codes ``` as demonstrated in the example file.

Changing the globally-shared code entry (affects pose)
![GlobalCodeExamples](/example_fakes_global.png)

Changing the codes that are shared by region (affects mid-level features such as age and accessories)
![SharedCodeExamples](/example_fakes_shared.png)

Changing all local codes (affects the fine details of the face)
![localCodeExamples](/example_fakes_alllocal.png)

Changing specific local codes (4x4 cells around the mouth)
![mouthCodeExamples](/example_fakes_mouth.png)

Changing specific local codes (3x7 cells covering the top of the head)
![hairCodeExamples](/example_fakes_hair.png)

# Training a network from scratch
The network can be trained similarly to training the original StyleGAN but with a different generator. The code for our generator is included under ``` training/networks_structurednoiseinjection.py ```.

To run training on the FFHQ datasets with the default settings:
``` python3 train.py ```

The expected performance of a trained network:


| Metric  | Score |
| ------------- | ------------- |
| fid50k  | 6.22  |
| ppl_zfull  | 354.07  |
| ppl_wfull  | 175.82  |
| ppl_zend  | Content Cell  |
| ppl_wend  | Content Cell  |
| ls_z  | 50.68 |
| ls_w  | 1.0012  |



# Testing new settings of structured noise injection
TODO

