# StyleGAN-Keras
StyleGAN made with Keras (without growth)

![alt text](https://i.imgur.com/o3jNlfn.jpg)
A set of 256x256 samples trained for 1 million steps with a batch size of 4.

![alt text](https://i.imgur.com/iV6Y0Dn.jpg)
Rows: 4^2 to 32^2 styles
Columns: 32^2 to 256^2 styles

This GAN is based off this paper:
https://arxiv.org/abs/1812.04948

"A Style-Based Generator Architecture for Generative Adversarial Networks"


Additionally, in AdaIN.py, you will find code for Spatially Adaptive Denormalization (a.k.a SPADE)
This is adapted (as best as I can) from this paper:
https://arxiv.org/abs/1903.07291

"Semantic Image Synthesis with Spatially-Adaptive Normalization"


This StyleGAN is missing growth. Feel free to contribute this, if you'd like!

Mixing regularities is left out in stylegan.py, but included in mixing-stylegan.py. It complicates the inputs of the generator.


To train this on your own dataset, adjust lines 10 to 15 respectively, and load your own images into the /data/ folder under the naming convention 'im (n).suffix'.

Enjoy!
