# StyleGAN-Keras
StyleGAN made with Keras (without growth or mixing regularization)

![alt text](https://i.imgur.com/6vnPOaG.jpg)
A set of 256x256 samples trained for 325,000 steps with a batch size of 4.

This GAN is based off this paper:
https://arxiv.org/abs/1812.04948

"A Style-Based Generator Architecture for Generative Adversarial Networks"


Additionally, in AdaIN.py, you will find code for Spatially Adaptive Denormalization (a.k.a SPADE)
This is adapted (as best as I can) from this paper:
https://arxiv.org/abs/1903.07291

"Semantic Image Synthesis with Spatially-Adaptive Normalization"


This StyleGAN is missing two components: dimension growth and mixing regularization. Feel free to contribute these, if you'd like!


To train this on your own dataset, adjust lines 18 to 23 respectively, and load your own images into the /data/ folder under the naming convention 'im (n).suffix'.

Enjoy!
