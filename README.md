

# Semi-Supervised Segmentation of Multi-vendor and Multi-center Cardiac MRI
Semi supervised learning with histogram matching

This repository is based on Keras

First part, we applied a fully supervised approach to predict the labels for data of vendor C which doesn't have annotation masks.
In order to improve the results of this step, we applied histogram matching.

The first step allows us to both improve the generalizability of our model in out-of-distribution data and increase the size of our training data.

Next step is to merge data and start training again with new data. In this manner we leverage the unlabeled data to enhance the performance of our model without the need for more labeled data which is certainly expensive.

![ssl_diagram](https://user-images.githubusercontent.com/80331448/113724322-e8673380-96fa-11eb-9297-f13ab5819d83.png)
 
 One of best semi supervised results:
![best2ssl](https://user-images.githubusercontent.com/80331448/113724571-27958480-96fb-11eb-857a-f69a2838e2cd.png)

You can find the paper here:
https://ieeexplore.ieee.org/abstract/document/9477818

Cite this paper:
@INPROCEEDINGS{9477818,  author={Bolhassani, Mahyar and Oksuz, Ilkay},  booktitle={2021 29th Signal Processing and Communications Applications Conference (SIU)},   title={Semi-Supervised Segmentation of Multi-vendor and Multi-center Cardiac MRI},   year={2021},  volume={},  number={},  pages={1-4},  doi={10.1109/SIU53274.2021.9477818}}

