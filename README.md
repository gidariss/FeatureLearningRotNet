## *Unsupervised Representation Learning by Predicting Image Rotations*

### Introduction

The *FeatureLearningRotNet* code implements the following ICLR2018 paper:    
**Title:**      "Unsupervised Representation Learning by Predicting Image Rotations"    
**Authors:**     Spyros Gidaris, Praveer Singh, Nikos Komodakis    
**Institution:** Universite Paris Est, Ecole des Ponts ParisTech    
**Code:**        https://github.com/gidariss/FeatureLearningRotNet   
**Link:**        https://openreview.net/forum?id=S1v4N2l0-

**Abstract:**  
Over the last years, deep convolutional neural networks (ConvNets) have transformed the field of computer vision thanks to their  unparalleled capacity to learn high level semantic image features. However, in order to successfully learn those features, they usually require massive amounts of manually labeled data, which is both expensive and impractical to scale. Therefore, unsupervised semantic feature learning, i.e., learning without requiring manual annotation effort, is of crucial importance in order to successfully harvest the vast amount of visual data that are available today. In our work we propose to learn image features by training ConvNets to recognize the 2d rotation that is applied to the image that it gets as input.  We demonstrate both qualitatively and quantitatively that this apparently simple task actually provides a very powerful supervisory signal for semantic feature learning.  We exhaustively evaluate our method in various unsupervised feature learning benchmarks and we exhibit in all of them state-of-the-art performance. Specifically, our results on those benchmarks demonstrate dramatic improvements w.r.t. prior state-of-the-art approaches in unsupervised representation learning and thus significantly close the gap with supervised feature learning. For instance, in PASCAL VOC 2007 detection task our unsupervised pre-trained AlexNet model achieves the state-of-the-art (among unsupervised methods) mAP of 54.4%$that is only 2.4 points lower from the supervised case.  We get similarly striking results when we transfer our unsupervised learned features on various other tasks, such as ImageNet classification, PASCAL classification, PASCAL segmentation, and CIFAR-10 classification.

### Citing FeatureLearningRotNet

If you find *FeatureLearningRotNet* useful in your research, please consider citing our ICLR2018 paper:
```
@inproceedings{
  gidaris2018unsupervised,
  title={Unsupervised Representation Learning by Predicting Image Rotations},
  author={Spyros Gidaris and Praveer Singh and Nikos Komodakis},
  booktitle={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=S1v4N2l0-},
}
```
