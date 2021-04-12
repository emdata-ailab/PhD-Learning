# PhD-Learning

<!--The whole project is established based on the open source project [**torchreid**](https://github.com/KaiyangZhou/deep-person-reid).-->

## Introduction

This repository contains the pytorch implementation of **Phd loss** introduced in CVPR21 paper **PhD Learning: Learning with Pompeiu-hausdorff Distances for Video-based Vehicle Re-Identification.** In this paper, we first create a video vehicle re-ID evaluation benchmark called [**VVeRI-901**](https://cove.thecvf.com/datasets/564) and verify the performance of video-based re-ID is far better than static image-based one. 
Then we propose a new Pompeiu-hausdorff distance (PhD) learning method for video-to-video matching.  It can alleviate the data noise problem caused by the occlusion in videos and thus improve re-ID performance significantly. Extensive empirical results on video-based vehicle and person re-ID datasets, i.e., VVeRI-901, MARS and PRID2011, demonstrate the superiority of the proposed method.

<img src='./docs/intro.png' width=600>

## VVeRI-901

The  proposed  dataset  contains **901** IDs (i.e.,451 IDs for training and 450 IDs for testing), **2,320** tracklets, and **488,195** bounding boxes. 
Besides the vehicle re-ID task, more  related  research  areas  can  be  facilitated, like 
- cross-resolution re-ID, 
- cross-view matching,
- multi-view  synthesis.  

### Samples in VVeRI-901 dataset
<img src='./docs/veri901_samples.png' width=600>

### Statistic of the VVeRI-901 dataset
<img src='./docs/veri901_statistic.png' width=600>

### Comparsion with other existing datasets
<img src='./docs/veri901_compar.png' width=600>

## PhD Loss

The pompeiu-hausdorff distance (PhD) is widely used to measure the similarity between two sets of points. In this work, we investigate the application of PhD metric learning in the field of person/vehicle video-based re-ID task and demonstrate the superiority of PhD metric learning in nosie resistance.

<img src='./docs/phd_intro.png' width=600>

## Evaluation Results

### Vehicle video-based re-ID (VVeRI-901)
<img src='./docs/veri901_result.png' width=400>

### Person video-based re-ID (Mars, PRID2011)
<img src='./docs/person_result.png' width=400>

## Citation
Please cite the following reference if you feel our work is useful to your research.
```
@inproceedings{PhD_2021_CVPR,
  author = {Jianan Zhao and Fengliang Qi and Guangyu Ren and Lin Xu},
  title = {PhD Learning: Learning with Pompeiu-hausdorff Distances for Video-based Vehicle Re-Identification},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year  = {2021},
}
```

## Contact

For any question, please file an issue or contact

```
Jianan Zhao (Shanghai Em-Data Technology Co., Ltd.) jianan.zhao24@gmail.com
Fengliang Qi (Shanghai Em-Data Technology Co., Ltd.) fengliang.qi07@gmail.com
Guangyu Ren (Imperial College London) g.ren19@imperial.ac.uk
Lin Xu (Shanghai Em-Data Technology Co., Ltd.) lin.xu5470@gmail.com
```
