
## Weakly-Supervised Navigating Challenges in Mitosis Detection: A Difficulty-Aware Learning Paradigm

## ABSTRACT
Mitosis nuclei count is an important indicator for breast cancer diagnosis and grading, currently relying on manual whole slide image 
review by pathologists, which is time-consuming, labor-intensive, and subjective. Due to extreme class imbalance of nuclei types,
intra-class heterogeneity, and inter-class similarity in breast pathology images, these data challenges hinder mitosis detection,
with inadequate handling of sample noise and hard samples further reducing robustness. In this paper, we propose a  Weakly-Supervised
Difficulty-Aware Learning (WS-DAL) framework based on the nuclei patch level. First, the difficulty and quantity of data are balanced 
by the Negative Sample Search (NS) method. Then, Dynamic Memory Units (DMU) and Difficulty Level Evaluation (DLE) are used to select 
simple samples, eliminate noisy samples, and subdivide and validate the difficulty levels of hard samples to obtain a clean and refined 
dataset. Finally, Class-Stratified Difficulty-Guided Learning (CSDGL) aims to address the issues of intra-class heterogeneity and 
inter-class similarity and optimize the learning of hard samples. Experimental results show that our method achieves state-of-the-art
detection on the ICPR2014, TUPAC2016, and MIDOG2021 datasets, and surpasses several widely adopted classical detection approaches on
the newly released clinical dataset GZMH-V2, introduced by our team. The proposed method in this paper is an easy-to-use classification 
model that only uses weakly supervised point annotations, highlighting its potential application for breast cancer mitosis detection.

#### Hardware

* 128GB of RAM
* 1*Nvidia 2080ti 12G GPU

## Updates / TODOs
Please follow this GitHub for more updates.
- [X] Add training code
- [X] Add testing code.
- [X] Add model.
- [ ] Add key code
###
#### 1.Preparations
* Data Preparation

   * Download training challenge 1.[MIDOG 2021](https://imig.science/midog/download-dataset/)
2.[MITOSIS14](https://mitos-atypia-14.grand-challenge.org/Dataset/)  3.[TUPAC-auxiliary](https://tupac.grand-challenge.org/Dataset/) 4.[GZMH-V2](https://doi.org/10.57760/sciencedb.08547/)
   * run python pre/cut_patch_2.py

  
#### 2.negative sample search method
```
python NS/sigle.py
```
Then
```
python python pre/cut_patch_2.py --fp_mdoel True
```

#### 3. Dynamic Memory Units
```
python DMU/main.py
```
Then
```
python clus_hard.py
```
#### 4. Difficulty Level Evaluation
```
python DMU/hard_main.py
```
Then
```
python clus_scl.py
```
#### 5. subclass-based difficulty grading-driven supervised contrastive learning 
```
python DMU/hard_scl_main.py
```



#### Inference
```
python predict.py
```

#### Acknowlegment
Many thanks to [UTS](https://github.com/cvblab/Mitosis-UTS)

#### Contact
If you have any question, please contact whd@guet.edu.cn.





