
## Navigating Challenges in Mitosis Detection: A Difficulty-Aware Learning Paradigm

## ABSTRACT
 Mitosis nuclei count is an important indicator for pathological diagnosis and histological grade of breast cancer. At present, it mainly
 relies on pathologists to manually review whole slide images (WSI), which is time-consuming, laborious, and subjective. Due to the
 extreme class imbalance of nuclei types, large intra-class heterogeneity, and high inter-class similarity in breast pathology images,
 existing automated methods for mitosis detection struggle to achieve optimal results. Additionally, they inadequately address
 sample noise and hard sample learning, reducing detection robustness. In this paper, we propose a Difficulty-Aware Supervised
 Contrastive Learning (DASC) classification framework based on the nuclei patch level. First, the difficulty and quantity of data
 are balanced by the Negative Sample Search (NS) method. Then, Dynamic Memory Units (DMU) and Difficulty Level Evaluation
 (DLE) are used to select simple samples, eliminate noisy samples, and subdivide and validate the difficulty levels of hard samples
 to obtain a clean and refined dataset. Finally, Subclass-based Difficulty-Driven Supervised Contrastive Learning (SDSCL) aims to
 address the issues of intra-class heterogeneity and inter-class similarity and optimize the learning of hard samples. Experimental
 results show that our method achieves state-of-the-art detection on ICPR2014, TUPAC2016, and MIDOG2021 breast pathology
 datasets. Furthermore, evaluation of the newly released GZMH-V2 dataset, a clinical dataset introduced by our team, demonstrates
 that our method surpasses several widely adopted classical detection approaches. The proposed method in this paper is a simple
 classification model that only uses point annotations, highlighting its potential application for breast cancer mitosis detection. 

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





