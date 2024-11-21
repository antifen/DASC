
## Mitosis detection in breast histopathology images via Difficulty-Aware Supervised Contrastive Learning

## ABSTRACT
Mitosis counting is an important indicator in the pathological diagnosis and histological grading of breast cancer. Currently, the counting of mitotic nuclei relies on pathologists manually reviewing whole slide images (WSI), which is time-consuming and susceptible to subjective biases. Due to the extreme class imbalance of nuclei types, large intra-class heterogeneity, and high inter-class similarity in breast pathology images, existing automated methods for mitosis detection struggle to achieve optimal results. Additionally, they inadequately address sample noise and hard sample learning, reducing detection robustness.
In this paper, we propose a difficulty-aware supervised contrast learning classification framework based on the nucleus patch level (DASC). First, the difficulty and quantity of data are balanced by the negative sample search method (NS). Then, Dynamic Memory Units (DMU) and Difficulty Level Evaluation (DLE) are used to filter out simple samples, eliminate noisy samples, and subdivide and validate the difficulty levels of hard samples to obtain a clean and refined dataset. Finally, subclass-based difficulty grading-driven supervised contrastive learning (SDSCL) aims to address the issues of intra-class heterogeneity and inter-class similarity and optimize the learning of hard samples.

#### Hardware

* 64GB of RAM
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
2.[MITOSIS14](https://mitos-atypia-14.grand-challenge.org/Dataset/)  3.[TUPAC-auxiliary](https://tupac.grand-challenge.org/Dataset/)
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





