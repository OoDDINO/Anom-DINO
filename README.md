






# OoDDINO

OodDINO is a state-of-the-art framework for anomaly detection on road anomaly datasets. It achieves top performance on both the [RoadAnomaly Benchmark](https://paperswithcode.com/sota/anomaly-detection-on-road-anomaly) and the [SegmentMeIfYouCan Benchmark](https://segmentmeifyoucan.com/leaderboard).




## Framework 

Our framework is built on top of [MMDetection](https://github.com/open-mmlab/mmdetection), a powerful open-source object detection toolbox. Please follow the [installation instructions](https://mmdetection.readthedocs.io/en/latest/install.html) to set up the environment.
![image](https://github.com/user-attachments/assets/1617dba5-09ef-4e39-b7fc-cdf51018dca2)



## Datasets Preparation

Before using the framework, you need to download and prepare the datasets. Here are the steps:

### Validation Set

Download the validation set from the following link:  
[Validation Dataset](https://drive.google.com/file/d/1IbD_zl5MecMCEj6ozGh40FE5u-MaxPLh/view?usp=sharing)

After downloading, the dataset structure should look like this:

-- val
-- road_anomaly ... -- segment_me ...



### Training Set

Download the training set from the following link:  
[Training Dataset](https://drive.google.com/file/d/1k25FpVP4pG3ER3eXsR-go_iprZMEdEae/view?usp=sharing)

The training dataset structure should look like this:

-- train_dataset -- offline_dataset ... -- offline_dataset_score ... -- offline_dataset_score_view ... -- ood.json

## Framework Details

The framework utilizes uncertainty fusion layers. These layers can be found in the `layers` directory of the codebase. Please refer to the code for detailed implementation and usage of these layers.

## Baseline

The baseline for our framework is provided by [RPL](https://github.com/yyliu01/RPL). Please follow the instructions below to install and set it up.


### Pre-trained Weights

You can download the pre-trained weights from the following link:  
[Pre-trained Weights](https://drive.google.com/file/d/1osPT__BIqCYrBT0F-Dmi2IfbUSmuUZPd/view?usp=drive_link)

Please place the downloaded weights in the appropriate directory as per the configuration files.

Replace the threshold-net section with the RPL training code as follows:

Locate the adt-net section of your code and replace it with the relevant code from the RPL repository for training. You can follow the instructions in the RPL repository to adapt the training process.

you can perform inference using the following command:

1. Go to the RPL directory, Run the inference script:

   ```
   cd /rpl_corocl.code
   python test.py
   ```

If the code fails to run, please ensure that **MMDetection** is properly installed according to its official installation guide.

2. Replace the `layers` and `configs` folders in MMDetection with the provided versions from our project.

3. Run the following command to perform inference:

   ```
   python inference_ra.py
 
Keep the box information in the output results for further processing. Then execute the following command to train the threshold:
  ```
 python train_adt.py
 ```



Acknowledgements
This framework is based on MMDetection.

The OodDINO framework has been evaluated on the RoadAnomaly and SegmentMeIfYouCan benchmarks.

The baseline for this framework is provided by RPL.

The train set for this framework is provided by S2M.


```

@article{chen2019mmdetection,
  title={MMDetection: Open mmlab detection toolbox and benchmark},
  author={Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and Liu, Ziwei and Xu, Jiarui and others},
  journal={ARXIV},
  year={2019}
}

@inproceedings{liu2023residual,
  title={Residual pattern learning for pixel-wise out-of-distribution detection in semantic segmentation},
  author={Liu, Yuyuan and Ding, Choubo and Tian, Yu and Pang, Guansong and Belagiannis, Vasileios and Reid, Ian and Carneiro, Gustavo},
  booktitle=ICCV,
  pages={1151--1161},
  year={2023}
}

@inproceedings{zhao2024segment,
  title={Segment every out-of-distribution object},
  author={Zhao, Wenjie and Li, Jia and Dong, Xin and Xiang, Yu and Guo, Yunhui},
  booktitle=CVPR,
  pages={3910--3920},
  year={2024}
}


```
More visual results for SMIYC are below:

![image](https://github.com/user-attachments/assets/e7909315-ec53-40c7-bb85-abf888452be6)

![image](https://github.com/user-attachments/assets/d92739b7-02cb-42eb-b3c6-0ec57841dda4)

