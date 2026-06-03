# Content Understanding and Reasoning Enhanced Small-Sample Blind Image Quality Assessment
This is the source code for [CURE-IQA: Content Understanding and Reasoning Enhanced Small-Sample Blind Image Quality Assessment](https://ieeexplore.ieee.org/document/11498414).![CURE-IQA Framework](https://github.com/esnthere/CURE-IQA/blob/main/framework.jpg)

## Dependencies and Installation
Pytorch: 2.11.0 
CUDA: 13.0 
Python: 3.11

## For test:
### 1. Data preparation  
   To ensure high speed, save images and lables of each dataset with 'mat' files. Only need to run '**data_preparation_example.py**' once for each dataset.
   
### 2. Load pre-trained weight for test  
   The models pre-trained on KonIQ-10k with 1%, 5%, 10%, 25%, 80% samples are released. The files in the folder of '**model'** are obtained from open accessed source code of [Long-CLIP](https://github.com/beichenzbc/Long-CLIP). Please download the checkpoints from Long-CLIP, and put it into the folder of 'checkpoints'. 
   
   The pre-trained models can be downloaded from: [Pre-trained models](https://pan.baidu.com/s/1jPi2MgzLLZUJiu-4J9YfOQ?pwd=d9cb). Please download these files and put them in the same folder of code and then run '**test_koniq_rt'*n*'.py**' to make intra/cross dataset test for models trained on *n%* samples.
   
   
## For train:  
The training code can be available at the 'training' folder.


## If you like this work, please cite:

{
  
  author={Song, Tianshu and Cheng, Deqiang and Kou, Qiqi and Zhang, Sanyou and Huang, Yipo and Li, Leida},
  
  journal={IEEE Transactions on Multimedia}, 
  
  title={Content Understanding and Reasoning Enhanced Small-Sample Blind Image Quality Assessment}, 
  
  year={2026, Early Access},
 
  doi={10.1109/TMM.2026.3688392}
  
}

  
## License
This repository is released under the Apache 2.0 license. 
