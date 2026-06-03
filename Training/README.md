# Training Code for CURE-IQA: Content Understanding and Reasoning Enhanced Small-Sample Blind Image Quality Assessment
This is the training example of CURE-IQA on the LIVEW dataset, which is small enough to re-train. The trainning process is the same for other datasets:

## 1. Data preparation

   To ensure high speed, save training images and lables into 'mat' files. The preparation process please refer to the published paper [CURE-IQA](https://ieeexplore.ieee.org/document/10355923).  Please run '**data_preparation_example_for_livew.py**' to save the training images and labels, and '***.json**' contains the text obtained from VLM/LLM.
   
## 2. Training the model

   Run'**livew_rt5.py.py**' to train the model. The files in the folder of '**model'** are obtained from open accessed source code of [Long-CLIP](https://github.com/beichenzbc/Long-CLIP) . 
  
 
 


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

