## STRONG: Spatio-Temporal Reinforcement Learning for Cross-Modal Video Moment Localization
This is our implementation for the paper:

Da Cao, Yawen Zeng, Meng Liu, Xiangnan He, Meng Wang, and Zheng Qin. 2020. STRONG: Spatio-Temporal Reinforcement Learning for Cross-Modal Video Moment Localization. In The ACM International Conference on Multimedia (ACM MM '20). ACM, Seattle, United States.

## Environment Settings
We use the framework pytorch.

* pytorch version: '1.2.0'
* python version: '3.5'

## STRONG
The released code consists of the following files.
```
--data
--log
--feature_all
--cal4log.py
--main.py
--MADDPG
--IMGDDPG
--model
--memory
--spp
--utils
--randomProcess
```

## Example to run the codes
Run STRONG：
```
python main.py
```

## Example to get the results
Run log:
```
python cal4log.py
```
There are a lot of experimental records in the ./log

## Dataset
We provide two processed datasets: Charades-STA && TACoS
The strategy of multi-scale sliding windows is utilized to segment each video with the size of [64, 128, 256, 512] frames with 80% overlap and we randomly selected 80% and 20% of them for training and testing, respectively.

All features are saved in ./feature_all_train, ./feature_all_test. 
* These two processed features are available for downloading here: https://drive.google.com/open?id=1-AMToMuTlPRY1C2n0ZoyKrwBsVPehbFK
* The original videos and their corresponding caption annotations/querys: https://github.com/jiyanggao/TALL and http://www.coli.uni-saarland.de/projects/smile/tacos

## Workspace
./data/video_cut2image that processes images in advance as a runtime workspace.

#


Last Update Date: Jul 28, 2020
