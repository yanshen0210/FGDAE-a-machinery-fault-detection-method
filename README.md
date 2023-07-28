# FGDAE:  A new machinery anomaly detection method towards complex operating conditions
* Core codes for the paper ["FGDAE: A new machinery anomaly detection method towards complex operating conditions"](https://www.sciencedirect.com/science/article/abs/pii/S0951832023002338)
* Created by Shen Yan, Haidong Shao, Zhishan Min, Jiangji Peng, Baoping Cai, Bin Liu.
* Journal: Reliability Engineering and System Safety

![FGDAE](https://github.com/yanshen0210/FGDAE-a-machinery-fault-detection-method/blob/main/framework.jpg)
## Our operating environment
* Python 3.8
* pytorch  1.10.1
* and other necessary libs

## Guide 
* This repository provides a concise framework for machinery anomaly detection. 
* It includes the pre-processing and graph composition process for the data and the model proposed in the paper. 
* We have also integrated 4 baseline methods for comparison.
* `Graph_train_val_test.py` is the train&val&test process of our proposed method; `Base_train_val_test.py` is the train&val&test process of base methods.
* You need to load the data in following Datasets link at first, and put them in the `data` folder. Then run in `General_procedure.py`
* You can also adjust the structure and parameters of the model to suit your needs.

## Datasets
* [Case1](https://drive.google.com/drive/folders/1NXA2eh_8OFHisuNl05y3KTWBbgqmCYTi?usp=sharing)
* [Case2](https://drive.google.com/drive/folders/1IVh5Y7LvUVxECtLDu08pfUEn93GLk79M?usp=sharing)

## Run the code
### Case1
* `General_procedure.py` --data_dir "./data/Case1"; --data_num ['200Hz_0N', '300Hz_1000N', '400Hz_1400N']; 
<br> --sensor_number 6; --fault_num 7; --unbalance_train [200, 100, 10]
### Case2
* `General_procedure.py` --data_dir "./data/Case2"; --data_num ['G_20_0', 'G_30_2']; 
<br> --sensor_number 8; --fault_num 5; --unbalance_train [200, 10]

## Pakages
* `data` needs loading the Datasets in above links
* `datasets` contians the pre-processing and graph composition process for the data
* `models` contians the proposed model and 4 base models
* `utils` contians two types of train&val&test processes

## Citation
If our work is useful to you, please cite the following paper, it is the greatest encouragement to our open source work, thank you very much!
```
@paper{FGDAE,
  title = {FGDAE: A new machinery anomaly detection method towards complex operating conditions},
  author = {Shen Yan, Haidong Shao, Zhishan Min, Jiangji Peng, Baoping Cai, Bin Liu},
  journal = {Reliability Engineering and System Safety},
  volume = {236},
  pages = {109319},
  year = {2023},
  doi = {https://doi.org/10.1016/j.ress.2023.109319},
  url = {https://www.sciencedirect.com/science/article/abs/pii/S0951832023002338},
}
```

## Contact
- yanshen0210@gmail.com
