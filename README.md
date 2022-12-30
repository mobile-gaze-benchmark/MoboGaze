# MoboGaze
## MoboGAZE: A Benchmark and Novel Baseline for Lightweight Deep Learning based Gaze Estimation


## Results:
[![results](/docs/results.jpg)](https://github.com/mobile-gaze-benchmark/MoboGaze)
## Dependencies
- python3.7+
- pytorch1.9+
- torchvision


### Datasets
We follow [Cheng et al.](http://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/) and use the same datasets. Please refer to [here](http://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/) to download the preprocessed datasets.

### Pretrained Models
- Baidu Netdisk (百度网盘)：https://pan.baidu.com/s/1D_qOD7mcy27nwCWgw1WZqw 
- Extraction Code (提取码)：xur9

- Google Drive: https://drive.google.com/drive/folders/1d-qAuiq4WL4X-ZQBKH9MjdIiW4dfzSKo?usp=share_link

Please put the model under `pretrain` folder.

### Train
```
./train_gaze360.sh
```
### Test
```
./test_gaze360.sh
```




## Licesnse
For academic and non-commercial use only. The whole project is under the MIT license.

## Citation
If you find this project useful in your research, please consider citing:

