
# Temporal Action Localization with Cross Layer Task Decoupling and Refinement（CLTDR-GMG）
Our code is built upon the codebase from [ActionFormer](https://github.com/happyharrycn/actionformer_release),[TemporalMaxer](https://github.com/TuanTNG/TemporalMaxer) and [Tridet](https://github.com/dingfengshi/TriDet), and we would like to express our gratitude for their outstanding work.

![](./doc/Fig1.png)

![](./doc/Fig2.png)

![](./doc/Fig3.png)

![](./doc/Fig4.png)

## Environment
- Ubuntu20.04
-  NVIDIA RTX 4090 GPU
-  Python3.8, Pytorch2.0 and CUDA11.8
-  `h5py,
joblib,
matplotlib,
numpy,
pandas,
PyYAML,
scikit_learn,
scipy,
setuptools`


## Install NMS
```
cd ./libs/utils
python setup.py install --user
cd ../..
```


## Pretrained Models



## Training and Evaluation
Train: `python ./train.py ./configs/xxxx.yaml --save_ckpt_dir ./ckpt/xxx`

Eval: `python ./eval.py ./configs/xxxx.yaml <path of the weights>`


