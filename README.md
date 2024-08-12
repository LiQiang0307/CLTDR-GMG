
# Temporal Action Localization with Cross Layer Task Decoupling and Refinement（CLTDR-GMG）
Our code is built upon the codebase from [ActionFormer](https://github.com/happyharrycn/actionformer_release),[TemporalMaxer](https://github.com/TuanTNG/TemporalMaxer) and [Tridet](https://github.com/dingfengshi/TriDet), and we would like to express our gratitude for their outstanding work.

![](./docs/fig2.png)

![](./docs/fig3.png)

## Environment
- Ubuntu20.04
-  NVIDIA RTX A6000 GPU
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

We provide our pre-trained models. 

Baidu link：https://pan.baidu.com/s/1Gzok3qnKYEsWo_KHtBfzEg?pwd=6wlu 


## Training and Evaluation
Train: `python ./train.py ./configs/xxxx.yaml --save_ckpt_dir ./ckpt/xxx`

Eval: `python ./eval.py ./configs/xxxx.yaml <path of the weights>`

## Results

![](./docs/table1.png)

![](./docs/table2.png)

![](./docs/table3.png)

![](./docs/table4.png)
