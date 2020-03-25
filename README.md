# kaggle-bengali
This code was used during the competition.

*Public LB  : 40/2,059*

*Private LB : 175/2,059*

The private test-set had unseen graphemes, for which we(our models) were not prepared and therefore, resulted in massive LB shake-down

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
conda create -n bengali python=3.6
source activate bengali
pip install -r requirements.txt
```

## Prepare dataset
The *data* folder already contains `train.csv` and the splits: `train_b_fold1_184855.npy` and `valid_b_fold1_15985.npy` used for training. So we only need to download the training images; will use the *feather* version of the dataset for faster loading, which can be downloaded via following commands

```
$ kaggle datasets download -d corochann/bengaliaicv19feather
$ unzip bengaliaicv19feather.zip -d data
```
After downloading and unzipping, the data directory should look like this:
```
data
  +- train.csv
  +- train_b_fold1_184855.npy
  +- valid_b_fold1_15985.npy
  +- train_image_data_0.feather
  +- train_image_data_1.feather
  +- train_image_data_2.feather
  +- train_image_data_3.feather
  +- test_image_data_0.feather
  +- test_image_data_1.feather
  +- test_image_data_2.feather
  +- test_image_data_3.feather
```

## Training
The training was done using `se_resnext50_32x4d` at original size of `137x236` and `efficient-netb3` at image size of `224x224`

The training followed the following strategy:


* train model with with only [cutmix](https://arxiv.org/abs/1905.04899) for `n` epochs
* take [swa](https://arxiv.org/abs/1803.05407) of last/top `k` checkpoints
* initialize with *swaed* checkpoint and add [GridMask](https://arxiv.org/abs/2001.04086) augmentation to train for `N` epochs
* take [swa](https://arxiv.org/abs/1803.05407) of last/top `k` checkpoints and submit

> In my experiments, `n` was 50, `k` was 30-50 and `N` was 50-100. These were tuned manually observing the validation score. So I recommend you to tune them observing the validation score while following the training strategy mentioned above

For training, pretrained model of `se_resnext50_32x4d` is needed which can be downloaded via
```
wget http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth
```

### Train models
Training without `GridMask` augmentation for `n` epochs. Training has to be stopped *manually* after `n` epochs
```
python train.py --model_name effnetb3
```
Training with `GridMask` augmentation for `N` epochs. Training has to be stopped *manually* after `N` epochs
```
python train.py --model_name effnetb3 --use_gridmask
```

### Take SWA
```
python swa.py --num_snapshots 30 --model_name effnetb3
```
### Validate with *swaed* model to get CV score 
```
python validate.py --initial_checkpoint ./runs/effnetb3/models_swa_30.pth --model_name effnetb3
```
This will save probabilities and ground-truth values, which can later used to check the ensemble score with different models

### Get CV score of ensembled models
Assuming the saved probabilites are at `./runs/effnetb3/submit` and `./runs/serex50/submit`, their ensemble score can be obtained via
```
python ensemble.py
```
>Train more models and add their saved probabilities-directory in `ensemble.py` for better ensemble score


