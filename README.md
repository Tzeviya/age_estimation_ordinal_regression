# age_estimation_ordinal_regression
Pytorch implementation of ordinal regression algorithms [OR-CNN](https://openaccess.thecvf.com/content_cvpr_2016/papers/Niu_Ordinal_Regression_With_CVPR_2016_paper.pdf) and [CORAL](https://arxiv.org/pdf/1901.07884.pdf) for age estimation datasets.


This code is in some places based on the code [here](https://github.com/Raschka-research-group/coral-cnn).

------


## Installation instructions

- Python 3.6+ 

- Pytorch 1.3.1

- numpy

- librosa

- soundfile

- Download the code:
    ```
    git clone https://github.com/Tzeviya/age_estimation_ordinal_regression.git
    ```


## How to use

This implementation uses the following datasets:

- [Historical Color Images](http://graphics.cs.cmu.edu/projects/historicalColor/)

- [Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection)

- [UTKFace](https://susanqq.github.io/UTKFace/)

- [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

- [AFAD](https://afad-dataset.github.io/)

- [TIDIGITS](https://catalog.ldc.upenn.edu/LDC93S10)

For the desired dataset, create files: ```train.txt```, ```val.txt```, ```test.txt``` and place them all under the ```dataset_name``` subfolder within the ```datasets``` folder. See example in the folder.


## ORCNN

To train ORCNN (e.g. on the historical color images dataset), run:

```
    python main.py  --num_classes [num_of_labels]
    			--historical
    			--pretrain
    			--arc [desired_cnn_architecture]
    			--bn 
    			--opt sgd
    			--save_file [model_name]
    			--cuda

```


## ORCNN

To train CORAL, run the same as above with the added ```--coral``` argument.