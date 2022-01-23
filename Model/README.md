**[!]** You can run on your own computer by following instruction below or use my setup on [Google Drive](https://drive.google.com/drive/folders/1B3t8Z3Gn0qrFY8_4z9MTQDOpC2EeATS9?usp=sharing)
(run **RunModel.ipynb** on **Google Colab**)
## Dependencies
**Environment** : Python 3.

You **must install** these Python libraries:
- [Numpy](https://numpy.org/)
```
pip install numpy
```
- [Pandas](https://pandas.pydata.org/)
```
pip install pandas
```
- [tqdm](https://tqdm.github.io/)
```
pip install tqdm
```
- [Pillow](https://pillow.readthedocs.io/en/stable/)
```
pip install Pillow
```
- [OpenCV](https://github.com/opencv/opencv-python)
```
pip install opencv-python
```
- [Keras](https://keras.io/)
- [Tensorflow 2](https://www.tensorflow.org/)
```
pip install tensorflow
```
## Generating Dataset
You can generate dataset by running the `generated_data.py` file.
```console
python generated_data.py --dir [des folder] --data_size [number of images] --save_image
```

**[!]** If you want to train the model, you can generate your own dataset or use our generated dataset at [Google Drive](https://drive.google.com/drive/folders/1-2u_ouIh00FhOH0tKHqWaHx6kmm1JoPd?usp=sharing)

## Training model
Before training the model, you should edit the path to data folder and data size in `parameters.py` file.

Running the codes below to train the model:

```console
python train.py --name [Name of the model]
```

The weights of model would be saved in *checkpoints* folder.
Moreover, you can setting the model by editing the `parameter.py` and `model.py` files.

## Predicting
You can predict the texts in prepared images by running the following codes:
```console
python predict.py --name [Name of the pretrained model] --dir [Folder contains dataset] --num [Number of images]
```

## Current result
Our best model **`OCR2`** predicts correct **89.31%** characters on 1000 images.