# CoronoVirus-Image-Project
## Approach

Download and merge the following data sets
1. Covid 19 X-Ray DataSet
2. Normal Chest X-Ray Images
3. Bacterial and Viral Pneumonia Chest X-Ray Images

After downloading and merging the datasets we need to load these images into Pytorch for Deep Learning.

To load these images into Pytorch

We need to first convert the raw data into `Pytorch Tensors` these tensors need to be split into `X_train`,  `y_train` and `X_test` and `y_test`.

- `X_train` - contains a list of images meant for training
- `y_train` - contains a list of labels for train images
- `X_test` - contains a lisf of images meant for testing
- `y_test` - contains a list of labels for test images

We might also want to have some sort of validation data to try and improve our model.

- `X_train` - contains a list of images to perform validation
- `y_train` - contains a list of labels for the validation images


### Reading Covid 19 Chest X-ray
 Before we go throug the Covid 19 X ray data set

 We need to understand the scheme documented [here](https://github.com/ieee8023/covid-chestxray-dataset/blob/master/SCHEMA.md)

## Libraries
1.  TorchXrayVision - an open source software to work with any chest xray datasets
2.  Pandas
3.  Pytorch
## Tools

1. Google Colab


