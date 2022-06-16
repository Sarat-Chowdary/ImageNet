ImageNet-mini

The dataset is an open source dataset available on kaggle
This dataset has 1000 classes with over 45k images and is called the ImageNet-mini

You can run the evaluate.py file after also downloading the hdf5 file to predict using the model

Because the number of classes is so large and the dataset being small in comparision the train accuracy and the validation accuracy could only reach about 65%
You could try other architectures on this problem which could do better.
Only a globalavgpooling layer, a dropout layer and an output dense layer were added onto the inceptionV3 model

More data and perhaps a better architecture would benefit in solving this problem with much better results.

