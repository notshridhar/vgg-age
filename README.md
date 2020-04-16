# Age Prediction using Transfer Learning on VGG Architecture

Inspired from paper: <br>
https://arxiv.org/abs/1709.01664 <br>
Z. Qawaqneh, A. Mallouh and B. Barkana - 
__"Deep Convolutional Neural Network for Age Estimation based on VGG-Face Model"__, University of Bridgeport - 2017
<br><br>

## Architecture
![](images/arch.png)

The convolutional layers are unchanged and are the same as those of VGG16 architecure. The fully connected layers, however are reduced because of the low number of target classes. <br>

__FC6__ : 512 neurons <br>
__FC7__ : 512 neurons <br>
__FC8__ : 8 neurons (output) 
<br><br>

## Dataset
The OUI-Adience Face Image Project is used to train the network. <br>
https://talhassner.github.io/home/projects/Adience/Adience-data <br>

__Total number of photos__: 26,580 <br>
__Total number of subjects__: 2,284 <br>
__Age groups__: (0-2), (4-6), (8-13), (15-20), (25-32), (38-43), (48-53), (60-100)
<br><br>

## Model
The above network is implemented in pytorch. Take the notebook and the source files with a grain of salt and ensure the files and directory structures rules are met. 

The weights of the convolutional network are loaded from VGG model trained on VGGFace dataset. The model is then fine-tuned. This is essential because of the small number of training data. This also prevents overfitting and ensures faster convergence. 

The weights for pytorch model can be found here : <br>
http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth

__Parameter count__: <br>
Conv &nbsp; : 14714688 <br>
Dense : 13112328 <br>
Total &nbsp;&nbsp; : 27827016 <br>
__Memory__ : 106.15MB
<br><br>

## Training
20% of the dataset is used for validation and the other 80% is used for training. 
The model was trained for 3 epochs with constant learning rate 0.01 and momentum 0.9.

The training was done on Google Colab with GPU acceleration. The corresponding notebook is `vgg16.ipynb`. The model took approximately 30 minutes to train.
<br><br>

## Results
__Accuracy__ : The final validation accuracy is 69.6% while final training accuracy is 68.6%. This shows that the model is generalizing to the inputs it has never seen before.

__Confusion Matrix__ : 
It appears that the model confuses between age groups 15-20 and 25-32 and also 38-43 and 25-32. This is intuitive because the facial features vary very slightly between those age groups.
| Pred→ |  0-2 |  4-6 | 8-13 | 15-20 | 25-32 | 38-43 | 48-53 | 60-|
| ---     | :---:| :---:| :---:| :---: | :---: | :---: | :---: | :---: |
|__0-2__  | __85.74__| 13.88|  0.00|  0.00 |  0.00 |  0.00 |  0.00 |  0.38 |
|__4-6__  |  4.03| __82.87__|  9.82|  0.25 |  2.02 |  0.50 |  0.25 |  0.25 |
|__8-13__ |  0.22|  5.41| __75.76__|  3.68 | 13.42 |  1.30 |  0.22 |  0.00 |
|__15-20__|  0.27|  0.82|  6.59| __33.24__ | 53.02 |  5.22 |  0.82 |  0.00 |
|__25-32__|  0.00|  0.09|  1.18|  2.09 | __78.56__ | 17.44 |  0.64 |  0.00 |
|__38-43__|  0.00|  0.58|  0.97|  0.39 | 26.99 | __63.50__ |  6.21 |  1.36 |
|__48-53__|  0.00|  1.10|  1.10|  0.55 |  7.18 | 39.23 | __39.78__ | 11.05 |
|__60-__  |  0.00|  0.56|  0.56|  0.00 |  2.25 | 23.60 | 29.78 | __43.26__ |
	
	
__One off accuracy__ : One off accuracy is the model predicting one more or one less than the actual age group. For this model, the one-off accuracy is 91.49%
	
__Predictions__ : Some of the predictions along with the images are shown. The labels specify the prediction class along with the actual classes. 
![](images/preds.png)
