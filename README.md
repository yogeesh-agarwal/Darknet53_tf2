# Darknet53_tf2
Face-Nonface classification Darknet-53 model trained from scratch using tf.keras (tf2) on wider face dataset.
This model also serves as backbone for the face detection model which uses Yolov3 structure.

By default training and testing dataset for this classification model is ***Wider face dataset***

**Requirements**  
1. python 3.7+
2. Tensorflow 2.2+
3. numpy
4. imgaug
5. opencv-python

*requirements.txt* can also be used to create the exact environment in which this model was created , trained and tested.

**Training**  
* This model can classfiy between images containing faces and ones without them , hence to train the classifier , choose face dataset of your choice , default is [**wider face dataset**](http://shuoyang1213.me/WIDERFACE/)
* To avoid the data imbalance problem , you can use the **gen_neg_samples.py** script to download new negative images from web which does not contain any faces.  
* The training script is programed to take pickle files containing the names of the image data, both for pos and neg data. Hence to create the pickle file , you can use the **wider_pre_processing.py** script to generate the pickle file from destined folder where your dataset is sitting and saving the pickle file into a folder of your choice. for neg samples pickle file , above mentioned script in (gen_neg_samples.py) will do the work for you.  
* Once we have the training and validation pickle files, we can now train the classifier leveraging the keras model api which btw has altered fit , train_step , test_step , compile methods to fit our need, feel free to experiment with this.  
To train, one can use the command : ```python keras_train.py```. while running this command please keep in mind to be in train directory, or else you would get path errors which can easily be resolved if wanted to.
Also setting proper path variables in the train script is also something that has to handled by the user.
* The training uses two callbacks , one of which is to save the model , ```ModelCheckpoint``` callback . Currently this is configured to save only the weights of the model , precisely storing as checkpoint files and can smoothly be used to resume training of large dataset and avoid large training hours by dividing training sessions of sub training hours and resume the training from previous cutoff everytime.  

**Testing**  
Testing of this model on default settings is fairly simple , using the command :  ```python inference.py```.  
* **inference.py** is a simple script to test the accuracy as well visualize the inferenced image with label on the window frame.  
* To give a different dataset , create a pickle file with above mentioned format and feed to the main function. default testing dataset is **celeba dataset**.
* Current trained model results to **92%** accuracy which can be experimented by increasing the number of epochs , trying dynamic LR scheduler e.t.c.

some good looking traditional metric charts from training this model (visualization using tensorboard)
![Screenshot](/data/training_charts.png)
