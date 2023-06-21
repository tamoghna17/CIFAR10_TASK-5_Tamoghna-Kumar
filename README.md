# CIFAR10_TASK-5_Tamoghna-Kumar

I took these project due to my interest in machine and deep learning.In these 4 days I have tried to implement various versions of neural networks to classify images in the cifar 10 dataset. I have learned about convolutional neural networks, the effect on accuracy due to the number of hidden layers in a neural network. It took me quite some time to learn about this and implement it. My approach revolved around cnns, mlps and different forms of neural networks on this dataset. I finaly built the neural network but have encountered limitations on creating the log file for it. 

{Dimensions of X: 32x32x3,50000}

I have created the run.sh file separately for it and my main code for it lies in the cifar10.py file. In the command for running, I have separately created train_images.npy and train_labels.npy files through the pickle module which then I added in a separate dataset folder. In the run.sh script , I have used the following command:

python train.py --lr 0.01 --momentum 0.5 --num_hidden 3 --sizes 100,100,100 --activation tanh --loss sq --opt gd --batch_size 20 --anneal true --save_dir pa1/ --expt_dir pa1/exp1/ --train dataset --test test.

I couldnt upload the train_images.npy and train_labels.npy since they were exceeding limit of 25MB. Please consider this and let me know if I need to provide any other files such as those above. 


