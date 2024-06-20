# cifar10

{Dimensions of X: 32x32x3,50000}

Classification on the cifar-10 dataset

Steps:
1)Clone this repository.
2)Download the [dataset](https://www.kaggle.com/competitions/cifar-10/data)
3)Change the hyperparamters as needed in run.sh file:
* –lr (initial learning rate for gradient descent based algorithms)
* –momentum (momentum to be used by momentum based algorithms)
* –num hidden (number of hidden layers - this does not include the 32x 32 dimensional input layer and the 10 dimensional output layer) 
* –sizes (a comma separated list for the size of each hidden layer) 
* –activation (the choice of activation function - valid values are tanh/sigmoid) 
* –loss (possible choices are squared error[sq] or cross entropy loss[ce]) 
* –opt (the optimization algorithm to be used: gd, momentum, nag, adam - you will be implementing the mini-batch version of these algorithms) 
* –batch size (the batch size to be used - valid values are 1 and multiples of 5)
* –anneal (if true the algorithm should halve the learning rate if at any epoch the validation loss decreases and then restart that epoch) 
* –save dir (the directory in which the pickled model should be saved - by model we mean all the weights and biases of the network) 
* –expt dir (the directory in which the log files will be saved - see below for a detailed description of which log files should be generated) 
* –train (path to the Training dataset) 
* –test (path to the Test dataset) 
argparse module in python for parsing these parameters

run code using below command
```
python train.py --lr 0.01 --momentum 0.5 --num_hidden 3 --sizes 100,100,100 --activation tanh --loss sq --opt gd --batch_size 20 --anneal true --save_dir pa1/ --expt_dir pa1/exp1/ --train dataset --test test.
```


