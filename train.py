import cifar10 
import numpy as np
import argparse

if __name__ == "__main__":
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Convolutional Neural Network for CIFAR-10')
        parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for momentum-based algorithms')
        parser.add_argument('--num_hidden', type=int, default=1, help='Number of hidden layers')
        parser.add_argument('--sizes', type=str, default='128', help='Sizes of hidden layers')
        parser.add_argument('--activation', type=str, default='tanh', choices=['tanh', 'sigmoid'], help='Activation function')
        parser.add_argument('--loss', type=str, default='ce', choices=['sq', 'ce'], help='Loss function')
        parser.add_argument('--opt', type=str, default='gd', choices=['gd', 'momentum', 'nag', 'adam'], help='Optimization algorithm')
        parser.add_argument('--batch_size', type=int, default=1, choices=[1, 5, 10, 15, 20], help='Batch size')
        parser.add_argument('--anneal', type=bool,default=True, help='Halve learning rate if validation loss decreases')
        parser.add_argument('--save_dir', type=str, default='model.pkl', help='Directory to save the model')
        parser.add_argument('--expt_dir', type=str, default='logs', help='Directory to save log files')
        parser.add_argument('--train', type=str, required=True, help='Path to the training dataset')
        parser.add_argument('--test', type=str, required=True, help='Path to the test dataset')
        args = parser.parse_args()
        return args

    args = parse_arguments()
    sizes = list(map(int, args.sizes.split(",")))

    X = np.load(args.train + "/train_images.npy", allow_pickle=True)
    Y = np.load(args.train + "/train_labels.npy", allow_pickle=True)
    Y = np.reshape(Y, (50000, 1))
    labels=[]

    list=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    classes={}
    for i in range(10):
        classes[list[i]]=i


    for i in Y:
        l = []
        n=classes[i[0]]
        for j in range(10):
            l.append(0)
        l[n]=1    
        labels.append(l)

    train_y=np.array(labels).reshape((50000, 10))
    NN = cifar10.Neural_Network(momentum=args.momentum, num_hidden=args.num_hidden, sizes=sizes, activation=args.activation, lr=args.lr, loss=args.loss, opt=args.opt)
    NN.train(X, train_y, epochs=20, batch_size=args.batch_size)





