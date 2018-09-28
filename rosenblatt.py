#Simple Rosenblatt perceptron with variable number of inputs n

import numpy as np
from random import seed
from random import uniform

class Rosenblatt():

    def __init__(self,n,nepoch,eta=0.1,thresh=1.0e-06,actfunc='heaviside',
                 randseed=123,initweight='random'):
        self.eta = eta # learning rate
        self.n = n # number of input attributes
        self.nepoch = nepoch # max. number of epochs (times to run through training data)
        self.thresh = thresh # error threshold for determining convergence
        self.actfunc = actfunc # choice of (step) activation function
                               # 'sgn' or 'heaviside' (default)
        self.randseed = randseed # random seed
        self.initweight = initweight # weight initialisation method
                                     # 'zero' or 'random' (default)

    def printsoln(self, weights, output, targets):
        print "Writing output\ninstance\toutput\ttarget"
        for i in range(np.shape(dataset)[0]):
            print i, "\t\t", output[i], "\t", targets[i]
        print "Writing weights (zeroth connection is from bias neuron)\nConnection\tweight"
        for i in range(len(weights)):
            print i, "\t\t", "%.5f" % (weights[i])
        return

    # Heaviside step function
    def heaviside(self, z):
        if z < 0:
            return 0
        elif z >= 0:
            return 1

    # Sign step function
    def sgn(self, z):
        if z < 0:
            return -1
        elif z == 0:
            return 0
        elif z > 0:
            return 1

    #make a prediction based on weights
    def predict(self, row, weights):
        activation = weights[0] # bias neuron
        for i in range(np.shape(row)[0]):
            activation += weights[i+1]*row[i]
        if self.actfunc == 'sgn':
            output = self.sgn(activation)
        elif self.actfunc == 'heaviside':
            output = self.heaviside(activation)
        return output

    #estimate perceptron weights using stochastic gradient descent
    def train(self, dataset, targets, weights):
        i = 0
        sum_error = 1.0
        output = [0 for m in range(np.shape(dataset)[0])]
        while i <= (self.nepoch - 1) and sum_error > self.thresh:
            sum_error = 0.0
            for j in range(np.shape(dataset)[0]):
                output[j] = self.predict(dataset[j,:], weights)
                error = targets[j] - output[j]
                sum_error += abs(error)
                # update weights
                weights[0] = weights[0] + self.eta*error # bias neuron
                for k in range(np.shape(dataset)[1]):
                    weights[k+1] = weights[k+1] + self.eta*error*dataset[j,k]
            sum_error = sum_error*(1.0/self.n)
            i += 1
        if i == (self.nepoch):
            print "Met maximum number of cycles = %i\nError = %.6f\n" \
                  % (i, sum_error)
        else:
            print "Error threshold met in %i cycles" % i
        return weights, output

    # read training data
    def readdata(self, fname):
    # data must be in format: input / n*output attributes / target neuron output
        dataset = []
        targets = [] # target values of neuron outputs
#        lines = [line.strip("\n") for line in open(fname)]
        with open(fname, "r") as fh:
            for line in fh:
                dataset.append(line.split()[0:self.n])
                targets.append(line.split()[-1])
        dataset = np.array(dataset,dtype=np.float)
        targets = np.array(targets,dtype=np.int)
        return dataset, targets

    # main function for driving perceptron learning
    def perceptron(self, dataset, targets):
        seed(self.randseed)
        # initialise weights to small random value
        if self.initweight == 'random':
            weights = [uniform(-1.0,1.0) for i in range(np.shape(dataset)[1]+1)]
        elif self.initweight == 'zero':
            weights = [0.0 for i in range(np.shape(dataset)[1]+1)]
        # use perceptron learning rule to train weights
        weights, output = self.train(dataset, targets, weights)
        self.printsoln(weights, output, targets)
        return weights

    # test trained perceptron using sample data
    def testperceptron(self, testset, weights, expect):
        output = [0 for i in range(np.shape(testset)[0])]
        sum_error = 0.0
        print "Predictions for test set\nInstance\toutput\texpected"
        for j in range(np.shape(testset)[0]):
            output[j] = self.predict(testset[j,:], weights)
            error = abs(output[j] - expect[j])
            sum_error += error
            print j, "\t\t", output[j], "\t", expect[j]
        print "Percentage error is: ", (1.0/self.n)*sum_error*100.0
        return

#Driver code
fname = 'rosenblatt_dataset.dat' # name of file containing training data
fname2 = 'rosenblatt_testdata.dat' # name of file containing test data
rb1 = Rosenblatt(2,5,initweight='zero')

dataset, targets = rb1.readdata(fname)
print dataset
weights = rb1.perceptron(dataset, targets)

testset, expect = rb1.readdata(fname2)
rb1.testperceptron(testset, weights, expect)
