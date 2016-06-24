import numpy as np
import matplotlib.pyplot as plt

# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
Y = np.array(([75], [82], [93]), dtype=float)

# Normalize
X = X/np.amax(X, axis=0)
Y = Y/100 #Max test score is 100


class NeuralNetwork(object):
    def __init__(self):
        self.inputLayers = 2
        self.outputLayers = 1
        self.hiddenLayers = 3
        
        #initialize weights
        self.w1 = np.random.randn(self.inputLayers,self.hiddenLayers)
        self.w2 = np.random.randn(self.hiddenLayers, self.outputLayers)
        
    def forward(self, X):
        #lets multiplicate matrices
        self.zh = np.dot(X, self.w1)
        self.ah = self.sigmoid(self.zh)
        self.zout = np.dot(self.ah,self.w2)
        self.yHat = self.sigmoid(self.zout)
        return self.yHat;
    
    def sigmoid(self, z):
        #Z is a matrix
        return 1/(1+np.exp(-z))

NN = NeuralNetwork()
yHat = NN.forward(X)
print(yHat)
#TEST SIGMOID    
# def sigmoid(z):
#         return 1/(1+np.exp(-z))
#     
# test = np.arange(-6,6,0.01)
# plt.plot(test, sigmoid(test),linewidth=2)
# plt.grid(1)
# plt.show()