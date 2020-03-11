import numpy as np

class NeuralNetwork:
    #def __init__(self, setup=[[68,25,"sigmoid",0],[25,1,"sigmoid",0]],lr=.05,seed=1,error_rate=0,bias=1,iter=500,lamba=.00001,simple=0):
    def __init__(self,inputs,outputs,activation='sigmoid',lr=0.05,bias=1,iter=500,lamda=0.00001,shape=(8,3,8)):
        self.activation = 'sigmoid'
        self.lr = lr
        self.init_bias = bias
        self.shape=shape
        self.weights = dict()
        self.weight_correction = dict()
        self.biases = dict()
        self.bias_correction = dict()
        self.lamda = lamda
        self.z = dict()
        self.a = dict()
        self.inputs = inputs
        self.outputs = outputs
        self.iter = iter
    def make_weights(self):
        self.a[1] = self.inputs
        for layer in range(1,len(self.shape),1):
            # initialize weights to random values for each layer
            weight_matrix = np.zeros((self.shape[layer-1],self.shape[layer]))
            weight_correction_matrix = np.zeros((self.shape[layer-1],self.shape[layer]))
            for i in range(weight_matrix.shape[0]):
                for j in range(weight_matrix.shape[1]):
                    weight_matrix[i,j] = np.random.random() * 0.1
            self.weights[layer] = weight_matrix
            self.weight_correction[layer] = weight_correction_matrix
            # initialize biases to the input value
            self.a[layer+1] = np.zeros((self.a.get(layer).shape[0],weight_matrix.shape[1]))
            bias_matrix = np.full((self.a.get(layer).shape[0],weight_matrix.shape[1]),fill_value=self.init_bias)
            bias_correction_matrix = np.zeros((self.a.get(layer).shape[0],weight_matrix.shape[1]))
            self.biases[layer] = bias_matrix
            self.bias_correction[layer] = bias_correction_matrix
        print('starting weights')
        print(self.weights)
    def feedforward(self):
        for layer in range(1,len(self.shape),1):
            z = np.dot(self.a.get(layer),self.weights.get(layer)) + self.biases.get(layer)
            self.z[layer+1] = z
            a = np.zeros((z.shape[0],z.shape[1]))
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    a[i,j] = activation(z[i,j], self.activation)
            self.a[layer+1] = a
        print('a')
        print(self.a)
    def backprop(self):
        deltas = dict()
        last_layer = len(self.shape)
        f_prime_z = np.zeros((self.z.get(last_layer).shape[0],self.z.get(last_layer).shape[1]))
        for i in range(f_prime_z.shape[0]):
            for j in range(f_prime_z.shape[1]):
                f_prime_z[i,j] = der_activation(self.z.get(last_layer)[i,j],self.activation)
        delta_output = np.multiply(-(self.outputs-self.a.get(last_layer)),f_prime_z)
        print('delta output')
        print(delta_output)
        deltas[last_layer] = delta_output
        # gradient of cost function for hidden layers (all but first and last)
        for layer in range(len(self.shape)-1,1,-1):
            f_prime_z = np.zeros((self.z.get(layer).shape[0],self.z.get(layer).shape[1]))
            for i in range(f_prime_z.shape[0]):
                for j in range(f_prime_z.shape[1]):
                    f_prime_z[i,j] = der_activation(self.z.get(layer)[i,j],self.activation)
            delta_layer = np.multiply(np.dot(deltas[layer+1],np.transpose(self.weights.get(layer))),f_prime_z)
            deltas[layer] = delta_layer
            print('delta hidden layer')
            print(delta_layer)
        for layer in range(1,len(self.shape),1):
            gradient_W = np.dot(np.transpose(self.a.get(layer)),deltas.get(layer+1))
            gradient_b = deltas[layer+1]
            self.weight_correction[layer] = self.weight_correction.get(layer) + gradient_W
            self.bias_correction[layer] = self.bias_correction.get(layer) + gradient_b
            m = self.inputs.shape[0]
            new_weights = self.weights.get(layer) + self.lr * (((1/m) * self.weight_correction[layer])+ self.lamda * self.weights[layer])
            new_biases = self.biases[layer] + self.lr * ((1/m) * self.bias_correction[layer])
    def fit(self):
        self.make_weights()
        for i in range(self.iter):
            self.feedforward()
            self.backprop()
        print('final weights')
        print(self.weights)
    def predict(self):
        self.feedforward()
        print('predicted outputs')
        print(self.a)
        
def activation(x,type):
    if type == 'sigmoid':
        # sigmoid activation function
        return (1 / (1 + np.exp(-x)))
def der_activation(x,type):
    if type == 'sigmoid':
        return (activation(x,'sigmoid')*(1-activation(x,'sigmoid')))
        
        

        
        