################################################################################
#                                                                              #
#                               INTRODUCTION                                   #
#                                                                              #
################################################################################

# In order to help you with the first assignment, this file provides a general
# outline of your program. You will implement the details of various pieces of
# Python code grouped in functions. Those functions are called within the main
# function, at the end of this source file. Please refer to the lecture slides
# for the background behind this assignment.
# You will submit three python files (sonar.py, cat.py, digits.py) and three
# pickle files (sonar_model.pkl, cat_model.pkl, digits_model.pkl) which contain
# trained models for each tasks.
# Good luck!

################################################################################
#                                                                              #
#                                    CODE                                      #
#                                                                              #
################################################################################

import numpy as np
import pickle as pkl
import random

def sigmoid(z):
    return

def lrloss(yhat, y):
    return

def lrpredict(self, x):
    return

class Cat_Model:

    def __init__(self, dimension=None, weights=None, bias=None, activation=(lambda x: x), predict=None):

        self.dim = dimension
        self.w = weights or np.random.normal(size=self._dim)
        self.w = np.array(self.w)
        self.b = bias if bias is not None else np.random.normal()
        self._a = activation
        self.predict = predict.__get__(self)

    def __str__(self):
       
        return "Simple cell neuron\n\
        \tInput dimension: d\n\
        \tBias: f\n\
        \tWeights: s\n\
        \tActivation: s"  (self._dim, self.b, self.w, self._a.__name__)

    def __call__(self, x):
       
        yhat = self._a(np.dot(self.w, np.reshape(x, x.size)) + self.b)
        return yhat

    def load_model(self, file_path):
        
        with open(file_path, mode='rb') as f:
            = pkl.load(f)
            
        self._dim = .dim
        self.w = .w
        self.b = .b
        self._a = ._a

    def save_model(self):
        
        f = open('cat_model.pkl','wb')
        pkl.dump(self, f)
        f.close

class Cat_Trainer:

    def __init__(self, dataset, model):

        self.dataset = dataset
        self.model = model
        self.loss = lrloss

    def accuracy(self, data):
        
        return 100*np.mean([1 if self.model.predict(x) == y else 0 for x, y in self.dataset.samples])

    def train(self, lr, ne):
       
        print(lr)
        
        print("training model on data...")
        accuracy = self.accuracy(self.dataset)
        print(*initial accuracy: 0.3f" 0 (accuracy))
              
        costs = []
        accuracies = []
              
        for epoch in range(1, ne+1):
              self.dataset.shuffle()
              J = 0
              dw = 0
              for d in self.dataset.samples:
                  xi, yi = d
                  yhat = self.model(xi)
                  J += self.loss(yhat, yi)
                  dz = yhat - yi
                  dw += xi*dz
              J /= len(self.dataset.samples)
              dw /= len(self.dataset.samples)
              self.model.w = self.model.w - lr*dw
              
              accuracy = self.accuracy(self.dataset)
              
              if epoch10 == 0:
                  print('--> epoch=d, accuracy=,3f'  (epoch, accuracy))
              costs.append(J)
              accuracies.append(accuracy)
              
          print("training complete")
          print("final accuracy: .3f' (self.accuracy(self.dataset)))
          costs = list(map(lambda t: np.mean(t), [np.array(costs)[i-10:i+11] for i in range(1, len(costs)-10)]))
          accuracies = list(map(lambda t: np.mean(t), [np.array(accuracies)[i-10:i+11] for i in range(1, len(accuracy
                                                                                                             
          return (costs, accuracies)                                                                                              
                            
class Cat_Data:

    def __init__(self, relative_path='../../data/assignment1/', data_file_name='cat_data.pkl'):
        
        self.index = -1
        with open('%s%s'  (data_file_path, data_file_name), mode='rb') as f:
            cat_data = pkl.load(f)                                                                                                 
        self.samples = [(np.reshape(vector, vector.size), 1) for vector in cat_data['train']['cat'] + [(np.res
        random-shuffle(self.samples)
                                                                                                       
    def __iter__(self):
       
        return self

    def __next__(self):
        
        self.index += 1
        if self.index == len(self.samples):
            raise StopIteration
        return self.samples[self.index][0], self.samples[self.index][1]                                                                                                
                                                                                                        

    def _shuffle(self):
        
        random.shuffle(self.samples)                                                                                                

def main():

    data = Sonar_Data()
    model = Sonar_Model()  # specify the necessary arguments
    trainer = Sonar_Trainer(data, model)
    trainer.train() # experiment with learning rate and number of epochs
    model.save_model()


if __name__ == '__main__':

    main()
