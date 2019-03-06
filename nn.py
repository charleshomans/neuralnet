import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# loads the data in a numpy matrix
train_data = pd.read_csv("TrainDigitX.csv.gz", sep=",")
train_data = train_data.values
#test_data = pd.read_csv("TrainDigitX2.csv.gz", sep=",")
#test_data = test_data.values

train_labels = np.loadtxt("TrainDigitY.csv")[1:]
#test_labels = np.loadtxt("TestDigitY.csv")
N_l = 25
L = 3
loss = 0
learning_rate = 0.1

layers = []

layers.append(np.zeros(785))
for i in np.arange(0, L-1):
    layers.append(np.zeros(N_l))
layers.append(np.zeros(10))

weights = []

errors = []


#make our initial weight matrices, we'll
#update during back pass
weights.append(np.random.normal(0, 0.1, (N_l, 785)))
errors.append(np.zeros(N_l))
for i in np.arange(1, L-1):
    weights.append(np.random.normal(0, 0.1, (N_l, N_l)))
    errors.append(np.zeros(N_l))
weights.append(np.random.normal(0, 0.1, (10, N_l)))
errors.append(np.zeros(10))

#for i, weight in enumerate(weights):
#    weights[i] = weight/weight.sum(axis=1)[:,None]

def sigma(x):
    return (1+np.exp(-x)) ** -1

def sigma_derivative(x):
    return sigma(x) * (1-sigma(x))

def loss_log_softmax(vec, true_vec):
    sm = softmax(vec)
    lls = -1*np.log(np.sum(sm * true_vec))
    #this is the CES as defined
    return lls

def softmax(vec):
    shifted_vec = vec - np.max(vec)
    shifted_exp_vec = np.exp(shifted_vec)
    sm = shifted_exp_vec/shifted_exp_vec.sum()
    return sm

def next_activation_vec(weight, prev_vector):
    weighted_vec = np.dot(weight, prev_vector)
    output = np.array([sigma(component) for component in weighted_vec])
    return output


def forward_pass(data):
    #plt.gray()
    #plt.imshow(data.reshape(28,28))
    layers[0] = np.append(data, 1)
    #print(layers[0])
    for i in np.arange(1, L+1):
            layers[i] = next_activation_vec(weights[i-1], layers[i-1])
            #print(len(layers[i]))
    loss = loss_log_softmax(layers[L], true_label)
    #plt.imshow(layers[3].reshape(5, 2))
    #plt.show()
    #print(loss)
    #print(np.sum(loss))

def backwards_pass():
    #errors[0] = lls[next_activation_vec()]
    errors[L-1] = softmax(layers[L]) - true_label
    #errors = no.gradient(loss1)
    for i in np.arange(L-2, -1, -1):
            #print(i)
            weighted_error = np.dot(np.transpose(weights[i+1]), errors[i+1])
            derivative = sigma_derivative(np.dot(weights[i], layers[i]))
            errors[i] = np.multiply(derivative, weighted_error)

def update_weights():
    for layer, weight_mat in enumerate(weights):
        for t in np.arange(0, len(weight_mat)):
            for s in np.arange(0, len(weight_mat[t])):
                derivative = errors[layer][t] * layers[layer][s]
                weight_mat[t][s] = weight_mat[t][s] - learning_rate*derivative
permutation = np.random.shuffle(np.arange(0, len(train_data)))
train_data = train_data[:,permutation,:]
train_labels = train_labels[:,permutation]
print(train_labels.shape)
print(train_data.shape)
for i, example in enumerate(train_data[:3000]):
        #plt.gray()
        #plt.imshow(example.reshape(28,28))
        #Plt.show()

        true_label = int(train_labels[i])
        vec = np.zeros(10)
        vec[true_label] = 1
        true_label = vec

        forward_pass(example)
        backwards_pass()
        update_weights()
        print("iteration: " + str(i))
        print("We guess the digit is " + str(np.argmax(layers[L])))
        print("The true digit is " + str(int(train_labels[i])))
        print("We compute -log(softmax dot true label): " + str(loss_log_softmax(layers[L], true_label)))

#for i, test_example in enumerate(test_data)
        #true_label = int(train_labels[i+1])
        #vec = np.zeros(10)
        #vec[true_label] = 1
        #true_label = vec
        #forward_pass(example)
