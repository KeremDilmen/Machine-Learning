import numpy as np
import scipy as scp
import matplotlib.pyplot as plt

# Load Data
wine_dataset = scp.io.loadmat('data.mat')
wine_training = wine_dataset['X']
wine_classes = wine_dataset['y']
wine_testing = wine_dataset['X_test']

wine_classes = wine_classes.reshape((wine_classes.shape[0], ))

# Append ones to training and testing sets
wine_training = np.append(wine_training, np.ones((wine_training.shape[0], 1)), axis=1)
wine_testing = np.append(wine_testing, np.ones((wine_testing.shape[0], 1)), axis=1)

# Partition Data
def partition(data, classes, size, rand_seed=42):
    if isinstance(size, float):
        size = int(data.shape[0] * size)
    
    training_size = data.shape[0] - size
    np.random.seed(rand_seed)
    i = np.random.permutation(data.shape[0])
    data_training = data[i][:training_size]
    data_validation = data[i][training_size:]
    classes_training = classes[i][:training_size]
    classes_validation = classes[i][training_size:]
    return data_training, data_validation, classes_training, classes_validation

train_data, val_data, train_classes, val_classes = partition(wine_training, wine_classes, 0.2)

# Normalize Data
for i in range(train_data.shape[1]-1):
    train_feature_mean = np.mean(train_data[:, i])
    train_feature_std = np.std(train_data[:, i])
    train_data[:, i] = (train_data[:, i] - train_feature_mean) / train_feature_std

    val_feature_mean = np.mean(val_data[:, i])
    val_feature_std = np.std(val_data[:, i])
    val_data[:, i] = (val_data[:, i] - val_feature_mean) / val_feature_std

# Choose Parameters
learn_rate = 1e-3
reg_const = 0.1
iter_num = 10000

# Initialize Weight, Cost, s Vectors
w = np.zeros(train_data.shape[1])
cost = np.zeros(iter_num+1)
s = scp.special.expit(train_data.dot(w))
cost[0] = -train_classes.dot(np.log(s)) - (1-train_classes).dot(np.log(1-s)) + (reg_const/2) * np.sum(w**2)

# Run Batch Gradient Descent
for i in range(1, iter_num+1):
    gradient = train_data.T.dot(s - train_classes) + reg_const * w
    w = w - learn_rate * gradient
    s = scp.special.expit(train_data.dot(w))
    cost[i] = -train_classes.dot(np.log(s)) - (1-train_classes).dot(np.log(1-s)) + (reg_const/2) * np.sum(w**2)

# Plot Cost versus Iterations Graph
plt.plot(range(iter_num+1), cost)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.title('Cost vs. Number of Iterations Graph for Batch Gradient Descent')
plt.show()

# Print Validation Accuracy
val_s=scp.special.expit(val_data.dot(w))
predicted_classes = (val_s > 0.5).astype(int)
val_accuracy = np.sum(val_classes == predicted_classes) / val_classes.shape[0]
print(f'Batch Gradient Descent Validation Accuracy: {val_accuracy}')

# Choose Parameters
learn_rate = 1e-2
reg_const = 0.1
iter_num = 10000

# Initialize Weight, Cost, s Vectors
w = np.zeros(train_data.shape[1])
cost = np.zeros(iter_num+1)
s = scp.special.expit(train_data.dot(w))
cost[0] = -train_classes.dot(np.log(s)) - (1-train_classes).dot(np.log(1-s)) + (reg_const/2) * np.sum(w**2)

# Run Stochastic Gradient Descent
index = 0
random_seed = 0
curr_training = train_data
curr_classes = train_classes
for i in range(1, iter_num+1):
    if index == curr_training.shape[0]:
        np.random.seed(random_seed)
        shuffle_index = np.random.permutation(curr_training.shape[0])
        curr_training = curr_training[shuffle_index]
        curr_classes = curr_classes[shuffle_index]
        random_seed += 1
        index = 0
    
    gradient = (s[index] - curr_classes[index]) * curr_training[index, :] + reg_const * w
    w = w - learn_rate * gradient
    s = scp.special.expit(curr_training.dot(w))
    cost[i] = -curr_classes.dot(np.log(s)) - (1-curr_classes).dot(np.log(1-s)) + (reg_const/2) * np.sum(w**2)
    index += 1


# Plot Cost versus Iterations Graph
plt.plot(range(iter_num+1), cost)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.title('Cost vs. Number of Iterations Graph for Stochastic Gradient Descent\n(Constant Learning Rate)')
plt.show()

# Print Validation Accuracy
val_s=scp.special.expit(val_data.dot(w))
predicted_classes = (val_s > 0.5).astype(int)
val_accuracy = np.sum(val_classes == predicted_classes) / val_classes.shape[0]
print(f'SGD (Constant Rate) Validation Accuracy: {val_accuracy}')

# Choose Parameters
delta = 1
reg_const = 0.1
iter_num = 10000

# Initialize Weight, Cost, s Vectors
w = np.zeros(train_data.shape[1])
cost = np.zeros(iter_num+1)
s = scp.special.expit(train_data.dot(w))
cost[0] = -train_classes.dot(np.log(s)) - (1-train_classes).dot(np.log(1-s)) + (reg_const/2) * np.sum(w**2)

# Run Stochastic Gradient Descent
index = 0
random_seed = 0
curr_training = train_data
curr_classes = train_classes
for i in range(1, iter_num+1):
    if index == curr_training.shape[0]:
        np.random.seed(random_seed)
        shuffle_index = np.random.permutation(curr_training.shape[0])
        curr_training = curr_training[shuffle_index]
        curr_classes = curr_classes[shuffle_index]
        random_seed += 1
        index = 0
    
    learn_rate = delta / i
    gradient = (s[index] - curr_classes[index]) * curr_training[index, :] + reg_const * w
    w = w - learn_rate * gradient
    s = scp.special.expit(curr_training.dot(w))
    cost[i] = -curr_classes.dot(np.log(s)) - (1-curr_classes).dot(np.log(1-s)) + (reg_const/2) * np.sum(w**2)
    index += 1

# Plot Cost versus Iterations Graph
plt.plot(range(iter_num+1), cost)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.title('Cost vs. Number of Iterations Graph for Stochastic Gradient Descent\n(Changing Learning Rate)')
plt.show()

# Print Validation Accuracy
val_s=scp.special.expit(val_data.dot(w))
predicted_classes = (val_s > 0.5).astype(int)
val_accuracy = np.sum(val_classes == predicted_classes) / val_classes.shape[0]
print(f'SGD (Changing Rate) Validation Accuracy: {val_accuracy}')

# Normalize Data
for i in range(wine_training.shape[1]-1):
    train_feature_mean = np.mean(wine_training[:, i])
    train_feature_std = np.std(wine_training[:, i])
    wine_training[:, i] = (wine_training[:, i] - train_feature_mean) / train_feature_std

    test_feature_mean = np.mean(wine_testing[:, i])
    test_feature_std = np.std(wine_testing[:, i])
    wine_testing[:, i] = (wine_testing[:, i] - test_feature_mean) / test_feature_std

# Choose Parameters
learn_rate = 1e-3
reg_const = 0.1
iter_num = 10000

# Initialize Weight, Cost, s Vectors
w = np.zeros(wine_training.shape[1])
cost = np.zeros(iter_num+1)
s = scp.special.expit(wine_training.dot(w))
cost[0] = -wine_classes.dot(np.log(s)) - (1-wine_classes).dot(np.log(1-s)) + (reg_const/2) * np.sum(w**2)

# Run Batch Gradient Descent
for i in range(1, iter_num+1):
    gradient = wine_training.T.dot(s - wine_classes) + reg_const * w
    w = w - learn_rate * gradient
    s = scp.special.expit(wine_training.dot(w))
    cost[i] = -wine_classes.dot(np.log(s)) - (1-wine_classes).dot(np.log(1-s)) + (reg_const/2) * np.sum(w**2)

test_s=scp.special.expit(wine_testing.dot(w))
predicted_classes = (test_s > 0.5).astype(int)


