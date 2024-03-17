import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt


with np.load("./toy-data.npz") as dataset:
    data = dataset['training_data']
    labels = dataset['training_labels']

w = np.array([-0.4528, -0.5190])
alpha = 0.1471

plt.scatter(data[:, 0], data[:, 1], c=labels)
# Plot the decision boundary
x = np.linspace(-5, 5, 100)
y = -(w[0] * x + alpha) / w[1]
plt.plot(x, y, 'k')
margin1 = -(w[0] * x + alpha - 1) / w[1]
margin2 = -(w[0] * x + alpha + 1) / w[1]
plt.plot(x, margin1, 'k--')
plt.plot(x, margin2, 'k--')
plt.show()

def partition(data, labels, size, rand_seed=30):
    if isinstance(size, float):
        size = int(len(data) * size)
    
    training_size = len(data) - size
    np.random.seed(rand_seed)
    i = np.random.permutation(len(data))
    data_training = data[i][:training_size]
    data_validation = data[i][training_size:]
    labels_training = labels[i][:training_size]
    labels_validation = labels[i][training_size:]
    return data_training, data_validation, labels_training, labels_validation

with np.load("./mnist-data.npz") as mnist_dataset:
    mnist_data = mnist_dataset['training_data']
    mnist_labels = mnist_dataset['training_labels']
    mnist_test = mnist_dataset['test_data']

with np.load("./spam-data.npz") as spam_dataset:
    spam_data = spam_dataset['training_data']
    spam_labels = spam_dataset['training_labels']
    spam_test = spam_dataset['test_data']

mnist_data_training, mnist_data_validation, mnist_labels_training, mnist_labels_validation = partition(mnist_data, mnist_labels, 10000)
spam_data_training, spam_data_validation, spam_labels_training, spam_labels_validation = partition(spam_data, spam_labels, 0.20)

def accuracy_score(y_true, y_pred):
    num_correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            num_correct += 1
    
    return num_correct / len(y_true)


def train(train_data, train_labels, val_data, val_labels, sizes, C=1.0):
    train_accuracies = []
    val_accuracies = []
    for size in sizes:
        model = svm.LinearSVC(C=C)
        model.fit(train_data[:size], train_labels[:size])

        training_accuracy = accuracy_score(train_labels[:size], model.predict(train_data[:size]))
        train_accuracies.append(training_accuracy)

        validation_accuracy = accuracy_score(val_labels, model.predict(val_data))
        val_accuracies.append(validation_accuracy)
    
    return train_accuracies, val_accuracies

def plot_accuracy(train_accuracies, val_accuracies, sizes, title):
    plt.title(title)
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Accuracy")
    plt.plot(sizes, train_accuracies, label="Training Accuracy")
    plt.plot(sizes, val_accuracies, label="Validation Accuracy")
    plt.legend()
    plt.show()

mnist_training_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
reshaped_data = mnist_data_training.reshape(50000, 784)
val_reshaped = mnist_data_validation.reshape(10000, 784)
mnist_train_accuracies, mnist_val_accuracies = train(reshaped_data, 
                                                     mnist_labels_training, 
                                                     val_reshaped, 
                                                     mnist_labels_validation, 
                                                     mnist_training_sizes)
plot_accuracy(mnist_train_accuracies, mnist_val_accuracies, mnist_training_sizes, "Result for MNIST Dataset")

spam_training_sizes = [100, 200, 500, 1000, 2000, 3337]
spam_train_accuracies, spam_val_accuracies = train(spam_data_training, 
                                                      spam_labels_training, 
                                                      spam_data_validation, 
                                                      spam_labels_validation, 
                                                      spam_training_sizes)
plot_accuracy(spam_train_accuracies, spam_val_accuracies, spam_training_sizes, "Result for Spam Dataset")


C = [10 ** -10, 10** -8, 10 ** -6, 10 ** -4, 10 ** -2, 1, 100, 10000]
hyper_mnist_val_accuracies = []
for c in C:
    hyper_mnist_train_accuracy, hyper_mnist_val_accuracy = train(reshaped_data, 
                                                     mnist_labels_training, 
                                                     val_reshaped, 
                                                     mnist_labels_validation, 
                                                     [10000], c)
    hyper_mnist_val_accuracies.append(hyper_mnist_val_accuracy)
    
for c, val_accuracy in zip(C, hyper_mnist_val_accuracies):
    print(f"C: {c}, val_accuracy: {val_accuracy}")


def cross_validation(train_data, train_labels, k, C_values):
    train_partitions, label_partitions = [], []
    for i in range(k):
        begin = i * (len(train_data) // k)
        if i == k-1:
            end = len(train_data)
        else:
            end = (i+1) * (len(train_data) // k)
        train_partitions.append(train_data[begin:end])
        label_partitions.append(train_labels[begin:end])
    val_accuracies = []

    for c in C_values:
        curr_val_accuracies = []
        for i in range(k):
            curr_train_data = np.concatenate(train_partitions[:i] + train_partitions[i+1:])
            curr_train_labels = np.concatenate(label_partitions[:i] + label_partitions[i+1:])
            curr_val_data = train_partitions[i]
            curr_val_labels = label_partitions[i]
            model = svm.LinearSVC(C=c)
            model.fit(curr_train_data, curr_train_labels)
            curr_val_accuracies.append(accuracy_score(curr_val_labels, model.predict(curr_val_data)))
        val_accuracies.append(np.mean(curr_val_accuracies))

    for c, val_accuracy in zip(C_values, val_accuracies):
        print(f"C: {c}, val_accuracy: {val_accuracy}")
                 
cross_validation(np.concatenate((spam_data_training, spam_data_validation)), np.concatenate((spam_labels_training, spam_labels_validation)), 5, C)

  

