import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

## PART 1 ##

with np.load("./mnist-data-hw3.npz") as mnist_dataset:
    mnist_data = mnist_dataset['training_data']
    mnist_labels = mnist_dataset['training_labels']
    mnist_test = mnist_dataset['test_data']

mnist_data = mnist_data.reshape(60000, 784)
mnist_labels = mnist_labels.reshape(60000, 1)
mnist_test = mnist_test.reshape(10000, 784)

norms = np.sqrt(np.sum(mnist_data**2, axis=1))
normalized_data = mnist_data / norms.reshape(-1, 1)

test_norms = np.sqrt(np.sum(mnist_test**2, axis=1))
normalized_test_data = mnist_test / test_norms.reshape(-1, 1)

mnist_distributions = {}
labels = np.unique(mnist_labels)
for label in labels:
    i = (mnist_labels == label).flatten()
    class_data = normalized_data[i]
    mean = np.mean(class_data, axis=0)
    sigma = np.cov(class_data.T)
    mnist_distributions[label] = [mean, sigma]

## PART 2 ##
    
i = (mnist_labels == labels[0]).flatten()
class_0_data = normalized_data[i]
sigma = np.abs(np.cov(class_0_data.T))
plt.imshow(sigma, cmap='binary_r')
plt.colorbar()
plt.show()

## PART 3 ##

# Part A #
def partition(data, labels, size, rand_seed=42):
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

mnist_data_training, mnist_data_validation, mnist_labels_training, mnist_labels_validation = partition(normalized_data, mnist_labels, 10000)
mnist_training_sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]

def train(training_data, training_labels):
    labels = np.unique(training_labels)
    num_images, num_features = training_data.shape
    avg_cov = np.zeros((num_features, num_features))

    class_distributions = []
    for label in labels:
        i = np.array(training_labels == label).flatten()
        class_data = training_data[i]

        prior_prob = class_data.shape[0] / num_images
        class_mean = class_data.mean(axis=0)
        centered = class_data - class_mean
        class_cov = np.cov(centered.T)
        avg_cov += class_cov * class_data.shape[0]
        class_cov /= class_data.shape[0]
        class_distributions.append([prior_prob, class_mean, class_cov])
        
    avg_cov /= num_images
    return class_distributions, avg_cov

def predict_lda(val_data, class_distributions, avg_cov):
    pred = []
    for prior_prob, mean, _ in class_distributions:
        log_likelihoods = scp.stats.multivariate_normal.logpdf(val_data, mean=mean, cov=avg_cov) + np.log(prior_prob)
        pred.append(log_likelihoods)
    pred = np.array(pred)
    return np.argmax(pred, axis=0).reshape(-1, 1)

digits = np.unique(mnist_labels_training)
errors_lda = []
errors_lda_per_digit = {digit: [] for digit in digits}
for size in mnist_training_sizes: 
    class_distributions, avg_cov = train(mnist_data_training[:size], mnist_labels_training[:size])
    avg_cov = avg_cov + (10**-6) * np.eye(avg_cov.shape[0])
    val_predicted = predict_lda(mnist_data_validation, class_distributions, avg_cov)
    val_accuracy = np.sum(mnist_labels_validation == val_predicted) / mnist_labels_validation.shape[0]
    errors_lda.append(1-val_accuracy)

    for digit in digits:
        i = np.array(mnist_labels_validation == digit).flatten()
        actual_digit_labels, predicted_digit_labels = mnist_labels_validation[i], val_predicted[i]
        digit_accuracy = np.sum(actual_digit_labels == predicted_digit_labels) / actual_digit_labels.shape[0]
        errors_lda_per_digit[digit].append(1 - digit_accuracy)

plt.figure(figsize=(7, 7))
plt.plot(mnist_training_sizes, errors_lda)
plt.xlabel("Training Size")
plt.ylabel("Error Rate")
plt.title("LDA Error Rate vs. Training Size")
plt.ylim((0, 1))
plt.show()

# Part B #
def predict_qda(val_data, class_distributions):
    pred = []
    for prior_prob, mean, cov in class_distributions:
        cov = cov + (10**-6) * np.eye(cov.shape[0])
        log_likelihoods = scp.stats.multivariate_normal.logpdf(val_data, mean=mean, cov=cov) + np.log(prior_prob)
        pred.append(log_likelihoods)
    pred = np.array(pred)
    return np.argmax(pred, axis=0).reshape(-1, 1)

digits = np.unique(mnist_labels_training)
errors_qda = []
errors_qda_per_digit = {digit: [] for digit in digits}
for size in mnist_training_sizes: 
    class_distributions, _ = train(mnist_data_training[:size], mnist_labels_training[:size])
    val_predicted = predict_qda(mnist_data_validation, class_distributions)
    val_accuracy = np.sum(mnist_labels_validation == val_predicted) / mnist_labels_validation.shape[0]
    errors_qda.append(1-val_accuracy)

    for digit in digits:
        i = np.array(mnist_labels_validation == digit).flatten()
        actual_digit_labels, predicted_digit_labels = mnist_labels_validation[i], val_predicted[i]
        digit_accuracy = np.sum(actual_digit_labels == predicted_digit_labels) / actual_digit_labels.shape[0]
        errors_qda_per_digit[digit].append(1 - digit_accuracy)

plt.figure(figsize=(7, 7))
plt.plot(mnist_training_sizes, errors_qda)
plt.xlabel("Training Size")
plt.ylabel("Error Rate")
plt.title("QDA Error Rate vs. Training Size")
plt.ylim((0, 1))
plt.show()

# Part D #
plt.figure(figsize=(7, 7))
for digit in digits:
    plt.plot(mnist_training_sizes, errors_lda_per_digit[digit], label=f'Digit {digit}')
plt.xlabel("Training Size")
plt.ylabel("Error Rate")
plt.title("LDA Error Rate vs. Training Size Per Digit")
plt.legend()
plt.ylim((0, 1))
plt.show()

plt.figure(figsize=(7, 7))
for digit in digits:
    plt.plot(mnist_training_sizes, errors_qda_per_digit[digit], label=f'Digit {digit}')
plt.xlabel("Training Size")
plt.ylabel("Error Rate")
plt.title("QDA Error Rate vs. Training Size Per Digit")
plt.legend()
plt.ylim((0, 1))
plt.show()


## Part 5 ## 
with np.load("./spam-data-hw3.npz") as spam_dataset:
    spam_data = spam_dataset['training_data']
    spam_labels = spam_dataset['training_labels']
    spam_test = spam_dataset['test_data']

spam_labels = spam_labels.reshape(4171, 1)

norms = np.sqrt(np.sum(spam_data**2, axis=1)) + 1e-10
normalized_spam_data = spam_data / norms.reshape(-1, 1)

test_norms = np.sqrt(np.sum(spam_test**2, axis=1)) + 1e-10
normalized_spam_test_data = spam_test / test_norms.reshape(-1, 1)

spam_training_data, spam_validation_data, spam_training_labels, spam_validation_labels = partition(normalized_spam_data, spam_labels, 0.2, 1150)
