
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# load mini training data and labels

## NEED TO CHANGE PATH TO WHEREVER YOUR SAVED .npy FILES ARE
path = '/Users/tonynewmark/PycharmProjects/MachineLearningHomework1/Homework1Problem1Files/'

mini_train = np.load(path+'knn_minitrain.npy')
mini_train_label = np.load(path+'knn_minitrain_label.npy')


# randomly generate test data
mini_test = np.random.randint(20, size=20)
mini_test = mini_test.reshape(10, 2)

# Define knn classifier
def kNNClassify(newInput, dataSet, labels, k):
    result = []

    # run for each test in the randomized test data
    for test in newInput:
        distances = []
        voter = [0,0,0,0]
        # calculate the distance of each train data from current test data
        for i in dataSet:
            squared_diff_X = np.square(test[0]-i[0])
            squared_diff_Y = np.square(test[1]-i[1])
            distance = np.sqrt(squared_diff_Y+squared_diff_X)
            distances.append(distance)
        # distances array is completed

        # finds k smallest indexes
        k_smallest_indexes = np.argsort(distances)[:k]

        # votes on which Label is Most Frequent (Winner)
        for i in range(len(k_smallest_indexes)):
            if labels[k_smallest_indexes[i]] == 0:
                voter[0] += 1
            elif labels[k_smallest_indexes[i]] == 1:
                voter[1] += 1
            elif labels[k_smallest_indexes[i]] == 2:
                voter[2] += 1
            else:
                voter[3] += 1
        winning_label = np.argmax(voter)
        result.append(winning_label)

    return result

outputlabels = kNNClassify(mini_test, mini_train, mini_train_label, 10)

print ('random test points are:', mini_test)
print ('knn classfied labels for test:', outputlabels)

# plot train data and classfied test data
train_x = mini_train[:, 0]
train_y = mini_train[:, 1]
fig = plt.figure()
plt.scatter(train_x[np.where(mini_train_label == 0)], train_y[np.where(mini_train_label == 0)], color='red')
plt.scatter(train_x[np.where(mini_train_label == 1)], train_y[np.where(mini_train_label == 1)], color='blue')
plt.scatter(train_x[np.where(mini_train_label == 2)], train_y[np.where(mini_train_label == 2)], color='yellow')
plt.scatter(train_x[np.where(mini_train_label == 3)], train_y[np.where(mini_train_label == 3)], color='black')

test_x = mini_test[:, 0]
test_y = mini_test[:, 1]
outputlabels = np.array(outputlabels)
plt.scatter(test_x[np.where(outputlabels == 0)], test_y[np.where(outputlabels == 0)], marker='^', color='red')
plt.scatter(test_x[np.where(outputlabels == 1)], test_y[np.where(outputlabels == 1)], marker='^', color='blue')
plt.scatter(test_x[np.where(outputlabels == 2)], test_y[np.where(outputlabels == 2)], marker='^', color='yellow')
plt.scatter(test_x[np.where(outputlabels == 3)], test_y[np.where(outputlabels == 3)], marker='^', color='black')

# save diagram as png file
plt.savefig("miniknn.png")
