# TODO:use SVM with another group of parameters
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import GridSearchCV

# download and read mnist
mnist = fetch_mldata('MNIST original', data_home='./')

# 'mnist.data' is 70k x 784 array, each row represents the pixels from a 28x28=784 image
# 'mnist.target' is 70k x 1 array, each row represents the target class of the corresponding image
images = mnist.data
targets = mnist.target

# make the value of pixels from [0, 255] to [0, 1] for further process
X = mnist.data / 255.
Y = mnist.target


# split data to train and test (for faster calculation, just use 1/10 data)
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)


md = LinearSVC()
#调参数赋值，C可更改以获得更精确的模型
tuned_parameters = [{'dual': [False], 'C':[1, 0.1, 0.01, 0.0001],}]

#scores = ['precision', 'recall']

#gridsearchhcv函数调优
clf = GridSearchCV(md, tuned_parameters)
clf.fit(X_train, Y_train)
#best_params函数即找出最佳的C
print(clf.best_params_)
#best_estimator即找出最佳模型
best_md = clf.best_estimator_

train_accuracy = best_md.score(X_train, Y_train)
test_accuracy = best_md.score(X_test, Y_test)

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))
