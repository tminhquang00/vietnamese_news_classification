from sklearn import metrics
import pickle
import pandas as pd

def evaluate_model(model, X_test, Y_test):
    '''
    Initial evaluate model on test set
    '''
    # Predict topic indices in x_test
    y_test_hat = model.predict(X_test)

    # Estimate the testing accuracy
    test_accuracy = metrics.accuracy_score(Y_test,y_test_hat)*100
    print("Accuracy for our testing dataset with tuning is : {:.2f}%".format(test_accuracy) )
    return test_accuracy


# This function is to plot the confusion matrix
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    

class Classifier(object):
    def __init__(self, X_train = None, Y_train = None, X_test = None, Y_test = None, classifier=None):
        self.X_train = X_train
        self.X_test = X_test  
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.classifier = classifier

    def training(self):
        self.classifier.fit(self.X_train, self.Y_train)
        self.__training_result()


    def __training_result(self): 
        y_true, y_pred = self.Y_test, self.classifier.predict(self.X_test)
        label_dict = dict(
                    zip(
                        sorted(set(self.Y_test), key=self.Y_test.index), 
                        range(len(self.Y_test))
                        )
                )
        print(metrics.classification_report(y_true, y_pred))
        confus_mat = metrics.confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(confus_mat, title='Confusion matrix')
