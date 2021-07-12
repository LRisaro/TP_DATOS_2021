from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def plot_roc(_fpr, _tpr, x):
    roc_auc = auc(_fpr, _tpr)
    plt.figure(figsize=(15, 10))
    plt.plot(_fpr, _tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.scatter(_fpr, x)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

def getMetrics(y_test, y_pred):
    print('\033[1m' + "Accuracy" + '\033[0m', str(accuracy_score(y_test, y_pred)) + "\n", sep=': ')
        
    print('\033[1m' + "Precision" + '\033[0m', str(precision_score(y_test, y_pred)) + "\n", sep=': ')
    
    print('\033[1m' + "Recall" + '\033[0m', str(recall_score(y_test, y_pred, pos_label=0)) + "\n", sep=': ')
    
    print('\033[1m' + "F1 Score" + '\033[0m', str(f1_score(y_test, y_pred)) + "\n", sep=': ')
    
    print('\033[1m' + "Matriz de confusion: " + '\033[0m')
    print(str(confusion_matrix(y_test, y_pred)) + "\n")
    
    print('\033[1m' + "AUC-ROC: " + '\033[0m' + "\n")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plot_roc(fpr, tpr, thresholds)
    display(roc_auc_score(y_test, y_pred))
