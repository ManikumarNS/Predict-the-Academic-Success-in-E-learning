import numpy as np
import pandas as pd
from Save_load import *
import matplotlib.pyplot as plt
import seaborn as sns

def bar_plot(label, value, metric):

    fig = plt.figure()

    # creating the bar plot
    colors = ['maroon', 'green', 'red', 'purple']
    plt.bar(label, value, color=colors,
            width=0.4)

    plt.xlabel("Method")
    plt.ylabel(metric)
    plt.savefig('./Results3/'+metric+'.png', dpi=400)
    plt.show(block=False)


def evaluate(X_train, y_train, X_test, y_test,soln):
    met_train, met_test = TLO(X_train, y_train, X_test, y_test, soln)
    return np.array(met_train), np.array(met_test)

def plot_res():

    X_train = load('X_train')
    X_test = load('X_test')
    y_train = load('y_train')
    y_test = load('y_test')

    met1 = load("metrices")
    met2 = load("metrices1")
    met3 = load("metrices2")

    methods = ["accuracy", "precision", "sensitivity", "specificity", "f_measure", "mcc", "npv", "fpr", "fnr"]

    # Data
    labels = ['60%', '70%', '80%']
    for j in range(len(methods)):
        dd = np.array([met2[j], met3[j], met1[j]]).T

        # Create positions for the bars on the x-axis
        x = np.arange(len(labels))

        # Set bar width
        width = 0.15

        # Create the bar plots for each dataset
        plt.bar(x - 1.5 * width, dd[0], width=width, label='CNN')
        plt.bar(x - 0.5 * width, dd[1], width=width, label='QDNN')
        plt.bar(x + 0.5 * width, dd[2], width=width, label='Stacked_Auto_Encoder')
        plt.bar(x + 1.5 * width, dd[3], width=width, label='Proposed')

        # Set labels and title
        plt.xlabel('Learning Rate')
        plt.ylabel(methods[j])
        plt.xticks(x, labels)

        # Add a legend
        plt.legend()
        plt.savefig('./Results3/' + methods[j] + '.png', dpi=400)
        # Show the plot
        plt.show()


    metrices = [met2,met3,met1]
    method1 = ["CNN", "QDNN", "Stacked_Auto_Encoder", "Proposed"]
    metrices_plot = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F-Measure', 'MCC', 'NPV', 'FPR', 'FNR']
    inte =[60,70,80]
    for k in range(3):
        print('Testing Metrices '+ str(inte[k]) +"% of learning rate")
        tab = pd.DataFrame(metrices[k], index=metrices_plot, columns=method1)
        print(tab)
plot_res()