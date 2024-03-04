import numpy as np
import pandas as pd
from Save_load import *
import matplotlib.pyplot as plt
import seaborn as sns


def plot_res():

    metrices=load('meti')

    method = ['ANN', 'CNN', 'RNN', 'Proposed']
    metrices_plot=['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F-Measure', 'MCC', 'NPV', 'FPR', 'FNR']

    plt.figure(figsize=(10, 6))

    # Plot the first set of data
    plt.plot(method, metrices[0], marker='o',label=metrices_plot[0])
    plt.plot(method, metrices[1], marker='o',label=metrices_plot[1])

    plt.xlabel('Models')
    plt.ylabel('Values')
    plt.title('Comparison of Models')
    plt.legend()
    plt.grid(True)  # Optional: Add a grid to the plot
    plt.tight_layout()  # Optional: Adjust layout to prevent labels from being cut off
    plt.savefig('./result4/' + "plot1" + '.png', dpi=400)
    plt.show()



    # Table
    print('Testing Metrices')
    tab=pd.DataFrame(metrices, index=metrices_plot, columns=method)
    print(tab)
    #tab.to_csv('./result4/' + "table1" + '.csv')


def plot_res1():
    metrices = load('meti')

    method = ['ANN', 'CNN', 'RNN', 'Proposed']
    metrices_plot = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F-Measure', 'MCC', 'NPV', 'FPR', 'FNR']

    plt.figure(figsize=(10, 6))

    # Plot the first set of data
    plt.plot(method, metrices[2], marker='o', label=metrices_plot[2])
    plt.plot(method, metrices[3], marker='o', label=metrices_plot[3])


    plt.xlabel('Models')
    plt.ylabel('Values')
    plt.title('Comparison of Models')
    plt.legend()
    plt.grid(True)  # Optional: Add a grid to the plot
    plt.tight_layout()  # Optional: Adjust layout to prevent labels from being cut off
    plt.savefig('./result4/' + "plot2" + '.png', dpi=400)
    plt.show()


def plot_res2():
    metrices = load('meti')

    method = ['ANN', 'CNN', 'RNN', 'Proposed']
    metrices_plot = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F-Measure', 'MCC', 'NPV', 'FPR', 'FNR']

    plt.figure(figsize=(10, 6))

    # Plot the first set of data
    plt.plot(method, metrices[4], marker='o', label=metrices_plot[4])
    plt.plot(method, metrices[5], marker='o', label=metrices_plot[5])
    plt.plot(method, metrices[6], marker='o', label=metrices_plot[6])

    plt.xlabel('Models')
    plt.ylabel('Values')
    plt.title('Comparison of Models')
    plt.legend()
    plt.grid(True)  # Optional: Add a grid to the plot
    plt.tight_layout()  # Optional: Adjust layout to prevent labels from being cut off
    plt.savefig('./result4/' + "plot3" + '.png', dpi=400)
    plt.show()

def plot_res3():

    metrices = load('meti')

    method = ['ANN', 'CNN', 'RNN', 'Proposed']
    metrices_plot = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F-Measure', 'MCC', 'NPV', 'FPR', 'FNR']

    plt.figure(figsize=(10, 6))

    # Plot the first set of data
    plt.plot(method, metrices[7], marker='o', label=metrices_plot[7])
    plt.plot(method, metrices[8], marker='o', label=metrices_plot[8])

    plt.xlabel('Models')
    plt.ylabel('Values')
    plt.title('Comparison of Models')
    plt.legend()
    plt.grid(True)  # Optional: Add a grid to the plot
    plt.tight_layout()  # Optional: Adjust layout to prevent labels from being cut off
    plt.savefig('./result4/' + "plot4" + '.png', dpi=400)
    plt.show()

plot_res()
plot_res1()
plot_res2()
plot_res3()






