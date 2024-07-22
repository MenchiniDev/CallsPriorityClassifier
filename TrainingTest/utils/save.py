from matplotlib import pyplot as plt


def saveResults(classifier, confusion_matrix, classification_report, type):
    
    path = f'../results/{classifier}_{type}.png'
    plt.savefig(path)
        