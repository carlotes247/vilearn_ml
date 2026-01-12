import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import numpy.typing as npt

class ConfusionMatrixPlotter:
    def __init__(self) -> None:
        pass

    def plot_confusion_matrices(self, confusion_matrices: dict[str, list[list[int]]], labels_text:list[str]):
        """
        Plot a list of confusion matrices into the same figure.
        
        Parameters:
        confusion_matrices: A list of 2D lists representing the confusion matrices.
        title: A string representing the title for the plot.
        """
        # Create a figure and axis
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15,10))
        x=0
        y=0
        # Iterate through the confusion matrices and plot each one
        for title, matrix in confusion_matrices.items():
            matrix = np.array(matrix)
            matrix = np.round(matrix/matrix.sum()*100, 1)
            # Create a ConfusionMatrixDisplay for the current matrix
            disp = metrics.ConfusionMatrixDisplay(
                confusion_matrix=matrix,
                display_labels=labels_text, 
                # cmap=plt.cm.Blues,
                # normalize=normalize
            )
            ax[x,y].title.set_text(title)
            disp.plot(ax=ax[x,y])
            # move to next axis in figure
            x = x+1
            if x == 2:
                y = y+1
                x = 0
        
        # Adjust the layout to fit the last matrix
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    labels = ["low", "high"]
    confusion_matrices = {
        'Best All_Groups, All Features (Naive Bayes, 63%)': [[65, 32],[34, 55]],
        'Best All_Groups, AIxVR Features (QDA, 61%)': [[64, 33],[36, 53]],
        'Best Dyads, All Features (QDA, 77%)': [[42, 4], [14, 20]],
        'Best Dyads, AIxVR Features (Knn, 70%)': [[34, 12],[13, 21]],
        'Best Triads, All Features (Linear SVM l2, 59%)': [[32,19],[27,28]],
        'Best Triads, AIxVR Features (Decision Tree, 57%)': [[28, 23],[21, 34]]
    }
    plotter = ConfusionMatrixPlotter()
    plotter.plot_confusion_matrices(confusion_matrices=confusion_matrices, labels_text=labels)
