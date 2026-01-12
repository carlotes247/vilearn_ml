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
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
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
    conf_matrices_dyads_all_f_combined_naive_bayes = [
        np.array([[10, 0], [2, 0]]),
        np.array([[9, 0], [2, 1]]),
        np.array([[3, 2], [1, 5]]),
        np.array([[9, 0], [2, 0]]),
        np.array([[7, 1], [1, 0]]),
        np.array([[4, 0], [4, 1]]),
        np.array([[0, 1], [1, 7]]),
        np.array([[0, 0], [2, 5]]),
    ]

    conf_matrices_dyads_aixvr_f_combined_qda = [
        np.array([[9, 1], [2, 0]]),
        np.array([[9, 0], [2, 1]]),
        np.array([[5, 0], [4, 2]]),
        np.array([[9, 0], [2, 0]]),
        np.array([[7, 1], [1, 0]]),
        np.array([[4, 0], [4, 1]]),
        np.array([[1, 0], [7, 1]]),
        np.array([[0, 0], [4, 2]]), 
    ]

    conf_matrices_triads_all_f_combined_qda = [
        np.array([[3, 2], [4, 4]]),
        np.array([[0, 8], [0, 3]]),
        np.array([[1, 3], [2, 5]]),
        np.array([[5, 4], [0, 2]]),
        np.array([[3, 3], [0, 4]]),
        np.array([[0, 3], [0, 5]]),
        np.array([[1, 4], [0, 3]]),
        np.array([[1, 2], [2, 3]]), 
        np.array([[5, 2], [0, 1]]),
        np.array([[0, 0], [1, 6]]),
        np.array([[0, 1], [2, 4]]),
        np.array([[0, 0], [3, 1]]),
    ]

    conf_matrices_triads_aixvr_f_combined_linear_svm_l1 = [
        np.array([[4, 1], [1, 7]]),
        np.array([[0, 8], [0, 3]]),
        np.array([[0, 4], [0, 7]]),
        np.array([[5, 4], [0, 2]]),
        np.array([[0, 6], [0, 4]]),
        np.array([[0, 3], [0, 5]]),
        np.array([[3, 2], [2, 1]]),
        np.array([[2, 1], [0, 5]]), 
        np.array([[0, 7], [0, 1]]),
        np.array([[0, 0], [1, 6]]),
        np.array([[0, 1], [0, 6]]),
        np.array([[0, 0], [3, 1]]),
    ]

    

    confusion_matrices = {
        # # Dyads
        # 'Best Dyads, All Features, Training Combined (Naive Bayes, 75%)': (sum(conf_matrices_dyads_all_f_combined_naive_bayes)),
        # 'Best Dyads, All Features, Training Specific (QDA, 77%)': [[42, 4], [14, 20]],
        # 'Best Dyads, AIxVR Features, Training Combined (QDA, 63%)': (sum(conf_matrices_dyads_aixvr_f_combined_qda)),
        # 'Best Dyads, AIxVR Features, Training Specific (Naive Bayes, 68%)': [[38, 8], [17, 17]], # why not knn? performs 2% better
        # Triads
        'Best Triads, All Features, Training Combined (QDA, 56%)': (sum(conf_matrices_triads_all_f_combined_qda)),
        'Best Triads, All Features, Training Specific (Linear SVM l1, 54%)': [[24, 27], [20, 35]],
        'Best Triads, AIxVR Features, Training Combined (Linear SVM l1, 57%)': (sum(conf_matrices_triads_aixvr_f_combined_linear_svm_l1)),
        'Best Triads, AIxVR Features, Training Specific (Linear SVM l1, 55%)': [[25, 26], [20, 35]],

        # # Separated by groups, original way
        # 'Best All_Groups, All Features (Naive Bayes, 63%)': [[65, 32],[34, 55]],
        # 'Best All_Groups, AIxVR Features (QDA, 61%)': [[64, 33],[36, 53]],
        # 'Best Dyads, All Features (QDA, 77%)': [[42, 4], [14, 20]],
        # 'Best Dyads, AIxVR Features (Knn, 70%)': [[34, 12],[13, 21]],
        # 'Best Triads, All Features (Linear SVM l2, 59%)': [[32,19],[27,28]],
        # 'Best Triads, AIxVR Features (Decision Tree, 57%)': [[28, 23],[21, 34]]
    }
    plotter = ConfusionMatrixPlotter()
    plotter.plot_confusion_matrices(confusion_matrices=confusion_matrices, labels_text=labels)
