import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import random
from mpl_toolkits import mplot3d


class BsplineCurve:
    def __init__(self, list_of_points):
        """
        list of points to interpolate
        """
        self.list_of_points = [np.array(i) for i in list_of_points if type(i) != np.ndarray]

    def get_basis_values(self, t_range, k):
        """
        :param t_range: t_values to iterate over
        :param k: the order
        :return: values of the bspline basis functions over the given t and order
        """
        length = 1/(2*k) #for variable legnth
        knot_vector = np.arange(0, 1.01, 0.1) #step is 0.1 because we want to have a knot vector length 10
        basis_at_t = []

        for t in t_range:
            function_output = []
            for i in range(len(knot_vector) - k):
                function_output.append(self.N(i,k,t, knot_vector))
            basis_at_t.append((function_output))

        outputs = np.array(basis_at_t)
        #because each lists contain values of basis functions at given t, transposing gives us seperate lists for all basis functions
        outputs = np.transpose(outputs)
        outputs = [i for i in outputs if any(i)] #excluding basis which all zero values
        return outputs

    def N(self, i, k, t, knot):
        """
        based on the textbook documentation of the N(i,k) function for a given t
        """
        if k == 1:
            if knot[i] <= t < knot[i + 1]:
                return 1
            return 0
        else:
            return (((t - knot[i]) / (knot[i + k - 1] - knot[i])) * self.N(i, k - 1, t, knot)) + (
                        ((knot[i + k] - t) / (knot[i + k] - knot[i + 1])) * self.N(i + 1, k - 1, t, knot))

    def plot_basis(self, k, labels = ['', ''], title = ''):
        """
        :param k: order
        :param labels: labels of x,y axis
        :param title: title of the graph

        plots the basis on the plt, for the given values, over the range 0 to 1 (unit interval)
        """
        STEP = 0.001
        t = np.arange(0, 1, STEP)
        basis_values = self.get_basis_values(t, k)
        for i in basis_values:
            plt.plot(t, i)
        plt.title(title)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
