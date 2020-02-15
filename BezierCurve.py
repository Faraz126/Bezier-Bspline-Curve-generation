import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import random
from mpl_toolkits import mplot3d
import math

class BezierCurve:
    def __init__(self, list_of_points):
        """
        initializer for the BezierCurve object
        :param list_of_points: the list of n-d points to interpolate through our bezier curve
        """

        self.list_of_points = []
        for i in list_of_points:
            if type(i) != np.array:
                self.list_of_points.append(np.array(i)) #converting to np array so that it is easier to deal with
            else:
                self.list_of_points.append(i)
        self.ax = None


    def bezier_basis(self, s, t, degree = 1):
        """
        returns a list of bazier basis values for a function of given degree.
        :param t: t-parameter
        :param s: s-parameter
        :param degree: degree of the function
        """
        if s == -1: #indicator that we are iterating over unit interval
            s = 0
        function_values = []
        for i in range(degree + 1):
            #conventional tensor product structure
            term = sp.comb(degree, i) * ((1 - s) ** (degree - i)) * (s ** i)
            for j in range(degree + 1):
                term2 = sp.comb(degree, j) * ((1 - t) ** (degree - j)) * (t ** j)
                function_values.append(term * term2)
        return function_values


    def bezier_basis_derivate(self, derivative = 1, translate = 0):

        degree  = len(self.list_of_points) - 1
        function_values = []
        if derivative == 1:
            output = self.get_basis_values(s_range = -1, t_range = np.arange(0, 1.01, 0.01), degree = degree - 1, func = self.bezier_basis)
            for j in range(len(output[0])):
                l = []
                for i in range(len(output)):
                    l.append( output[i][j] * (degree) * (self.list_of_points[i + 1] - self.list_of_points[i]))
                a = sum(l)
                function_values.append(a)

        elif derivative == 2:
            output = self.get_basis_values(s_range=-1, t_range=np.arange(0, 1.01, 0.01), degree= degree - 2,func=self.bezier_basis)
            for j in range(len(output[0])):
                l = []
                for i in range(len(output)):
                    l.append(output[i][j] * degree *  (degree - 1) * (self.list_of_points[i + 2] - 2 * self.list_of_points[i+1] + self.list_of_points[i]))
                a = sum(l)
                function_values.append(a)

        x = np.array([i[0] for i in function_values])
        y = np.array([i[1] for i in function_values])
        plt.ylabel('dy')
        plt.xlabel('t')
        #plt.plot(*(zip(*function_values)))
        #plt.plot(np.arange(translate + 0, translate + 1.01, 0.01), x)
        plt.plot(np.arange(translate + 0, translate + 1.01, 0.01), y)
        return

    def make_continuous(self, Q, c = 1):
        if c >= 0:
            Q.list_of_points[0] = self.list_of_points[-1]
        if c >= 1:
            Q.list_of_points[1] = 2 * Q.list_of_points[0] - self.list_of_points[-2]
        if c >= 2:
            Q.list_of_points[2] = self.list_of_points[-3] - (4 * self.list_of_points[-2]) + (4 *self.list_of_points[-1])

    def bezier_basis_interpolate(self, s, t, degree):
        """
        returns a set of basis functions which passes through all given points
        """
        functions = []
        matrix = []
        i = 0

        #to determine co_efficients. We are Basically, finding the Ax = b solution by constructing A using inputs and b as given t.
        if degree == 2:
            t_s = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            for output in t_s:
                matrix.append(function_co_efficients(degree, output))
                f = lambda x: ((matrix[i][0]) * x ** 2) + ((matrix[i][1]) * x) + (matrix[i][2])
                functions.append(f)
                i += 1
            # functions = [lambda x: 2*(x**2) - 3*x +1, lambda x: -4*(x**2) + 4*x, lambda x:2*(x**2) - x]
        if degree == 3:
            t_s = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            for output in t_s:
                matrix.append(function_co_efficients(degree, output))
                f = lambda x: ((matrix[i][0]) * x ** 3) + ((matrix[i][1]) * x ** 2) + (matrix[i][2] * x) + (
                matrix[i][3])
                functions.append(f)
                i += 1

        if degree == 4:
            t_s = [[1,0,0,0,0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
            for output in t_s:
                matrix.append(function_co_efficients(degree, output))
                f = lambda x: ((matrix[i][0]) * x ** 4) + ((matrix[i][1]) * x ** 3) + ((matrix[i][2]) * x ** 2) + (matrix[i][3] * x) + (matrix[i][4])
                functions.append(f)
                i += 1


        function_values = []
        if s == -1:  #for 2d case, applying t to every function
            i = 0
            for func in functions:
                function_values.append(func(t))
                i += 1
        else:
            #applying tensor product for given set of s and t. We fix s, and iterate over all functions for t.
            i = 0
            mid_func = []
            for func in functions:
                mid_func.append(func(s))
                i += 1

            for val in mid_func:
                i = 0
                for func in functions:
                    function_values.append(val * func(t))
                    i += 1

        return function_values





    def get_basis_values(self, s_range, t_range, degree, func ):
        """
        returns a list of basis values after applying t_range and s_range as inputs.
        s_range should be 0 if the parameterization relies only on one variable.
        :param s_range: range of second variable
        :param degree: range of first variable
        :param func: uses the function given as func to output values
        :return: a nested list of the form,
        [ [N_0(0), N_0(0.1)..., N_0(1)], [N_1(0), N_1(0.1)..., N_1(1)], ..]
        """

        if type(s_range) == int: #2d case
            s_range = [0]
        outputs = []
        for s in s_range:
            for t in t_range:
                outputs.append(func(s, t, degree = degree))
        outputs = np.array(outputs)
        # because each lists contain values of basis functions at given t, transposing gives us seperate lists for all basis functions
        outputs = np.transpose(outputs)
        outputs = [i for i in outputs if any(i)]  #filtering out all rows with all zeros
        return outputs

    def get_basis_values(self, s_range, t_range, degree, func ):
        """
        returns a list of basis values after applying t_range and s_range as inputs.
        s_range should be 0 if the parameterization relies only on one variable.
        :param s_range: range of second variable
        :param degree: range of first variable
        :param func: uses the function given as func to output values
        :return: a nested list of the form,
        [ [N_0(0), N_0(0.1)..., N_0(1)], [N_1(0), N_1(0.1)..., N_1(1)], ..]
        """

        if type(s_range) == int: #2d case
            s_range = [-1]
        outputs = []
        for s in s_range:
            for t in t_range:
                outputs.append(func(s, t, degree = degree))
        outputs = np.array(outputs)
        # because each lists contain values of basis functions at given t, transposing gives us seperate lists for all basis functions
        outputs = np.transpose(outputs)
        outputs = [i for i in outputs if any(i)]  #filtering out all rows with all zeros
        return outputs

    def plot_basis(self,no_of_variables, degree, func = bezier_basis, labels = [], title = ''):
        """
        the no of variables on which the paramerization is done.
        :param degree: degree of the functions
        plots the basis for the given no_of_variables, and according to the given degree.
        """

        t = np.arange(0, 1.01, 0.005)
        outputs = []
        if no_of_variables != 1:
            t = np.arange(0, 1.01, 0.05)
            s = np.arange(0, 1.01, 0.05)
            inputs = [[], []]
            for j in s:
                for i in t:
                    outputs.append(func(j, i, degree = degree))
                    inputs[0].append(j)
                    inputs[1].append(i)
            t = inputs[1]
            s = inputs[0]
        else:
            for i in t:
                outputs.append(func(-1, i, degree = degree))

        outputs = np.array(outputs)
        outputs = np.transpose(outputs)
        fig = plt.figure()
        if no_of_variables != 1:
            ax = plt.axes(projection='3d')
        for i in outputs:
            if (any(i)):
                if no_of_variables != 1:
                    ax.set_zlabel(labels[2])
                    ax.plot_trisurf(t, s, i)
                else:
                    plt.plot(t, i)

        plt.title(title)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])


    def plot_curve(self, passing_through_all_point = False, color = 'blue', ax = None, derivative = 0):
        """
        plots curve to ax, if ax is None, plots on default plt.
        If passing_through_all_point is True, it uses interpolating basis, else it uses bezier basis function.
        """
        n = len(self.list_of_points)
        if derivative == 0:
            if passing_through_all_point:
                func = self.bezier_basis_interpolate
            else:
                func = self.bezier_basis
        elif derivative == 1:
            func = self.bezier_basis_derivate

        if (len(self.list_of_points[0]) == 2):
            basis_values = np.transpose(
                self.get_basis_values(s_range = -1, t_range = np.arange(0, 1.01, 0.01), degree = n - 1, func = func)) #generating basis
            function_values = [sum([j[i] * self.list_of_points[i] for i in range(len(self.list_of_points))]) for j in basis_values]
            if passing_through_all_point:
                label = "Curve Passing Through all points"
            else:
                label = "Bezier Curve"
            plt.plot(*(zip(*function_values)), label = label)

        else:
            basis_values = np.transpose(
                self.get_basis_values(s_range = np.arange(0, 1.01, 0.05),t_range= np.arange(0, 1.01, 0.05), degree = int(math.sqrt(n))- 1 , func = func))
            function_values = [sum([j[i] * self.list_of_points[i] for i in range(len(self.list_of_points))]) for j in
                               basis_values] #summing values to plot

            """
            to produce smoother graphs
            """
            X = [i[0] for i in function_values]
            Y = [i[1] for i in function_values]
            Z = [i[2] for i in function_values]
            x = np.reshape(X, (int(math.sqrt(len(X))), int(math.sqrt(len(X)))))
            y = np.reshape(Y, (int(math.sqrt(len(X))), int(math.sqrt(len(X)))))
            z = np.reshape(Z, (int(math.sqrt(len(X))), int(math.sqrt(len(X)))))
            if ax == None:
                if self.ax == None:
                    self.ax = plt.axes(projection='3d')
                surf = self.ax.plot_surface(x, y, z, label = "Patch")
            else:
                surf = ax.plot_surface(x, y, z)
            surf._facecolors2d = surf._facecolors3d
            surf._edgecolors2d = surf._edgecolors3d

        #plt.title("Cubic Bezier Patch")
        plt.xlabel("x")
        plt.ylabel("y")

    def plot_points(self, ax = None):
        """
        plots its own set of points in scatter plot
        """
        if len(self.list_of_points[0]) == 2:
            plt.scatter(*zip(*self.list_of_points), color='red', label = "Control Points")
        else:
            if ax == None:
                if self.ax == None:
                    self.ax = plt.axes(projection = '3d')
                self.ax.scatter(*zip(*self.list_of_points), color = 'red', label = "Control Points")
            else:
                ax.scatter(*zip(*self.list_of_points), color = 'red', label = "Control Points")
        return

def function_co_efficients(degree, t):
    """
    to determine co_efficients of a function of degree d, given the right hand side t.
    """
    matrix = []
    t_values = [round(i/degree, 3) for i in range(degree + 1)]
    for i in t_values:
        row = [i**k for k in range(degree , -1, -1)]
        matrix.append(row)
    return np.linalg.solve(matrix , t)

