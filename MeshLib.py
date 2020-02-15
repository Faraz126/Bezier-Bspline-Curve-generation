import math
import numpy as np
import statistics
import time
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from BezierCurve import *

class Vector:
    def __init__(self, co_ordinates):
        """
        :param co_ordinates: a tuple ot co_ordinates of the given vertex
        """
        self.co_ordinates = np.array(co_ordinates)

class Face:
    def __init__(self, vertices_index):
        self.n  = len(vertices_index)
        self.vertices_index = np.array(vertices_index)
        self.curve = None

    def make_curve(self, vertices, curve):
        """
        :param vertices: vertices of the face
        :param curve: which curve to make, could be Bezier or Bspline
        produces a curve on the given set of points
        """
        list_of_points = deepcopy([vertices[i].co_ordinates for i in self.vertices_index])
        if len(list_of_points) == 9:
            list_of_points = [list_of_points[0],list_of_points[4], list_of_points[1], list_of_points[7], list_of_points[8], list_of_points[5], list_of_points[3], list_of_points[6], list_of_points[2]]
        self.curve = curve(list_of_points)

    def plot_curve(self, color, ax = None):
        """
        :param color: color of the plot
        :param ax: on which to plot, if ax is None, it plot to the default plot, i.e. plt
        produces a curve on the given set of points
        """
        if self.curve != None:
            if ax != None:

                return self.curve.plot_curve(color = color, ax = ax)
            else:
                self.curve.plot_curve(color = color)



class Mesh:
    def __init__(self, filename):
        """
        :param filename: the name of the text file to open and read
        """
        txtFile = open(filename)
        self.file_name = filename.replace('.txt','')
        self.vertices, self.faces = (int(i) for i in txtFile.readline().strip().split())
        self.list_of_vertices = []
        self.list_of_faces = []
        self.directed_edges = dict()
        for i in range(self.vertices):
            line = txtFile.readline()
            vertices = tuple(float(i) for i in line.strip().split())
            self.list_of_vertices.append(Vector(vertices))
        self.dimensions = len(vertices)
        self.edges = dict()
        for i in range(self.faces):
            line = txtFile.readline()
            faces = tuple(int(k) for k in line.strip().split())
            self.list_of_faces.append(Face(faces))
        if len(vertices) == 3:
            self.ax = plt.axes(projection='3d')  #creating a new plot for 3d plots
        else:
            self.ax = None

    def fit_curve(self, curve):
        for i in self.list_of_faces:
            i.make_curve(self.list_of_vertices, curve) #fiting the curve on all faces

    def show_curve(self, color = 'blue'):
        for i in self.list_of_faces:
            i.plot_curve(color = color, ax = self.ax) #ploting the curve of each face


    def show_points(self, color = 'red'):
        if len(self.list_of_vertices[0].co_ordinates) == 2:
            plt.scatter(*zip(*[i.co_ordinates for i in self.list_of_vertices]), color=color)
        else:
            self.ax.scatter(*zip(*[i.co_ordinates for i in self.list_of_vertices]), color=color)

