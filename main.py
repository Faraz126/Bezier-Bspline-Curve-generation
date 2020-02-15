from MeshLib import *
from BezierCurve import *
from BsplineCurve import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    curve1 = BezierCurve([])
    curve2 = BsplineCurve([])
    LINEAR = 1
    QUADRATIC = 2
    CUBIC = 3
    QUARTIC = 4

    # Q1(a)
    """
    uncomment each line to see different basis
    """
    #curve1.plot_basis(no_of_variables = 1,  degree = LINEAR, func = curve1.bezier_basis,  labels=['t', 'N(t)'], title = 'Linear Basis Function over unit interval') #Q1(a)
    #curve1.plot_basis(no_of_variables = 1,  degree = QUADRATIC, func = curve1.bezier_basis, labels=['t', 'N(t)'], title='Quadratic Basis Function over unit interval') #Q1(a)
    #curve1.plot_basis(no_of_variables = 2, degree = LINEAR, func = curve1.bezier_basis, labels=['t', 's', 'N(s,t)'], title='Linear Basis Function over unit square')  # Q1(a)
    #curve1.plot_basis(no_of_variables = 2, degree = QUADRATIC, func = curve1.bezier_basis, labels=['t', 's', 'N(s,t)'], title='Quadratic Basis Function over unit square')  # Q1(a)

    """
        uncomment each line to see different basis
    """
    #curve2.plot_basis(k = 2, labels = ['t', 'N_i,2(t)'], title = "Bspline Basis with linear basis") #Q1(b)
    #curve2.plot_basis(k = 3, labels = ['t', 'N_i,3(t)'], title = "Bspline Basis with Quadratic basis") #Q1(b)
    #curve2.plot_basis(k = 4, labels = ['t', 'N_i,4(t)'], title = "Bspline Basis with Qubic basis") #Q1(b)
    #curve2.plot_basis(k= 5, labels=['t', 'N_i,5(t)'], title="Bspline Basis with Quartic basis")  # Q1(b)

    """
        uncomment one set of points, and then uncomment plot curve function to see the curve. Uncomment, show points, to view control points.
    """
    #curve1.list_of_points = [np.array(i) for i in [(1,2), (4,5), (3,2)]] #Q1(c) #quadratic
    #curve1.list_of_points = [np.array(i) for i in [(1.5, 2), (4, 5), (2,5), (4,9)]] #Q1(c) qubic
    #curve1.list_of_points = [np.array(i) for i in [(1, 2), (7, -5), (3, -5), (5,9), (10,15)]]
    #curve1.plot_points() #Q1(c)
    #curve1.plot_curve() #Q1(c) #uncomment to see
    #curve1.plot_curve(passing_through_all_point= True, color = 'orange')

    """
        uncomment each line to see different basis.
    """
    #curve1.plot_basis(no_of_variables = 1,  degree = QUADRATIC, func = curve1.bezier_basis_interpolate,  labels=['t', 'N(t)'], title = 'Quadratic Basis function to pass over all points') #Q1(d)
    #curve1.plot_basis(no_of_variables=1, degree= CUBIC, func=curve1.bezier_basis_interpolate, labels=['t', 'N(t)'],title='Cubic Basis function to pass over all points')  # Q1(d)


    """
    for Q1(d)
    Uncomment one set of points, and the remaining functions (plot_points, plot_curve)  to see the plots
    """
    #curve1.list_of_points = [np.array(i) for i in [(1,2), (4,5), (3,2)]]
    #curve1.list_of_points = [np.array(i) for i in [(1.5, 2), (4, 5), (2,5), (4,9)]]
    #curve1.list_of_points = [np.array(i) for i in [(0, 0), (2, 4), (3,9)]]
    #curve1.list_of_points = [np.array(i) for i in [(0, 0), (1.5, 2.25), (2,4), (3, 9)]]
    #curve1.plot_points()
    #curve1.plot_curve()
    #curve1.plot_curve(passing_through_all_point=True, color = 'orange')


    """
        for Q1(e)
    """
    #curve1.list_of_points = [np.array(i) for i in [(1.5, 2), (4, 5), (2, 5), (4, 9)]]
    #curve2 = BezierCurve([(0, 0), (1.5, 2.25), (2,4), (12, 15)])
    #curve1.make_continuous(curve2, c = 2)
    #curve1.plot_curve()
    #curve1.plot_points()
    #curve2.plot_curve()
    #curve2.plot_points()




    """
        for Q1(g)
        Uncomment one set of points, and the remaining functions (plot_points, plot_curve)  to see the plots
    """

    #curve1.list_of_points = [np.array(i) for i in [[0,0,0], [1, 0, 2], [0, 1, 2], [1, 1, 2]]]
    #curve1.list_of_points = [np.array(i) for i in [[0,0,0], [1, 0, 1], [2, 0, 0], [0, 1, 1], [1, 1, 1], [2, 1, 1], [0,2, 0], [1,2,1], [2,2,0]]]
    #curve1.list_of_points = [np.array(i) for i in [[0,0,0], [1,0,0], [2,0,0],  [3,0,0], [0,1,0], [1,1,4], [2,1,4],  [3,1,0], [0,2,0], [1,2,8], [2,2,8],  [3,2,0], [0,3,0], [1,3,0], [2,3,0],  [3,3,0]]]
    #curve1.list_of_points = [np.array(i) for i in [[0,0,0], [1,0,0], [2,0,0],  [3,0,0], [0,1,0], [1,1,4], [2,1,4],  [3,1,0], [0,2,0], [1,2,8], [2,2,8],  [3,2,0], [0,3,0], [1,3,0], [2,3,0],  [3,3,0]]]
    #curve1.plot_points() #Q1(g)
    #curve1.plot_curve() #Q1(g)
    #curve1.plot_points()
    #curve1.plot_curve(passing_through_all_point=True)

    """
        uncomment each line to see different basis.
    """
    #curve1.plot_basis(no_of_variables = 2,  degree = QUADRATIC, func = curve1.bezier_basis_interpolate,  labels=['t', 's', "N(t)"], title = 'Quadratic Basis function to pass over all points') #Q1(g)
    #curve1.plot_basis(no_of_variables= 2, degree= CUBIC, func=curve1.bezier_basis_interpolate, labels=['t', "s", 'N(t)'], title='Cubic Basis function to pass over all points') #Q1(g)

    """
        #for Q1(h)
        curve1.list_of_points = [np.array(i) for i in [(0,0,0), (1,0,0), (2,0,0), (0,1,0), (1,1,1), (2,1,0), (0,2,0), (1,2,0), (2,2,0)]]
        curve2 = BezierCurve([(2,0,0), (3, 0, 0), (4, 0, 0), (2, 1, 0), (3, 1, -1), (4,1,1), (2,2,0),(3,2,0),(4,2,6)])
        ax = plt.axes(projection='3d')
        curve1.plot_curve(ax = ax)
        curve2.plot_curve(ax = ax)
        """


    """
    #uncomment the whole block for Q2
    myM = Mesh("circle.txt") 
    myM.fit_curve(BezierCurve) 
    myM.show_curve() 
    myM.show_points()  
    """


    #Uncommed the whole block for Q3
    myM = Mesh("sphere.txt") 
    myM.fit_curve(BezierCurve)
    myM.show_points() 
    myM.show_curve() 


    plt.legend()
    plt.show()