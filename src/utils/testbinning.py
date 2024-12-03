import matplotlib.pyplot as plt

from src.utils.plotting import Plotter
from src.utils.binning import power_function, generate_points, bin_and_average
def test_plot():
    '''
    Strategy in ["y_mean", "y_median", "slope_mean", "slope_median", "fit", "spline"].
    Additionally, can be in log domain or not.
    '''
    npoints = 5000
    power = -1
    # Represents Nq, MSE^0.5.
    fixed_point = (100, 1e-2) 
    mean = 0.7
    xrange = (fixed_point[0], 10**5)
    f = power_function(power, fixed_point)
    xs, ys = generate_points(npoints, xrange, mean, f, noise = True)
    
    gxs, gys, bins, grouped_points = bin_and_average(xs, ys, nbins = 10, 
                                                     fixed_point = fixed_point,
                                                     strategy = "y_mean",
                                                     logdomain = False,
                                                     full_output = True)
    
    plotter = Plotter()
    plotter.scatter_by_groups(grouped_points, ypower = 0.5)
    plotter.curve(f, xrange, npoints = 100, style = "--")
    plotter.vertical_lines(bins)
    plotter.scatter(gxs, gys)

    xlabel = "Number of queries"
    ylabel = "Root mean squared error"
    plotter.set_labels(xlabel, ylabel)


test_plot()
plt.show()