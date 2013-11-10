import matplotlib.pyplot as plt

def plot_cost(debug_x, debug_y, filename=None):
    plt.figure()
    plt.subplot(111)
    plt.ylim(0, 10)
    plt.plot(debug_x, debug_y)
    if filename:
        plt.savefig(filename)
    plt.show()


def plot_regression(X, T, regression_x, regression_y, filename=None):
    plt.figure()
    plt.subplot(111)
    plt.xlim(-1, 1)
    plt.scatter(X, T)
    plt.plot(regression_x, regression_y)
    if filename:
        plt.savefig(filename)
    plt.show()


def plot_classification():
    pass

