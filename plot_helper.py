import matplotlib.pyplot as plt
import itertools as it
    
COLOURS = it.cycle(('-bo', '-ro', '-go', '-ko', '-yo', '-co', '-mo',
                    '-bs', '-rs', '-gs', '-ks', '-ys', '-cs', '-ms',
                    '-b*', '-r*', '-g*', '-k*', '-y*', '-c*', '-m*'))

def define_plot(x_title='', y_title='', title='', label_prefix='', labels=''):
    plt.ion()
    plt.figure(1)
    plt.clf()
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.draw()
    plt.show()

def plot_accuracy(x, y, label=''):
    plt.ion()
    plt.plot(x, y, next(COLOURS), label=label)

def ylim(min, max):
    plt.ylim((min - 0.1 * min, max + 0.1 * max))

def legend(loc):
    plt.legend(loc=loc)
