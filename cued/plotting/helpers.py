import matplotlib as mpl
import matplotlib.pyplot as plt

default_labels = ['(' + chr(ord('a') + i) + ')' for i in range(26)]
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def label_inner(ax, idx=None, color='black'):
    '''
    Counts its calls via counter and increments alphabet
    Labels plots consecutively when called.
    If idx is given it puts that label
    '''
    if idx is None:
        ax.text(0.02, 0.88, r'{}'.format(default_labels[plt_fig_idx_inner.counter]), transform=ax.transAxes, color=color)
        label_inner.counter += 1
    else:
        ax.text(0.02, 0.88, r'{}'.format(default_labels[idx]), transform=ax.transAxes, color=color)

label_inner.counter = 0

def contourf_remove_white_lines(contour):
    '''
    Remove artifact white lines from contour plots 
    '''
    for c in contour.collections:
        c.set_edgecolor('face')
        c.set_linewidth(0.000000000001)

