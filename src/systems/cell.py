from scipy.signal import convolve2d
import numpy as np


OUTER_TOTAL_FILTER = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]])


def no_pad_step(state, rule):
    outer_totals = convolve2d(state, OUTER_TOTAL_FILTER, mode='valid')
    cell_centers = state[1:-1, 1:-1]
    return (rule & (1 << (2 * outer_totals + cell_centers))).clip(0, 1)


def toroidal_step(state, rule):
    outer_totals = convolve2d(state, OUTER_TOTAL_FILTER, boundary='wrap', mode='same')
    return (rule & (1 << (2 * outer_totals + state))).clip(0, 1)
