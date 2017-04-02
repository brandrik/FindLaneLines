# mathmatical functions for lines
from collections import namedtuple
from typing import Sequence
import numpy as np

Line = namedtuple("Line", "x1 y1 x2 y2")


def slope(line: np.array) -> float:
    """ Returns: slope of the line. """

    #import pdb; pdb.set_trace()
    
    delta_y = line[3] - line[1]
    delta_x = line[2] - line[0]

    # avoid crashing, compromising accuracy
    if (delta_x == 0):  delta_x = 0.00000001
    return (delta_y / delta_x)


def extrapolate_line(line: Line) -> Line:
    """ Returns: extrapolated line """
    return (line)


def average_of_lines(lines: Sequence[Line]) -> Sequence[Line]:
    """ Returns: average of sequence of lines, if list is not empty. """
    return(1)

#def filter
