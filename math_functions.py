# mathmatical functions for lines
from collections import namedtuple
from typing import Sequence
import numpy as np

StraightLine = namedtuple("line", "x1 y1 x2 y2")


def slope(line: StraightLine) -> float:
    """ Returns: slope of the line. """

    #import pdb; pdb.set_trace()

    delta_y = line.y2 - line.y1
    delta_x = line.x2 - line.x1

    # avoid crashing, compromising accuracy
    if (delta_x == 0):  delta_x = 0.00000001
    return (delta_y / delta_x)

def line_interception_y_axis(line: StraightLine) -> int:
    """ Returns: y coordinate of interception with  y axis
    """
    # b = y - mx

    b = line.y1 - slope(line)*line.x1
    return (b)


def extrapolate_line(line: StraightLine, start_y, end_y) -> StraightLine:
    """ Returns: extrapolated staight line according to start and end value of y """
    # calculate corresponding x coordinates (y = mx + b) -> x = (y-b)/m
    m = slope(line)
    b = line_interception_y_axis(line)
    start_x = int((start_y - b) / m)
    end_x   = int((end_y - b) / m)

    extrapolatedLine = StraightLine(*[start_x, start_y, end_x, end_y])

    return (extrapolatedLine)


def average_straight_lines(lines: Sequence[StraightLine]) -> Sequence[StraightLine]:
    """ Returns: the average of sequence of lines, if list is not empty. """
    # y = m1*x + b1
    # y = m2*x + b2
    # -> average over slope 'm'' and interception w/ y axis 'b', easier -> average over start and endpoints
    num_lines = len(lines)
    if num_lines < 2: return(lines)
    average_x1 = int(sum(line.x1 for line in lines)/num_lines)
    average_x2 = int(sum(line.x2 for line in lines)/num_lines)
    average_y1 = int(sum(line.y1 for line in lines)/num_lines)
    average_y2 = int(sum(line.y2 for line in lines)/num_lines)
    #import pdb; pdb.set_trace()
    average_line = StraightLine(*[average_x1, average_y1, average_x2, average_y2])

    return(average_line)

#def filter
