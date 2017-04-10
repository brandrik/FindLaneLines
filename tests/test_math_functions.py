""" Tests for math_functions.py"""
# also import from parant directory
import pytest
from inspect import getsourcefile
import os.path
import sys


# enable import from modules lying in parent directory
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]

sys.path.insert(0, parent_dir)

from lib.math_functions import StraightLine, slope, line_interception_y_axis, average_straight_lines, extrapolate_line


def test_slope():
    test_line = StraightLine(0, 0, 10, 10)
    expected_slope = 1
    assert expected_slope == slope(test_line)
    
    
def test_line_interception_y_axis():
    test_line = StraightLine(1, 0, 2, 2)
    expected_interception_point = -2
    assert expected_interception_point == line_interception_y_axis(test_line)
  

def test_extrapolate_line():
    ymin = 0
    ymax = 540
    test_line = StraightLine(1, 0, 2, 2)
    expected_extrapolated = StraightLine(1, 0, 271, 540)
    assert expected_extrapolated == extrapolate_line(test_line, ymin, ymax)
    
    
def test_average_straight_lines():
    test_line_one = StraightLine(0, 0, 10, 10)
    test_line_two = StraightLine(4, 4, 6, 6)

    expected_average = StraightLine(2, 2, 8, 8)
    assert expected_average == average_straight_lines([test_line_one, test_line_two])
    

