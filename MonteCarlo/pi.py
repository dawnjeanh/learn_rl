import numpy as np
import math
import random

square_size = 1
points_inside_circle = 0
points_inside_square = 0
sample_size = 100000

def generate_points(size):
    x = random.random() * size
    y = random.random() * size
    return (x, y)

def is_in_circle(point, size):
    return math.sqrt(point[0] ** 2 + point[1] ** 2) <= size

def compute_pi(points_inside_circle, points_inside_square):
    return 4 * (points_inside_circle / points_inside_square)

for i in range(sample_size):
    point = generate_points(square_size)
    points_inside_square += 1
    if is_in_circle(point, square_size):
        points_inside_circle += 1

print('pi is {}'.format(compute_pi(points_inside_circle, points_inside_square)))