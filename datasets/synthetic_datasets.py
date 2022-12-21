import numpy as np
from sklearn import datasets
import math


def make_alternate_stripes():
    np.random.seed(0)

    data = []

    for i in [np.pi * i for i in [-3, -1, 1, 3, 5]]:
        noise = np.random.randn(200, 1)*0.1
        x = np.expand_dims(np.repeat(i, 200), axis=1) + noise
        y = noise

        concat = np.concatenate((x, y), axis=1)
        data.append(concat)

    for i in [np.pi * i for i in [-4, -2, 0, 2, 4]]:
        noise = np.random.randn(200, 1)*0.1
        x = np.expand_dims(np.repeat(i, 200), axis=1) + noise
        y = noise

        concat = np.concatenate((x, y), axis=1)
        data.append(concat)

    data_x = np.reshape(np.array(data), (2000, 2))

    return data_x


def fibonacci_sphere(r, samples=1000, mu=0, sigma=0.05):
    np.random.seed(0)
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

        noise = np.random.randn(1000, 3)
        noise = noise * sigma + mu

    return np.array(points) * r + noise


def make_two_spheres():
    sphere_one = fibonacci_sphere(0.1)
    sphere_two = fibonacci_sphere(0.5)

    x = np.concatenate((sphere_one, sphere_two), axis=0)

    return x


def circles_data(x0=0, y0=0):
    np.random.seed(0)
    x, y = datasets.make_circles(n_samples=1000, factor=0.1, noise=0.05)
    x[:, 0] += x0
    x[:, 1] += y0

    return x
