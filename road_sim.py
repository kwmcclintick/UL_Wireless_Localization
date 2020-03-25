"""
TPMS RSS simulation
"""
import itertools
from itertools import combinations
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

'''
checks to see if area of two cars overlap for initialization purposes given the radii and centroids of the cars
'''
def overlap(car1, car2, car_len_rad, car_wid_rad):
    # is one rectangle on the left side of another
    if (car1[0]-car_len_rad > car2[0]+car_len_rad or car2[0]-car_len_rad > car1[0]+car_len_rad):
        return False
    # is one rectangle above the other
    if (car1[1]+car_wid_rad < car2[1]-car_wid_rad or car2[1]+car_wid_rad < car1[1]-car_wid_rad):
        return False
    return True

'''
check to see if the area of any car obstructs the direct path from a tire-receiver pair 
'''
def LOS(tire, receiver, cars):
    #########################
    # YOUR CODE HERE
    #########################
    return True


'''
compute the distance of a ray in meters given its transmitting tire and receiving road side radio
'''
def ray_length(tire, receiver):
    distance = None
    #########################
    # YOUR CODE HERE
    #########################
    return distance


'''
Plots the road, all cars, receivers
'''
def plot_road(cars, bound_boxes, car_wid_rad, car_len_rad, xMin, xMax, yMin, yMax, receivers):
    shoulder = 2  # off-road shoulder for receivers placement, meters

    plt.scatter(cars[:, 0], cars[:, 1], edgecolor='b', facecolor='none')
    plt.scatter(bound_boxes[:, :, 0], bound_boxes[:, :, 1])
    plt.scatter(receivers[:, 0], receivers[:, 1], edgecolor='r', facecolor='none')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim([xMin - car_len_rad, xMax + car_len_rad])
    plt.ylim([yMin - car_wid_rad-shoulder, yMax + car_wid_rad+shoulder])
    plt.legend(['Car Centroids', 'Tires', 'Receivers'])
    plt.scatter(np.linspace(xMin-car_len_rad, xMax + car_len_rad), np.ones(50)*(yMax+car_wid_rad), marker='_', c='k')
    plt.scatter(np.linspace(xMin - car_len_rad, xMax + car_len_rad), np.ones(50)*(yMin - car_wid_rad), marker='_', c='k')
    plt.show()

'''
returns the centroids and bounding boxes of all inital cars. Does not create cars that are off the road or overlapping
with other cars
'''
def init_cars(car_len_rad, car_wid_rad, xMin, xMax, yMin, yMax):

    xDelta = xMax - xMin
    yDelta = yMax - yMin  # rectangle dimensions
    areaTotal = xDelta * yDelta

    # Point process parameters
    lambda0 = 0.5  # intensity (ie mean density) of the Poisson process

    # Simulate Poisson point process
    numbPoints = scipy.stats.poisson(lambda0 * areaTotal).rvs()  # Poisson number of points
    xx = xDelta * scipy.stats.uniform.rvs(0, 1, ((numbPoints, 1))) + xMin  # x coordinates of Poisson points
    yy = yDelta * scipy.stats.uniform.rvs(0, 1, ((numbPoints, 1))) + yMin  # y coordinates of Poisson points

    # remove overlapping inits
    cars = np.array([list(a) for a in zip(xx, yy)])
    any_overlaps = False
    for two_cars in combinations(cars, 2):
        # check to see if two cars overlap
        overlaps = overlap(two_cars[0], two_cars[1], car_len_rad, car_wid_rad)
        # remove random car
        if overlaps:
            cars = np.delete(cars, np.argwhere(cars == two_cars[0]))
            any_overlaps = True
    # removing overlapping cars changes shape, must change back
    if any_overlaps:
        cars = cars.reshape((int(len(cars) / 2), 2))

    # define bounding box for each car as [bottom left, top left, bottom right, top right]
    bound_boxes = []
    for i in range(len(cars)):
        bound_boxes.append(
            [[cars[i, 0] - car_len_rad, cars[i, 1] - car_wid_rad], [cars[i, 0] - car_len_rad, cars[i, 1] + car_wid_rad],
             [cars[i, 0] + car_len_rad, cars[i, 1] - car_wid_rad],
             [cars[i, 0] + car_len_rad, cars[i, 1] + car_wid_rad]])
    bound_boxes = np.array(bound_boxes)

    return cars, bound_boxes



if __name__== "__main__":
    car_len_rad = 4.56 / 2.  # 2013 subaru forester length in meters
    car_wid_rad = 2.006 / 2.  # width in meters
    # Simulation window parameters
    xMin = 0 + car_len_rad  # road start
    xMax = 25 - car_len_rad  # road end
    yMin = 0 + car_wid_rad  # road bottom
    yMax = 4 - car_wid_rad  # road top

    # initialize cars
    cars, bound_boxes = init_cars(car_len_rad, car_wid_rad, xMin, xMax, yMin, yMax)

    # manual placement of receivers, [3, 2] array
    receivers = np.array([[25 / 2., -1], [25 / 3., 5], [25 * 2 / 3., 5]])

    # coordinates of all tires, a [num_cars*4, 2] array
    tires = bound_boxes.reshape((int(len(bound_boxes)*4),2))
    #Determine if each tire - receiver pair is LOS or NLOS
    #########################
    # YOUR CODE HERE
    # for tire_receiver in ...:
    #      los = LOS(tire, receiver, cars)
    #      distance = ray_length(tire, receiver)
    #########################

    # plot road, cars, recievers
    plot_road(cars, bound_boxes, car_wid_rad, car_len_rad, xMin, xMax, yMin, yMax, receivers)
