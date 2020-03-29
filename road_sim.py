"""
TPMS RSS simulation returning a matrix of RSS values and the true location of the transmitters giving those values
"""
import itertools
from itertools import combinations
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt



'''
check to see if the area of any car obstructs the direct path from a tire-receiver pair 
'''
def LOS(transmitter, receiver, cars):
    #########################
    # YOUR CODE HERE
    #########################
    return True


'''
compute the distance of a ray in meters given its transmitting tire and receiving road side radio
'''
def ray_length(transmitter, receiver):
    #########################
    # YOUR CODE HERE
    #########################
    return distance


'''
Compute the RSS of a signal sent from a transmitter to a receiver
'''
def compute_rss(transmitter, receiver, LOS):
    #########################
    # YOUR CODE HERE
    #########################
    return RSS


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
Plots the road, all cars, receivers
'''
def plot_road(cars, car_wid_rad, car_len_rad, xMin, xMax, yMin, yMax, receivers, t):
    extra_plot_y = 10  # extra distance for plot window
    extra_plot_x = 25

    centers = []
    corners = []
    transmitters = []
    for car in cars:
        centers.append(car.center)
        corners.append(car.corners)
        transmitters.append(car.transmitters)
    centers = np.array(centers)
    corners = np.array(corners)
    transmitters = np.array(transmitters)

    plt.scatter(centers[:, 0], centers[:, 1])
    plt.scatter(corners[:,:, 0], corners[:,:, 1])
    plt.scatter(receivers[:, 0], receivers[:, 1],  marker='^')
    plt.scatter(transmitters[:,:,0], transmitters[:,:,1], marker='^')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title('Time = '+str(np.around(t,decimals=3)))
    plt.xlim([xMin - car_len_rad, xMax + car_len_rad+extra_plot_x])
    plt.ylim([yMin - car_wid_rad-extra_plot_y, yMax + car_wid_rad+extra_plot_y])
    plt.legend(['Car Centroids', 'Car Edges', 'Receivers', 'Transmitters'])
    for car in cars:
        plt.fill(car.corners[:, 0], car.corners[:, 1], c='b')
    plt.scatter(np.linspace(xMin-car_len_rad, xMax + car_len_rad + extra_plot_x), np.ones(50)*(yMax+car_wid_rad), marker='_', c='k')
    plt.scatter(np.linspace(xMin - car_len_rad, xMax + car_len_rad+extra_plot_x), np.ones(50)*(yMin - car_wid_rad), marker='_', c='k')
    plt.show()

'''
A car has a center (x,y) a velocity (x,y), four corners ((x,y),(x,y),(x,y),(x,y)), and some set of transmitters
((x,y),...(x,y)) 
'''
class Car:
    def __init__(self, center, velocity, corners, transmitters):
        self.center = center
        self.velocity = velocity
        self.corners = corners
        self.transmitters = transmitters

    def set_center(self, center):
        self.center = center

    def set_velocity(self, velocity):
        self.velocity = velocity

    def set_corners(self, corners):
        self.corners = corners

    def set_transmitters(self, transmitters):
        self.transmitters = transmitters


'''
returns the centroids and bounding boxes of all inital cars. Does not create cars that are off the road or overlapping
with other cars
'''
def init_cars(car_len_rad, car_wid_rad, xMin, xMax, yMin, yMax, freq, tire_corner_dist, tire_thickness, rim_diameter):
    xDelta = xMax - xMin
    yDelta = yMax - yMin  # rectangle dimensions
    areaTotal = xDelta * yDelta

    # Point process parameters
    lambda0 = 1.  # intensity (ie mean density) of the Poisson process

    # Simulate Poisson point process
    numbPoints = scipy.stats.poisson(lambda0 * areaTotal).rvs()  # Poisson number of points
    xx = xDelta * scipy.stats.uniform.rvs(0, 1, ((numbPoints, 1))) + xMin  # x coordinates of Poisson points
    yy = yDelta * scipy.stats.uniform.rvs(0, 1, ((numbPoints, 1))) + yMin  # y coordinates of Poisson points

    # remove overlapping inits
    centers = np.array([list(a) for a in zip(xx, yy)])
    any_overlaps = False
    for two_cars in combinations(centers, 2):
        # check to see if two cars overlap
        overlaps = overlap(two_cars[0], two_cars[1], car_len_rad, car_wid_rad)
        # remove random car
        if overlaps:
            centers = np.delete(centers, np.argwhere(centers == two_cars[0]))
            any_overlaps = True
    # removing overlapping cars changes shape, must change back
    if any_overlaps:
        centers = centers.reshape((int(len(centers) / 2), 2))

    # define bounding box for each car as [bottom left, top left, bottom right, top right]
    bound_boxes = []
    for i in range(len(centers)):
        bound_boxes.append(
            [[centers[i, 0] - car_len_rad, centers[i, 1] - car_wid_rad],
             [centers[i, 0] - car_len_rad, centers[i, 1] + car_wid_rad],
             [centers[i, 0] + car_len_rad, centers[i, 1] - car_wid_rad],
             [centers[i, 0] + car_len_rad, centers[i, 1] + car_wid_rad]])
    bound_boxes = np.array(bound_boxes)

    # define transmitter locations for each car
    ################  TPMS  #####################
    t = 0  # computed at t=0
    h_tpms = 2 * tire_thickness + rim_diameter * (1 + np.cos(2 * np.pi * freq * t))
    x_tpms = np.sin(2 * np.pi * freq * t)

    transmitters = []
    for i in range(len(centers)):
        # bottom left tire's tpms transmitter
        transmitters.append([[centers[i, 0] - car_len_rad + tire_corner_dist + x_tpms, centers[i, 1] - car_wid_rad, h_tpms],
                             [centers[i, 0] - car_len_rad + tire_corner_dist + x_tpms, centers[i, 1] + car_wid_rad, h_tpms],
                             [centers[i, 0] + car_len_rad - tire_corner_dist + x_tpms, centers[i, 1] - car_wid_rad, h_tpms],
                             [centers[i, 0] + car_len_rad - tire_corner_dist + x_tpms, centers[i, 1] + car_wid_rad, h_tpms]])
    transmitters = np.array(transmitters)

    # define initial velocity
    x_speed = 26.+np.random.randn(1)[0] # m/s
    y_speed = 0.+np.random.randn(1)[0]

    # create instances of car class
    car_list = []
    for i in range(len(centers)):
        car_list.append(Car(centers[i], [x_speed, y_speed], bound_boxes[i], transmitters[i]))

    return car_list



'''
Updates the position of all cars, their edges, and their transmitters
'''
def MP(cars, freq, t_res, tire_corner_dist, tire_thickness, rim_diameter, t):
    # define MP matrices
    F = np.array([[1, 0, t_res, 0], [0, 1, 0, t_res], [0, 0, 1, 0], [0, 0, 0, 1]])  # state advance matrix
    G = np.array([[t_res**2/2, 0], [0, t_res**2/2], [t_res, 0], [0, t_res]])  # process advance matrix
    var = 5.
    Q = var*np.eye(2)  # process noise covariance

    # TPMS parameters
    x_tpms = np.sin(2 * np.pi * freq * t)
    h_tpms = 2 * tire_thickness + rim_diameter * (1 + np.cos(2 * np.pi * freq * t))

    # step through markov process one step
    for i in range(len(cars)):
        U = np.array([np.random.multivariate_normal([0,0], Q)]).T  # process noise
        old_state = np.array([[cars[i].center[0], cars[i].center[1], cars[i].velocity[0], cars[i].velocity[1]]]).T
        new_state = (np.matmul(F, old_state) + np.matmul(G, U)).flatten()

        # update instance variales of car
        cars[i].set_center(new_state[0:2])
        cars[i].set_velocity(new_state[2:4])
        cars[i].set_corners(np.array([[cars[i].center[0] - car_len_rad, cars[i].center[1] - car_wid_rad],
                                     [cars[i].center[0] - car_len_rad, cars[i].center[1] + car_wid_rad],
                                     [cars[i].center[0] + car_len_rad, cars[i].center[1] - car_wid_rad],
                                     [cars[i].center[0] + car_len_rad, cars[i].center[1] + car_wid_rad]]))
        cars[i].set_transmitters(np.array([[cars[i].center[0] - car_len_rad + tire_corner_dist + x_tpms, cars[i].center[1] - car_wid_rad, h_tpms],
                             [cars[i].center[0] - car_len_rad + tire_corner_dist + x_tpms, cars[i].center[1] + car_wid_rad, h_tpms],
                             [cars[i].center[0] + car_len_rad - tire_corner_dist + x_tpms, cars[i].center[1] - car_wid_rad, h_tpms],
                             [cars[i].center[0] + car_len_rad - tire_corner_dist + x_tpms, cars[i].center[1] + car_wid_rad, h_tpms]]))
    return cars



if __name__== "__main__":
    car_len_rad = 4.56 / 2.  # 2013 subaru forester length in meters
    car_wid_rad = 2.006 / 2.  # width in meters
    # Simulation window parameters
    xMin = 0 + car_len_rad  # road start
    xMax = 25 - car_len_rad  # road end
    yMin = 0 + car_wid_rad  # road bottom
    yMax = 4 - car_wid_rad  # road top

    # define Markov process parameters
    t_steps = 10  # number of time steps.
    time = np.linspace(0, 1., t_steps)  # 26.822 m/s is 60 MPH

    # TPMS parameters
    tire_corner_dist = 1
    freq = 12.5  # avg tire has 12.5 RPS at 60 MPH
    # TPMS height changes as the tire rotates, sizes are for 2013 subaru forester
    tire_thickness = 0.225  # meters
    rim_diameter = 0.381

    # initialize cars
    cars = init_cars(car_len_rad, car_wid_rad, xMin, xMax, yMin, yMax, freq, tire_corner_dist, tire_thickness, rim_diameter)

    # manual placement of receivers, [3, 2] array
    receiver_curb_dist = 1  # how many meters offroad the receivers are
    receivers = np.array([[(xMax-xMin+2*car_len_rad) / 2., yMin-car_wid_rad-receiver_curb_dist],
                          [xMin, yMax+car_wid_rad+receiver_curb_dist],
                          [xMax, yMax+car_wid_rad+receiver_curb_dist]])

    # initialize RSS data matrix
    num_transmitters = 0
    for car in cars:
        num_transmitters += len(car.transmitters)
    RSS = np.zeros(shape=(len(time), num_transmitters * len(receivers)))
    true_loc = np.zeros(shape=(len(time), num_transmitters))

    # loop over time
    for i in range(len(time)):

        # Compute RSS of each transmit receive pair
        #########################
        # YOUR CODE HERE
        # for transmit_receive in ...:
        #      los = LOS(transmitter, receiver, cars)
        #      distance = ray_length(transmitter, receiver)
        #      RSS[i,...] = compute_rss(transmitter, receiver, LOS)
        #      true_loc[i,...] =
        #########################



        # move simulation one time step forward
        cars = MP(cars, freq, time[-1]/t_steps, tire_corner_dist, tire_thickness, rim_diameter, time[i])

        # plot road, cars, recievers
        plot_road(cars, car_wid_rad, car_len_rad, xMin, xMax, yMin, yMax, receivers, time[i])

