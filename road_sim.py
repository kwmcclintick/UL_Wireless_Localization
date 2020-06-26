"""
TPMS RSS simulation returning a matrix of RSS values and the true location of the transmitters giving those values
"""
import itertools
from itertools import combinations
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import math


'''
check to see if the area of any car obstructs the direct path from a tire-receiver pair , returns True if LOS, False if NLOS
'''
def LOS(transmitter, receiver, cars): #Assumes transmitter and receiver are arrays containing x,y coordinates.

    tx2d = [] #use only 2D location for LOS calculation
    tx2d.append(transmitter[0])
    tx2d.append(transmitter[1])
    tx2d = np.array(tx2d)

    a = [tx2d, receiver] #Tx Rx Vector

    #for each car
    for n in range(len(cars)):

        if np_seg_intersect(a, [cars[n].corners[0], cars[n].corners[3]]): #Check bottom left to top right diagonal, if they intersect LOS is false
            return False
        elif np_seg_intersect(a, [cars[n].corners[1], cars[n].corners[2]]): #Check top left to bottom right diagonal, if they intersect LOS is false
            return False
    return True #LOS if no intersects


'''
Supporting functions for LOS function
Sources:
# https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/36416304#36416304
# https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
# http://www.codeproject.com/Tips/862988/Find-the-intersection-point-of-two-line-segments
'''
def np_perp( a ) :
    b = np.empty_like(a)
    b[0] = a[1]
    b[1] = -a[0]
    return b

def np_cross_product(a, b):
    return np.dot(a, np_perp(b))

def np_seg_intersect(a, b):
    considerCollinearOverlapAsIntersect = False
    r = a[1] - a[0]
    s = b[1] - b[0]
    v = b[0] - a[0]
    num = np_cross_product(v, r)
    denom = np_cross_product(r, s)
    # If r x s = 0 and (q - p) x r = 0, then the two lines are collinear.
    if np.isclose(denom, 0) and np.isclose(num, 0):
        # 1. If either  0 <= (q - p) * r <= r * r or 0 <= (p - q) * s <= * s
        # then the two lines are overlapping,
        if(considerCollinearOverlapAsIntersect):
            vDotR = np.dot(v, r)
            aDotS = np.dot(-v, s)
            if (0 <= vDotR  and vDotR <= np.dot(r,r)) or (0 <= aDotS  and aDotS <= np.dot(s,s)):
                return True
        # 2. If neither 0 <= (q - p) * r = r * r nor 0 <= (p - q) * s <= s * s
        # then the two lines are collinear but disjoint.
        # No need to implement this expression, as it follows from the expression above.
        return False
    if np.isclose(denom, 0) and not np.isclose(num, 0):
        # Parallel and non intersecting
        return False
    u = num / denom
    t = np_cross_product(v, s) / denom
    if u >= 0 and u <= 1 and t >= 0 and t <= 1:
        res = b[0] + (s*u)
        return True
    # Otherwise, the two line segments are not parallel but do not intersect.
    return False


'''
compute the distance of a ray in meters given its transmitting tire and receiving road side radio
'''
def ray_length(transmitter, receiver):
    tx2d = [] #use only 2D location for distance calculation since no Rx height is available
    tx2d.append(transmitter[0])
    tx2d.append(transmitter[1])
    tx2d = np.array(tx2d)

    distance = math.sqrt((receiver[0]-tx2d[0])**2 + (receiver[1]-tx2d[1])**2)
    return distance


'''
Compute the RSS of a signal sent from a transmitter to a receiver
'''
def compute_rss(distance, LOS):
    #Parameters
    P0 = -59.78 #dBm
    #  9.391907167846293 for los
    # 4.435287398731233 for grid
    txPwrVar = 4.435287398731233 #dB
    LOS_grad = -1.12
    NLOS_grad = -2.38
    #Account for variation in Tx Power
    norm = np.random.normal(0, 1)
    P0var = P0 + norm * txPwrVar #Tx Power varies by about 4 dB
    #Calculate RSS
    if LOS:
        RSS = P0var + 10*LOS_grad*math.log(distance,10)
    else:
        RSS = P0var + 10*NLOS_grad*math.log(distance,10)
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
def plot_road(num_lanes, lane_width, cars, car_wid_rad, car_len_rad, xMin, xMax, yMin, yMax, receivers, t):
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
    plt.ylim([yMin - car_wid_rad-extra_plot_y, yMax + car_wid_rad+extra_plot_y + num_lanes*lane_width])
    plt.legend(['Car Centroids', 'Car Edges', 'Receivers', 'Transmitters'])
    for car in cars:
        plt.fill(car.corners[:, 0], car.corners[:, 1], c='b')

    for i in range(num_lanes):
        plt.scatter(np.linspace(xMin-car_len_rad, xMax + car_len_rad + extra_plot_x), np.ones(50)*(i*lane_width), marker='_', c='k')
        plt.scatter(np.linspace(xMin - car_len_rad, xMax + car_len_rad+extra_plot_x), np.ones(50)*((i+1)*lane_width), marker='_', c='k')
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
returns the centroids and bounding boxes of all initial cars. Does not create cars that are off the road or overlapping
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
    x_speed = 38.+np.random.randn(1)[0] # m/s
    y_speed = 0.+np.random.randn(1)[0]

    # create instances of car class
    car_list = []
    for i in range(len(centers)):
        car_list.append(Car(centers[i], [x_speed, y_speed], bound_boxes[i], transmitters[i]))

    return car_list


'''
Updates the position of all cars, their edges, and their transmitters
'''
def MP(cars, t_res, tire_corner_dist, tire_thickness, rim_diameter, t):
    # define MP matrices
    F = np.array([[1, 0, t_res, 0], [0, 1, 0, t_res], [0, 0, 1, 0], [0, 0, 0, 1]])  # state advance matrix
    G = np.array([[t_res**2/2, 0], [0, t_res**2/2], [t_res, 0], [0, t_res]])  # process advance matrix
    var = 5.
    Q = var*np.eye(2)  # process noise covariance

    # step through markov process one step
    for i in range(len(cars)):
        # TPMS transmitter changes x position and height with rotation of the tire
        freq = 1 / (np.pi * rim_diameter)  # rotations per meter
        freq = freq * cars[i].velocity[0]  # rotations per second, as vel is in m/s
        x_tpms = np.sin(2 * np.pi * freq * t)
        h_tpms = 2 * tire_thickness + rim_diameter * (1 + np.cos(2 * np.pi * freq * t))

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


'''
A grid of stationary cars spread across an area of interest, representing training data collected
'''
def grid_cars(car_len_rad, car_wid_rad, xMin, xMax, yMin, yMax, freq, tire_corner_dist, tire_thickness, rim_diameter):
    num_cars_x = 3  # how many cars from end to end of a lane
    num_cars_y = 3 # how many cars from top to bottom of a lane
    x, y = np.meshgrid(np.linspace(xMin+1., xMax-1., num_cars_x), np.linspace(yMin-car_wid_rad-2., yMax+car_wid_rad+2., num_cars_y))
    centers = np.concatenate((np.expand_dims(x.flatten(), axis=1), np.expand_dims(y.flatten(), axis=1)), axis=1)

    # # experimental training grid locations
    centers = np.array([[ 3.667,    8.94775],
             [ 3.8795,   6.761  ],
             [ 4.00075,  4.34425],
             [ 6.77125, 10.4065 ],
             [ 6.89075,  7.9435 ],
             [ 7.155,    4.8235 ],
             [ 9.8735,   9.213  ],
             [ 9.91125,  8.04275],
             [10.0495,   5.91475]])

    # # experimental los test locations
    # centers = np.array([[5.3495, 9.2845],
    #         [5.9715, 6.7265]])

    # 150, 1000
    centers = np.repeat(centers,1000,axis=0) # repeat each position so many times to simulate a window of recording



    # define bounding box for each car as [bottom left, top left, bottom right, top right]
    bound_boxes = []
    for i in range(len(centers)):
        # if i %2==0:
        # horizontal cars
        bound_boxes.append(
            [[centers[i, 0] - car_len_rad, centers[i, 1] - car_wid_rad],
             [centers[i, 0] - car_len_rad, centers[i, 1] + car_wid_rad],
             [centers[i, 0] + car_len_rad, centers[i, 1] - car_wid_rad],
             [centers[i, 0] + car_len_rad, centers[i, 1] + car_wid_rad]])
        # else:
        # # vertical cars
        #     bound_boxes.append(
        #         [[centers[i, 0] - car_wid_rad, centers[i, 1] - car_len_rad],
        #          [centers[i, 0] - car_wid_rad, centers[i, 1] + car_len_rad],
        #          [centers[i, 0] + car_wid_rad, centers[i, 1] - car_len_rad],
        #          [centers[i, 0] + car_wid_rad, centers[i, 1] + car_len_rad]])
    bound_boxes = np.array(bound_boxes)

    # define transmitter locations for each car
    ################  TPMS  #####################
    t = 0  # computed at t=0
    h_tpms = 2 * tire_thickness + rim_diameter * (1 + np.cos(2 * np.pi * freq * t))
    x_tpms = np.sin(2 * np.pi * freq * t)

    transmitters = []
    for i in range(len(centers)):
        # bottom left tire's tpms transmitter
        # if i % 2 ==0:
        transmitters.append(
            [[centers[i, 0] - car_len_rad + tire_corner_dist + x_tpms, centers[i, 1] - car_wid_rad, h_tpms],
             [centers[i, 0] - car_len_rad + tire_corner_dist + x_tpms, centers[i, 1] + car_wid_rad, h_tpms],
             [centers[i, 0] + car_len_rad - tire_corner_dist + x_tpms, centers[i, 1] - car_wid_rad, h_tpms],
             [centers[i, 0] + car_len_rad - tire_corner_dist + x_tpms, centers[i, 1] + car_wid_rad, h_tpms]])
        # else:
        #     transmitters.append(
        #         [[centers[i, 0] - car_wid_rad, centers[i, 1] - car_len_rad + tire_corner_dist + x_tpms, h_tpms],
        #          [centers[i, 0] - car_wid_rad, centers[i, 1] + car_len_rad - tire_corner_dist + x_tpms, h_tpms],
        #          [centers[i, 0] + car_wid_rad, centers[i, 1] - car_len_rad + tire_corner_dist + x_tpms, h_tpms],
        #          [centers[i, 0] + car_wid_rad, centers[i, 1] + car_len_rad - tire_corner_dist + x_tpms, h_tpms]])
    transmitters = np.array(transmitters)

    # define initial velocity
    x_speed = 0.
    y_speed = 0.

    # create instances of car class
    car_list = []
    for i in range(len(centers)):
        car_list.append(Car(centers[i], [x_speed, y_speed], bound_boxes[i], transmitters[i]))

    return car_list

if __name__== "__main__":
    mode = 'test'  # train for grid, test for path

    car_len_rad = 4.56 / 2.  # 2013 subaru forester length in meters
    car_wid_rad = 2.006 / 2.  # width in meters
    # Simulation window parameters
    num_lanes = 3
    lane_width = 4
    xMin = 0 + car_len_rad  # road start
    xMax = 15 - car_len_rad  # road end
    yMin = 0 + car_wid_rad  # road bottom
    yMax = lane_width - car_wid_rad  # road top

    # TPMS parameters
    tire_corner_dist = .9725
    freq = 12.5  # avg tire has 12.5 RPS at 60 MPH
    # TPMS height changes as the tire rotates, sizes are for 2013 subaru forester
    tire_thickness = 0.225  # meters
    rim_diameter = 0.381

    # initialize cars
    cars = []
    for i in range(num_lanes):
        if mode == 'test':
            cars.extend(init_cars(car_len_rad, car_wid_rad, xMin, xMax, yMin+i*lane_width, yMax+i*lane_width, freq,
                                  tire_corner_dist, tire_thickness, rim_diameter))
        if mode == 'train' and i==1:
            cars.extend(grid_cars(car_len_rad, car_wid_rad, xMin, xMax, yMin+i*lane_width, yMax+i*lane_width, freq,
                                  tire_corner_dist, tire_thickness, rim_diameter))

    if mode == 'test':
        # define Markov process parameters
        t_steps = 1000  # number of time steps.
        step_size = 0.002
        end = 1.
        time = np.linspace(start=step_size, stop=end, num=int(end/step_size))  # 26.822 m/s is 60 MPH
    else:
        time = np.arange(len(cars))



    # # manual placement of receivers, [3, 2] array
    receiver_curb_dist = 1  # how many meters offroad the receivers are
    # receivers = np.array([[(xMax-xMin+2*car_len_rad) / 2., yMin-car_wid_rad-receiver_curb_dist],
    #                       [xMin, (yMax+car_wid_rad)*num_lanes+receiver_curb_dist],
    #                       [xMax, (yMax+car_wid_rad)*num_lanes+receiver_curb_dist]])

    receivers = np.array([[7.5, 0], [0, 12.99], [15, 12.99]])

    # num_recs = 20
    #
    # a = np.expand_dims(np.linspace(xMin, xMax, num_recs), axis=1)
    # b = np.array((yMin - car_wid_rad - receiver_curb_dist) * np.ones(shape=(num_recs,1)))
    # receivers = np.array(np.concatenate((a,b), axis=1))
    #
    # c = np.array(((yMax+car_wid_rad)*num_lanes+receiver_curb_dist) * np.ones(shape=(num_recs,1)))
    # receivers_top = np.array(np.concatenate((a,c), axis=1))
    # receivers = np.concatenate((receivers, receivers_top), axis=0)

    # initialize machine learning input/output data matrix
    if mode=='test':
        num_transmitters = 0
        for car in cars:
            num_transmitters += len(car.transmitters)
        RSS = np.zeros(shape=(len(receivers), num_transmitters * len(time)))
        true_loc = np.zeros(shape=(2, num_transmitters * len(time))) #2 dimensions since no rx height
    else:
        num_transmitters = 4
        RSS = np.zeros(shape=(len(receivers), num_transmitters * len(time)))
        true_loc = np.zeros(shape=(2, num_transmitters * len(time)))  # 2 dimensions since no rx height

    if mode == 'train':
        print(time.shape)
        # loop over time
        for i in range(len(time)):
            # Compute RSS of each transmit receive pair
            for j in range(len(receivers)):
                car = cars[i]
                n = 0
                for m in range(len(car.transmitters)):
                    k = (num_transmitters * i) + (len(car.transmitters) * n) + m #index for recording RSS values within a receiver row. Will be sorted by Time, then Car within that, then Tx/Tire within that
                    los = LOS(car.transmitters[m], receivers[j], [car])
                    distance = ray_length(car.transmitters[m], receivers[j])
                    RSS[j,k] = compute_rss(distance, los)
                    # true_loc[0, k] = np.mean(car.transmitters[:,0])
                    # true_loc[1, k] = np.mean(car.transmitters[:,1])

                    true_loc[0, k] = car.transmitters[m,0]
                    true_loc[1, k] = car.transmitters[m,1]

            # plot road, cars, recievers
            # if i == 100:
            # plot_road(num_lanes, lane_width, [car], car_wid_rad, car_len_rad, xMin, xMax, yMin, yMax, receivers, time[i])

    else:
        # loop over time
        for i in range(len(time)):
            # Compute RSS of each transmit receive pair
            for j in range(len(receivers)):
                for n in range(len(cars)):
                    for m in range(len(car.transmitters)):
                        k = (num_transmitters * i) + (len(
                            car.transmitters) * n) + m  # index for recording RSS values within a receiver row. Will be sorted by Time, then Car within that, then Tx/Tire within that
                        los = LOS(cars[n].transmitters[m], receivers[j], cars)
                        distance = ray_length(cars[n].transmitters[m], receivers[j])
                        RSS[j, k] = compute_rss(distance, los)
                        true_loc[0, k] = cars[n].transmitters[m][0]
                        true_loc[1, k] = cars[n].transmitters[m][1]

            # plot road, cars, recievers
            if i % 100 == 1:
                plot_road(num_lanes, lane_width, cars, car_wid_rad, car_len_rad, xMin, xMax, yMin, yMax, receivers, time[i])
            # move simulation one time step forward
            cars = MP(cars, time[-1] / t_steps, tire_corner_dist, tire_thickness, rim_diameter, time[i])

    np.save('X_sim_grid', RSS)
    np.save('Y_sim_grid',true_loc)



