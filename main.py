import timeit
import load_data
import slam_lib
import p3_utils
import numpy as np
import matplotlib.pyplot as plt
from testing import testing
import pickle

''' 
user input 
change the file name of dataset and SLAM result
'''
joint_file = "joint/train_joint0"
lidar_file = "lidar/train_lidar0"
slam_data = "slam_data" # the name of slam data that will be saved, needed for texture mapping

if __name__ == '__main__':

    # load data
    j0 = load_data.get_joint(joint_file)
    head_idx = load_data.get_joint_index('Head')
    l0 = load_data.get_lidar(lidar_file)

    particle_num = 20  # num of particles
    n = len(l0)  # length of iterations

    particles = np.zeros((3, particle_num))  # empty array for storing particles
    weights = np.log(np.ones((1, particle_num))[0] / particle_num)  # init weights array
    odom_prev = np.array([0, 0, 0])  # previous odometry
    i_best = 0  # index of particle with highest weight
    MAP = slam_lib.init_map()  # init a empty map
    x_array = np.zeros((3, n))  # trajectory array
    ts_array = np.zeros((3, n))[0]  # timestamp array

    start = timeit.default_timer()  # start timer
    for i in range(n):
        ts = np.asscalar(l0[i]['t'])
        head = slam_lib.find_head_angles(j0, ts)  # find the head angles of corresponding timestamp
        odom = l0[i]['pose'][0]
        # x = slam_lib.dead_reckoning(x, odom_prev, odom) # deck reckoning
        # particles = slam_lib.prediction(particles, odom_prev, odom) # prediction only
        particles, weights, MAP, i_best = slam_lib.slam(particles, weights, l0[i], MAP, head, odom_prev, odom, i_best,
                                                        i)
        odom_prev = odom

        x_array[:, i] = particles[:, i_best]
        ts_array[i] = ts

        remaintime = round((timeit.default_timer() - start) / (i + 1.0) * (n - i) / 60)
        print str(round(i * 100.0 / n, 1)) + "%" + ' remain: ' + str(remaintime) + ' mins'

    print "done in " + str(round(timeit.default_timer() - start, 2)) + ' seconds'

    data = [MAP, x_array, ts_array]  # save data

    with open(slam_data, 'wb') as sd:
        pickle.dump(data, sd)

    slam_lib.plot_results(MAP, x_array)  # plot map and trajectory
