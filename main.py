import timeit
import load_data
import slam_lib
import p3_utils
import numpy as np
import matplotlib.pyplot as plt
from testing import testing
import pickle

if __name__ == '__main__':
    start = timeit.default_timer()  # start timer
    timer = start
    # load data
    j0 = load_data.get_joint("joint/train_joint0")
    head_idx = load_data.get_joint_index('Head')
    l0 = load_data.get_lidar("lidar/train_lidar0")
    # r0 = load_data.get_rgb("cam/RGB_0")
    # d0 = load_data.get_depth("cam/DEPTH_0")
    # exIR_RGB = load_data.getExtrinsics_IR_RGB()
    # IRCalib = load_data.getIRCalib()
    # RGBCalib = load_data.getRGBCalib()

    # visualize data
    # load_data.replay_lidar(l0)
    # load_data.replay_rgb(r0)
    # load_data.replay_depth(d0)

    n = len(l0)
    particle_num = 30
    x = np.array([0, 0, 0])
    particles = np.zeros((3, particle_num))
    weights = np.log(np.ones((1, particle_num))[0] / particle_num)
    odom_prev = np.array([0, 0, 0])
    MAP = slam_lib.init_map(l0[0], slam_lib.find_head_angles(j0, np.asscalar(l0[0]['t'])), np.array([0, 0, 0]))
    # x_array = np.zeros((3, n))
    x_array = np.zeros((3, particle_num * n))
    for i in range(n):
        ts = np.asscalar(l0[i]['t'])
        head = slam_lib.find_head_angles(j0, ts)
        odom = l0[i]['pose'][0]
        # x = slam_lib.dead_reckoning(x, odom_prev, odom)
        # particles = slam_lib.prediction(particles, odom_prev, odom)
        particles, weights, MAP = slam_lib.slam(particles, weights, l0[i], MAP, head, odom_prev, odom)
        odom_prev = odom

        # x_array[:,i] = x
        x_array[:, i * particle_num:(i + 1) * particle_num] = particles

        remaintime = round((timeit.default_timer() - timer) * (n - i) / 60)
        print str(round(i * 100.0 / n, 1)) + "%" + ' remain: ' + str(remaintime) + ' mins'
        timer = timeit.default_timer()

    print "done in " + str(round(timeit.default_timer() - start, 2)) + ' seconds'

    data = [MAP, x_array]
    with open('slam_data', 'wb') as sd:
        pickle.dump(data, sd)
