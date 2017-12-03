import numpy as np
import transforms3d
import p3_utils
import matplotlib.pyplot as plt
import timeit


def slam(particles, weights, lidar, MAP, head, odom_prev, odom, x_best):
    particles = prediction(particles, odom_prev, odom)
    particles, weights, new_best = update(particles, weights, lidar, MAP, head)
    MAP = mapping(MAP, lidar, head, x_best)
    return particles, weights, MAP, new_best


def resampling(particles, weights):
    n = np.shape(particles)[1]
    weights = np.exp(weights)
    newparticles = particles
    newweights = np.log(np.ones((1, n))[0] / n)
    j = 0
    c = weights[0]
    for k in range(0, n):
        u = np.random.uniform(0, 1.0 / n)
        b = u + k / n
        while b > c:
            j += 1
            c += weights[j]
        newparticles[:, k] = particles[:, j]
    return newparticles, newweights


def update(particles, weights, lidar, MAP, head):
    start = timeit.default_timer()

    # print 'update'
    n = np.shape(particles)[1]
    corr_array = np.zeros((1, n))[0]
    angles = np.array([np.arange(-135, 135.25, 0.25) * np.pi / 180.])
    ranges = np.double(lidar['scan'])

    # take valid indices
    indValid = np.logical_and((ranges < 30), (ranges > 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]

    # xy position in the sensor frame
    xs0 = np.array([ranges * np.cos(angles)])
    ys0 = np.array([ranges * np.sin(angles)])

    x_im = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])  # x-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res'])  # y-positions of each pixel of the map

    x_range = np.arange(-0.2, 0.2 + 0.05, 0.05)
    y_range = np.arange(-0.2, 0.2 + 0.05, 0.05)
    yaw_range = np.arange(-0.1, 0.1 + 0.05, 0.05)

    binary_map = 1 - np.power(1 + np.exp(MAP['map']), -1)
    binary_map[binary_map > 0.6] = 1
    binary_map[binary_map <= 0.6] = 0

    for i in range(n):
        pose = particles[:, i]
        c_array = []
        pose_array = []
        for w_yaw in yaw_range:
            newpose = pose + [0, 0, w_yaw]

            xs0, ys0 = Cartesian2World(xs0, ys0, head, newpose)

            # convert position in the map frame here
            Y = np.concatenate([np.concatenate([xs0, ys0], axis=0), np.zeros(xs0.shape)], axis=0)

            c = p3_utils.mapCorrelation(binary_map.astype(np.int8), x_im, y_im, Y[0:3, :], x_range, y_range)
            c_array.append(np.amax(c))
            # ix, iy = np.unravel_index(np.argmax(c), (9, 9))
            # pose_array.append(newpose + [x_range[ix], y_range[iy], 0])
        corr_array[i] = np.amax(c_array)
        particles[:, i] += [0, 0, yaw_range[np.argmax(c_array)]]
        # pose = pose_array[np.argmax(c_array)]
        # particles[:, i] = pose

    weights += corr_array
    weights = weights - np.amax(weights) - np.log(np.sum(np.exp(weights - np.amax(weights))))
    N_eff = 1 / np.sum(np.power(np.exp(weights), 2))

    x_best = particles[:, np.argmax(weights)]

    if N_eff < n * 0.7:
        particles, weights = resampling(particles, weights)
    print "update " + str(round(timeit.default_timer() - start, 3)) + ' seconds',
    return particles, weights, x_best


def prediction(particles, odom_prev, odom):
    p2, R2 = pose_transform(odom)
    p1, R1 = pose_transform(odom_prev)
    p_u, R_u = smart_minus(p1, R1, p2, R2)
    for i in range(np.shape(particles)[1]):
        sigma = np.array([0.0001, 0.0001, 0.0001])  # ([0.00005, 0.00005, 0.00001])
        w = sigma * np.random.randn(1, 3)[0]
        p_x, R_x = pose_transform(particles[:, i] + w)
        p, R = smart_plus(p_x, R_x, p_u, R_u)
        roll, pitch, yaw = transforms3d.euler.mat2euler(R, axes='sxyz')
        particles[:, i] = np.array([p[0], p[1], yaw])
    return particles


def dead_reckoning(x_prev, odom_prev, odom):
    p2, R2 = pose_transform(odom)
    p1, R1 = pose_transform(odom_prev)
    p_u, R_u = smart_minus(p1, R1, p2, R2)
    p_x, R_x = pose_transform(x_prev)
    p, R = smart_plus(p_x, R_x, p_u, R_u)
    roll, pitch, yaw = transforms3d.euler.mat2euler(R, axes='sxyz')
    return np.array([p[0], p[1], yaw])


def init_map(lidar, head, pose):
    # init MAP
    MAP = {}
    MAP['res'] = 0.05  # meters
    MAP['xmin'] = -20  # meters
    MAP['ymin'] = -20
    MAP['xmax'] = 20
    MAP['ymax'] = 20
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=float)  # DATA TYPE: char or int8
    MAP = mapping(MAP, lidar, head, pose)
    return MAP


def mapping(MAP, lidar, head, pose):
    start = timeit.default_timer()

    # print 'mapping'
    angles = np.array([np.arange(-135, 135.25, 0.25) * np.pi / 180.])
    ranges = np.double(lidar['scan'])

    # take valid indices
    indValid = np.logical_and((ranges < 30), (ranges > 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]

    # xy position in the sensor frame
    xs0 = np.array([ranges * np.cos(angles)])
    ys0 = np.array([ranges * np.sin(angles)])

    xs0, ys0 = Cartesian2World(xs0, ys0, head, pose)

    # convert from meters to cells
    xio = np.ceil((xs0 - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    yio = np.ceil((ys0 - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

    cx = np.ceil((pose[0] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    cy = np.ceil((pose[1] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

    # xy position of free cell
    xif, yif = free(xio, yio, cx, cy)

    # build an arbitrary map
    indOccupied = np.logical_and(np.logical_and(np.logical_and((xio > 1), (yio > 1)), (xio < MAP['sizex'])),
                                 (yio < MAP['sizey']))
    MAP['map'][xio[0][indOccupied[0]], yio[0][indOccupied[0]]] += np.log(4) * 2

    indFree = np.logical_and(np.logical_and(np.logical_and((xif > 1), (yif > 1)), (xif < MAP['sizex'])),
                             (yif < MAP['sizey']))
    MAP['map'][xif[0][indFree[0]], yif[0][indFree[0]]] += np.log(0.25)

    print "mapping " + str(round(timeit.default_timer() - start, 3)) + ' seconds',
    return MAP


def plot_map(MAP):
    binary_map = 1 - np.power(1 + np.exp(MAP['map']), -1)
    plt.imshow(binary_map, cmap="binary")
    plt.show()


def free(x, y, cx, cy):
    cell = np.vstack(([], []))
    for j in range(np.shape(x)[1]):
        cell = np.hstack((cell, p3_utils.bresenham2D(cx, cy, x[0, j], y[0, j])))
    return np.array([cell[0]]).astype(np.int16), np.array([cell[1]]).astype(np.int16)


def T_l2b(head):
    pitch, yaw = head
    R = transforms3d.euler.euler2mat(0, pitch, yaw, 'sxyz')
    p = np.array([0, 0, 0.48])
    return p, R


def find_head_angles(j0, ts):
    i = closest_ts(j0['ts'], ts)
    yaw = j0['head_angles'][0][i]
    pitch = j0['head_angles'][1][i]
    return pitch, yaw


def pose_transform(pose):
    x, y, theta = pose
    R = transforms3d.euler.euler2mat(0, 0, theta, 'sxyz')
    p = np.array([x, y, 0])
    return p, R


def Cartesian2World(x, y, head, pose):
    p2, R2 = T_l2b(head)
    p1, R1 = pose_transform(pose)
    p, R = smart_plus(p1, R1, p2, R2)

    T = np.matrix(np.zeros((4, 4)))
    T[:3, :3] = R
    T[:3, 3] = np.matrix(p).T
    T[3, 3] = 1

    z = np.zeros(np.shape(x))
    a = np.ones(np.shape(x))
    pos = np.matrix(np.vstack((x, y, z, a)))
    newpos = T * pos

    return np.asarray(newpos[0]), np.asarray(newpos[1])


def smart_plus(p1, R1, p2, R2):
    p1 = np.matrix(p1).T
    p2 = np.matrix(p2).T
    R1 = np.matrix(R1)
    R2 = np.matrix(R2)

    p = p1 + R1 * p2
    R = R1 * R2
    return np.asarray(p.T)[0], np.asarray(R)


def smart_minus(p1, R1, p2, R2):
    p1 = np.matrix(p1).T
    p2 = np.matrix(p2).T
    R1 = np.matrix(R1)
    R2 = np.matrix(R2)

    p = R1.T * (p2 - p1)
    R = R1.T * R2
    return np.asarray(p.T)[0], np.asarray(R)


def closest_ts(joint_ts, ts):
    ref_ts = np.abs(np.matrix(joint_ts) - ts)
    return np.argmin(ref_ts)
