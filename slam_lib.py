import numpy as np
import transforms3d
import p3_utils
import matplotlib.pyplot as plt


def texture_mapping(rgb, depth, pose, MAP, exIR_RGB, IRCalib, RGBCalib):
    img = rgb['image']  # read image
    dimg = np.multiply(depth['depth'], 0.001)  # millimeter to meter
    width = np.asscalar(rgb['width'])
    height = np.asscalar(rgb['height'])
    width_d = np.asscalar(depth['width'])
    height_d = np.asscalar(depth['height'])
    yaw = rgb['head_angles'][0, 0]
    pitch = rgb['head_angles'][0, 1]
    head = [pitch, yaw]  # get head angles

    Kd = np.matrix(camera_matrix(IRCalib))  # get depth camera matrix
    Kr = np.matrix(camera_matrix(RGBCalib))  # get rgb camera matrix
    T_ex = np.matrix(np.zeros((4, 4)))  # transformation from IR to RGB
    T_ex[:3, :3] = exIR_RGB['rgb_R_ir']
    T_ex[:3, 3] = np.matrix(exIR_RGB['rgb_T_ir']).T
    T_ex[3, 3] = 1
    Kd_inv = np.matrix(np.linalg.inv(Kd))  # inverse of depth camera matrix

    pixels = []  # structure the u,v and depth for pixels
    for u in range(width_d):
        for v in range(height_d):
            if np.logical_and((dimg[v, u] < 30), (dimg[v, u] > 0.1)):
                pixels.append([dimg[v, u] * u, dimg[v, u] * v, dimg[v, u]])
    pixels = np.matrix(pixels).T

    point_cloud = Kd_inv * pixels  # from pixels to point cloud
    point_cloud = np.vstack((point_cloud, np.ones((1, np.shape(point_cloud)[1]))))

    point_cloud = T_ex * point_cloud  # point cloud in RGB frame
    dimg_in_rgb = Kr * point_cloud[0:3, :]  # point cloud to RGB pixel frame
    dimg_in_rgb = np.divide(dimg_in_rgb, dimg_in_rgb[2, :])

    point_cloud = np.matrix(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]) * point_cloud  # optical frame to real-world frame

    p2, R2 = T_l2b(head, 0.4)
    p1, R1 = pose_transform(pose)
    p, R = smart_plus(p1, R1, p2, R2)  # body to world frame

    T = np.matrix(np.zeros((4, 4)))
    T[:3, :3] = R
    T[:3, 3] = np.matrix(p).T
    T[3, 3] = 1

    point_cloud = T * point_cloud  # point cloud in world frame

    # valid point cloud in grid cell
    xi0 = np.ceil((point_cloud[0, :] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    yi0 = np.ceil((point_cloud[1, :] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1
    indGood = np.logical_and(np.logical_and(np.logical_and((xi0 > 1), (yi0 > 1)), (xi0 < MAP['sizex'])),
                             (yi0 < MAP['sizey']))

    # point cloud hits ground plane
    ground_plane = np.logical_and(indGood, np.logical_and((point_cloud[2] >= -1.5), (point_cloud[2] <= 1.5)))

    point_cloud = np.vstack((point_cloud[0, :][ground_plane], point_cloud[1, :][ground_plane],
                             point_cloud[2, :][ground_plane], point_cloud[3, :][ground_plane]))

    dimg_in_rgb = np.vstack((dimg_in_rgb[0, :][ground_plane], dimg_in_rgb[1, :][ground_plane],
                             dimg_in_rgb[2, :][ground_plane]))

    xi0 = xi0[ground_plane]
    yi0 = yi0[ground_plane]

    # print 'len' + str(np.shape(xi0))
    cnt = 0
    for i in range(np.shape(point_cloud)[1]):
        xi = xi0[0, i]
        yi = yi0[0, i]

        # add valid rgb to texture map for corresponding ground plane points
        if np.all(MAP['tmap'][xi, yi, :] == np.array([255, 255, 255])):
            u = int(dimg_in_rgb[0, i])
            v = int(dimg_in_rgb[1, i])
            if np.logical_and(np.logical_and(u >= 0, u < width), np.logical_and(v >= 0, v < height)):
                MAP['tmap'][xi, yi, :] = img[v, u, :]
                cnt += 1

    # print 'cnt ' + str(cnt)
    return MAP


# camera matrix K
def camera_matrix(calib):
    fc = calib['fc']
    alpha_c = calib['ac']
    cc = calib['cc']
    K = np.array([[fc[0], alpha_c * fc[0], cc[0]], [0, fc[1], cc[1]], [0, 0, 1]])
    return K


# SLAM with prediction every iteration, update every 100 iterations
def slam(particles, weights, lidar, MAP, head, odom_prev, odom, i_best, iter):
    particles = prediction(particles, odom_prev, odom)
    if iter % 100 == 0:
        particles, weights, new_best = update(particles, weights, lidar, MAP, head)
        MAP = mapping(MAP, lidar, head, particles[:, i_best])
    else:
        new_best = i_best
    return particles, weights, MAP, new_best


# resampling using stratified resampling
def resampling(particles, weights, i_best):
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
        if np.all(newparticles[:, k] == particles[:, i_best]):
            new_best = k
    return newparticles, newweights, new_best


def update(particles, weights, lidar, MAP, head):
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
    yaw_range = np.arange(-0.2, 0.2 + 0.05, 0.05)

    # convert log-odds maps to occupancy map
    converted_map = 1 - np.power(1 + np.exp(MAP['map']), -1)
    binary_map = converted_map
    binary_map[converted_map > 0.6] = 1
    binary_map[np.logical_and(converted_map <= 0.6, converted_map >= 0.4)] = 0
    binary_map[converted_map < 0.4] = -1

    for i in range(n):
        pose = particles[:, i]
        c_array = []
        pose_array = []
        for w_yaw in yaw_range:
            # position with yaw variation
            xs1, ys1 = Cartesian2World(xs0, ys0, head, pose + [0, 0, w_yaw])

            # convert position in the map frame here
            Y = np.concatenate([np.concatenate([xs1, ys1], axis=0), np.zeros(xs1.shape)], axis=0)

            # find the max of map correlation, save the pose
            c = p3_utils.mapCorrelation(binary_map.astype(np.int8), x_im, y_im, Y[0:3, :], x_range, y_range)
            c_array.append(np.amax(c))
            ix, iy = np.unravel_index(np.argmax(c), (9, 9))
            pose_array.append(pose + [x_range[ix], y_range[iy], w_yaw])
        corr_array[i] = np.amax(c_array)
        # shift the pose to the location with highest map correlation
        particles[:, i] = pose_array[np.argmax(c_array)]

    # update the weights
    weights += corr_array
    weights = weights - np.amax(weights) - np.log(np.sum(np.exp(weights - np.amax(weights))))
    N_eff = 1 / np.sum(np.power(np.exp(weights), 2))

    # save the index of particle with highest weight
    i_best = np.argmax(weights)

    # resampling and find the new index of that best particle
    if N_eff < n / 2:
        particles, weights, i_best = resampling(particles, weights, i_best)
    return particles, weights, i_best


def prediction(particles, odom_prev, odom):
    n = np.shape(particles)[1]
    p2, R2 = pose_transform(odom)
    p1, R1 = pose_transform(odom_prev)
    p_u, R_u = smart_minus(p1, R1, p2, R2)  # u_t
    noise = np.random.randn(n, 3)
    for i in range(n):
        sigma = np.array([0.0001, 0.0001, 0.0005])
        w = sigma * noise[i, :]  # add noise from Gaussian
        p_x, R_x = pose_transform(particles[:, i] + w)
        p, R = smart_plus(p_x, R_x, p_u, R_u)  # xt to xt+1
        roll, pitch, yaw = transforms3d.euler.mat2euler(R, axes='sxyz')
        particles[:, i] = np.array([p[0], p[1], yaw])  # update particle location
    return particles


# simple dead reckoning
def dead_reckoning(x_prev, odom_prev, odom):
    p2, R2 = pose_transform(odom)
    p1, R1 = pose_transform(odom_prev)
    p_u, R_u = smart_minus(p1, R1, p2, R2)
    p_x, R_x = pose_transform(x_prev)
    p, R = smart_plus(p_x, R_x, p_u, R_u)
    roll, pitch, yaw = transforms3d.euler.mat2euler(R, axes='sxyz')
    return np.array([p[0], p[1], yaw])


# init texture map in MAP based on the occupancy grid map
def init_texture_MAP(MAP):
    MAP['tmap'] = np.zeros((MAP['sizex'], MAP['sizey'], 3), dtype=np.uint8)
    converted_map = 1 - np.power(1 + np.exp(MAP['map']), -1)
    for x in range(MAP['sizex']):
        for y in range(MAP['sizey']):
            if converted_map[x, y] > 0.6:
                MAP['tmap'][x, y, :] = [0, 0, 0]
            elif converted_map[x, y] < 0.4:
                MAP['tmap'][x, y, :] = [255, 255, 255]
            else:
                MAP['tmap'][x, y, :] = [128, 128, 128]
    return MAP


def init_map():
    # init MAP
    MAP = {}
    MAP['res'] = 0.05  # meters
    MAP['xmin'] = -20  # meters
    MAP['ymin'] = -20
    MAP['xmax'] = 20
    MAP['ymax'] = 20
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=float)
    return MAP


# find the path in grid cells
def path_in_grid(x, y, MAP):
    gridr = map(int, (x - MAP['xmin']) / MAP['res'])
    gridc = map(int, (y - MAP['xmin']) / MAP['res'])
    return [gridr, gridc]


def mapping(MAP, lidar, head, pose):
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
    # accumulate log-odds of occupied considering the overlapping (reason of *2) of occupied in free cell array
    MAP['map'][xio[0][indOccupied[0]], yio[0][indOccupied[0]]] += np.log(4) * 2

    indFree = np.logical_and(np.logical_and(np.logical_and((xif > 1), (yif > 1)), (xif < MAP['sizex'])),
                             (yif < MAP['sizey']))
    MAP['map'][xif[0][indFree[0]], yif[0][indFree[0]]] += np.log(0.25)  # accumulate log-odds of free

    return MAP


def plot_results(MAP, x_array):
    binary_map = 1 - np.power(1 + np.exp(MAP['map']), -1)
    plt.imshow(binary_map, cmap="binary")  # plot occupancy grid
    [row, col] = path_in_grid(x_array[0, :], x_array[1, :], MAP)  # plot path
    plt.plot(col, row, 'b')
    plt.axis([0, MAP['sizex'], MAP['sizey'], 0])
    plt.show()


# find free cells using bresenham2D, (contain occupied cell, should be considered)
def free(x, y, cx, cy):
    cell = np.vstack(([], []))
    for j in range(np.shape(x)[1]):
        cell = np.hstack((cell, p3_utils.bresenham2D(cx, cy, x[0, j], y[0, j])))
    return np.array([cell[0]]).astype(np.int16), np.array([cell[1]]).astype(np.int16)


# sensor to body, default 0.48m height
def T_l2b(head, height=0.48):
    pitch, yaw = head
    R = transforms3d.euler.euler2mat(0, pitch, yaw, 'sxyz')
    p = np.array([0, 0, height])
    return p, R


# find head angles of the corresponding ts
def find_head_angles(j0, ts):
    i = closest_ts(j0['ts'], ts)
    yaw = j0['head_angles'][0][i]
    pitch = j0['head_angles'][1][i]
    return pitch, yaw


# pose (x,y,yaw) to p and R, default height 0.93m
def pose_transform(pose, height=0.93):
    x, y, theta = pose
    R = transforms3d.euler.euler2mat(0, 0, theta, 'sxyz')
    p = np.array([x, y, height])
    return p, R


# transform from sensor frame to world frame
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

    aboveGround = newpos[2] > 0.1

    return np.asarray(newpos[0][aboveGround]), np.asarray(newpos[1][aboveGround])


# smart plus
def smart_plus(p1, R1, p2, R2):
    p1 = np.matrix(p1).T
    p2 = np.matrix(p2).T
    R1 = np.matrix(R1)
    R2 = np.matrix(R2)

    p = p1 + R1 * p2
    R = R1 * R2
    return np.asarray(p.T)[0], np.asarray(R)


# smart minus
def smart_minus(p1, R1, p2, R2):
    p1 = np.matrix(p1).T
    p2 = np.matrix(p2).T
    R1 = np.matrix(R1)
    R2 = np.matrix(R2)

    p = R1.T * (p2 - p1)
    R = R1.T * R2
    return np.asarray(p.T)[0], np.asarray(R)


# find the index of closest timestamp
def closest_ts(joint_ts, ts):
    ref_ts = np.abs(np.matrix(joint_ts) - ts)
    return np.argmin(ref_ts)
