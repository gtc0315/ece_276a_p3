import slam_lib
import transforms3d
import numpy as np


def testing():
    R1 = transforms3d.euler.euler2mat(0, 0, slam_lib.deg2rad(30), 'sxyz')
    R2 = transforms3d.euler.euler2mat(0, 0, slam_lib.deg2rad(45), 'sxyz')
    p1 = np.array([1,0,0])
    p2 = np.array([1,0,0])
    p,R = slam_lib.smart_plus(p1,R1,p2,R2)
    print p
    print R
    print '\n'

    R1 = transforms3d.euler.euler2mat(0, slam_lib.deg2rad(30), slam_lib.deg2rad(30), 'sxyz')
    R2 = transforms3d.euler.euler2mat(0, slam_lib.deg2rad(20), slam_lib.deg2rad(45), 'sxyz')
    p1 = np.array([1,1,0])
    p2 = np.array([1,0,0])
    p,R = slam_lib.smart_plus(p1,R1,p2,R2)
    print p
    print R
    print '\n'
