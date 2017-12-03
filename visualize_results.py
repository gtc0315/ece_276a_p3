import pickle
import slam_lib
import matplotlib.pyplot as plt


if __name__ == '__main__':
    with open('slam_data', 'rb') as sd:
        [MAP, x_array] = pickle.load(sd)

    slam_lib.plot_map(MAP)
    plt.plot(x_array[0, :], x_array[1, :])
    plt.show()