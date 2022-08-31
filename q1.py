import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def points_file_reader(file_name, output_name='./res01.jpg'):
    """
    This function takes the name of the file containing coordinates of the points and returns the points as an array.
    :param file_name: name of the file containing coordinates of the points
    :param output_name: the directory where the scatter plot of the points should be saved.
    :return: a numeric array containing coordinates of the points
    """
    with open(file_name, 'r') as f:
        points_str = f.readlines()  # Reads all the lines at once

    # The first line is the number of points:
    n_points = points_str[0]
    # Remove the next line character
    n_points = int(n_points[:-1])
    # Separate coordinates by space and assign store them in a numpy array with shape = (n_points, dim)
    dim = len(points_str[2].split(' '))
    points = np.zeros((n_points, dim))
    points_str = points_str[1:]
    for i in range(n_points):
        point_i = points_str[i].split(' ')
        for j in range(dim):
            points[i, j] = float(point_i[j])

    # Showing the points and saving the figure
    fig = plt.figure()
    plt.scatter(points[:, 0], points[:, 1])
    plt.title("Scatter Plot of the Input Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(output_name)
    plt.close(fig)  # Comment this line to show the plot
    plt.show()
    return points


def k_means(points, k, output_file='./k-means.jpg', mapping=False, initial_centers=None, tol=0.0001, max_iter=300):
    """
    This function takes some points and uses k-means algorithm to cluster them.
    :param points: input data points
    :param k: number of clusters
    :param output_file: the directory where the scatter plot of the clustered points should be saved.
    :param mapping: Determines whether or not data points should be mapped.
    :param initial_centers: an array containing initial points if provided. Otherwise, set as None type.
    :param tol: if L2 norm of the means' difference in two consecutive iterations is less than tol, the algorithm will
     stop.
    :param max_iter: maximum allowable number of iterations.
    :return: labels and centers
    """
    # Find number of given points
    n_points = points.shape[0]
    # If user hasn't provided initial centers, initialize them randomly.
    if initial_centers is None:
        # Choose k points randomly. Note that replace must be false.
        idx = np.random.choice(np.arange(n_points), size=k, replace=False)
        initial_centers = points[idx, :].copy()
    # Save initializing points for future plotting
    random_initial_centers = initial_centers.copy()
    # Mapping input points
    points_holder = points.copy()
    if mapping:
        points_center = np.mean(points, axis=0)
        # points_center = np.array([[0, 0]])
        points = np.sqrt(np.sum(np.square(points - points_center), axis=1)).reshape((-1, 1))
        initial_centers = np.sqrt(np.sum(np.square(initial_centers - points_center), axis=1)).reshape((-1, 1))

    # This is just to avoid an error in case max_iter <= 0
    clusters = np.ones((n_points, 1))

    for _ in range(max_iter):
        # Preallocate a distance array. This array is of shape (n_points, k). i-th column of this array is the distance
        # between all points and i-th center.
        distance_array = np.zeros((n_points, k))
        for i in range(k):
            distance_array[:, i] = np.sqrt(np.sum(np.square(points - initial_centers[i, :]), axis=1))
        # For each data point set the cluster label as the argmin of distance with all centers
        clusters = np.argmin(distance_array, axis=1)
        # Save current centers to check stopping criterion
        prev_center = initial_centers.copy()
        for i in range(k):
            idx_i = clusters == i
            # Update mean of each cluster
            initial_centers[i, :] = np.mean(points[idx_i, :], axis=0)
        # If means' difference L2 norm is less than tolerance stop fitting
        if np.sqrt(np.sum(np.square(initial_centers - prev_center))) < tol:
            break

    # Uncomment the following lines to check your answers with sklearn,KMeans result.
    # sklearn_clusters = KMeans(n_clusters=k, init=initial_centers, n_init=1).fit(points)
    # # Return to R^n
    # points = points_holder.copy()
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.scatter(points[:, 0], points[:, 1], s=40, c=sklearn_clusters.labels_, cmap='PiYG')
    # plt.scatter(random_initial_centers[:, 0],
    #             random_initial_centers[:, 1],
    #             s=100,
    #             c=np.arange(k),
    #             cmap='bwr',
    #             marker='*')
    # plt.legend(["Clustered Data Points", "Initial Cluster Centers"])
    # plt.title("sklearn.KMeans")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    #
    # plt.subplot(1, 2, 2)
    # plt.scatter(points[:, 0], points[:, 1], s=40, c=clusters, cmap='PiYG')
    # plt.scatter(random_initial_centers[:, 0],
    #             random_initial_centers[:, 1],
    #             s=100,
    #             c=np.arange(k),
    #             cmap='bwr',
    #             marker='*')
    # plt.legend(["Clustered Data Points", "Initial Cluster Centers"])
    # plt.title("Clustered Date Points by K-Means with Initial Centers")
    # plt.xlabel("X")
    # plt.ylabel("Y")



    # Save scatter plots
    # Use c and cmap parameters of scatter to set a different color for each cluster.
    points = points_holder.copy()
    fig = plt.figure()
    plt.scatter(points[:, 0], points[:, 1], s=40, c=clusters, cmap='PiYG')
    plt.scatter(random_initial_centers[:, 0],
                random_initial_centers[:, 1],
                s=100,
                c=np.arange(k),
                cmap='bwr',
                marker='*')
    plt.legend(["Clustered Data Points", "Initial Cluster Centers"])
    plt.title("Clustered Date Points by K-Means with Initial Centers")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(output_file)
    plt.close(fig)  # Comment this line to show the plot
    plt.show()
    return clusters, initial_centers


f_name = "./Points.txt"
points_arr = points_file_reader(f_name, './res01.jpg')
k_means(points_arr, 2, './res02.jpg')
k_means(points_arr, 2, './res03.jpg')
k_means(points_arr, 2, './res04.jpg', mapping=True)
