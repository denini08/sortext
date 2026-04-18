import random
import numpy as np

class Kmeans:
    '''
    Responsible for all k-means clustering functionality.

    Methods:
        initialize_centroids(matrix: np.ndarray, number_of_centroids: int): Creates initial centroids for each cluster.
        euclidean_distance(centroid: np.ndarray, document: np.ndarray): Calculates the euclidean distance between a document and centroid.
        get_clusters(distances: np.ndarray): Returns an array that indicates which clusters the documents belong to.
        calculate_distances(matrix: np.ndarray, distances: np.ndarray, centroid_coordinates: np.ndarray): Returns an array that
        that indicates the distances of all the documents to all centroids.
        calculate_new_centroid_coordinates(matrix: np.ndarray, clusters: np.ndarray, centroid_coordinates: np.ndarray): Calculates new
        centroid coordinates using the averages of each document that is included in the specific cluster.
        run_k_means(matrix: np.ndarray, centroid_coordinates: np.ndarray, distances: np.ndarray, max_iterations: int): Runs the K-means
        algorithm.
    '''

    def initialize_centroids(self, matrix: np.ndarray, number_of_centroids: int, random_state: int = None):
        '''
        Initializes centroids for clusters. The first centroid is selected randomly from the documents. The next centroids will
        be given the coordinates of the farthest document from other centroids.

        Args:
            matrix (np.ndarray): TF-IDF matrix.
            number_of_centroids (int): An integer indicating the desired number of clusters.
            random_state (int, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            centroid_coordinates (np.ndarray): An array of centroid coordinates. Rows are centroids, columns are terms.
            distances (np.ndarray): An array of distances between documents and centroids. Rows are centroids, columns are documents.
        '''

        random.seed(random_state)
        matrix_shape = matrix.shape
        centroid_coordinates = np.zeros((number_of_centroids, matrix_shape[1]))#Rows are centroids, columns are terms
        distances = np.zeros((number_of_centroids, matrix_shape[0]))#Rows are centroids, columns are documents
        centroid_coordinates[0] = matrix[random.randint(0, matrix_shape[0]-1)]

        #Calculate the document distances for the first centroid
        document_index = 0
        for document in matrix:
            distances[0, document_index] = self.euclidean_distance(centroid_coordinates[0], document)
            document_index += 1

        #Add the rest of the centroids using k-means++
        for centroid_index in range(1,number_of_centroids):
            farthest_document = np.argmax(np.amin(distances[:centroid_index], axis=0))
            centroid_coordinates[centroid_index] = matrix[farthest_document]

            #Calculate the document distances for each new centroid
            document_index = 0
            for document in matrix:
                distances[centroid_index, document_index] = self.euclidean_distance(centroid_coordinates[centroid_index], document)
                document_index += 1

        return centroid_coordinates, distances

    def euclidean_distance(self, centroid: np.ndarray, document: np.ndarray):
        return np.sqrt(np.sum((centroid - document)**2))

    def get_clusters(self, distances: np.ndarray):
        return np.argmin(distances, axis=0)

    def calculate_distances(self, matrix: np.ndarray, distances: np.ndarray, centroid_coordinates: np.ndarray):
        for centroid_index in range(centroid_coordinates.shape[0]):
            document_index = 0
            for document in matrix:
                distances[centroid_index, document_index] = self.euclidean_distance(centroid_coordinates[centroid_index], document)
                document_index += 1
        return distances

    def calculate_new_centroid_coordinates(self, matrix: np.ndarray, clusters: np.ndarray, centroid_coordinates: np.ndarray):
        centroid_coordinates_shape = centroid_coordinates.shape
        for centroid_index in range(centroid_coordinates_shape[0]):
            cluster_document_indices = np.where(clusters == centroid_index)
            for term_index in range(centroid_coordinates_shape[1]):
                centroid_coordinates[centroid_index, term_index] = np.mean(matrix[[cluster_document_indices], term_index])
        return centroid_coordinates

    def run_k_means(self, matrix: np.ndarray, centroid_coordinates: np.ndarray, distances: np.ndarray, max_iterations: int):
        '''
        Runs the K-means algorithm until convergence or max_iterations is reached.

        Args:
            matrix (np.ndarray): TF-IDF array of the dataset.
            centroid_coordinates (np.ndarray): The coordinates of all the centroids. Rows are centroids and columns terms.
            distances (np.ndarray): Distances of all documents to all centroids. Rows are centroids and columns documents.
            max_iterations (int): Maximum number of iterations of the algorithm if convergence is not yet reached.

        Returns:
            centroid_coordinates (np.ndarray): The coordinates of all the centroids. Rows are centroids and columns terms.
            clusters (np.ndarray): An array that indicates which cluster each document belongs to. Columns are documents and values clusters.
        '''

        iterations = 0
        while iterations < max_iterations:
            previous_centroid_coordinates = centroid_coordinates.copy()
            clusters = self.get_clusters(distances)
            centroid_coordinates = self.calculate_new_centroid_coordinates(matrix, clusters, centroid_coordinates)
            if np.array_equal(centroid_coordinates, previous_centroid_coordinates):
                print(f'Centroid coordinates unchanged at iteration {iterations+1}. Converged.')
                break
            distances = self.calculate_distances(matrix, distances, centroid_coordinates)
            iterations += 1
        clusters = self.get_clusters(distances)
        return centroid_coordinates, clusters
