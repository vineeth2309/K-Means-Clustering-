import numpy as np
import cv2
import math
import time
from sklearn import datasets
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class K_Means:
	def __init__(self, no_samples=1000, no_features = 2, no_classes = 6, clusters = 6, max_iterations = 10):
		self.no_classes = no_classes
		self.clusters = clusters
		self.max_iterations = max_iterations
		self.X1, self.Y1 = datasets.make_blobs(n_samples = no_samples, centers = self.no_classes, n_features = no_features, random_state = 0)
		self.cluster_centers = self.X1[np.random.randint(0, self.X1.shape[0], self.clusters)]
		self.dists = np.zeros((self.X1.shape[0], self.clusters))
		self.mean_dist = np.zeros(self.max_iterations)
		self.cmap = ['green', 'blue', 'yellow', 'black', 'orange', 'red', 'purple', 'magenta']

		for self.itr in range(self.max_iterations):
			self.calc_distance_to_cluster_centers()
			self.assign_clusters_and_update_centers_and_calc_mean_dist()
		self.plot_data()

	def calc_distance_to_cluster_centers(self):
		''' dist = ((x1-x2)**2 + (y1-y2)**2)**0.5 '''
		for i in range(self.clusters):
			self.dists[:, i] = np.sum((self.X1- self.cluster_centers[i])**2, axis=1)**0.5

	def assign_clusters_and_update_centers_and_calc_mean_dist(self):
		''' 
			1) Each point is assigned to the closest cluster center
			2) Update the cluster center to be the mean of the cluster
		'''
		self.cluster_idx = np.argmin(self.dists, axis=1)
		self.dist = []
		for i in range(self.clusters):
			self.cluster_centers[i] = np.mean(self.X1[np.where(self.cluster_idx==i)], axis=0)
			self.dist.append(np.mean(self.dists[np.where(self.cluster_idx==i), i]))
		self.mean_dist[self.itr] = np.mean(np.array(self.dist).reshape(-1,1))

	def plot_data(self):
		fig, axs = plt.subplots(3)
		for i in range(self.clusters):
			cluster = self.X1[np.where(self.cluster_idx==i)]
			axs[0].scatter(cluster[:,0], cluster[:,1], color=self.cmap[i])
			axs[0].scatter(self.cluster_centers[i,0], self.cluster_centers[i,1], color=self.cmap[-1])
		axs[1].scatter(self.X1[:,0], self.X1[:,1], c = self.Y1)
		axs[2].plot(self.mean_dist, color='black')
		plt.show()


if __name__ == "__main__":
	obj = K_Means()
	# obj.plot_data(obj.X1, obj.Y1)