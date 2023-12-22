import csv
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Read location data from CSV file
locations = []
with open('../../FINAL/FINAL_LATITUDE_LONGITUDE.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        locations.append([float(row[3]), float(row[4])])

num_locations = len(locations)

# Convert locations into a NumPy array
coordinates = np.array(locations)

# Calculate adjacency matrix using NearestNeighbors
nn = NearestNeighbors(n_neighbors=2, metric='haversine')
nn.fit(np.radians(coordinates))
distances, indices = nn.kneighbors()

# Generate adjacency matrix
adjacency_matrix = np.zeros((num_locations, num_locations))
for i in range(num_locations):
    nearest_neighbor = indices[i][1]  # Index 0 is the location itself, so index 1 is the nearest neighbor
    adjacency_matrix[i][nearest_neighbor] = 1

print(adjacency_matrix)
