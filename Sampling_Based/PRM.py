# Standard Algorithm Implementation
# Sampling-based Algorithms PRM

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy import spatial

# Class for PRM
class PRM:
    # Constructor
    def __init__(self, map_array):
        self.map_array = map_array            # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]    # map size
        self.size_col = map_array.shape[1]    # map size

        self.samples = []                     # list of sampled points
        self.graph = nx.Graph()               # constructed graph
        self.path = []                        # list of nodes of the found path


    def check_collision(self, p1, p2):
        '''Check if the path between two points collide with obstacles
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            True if there are obstacles between two points
        '''
        ### YOUR CODE HERE ###
        pt_set = set()
        points = np.linspace(p1, p2, dtype=int)
        [pt_set.add(tuple(x)) for x in points if tuple(x) not in pt_set]
        for j in pt_set:
            if self.map_array[j[0]][j[1]] == 0:
                return True
        return False

    def dis(self, point1, point2):
        '''Calculate the euclidean distance between two points
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            euclidean distance between two points
        '''
        ### YOUR CODE HERE ###
        return np.sqrt((point1[1]-point2[1])**2 + (point1[0]-point2[0])**2)

    def distribution(self, n_pts, random):
        '''
        Function randomly/uniformly distributes points across the space
        '''
        w = self.size_row
        h = self.size_col
        n_y = int(np.sqrt(w * n_pts / h + ((w - h) ** 2) / 4 * (h ** 2)) - ((w - h) / 2 * h))
        n_x = int(n_pts / n_y)
        if not random:
            vec_x = np.linspace(0, w - 1, n_x, dtype=int)
            vec_y = np.linspace(0, h - 1, n_y, dtype=int)
            grid_x, grid_y = np.meshgrid(vec_x, vec_y)
            row_idx = grid_x.ravel()
            col_idx = grid_y.ravel()
            return zip(row_idx,col_idx)
        else:
            vec_x = np.random.randint(0, w - 1, n_x, dtype=int)
            vec_y = np.random.randint(0, h - 1, n_y, dtype=int)
            grid_x, grid_y = np.meshgrid(vec_x, vec_y)
            row_idx = grid_x.ravel()
            col_idx = grid_y.ravel()
            return zip(row_idx, col_idx)

    def uniform_sample(self, n_pts):
        '''Use uniform sampling and store valid points
        arguments:
            n_pts - number of points try to sample,
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()
        ### YOUR CODE HERE ###
        self.samples.append((0, 0))
        grid = self.distribution(n_pts, random=False)
        for g in grid:
            if self.map_array[g[0]][g[1]] == 1:
                self.samples.append(g)

    def random_sample(self, n_pts):
        '''Use random sampling and store valid points
        arguments:
            n_pts - number of points try to sample,
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()
        ### YOUR CODE HERE ###
        self.samples.append((0, 0))
        grid = self.distribution(n_pts, random=True)
        for g in grid:
            if self.map_array[g[0]][g[1]] == 1:
                self.samples.append(g)

    def gaussian_sample(self, n_pts):
        '''Use gaussian sampling and store valid points
        arguments:
            n_pts - number of points try to sample,
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()
        ### YOUR CODE HERE ###
        # self.samples.append((0, 0))
        grid = self.distribution(n_pts, random=True)
        for g in grid:
            if self.map_array[g[0]][g[1]] == 0:
                # obstacle points
                # normal distribution
                f_x = int(np.random.normal(g[0], 5))
                f_y = int(np.random.normal(g[1], 5))
                if 0 < f_x < self.size_row and 0 < f_y < self.size_col:
                    if self.map_array[f_x, f_y] == 1:
                        self.samples.append([f_x, f_y])
            else:
                # free points
                # normal distribution
                ob_x = int(np.random.normal(g[0], 5))
                ob_y = int(np.random.normal(g[1], 5))
                if 0 < ob_x < self.size_row and 0 < ob_y < self.size_col:
                    if self.map_array[ob_x, ob_y] == 0:
                        self.samples.append(g)


    def bridge_sample(self, n_pts):
        '''Use bridge sampling and store valid points
        arguments:
            n_pts - number of points try to sample,
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        grid = self.distribution(n_pts, random=True)
        for g in grid:
            # check obstacle
            if self.map_array[g[0]][g[1]] == 0:
                f_x = int(np.random.normal(g[0], 10))
                f_y = int(np.random.normal(g[1], 10))
                if 0 < f_x < self.size_row and 0 < f_y < self.size_col:
                    # check obstacle
                    if self.map_array[f_x, f_y] == 0:
                        # find middle point devoid of obstacle
                        mid_x = int((f_x + g[0])/2)
                        mid_y = int((f_y + g[1]) / 2)
                        if self.map_array[mid_x, mid_y] == 1:
                            self.samples.append([mid_x, mid_y])


    def draw_map(self):
        '''Visualization of the result
        '''
        # Create empty map
        fig, ax = plt.subplots()
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw graph
        # get node position (swap coordinates)
        node_pos = np.array(self.samples)[:, [1, 0]]
        pos = dict( zip( range( len(self.samples) ), node_pos) )
        pos['start'] = (self.samples[-2][1], self.samples[-2][0])
        pos['goal'] = (self.samples[-1][1], self.samples[-1][0])

        # draw constructed graph
        nx.draw(self.graph, pos, node_size=3, node_color='y', edge_color='y' ,ax=ax)

        # If found a path
        if self.path:
            # add temporary start and goal edge to the path
            final_path_edge = list(zip(self.path[:-1], self.path[1:]))
            nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=self.path, node_size=8, node_color='b')
            nx.draw_networkx_edges(self.graph, pos=pos, edgelist=final_path_edge, width=2, edge_color='b')

        # draw start and goal
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=['start'], node_size=12,  node_color='g')
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=['goal'], node_size=12,  node_color='r')

        # show image
        plt.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.show()

    def sample(self, n_pts=1000, sampling_method="uniform"):
        '''Construct a graph for PRM
        arguments:
            n_pts - number of points try to sample,
                    not the number of final sampled points
            sampling_method - name of the chosen sampling method

        Sample points, connect, and add nodes and edges to self.graph
        '''
        # Initialize before sampling
        self.samples = []
        self.graph.clear()
        self.path = []
        radius = 0
        # different radii for different methods
        # Sample methods
        if sampling_method == "uniform":
            self.uniform_sample(n_pts)
            radius = 20
        elif sampling_method == "random":
            self.random_sample(n_pts)
            radius = 35
        elif sampling_method == "gaussian":
            self.gaussian_sample(n_pts)
            radius = 30
        elif sampling_method == "bridge":
            self.bridge_sample(n_pts)
            radius = 70
        ### YOUR CODE HERE ###
        pairs = []
        node_list = []
        kdtree = spatial.KDTree(np.array(self.samples))
        # kd_tree pairs list
        kd_pairs = list(kdtree.query_pairs(radius))

        # All points in Kdtree are checked
        for query in kd_pairs:
            collision = self.check_collision(self.samples[query[0]], self.samples[query[1]])
            if not collision:
                weight = self.dis(self.samples[query[0]], self.samples[query[1]])
                if weight != 0:
                    pairs.append((query[0], query[1], weight))
                    node_list.append(query[0])
                    node_list.append(query[1])

        self.graph.add_nodes_from(node_list)
        self.graph.add_weighted_edges_from(pairs)

        # Print constructed graph information
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        print("The constructed graph has %d nodes and %d edges" % (n_nodes, n_edges))

    def search(self, start, goal):
        '''Search for a path in graph given start and goal location
        arguments:
            start - start point coordinate [row, col]
            goal - goal point coordinate [row, col]

        Temporary add start and goal node, edges of them and their nearest neighbors
        to graph for self.graph to search for a path.
        '''
        # Clear previous path
        self.path = []

        # Temporarily add start and goal to the graph
        self.samples.append(start)
        self.samples.append(goal)
        # start and goal id will be 'start' and 'goal' instead of some integer
        self.graph.add_nodes_from(['start', 'goal'])

        ### YOUR CODE HERE ###
        start_pairs = []
        goal_pairs = []
        # Setting up a goal radius
        radius = 100

        # Evaluation for start pairs:
        for i in range(len(self.samples)):
            # Calculating the starting point
            start = self.samples[len(self.samples) - 2]
            node = self.samples[i]
            if start != node:
                collision = self.check_collision(start, node)
                if not collision:
                    # calculate weight
                    weight = self.dis(start, node)
                    if weight != 0 and weight < radius:
                        start_pairs.append(('start', self.samples.index(node), weight))

        # Evaluation for goal pairs:
        for j in range(len(self.samples)):
            # Calculating the goal point
            goal = self.samples[len(self.samples) - 1]
            node = self.samples[j]
            if goal != node:
                collision = self.check_collision(goal, node)
                if not collision:
                    # calculate weight
                    weight = self.dis(goal, node)
                    if weight != 0 and weight < radius:
                        goal_pairs.append(('goal', self.samples.index(node), weight))

        self.graph.add_weighted_edges_from(start_pairs)
        self.graph.add_weighted_edges_from(goal_pairs)

        # Search using Dijkstra
        try:
            self.path = nx.algorithms.shortest_paths.weighted.dijkstra_path(self.graph, 'start', 'goal')
            path_length = nx.algorithms.shortest_paths.weighted.dijkstra_path_length(self.graph, 'start', 'goal')
            print("The path length is %.2f" % path_length)
        except nx.exception.NetworkXNoPath:
            print("No path found")

        # Draw result
        self.draw_map()

        # Remove start and goal node and their edges
        self.samples.pop(-1)
        self.samples.pop(-1)
        self.graph.remove_nodes_from(['start', 'goal'])
        self.graph.remove_edges_from(start_pairs)
        self.graph.remove_edges_from(goal_pairs)
