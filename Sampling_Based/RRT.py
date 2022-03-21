# Standard Algorithm Implementation
# Sampling-based Algorithms RRT and RRT*

import matplotlib.pyplot as plt
import numpy as np
import math
# import bresenham

# Class for each tree node
class Node:
    def __init__(self, row, col):
        self.row = row        # coordinate
        self.col = col        # coordinate
        self.parent = None    # parent node
        self.cost = 0.0       # cost


# Class for RRT
class RRT:
    # Constructor
    def __init__(self, map_array, start, goal):
        self.map_array = map_array            # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]    # map size
        self.size_col = map_array.shape[1]    # map size

        self.start = Node(start[0], start[1]) # start node
        self.goal = Node(goal[0], goal[1])    # goal node
        self.vertices = []                    # list of nodes
        self.found = False                    # found flag


    def init_map(self):
        '''Intialize the map before each search
        '''
        self.found = False
        self.vertices = []
        self.vertices.append(self.start)


    def dis(self, node1, node2):
        '''Calculate the euclidean distance between two nodes
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            euclidean distance between two nodes
        '''
        ### YOUR CODE HERE ###
        return np.sqrt((node1.row - node2.row)**2 + (node1.col - node2.col)**2)


    def check_collision(self, node1, node2):
        '''Check if the path between two nodes collide with obstacles
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            True if the new node is valid to be connected
        '''
        ### YOUR CODE HERE ###
        # x1, y1 = node1.row, node1.col
        # x2, y2 = node2.row, node2.col
        # points = int(list(bresenham.bresenham(x1, y1, x2, y2)))
        points = np.linspace([node1.row, node1.col], [node2.row, node2.col], dtype=int)
        for p in points:
            x, y = p
            # colliding condition
            if self.map_array[x][y] == 0:
                return False
        return True


    def get_new_point(self, goal_bias):
        '''Choose the goal or generate a random point
        arguments:
            goal_bias - the possibility of choosing the goal instead of a random point

        return:
            point - the new point
        '''
        ### YOUR CODE HERE ###
        rand_row = np.random.randint(0, self.size_row - 1)
        rand_col = np.random.randint(0, self.size_col - 1)
        rand_node = Node(rand_row, rand_col)
        # random point probability
        if np.random.rand() < goal_bias:
            return self.goal
        else:
            return rand_node


    def get_nearest_node(self, point):
        '''Find the nearest node in self.vertices with respect to the new point
        arguments:
            point - the new point

        return:
            the nearest node
        '''
        ### YOUR CODE HERE ###
        minimum = 999999
        nearest_node = self.vertices[0]
        #  find minimum of all nodes
        for node in self.vertices:
            if self.dis(node, point) < minimum:
                nearest_node = node
                minimum = self.dis(node, point)
        return nearest_node

    def get_neighbors(self, new_node, neighbor_size):
        '''Get the neighbors that are within the neighbor distance from the node
        arguments:
            new_node - a new node
            neighbor_size - the neighbor distance

        return:
            neighbors - a list of neighbors that are within the neighbor distance
        '''
        ### YOUR CODE HERE ###
        neighbours = []
        for node in self.vertices:
            # distance condition
            if self.dis(node, new_node) < neighbor_size:
                # collision condition
                if self.check_collision(node, new_node):
                    neighbours.append(node)
        return neighbours
        # return [self.vertices[0]]


    def rewire(self, new_node, neighbors):
        '''Rewire the new node and all its neighbors
        arguments:
            new_node - the new node
            neighbors - a list of neighbors that are within the neighbor distance from the node

        Rewire the new node if connecting to a new neighbor node will give least cost.
        Rewire all the other neighbor nodes.
        '''
        ### YOUR CODE HERE ###
        distances = []
        for n in neighbors:
            distances.append(n.cost + self.dis(n, new_node))
        # get the best neighbour distance
        closest_neigh = neighbors[np.argmin(distances)]
        # make parent
        new_node.parent = closest_neigh
        # change cost
        new_node.cost = closest_neigh.cost + self.dis(new_node, closest_neigh)
        neighbors.remove(closest_neigh)
        # rewire other neighbours
        for n in neighbors:
            dist_new_node = new_node.cost + self.dis(new_node, n)
            if n.cost > dist_new_node and not self.check_collision(new_node, n):
                self.vertices.remove(n)
                n.parent = new_node
                n.cost = new_node.cost + dist_new_node
                self.vertices.append(n)
        self.vertices.append(new_node)



    def draw_map(self):
        '''Visualization of the result
        '''
        # Create empty map
        fig, ax = plt.subplots(1)
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw Trees or Sample points
        for node in self.vertices[1:-1]:
            plt.plot(node.col, node.row, markersize=3, marker='o', color='y')
            plt.plot([node.col, node.parent.col], [node.row, node.parent.row], color='y')

        # Draw Final Path if found
        if self.found:
            cur = self.goal
            while cur.col != self.start.col or cur.row != self.start.row:
                plt.plot([cur.col, cur.parent.col], [cur.row, cur.parent.row], color='b')
                cur = cur.parent
                plt.plot(cur.col, cur.row, markersize=3, marker='o', color='b')

        # Draw start and goal
        plt.plot(self.start.col, self.start.row, markersize=5, marker='o', color='g')
        plt.plot(self.goal.col, self.goal.row, markersize=5, marker='o', color='r')

        # show image
        plt.show()


    def RRT(self, n_pts=1000):
        '''RRT main search function
        arguments:
            n_pts - number of points try to sample,
                    not the number of final sampled points

        In each step, extend a new node if possible, and check if reached the goal
        '''
        # Remove previous result
        self.init_map()
        ### YOUR CODE HERE ###
        # In each step,
        goal_bias = 0.05
        for i in range(n_pts):
            # get a new point,
            curr_node = self.get_new_point(goal_bias)
            # get its nearest node,
            near_node = self.get_nearest_node(curr_node)
            # extension
            extend_dis = 10
            theta = math.atan2((curr_node.col - near_node.col), (curr_node.row - near_node.row))
            # extend the node and check collision to decide whether to add or drop,
            new_row = int(near_node.row + extend_dis * (math.cos(theta)))
            new_col = int(near_node.col + extend_dis * (math.sin(theta)))
            new_node = Node(new_row, new_col)
            if (0 <= new_row < self.size_row) and (0 <= new_col < self.size_col) \
                    and self.check_collision(new_node, near_node):
                new_node.parent = near_node
                new_node.cost = extend_dis + near_node.cost
                self.vertices.append(new_node)
                if not self.found:
                    short_dis = self.dis(self.goal, new_node)
                    # and check if reach the neighbor region of the goal if the path is not found.
                    if short_dis < extend_dis and self.check_collision(self.goal, new_node):  # in neighbourhood
                        self.goal.parent = new_node
                        self.found = True
                        self.goal.cost = short_dis + new_node.cost
                        self.vertices.append(self.goal)
                        break
        # Output
        if self.found:
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" %steps)
            print("The path length is %.2f" %length)
        else:
            print("No path found")

        # Draw result
        self.draw_map()


    def RRT_star(self, n_pts=1000, neighbor_size=40):
        '''RRT* search function
        arguments:
            n_pts - number of points try to sample,
                    not the number of final sampled points
            neighbor_size - the neighbor distance

        In each step, extend a new node if possible, and rewire the node and its neighbors
        '''
        # Remove previous result
        self.init_map()

        ### YOUR CODE HERE ###
        # In each step,
        goal_bias = 0.05
        for i in range(n_pts):
            # get a new point,
            curr_node = self.get_new_point(goal_bias)
            # get its nearest node,
            near_node = self.get_nearest_node(curr_node)
            # extension
            extend_dis = 10
            theta = math.atan2((curr_node.col - near_node.col), (curr_node.row - near_node.row))
            # extend the node and check collision to decide whether to add or drop,
            new_row = int(near_node.row + extend_dis * (math.cos(theta)))
            new_col = int(near_node.col + extend_dis * (math.sin(theta)))
            new_node = Node(new_row, new_col)
            if (0 <= new_row < self.size_row) and (0 <= new_col < self.size_col) \
                    and self.check_collision(new_node, near_node):
                # find neighbours
                neighbours = self.get_neighbors(new_node, neighbor_size)
                # rewire them
                self.rewire(new_node, neighbours)
                if not self.found:
                    short_dis = self.dis(self.goal, new_node)
                    # and check if reach the neighbor region of the goal if the path is not found.
                    if short_dis < extend_dis and self.check_collision(self.goal, new_node):  # in neighbourhood
                        self.goal.parent = new_node
                        self.found = True
                        self.goal.cost = short_dis + new_node.cost
                        self.vertices.append(self.goal)

        # Output
        if self.found:
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" %steps)
            print("The path length is %.2f" %length)
        else:
            print("No path found")

        # Draw result
        self.draw_map()
