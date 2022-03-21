# Basic searching algorithms
from collections import deque
import math
# Class for each node in the grid
class Node:
    def __init__(self, row, col, is_obs, h):
        self.row = row        # coordinate
        self.col = col        # coordinate
        self.is_obs = is_obs  # obstacle?
        self.g = None         # cost to come (previous g + moving cost)
        self.h = h            # heuristic
        self.cost = None      # total cost (depend on the algorithm)
        self.parent = None    # previous node


def bfs(grid, start, goal):
    '''Return a path found by BFS alogirhm
       and the number of steps it takes to find it.

    arguments:
    grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
           e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    start - The start node in the map. e.g. [0, 0]
    goal -  The goal node in the map. e.g. [2, 2]

    return:
    path -  A nested list that represents coordinates of each step (including start and goal node),
            with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
    steps - Number of steps it takes to find the final solution,
            i.e. the number of nodes visited before finding a path (including start and goal node)

    >>> from main import load_map
    >>> grid, start, goal = load_map('test_map.csv')
    >>> bfs_path, bfs_steps = bfs(grid, start, goal)
    It takes 10 steps to find a path using BFS
    >>> bfs_path
    [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]]
    '''
    ### YOUR CODE HERE ###
    found = False
    visit = set()
    q = deque()
    steps = 0
    path = []
    rows, cols = len(grid), len(grid[0])
    if grid[start[0]][start[1]] == 1:
        return path, steps

    s_node = Node(start[0], start[1], False, None)
    s_node.parent = None
    visit.add(tuple(start))
    q.append(s_node)

    while q:
        if not found:
            s_node = q.popleft()
            row, col = s_node.row, s_node.col
            if (row, col) == tuple(goal):
                found = True
                break
            directions = [[0, +1], [+1, 0], [0, -1], [-1, 0]]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                if (r, c) == tuple(goal):
                    found = True
                    new_node = Node(r, c, False, None)
                    new_node.parent = s_node
                    q.append(new_node)
                    steps += 1
                    break
                else:
                    if r in range(rows) and c in range(cols) and (r, c) not in visit and grid[r][c] == 0:
                        new_node = Node(r, c, False, None)
                        new_node.parent = s_node
                        visit.add((r, c))
                        q.append(new_node)
                        steps += 1

        else:
            break

    while s_node.parent is not None:
        path.append([s_node.row, s_node.col])
        s_node = s_node.parent
    path.append(start)
    path.reverse()
    path.append(goal)

    if found:
        steps += 1
        print(f"It takes {steps} steps to find a path using BFS")
    else:
        print("No path found")
    return path, steps


def _neighbour(grid, row, col):
    rows, cols = len(grid), len(grid[0])
    directions = [[0, +1], [+1, 0], [0, -1], [-1, 0]]
    n_list = []
    for dr, dc in directions:
        r, c = row + dr, col + dc
        if r in range(rows) and c in range(cols) and grid[r][c] == 0:
            n_list.append((r,c))
    #n_list.reverse()
    return n_list

def dfs(grid, start, goal):
    '''Return a path found by DFS alogirhm
       and the number of steps it takes to find it.

    arguments:
    grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
           e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    start - The start node in the map. e.g. [0, 0]
    goal -  The goal node in the map. e.g. [2, 2]

    return:
    path -  A nested list that represents coordinates of each step (including start and goal node),
            with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
    steps - Number of steps it takes to find the final solution,
            i.e. the number of nodes visited before finding a path (including start and goal node)

    >>> from main import load_map
    >>> grid, start, goal = load_map('test_map.csv')
    >>> dfs_path, dfs_steps = dfs(grid, start, goal)
    It takes 9 steps to find a path using DFS
    >>> dfs_path
    [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 3], [3, 3], [3, 2], [3, 1]]
    '''
    ### YOUR CODE HERE ###
    found = False
    visit = set()
    q = deque()
    steps = 0
    path = []
    #rows, cols = len(grid), len(grid[0])
    if grid[start[0]][start[1]] == 1:
        return path, steps

    s_node = Node(start[0], start[1], False, None)
    s_node.parent = None
    visit.add(tuple(start))
    while not found:
        row, col = s_node.row, s_node.col
        if (row, col) == tuple(goal):
            found = True
            break
        else:
            node_list = _neighbour(grid, row, col)
            node_list.reverse()
            for node in node_list:
                if node not in visit:
                    temp_object = Node(node[0], node[1], False, None)
                    temp_object.parent = s_node
                    q.append(temp_object)
            s_node = q.pop()
            visit.add((s_node.row, s_node.col))

    steps = len(visit)
    while s_node.parent is not None:
        path.append([s_node.row, s_node.col])
        s_node = s_node.parent
    path.append(start)
    path.reverse()

    if found:
        print(f"It takes {steps} steps to find a path using DFS")
    else:
        print("No path found")
    return path, steps

def dijkstra(grid, start, goal):
    '''Return a path found by Dijkstra alogirhm
       and the number of steps it takes to find it.

    arguments:
    grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
           e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    start - The start node in the map. e.g. [0, 0]
    goal -  The goal node in the map. e.g. [2, 2]

    return:
    path -  A nested list that represents coordinates of each step (including start and goal node),
            with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
    steps - Number of steps it takes to find the final solution,
            i.e. the number of nodes visited before finding a path (including start and goal node)

    >>> from main import load_map
    >>> grid, start, goal = load_map('test_map.csv')
    >>> dij_path, dij_steps = dijkstra(grid, start, goal)
    It takes 10 steps to find a path using Dijkstra
    >>> dij_path
    [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]]
    '''
    ### YOUR CODE HERE ###
    found = False
    visit = set()
    q = []
    steps = 0
    path = []
    # rows, cols = len(grid), len(grid[0])
    if grid[start[0]][start[1]] == 1:
        return path, steps

    s_node = Node(start[0], start[1], False, None)
    s_node.parent = None
    visit.add(tuple(start))
    s_node.g = 0
    #i , j, k = 1, 1, 1
    while not found:
        #i += 1
        #print(i, "While Loop")
        row, col = s_node.row, s_node.col
        if (row, col) == tuple(goal):
            #j += 1
            #print(j, "Goal Loop")
            found = True
            #visit.add((row, col))
            break
        else:
            for node in _neighbour(grid, row, col):
                #k += 1
                #print(k, "Visit Loop")
                if node not in visit:
                    temp_object = Node(node[0], node[1], False, None)
                    temp_object.parent = s_node
                    temp_object.g = s_node.g + 1
                    q.append((temp_object, temp_object.g))

            q.sort(key=lambda x: x[1])
            #print([[e[0].row, e[0].col] for e in q])
            priority_ = q.pop(0)
            s_node = priority_[0]
            visit.add((s_node.row, s_node.col))

    #print(visit)
    steps = len(visit)
    while s_node.parent is not None:
        path.append([s_node.row, s_node.col])
        s_node = s_node.parent
    path.append(start)
    path.reverse()
    #print(path)
    if found:
        print(f"It takes {steps} steps to find a path using Dijkstra")
    else:
        print("No path found")
    return path, steps

def astar(grid, start, goal):
    '''Return a path found by A* alogirhm
       and the number of steps it takes to find it.

    arguments:
    grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
           e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    start - The start node in the map. e.g. [0, 0]
    goal -  The goal node in the map. e.g. [2, 2]

    return:
    path -  A nested list that represents coordinates of each step (including start and goal node),
            with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
    steps - Number of steps it takes to find the final solution,
            i.e. the number of nodes visited before finding a path (including start and goal node)

    >>> from main import load_map
    >>> grid, start, goal = load_map('test_map.csv')
    >>> astar_path, astar_steps = astar(grid, start, goal)
    It takes 7 steps to find a path using A*
    >>> astar_path
    [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]]
    '''
    ### YOUR CODE HERE ###
    found = False
    visit = set()
    q = []
    steps = 0
    path = []
    # rows, cols = len(grid), len(grid[0])
    if grid[start[0]][start[1]] == 1:
        return path, steps

    s_node = Node(start[0], start[1], False, None)
    s_node.parent = None
    visit.add(tuple(start))
    s_node.g = 0
    while not found:
        row, col = s_node.row, s_node.col
        if (row, col) == tuple(goal):
            found = True
            # visit.add((row, col))
            break
        else:
            for node in _neighbour(grid, row, col):
                if node not in visit:
                    h = (abs(node[0] - goal[0]) + abs(node[1] - goal[1]))
                    temp_object = Node(node[0], node[1], False, h)
                    temp_object.parent = s_node
                    temp_object.g = s_node.g + 1
                    q.append((temp_object, temp_object.g + temp_object.h))
            q.sort(key=lambda x: x[1])
            #print([[e[0].row, e[0].col] for e in q])
            priority_ = q.pop(0)
            s_node = priority_[0]
            visit.add((s_node.row, s_node.col))
    #print(visit)
    steps = len(visit)
    while s_node.parent is not None:
        path.append([s_node.row, s_node.col])
        s_node = s_node.parent
    path.append(start)
    path.reverse()

    if found:
        print(f"It takes {steps} steps to find a path using A*")
    else:
        print("No path found")
    return path, steps

# Doctest
if __name__ == "__main__":
    # load doc test
    from doctest import testmod, run_docstring_examples
    # Test all the functions
    testmod()
