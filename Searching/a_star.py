def euclidean_dist_heuristic(graph, v, goal):
    """
   Implement the euclidean distance heuristic.
    Args:
        graph (AjacentList): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    fst = graph.nodes[v]
    lst = graph.nodes[goal]
    fst_x = fst['pos'][0]
    fst_y = fst['pos'][1]
    lst_x = lst['pos'][0]
    lst_y = lst['pos'][1]
    dist = math.sqrt((fst_x-lst_x)**2 + (fst_y-lst_y)**2)
    return dist

def A_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Implement A* algorithm.

    Args:
        graph (AjacentList): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """ 
    if start == goal:
        return []

    paths = dict()    
    frontier = PriorityQueue()
    frontier.append((heuristic(graph, start, goal), start))
    paths[start] = [0, [start]]

    explored = set()
    
    while frontier.size() > 0:
        node = frontier.pop()[-1]
        if node == goal:
            return paths[goal][1]
        explored.add(node)
        for n in graph.neighbors(node):
            if not(n in explored) and not(frontier.__contains__(n)):
                g = paths[node][0] + graph.get_edge_weight(node, n)
                frontier.append((g+heuristic(graph, n, goal), n))
                paths[n] = [g, paths[node][1]+[n]]
            elif frontier.__contains__(n):
                g = paths[node][0] + graph.get_edge_weight(node, n)
                f = g+heuristic(graph, n, goal)
                if f < frontier.values(n):
                    frontier.replace_value(n, f)
                    paths[n] = [g, paths[node][1]+[n]]
              
