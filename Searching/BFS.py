import PriorityQueue
import AdjacencyList
def BFS(graph, start, goal):
    """
    Implement breadth-first-search.
    Args:
        graph (Adjacency List): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    best_path = []
    if start == goal:
        return best_path
    frontier = PriorityQueue()
    i = 0
    frontier.append_alpha((i,start))    #<- a priority queue ordered by PATH-Cost, with node(dis,'name')
    explored = PriorityQueue()          #<- an empty list
    find = False
    father = None
    path = [[None,start]]
    while frontier.size() > 0:         #loop when frontier is not empty
        i += 1                          #layer number
        node = frontier.pop()[-1]       #chose the shallowest node in frontier by caption seq
        father = node
        explored.append_alpha((i,node))#explore, add node in the explore list.
        for n in graph.neighbors(node): #for each child node of node(AKA explore a node)
            if not(explored.__contains__(n)) and not(frontier.__contains__(n)): #if it is new add to frontier
                frontier.append_alpha((i,n))
                path.append([father,n])
                if n == goal:
                    find = True
                    break 
        if find is True:
            explored.append_alpha(((i+1),goal))
            break
    path_len = path.__len__()
    back = goal
    for i in range(path_len-1,-1,-1):
        if back is path[i][-1]:
            back = path[i][0]
            if back is not None:
                best_path.insert(0, back)
            else:
                best_path.append(goal)
                break
    return best_path
  
if __name__ == '__main__':
   print("Implement an AdjacencyList Graph to test this function.")
