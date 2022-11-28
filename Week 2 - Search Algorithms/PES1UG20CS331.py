# Renita Kurian - PES1UG20CS331
# Lab 2 - A* and DFS Algorithms
import queue
import copy

def A_star_Traversal(cost, heuristic, start_point, goals):
    path = []
    n=len(cost)
    explored=[0 for i in range(n)]
    priority_queue_frontier = queue.PriorityQueue()             
    priority_queue_frontier.put((heuristic[start_point], ([start_point], start_point, 0)))

    while(priority_queue_frontier.qsize() != 0):
        total_cost_estimated, nodes_tuple = priority_queue_frontier.get()
        Astar_path_till_node = nodes_tuple[0]
        node = nodes_tuple[1]
        node_cost = nodes_tuple[2]

        if explored[node] == 0:
            explored[node] = 1

            if node in goals:
                return Astar_path_till_node
            
            for neighbour_node in range(1, n):
                if cost[node][neighbour_node] > 0 and explored[neighbour_node] == 0:
                    total_cost_till_node = node_cost + cost[node][neighbour_node]
                    total_cost_estimated = total_cost_till_node + heuristic[neighbour_node]                    
                    Astar_path_till_neighbour_node = copy.deepcopy(Astar_path_till_node)
                    Astar_path_till_neighbour_node.append(neighbour_node)
                    priority_queue_frontier.put((total_cost_estimated, (Astar_path_till_neighbour_node, neighbour_node, total_cost_till_node)))
    return path


def DFS_Traversal(cost, start_point, goals):
    path = []
    n = len(cost)                                  
    explored = [0 for i in range(n)]                
    stack_frontier = queue.LifoQueue()            
    stack_frontier.put((start_point, [start_point]))
    while(stack_frontier.qsize() != 0):
        node, dfspath_till_node = stack_frontier.get()
        if explored[node] == 0:
            explored[node] = 1
            if node in goals:
                return dfspath_till_node

           
            for neighbour_node in range(n-1, 0, -1):
                if cost[node][neighbour_node] > 0:
                    if explored[neighbour_node] == 0:
                        dfspath_till_neighbour_node = copy.deepcopy(dfspath_till_node)
                        dfspath_till_neighbour_node.append(neighbour_node)
                        stack_frontier.put((neighbour_node, dfspath_till_neighbour_node))
    return path
