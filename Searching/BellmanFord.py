 def _neighbor_from(self, graph, w_graph, v, box=None):
        M = len(graph)
        N = len(graph[0])
        neighbors = {}
        if graph[v[0]][v[1]] == '#':
            return neighbors
        for i in range(-1,2):
            for j in range(-1,2):
                if i ==0 and j == 0:
                    pass
                else:
                    ni = v[0]+i
                    nj = v[1]+j
                    if 0<=ni<M and 0<=nj<N:
                        if graph[ni][nj] == '.' or graph[ni][nj] == box or graph[ni][nj] == '*':
                            cost = 1 + (i**2) + (j**2) + w_graph[ni][nj]
                            neighbors[(ni,nj)] = cost     
        return neighbors
  
  def BellmanFord(self, G, W, t, t_name, pickup_box=True):
        '''
            G : 2-D list
            W : 2-D list
            t : (i,j)
        '''
        # initial a 2-D Table(dic), key: vertex v value: number of steps from v to t
        N = len(G)
        M = len(G[0])
        size = (N,M)
        P = [['-1' for j in range(M)] for i in range(N)]
        T = [[float("inf") for j in range(N*M)] for i in range(N*M)]    # 从上到下，从左到右给所有格子编号，(0,0)->0;(1,0)->1*M+0; 每行代表所用步数，每列代表所有可能的源节点;每格(i,j)的含义是从该节点j出发经过i步到达t的成本
        index_t = self._axis2ind(t, size)
        if pickup_box:
            # 唯一可能通过0步到目标节点的源只有它自己。
            T[0][index_t] = 0
            P[t[0]][t[1]] = 'B'
            # 唯一可能通过1步到目标节点的源都可以举起它。
            lift_area = self._neighbor(G, t)
            lift_area_axises = list(lift_area.keys())
            for y in lift_area_axises:
                y_index = self._axis2ind(y, size)
                T[1][y_index] = 4+W[t[0]][t[1]]
                P[y[0]][y[1]] = 'lift 1'
        else:
            # 唯一可能通过0步到目标节点的源只有它自己。
            T[0][index_t] = float("inf")
            P[t[0]][t[1]] = 'error' # 但此时还不能确定如果自己站在dropzone上该咋办
            # 唯一可能通过1步到目标节点的源都可以放下箱子。
            drop_area = self._neighbor_from(G, W, t, box=t_name) # 内涵从目标t到这些drop点的成本
            drop_area_axises = list(drop_area.keys())
            for y in drop_area_axises:
                cost_ty = drop_area[y]
                if T[0][index_t] > cost_ty:
                    T[0][index_t] = cost_ty
                    P[t[0]][t[1]] = "move " + self._axis2policy(t, y)
                y_index = self._axis2ind(y, size)
                T[1][y_index] = 2+W[t[0]][t[1]]
                P[y[0]][y[1]] = "down " + self._axis2policy(y, t)
        for i in range(2,N*M-1):    # 对于所有可能的步数1~N*M-1
            for j in range(N*M):    # 从(0,0)开始的节点到(N-1,M-1)
                if j == index_t:
                    continue
                T[i][j] = T[i-1][j]   # T[i,j] 从j(z)出发经过i步到达t的成本
                z = self._ind2axis(j, size)
                neighbors = self._neighbor_from(G, W, z, box=t_name) # 从z出发可以到达的所有节点
                neighbors_axis = list(neighbors.keys())
                for y in neighbors_axis:    # 从z出发可以到达的所有节点
                    # 先把邻居坐标转换成全局索引
                    neighbor_index = self._axis2ind(y, size)
                    # 从z出发到达y的成本:
                    cost = neighbors[y]
                    # 从z出发到达y的策略：
                    p = self._axis2policy(z, y)
                    # 从z出发到达y,再从y出发经过(i-1)步到达t的成本
                    # t_cost = cost + T[i-1][y]
                    t_cost = cost + T[i-1][neighbor_index]
                    # 如果从从z->y->t的成本小于z->t的成本
                    if T[i][j] >= t_cost:
                        T[i][j] = t_cost
                        P[z[0]][z[1]] = "move " + p
            if (T[i-1] == T[i]):
                return P
        return P
