import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import random
import os
from scripts.data_loader import data_loader
import json
from collections import OrderedDict
import csv

import pdb;

class meta_path:
    visited = {}
    curr_meta = []
    metas = [{} for i in range(8)] # 8: node type amount

    def __init__(self, path, meta_length, directed):
        self.path = path    # file path
        self.directed = directed    # graph is directed if true
        self.meta_length = meta_length  # length of meta path
        self.vertices, self.id_to_type = self.load_nodes()  # list of all vertices, dict from node id to node type
        self.total_num = len(self.vertices) # total number of vertices
        self.graph, self.ids_to_type = self.load_links(self.total_num)  # graph as sparse matrix, dict from [first node id][second node id] to link type

    def load_nodes(self):
        id_to_type = {}
        vertices = []

        with open(os.path.join(path, 'node.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                if len(th) == 4:
                    node_id, node_name, node_type, node_attr = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    id_to_type[node_id] = node_type
                    vertices.append(node_id)
                elif len(th) == 3:
                    node_id, node_name, node_type = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    id_to_type[node_id] = node_type
                    vertices.append(node_id)
                else:
                    raise Exception("Too few information to parse!")
            return vertices, id_to_type

    def load_links(self, total):
        id_to_adj = {}
        row = []
        col = []
        data = []

        with open(os.path.join(path, 'link.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                h_id, t_id, r_id, link_weight = int(th[0]), int(th[1]), int(th[2]), float(th[3])
                row.append(h_id)
                col.append(t_id)
                data.append(r_id)
                if h_id in id_to_adj:
                    id_to_adj[h_id][t_id] = r_id
                else:
                    id_to_adj[h_id] = {}
                    id_to_adj[h_id][t_id] = r_id

                if not self.directed:
                    new_id = - (r_id + 1)
                    row.append(t_id)
                    col.append(h_id)
                    data.append(new_id)
                    if t_id in id_to_adj:
                        id_to_adj[t_id][h_id] = new_id
                    else:
                        id_to_adj[t_id] = {}
                        id_to_adj[t_id][h_id] = new_id

            graph = sp.csr_matrix((np.array(data), (np.array(row), np.array(col))), shape=(total, total), dtype=np.int64)

            return graph, id_to_adj

    def get_edge_type(self):
        # convert list of vertices in a path to a string of edges type in the path
        edge_type = []
        for i in range(len(self.curr_meta)-1):
            v = self.curr_meta[i]
            u = self.curr_meta[i + 1]
            edge_type.append(self.ids_to_type[v][u])
        return ','.join(str(e) for e in edge_type)

    def get_meta(self, start_vertex, leng, typ):

        # if the vertex has been visited, return
        if start_vertex in self.visited:
            if self.visited[start_vertex] == True:
                return 
        
        self.visited[start_vertex] = True 
        
        leng += 1
        self.curr_meta.append(start_vertex)

        # if length of a path is the length of targeted length, add the path
        if leng >= self.meta_length:
            edge_type = self.get_edge_type()
            if edge_type in self.metas[typ]:
                self.metas[typ][edge_type] = self.metas[typ][edge_type] + 1
            else:
                self.metas[typ][edge_type] = 1
            self.visited[start_vertex] = False
            self.curr_meta.pop()
            return

        next_vertices = np.nonzero(self.graph[start_vertex].toarray()[0])[0]

        for v in next_vertices:
            self.get_meta(v, leng, typ)

        self.curr_meta.pop()
        self.visited[start_vertex] = False

    def get_meta_from_nodes(self):
        print('start')

        # get meta path starting from each vertex in vertices
        for n in self.vertices:
            n_t = self.id_to_type[n]
            self.visited = {}
            self.curr_meta = []
            self.get_meta(n, 0, n_t)

        json_data = {}
        json_data["Dataset"] = "Freebase"
        json_data["Meta_length"] = str(self.meta_length)

        idx = 0
        for m in self.metas:
            m = sorted(m.items(), key=lambda x: x[1], reverse=True)
            node_name = "node_"+ str(idx)
            json_data[node_name] = []
            for (k, v) in m:
                json_data[node_name].append({"path": k, "amount": v})
            idx += 1

        if self.directed:
            json_file = 'meta_' + str(self.meta_length) + ".json"
        elif not self.directed:
            json_file = 'not_directed_meta_' + str(self.meta_length) + ".json"

        with open(json_file, "w") as write_file:
            json.dump(json_data, write_file)

        '''
        for m in self.metas:
            m = sorted(m.items(), key=lambda x: x[1], reverse=True)
            csv_file = 'meta_' + str(self.meta_length) + "_node_" + str(idx) + ".csv"
            w = csv.writer(open(csv_file, "w"))
            for (k, v) in m:
                w.writerow([k, v])
            idx += 1
        '''

    def DFS(self, start, end, typ):
        if start in self.visited:
            if self.visited[start] == True:
                return 
        
        self.visited[start] = True 
        
        self.curr_meta.append(start)

        if start == end:
            edge_type = self.get_edge_type()
            if edge_type in self.metas[typ]:
                self.metas[typ][edge_type] = self.metas[typ][edge_type] + 1
            else:
                self.metas[typ][edge_type] = 1
            self.visited[start] = False
            self.curr_meta.pop()
            return 
        elif len(self.curr_meta) == self.meta_length:
            self.visited[start] = False
            self.curr_meta.pop()
            return 

        next_vertices = np.nonzero(self.graph[start].toarray()[0])[0]

        for v in next_vertices:
            self.DFS(v, end, typ)

        self.curr_meta.pop()
        self.visited[start] = False

    def get_nodes_type_of_edge(self):
        egde_types = {}
        for u in self.vertices:
            u_t = self.id_to_type[u]
            neighbors = np.nonzero(self.graph[u].toarray()[0])[0]
            for v in neighbors:
                v_t = self.id_to_type[v]
                edge_t = self.ids_to_type[u][v]
                if edge_t in egde_types:
                    if (u_t, v_t) != egde_types[edge_t]:
                        print('prev edge_t', egde_types[edge_t])
                        print('edge_t', edge_t, 'node1', u_t, 'node2', v_t)
                else:
                    egde_types[edge_t] = (u_t, v_t)


path = '../../../data/Freebase'

mp = meta_path(path, 3) # find all meta path of length 3
mp.get_meta_from_nodes()

#mp.adj_matrix()
#mp.get_nodes_type_of_edge()
