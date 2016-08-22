import networkx
import pickle

### the intent of this program is to generate a list of 2-vertex candidate graphs which can minimally separate a genus 3 surface

def to_networkx(matrix):
    x_graph = networkx.Multigraph()
    row_count = 0
    for row in matrix:
        col_count = row_count
        for col in row[row_count:]:
            edge = 0
            if col_count == row_count: col = col/2
            while edge < col:
                x_graph.add_edge(row_count, col_count)
                edge += 1
            col_count += 1
        row_count += 1
    return x_graph

def to_matrix(graph):
    return [(graph[0]*2, graph[1]), (graph[1], graph[2]*2)]

g = 3
v = 2
E = range(3, 2*g + v + 1) ## needs more than 2 edges to be a potential minsep

graphs = []

###     a loop is an edge connecting a vertex to itself
###     a bridge is an edge connecting v_0 to v_1
for e in E[::-1]:
    bridges = e
    if e % 2 == 0: graph = [0, e, 0]
    if e % 2 == 1: graph = [1, e-1, 0]
    if graph[1] == 2 and graph[2] == 0: break
    graphs.append(graph)
    prev = 1
    while bridges > 0:
        if bridges-2 <= 0: break
        new_graphs = []
        for graph in graphs[-prev:]:
            new_graph = [graph[0]+2, graph[1]-2, graph[2]]
            new_graphs.append(new_graph)

            new_graph = [graph[0]+1, graph[1]-2, graph[2]+1]
            new_graphs.append(new_graph)

        new_graphs = [graph for graph in new_graphs if graph[0] >= graph[2] and graph[1] > 0]
        new_graphs = [graph for graph in new_graphs if not (graph[1] == 2 and graph[2] == 0)]
        prev = len(new_graphs)
        unique_new = []
        for new in new_graphs:
            is_new = True
            for graph in graphs:
                if new == graph: is_new = False
            if is_new: graphs.append(new)
                    
        bridges -= 2

matrices = []
for graph in graphs: matrices.append(to_matrix(graph))

with open("g3_v2_out", "wb") as out:
    pickle.dump(matrices, out)
