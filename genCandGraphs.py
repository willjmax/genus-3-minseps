import argparse
import pickle
import itertools
import igraph
import networkx

def generateAllGraphs(v, e):
    allGraphs = []

    allPossibleEdges = list(itertools.combinations_with_replacement(range(0,v),2))
    allPossibleWaysOfChoosingEdges = itertools.combinations_with_replacement(allPossibleEdges,e)

    for choice in allPossibleWaysOfChoosingEdges:
        newGraph = igraph.Graph(v)
        newGraph.add_edges(list(choice))

        if newGraph.degree().count(0) == 0:
            allGraphs.append(newGraph)
            del newGraph

    return allGraphs

def hasNoVertexOfOddDegree(graph):
    listOfDegrees = graph.degree()
    numOfVerticesWithOddDegree = [x%2 for x in listOfDegrees].count(1)
    return numOfVerticesWithOddDegree == 0

def allDegreeTwoVerticesAreBoomerangs(graph):
    listOfDegrees = graph.degree()
    verticesWithDegreeTwo = [i for i, j in enumerate(listOfDegrees) if j == 2]
    while len(verticesWithDegreeTwo) > 0:
        vertexToExamine = verticesWithDegreeTwo.pop()
        if not isABoomerang(vertexToExamine, graph):
            return False
    return True

def atMostGPlusOneBoomerangs(graph):
    global g
    listOfDegrees = graph.degree()
    verticesWithDegreeTwo = [i for i, j in enumerate(listOfDegrees) if j == 2]
    numberOfBoomerangs = [isABoomerang(vertex, graph) for vertex in verticesWithDegreeTwo].count(True)
    if numberOfBoomerangs <= g+1:
        return True
    return False

def isABoomerang(vertexToExamine, graph):
    if graph.degree(vertexToExamine) != 2:
        return False
    neighbors = graph.neighbors(vertexToExamine)
    numberOfNonSelfNeighbors = len([i for i, j in enumerate(neighbors) if j != vertexToExamine])
    if numberOfNonSelfNeighbors == 0:
        return True
    return False

def removeIsomorphicCopies(temp):
    C_g = []
    while len(temp) > 0:
        C_g.insert(0, temp.pop(0))
        temp = [x for x in temp if not networkx.is_isomorphic(convertFromIgraphToNetworkX(C_g[0]), convertFromIgraphToNetworkX(x))]
    C_g.reverse()
    return C_g

def convertFromIgraphToNetworkX(igraph_graph):
    E = igraph_graph.get_edgelist()
    networkx_graph = networkx.MultiGraph()
    networkx_graph.add_edges_from(E)
    return networkx_graph

parser = argparse.ArgumentParser(description='Select the number of vertices')
parser.add_argument('vertices', type=int, help='the number of vertices in the graphs to be generated')
args = parser.parse_args()

global g
g = 3
v = args.vertices

C_g = []
temp = []

for e in range(v, 2*g + v + 1):
    allGraphs = generateAllGraphs(v, e)

    while len(allGraphs) > 0:
        graph = allGraphs.pop()

        if hasNoVertexOfOddDegree(graph) and allDegreeTwoVerticesAreBoomerangs(graph) and atMostGPlusOneBoomerangs(graph):
            temp.append(graph.copy())

print len(temp)
C_g = removeIsomorphicCopies(temp)
print len(C_g)
with open("g3_v{0}_out".format(v), "wb") as candidate_graphs:
    pickle.dump(C_g, candidate_graphs)
