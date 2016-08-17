import itertools
import copy
import networkx
import pickle

class unique_element:
    def __init__(self, value, occurrences):
        self.value = value
        self.occurrences = occurrences

def perm_unique(elements):
    eset = set(elements)
    listunique = [unique_element(i, elements.count(i)) for i in eset]
    u = len(elements)
    return perm_unique_helper(listunique, [0]*u,u-1)

def perm_unique_helper(listunique, result_list, d):
    if d < 0:
        yield tuple(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d] = i.value
                i.occurrences -= 1
                for g in perm_unique_helper(listunique, result_list, d-1):
                    yield g
                i.occurrences += 1

vertices = 6
degree = 4
genus = 3

sum_to = []
sum_to.append([])
sum_to.append([[1]])
sum_to.append([[2], [1, 1]])
sum_to.append([[3], [2, 1], [1, 1, 1,]])
sum_to.append([[4], [2, 2], [3, 1], [2, 1, 1], [1, 1, 1, 1]])

def pad_zeros(combination, row_number):
    zeros_needed = vertices - row_number - len(combination)
    for i in range(zeros_needed): combination.append(0)
    return combination

def possible_rows(combinations, row_number):
    total_perms = []
    if not combinations:
        zeros = pad_zeros(combinations, row_number)
        return tuple(zeros)
    for combination in combinations:
        padded_combination = pad_zeros(combination, row_number)
        perms = perm_unique(padded_combination)
        total_perms.append(perms)
    return total_perms

def valid_column_sum(matrix):
    for i in range(vertices):
        column_sum = 0
        for row in matrix:
            column_sum += row[i]
            if column_sum > degree: return False
    return True

def final_column_sum(matrix):
    for i in range(vertices):
        column_sum = 0
        for row in matrix: column_sum += row[i]
        if column_sum != degree: return False
    return True

def is_diagonal_even(matrix):
    for i in range(len(matrix)):
        if matrix[i][i] % 2 == 1: return False
    return True

def filter_initial_isos(matrix):
    if matrix[0][5] == 0 and matrix[0][4] == 0: return True
    if matrix[0][5] == 0 and matrix[0][4] == 1: return True
    return False

def filter_loops(matrix):
    loop_count = 0
    count = 0
    for row in matrix:
        if matrix[0][0] >= 1:
            loop_count += 1
    count += 1
#    return loop_count <= 4 
    return loop_count == 0 ## all 6 vertex graphs min-separating genus 3 are connected

def filter_isomorphic_copies(non_iso_matrices):
    matrices = []
    while len(non_iso_matrices) > 0:
        print len(non_iso_matrices)
        matrix = non_iso_matrices.pop()
        matrices.append(matrix)
        x_matrix = to_networkx(matrix)
        non_iso_matrices = [x for x in non_iso_matrices if not networkx.is_isomorphic(x_matrix, to_networkx(x))]
    return matrices

''' YOU'VE BEEN A BAD FUNCTION.... A VERY BAD FUNCTION
def to_networkx(graph):
    x_graph = networkx.MultiGraph()
    row_count = 0
    rows = len(matrix)
    for row in graph:
        col_count = 0
        for col in row[row_count:]:
            while col > 0:
                x_graph.add_edge(row_count, col_count)
                col -= 1
            col_count += 1
        row_count += 1
    print graph, x_graph.edges()
    return x_graph
'''

def to_networkx(matrix):
    x_graph = networkx.MultiGraph()
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

def bad_two_cycles(matrix):
    row_count = 0
    two_cycles = []
    for row in matrix:
        row_count += 1
        col_count = 0
        for col in row:
            col_count += 1
            if col == 2 and row_count != col_count:
                two_cycles.append([row_count, col_count])
    unique_two_cycles = []
    while two_cycles:
        cycle = two_cycles.pop()
        unique_two_cycles.append(cycle)
        two_cycles = [cy for cy in two_cycles if set(cycle) != set(cy)]
    return len(unique_two_cycles) >= 3 and all_disjoint(unique_two_cycles)

def all_disjoint(sets):
    elems = set()
    for s in sets:
        for x in s:
            if x in elems: return False
            elems.add(x)
    return True

current_row = 0
initial_sum = copy.deepcopy(sum_to[4])
initial_rows = possible_rows(initial_sum, current_row)
chain = itertools.chain(*initial_rows)

matrices = [[matrix] for matrix in chain]
matrices = [matrix for matrix in matrices if filter_initial_isos(matrix)]

current_row += 1
while current_row < vertices:
    new_matrices = []
    for matrix in matrices:
        total = 0

        for i in range(len(matrix)): total += matrix[i][current_row]

        remainder = copy.deepcopy(sum_to[degree - total])
        remainder = [i for i in remainder if len(i) <= vertices - current_row]
        perms = possible_rows(remainder, current_row)
        if type(perms) == tuple:
            perms = [perms]
        else:
            perms = itertools.chain(*perms)

        next_row = tuple([row[current_row] for row in matrix])
        for perm in perms:
            new_matrix = copy.copy(matrix)
            new_matrix.append(next_row + perm)
            if valid_column_sum(new_matrix) and is_diagonal_even(new_matrix): 
                new_matrices.append(new_matrix)

    matrices = copy.deepcopy(new_matrices)
    current_row += 1

matrices = [matrix for matrix in matrices if final_column_sum(matrix)]
matrices = [matrix for matrix in matrices if filter_loops(matrix)]
matrices = [matrix for matrix in matrices if not bad_two_cycles(matrix)]
matrices = filter_isomorphic_copies(matrices)

with open('candidate_graphs', 'wb') as candidate_graphs:
    pickle.dump(matrices, candidate_graphs)
