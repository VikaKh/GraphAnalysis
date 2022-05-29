import csv
import random
import numpy as np
import matplotlib.pyplot as plt

def dfs(graph):
    stack = []
    number_comp = 1
    max_node = [0,0] # max_node[0] - count nodes, max_node[1] - index component
    max_comp = []
    for b in graph:
        if not graph[b]['is_delete'] and not graph[b]['marker']:
            node = 0
            stack.insert(0, b)
            node += 1
            comp = [b]
            graph[b]['marker'] = True
            graph[b]['component'] = number_comp
            while stack:
                c = False
                for i in graph[stack[0]]['adjacent']:
                    if not graph[i]['marker'] and not graph[i]['is_delete']:
                        graph[i]['marker'] = True
                        graph[i]['component'] = number_comp
                        stack.insert(0, i)
                        comp.append(i)
                        node += 1
                        c = True
                        break
                if not c:
                    stack.pop(0)
                    c = 0
            number_comp += 1
            if max_node[0] < node:
                max_node[0] = node
                max_node[1] = number_comp - 1
                max_comp = comp
    
    return number_comp-1, max_node


def read_graph(filename, is_oriented, is_csv):
    if not is_csv:
        f = open(filename)
        graph = {}
        count_node = 0
        count_edge = 0
        edge = f.readline().split()
        while edge:
            edge[0] = int(edge[0])
            edge[1] = int(edge[1])
            if edge[0] in graph:
                if edge[0] == edge[1]:
                    graph[edge[0]]['degree'] += 2
                    count_edge += 1
                elif edge[1] not in graph[edge[0]]['adjacent']:
                    graph[edge[0]]['adjacent'].append(edge[1])
                    graph[edge[0]]['degree'] += 1
                    count_edge += 1
            else:
                if edge[0] != edge[1]:
                    graph[edge[0]] = {'adjacent': [edge[1]], 'reverse': [], 'degree': 1, 'component': '', 'marker': False, 'is_delete': False}
                    count_node += 1
                    count_edge += 1
                else:
                    graph[edge[0]] = {'adjacent': [], 'reverse': [], 'degree': 2, 'component': '', 'marker': False, 'is_delete': False}
                    count_node += 1
                    count_edge += 1
            if not is_oriented:
                if edge[1] in graph:
                    if edge[0] not in graph[edge[1]]['adjacent'] and edge[0] != edge[1]:
                        graph[edge[1]]['adjacent'].append(edge[0])
                        graph[edge[1]]['degree'] += 1
                else:
                    graph[edge[1]] = {'adjacent': [edge[0]], 'reverse': [], 'degree': 1, 'component': '', 'marker': False, 'is_delete': False}
                    count_node += 1
            else:
                if edge[1] in graph:
                    if edge[0] not in graph[edge[1]]['reverse'] and edge[0] != edge[1]:
                        graph[edge[1]]['reverse'].append(edge[0])
                        graph[edge[1]]['degree'] += 1
                else:
                    graph[edge[1]] = {'adjacent': [], 'reverse': [edge[0]], 'degree': 1, 'component': '', 'marker': False, 'is_delete': False}
                    count_node += 1
            edge = f.readline().split()
        f.close()
    else:
        graph = {}
        count_node = 0
        count_edge = 0
        with open(filename) as file:
            reader = csv.reader(file)
            next(reader)
            for edge in reader:
                edge[0] = int(edge[0])
                edge[1] = int(edge[1])
                if edge[0] in graph:
                    if edge[0] == edge[1]:
                        graph[edge[0]]['degree'] += 2
                        count_edge += 1
                    elif edge[1] not in graph[edge[0]]['adjacent']:
                        graph[edge[0]]['adjacent'].append(edge[1])
                        graph[edge[0]]['degree'] += 1
                        count_edge += 1
                else:
                    if edge[0] != edge[1]:
                        graph[edge[0]] = {'adjacent': [edge[1]], 'reverse': [], 'degree': 1, 'component': '', 'marker': False, 'is_delete': False}
                        count_node += 1
                        count_edge += 1
                    else:
                        graph[edge[0]] = {'adjacent': [], 'reverse': [], 'degree': 2, 'component': '', 'marker': False, 'is_delete': False}
                        count_node += 1
                        count_edge += 1
                if not is_oriented:
                    if edge[1] in graph:
                        if edge[0] not in graph[edge[1]]['adjacent'] and edge[0] != edge[1]:
                            graph[edge[1]]['adjacent'].append(edge[0])
                            graph[edge[1]]['degree'] += 1
                    else:
                        graph[edge[1]] = {'adjacent': [edge[0]], 'reverse': [], 'degree': 1, 'component': '', 'marker': False, 'is_delete': False}
                        count_node += 1
                else:
                    if edge[1] in graph:
                        if edge[0] not in graph[edge[1]]['reverse'] and edge[0] != edge[1]:
                            graph[edge[1]]['reverse'].append(edge[0])
                            graph[edge[1]]['degree'] += 1
                    else:
                        graph[edge[1]] = {'adjacent': [], 'reverse': [edge[0]], 'degree': 1, 'component': '', 'marker': False, 'is_delete': False}
                        count_node += 1
    return graph, count_node, count_edge

def heapsort(node, graph):
    build_max_heap(node, graph)
    for i in range(len(node) - 1, 0, -1):
        node[0], node[i] = node[i], node[0]
        max_heapify(node, graph, index=0, size=i)
 
def parent(i):
    return (i - 1)//2
 
def left(i):
    return 2*i + 1
 
def right(i):
    return 2*i + 2
 
def build_max_heap(node, graph):
    length = len(node)
    start = parent(length - 1)
    while start >= 0:
        max_heapify(node, graph, index=start, size=length)
        start = start - 1
 
def max_heapify(node, graph, index, size):
    l = left(index)
    r = right(index)
    if (l < size and graph[node[l]]['degree'] > graph[node[index]]['degree']):
        largest = l
    else:
        largest = index
    if (r < size and graph[node[r]]['degree'] > graph[node[largest]]['degree']):
        largest = r
    if (largest != index):
        node[largest], node[index] = node[index], node[largest]
        max_heapify(node, graph, largest, size)


#graph, count_node, count_edge = read_graph('vk.csv', False, True)
#graph, count_node, count_edge = read_graph('web-Google.txt', False, False)
graph, count_node, count_edge = read_graph('vk.csv', False, True)
#graph, count_node, count_edge = read_graph('CA-AstroPh.txt', False, False)
#graph, count_node, count_edge = read_graph('1.txt', False, False)

nodes = []
nodes_sort = []
for i in graph:
    nodes.append(i)
    nodes_sort.append(i)

heapsort(nodes_sort, graph)
nodes_sort.reverse()
#x1 =[]
#y =[]
#y1 = []
# для вк, чтоб не пересчитывать для изначального графа
x1 = [0]
y = [0.983]
y1 = [0.983]
for i in range(20, 100, 20):
    x = round(len(nodes)*(i/100))
    x1.append(i)
    y_for_medium = 0
    number_comp = 0
    max_node = [0, 0]
    #for t in range(1, 4):
    #    node_d = random.sample(nodes, x)
    #
    #    for j in node_d:
    #        graph[j]['is_delete'] = True
   
        #for i in graph:
        #    print(i, ': ', graph[i])
        #print ('\n')
    #    number_comp, max_node = dfs(graph)

    #    for j in node_d:
    #        graph[j]['is_delete'] = False
    #    
    #    for j in graph:
    #        graph[j]['marker'] = False

    #    y_for_medium += max_node[0]/count_node

    #y.append(y_for_medium/3)


    node_d = random.sample(nodes, x)

    for j in node_d:
        graph[j]['is_delete'] = True
   
        
    number_comp, max_node = dfs(graph)

    for j in node_d:
        graph[j]['is_delete'] = False
        
    for j in graph:
        graph[j]['marker'] = False

    y.append(max_node[0]/(count_node-x))


    for j in range(x):
        graph[nodes_sort[j]]['is_delete'] = True


    number_comp, max_node = dfs(graph)


    for j in range(x):
        graph[nodes_sort[j]]['is_delete'] = False
    for j in graph:
        graph[j]['marker'] = False
    y1.append(max_node[0]/(count_node-x))


plt.plot(x1, y1, color = 'r', label = 'наибольшей степени') # удаление наибольшей степени
plt.plot(x1, y, color = 'b', label = 'случайным образом') # случайное удаление
plt.xlabel("x%")
plt.ylabel("Доля вершин в наибольшей компоненте связности при удалении x% вершин")
plt.legend()
plt.show()

print(y1)
