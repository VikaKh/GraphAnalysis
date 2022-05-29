import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import networkx as nx

def dfs(graph, a = 0, x = []):
    # 0 - for components
    # 1 or 2 - for strong components: 1 - on inverted arcs, 2 - with new order
    stack = []
    if not a:
        number_comp = 1
        max_node = [0,0] # max_node[0] - count nodes, max_node[1] - index component
        max_comp = []
        for b in graph:
            node = 0
            if not graph[b]['marker']:
                stack.insert(0, b)
                node += 1
                comp = [b]
                graph[b]['marker'] = True
                graph[b]['component'] = number_comp
                while stack:
                    c = False
                    for i in graph[stack[0]]['adjacent']:
                        if not graph[i]['marker']:
                            graph[i]['marker'] = True
                            graph[i]['component'] = number_comp
                            stack.insert(0, i)
                            comp.append(i)
                            node += 1
                            c = True
                            break
                    if not c and not a:
                        for i in graph[stack[0]]['reverse']:
                            if not graph[i]['marker']:
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
        return number_comp-1, max_node, max_comp
    elif a == 1:
        array = []
        for b in graph:
            if not graph[b]['marker']:
                stack.insert(0, b)
                graph[b]['marker'] = True
                while stack:
                    c = False
                    for i in graph[stack[0]]['reverse']:
                        if not graph[i]['marker']:
                            graph[i]['marker'] = True
                            stack.insert(0, i)
                            c = True
                            break
                    if not c:
                        array.append(stack[0])
                        stack.pop(0)
                        c = 0
        return array
    else:
        number_comp = 1
        max_node = [0,0] # max_node[0] - count nodes, max_node[1] - index component
        components = []
        for b in x:
            node = 0
            if graph[b]['marker']:
                stack.insert(0, b)
                comp = [b]
                node += 1
                graph[b]['marker'] = False
                graph[b]['component'] = number_comp
                while stack:
                    c = True
                    for i in graph[stack[0]]['adjacent']:
                        if graph[i]['marker']:
                            graph[i]['marker'] = False
                            graph[i]['component'] = number_comp
                            stack.insert(0, i)
                            comp.append(i)
                            node += 1
                            c = False
                            break
                    if c:
                        stack.pop(0)
                        c = 0
                number_comp += 1
                components.append(comp)
                if max_node[0] < node:
                    max_node[0] = node
                    max_node[1] = number_comp - 1
        return number_comp-1, max_node, components

def strong_comp(graph): # searching for strong conesion components
    array = dfs(graph, 1)
    array.reverse()
    return dfs(graph, 2, array)

def meta_graph(graph, comp):
    m_graph = {}
    for i in range(len(comp)):
        m_graph[i+1] = []
    for i in range(len(comp)):
        for j in range(i+2, len(comp)+1):
            for t in comp[i]:
                if any(element in comp[j-1] for element in graph[t]['adjacent']) and j+1 not in m_graph[i+1]:
                    m_graph[i+1].append(j)
                if any(element in comp[j-1] for element in graph[t]['reverse']) and i+1 not in m_graph[j]:
                    m_graph[j].append(i+1)
    return m_graph

def bfs(graph, u, d_list):
    graph[u]['color'] = 'grey'
    d_max = 0
    queue = [u]
    graph[u]['d'] = 0
    while queue:
        a = queue[0]
        queue.pop(0)
        for i in graph[a]['adjacent']:
            if graph[i]['color'] == 'white':
                graph[i]['color'] = 'grey'
                graph[i]['d'] = graph[a]['d'] + 1
                d = graph[i]['d']
                d_list.append(d)
                queue.append(i)
                if d_max < d:
                    d_max = d
            else:
                graph[a]['color'] = 'black'
        for i in graph[a]['reverse']:
            if graph[i]['color'] == 'white':
                graph[i]['color'] = 'grey'
                graph[i]['d'] = graph[a]['d'] + 1
                d = graph[i]['d']
                d_list.append(d)
                queue.append(i)
                if d_max < d:
                    d_max = d
            else:
                graph[a]['color'] = 'black'
    return d_max

def triangle(graph, u, v):
    #u - min degree
    #v - max dergee
    count = 0
    for i in graph[u]['adjacent']:
        if (i in graph[v]['adjacent'] or i in graph[v]['reverse']) and i != v and i != u:
            graph[i]['triangle'] += 1
            graph[u]['triangle'] += 1
            graph[v]['triangle'] += 1
            count += 1
    for i in graph[u]['reverse']:
        if (i in graph[v]['adjacent'] or i in graph[v]['reverse']) and i != v and i != u:
            graph[i]['triangle'] += 1
            graph[u]['triangle'] += 1
            graph[v]['triangle'] += 1
            count += 1
    return count

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
                    graph[edge[0]] = {'adjacent': [edge[1]], 'reverse': [], 'degree': 1, 'component': '', 'marker': False, 'color': 'white', 'd': 0, 'triangle': 0, 'cl_c': 0}
                    count_node += 1
                    count_edge += 1
                else:
                    graph[edge[0]] = {'adjacent': [], 'reverse': [], 'degree': 2, 'component': '', 'marker': False, 'color': 'white', 'd': 0, 'triangle': 0, 'cl_c': 0}
                    count_node += 1
                    count_edge += 1
            if not is_oriented:
                if edge[1] in graph:
                    if edge[0] not in graph[edge[1]]['adjacent'] and edge[0] != edge[1]:
                        graph[edge[1]]['adjacent'].append(edge[0])
                        graph[edge[1]]['degree'] += 1
                else:
                    graph[edge[1]] = {'adjacent': [edge[0]], 'reverse': [], 'degree': 1, 'component': '', 'marker': False, 'color': 'white', 'd': 0, 'triangle': 0, 'cl_c': 0}
                    count_node += 1
            else:
                if edge[1] in graph:
                    if edge[0] not in graph[edge[1]]['reverse'] and edge[0] != edge[1]:
                        graph[edge[1]]['reverse'].append(edge[0])
                        graph[edge[1]]['degree'] += 1
                else:
                    graph[edge[1]] = {'adjacent': [], 'reverse': [edge[0]], 'degree': 1, 'component': '', 'marker': False, 'color': 'white', 'd': 0, 'triangle': 0, 'cl_c': 0}
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
                        graph[edge[0]] = {'adjacent': [edge[1]], 'reverse': [], 'degree': 1, 'component': '', 'marker': False, 'color': 'white', 'd': 0, 'triangle': 0, 'cl_c': 0}
                        count_node += 1
                        count_edge += 1
                    else:
                        graph[edge[0]] = {'adjacent': [], 'reverse': [], 'degree': 2, 'component': '', 'marker': False, 'color': 'white', 'd': 0, 'triangle': 0, 'cl_c': 0}
                        count_node += 1
                        count_edge += 1
                if not is_oriented:
                    if edge[1] in graph:
                        if edge[0] not in graph[edge[1]]['adjacent'] and edge[0] != edge[1]:
                            graph[edge[1]]['adjacent'].append(edge[0])
                            graph[edge[1]]['degree'] += 1
                    else:
                        graph[edge[1]] = {'adjacent': [edge[0]], 'reverse': [], 'degree': 1, 'component': '', 'marker': False, 'color': 'white', 'd': 0, 'triangle': 0, 'cl_c': 0}
                        count_node += 1
                else:
                    if edge[1] in graph:
                        if edge[0] not in graph[edge[1]]['reverse'] and edge[0] != edge[1]:
                            graph[edge[1]]['reverse'].append(edge[0])
                            graph[edge[1]]['degree'] += 1
                    else:
                        graph[edge[1]] = {'adjacent': [], 'reverse': [edge[0]], 'degree': 1, 'component': '', 'marker': False, 'color': 'white', 'd': 0, 'triangle': 0, 'cl_c': 0}
                        count_node += 1
    return graph, count_node, count_edge

"""def dijkstra(graph, comp, u):
    d = []
    is_visit = []
    for i in range(len(comp)):
        d.append('')
        is_visit.append(True)
    d[comp.index(u)] = 0
    for i in graph[u]['adjacent']:
        if i in comp:
            if d[comp.index(i)] == '' or d[comp.index(i)] > d[comp.index(u)] + 1:
                d[comp.index(i)] = d[comp.index(u)] + 1
    is_visit[comp.index(u)] = False
    while any(is_visit):
        v = 0
        for i in comp:
            if is_visit[comp.index(i)] and d[comp.index(i)] != '':
                v = i
                break
        for i in graph[v]['adjacent']:
            if i in comp:
                if d[comp.index(i)] == '' or d[comp.index(i)] > d[comp.index(v)] + 1:
                    d[comp.index(i)] = d[comp.index(v)] + 1
        is_visit[comp.index(v)] = False
    d.remove(0)
    return max(d), d"""

def metagraph(graph, number_st_comp):
    m_graph = {}
    for i in range(number_st_comp):
        m_graph[i+1] = []
    for i in graph:
        for j in graph[i]['adjacent']:
            if graph[i]['component'] != graph[j]['component'] and graph[j]['component'] not in m_graph[graph[i]['component']]:
                m_graph[graph[i]['component']].append(graph[j]['component'])
    return m_graph

#------------------------------------------------------------------------------------------------------------------------------------------------------------

is_oriented = True # for web-Google.txt.txt, soc-wiki-Vote.txt, email-Eu-core.txt
#is_oriented = False # other

#graph, count_node, count_edge = read_graph('CA-AstroPh.txt', is_oriented, False)
#graph, count_node, count_edge = read_graph('web-Google.txt', is_oriented, False)
#graph, count_node, count_edge = read_graph('vk.csv', is_oriented, True)
graph, count_node, count_edge = read_graph('soc-wiki-Vote.txt', is_oriented, False)
#graph, count_node, count_edge = read_graph('socfb-Reed98.txt', is_oriented, False)

"""G = nx.Graph()
with open('socfb-Reed98.txt', mode='r') as f:
    for line in f:
        u, v = line.split()
        G.add_edge(u, v)"""

print('Count nodes: ', count_node)
print('Count edges: ', count_edge)
if not is_oriented:
    print('Density: ', (2*count_edge)/(count_node*(count_node-1)))
else:
    print('Density: ', (count_edge)/(count_node*(count_node-1)))


if is_oriented:
    tic = time.perf_counter()
    number_st_comp, max_node_st_c, st_comp = strong_comp(graph)
    toc = time.perf_counter()
    print('Count strong components: ', number_st_comp)
    print('     number nodes: ', max_node_st_c[0])
    print('     fraction of nodes: ', max_node_st_c[0]/count_node, ' or ', 100*max_node_st_c[0]/count_node, '%')
    print(f'Time: {toc - tic:0.4f} sec')
    tic = time.perf_counter()
    m_graph = metagraph(graph, number_st_comp)
    toc = time.perf_counter()
    f = open('metagraph.txt', 'w')
    f.write('Nodes in components' + '\n')
    for i in range(number_st_comp):
        f.write(str(i+1))
        f.write(':' + str(st_comp[i]) + '\n')
    f.write('Meta-graph \n')
    for i in range(number_st_comp):
        f.write(str(i+1))
        f.write(':' + str(m_graph[i+1]))
    print(f'Time: {toc - tic:0.4f} sec')
    for i in graph:
        for j in graph[i]['reverse']:
            if j not in graph[i]['adjacent']:
                graph[i]['adjacent'].append(j)
            graph[i]['reverse'].remove(j)

tic = time.perf_counter()
number_comp, max_node_comp, comp = dfs(graph)
toc = time.perf_counter()
print('Count components: ', number_comp)
print('     number nodes: ', max_node_comp[0])
print('     fraction of nodes: ', max_node_comp[0]/count_node, ' or ', 100*max_node_comp[0]/count_node, '%')
print(f'Time: {toc - tic:0.4f} sec')

"""tic = time.perf_counter()
print("Count components from networkx: ",nx.number_connected_components(G))
toc = time.perf_counter()
print(f'Time: {toc - tic:0.4f} sec')
largest_component = G.subgraph(nodes=max(nx.connected_components(G), key=len))"""

max_degree = 0
min_degree = graph[next(iter(graph))]['degree']
medium_degree = 0
for i in graph:
    if max_degree < graph[i]['degree']:
        max_degree = graph[i]['degree']
    if min_degree > graph[i]['degree']:
        min_degree = graph[i]['degree']
    medium_degree += graph[i]['degree']
print('Max degree: ', max_degree)
print('Min degree: ', min_degree)
print('Medium degree: ', medium_degree/count_node)


tic = time.perf_counter()
node500 = random.sample(comp, 500)

d = []
d_all = []
for i in node500:
    d.append(bfs(graph, i, d_all))
    for j in comp: #graph:
        graph[j]['color'] = 'white'
    #a, b = dijkstra(graph, comp, i)
    #d_all += b
    #d.append(a)
d.sort()
d_all.sort()

print('Diameter: ', max(d))
print('Radius: ', min(d))
print('90 percentile: ', d_all[round(0.9*len(d_all))])
toc = time.perf_counter()
print(f'Time: {toc - tic:0.4f} sec')

"""tic = time.perf_counter()
diam = nx.diameter(largest_component)
toc = time.perf_counter()
radius = nx.radius(largest_component)
print("Networks diametr: ", diam)
print("Networks radius: ", radius)
print(f'Time: {toc - tic:0.4f} sec')"""

"""tic = time.perf_counter()
listrain = sum(nx.triangles(G).values())
print("Count triangle from networkx: ", int(listrain/3))
toc = time.perf_counter()
print(f'Time: {toc - tic:0.4f} sec')"""

tic = time.perf_counter()
count_tr = 0
for i in graph:
    dergee_i = graph[i]['degree']
    for j in graph[i]['adjacent']:
        if j != i:
            if dergee_i < graph[j]['degree']:
                 count_tr += triangle(graph, i, j)
            else:
                 count_tr += triangle(graph, j, i)
    for j in graph[i]['reverse']:
        if j != i:
            if dergee_i < graph[j]['degree']:
                count_tr += triangle(graph, i, j)
            else:
                count_tr += triangle(graph, j, i)
toc = time.perf_counter()
print('Count triangle: ', int(count_tr/6))
print(f'Time: {toc - tic:0.4f} sec')

C = 0
for i in graph:
    n = len(graph[i]['adjacent']) + len(graph[i]['reverse'])
    C += (n*(n-1))/2

Cl_medium = 0
Cl_global = 0

for i in graph:
    t = int(graph[i]['triangle']/6)
    graph[i]['triangle'] = t
    n = len(graph[i]['adjacent']) + len(graph[i]['reverse'])
    if n >= 2:
        graph[i]['cl_c'] = (2*t)/(n*(n-1))
    c = graph[i]['cl_c']
    Cl_medium += c
    Cl_global += t
Cl_medium = Cl_medium/count_node
Cl_global = Cl_global/C

print('Average cluster coefficient: ', Cl_medium)
print('Global cluster coefficient: ', Cl_global)

ver_deg = []
for i in graph:
    ver_deg.append(graph[i]['degree'])

ver_deg = np.bincount(ver_deg)
ver_deg = ver_deg/sum(ver_deg)

f = 0
fig, ax = plt.subplots()
#for i in range(1, len(ver_deg)): # distribution function
#    ax.arrow(i, f, -1, 0)
#    f += ver_deg[i]
plt.plot(range(len(ver_deg)), ver_deg) 

fig, ax = plt.subplots()
#f = 0
#for i in range(1, len(ver_deg)): # distribution function
#    ax.arrow(i, f, -1, 0)
#    f += ver_deg[i]
ax.set_xscale('log')
ax.set_yscale('log')
plt.plot(range(len(ver_deg)), ver_deg)


plt.show()


