FILENAME = 'CA-AstroPh.txt'
ORIENTED = False
LANDMARKS_COUNT = range(20, 101, 40)
import datetime as tm
import networkx as nx







 





 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#считывание данных
import csv
from operator import truediv

def read_from_txt(filename, oriented=False):
    graph = {}
    with open(filename) as file:

        row = file.readline()
        while row:

            parent, child = row.split()
            parent = int(parent)
            child = int(child)

            if parent in graph:
                if child not in graph[parent]['linked']:
                    graph[parent]['linked'].append(child)
                    graph[parent]['degree'] += 1
            else:
                graph[parent] = {
                    'linked': [child],
                    'length': {},
                    'shortest_paths':{},
                    'degree': 1,
                }

            if oriented:
                if child not in graph:
                    graph[child] = {
                        'linked': [],
                        'length': {},
                        'shortest_paths':{},
                        'degree': 0,
                    }

            else:
                if child in graph:
                    if parent not in graph[child]['linked']:
                        graph[child]['linked'].append(parent)
                        graph[child]['degree'] += 1

                else:
                    graph[child] = {
                        'linked': [parent],
                        'length': {},
                        'shortest_paths':{},
                        'degree': 1,
                    }

            row = file.readline()

    return graph

def read_from_csv(filename, oriented=False):
    graph = {}

    with open(filename) as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:

            parent = int(row[0])
            child = int(row[1])

            if parent in graph:
                if child not in graph[parent]['linked']:
                    graph[parent]['linked'].append(child)
                    graph[parent]['degree'] += 1
            else:
                graph[parent] = {
                    'linked': [child],
                    'length': {},
                    'shortest_paths':{},
                    'degree': 1,
                }

            if oriented:
                if child not in graph:
                    graph[child] = {
                        'linked': [],
                        'length': {},
                        'shortest_paths':{},
                        'degree': 0,
                    }

            else:
                if child in graph:
                    if parent not in graph[child]['linked']:
                        graph[child]['linked'].append(parent)
                        graph[child]['degree'] += 1

                else:
                    graph[child] = {
                        'linked': [parent],
                        'length': {},
                        'shortest_paths':{},
                        'degree': 1,
                    }

    return graph

def parse(filename, oriented=False):
    if filename.split('.')[0] == 'vk':
        return read_from_csv(filename, oriented)
    elif filename.split('.')[-1] == 'txt':
        return read_from_txt(filename, oriented)













#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#вычисление кратчайших путей между двумя вершинами
#это просто bfs, только на выход дает маршрут
def short_path (start, finish, graph):
    from collections import deque 
    found = False
    visited = []
    visited.append(start)
    queue = deque()
    queue.append(start)
    pathes = {}
    pathes [start]={'shortest_path' : [start]}
    while queue:
        v = queue.popleft() 
        if v == finish:
            found = True
            path = pathes[v]['shortest_path']
            
            break

        for neighbor in graph[v]['linked']:
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
                p = pathes[v]['shortest_path'].copy()
                p.append(neighbor)
                pathes[neighbor] = {'shortest_path' : []}
                pathes[neighbor]['shortest_path'] = p


    if found:
        return path
    else:
        return -1













#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#выбор марок
import random


def landmarks_choose(start, finish, graph, selection_type, landmarks_count):
    start_time= tm.datetime.now()
    graph_items = graph.items()
    graph_size = len(graph_items)

    
    

    if selection_type == 'random':
        rand_nod = [i[0] for i in graph_items]
        random.shuffle(rand_nod)
        rand_nod = rand_nod[:landmarks_count]
        
        for v in rand_nod:
            path_from_start = short_path(start, v, graph)
            graph[start]['shortest_paths'][v] = path_from_start
            if path_from_start != -1:
                graph[start]['length'][v] = len(path_from_start)
            else: 
                graph[start]['length'][v] = -1

            path_from_finish = short_path(v, finish, graph)
            graph[finish]['shortest_paths'][v] = path_from_finish
            if path_from_finish != -1:
                graph[finish]['length'][v] = len(path_from_finish)
            else:
                graph[finish]['length'][v] = -1
        return rand_nod, (tm.datetime.now() - start_time).total_seconds()



    elif selection_type == 'degree':
        sort_degree = sorted(graph_items, key=lambda x: x[1]['degree'], reverse=True) 
        degree_nod = [i[0] for i in sort_degree[:landmarks_count]]


        for v in degree_nod:
            path_from_start = short_path(start, v, graph)
            graph[start]['shortest_paths'][v] = path_from_start
            if path_from_start != -1:
                graph[start]['length'][v] = len(path_from_start)
            else: 
                graph[start]['length'][v] = -1

            path_from_finish = short_path(v, finish, graph)
            graph[finish]['shortest_paths'][v] = path_from_finish
            if path_from_finish != -1:
                graph[finish]['length'][v] = len(path_from_finish)
            else:
                graph[finish]['length'][v] = -1

        return degree_nod, (tm.datetime.now() - start_time).total_seconds()




    elif selection_type == 'coverege':
        number_of_uses = {}
        while len(number_of_uses) < landmarks_count:
            uses_part = {}
            rand_nod = [i[0] for i in graph_items]
            random.shuffle(rand_nod)
            start_nod = rand_nod[:landmarks_count]
            finish_nod = rand_nod[graph_size - landmarks_count:]
            for i, j in zip(start_nod, finish_nod):
                ##used_nodes - список с номерами вершин, которые попали в кратчайший путь
                used_nodes = short_path (i,j, graph)
                if used_nodes == -1:
                    continue
                for v in used_nodes:
                    if v not in uses_part:
                        uses_part[v] = {
                            'count' : 1
                        }
                    else:
                        uses_part[v]['count'] += 1
            number_of_uses = {**number_of_uses, **uses_part}
        if len(number_of_uses) < landmarks_count:
            a = 5
        number_of_uses = sorted(number_of_uses, reverse=True)
        number_of_uses = number_of_uses[:landmarks_count]





        for v in number_of_uses:
            path_from_start = short_path(start, v, graph)
            graph[start]['shortest_paths'][v] = path_from_start
            if path_from_start != -1:
                graph[start]['length'][v] = len(path_from_start)
            else: 
                graph[start]['length'][v] = -1

            path_from_finish = short_path(v, finish, graph)
            graph[finish]['shortest_paths'][v] = path_from_finish
            if path_from_finish != -1:
                graph[finish]['length'][v] = len(path_from_finish)
            else:
                graph[finish]['length'][v] = -1
        return number_of_uses, (tm.datetime.now() - start_time).total_seconds()

        """берется M случайных пар вершин
        вычисляется кратчайший путь между ними
        для каждой вершины в этом пути увеличивается количество ее вхождений в кратчайшие пути
        выбираются вершины с самым большим количеством вхождений
        """















 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#bfs возвращает расстояние между двумя вершинами
def bfs(start, finish, graph):
    start_time= tm.datetime.now()
    if len(graph) == 0 or start not in graph or finish not in graph:
        return -1, (tm.datetime.now() - start_time).total_seconds()
    from collections import deque 
    found = False
    visited = []
    visited.append(start)
    queue = deque()
    queue.append(start)
    pathes = {}
    pathes [start]={'shortest_path' : [start]}
    
    while queue:
        v = queue.popleft() 
        if v == finish:
            found = True
            path = pathes[v]['shortest_path']
            break

        for neighbor in graph[v]['linked']:
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
                p = pathes[v]['shortest_path'].copy()
                p.append(neighbor)
                pathes[neighbor] = {'shortest_path' : []}
                pathes[neighbor]['shortest_path'] = p


    if found:
        return path, (tm.datetime.now() - start_time).total_seconds()
    else:
        return -1, (tm.datetime.now() - start_time).total_seconds()












 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#bfs+landmarks
def landmarks_bfs(start, finish, graph):
    start_time= tm.datetime.now()
    little_graph = {}
    start_path = graph[start]['shortest_paths']
    finish_path = graph[finish]['shortest_paths']
    


    for mark, path in list(start_path.items()) + list(finish_path.items()):
        if path == -1 or path == []:
            continue
        if path[len(path)-1] not in little_graph:
            little_graph[path[len(path)-1]] = { 
                'linked': [],
            }
        for v in range(len(path)-1):
            if path[v] in little_graph and path[v+1] not in little_graph[path[v]]['linked']:
                little_graph[path[v]]['linked'].append(path[v+1])
            elif path[v] not in little_graph:
                little_graph[path[v]] = {
                    'linked': [path[v+1]],
                }


            if path[v+1] in little_graph and path[v] not in little_graph[path[v+1]]['linked']:
                little_graph[path[v+1]]['linked'].append(path[v])
            elif path[v+1] not in little_graph:
                little_graph[path[v+1]] = {
                    'linked': [path[v]],
                }
                
    path, bfs_time = bfs(start, finish, little_graph)
    return path, (tm.datetime.now() - start_time).total_seconds()













 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#basic+landmarks
def landmarks_basic(start, finish, graph):
    start_time= tm.datetime.now()
    start_path = graph[start]['shortest_paths']
    finish_path = graph[finish]['shortest_paths']
    shortest_start_path = []
    shortest_finish_path = []

    
    d_up = 10**6
    
    for mark, from_start in start_path.items():
        
        to_finish = finish_path.get(mark, -1)
        if to_finish == -1 or to_finish == [] or from_start == [] or from_start == -1:
            continue
        to_finish.pop(0)
        d = len(from_start) + len(to_finish)
        
        if d < d_up:
            d_up = d
            shortest_start_path = from_start
            shortest_finish_path = to_finish

            
    #проверка на достижимость
    if d_up == 10**6:
        return -1, (tm.datetime.now() - start_time).total_seconds()
    shortest_path = shortest_start_path + shortest_finish_path
    return shortest_path, (tm.datetime.now() - start_time).total_seconds()












 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#основная часть
import random
import networkx as nx
import collections



results = {'basic':{
                    'random': {}, 
                    'degree': {},
                    'coverege': {}
                 },
        'landmarks_bfs': {
            'random': {}, 
            'degree': {},
            'coverege': {}
            },
        'bfs': {
            'random': {}, 
            'degree': {},
            'coverege': {}
            },
        'choose_landmarks': {
            'random': {}, 
            'degree': {},
            'coverege': {}
            }
         }
for landmarks_count in LANDMARKS_COUNT:
    results['basic']['random'][landmarks_count] = {'time': 0, 'accuracy': 0}
                                  
    results['basic']['degree'][landmarks_count] = {'time': 0,'accuracy': 0}
                                  
    results['basic']['coverege'][landmarks_count] = {'time': 0,'accuracy': 0}
                                  
    results['landmarks_bfs']['random'][landmarks_count] = {'time': 0, 'accuracy': 0}
                                  
    results['landmarks_bfs']['degree'][landmarks_count] = {'time': 0,'accuracy': 0}
                                  
    results['landmarks_bfs']['coverege'][landmarks_count] = {'time': 0,'accuracy': 0}

    results['bfs']['random'][landmarks_count] = {'time': 0}
                                  
    results['bfs']['degree'][landmarks_count] = {'time': 0}
                                  
    results['bfs']['coverege'][landmarks_count] = {'time': 0}
                                  
    results['choose_landmarks']['random'][landmarks_count] = {'time': 0}
                                  
    results['choose_landmarks']['degree'][landmarks_count] = {'time': 0}
                                  
    results['choose_landmarks']['coverege'][landmarks_count] = {'time': 0}


with open('results_test_CA-GrQc.txt', 'w') as file:
    tic = tm.datetime.now()

    graph = parse(FILENAME, ORIENTED)
    graph_items = graph.items()
    graph_size = len(graph_items)
    marks_selection = ('random', 'degree', 'coverege')


    #выбор начальной и конечной вершины

    number_of_tests = 7



    nodes = [i[0] for i in graph_items]
    nodes_start = nodes.copy()
    random.shuffle(nodes_start)
    nodes_start = nodes_start[:number_of_tests]

    nodes_finish = nodes.copy()
    random.shuffle(nodes_finish)
    nodes_finish = nodes_finish[:number_of_tests]

    #запуск алгоритмов с разными параметрами
    for selection in marks_selection:
        for landmarks_count in LANDMARKS_COUNT:
            for start, finish in zip(nodes_start,nodes_finish):
            
                for k, v in graph.items():
                    v['length'] = {}
                    v['shortest_paths'] = {}
                while start == finish:
                    if start < graph_size-1:
                        start += 1
                    else:
                        start -= 1

                landmarks, timer_landmarks = landmarks_choose (start, finish, graph, selection, landmarks_count)
                results['choose_landmarks'][selection][landmarks_count]['time'] += timer_landmarks

                path_bfs, timer_bfs = landmarks_bfs (start, finish, graph)

                results['landmarks_bfs'][selection][landmarks_count]['time'] += timer_bfs



                path_basic, timer_basic = landmarks_basic (start, finish, graph)
                results['basic'][selection][landmarks_count]['time'] += timer_basic


                s_path, timer_exact = bfs(start, finish, graph)
                results['bfs'][selection][landmarks_count]['time'] += timer_exact

                if s_path != -1 and path_bfs != -1:
                    approximation_error_bfs = (len(path_bfs) - len(s_path))/len(s_path)
                    results['landmarks_bfs'][selection][landmarks_count]['accuracy'] += approximation_error_bfs

                if s_path != -1 and path_bfs != -1 and path_basic != -1:
                    approximation_error_basic = (len(path_basic) - len(s_path))/len(s_path)
                    results['basic'][selection][landmarks_count]['accuracy'] += approximation_error_basic

            results['basic'][selection][landmarks_count]['accuracy'] /= number_of_tests
            results['landmarks_bfs'][selection][landmarks_count]['accuracy'] /= number_of_tests
            results['choose_landmarks'][selection][landmarks_count]['time'] /= number_of_tests
            results['landmarks_bfs'][selection][landmarks_count]['time'] /= number_of_tests
            results['basic'][selection][landmarks_count]['time'] /= number_of_tests
            results['bfs'][selection][landmarks_count]['time'] /= number_of_tests
 
    toc = tm.datetime.now()
    file.write(str(toc - tic) + '\n')



import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt




x = []
y_basic_random_time = []
y_basic_degree_time = []
y_basic_coverege_time = []
y_landmarks_bfs_random_time = []
y_landmarks_bfs_degree_time = []
y_landmarks_bfs_coverege_time = []
y_bfs_random_time = []
y_bfs_degree_time = []
y_bfs_coverege_time = []
y_choose_landmarks_random_time = []
y_choose_landmarks_degree_time = []
y_choose_landmarks_coverege_time = []

y_basic_random_accuracy = []
y_basic_degree_accuracy = []
y_basic_coverege_accuracy = []
y_landmarks_bfs_random_accuracy = []
y_landmarks_bfs_degree_accuracy = []
y_landmarks_bfs_coverege_accuracy = []
y_bfs_random_accuracy = []
y_bfs_degree_accuracy = []
y_bfs_coverege_accuracy = []
y_choose_landmarks_random_accuracy = []
y_choose_landmarks_degree_accuracy = []
y_choose_landmarks_coverege_accuracy = []

for count in LANDMARKS_COUNT:
    x.append(count)
    y_basic_random_time.append(results['basic']['random'][count]['time'])
    y_basic_degree_time.append(results['basic']['degree'][count]['time'])
    y_basic_coverege_time.append(results['basic']['coverege'][count]['time'])###########################################################
    y_landmarks_bfs_random_time.append(results['landmarks_bfs']['random'][count]['time'])
    y_landmarks_bfs_degree_time.append(results['landmarks_bfs']['degree'][count]['time'])
    y_landmarks_bfs_coverege_time.append(results['landmarks_bfs']['coverege'][count]['time'])###########################################################
    y_bfs_random_time.append(results['bfs']['random'][count]['time'])
    y_bfs_degree_time.append(results['bfs']['degree'][count]['time'])
    y_bfs_coverege_time.append(results['bfs']['coverege'][count]['time'])###########################################################
    y_choose_landmarks_random_time.append(results['choose_landmarks']['random'][count]['time'])
    y_choose_landmarks_degree_time.append(results['choose_landmarks']['degree'][count]['time'])
    y_choose_landmarks_coverege_time.append(results['choose_landmarks']['coverege'][count]['time'])###########################################################

    y_basic_random_accuracy.append(results['basic']['random'][count]['accuracy'])
    y_basic_degree_accuracy.append(results['basic']['degree'][count]['accuracy'])
    y_basic_coverege_accuracy.append(results['basic']['coverege'][count]['accuracy'])###########################################################
    y_landmarks_bfs_random_accuracy.append(results['landmarks_bfs']['random'][count]['accuracy'])
    y_landmarks_bfs_degree_accuracy.append(results['landmarks_bfs']['degree'][count]['accuracy'])
    y_landmarks_bfs_coverege_accuracy.append(results['landmarks_bfs']['coverege'][count]['accuracy'])###########################################################





fig, ax = plt.subplots()
plt.plot(x, y_basic_random_time, color='blue')
plt.plot(x, y_basic_degree_time, color='red')
plt.plot(x, y_basic_coverege_time, color='green')
plt.xlabel('Количество марок')
plt.ylabel('Время работы')
plt.title('Время работы basic алгоритма на марках для разных стратегий', fontsize=30)
plt.legend(['random landmarks', 'degree landmarks', 'coverage landmarks'])


fig, ax = plt.subplots()
plt.plot(x, y_landmarks_bfs_random_time, color='blue')
plt.plot(x, y_landmarks_bfs_degree_time, color='red')
plt.plot(x, y_landmarks_bfs_coverege_time, color='green')
plt.xlabel('Количество марок')
plt.ylabel('Время работы')
plt.title('Время работы landmarks_bfs алгоритма на марках для разных стратегий', fontsize=30)
plt.legend(['random landmarks', 'degree landmarks', 'coverage landmarks'])


fig, ax = plt.subplots()
plt.plot(x, y_choose_landmarks_random_time, color='blue')
plt.plot(x, y_choose_landmarks_degree_time, color='red')
plt.plot(x, y_choose_landmarks_coverege_time, color='green')
plt.xlabel('Количество марок')
plt.ylabel('Время работы')
plt.title('Время выбора марок для разных стратегий', fontsize=30)
plt.legend(['random landmarks', 'degree landmarks', 'coverage landmarks'])


fig, ax = plt.subplots()
plt.plot(x, y_basic_random_accuracy, color='blue')
plt.plot(x, y_basic_degree_accuracy, color='red')
plt.plot(x, y_basic_coverege_accuracy, color='green')
plt.xlabel('Количество марок')
plt.ylabel('Точность')
plt.title('Точность basic алгоритма на марках для разных стратегий', fontsize=30)
plt.legend(['random landmarks', 'degree landmarks', 'coverage landmarks'])


fig, ax = plt.subplots()
plt.plot(x, y_landmarks_bfs_random_accuracy, color='blue')
plt.plot(x, y_landmarks_bfs_degree_accuracy, color='red')
plt.plot(x, y_landmarks_bfs_coverege_accuracy, color='green')
plt.xlabel('Количество марок')
plt.ylabel('Точность')
plt.title('Точность landmarks_bfs алгоритма на марках для разных стратегий', fontsize=30)
plt.legend(['random landmarks', 'degree landmarks', 'coverage landmarks'])


plt.show()
