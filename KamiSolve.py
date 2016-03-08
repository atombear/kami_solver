from __future__ import division
import networkx as nx
import matplotlib.pyplot as plt
import copy
import scipy
import scipy.linalg
import operator
from scipy.constants import pi
import cmath

# A long-winded means of having the user manually enter the details of 
# the color graph corresponding to the puzzle
def build_color_graph():
    c_graph = {}
    
    while True:
        node = raw_input('What is the name of the node? ')
        if node == 'done':
            return c_graph
        color = raw_input('What is the color of the node? ')
        con = raw_input('What are the connections of the node? ')
        
        c_graph[node] = [int(color)] + list(con)



# A bad function that should draw the graph, but it does so, badly
def draw_graph(graph):

    # create networkx graph
    G=nx.Graph()
    
    ordered_node_list = scipy.sort([int(i[1::]) for i in graph])

    # add nodes
    #for node in graph:
    #    G.add_node(node)
    for num in ordered_node_list:
        G.add_node('n'+str(num))
    

    # add edges
    for i in graph:
        for j in graph[i][1::]:
            G.add_edge(i,j)
            
    colors = ['b','r','g','c','w','k']
    
    node_color = [colors[graph[node][0]] for node in graph]

    # draw graph
    #pos = nx.shell_layout(G)
    pos = nx.spring_layout(G,iterations=100)
    nx.draw(G, pos, node_color = node_color)

    # show graph
    plt.axis('off')
    plt.show()


# Changes the color of one node of a graph to new_color
def change_color(graph,node,new_color):
    graph_copy = copy.deepcopy(graph)
    graph_copy[node][0] = new_color
    return graph_copy
    

# Finds all pairs of adjacent nodes that have the same color
def find_neighbors(graph):
    nlist = []
    for i in graph:
        for j in graph[i][1::]:
            if graph[i][0] == graph[j][0]:
                nlist.append(tuple(set((i,j))))
         
    return list(set(nlist))

# Returns a graph having merged the nodes in ntup, which are the same color. One
# node adopts the connections of both.
def merge_nodes(graph,ntup):
    graph_copy = copy.deepcopy(graph)
    new_con = graph_copy[ntup[0]][1::] + graph_copy[ntup[1]][1::]
    new_con.remove(ntup[0])
    new_con.remove(ntup[1])
    new_con = list(set(new_con))
    new_con = [graph_copy[ntup[0]][0]] + new_con
    del graph_copy[ntup[0]]
    del graph_copy[ntup[1]]
    graph_copy[ntup[1]] = new_con
    for i in graph_copy:
        if ntup[0] in graph_copy[i]:
            graph_copy[i][graph_copy[i].index(ntup[0])] = ntup[1]
            graph_copy[i][1::] = list(set(graph_copy[i][1::]))
    return graph_copy


# Determines the minimal graph from the 16x10 puzzle colorboard
def determine_graph(c):
    graph = {}
    num_colors = len(c)
    c0 = c[0]
    c1 = c[1]
    c2 = c[2]
    if num_colors == 4:
        c3 = c[3]
    for i in range(1,9):
        for j in range(1,15):
            p = (i,j)
            if num_colors == 3:
                color = 0*(p in c0) + 1*(p in c1) + 2*(p in c2)
            if num_colors == 4:
                color = 0*(p in c0) + 1*(p in c1) + 2*(p in c2) + 3*(p in c3)
            graph['n'+str(10*j+i)] = [color, 'n'+str(10*j+i-10), 'n'+str(10*j+i+1), 'n'+str(10*j+i+10), 'n'+str(10*j+i-1)]
            
    for i in range(1,9):
        j = 0
        p = (i,j)
        if num_colors == 3:
            color = 0*(p in c0) + 1*(p in c1) + 2*(p in c2)
        if num_colors == 4:
            color = 0*(p in c0) + 1*(p in c1) + 2*(p in c2) + 3*(p in c3)
        graph['n'+str(10*j+i)] = [color, 'n'+str(10*j+i+1), 'n'+str(10*j+i+10), 'n'+str(10*j+i-1)]
        j = 15
        p = (i,j)
        if num_colors == 3:
            color = 0*(p in c0) + 1*(p in c1) + 2*(p in c2)
        if num_colors == 4:
            color = 0*(p in c0) + 1*(p in c1) + 2*(p in c2) + 3*(p in c3)
        graph['n'+str(10*j+i)] = [color, 'n'+str(10*j+i+1), 'n'+str(10*j+i-10), 'n'+str(10*j+i-1)]
    
    
    for j in range(1,15):
        i = 0
        p = (i,j)
        if num_colors == 3:
            color = 0*(p in c0) + 1*(p in c1) + 2*(p in c2)
        if num_colors == 4:
            color = 0*(p in c0) + 1*(p in c1) + 2*(p in c2) + 3*(p in c3)
        graph['n'+str(10*j+i)] = [color, 'n'+str(10*j+i-10), 'n'+str(10*j+i+1), 'n'+str(10*j+i+10)]
        
        i = 9
        p = (i,j)
        if num_colors == 3:
            color = 0*(p in c0) + 1*(p in c1) + 2*(p in c2)
        if num_colors == 4:
            color = 0*(p in c0) + 1*(p in c1) + 2*(p in c2) + 3*(p in c3)
        graph['n'+str(10*j+i)] = [color, 'n'+str(10*j+i-10), 'n'+str(10*j+i+10), 'n'+str(10*j+i-1)]
    
    i = 0
    j = 0
    p = (i,j)
    if num_colors == 3:
        color = 0*(p in c0) + 1*(p in c1) + 2*(p in c2)
    if num_colors == 4:
        color = 0*(p in c0) + 1*(p in c1) + 2*(p in c2) + 3*(p in c3)
    graph['n'+str(10*j+i)] = [color, 'n'+str(10*j+i+1), 'n'+str(10*j+i+10)]
    i = 9
    j = 0
    p = (i,j)
    if num_colors == 3:
        color = 0*(p in c0) + 1*(p in c1) + 2*(p in c2)
    if num_colors == 4:
        color = 0*(p in c0) + 1*(p in c1) + 2*(p in c2) + 3*(p in c3)
    graph['n'+str(10*j+i)] = [color, 'n'+str(10*j+i+10), 'n'+str(10*j+i-1)]
    i = 0
    j = 15
    p = (i,j)
    if num_colors == 3:
        color = 0*(p in c0) + 1*(p in c1) + 2*(p in c2)
    if num_colors == 4:
        color = 0*(p in c0) + 1*(p in c1) + 2*(p in c2) + 3*(p in c3)
    graph['n'+str(10*j+i)] = [color, 'n'+str(10*j+i-10), 'n'+str(10*j+i+1)]
    i = 9
    j = 15
    p = (i,j)
    if num_colors == 3:
        color = 0*(p in c0) + 1*(p in c1) + 2*(p in c2)
    if num_colors == 4:
        color = 0*(p in c0) + 1*(p in c1) + 2*(p in c2) + 3*(p in c3)
    graph['n'+str(10*j+i)] = [color, 'n'+str(10*j+i-10), 'n'+str(10*j+i-1)]
    
    
    neigh_list = find_neighbors(graph)
    while neigh_list:
        graph = merge_nodes(graph,neigh_list[0])
        neigh_list = find_neighbors(graph)
    return graph

# Turns a RGB tuple into a HSL tuple
def rgb2hsl(rgb,n):
    R = rgb[0]/n
    G = rgb[1]/n
    B = rgb[2]/n
    
    M = max([R,G,B])
    m = min([R,G,B])
    
    L = .5*(M + m)
    
    C = M - m
    
    if C == 0:
        H = 0
    else:
        S = C/(1-abs(2*L-1))
        if M == R:
            H = 60*(((G-B)/C)%6)
        if M == G:
            H = 60*((B-R)/C + 2)
        if M == B:
            H = 60*((R-G)/C + 4)
            
    return [H,S,L]
            

# Turns a RGB tuple into a CIELAB tuple
def rgb2CIELAB(RGB):
    def f(t):
        if t>(6/29)**3:
            return t**(1/3)
        else:
            return t*(1/3)*(29/6)**2 + (4/29)
            
    b = scipy.array([[.49,.31,.2],[.17697,.81240,.01063],[0.0,.01,.99]])/.17697
    X,Y,Z = scipy.inner(b,scipy.array(RGB))
    Xn,Yn,Zn = [95.047,100,108.883]
    
    L = 116*f(Y/Yn)-16
    a = 500*(f(X/Xn)-f(Y/Yn))
    b = 200*(f(Y/Yn)-f(Z/Zn))
    
    return scipy.array([L,a,b])
    
    
# Backpropagation algorithm to solve the graph
def solve_graph(graph,move_list,move_limit):
    
    #color_list = list(set([graph[i][0] for i in graph]))
    node_list = [i for i in graph]
    
    if len(graph) == 1:
        return move_list
    
    # Heuristic for choosing the node with the most connections
    temp = sorted([(i,len(graph[i])-1) for i in graph], key=lambda x: x[1])[::-1]
    organized_node_list = [i[0] for i in temp]
    for node in organized_node_list:
    
    # Heuristic for choosing the node closest to the center
    #cen_node = scipy.array([5,8])
    #temp = sorted([(i,scipy.linalg.norm(scipy.array([int(i[1::])%10,int(int(i[1::])/10)])-cen_node)) for i in graph], key=lambda x: x[1])
    #organized_node_list = [i[0] for i in temp]
    #for node in organized_node_list:
        
    #for node in node_list:
        # new_color_list contains all colors not including node_color
        #node_color = graph[node][0]
        #new_color_list = list(set(color_list) - set([node_color]))
        
        # new_color_list contains colors of nodes connected to node
        #new_color_list = list(set([graph[i][0] for i in graph[node][1::]]))
        
        # new_color_list contains colors of nodes connected to node ordered by most frequent color
        neighbor_color_list = [graph[i][0] for i in graph[node][1::]]
        neighbor_color_set = list(set(neighbor_color_list))
        temp = sorted([(i, neighbor_color_list.count(i)) for i in neighbor_color_set],key=lambda x: x[1])[::-1]
        new_color_list = [i[0] for i in temp]
        
        for color in new_color_list:
            
            new_graph = change_color(graph,node,color)
            
            while find_neighbors(new_graph):
                new_graph = merge_nodes(new_graph,find_neighbors(new_graph)[0])
            
            new_move_list = copy.deepcopy(move_list)
            new_move_list.append((node,color))
            if len(move_list)<move_limit:
                print len(new_graph)
                soln = solve_graph(new_graph,new_move_list,move_limit)
                if soln:
                    return soln
    return []

# Helper function for k-means that assigns every square a color in accordance
# with the k means m
def assign_to_set(color_board,m):
    for i in color_board:
        dist_list = [scipy.linalg.norm(color_board[i][0]-mi) for mi in m]
        #dist_list = [abs(scipy.exp(1j*color_board[i][0]*pi/180)-scipy.exp(1j*mi*pi/180)) for mi in m]
        #dist_list = [abs(scipy.exp(1j*color_board[i][0][0]*pi/180)-scipy.exp(1j*mi[0]*pi/180))**2 + abs(color_board[i][0][1]-mi[1])**2 for mi in m]
        min_index, min_value = min(enumerate(dist_list), key=operator.itemgetter(1))
        color_board[i][1] = min_index

# Helper function for k-means that determines nad returns the k means m
def find_centroids(color_board,k):
    ms = [0 for i in range(k)]
    ns = [0 for i in range(k)]
    for i in color_board:
        ms[color_board[i][1]] += color_board[i][0]
        #ms[color_board[i][1]] += scipy.array([scipy.exp(1j*color_board[i][0][0]*pi/180), color_board[i][0][1]])
        ns[color_board[i][1]] += 1
    #m = [scipy.array([(180/pi)*cmath.phase(ms[i][0]/ns[i]), ms[i][1]/ns[i]]) for i in range(k)]
    m = [ms[i]/ns[i] for i in range(k)]
    return m
    
# Performs k-means clustering on the colorboard that is the puzzle
def k_means(color_board,k):
    old_color_board = copy.deepcopy(color_board)
    old_color_board[(0,0)][1] += 1
    while sum([old_color_board[i][1]-color_board[i][1] for i in color_board]) != 0:
        old_color_board = copy.deepcopy(color_board)
        m = find_centroids(color_board,k)
        assign_to_set(color_board,m)
    return m
    

def show_color_board(color_board):
    pic = scipy.ones((16,10))
    for i in color_board:
        pic[i[1],i[0]]*=color_board[i][1]
    plt.imshow(pic,interpolation='none')
    
    
def translate_moves(moves,colors):
    #colors = ['Blue', 'Black', 'Red', 'White']
    for move in moves:
        tile_num = int(move[0][1::])
        i = tile_num%10
        j = int((tile_num-i)/10)
        print 'Make (' + str(i) + ',' + str(j) + ') ' + colors[move[1]]
        #plt.figure()
        #plt.imshow(pic + 10*((X-i)**2 + (Y-j)**2<1),interpolation='none')