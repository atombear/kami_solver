import cv2
import scipy
import matplotlib.pyplot as plt
from kami_solver import kami_solve
import operator
from scipy.constants import pi
        

BGR_weights = scipy.array([0.1140,0.5870,0.2989])


kami_dir = '/Users/alexanderpapageorge/Kami7.png'
colors = ['Blue','Red','White','Black']
colors = ['Blue','Red','Black']
k = len(colors)


#colors = ['Blue','White','Red','Black']
m = [0 for i in range(k)]
#color_dict = {'Blue': scipy.array([ 132.,154.,49.]), 'Black': scipy.array([52.,74.,89.]), 'Red': scipy.array([77.,83.,231.]), 'White': scipy.array([90.,123.,153.])}
#color_dict = {'Blue': scipy.array([ 1.1,1.28,.4,121/100]), 'Black': scipy.array([.68,.97,1.16,79/100]), 'Red': scipy.array([.61,.67,1.77,125/100]), 'White': scipy.array([.74,.98,1.13,195/100])}
color_dict = {'Blue': scipy.array([167,.4]), 'Black': scipy.array([35,.2]), 'Red': scipy.array([2,.6]), 'White': scipy.array([35,.7])}
#color_dict = {'Blue': scipy.array([167,.6,.8]), 'Black': scipy.array([35,.1,.2]), 'Red': scipy.array([2,.6,.7]), 'White': scipy.array([35,.4,.7])}
color_dict = {'Blue': scipy.array([212,-88,13]), 'Black': scipy.array([166,17,45]), 'Red': scipy.array([193,134,48]), 'White': scipy.array([230,19,52])}


for ni,i in enumerate(colors):
    m[ni] = color_dict[i]
    
kami = cv2.imread(kami_dir)
kami = kami[0:1720,:,:]

x = scipy.arange(10)
y = scipy.arange(16)
X,Y = scipy.meshgrid(x,y)

color_board = {}
for i in range(10):
    for j in range(16):
        p = (i,j)
        xc = 108*i+54
        yc = 108*j+54
        color = (kami[yc-20:yc+20,xc-20:xc+20].sum(0).sum(0)/1600)
        color_hsl = scipy.array([kami_solve.rgb2hsl(color[::-1],255)[l] for l in [0,2]])
        #color_hsl = scipy.array(kami_solve.rgb2hsl(color[::-1],255))
        
        color_Lab = kami_solve.rgb2CIELAB(color[::-1])

        #gw = color.dot(BGR_weights)
        #color = color/gw
        #color = scipy.append(color,gw/100)
        
        #dist_list = [scipy.linalg.norm(color-mi) for mi in m]
        #dist_list = [abs(scipy.exp(1j*color_hsl[0]*pi/180)-scipy.exp(1j*mi[0]*pi/180)) + abs(color_hsl[1]-mi[1]) for mi in m]
        dist_list = [scipy.linalg.norm(color_Lab-mi) for mi in m]
        
        min_index, min_value = min(enumerate(dist_list), key=operator.itemgetter(1))
        #color_board[p] = [color,min_index]
        #color_board[p] = [color_hsl,min_index]
        color_board[p] = [color_Lab,min_index]

kami = scipy.fliplr(kami.reshape(-1,3)).reshape(kami.shape)
plt.imshow(kami)
plt.figure()
kami_solve.show_color_board(color_board)

m = kami_solve.k_means(color_board,k)
plt.figure()
kami_solve.show_color_board(color_board)

plt.show()

c = [ [], [], [], [] ]
for i in color_board:
    c[color_board[i][1]].append(i)
while [] in c:
    c.remove([])
graph = kami_solve.determine_graph(c)

moves = kami_solve.solve_graph(graph,[],4)
kami_solve.translate_moves(moves,colors)



#c0 = []
#c1 = []
#c2 = []
#c3 = []


#c = [c0, c1, c2, c3]

        #print color
        #red > 190
        #if color[2]>150:
        #    c2.append(p)
        #    color_board[p] = [color,2]
        #else:
        #    if (color[0]<100):
        #        c1.append(p)
        #        color_board[p] = [color,1]
        #    else:
        #        c0.append(p)
        #        color_board[p] = [color,0]

        #if (color[2]>150) & (color[1]>122):
        #    c1.append((i,j))
        #    color_board[p] = [color,1]
        #else:
        #    if (color[2]>170):
        #        c2.append((i,j))
        #        color_board[p] = [color,2]
        #    else:
        #        c0.append((i,j))
        #        color_board[p] = [color,0]
        
        #if (color[2]>150) & (color[1]>122):
        #    c1.append((i,j))
        #    color_board[p] = [color,1]
        #else:
        #    if (color[2]>150):
        #        c2.append((i,j))
        #        color_board[p] = [color,2]
        #    else:
        #        if (color[0]>92.64):
        #            c0.append((i,j))
        #            color_board[p] = [color,0]
        #        else:
        #            c3.append((i,j))
        #            color_board[p] = [color,3]
        
#for i in c0:
#    pic[i[1],i[0]]*=0
#for i in c1:
#    pic[i[1],i[0]]*=1
#for i in c2:
#    pic[i[1],i[0]]*=2
#for i in c3:
#    pic[i[1],i[0]]*=3
