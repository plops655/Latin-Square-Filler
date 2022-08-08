import pygame, sys, random, copy
from pygame.locals import *
from collections import defaultdict, deque

pygame.init()

ROWS=COLUMNS=color_count=20
WINDOW_LENGTH=400
WINDOW_SIZE=(WINDOW_LENGTH, WINDOW_LENGTH)
gap=WINDOW_LENGTH/ROWS
screen=pygame.display.set_mode(WINDOW_SIZE, 0, 32)
pygame.display.set_caption("Latin Squares Filler")
clicked=(0,0) #initialize clicked to this after each setting
background_color="ORANGE"
colors=[]
filled_cells=0
run=False
lose=False
color_index=color_count-1

class Cell:
    clicked_alg2=defaultdict(list)
    clicked_cells=defaultdict(list)
    red_edges=defaultdict(set) #paired B vertices and corresponding edges
    #unpaired_vertices_A=[] #unpaired A vertices (eventually empty via Hopcroft_Karp and Hall's theorem)
    #unpaired_vertices_B=[] #unpaired A vertices (eventually empty via Hopcroft_Karp and Hall's theorem)
    def __init__(self, row, col, value, color):
        self.row=row
        self.col=col
        self.value=value
        self.color=color
        self.clicked=False

    def set_clicked(self):
        Cell.clicked_cells["R"+str(self.row)].append(self.col)
        Cell.clicked_cells["V"+str(self.value)].append((self.row, self.col))
        Cell.clicked_cells["C"+str(self.col)].append(self.row)
        self.clicked=True

    def fill(self, color, value):
        col=self.col
        row=self.row
        self.color=color
        self.value=value
        rect=Rect((col-1)*gap, (row-1)*gap, gap, gap)
        screen.fill(Color(color), rect)
        pygame.display.flip()
        pygame.draw.line(screen, "BLACK", ((col-1)*gap, (row-1)*gap),(col*gap, (row-1)*gap))
        pygame.draw.line(screen, "BLACK", ((col-1)*gap, (row-1)*gap),((col-1)*gap, row*gap))

    def __lt__(self, other):
        return False

for row in range(1, ROWS+1):
    Cell.clicked_cells["R"+str(row)]=list()
for col in range(1, COLUMNS+1):
    Cell.clicked_cells["C"+str(col)]=list()

cells=defaultdict(Cell)
alg_med=defaultdict(int)

def pretty_print(pi_row, pi_col, unmarked_set):
    for r in range(1, ROWS+1):
        line=""
        for c in range(1, COLUMNS+1):
            unmarked=False
            for v in unmarked_set:
                if (pi_row[r], pi_col[c]) in Cell.clicked_cells["V"+str(v)]:
                    line+=(str(v)+ " ")
                    unmarked=True
            if not unmarked:
                line+="0 "
        print(line)
    
    print("")

        
    

golden_ratio_inverse=0.618033988749895
S, V = 0.99, 0.99

def check_color(row, col):
    global lose 
    for c in range(1, COLUMNS+1):
        if(cells[(row,c)].clicked):
            if(cells[(row,col)].value==cells[(row,c)].value):
                pygame.draw.line(screen, "RED", ((col-0.5)*gap, (row-0.5)*gap), ((c-0.5)*gap, (row-0.5)*gap))
                lose=True
    for r in range(1, ROWS+1):
        if(cells[(r,col)].value):
            if(cells[(row,col)].value==cells[(r,col)].value):
                pygame.draw.line(screen, "RED", ((col-0.5)*gap, (row-0.5)*gap), ((col-0.5)*gap, (r-0.5)*gap))
                lose=True

def hsv_to_rgb(h,s,v, index):
    c=s*v
    h_new=h*6
    x=c*(1-abs(h_new % 2 -1))
    if(0<=h_new<1):
        r,g,b=c,x,0
    if(1<=h_new<2):
        r,g,b=x,c,0
    if(2<=h_new<3):
        r,g,b=0,c,x
    if(3<=h_new<4):
        r,g,b=0,x,c
    if(4<=h_new<5):
        r,g,b=x,0,c
    if(5<=h_new<6):
        r,g,b=c,0,x
    m=v-c
    color=(int(256*(r+m)), int(256*(g+m)), int(256*(b+m)))
    colors.append((color, index+1)) 


H=random.random()
for c in range(0, color_count):
    if(c % 4 == 0):
        S-=0.1
        V-=0.1
    hsv_to_rgb(H, S, V, c)
    H+=golden_ratio_inverse
    H = H % 1

def set_red(edge_tuple):
    Cell.red_edges[edge_tuple[1]]=edge_tuple[0] #B to A

def DFS1(parents_dictionary, curr, upper_level): #upper_level is a set
    lower_level=set(curr) #if this leads to errors just use for loop
    curr_vert=[[i] for i in curr]
    unclicked_upper=upper_level.copy()
    unclicked_lower=set()
    paths=[]
    path_curr=[]
    marked=set()
    while curr_vert:
        while curr_vert[-1]:
            if path_curr:
                if curr_vert[-1][-1] not in parents_dictionary[path_curr[-1]]:
                    if path_curr[-1] in lower_level:
                        unclicked_lower.add(path_curr[-1])
                    path_curr.pop()
                    break
            val=curr_vert[-1].pop()
            if val in upper_level:
                path_curr.append(val)
                paths.append(path_curr)
                marked.update(path_curr)
                unclicked_upper.remove(val)
                path_curr=[]
                curr_vert.pop()
                break
            parents_exist=False
            for p in parents_dictionary[val]:
                if p not in marked:
                    parents_exist=True
                    curr_vert[-1].append(p)
            if parents_exist:
                path_curr.append(val)
            else:
                if val in lower_level: 
                    unclicked_lower.add(val)
            if not curr_vert[-1]: 
                curr_vert.pop()
                break
            
    return paths, unclicked_lower, unclicked_upper


def DFS(parents_dictionary, curr, upper_level): #upper_level is a set
    curr_vert=[[i] for i in curr]
    actual_upper=[e+str(1) for e in upper_level]
    unclicked_upper=upper_level.copy()
    unclicked_lower=set(i[:-1] for i in curr)
    paths=[]
    path_curr=[]
    marked=set()
    while curr_vert:
        while curr_vert[-1]:
            if path_curr:
                if curr_vert[-1][-1] not in parents_dictionary[path_curr[-1]]:
                    path_curr.pop()
                    break
            val=curr_vert[-1].pop()
            if val in actual_upper:
                path_curr.append(val)
                paths.append(path_curr)
                marked.update(path_curr)
                path_curr=[]
                curr_vert.pop()
                break
            parents_exist=False
            for p in parents_dictionary[val]:
                if p not in marked:
                    parents_exist=True
                    curr_vert[-1].append(p)
            if parents_exist:
                path_curr.append(val)
            if not curr_vert[-1]: 
                curr_vert.pop()
                break
    for path in paths:
        for i in range(0, len(path)):
            path[i]=path[i][:-1]
        unclicked_lower.remove(path[0])
        unclicked_upper.remove(path[-1])
    #what if we don't care about unclicked_lower, upper
    return paths, unclicked_lower, unclicked_upper
#paths, unclicked_lower, unclicked_upper
def twolist_into_dict(input_twolist): #twolist is [[a,b,c], [d, ...], ...]
    output_dict=defaultdict(str)
    for list in input_twolist:
        for i in range(len(list)):
            if i%2 == 0:
                output_dict[list[i]]=list[i+1]
    output_reversed=dict(reversed(i) for i in output_dict.items())
    output_dict=output_dict | output_reversed 
    return output_dict

def BFS(edge_dictionary, start_vert, end_vert, pair_edges, A_letter, B_letter): #start/end_vert are sets. pair_edges~M(dict)
    parents_dictionary=defaultdict(set)
    level=1
    A_level=True
    queue=deque(start_vert) 
    encounter_unpaired=set()
    while queue:
        val=queue.popleft()
        if(val[0]==A_letter):
            if not A_level:
                level+=1
                A_level=True
            if(val not in start_vert):
                for neighbor in edge_dictionary[val]:
                    if(neighbor not in pair_edges[val]):
                        parents_dictionary[neighbor+str(level+1)].add(val+str(level))
                        if(neighbor in end_vert):
                            encounter_unpaired.add(neighbor+str(level+1))
                        queue.append(neighbor)
            else:
                for neighbor in edge_dictionary[val]:
                    parents_dictionary[neighbor+str(level+1)].add(val+str(level))
                    if(neighbor in end_vert):
                            encounter_unpaired.add(neighbor+str(level+1))
                    queue.append(neighbor)

        elif(val[0]==B_letter):
            if encounter_unpaired:
                return parents_dictionary, encounter_unpaired
            if A_level:
                level+=1
                A_level=False
            if val in pair_edges.keys():
                parents_dictionary[pair_edges[val]+str(level+1)].add(val+str(level))
                queue.append(pair_edges[val])
    return parents_dictionary, encounter_unpaired #encounter_unpaired is bottom set of DFS graph
#parent_dictionary, encounter_unpaired

def initialize_pairings(edge_dictionary, A_letter, B_letter):
    A_list=[]
    B_set=set()
    M_med=[]
    M=defaultdict(str)
    for vertex in edge_dictionary.keys():
        if(vertex[0]==A_letter):
            A_list.append(vertex)
        elif(vertex[0]==B_letter):
            B_set.add(vertex)
    
    M_med, unclicked_A, unclicked_B = DFS1(edge_dictionary, A_list, B_set)
    M=twolist_into_dict(M_med)
    return M, unclicked_A, unclicked_B


def Hopcroft_Karp(edge_dictionary, A_letter, B_letter, M, unclicked_A, unclicked_B):#edit hopcroft karp in main alg
    if(not unclicked_A):
        return M
    #wrong
    parents_dictionary, free_B = BFS(edge_dictionary, unclicked_A, unclicked_B, M, A_letter, B_letter)
    P, unclicked_in_free_B, unclicked_A = DFS(parents_dictionary, free_B, unclicked_A)
    free_B=set(i[:-1] for i in free_B)
    unclicked_B=unclicked_B-free_B | unclicked_in_free_B
    M_new=defaultdict(str)
    for list in P:
        for i in range(len(list)):
            if (i%2 == 0):
                M_new[list[i]]=list[i+1]
    M_new_inv=dict(reversed(i) for i in M_new.items())
    M_new=M_new | M_new_inv
    M=M | M_new
    return Hopcroft_Karp(edge_dictionary, A_letter, B_letter, M, unclicked_A, unclicked_B)

#edge_dict~graph, A/B_letter (bipartition), M~pairing, unclicked_A/B~free vertices per partition
def merge(list1, list2):
    list=[]
    i, j = 0, 0
    while i<len(list1) and j<len(list2):
        if(list1[i]>=list2[j]):
            list.append(list1[i])
            i+=1
        else:
            list.append(list2[j])
            j+=1
    if i==len(list1):
        for index in range(j, len(list2)):
            list.append(list2[index])
        return list
    for index in range(i, len(list1)):
            list.append(list1[index])
    return list

def mergeSort(list):
    if len(list)==1:
        return list
    med=len(list)//2
    list1=mergeSort(list[:med])
    list2=mergeSort(list[med:])
    return merge(list1, list2)

#fix edge definition
def algorithm1(pi_row, pi_col, j, marked_set=None, unmarked_set=None): 
    global colors
    unclicked_values=[]
    ordered_values=defaultdict(list)
    values_list=set()
    for v in unmarked_set: 
        if(Cell.clicked_cells["V"+str(v)]):
            ordered_values[len(Cell.clicked_cells["V"+str(v)])].append(v)
            values_list.add(len(Cell.clicked_cells["V"+str(v)]))
        else:
            unclicked_values.append(v)
    values_list=list(values_list)
    values_list=mergeSort(values_list)
    for values in values_list:
        for v in ordered_values[values]:
            edges=defaultdict(list)
            column_list=["C"+str(pi_col[c]) for c in range(1, COLUMNS+2-j)]
            row_list=["R"+str(pi_row[r]) for r in range(j, ROWS+1)]
            M, unclicked_R, unclicked_C = defaultdict(str), set(), set()
            for tuple in Cell.clicked_cells["V"+str(v)]: #tuple[0], [1] ~ row, col
                row_list.remove("R"+str(tuple[0]))
                column_list.remove("C"+str(tuple[1]))
            for r in row_list:
                edges[r]=column_list.copy()
            for c in column_list:
                edges[c]=row_list.copy()
                for row in Cell.clicked_cells[c]: 
                    if "R"+str(row) in row_list:
                        edges["R"+str(row)].remove(c)
                        edges[c].remove("R"+str(row))
                        #what is happening
            M, unclicked_R, unclicked_C = initialize_pairings(edges, "R", "C")
            M=Hopcroft_Karp(edges, "R", "C", M, unclicked_R, unclicked_C)
            for key in M.keys():
                if key[0]=="R":
                    cells[(int(key[1:]), int(M[key][1:]))].fill(colors[v-1][0], v)
                    cells[(int(key[1:]), int(M[key][1:]))].set_clicked()
    #check how edges is made
    for v in unclicked_values:
        edges=defaultdict(list)
        column_list=["C"+str(pi_col[c]) for c in range(1, COLUMNS+2-j)]
        row_list=["R"+str(pi_row[r]) for r in range(j, ROWS+1)]
        M_temp, unclicked_R, unclicked_C = defaultdict(str), set(), set()
        for r in row_list:
            edges[r]=column_list.copy()
        #we need to make value_special dissappear
        for c in column_list:
            edges[c]=row_list.copy()
            for row in Cell.clicked_cells[c]:
                if "R"+str(row) in row_list: 
                    edges["R"+str(row)].remove(c)
                    edges[c].remove("R"+str(row))
        M_temp, unclicked_R, unclicked_C = initialize_pairings(edges, "R", "C")
        M=Hopcroft_Karp(edges, "R", "C", M_temp, unclicked_R, unclicked_C)
        for key in M.keys():
            if key[0]=="R":
                cells[(int(key[1:]), int(M[key][1:]))].fill(colors[v-1][0], v)
                cells[(int(key[1:]), int(M[key][1:]))].set_clicked()
    if unmarked_set==set(range(1, ROWS+1)):
        print(True)

def swap(perm, i, j, perm_inv=None):
    med=perm[i]
    perm[i]=perm[j]
    if perm_inv is not None:
        perm_inv[perm[j]]=i
        perm_inv[med]=j
    perm[j]=med
    
def exchange(pi_row, pi_col, val, marked, row_dict, row, col, obs):
    exchange=True
    row1=row
    value1=val
    while exchange:
        #below line is an issue
        value=cells[(pi_row[row1], pi_col[col])].value
        cells[(pi_row[row1], pi_col[col])].fill(colors[value1-1][0], value1)
        cells[(pi_row[row1], pi_col[col])].set_clicked()
        Cell.clicked_cells["V"+str(value)].remove((pi_row[row1], pi_col[col]))
        cells[(pi_row[row1], pi_col[COLUMNS-obs])].fill(colors[value-1][0], value)
        cells[(pi_row[row1], pi_col[COLUMNS-obs])].set_clicked()
        if value1 != val:
            Cell.clicked_cells["V"+str(value1)].remove((pi_row[row1], pi_col[COLUMNS-obs]))
        if value in marked: #marked starts empty
            temp=row_dict[value]
            row_dict[value]=row1
            row1=temp
            value1=value
        else:
            marked.add(value)
            row_dict[value]=row1
            exchange=False
            break

def final_swap(pi_row, pi_col, obs, val, marked, unmarked_set):
    row_dict=defaultdict(int)
    #colors is the issue
    cells[(pi_row[obs+1], pi_col[1])].fill(colors[val-1][0], val)
    cells[(pi_row[obs+1], pi_col[1])].set_clicked()
    cells[(pi_row[ROWS], pi_col[COLUMNS-obs])].fill(colors[val-1][0], val)
    cells[(pi_row[ROWS], pi_col[COLUMNS-obs])].set_clicked()
    for i in range(obs+2, ROWS):
        exchange(pi_row, pi_col, val, marked, row_dict, i, i-obs, obs)
    for c in range(2, ROWS+1-obs):
        for v in unmarked_set:
            not_in=True
            for r in range(obs+2, ROWS+1):
                if cells[(pi_row[r], pi_col[c])].value == v:
                    not_in=False
                    break
            if not_in:    
                unmarked_set.remove(v)
                cells[(pi_row[obs+1], pi_col[c])].fill(colors[v-1][0], v)
                cells[(pi_row[obs+1], pi_col[c])].set_clicked()
                break
            else:
                continue

def alg2_sub(pi_row, pi_col, pi_rinv, pi_cinv, j, marked_set, row_special, fixed, rows):
    rows_copy=rows.copy()
    swap(pi_row, pi_rinv[row_special], len(Cell.clicked_alg2["R"+str(row_special)])+j-1, pi_rinv)
    index=len(Cell.clicked_alg2["R"+str(row_special)])+1 
    while rows:
        row=rows.pop()
        index+=len(Cell.clicked_alg2["R"+str(row)])
        swap(pi_row, pi_rinv[row], index+j-1, pi_rinv)

    #pi_cinv=dict([reversed(i) for i in pi_col.items()])
    #fixed=singlet[1] 
    #this line is affected by emptying clicked_cells heeeeeeeeee
    swap(pi_col, pi_cinv[fixed], len(Cell.clicked_alg2["R"+str(row_special)]), pi_cinv) 
    marked=1
    for col in Cell.clicked_alg2["R"+str(row_special)]:
        if cells[(row_special, col)].value in marked_set or marked>pi_cinv[col] or col==fixed: 
            continue 
        while pi_col[marked] in Cell.clicked_alg2["R"+str(row_special)]:
            marked+=1
        if marked<=pi_cinv[col]:
            swap(pi_col, pi_cinv[col], marked, pi_cinv)
            marked+=1
    for row in rows_copy: 
        for col in Cell.clicked_alg2["R"+str(row)]: 
            if cells[(row, col)].value in marked_set or marked>pi_cinv[col]: 
                continue 
            while pi_col[marked] in Cell.clicked_alg2["R"+str(row)] or marked == pi_cinv[fixed]:
                marked+=1 
            if marked<=pi_cinv[col]:
                swap(pi_col, pi_cinv[col], marked, pi_cinv)
                marked+=1
    return rows_copy
#pi_row, pi_col, j, marked_set ~ set, dict, int, defaultdict
def algorithm2(pi_row, pi_col, pi_rinv, pi_cinv, j, marked_set, unmarked): 
    #pi_rinv=dict([reversed(i) for i in pi_row.items()])

    for index in Cell.clicked_alg2.keys():
        if index[0] != "V":
            continue
        if int(index[1:]) in marked_set: 
            continue
        if len(Cell.clicked_alg2[index])==1: 
            singlet = Cell.clicked_alg2[index][0] #list(Cell.clicked_alg2[index])[0] 
            row_special=singlet[0]
            fixed=singlet[1] 
            value_special=index[1:] 
            break
    rows=set() 

    for i in range(j, ROWS+1): 
        r=pi_row[i]
        if r == row_special: 
            continue
        if Cell.clicked_alg2["R"+str(r)]:
            rows.add(r)
    #assume alg2_sub correctly permutes rows and columns
    rows_copy=alg2_sub(pi_row, pi_col, pi_rinv, pi_cinv, j, marked_set, row_special, fixed, rows)

    marked_set.add(int(value_special))
    unmarked.remove(int(value_special))
    Cell.clicked_alg2["R"+str(singlet[0])].remove(singlet[1])
    Cell.clicked_alg2["C"+str(singlet[1])].remove(singlet[0])
    Cell.clicked_alg2["V"+value_special].remove((singlet[0], singlet[1]))
    Cell.clicked_cells["R"+str(singlet[0])].remove(singlet[1])
    Cell.clicked_cells["C"+str(singlet[1])].remove(singlet[0])
    Cell.clicked_cells["V"+value_special].remove((singlet[0], singlet[1]))
    #at this stage we have cells with an associated value (val), but whose dictionary[val] is empty (possibility of ValueError)
    algorithm(pi_row, pi_col, pi_rinv, pi_cinv, j+1, marked_set, unmarked, ROWS-j) 

    Cell.clicked_alg2["R"+str(singlet[0])].append(singlet[1])
    Cell.clicked_alg2["C"+str(singlet[1])].append(singlet[0])
    Cell.clicked_alg2["V"+value_special].append((singlet[0], singlet[1]))

    marked_set.remove(int(value_special))
    unmarked_set.add(int(value_special))

    alg2_sub(pi_row, pi_col, pi_rinv, pi_cinv, j, marked_set, row_special, fixed, rows_copy)
    final_swap(pi_row, pi_col, j-1, int(value_special), set(), unmarked_set.copy())
    if unmarked_set==set(range(1, ROWS+1)):
        print(True)

def algorithm(pi_row, pi_col, pi_rinv, pi_cinv, j, marked_set, unmarked_set, rows):
    if(color_index>=rows/2-1):
        algorithm1(pi_row, pi_col, j, marked_set, unmarked_set)
    else:
        algorithm2(pi_row, pi_col, pi_rinv, pi_cinv, j, marked_set, unmarked_set)

screen.fill("ORANGE")

screen.fill(background_color)
#initialize cell objects

for r in range(1, ROWS+1):
    for c in range(1, COLUMNS+1):
        cells[(r,c)]=Cell(r,c,0, None)
        cells[(r,c)].fill(background_color,0)

#pretty_print(cells, 1, COLUMNS)

while True:
    events = pygame.event.get()
    for event in events:
        if event.type==QUIT:
            pygame.quit()
            sys.exit()

        if (event.type == pygame.MOUSEBUTTONDOWN) and not run:
            #pretty_print(cells, 1, COLUMNS)
            x, y = pygame.mouse.get_pos()
            row, col = y//gap+1, x//gap+1
            row, col = int(row), int(col)
            if cells[(row,col)].clicked:
                rect=Rect((col-1)*gap, (row-1)*gap, gap, gap)
                if (clicked != (0,0)):
                    cells[clicked].fill(cells[clicked].color, cells[clicked].value)
                screen.fill("RED", rect)
                clicked=(row,col)
            elif not cells[(row,col)].clicked and clicked != (0,0):
                cells[(row, col)].fill(cells[clicked].color, cells[clicked].value)
                cells[(row,col)].set_clicked()
                cells[clicked].fill(cells[clicked].color, cells[clicked].value)
                clicked=(0,0)
                filled_cells+=1
            else:
                values=colors[color_index]
                color_curr, val_curr = values[0], values[1]
                cells[(row, col)].fill(color_curr, val_curr)
                cells[(row,col)].set_clicked()
                filled_cells+=1
                color_index-=1
            if filled_cells==color_count-1:
                run=True
                Cell.clicked_alg2=copy.deepcopy(Cell.clicked_cells)
                print(Cell.clicked_cells)
                for row in range(1, ROWS):
                    for col in Cell.clicked_cells["R"+str(row)]:
                        check_color(row, col)
                if(lose==True):
                    print(False)
                pi_row=dict((r,r) for r in range(1, ROWS+1))
                pi_rinv=dict((r,r) for r in range(1, ROWS+1))
                pi_col=dict((c,c) for c in range(1, COLUMNS+1))
                pi_cinv=dict((c,c) for c in range(1, COLUMNS+1))
                unmarked_set=set(range(1, ROWS+1))
                algorithm(pi_row, pi_col, pi_rinv, pi_cinv, 1, set(), unmarked_set, ROWS)

    pygame.display.update()


