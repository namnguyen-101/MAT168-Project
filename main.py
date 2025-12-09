from preprocessing import preprocessing
import numpy as np
from pyscipopt import Model, quicksum
from itertools import product
from tabulate import tabulate

courses, rooms = preprocessing()

m = courses.shape[0]
n = rooms.shape[0]
conflict_cap_array = np.zeros((m, n))
time = np.array([i for i in range(8, 18)])
conflict_instructor_list = []

for i, c in enumerate(courses):
    for j, r in enumerate(rooms):
        if c[2] > r[1]:
            conflict_cap_array[i, j] = 0
        else:
            conflict_cap_array[i, j] = 1

np.set_printoptions(threshold=np.inf)

conflict_cap_array = conflict_cap_array.astype(int)

for i in range(m):
    for j in range(i + 1, m):
        if courses[i][1] == courses[j][1]:
            conflict_instructor_list.append((i, j))    

def solve(courses: np.ndarray, rooms: np.ndarray, time: np.ndarray, M: np.ndarray, L: list):
    C = courses.shape[0]
    R = rooms.shape[0]
    T = time.shape[0]
    model = Model()
    x = {}
    y = {}
    valid_indices = np.argwhere(M == 1)
    for c, r in valid_indices:
        for t in range(T):
            x[(c, r, t)] = model.addVar(name=f"x_{c}_{r}_{t}", vtype="B")
    

    for r in range(R):
        y[r] = model.addVar(name=f"y_{r}", vtype="B")

    
    for c in range(C):
        model.addCons(quicksum(x[(c, r, t)] 
                        for r in range(R)
                        for t in range(T)
                        if M[c, r] == 1) == 1)
    
    for r, t in product(range(R), range(T)):
        model.addCons(quicksum(x[(c, r, t)]
                        for c in range(C)
                        if M[c, r] == 1) <= 1)
        
    for r in range(R):
        model.addCons(quicksum(x[(c, r, t)]
                          for c in range(C)
                          for t in range(T)
                          if M[c, r] == 1) <= (C*T + 1) * y[r])
        
    for t in range(T):
        for c1, c2 in L:
            model.addCons(quicksum(x[c1, r, t] for r in range(R) if M[c1, r] == 1) +
                          quicksum(x[c2, r, t] for r in range(R) if M[c2, r] == 1) 
                          <= 1)
        
    model.setObjective(quicksum(y[r] for r in range(R)), sense="minimize")
    model.optimize()
    return model, x

model, x = solve(courses, rooms, time, conflict_cap_array, conflict_instructor_list)

res = {}
for (c, r, t), var in x.items():
    if model.getVal(var) > 0.5:
        res[(r, t)] = c
    
rows = sorted({r for r, _ in res.keys()})
cols = sorted({t for _, t in res.keys()})

table = []
for r in rows:
    row = [f"{rooms[r][0]}\nCapacity = {rooms[r][1]}"]
    for t in cols:
        if res.get((r, t)) == None:
            row.append("")
        else:
            row.append(f"{courses[res.get((r, t))][0]}\nProf:{courses[res.get((r, t))][1]}\nEnrollment = {courses[res.get((r, t))][2]}")
    table.append(row)

headers = [""] + list(time[cols])
print(tabulate(table, headers=headers, tablefmt="grid"))