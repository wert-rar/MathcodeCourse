import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque
import heapq

# ---------------- CONFIG ----------------
ALGORITHM = 'bfs'    # options: 'bfs', 'dfs', 'astar'  (CHANGE)
MAZE_W = 40            # cells (CHANGE)
MAZE_H = 25            # cells (CHANGE)
SPEED = 30             # ms between frames (CHANGE)
USE_GENERATOR = True   # if True, generate maze via backtracker; else load simple pattern (CHANGE)
SEED = None            # for reproducibility (CHANGE)
# ----------------------------------------

if SEED is not None:
    random.seed(SEED)


def make_maze_backtracker(W, H):
    visited = [[False]*W for _ in range(H)]
    passages = [[set() for _ in range(W)] for _ in range(H)]
    stack = [(0,0)]
    visited[0][0] = True
    while stack:
        y, x = stack[-1]
        nbrs = []
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W and not visited[ny][nx]:
                nbrs.append((ny, nx))
        if nbrs:
            ny, nx = random.choice(nbrs)
            passages[y][x].add((ny,nx))
            passages[ny][nx].add((y,x))
            visited[ny][nx] = True
            stack.append((ny, nx))
        else:
            stack.pop()
    return passages

def passages_to_grid(passages):
    H = len(passages); W = len(passages[0])
    gh, gw = 2*H+1, 2*W+1
    grid = np.ones((gh, gw), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            gy, gx = 2*y+1, 2*x+1
            grid[gy,gx] = 0
            for ny,nx in passages[y][x]:
                gy2, gx2 = 2*ny+1, 2*nx+1
                grid[(gy+gy2)//2, (gx+gx2)//2] = 0
    return grid


passages = make_maze_backtracker(MAZE_W, MAZE_H)
grid = passages_to_grid(passages)
start = (1,1)  # grid coords (y,x)
goal = (grid.shape[0]-2, grid.shape[1]-2)


def neighbors_grid(y,x):
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        ny, nx = y+dy, x+dx
        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] and grid[ny,nx]==0:
            yield ny, nx


def bfs(start, goal):
    q = deque([start])
    parent = {start: None}
    order = []
    while q:
        cur = q.popleft()
        order.append(cur)
        if cur == goal:
            break
        for nb in neighbors_grid(*cur):
            if nb not in parent:
                parent[nb] = cur
                q.append(nb)
    path = []
    if goal in parent:
        cur = goal
        while cur:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
    return order, path

def dfs(start, goal):
    stack = [start]
    parent = {start: None}
    order = []
    visited = set([start])
    while stack:
        cur = stack.pop()
        order.append(cur)
        if cur == goal:
            break
        for nb in neighbors_grid(*cur):
            if nb not in visited:
                visited.add(nb)
                parent[nb] = cur
                stack.append(nb)
    path = []
    if goal in parent:
        cur = goal
        while cur:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
    return order, path

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(start, goal):
    open_heap = []
    heapq.heappush(open_heap, (0 + manhattan(start,goal), 0, start))
    parent = {start: None}
    gscore = {start: 0}
    order = []
    closed = set()
    while open_heap:
        f, gcur, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        order.append(cur)
        if cur == goal:
            break
        for nb in neighbors_grid(*cur):
            tentative_g = gcur + 1
            if nb in closed:
                continue
            if tentative_g < gscore.get(nb, 1e9):
                gscore[nb] = tentative_g
                parent[nb] = cur
                heapq.heappush(open_heap, (tentative_g + manhattan(nb,goal), tentative_g, nb))
    path = []
    if goal in parent:
        cur = goal
        while cur:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
    return order, path

# Выбор алгоритма
if ALGORITHM == 'bfs':
    order, path = bfs(start, goal)
elif ALGORITHM == 'dfs':
    order, path = dfs(start, goal)
elif ALGORITHM == 'astar':
    order, path = astar(start, goal)
else:
    raise ValueError("Unknown ALGORITHM")

# Prepare visualization frames:
# We'll color: 1=wall(black), 0=free(white), 2=visited(blue), 3=path(green), 4=start(red), 5=goal(yellow)
frames = []
color_grid = np.zeros((grid.shape[0], grid.shape[1]), dtype=np.uint8)
color_grid[grid==1] = 1
color_grid[grid==0] = 0
color_grid[start] = 4
color_grid[goal] = 5
frames.append(color_grid.copy())

# add frames for order (visited)
for idx, cell in enumerate(order):
    if cell != start and cell != goal:
        color_grid[cell] = 2
        frames.append(color_grid.copy())

# then add frames for path
for cell in path:
    if cell != start and cell != goal:
        color_grid[cell] = 3
        frames.append(color_grid.copy())

# visualization
cmap = plt.cm.get_cmap('tab20c')
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
im = ax.imshow(frames[0], interpolation='nearest')
title = ax.set_title(f"Solver: {ALGORITHM}")

def update(i):
    im.set_data(frames[i])
    return (im,)

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=SPEED, blit=True, repeat=False)
plt.show()