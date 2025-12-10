"""
maze_generator_visual.py

Генератор лабиринтов с анимацией в matplotlib.
Поддерживаемые алгоритмы:
 - 'backtracker' (рекурсивный backtracker)
 - 'prim' (рандомизированный Prim)

Как использовать:
    python maze_generator_visual.py

CHANGE: в блоке CONFIG можно менять параметры: WIDTH, HEIGHT, ALGORITHM, SPEED.
"""

import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque

# ---------------- CONFIG ----------------
WIDTH = 40       # number of cells horizontally (CHANGE)
HEIGHT = 25      # number of cells vertically (CHANGE)
CELL_SIZE = 1    # визуальный масштаб
ALGORITHM = 'prim'  # 'backtracker' or 'prim' (CHANGE)
SPEED = 30       # колличество миллисекунд между кадрами
SEED = None      # int or None (CHANGE to reproduce)
# ----------------------------------------

if SEED is not None:
    random.seed(SEED)

# Grid for drawing: we will build a (2*H+1) x (2*W+1) grid of 0/1 (0 = passage, 1 = wall)
grid_h = 2 * HEIGHT + 1
grid_w = 2 * WIDTH + 1

def empty_grid():
    g = np.ones((grid_h, grid_w), dtype=np.uint8)
    # mark cell centers as passages (we will carve between them)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            gy, gx = 2*y+1, 2*x+1
            g[gy, gx] = 0
    return g

# Utility: convert cell coords <-> grid coords
def cell_to_grid(c):
    y, x = c
    return 2*y+1, 2*x+1

# Backtracker algorithm (iterative)
def generate_backtracker():
    visited = [[False]*WIDTH for _ in range(HEIGHT)]
    stack = []
    order = []  # list of grid states (for animation) -> we will yield the updates incrementally
    g = empty_grid()
    start = (0, 0)
    stack.append(start)
    visited[0][0] = True
    order.append(g.copy())
    while stack:
        y, x = stack[-1]
        # find unvisited neighbors
        nbrs = []
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < HEIGHT and 0 <= nx < WIDTH and not visited[ny][nx]:
                nbrs.append((ny, nx))
        if nbrs:
            ny, nx = random.choice(nbrs)
            # remove wall between (y,x) and (ny,nx)
            gy1, gx1 = cell_to_grid((y,x))
            gy2, gx2 = cell_to_grid((ny,nx))
            wall_y, wall_x = (gy1+gy2)//2, (gx1+gx2)//2
            g[wall_y, wall_x] = 0
            visited[ny][nx] = True
            stack.append((ny, nx))
            order.append(g.copy())
        else:
            stack.pop()
            # we still append to visualize backtracking if desired (optional)
            order.append(g.copy())
    return order

# Prim algorithm (randomized)
def generate_prim():
    g = empty_grid()
    in_maze = [[False]*WIDTH for _ in range(HEIGHT)]
    walls = []  # walls are tuples: (cell_from, cell_to)
    start = (random.randrange(HEIGHT), random.randrange(WIDTH))
    in_maze[start[0]][start[1]] = True
    def push_walls(cell):
        y, x = cell
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < HEIGHT and 0 <= nx < WIDTH and not in_maze[ny][nx]:
                walls.append((cell, (ny,nx)))
    push_walls(start)
    order = [g.copy()]
    while walls:
        idx = random.randrange(len(walls))
        a, b = walls.pop(idx)
        by, bx = b
        if in_maze[by][bx]:
            continue
        # connect a <-> b
        ay, ax = a
        gy1, gx1 = cell_to_grid((ay,ax))
        gy2, gx2 = cell_to_grid((by,bx))
        wall_y, wall_x = (gy1+gy2)//2, (gx1+gx2)//2
        g[wall_y, wall_x] = 0
        in_maze[by][bx] = True
        push_walls(b)
        order.append(g.copy())
    return order

# Select generator
if ALGORITHM == 'backtracker':
    frames = generate_backtracker()
elif ALGORITHM == 'prim':
    frames = generate_prim()
else:
    raise ValueError("Unknown ALGORITHM: choose 'backtracker' or 'prim'")

# Set up matplotlib figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title(f"Maze generator — {ALGORITHM}")
ax.axis('off')
cmap = plt.cm.get_cmap('gray_r')  # white passages on dark background
im = ax.imshow(frames[0], cmap=cmap, interpolation='nearest')

def update(i):
    im.set_data(frames[i])
    return (im,)

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=SPEED, blit=True, repeat=False)
plt.show()
