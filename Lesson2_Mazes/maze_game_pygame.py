import pygame
import sys
import random
from collections import deque
import heapq

# ---------------- CONFIG ----------------
CELL_PIX = 20
MAZE_W = 30
MAZE_H = 20
FPS = 60
SEED = None
MOVE_DELAY_MS = 100
# ----------------------------------------

if SEED is not None:
    random.seed(SEED)

# Maze generation (backtracker) but storing passages
def generate_maze(W, H):
    visited = [[False]*W for _ in range(H)]
    passages = [[set() for _ in range(W)] for _ in range(H)]
    stack = [(0,0)]
    visited[0][0] = True
    while stack:
        y,x = stack[-1]
        nbrs = []
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W and not visited[ny][nx]:
                nbrs.append((ny,nx))
        if nbrs:
            ny,nx = random.choice(nbrs)
            passages[y][x].add((ny,nx))
            passages[ny][nx].add((y,x))
            visited[ny][nx] = True
            stack.append((ny,nx))
        else:
            stack.pop()
    return passages

def passages_to_walls(passages):
    H = len(passages); W = len(passages[0])
    # walls for cells: True if wall between cell and neighbor (or border)
    # We will build a grid 2*H+1 x 2*W+1 of 0/1 like before
    gh, gw = 2*H+1, 2*W+1
    grid = [[1]*gw for _ in range(gh)]
    for y in range(H):
        for x in range(W):
            gy, gx = 2*y+1, 2*x+1
            grid[gy][gx] = 0
            for ny,nx in passages[y][x]:
                gy2, gx2 = 2*ny+1, 2*nx+1
                grid[(gy+gy2)//2][(gx+gx2)//2] = 0
    return grid

# A* path for grid
def manhattan(a,b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar_grid(grid, start, goal):
    h, w = len(grid), len(grid[0])
    def neighbors(cell):
        y,x = cell
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w and grid[ny][nx]==0:
                yield (ny,nx)
    open_heap = []
    heapq.heappush(open_heap, (manhattan(start,goal), 0, start))
    parent = {start: None}
    gscore = {start: 0}
    closed = set()
    while open_heap:
        f, gcur, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        if cur == goal:
            break
        for nb in neighbors(cur):
            tentative = gcur + 1
            if tentative < gscore.get(nb, 1e9):
                gscore[nb] = tentative
                parent[nb] = cur
                heapq.heappush(open_heap, (tentative + manhattan(nb,goal), tentative, nb))
    if goal not in parent:
        return []
    path = []
    cur = goal
    while cur:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

# Initialize pygame
pygame.init()
grid_silent = None
passages = generate_maze(MAZE_W, MAZE_H)
grid_silent = passages_to_walls(passages)
GH, GW = len(grid_silent), len(grid_silent[0])

# Screen sizing
SCREEN_W = GW * CELL_PIX
SCREEN_H = GH * CELL_PIX
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("Пройди лабиринт (пробел - подсказка, R - новый лабиринт")
clock = pygame.time.Clock()

# Colors
C_BG = (30, 30, 30)
C_WALL = (20, 20, 20)
C_PASS = (240, 240, 240)
C_PLAYER = (200, 50, 50)
C_GOAL = (50, 200, 50)
C_HINT = (50, 150, 250)

# Player starts at first open cell (1,1), goal at last open
start = (1,1)
goal = (GH-2, GW-2)
player = list(start)
hint_path = []

def draw_grid(show_hint=False):
    screen.fill(C_BG)
    for y in range(GH):
        for x in range(GW):
            rect = pygame.Rect(x*CELL_PIX, y*CELL_PIX, CELL_PIX, CELL_PIX)
            if grid_silent[y][x] == 1:
                pygame.draw.rect(screen, C_WALL, rect)
            else:
                pygame.draw.rect(screen, C_PASS, rect)
    # draw hint path if any
    if show_hint and hint_path:
        for (y,x) in hint_path:
            rect = pygame.Rect(x*CELL_PIX, y*CELL_PIX, CELL_PIX, CELL_PIX)
            pygame.draw.rect(screen, C_HINT, rect)
    # draw goal
    rect_goal = pygame.Rect(goal[1]*CELL_PIX, goal[0]*CELL_PIX, CELL_PIX, CELL_PIX)
    pygame.draw.rect(screen, C_GOAL, rect_goal)
    # draw player
    rect_player = pygame.Rect(player[1]*CELL_PIX, player[0]*CELL_PIX, CELL_PIX, CELL_PIX)
    pygame.draw.rect(screen, C_PLAYER, rect_player)

def regenerate():
    global passages, grid_silent, GH, GW, start, goal, player, hint_path, direction
    passages = generate_maze(MAZE_W, MAZE_H)
    grid_silent = passages_to_walls(passages)
    GH, GW = len(grid_silent), len(grid_silent[0])
    start = (1,1)
    goal = (GH-2, GW-2)
    player[:] = start
    hint_path = []
    direction = None

# Movement handler
def can_move(to):
    y, x = to
    if 0 <= y < GH and 0 <= x < GW and grid_silent[y][x]==0:
        return True
    return False

# Determine direction from current pressed keys (priority order)
def get_direction_from_keys(pressed):
    # return a tuple dy,dx or None
    if pressed[pygame.K_UP] or pressed[pygame.K_w]:
        return (-1,0)
    if pressed[pygame.K_DOWN] or pressed[pygame.K_s]:
        return (1,0)
    if pressed[pygame.K_LEFT] or pressed[pygame.K_a]:
        return (0,-1)
    if pressed[pygame.K_RIGHT] or pressed[pygame.K_d]:
        return (0,1)
    return None

# Main loop
show_hint = False
running = True

# direction is the current held movement direction as (dy,dx) or None
direction = None
last_move_time = 0  # pygame.time.get_ticks() of last step (ms)

while running:
    dt = clock.tick(FPS)
    now = pygame.time.get_ticks()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                regenerate()
            elif event.key == pygame.K_SPACE:
                # toggle hint (A*)
                if not show_hint:
                    hint_path = astar_grid(grid_silent, tuple(player), goal)
                    show_hint = True
                else:
                    show_hint = False
                    hint_path = []
            elif event.key == pygame.K_ESCAPE:
                running = False
            elif event.key in (pygame.K_UP, pygame.K_w, pygame.K_DOWN, pygame.K_s,
                               pygame.K_LEFT, pygame.K_a, pygame.K_RIGHT, pygame.K_d):
                # set direction based on current keys and do an immediate step (felt-responsive)
                pressed = pygame.key.get_pressed()
                dir_now = get_direction_from_keys(pressed)
                direction = dir_now
                if direction is not None:
                    ny = player[0] + direction[0]
                    nx = player[1] + direction[1]
                    if can_move((ny, nx)):
                        player[0], player[1] = ny, nx
                        last_move_time = now
                    else:
                        # blocked immediately -> stop automatic movement
                        direction = None

        elif event.type == pygame.KEYUP:
            # when key released recalc direction from currently pressed keys;
            # if no movement key is pressed, stop.
            pressed = pygame.key.get_pressed()
            direction = get_direction_from_keys(pressed)

    # Automatic movement while key held
    if direction is not None:
        if now - last_move_time >= MOVE_DELAY_MS:
            ny = player[0] + direction[0]
            nx = player[1] + direction[1]
            if can_move((ny, nx)):
                player[0], player[1] = ny, nx
                last_move_time = now
            else:
                # hit wall -> stop moving automatically
                direction = None

    # Check win
    if tuple(player) == goal:
        print("Вы дошли до цели!")
        regenerate()

    draw_grid(show_hint)
    pygame.display.flip()

pygame.quit()
sys.exit()
