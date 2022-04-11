import pygame
import sys
from pygame.locals import * 
import numpy as np
import random

class Node():
    def __init__(self,state,parent,action):
        self.state = state
        self.parent = parent
        self.action = action
    
class StackFrontier():
    def __init__(self):
        self.frontier = []


    def add(self,node):
        self.frontier.append(node)

    def contains_state(self,state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("Empty")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node

class QueueFrontier(StackFrontier):
    def remove(self):
        if self.empty():
            raise Exception("Empty")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node

EMPTY = None

class maze():
    def __init__(self):
        self.maze = []
        try:   
            self.height = int(sys.argv[3])
        except:
            raise ValueError()

        self.width = self.height

        for i in range(self.height):    
            self.maze_row = []
            for j in range(self.height):
                self.maze_row.append(EMPTY)
            self.maze.append(self.maze_row)
        
        self.goal = (random.randint(0,self.height-1),random.randint(0,self.width-1))
        self.start = (random.randint(0,self.height-1),random.randint(0,self.width-1))
        self.maze[self.goal[0]][self.goal[1]] = "B"
        self.maze[self.start[0]][self.start[1]] = "A"

    def maze_change(self):
        self.wall = []
        for i,row in enumerate(self.maze):
            self.wall_row = []
            for j,col in enumerate(row):
                if self.maze[i][j] ==  "A":
                    self.start = (i,j)
                    self.wall_row.append(False)
                if self.maze[i][j] == "B":
                    self.goal = (i,j)
                    self.wall_row.append(False)
                if self.maze[i][j] == "Wall":
                    self.wall_row.append(True)
                if self.maze[i][j] == EMPTY:
                    self.wall_row.append(False)
            self.wall.append(self.wall_row)
        self.solution = None
    
    def neighbors(self,state):
        row, col = state
        print(sys.argv[2])
        
        if sys.argv[2] == 'True':
            candidates = [
                ("up", (row - 1, col)),
                ("down", (row + 1, col)),
                ("left", (row, col - 1)),
                ("right", (row, col + 1)),
                ("downriht", (row+1,col+1)),
                ("downleft", (row+1,col-1)),
                ("upright", (row-1,col+1)),
                ("upleft",(row-1,col-1))
            ]
        elif sys.argv[2] == 'False':
            candidates = [
                ("up", (row - 1, col)),
                ("down", (row + 1, col)),
                ("left", (row, col - 1)),
                ("right", (row, col + 1))
            ]
        else:
            raise Exception("Invalid Command!",print(sys.argv))
        
        result = []

        for action,(r,c) in candidates:
            if 0 <= r < self.height and 0<= c < self.width and not self.wall[r][c]:
                result.append((action, (r,c)))

        return result  

    def solve(self):

        self.number_of_exp = 0

        start = Node(state=self.start,parent=None,action=None)
        if sys.argv[1] == 'Stack':
            frontier = StackFrontier()
        elif sys.argv[1] == "Queue":
            frontier = QueueFrontier()
        else:
            raise Exception("Invalid algorithm.")

        frontier.add(start)


        self.explored = set()

        while True:
            
            if frontier.empty():
                raise Exception("no solution")
            
            node = frontier.remove()
            self.number_of_exp += 1

            if node.state == self.goal:
                actions = []
                cells = []
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (cells)
                return

            self.explored.add(node.state)

            for action,state in self.neighbors(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    child = Node(state=state,parent=node,action=action)
                    frontier.add(child)


class draw():
    def __init__(self):
        self.block_size = 1
        self.m = maze()
        self.m.maze_change()
        pygame.init()
        self.size = 600,600
        self.screen = pygame.display.set_mode(self.size)
        self.tile_size = 10
        self.border_size = 5
        self.block_size = 1

    def draw_maze(self):
        for i in range(self.m.height):
            for j in range(self.m.width):
                if self.m.maze[i][j] == "B":
                    rect = pygame.Rect(
                                j * self.tile_size,
                                i * self.tile_size,
                                self.tile_size, self.tile_size
                            )
                    pygame.draw.rect(self.screen, (255,0,0), rect, 0)
                    rect = pygame.Rect(
                                j * self.tile_size,
                                i * self.tile_size,
                                self.tile_size, self.tile_size
                            )
                    pygame.draw.rect(self.screen, (0,0,0), rect, 1)
                
                if self.m.maze[i][j] == "A":
                    rect = pygame.Rect(
                                j * self.tile_size,
                                i * self.tile_size,
                                self.tile_size, self.tile_size
                            )
                    pygame.draw.rect(self.screen, (0,255,0), rect, 0)
                    rect = pygame.Rect(
                                j * self.tile_size,
                                i * self.tile_size,
                                self.tile_size, self.tile_size
                            )
                    pygame.draw.rect(self.screen, (0,0,0), rect, 1)

                if self.m.maze[i][j] == "Wall":
                    rect = pygame.Rect(
                                j * self.tile_size,
                                i * self.tile_size,
                                self.tile_size, self.tile_size
                            )
                    pygame.draw.rect(self.screen, (0,0,0), rect, 0)
                    rect = pygame.Rect(
                                j * self.tile_size,
                                i * self.tile_size,
                                self.tile_size, self.tile_size
                            )
                    pygame.draw.rect(self.screen, (0,0,0), rect, 1)

                if self.m.maze[i][j] == EMPTY:
                    rect = pygame.Rect(
                                j * self.tile_size,
                                i * self.tile_size,
                                self.tile_size, self.tile_size
                            )
                    pygame.draw.rect(self.screen, (255,255,255), rect, 0)
                    rect = pygame.Rect(
                                j * self.tile_size,
                                i * self.tile_size,
                                self.tile_size, self.tile_size
                            )
                    pygame.draw.rect(self.screen, (0,0,0), rect, 1)
    def past_road(self):
        for k,i in enumerate(self.m.solution):
            if i != self.m.goal and i != self.m.start:
                rect = pygame.Rect(
                                i[1] * self.tile_size,
                                i[0] * self.tile_size,
                                self.tile_size, self.tile_size
                            )
                pygame.draw.rect(self.screen, (255,255,0), rect, 0)
                rect = pygame.Rect(
                                i[1] * self.tile_size,
                                i[0] * self.tile_size,
                                self.tile_size, self.tile_size
                            )
                pygame.draw.rect(self.screen, (0,0,0), rect, 1)

    def search(self):
        for k,i in enumerate(self.m.solution):
            if i != self.m.goal and i != self.m.start:
                rect = pygame.Rect(
                                i[1] * self.tile_size,
                                i[0] * self.tile_size,
                                self.tile_size, self.tile_size
                            )
                pygame.draw.rect(self.screen, (255,255,0), rect, 0)
                rect = pygame.Rect(
                                i[1] * self.tile_size,
                                i[0] * self.tile_size,
                                self.tile_size, self.tile_size
                            )
                pygame.draw.rect(self.screen, (0,0,0), rect, 1)
                
                rect = pygame.Rect(
                                self.m.goal[1] * self.tile_size,
                                self.m.goal[0] * self.tile_size,
                                self.tile_size, self.tile_size
                            )
                pygame.draw.rect(self.screen, (255,0,0), rect, 0)
                
                rect = pygame.Rect(
                                self.m.goal[1]  * self.tile_size,
                                self.m.goal[0] * self.tile_size,
                                self.tile_size, self.tile_size
                            )
                pygame.draw.rect(self.screen, (0,0,0), rect, 1)
                pygame.display.update()
                pygame.time.delay(100)
                
                j = self.m.solution[k]
                if j:
                    rect = pygame.Rect(
                                j[1] * self.tile_size,
                                j[0] * self.tile_size,
                                self.tile_size, self.tile_size
                            )
                    pygame.draw.rect(self.screen, (50,40,200), rect, 0)
                    rect = pygame.Rect(
                                    j[1] * self.tile_size,
                                    j[0] * self.tile_size,
                                    self.tile_size, self.tile_size
                                )
                    pygame.draw.rect(self.screen, (0,0,0), rect, 1)
            if i == self.m.goal:
                self.block_size += 1
                self.m.maze[i[0]][i[1]] = EMPTY
                self.m.start = self.m.goal
                self.m.goal = (random.randint(0,self.m.height-1),random.randint(0,self.m.width-1))
                while self.m.maze[self.m.goal[0]][self.m.goal[1]] == "Wall":
                    print('wall',self.m.goal)
                    self.m.goal = (random.randint(0,self.m.height-1),random.randint(0,self.m.width-1))
                    self.m.maze[self.m.goal[0]][self.m.goal[1]] = "B"
                self.draw_maze()
    
    def run(self):
        self.pause = False
        while True:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        sys.exit()
                    if event.key == K_s and self.pause == False:
                        self.pause = True
                
                if event.type == MOUSEBUTTONUP:
                    x_loc = mouse_x // (self.tile_size)
                    y_loc = mouse_y // (self.tile_size)
                    if event.button == 1 or event.button == 3 or event.button==2:
                        self.m.maze[y_loc][x_loc] = "Wall"
                        self.m.maze_change()
                
                self.draw_maze()
                pygame.display.flip()
                
                while self.pause:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            sys.exit()
                        if event.type == KEYDOWN:
                            if event.key == K_ESCAPE:
                                sys.exit()
                            if event.key == K_p:
                                self.pause = False
                    
                    self.screen.fill((255,255,255))
                    self.m.solve()
                    self.draw_maze()
                    self.search()

if len(sys.argv) != 4:
    sys.exit("Usage python <Stack or Queue> <True or False for corners> <size>")

if __name__ == "__main__":
    solution = draw()
    solution.run()

