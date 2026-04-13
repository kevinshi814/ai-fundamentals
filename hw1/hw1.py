import time
import numpy as np
from gridgame import *

##############################################################################################################################

# You can visualize what your code is doing by setting the GUI argument in the following line to true.
# The render_delay_sec argument allows you to slow down the animation, to be able to see each step more clearly.

# For your final submission, please set the GUI option to False.

# The gs argument controls the grid size. You should experiment with various sizes to ensure your code generalizes.
# Please do not modify or remove lines 18 and 19.

##############################################################################################################################

game = ShapePlacementGrid(GUI=True, render_delay_sec=0.5, gs=6, num_colored_boxes=5)
shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('export')
np.savetxt('initial_grid.txt', grid, fmt="%d")

##############################################################################################################################

# Initialization

# shapePos is the current position of the brush.

# currentShapeIndex is the index of the current brush type being placed (order specified in gridgame.py, and assignment instructions).

# currentColorIndex is the index of the current color being placed (order specified in gridgame.py, and assignment instructions).

# grid represents the current state of the board. 
    
    # -1 indicates an empty cell
    # 0 indicates a cell colored in the first color (indigo by default)
    # 1 indicates a cell colored in the second color (taupe by default)
    # 2 indicates a cell colored in the third color (veridian by default)
    # 3 indicates a cell colored in the fourth color (peach by default)

# placedShapes is a list of shapes that have currently been placed on the board.
    
    # Each shape is represented as a list containing three elements: a) the brush type (number between 0-8), 
    # b) the location of the shape (coordinates of top-left cell of the shape) and c) color of the shape (number between 0-3)

    # For instance [0, (0,0), 2] represents a shape spanning a single cell in the color 2=veridian, placed at the top left cell in the grid.

# done is a Boolean that represents whether coloring constraints are satisfied. Updated by the gridgames.py file.

##############################################################################################################################

shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('export')

print(shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done)


####################################################
# Timing your code's execution for the leaderboard.
####################################################

start = time.time()  # <- do not modify this.



##########################################
# Write all your code in the area below. 
##########################################

# States: any coloring including the given ones
# Actions: switch a color
# Objective function: Penalize adjacent pairs of same color, bonus for diagonal pairs with same color
def objective(coloring):
    ret = 0
    adj_penalty = coloring.size * 2
    diag_bonus = 1
    # Rows, downward diagonals
    for row in range(coloring.shape[0] - 1):
        for col in range(coloring.shape[1] - 1):
            if coloring[row, col] == coloring[row, col + 1]:
                ret -= adj_penalty
            if coloring[row, col] == coloring[row + 1, col + 1]:
                ret += diag_bonus
    for col in range(coloring.shape[1] - 1):
        if coloring[coloring.shape[0] - 1, col] == coloring[coloring.shape[0] - 1, col + 1]:
            ret -= adj_penalty

    # Cols, upward diagonals
    for col in range(1, coloring.shape[1]):
        for row in range(coloring.shape[0] - 1):
            if coloring[row, col] == coloring[row + 1, col]:
                ret -= adj_penalty
            if coloring[row, col] == coloring[row + 1, col - 1]:
                ret += diag_bonus
    for row in range(coloring.shape[0] - 1):
        if coloring[row, 0] == coloring[row + 1, 0]:
            ret -= adj_penalty

    return ret

# Create an initial solution
starting_filled = []
coloring = grid.copy()
for row in range(grid.shape[0]):
    for col in range(grid.shape[1]):
        if grid[row][col] == -1:
            coloring[row][col] = np.random.randint(4)
        else:
            starting_filled.append((row, col))

rows = coloring.shape[0]
cols = coloring.shape[1]
curr_obj = objective(coloring)
while curr_obj < 0:
    # Select a move
    pos = (np.random.randint(rows), np.random.randint(cols))
    if pos in starting_filled:
        continue
    new_color = np.random.randint(3)
    if new_color == coloring[pos]:
        new_color = 3

    # Compare the change to the current state
    curr_color = coloring[pos]
    coloring[pos] = new_color
    new_obj = objective(coloring)
    if new_obj > curr_obj:
        curr_obj = new_obj
    else:
        coloring[pos] = curr_color

# Place the solution in the game

# Check if the shape of a solid color fits there
# Returns the color index if it fits; -1 if not
def try_shape(pos, shape_index):
    seen = set()
    shape = game.shapes[shape_index]
    for i, row in enumerate(shape):
        for j, cell in enumerate(row):
            if cell:
                if pos[0] + j >= game.gridSize or pos[1] + i >= game.gridSize:
                    return -1
                if grid[pos[1] + i, pos[0] + j] != -1:
                    return -1
                else:
                    seen.add(coloring[pos[1] + i, pos[0] + j])
    if len(seen) == 1:
        return seen.pop()
    else:
        return -1

# Switch to shape 3
game.execute('switchshape')
game.execute('switchshape')
game.execute('switchshape')

# Go through different shapes and place as many as possible
for i in range(8):
    # Skip the single block for now
    if game.currentShapeIndex == 0:
        game.execute('switchshape')
        continue

    # Sweep across rows
    r = 0
    c = 0
    right = True
    while r + game.shapesDims[game.currentShapeIndex][1] <= grid.shape[1]:
        while 0 <= c and c + game.shapesDims[game.currentShapeIndex][0] <= grid.shape[0]:
            color = try_shape(game.shapePos, game.currentShapeIndex)
            if color >= 0:
                while game.currentColorIndex != color:
                    game.execute('switchcolor')
                game.execute('place')
            game.execute('right' if right else 'left')
            c += 1 if right else -1
        game.execute('down')
        r += 1
        right = not right
        c += 1 if right else -1

    while game.shapePos[0] > 0:
        game.execute('left')
    while game.shapePos[1] > 0:
        game.execute('up')
    game.execute('switchshape')

# Switch to single block
while game.currentShapeIndex != 0:
    game.execute('switchshape')

# Sweep to fill in holes
right = True
r = 0
c = 0
while r < rows:
    while c < cols and c >= 0:
        color = try_shape(game.shapePos, 0)
        if color != -1:
            while game.currentColorIndex != color:
                game.execute('switchcolor')
            game.execute('place')
        game.execute('right' if right else 'left')
        c += 1 if right else -1
    right = not right
    c += 1 if right else -1
    game.execute('down')
    r += 1


########################################

# Do not modify any of the code below. 

########################################

end=time.time()

np.savetxt('grid.txt', grid, fmt="%d")
with open("shapes.txt", "w") as outfile:
    outfile.write(str(placedShapes))
with open("time.txt", "w") as outfile:
    outfile.write(str(end-start))
