"""
Tic Tac Toe Player
"""
import copy
import math


X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    empty_count = 0
    x_count = 0
    o_count = 0
    for i in board:
        for j in i:
            if j == None:
                empty_count += 1
            if j == "X":
                x_count += 1
            if j == "O":
                o_count += 1
    
    
    if empty_count == 9:
        return X
    if x_count > o_count:
        return O
    else:
        return X



def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_moves = set()
    for i in range(0, len(board)):
        for j in range(0, len(board[0])):
            if board[i][j] == EMPTY:
                possible_moves.add((i,j))

    return possible_moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    player_turn = player(board)
    copy_board = copy.deepcopy(board)
    if copy_board[action[0]][action[1]] == EMPTY:
        if player_turn == X:
            copy_board[action[0]][action[1]] = X
            return copy_board
        if player_turn == O:
            copy_board[action[0]][action[1]] = O
            return copy_board

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    row = 0
    column = 0
    
    for i in range(3):
        if board[i][column] == board[i][column+1] == board[i][column+2] == X:
            return X
        if board[i][column] == board[i][column+1] == board[i][column+2] == O:
            return O
        if board[row][i] == board[row+1][i] == board[row+2][i] == X:
            return X
        if board[row][i] == board[row+1][i] == board[row+2][i] == O:
            return O
    
    if board[row][column] == board[row+1][column+1] == board[row+2][column+2] == X:
        return X
    if board[row][column] == board[row+1][column+1] == board[row+2][column+2] == O:
        return O
    if board[row][column+2] == board[row+1][column+1] == board[row+2][column] == X:
        return X
    if board[row][column+2] == board[row+1][column+1] == board[row+2][column] == O:
        return O

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None or (not any(EMPTY in sublist for sublist in board) and winner(board) is None):
        return True
    else:
        return False



def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """


    if terminal(board):
        win = winner(board)
        if win == X:
            return 1
        if win == O:
            return -1
        else:
            return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    
    if terminal(board):
        return None
    else:
        if player(board) == X:
            value, move = max_value(board)
            return move
        else:
            value, move = min_value(board)
            return move


def max_value(board):
    if terminal(board):
        return utility(board), None
    
    value = float("-inf")
    move = None
    for action in actions(board):
        maxval, act = min_value(result(board,action))
        if maxval > value:
            value = maxval
            move = action
            if value == 1:
                return value, move
        
    return value, move
    

def min_value(board):
    if terminal(board):
        return utility(board),None
    
    value = float("inf")
    move = None
    
    for action in actions(board):
        minval, act = max_value(result(board,action))
        if minval < value:
            value = minval
            move = action

            if value == -1:
                return value, move

    return value, move


