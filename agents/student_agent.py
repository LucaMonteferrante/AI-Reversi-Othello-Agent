# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()

    a = -np.inf
    b = np.inf

    best_move = (0,0)
    best_move = get_valid_moves(chess_board, player)[0]
    max_depth = 5

    if (player == 1): # 1 is alpha
      _, best_move = self.MaxValue(chess_board, a, b, 0, best_move, start_time)
    else:
      _, best_move = self.MinValue(chess_board, a, b, 0, best_move, start_time)

    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    #return random_move(chess_board,player)
    return best_move
  
  def successor(self, s, p):
    moves = get_valid_moves(s, p)
    #successors = np.array([])
    successors = []
    for move in moves:
      s_copy = deepcopy(s)
      execute_move(s_copy, move, p)
      #successors = np.append(successors, s_copy)
      successors.append((s_copy, move))
    #return np.array(successors)
    return successors

  def eval(self, s, p, scores):
    size = s.shape[0]
    ##################### CORNER EVAL #####################
    top_left_corner = 0
    if s[0][0] == 1:
      top_left_corner += 1
    elif s[0][0] == 2:
      top_left_corner -= 1
    top_right_corner = 0
    if s[0][-1] == 1:
      top_right_corner += 1
    elif s[0][-1] == 2:
      top_right_corner -= 1
    bot_left_corner = 0
    if s[-1][0] == 1:
      bot_left_corner += 1
    elif s[-1][0] == 2:
      bot_left_corner -= 1
    bot_right_corner = 0
    if s[-1][-1] == 1:
      bot_right_corner += 1
    elif s[-1][-1] == 2:
      bot_right_corner -= 1
    corner = top_left_corner + top_right_corner + bot_left_corner + bot_right_corner
    ##################### CORNER EVAL #####################

    ###################### WALL EVAL ######################
    top_wall = 0
    for j in range(1, size - 1):
      if s[0][j] == 1:
        top_wall += 1
      elif s[0][j] == 2:
        top_wall -= 1
    right_wall = 0
    for i in range(1, size - 1):
      if s[i][-1] == 1:
        right_wall += 1
      elif s[i][-1] == 2:
        right_wall -= 1
    bot_wall = 0
    for j in range(1, size - 1):
      if s[-1][j] == 1:
        bot_wall += 1
      elif s[-1][j] == 2:
        bot_wall -= 1
    left_wall = 0
    for i in range(1, size - 1):
      if s[i][0] == 1:
        left_wall += 1
      elif s[i][0] == 2:
        left_wall -= 1
    wall = top_wall + right_wall + bot_wall + left_wall
    ###################### WALL EVAL ######################

    #################### ADJACENT EVAL ####################
    top_left_adj = 0
    adj = [(0, 1), (1, 0), (1, 1)]
    for i, j in adj:
      if s[i][j] == 1:
        top_left_adj += 1
      elif s[i][j] == 2:
        top_left_adj -= 1
    top_right_adj = 0
    adj = [(0, -2), (1, -1), (1, -2)]
    for i, j in adj:
      if s[i][j] == 1:
        top_right_adj += 1
      elif s[i][j] == 2:
        top_right_adj -= 1
    bot_left_adj = 0
    adj = [(-1, 1), (-2, 0), (-2, 1)]
    for i, j in adj:
      if s[i][j] == 1:
        bot_left_adj += 1
      elif s[i][j] == 2:
        bot_left_adj -= 1
    bot_right_adj = 0
    adj = [(-1, -2), (-2, -1), (-2, -2)]
    for i, j in adj:
      if s[i][j] == 1:
        bot_right_adj += 1
      elif s[i][j] == 2:
        bot_right_adj -= 1
    adj = top_left_adj + top_right_adj + bot_left_adj + bot_right_adj
    #################### ADJACENT EVAL ####################

    #################### MOBILITY EVAL ####################
    mobility = len(get_valid_moves(s, p))
    if p == 2:
      mobility *= -1
    #################### MOBILITY EVAL ####################

    ################### STABILITY EVAL ####################
    stable = 0
    if top_left_corner == 1:
      #P1
      stable += 1
      i = 0
      j = 0
      xlim = i
      ylim = j
      while s[i][j] == s[i][j + 1]:
        stable += 1
        j += 1
        ylim = j
        if j == size // 2 - 1:
          break
      j = 0
      while s[i][j] == s[i + 1][j]:
        stable += 1
        i += 1
        xlim = i
        if i == size // 2 - 1:
          break
      i = 0
      j = 0
      temp = 0
      while i < xlim and j < ylim and s[i][j] == s[i + 1][j + 1]:
        i += 1
        j += 1
        temp += 1
        stable += 1
        while j < xlim - 1 and s[i][j] == s[i][j + 1]:
          stable += 1
          j += 1
          if j == ylim - 1:
            break
        ylim = j
        j = temp
        while i < xlim - 1 and s[i][j] == s[i + 1][j]:
          stable += 1
          i += 1
          if i == xlim - 1:
            break
        xlim = i
        i = temp
        if i == xlim - 1 and j == ylim - 1:
          break
    elif top_left_corner == -1:
      #P2
      stable -= 1
      i = 0
      j = 0
      xlim = i
      ylim = j
      while s[i][j] == s[i][j + 1]:
        stable -= 1
        j += 1
        ylim = j
        if j == size // 2 - 1:
          break
      j = 0
      while s[i][j] == s[i + 1][j]:
        stable -= 1
        i += 1
        xlim = i
        if i == size // 2 - 1:
          break
      i = 0
      j = 0
      temp = 0
      while i < xlim and j < ylim and s[i][j] == s[i + 1][j + 1]:
        i += 1
        j += 1
        temp += 1
        stable -= 1
        while j < ylim - 1 and s[i][j] == s[i][j + 1]:
          stable -= 1
          j += 1
          if j == ylim - 1:
            break
        ylim = j
        j = temp
        while i < xlim - 1 and s[i][j] == s[i + 1][j]:
          stable -= 1
          i += 1
          if i == xlim - 1:
            break
        xlim = i
        i = temp
        if i == xlim - 1 and j == ylim - 1:
          break
    if top_right_corner == 1:
      #P1
      stable += 1
      i = 0
      j = size - 1
      xlim = i
      ylim = j
      while s[i][j] == s[i][j - 1]:
        stable += 1
        j -= 1
        ylim = j
        if j == size // 2:
          break
      j = size - 1
      while s[i][j] == s[i + 1][j]:
        stable += 1
        i += 1
        xlim = i
        if i == size // 2 - 1:
          break
      i = 0
      j = size - 1
      temp = size - 1
      while i < xlim and j - 1 > ylim and s[i][j] == s[i + 1][j - 1]:
        i += 1
        j -= 1
        temp -= 1
        stable += 1
        while j - 1 > ylim and s[i][j] == s[i][j - 1]:
          stable += 1
          j -= 1
          if j == ylim + 1:
            break
        ylim = j
        j = temp
        while i < xlim - 1 and s[i][j] == s[i + 1][j]:
          stable += 1
          i += 1
          if i == xlim - 1:
            break
        xlim = i
        i = temp
        if i == xlim - 1 and j == ylim + 1:
          break
    elif top_right_corner == -1:
      #P2
      stable -= 1
      i = 0
      j = size - 1
      xlim = i
      ylim = j
      while s[i][j] == s[i][j - 1]:
        stable -= 1
        j -= 1
        ylim = j
        if j == size // 2:
          break
      j = size - 1
      while s[i][j] == s[i + 1][j]:
        stable -= 1
        i += 1
        xlim = i
        if i == size // 2 - 1:
          break
      i = 0
      j = size - 1
      temp = size - 1
      while i < xlim and j - 1 > ylim and s[i][j] == s[i + 1][j - 1]:
        i += 1
        j -= 1
        temp -= 1
        stable -= 1
        while j - 1 > ylim and s[i][j] == s[i][j - 1]:
          stable -= 1
          j -= 1
          if j == ylim + 1:
            break
        ylim = j
        j = temp
        while i < xlim - 1 and s[i][j] == s[i + 1][j]:
          stable -= 1
          i += 1
          if i == xlim - 1:
            break
        xlim = i
        i = temp
        if i == xlim - 1 and j == ylim + 1:
          break
    if bot_left_corner == 1:
      #P1
      stable += 1
      i = size - 1
      j = 0
      xlim = i
      ylim = j
      while s[i][j] == s[i][j + 1]:
        stable += 1
        j += 1
        ylim = j
        if j == size // 2 - 1:
          break
      j = 0
      while s[i][j] == s[i - 1][j]:
        stable += 1
        i -= 1
        xlim = i
        if i == size // 2:
          break
      i = size - 1
      j = 0
      temp = 0
      while i - 1 > xlim and j < ylim and s[i][j] == s[i - 1][j + 1]:
        i -= 1
        j += 1
        temp += 1
        stable += 1
        while j - 1 < ylim and s[i][j] == s[i][j + 1]:
          stable += 1
          j += 1
          if j == ylim - 1:
            break
        ylim = j
        j = temp
        while i - 1 > xlim and s[i][j] == s[i - 1][j]:
          stable += 1
          i -= 1
          if i == xlim + 1:
            break
        xlim = i
        i = temp
        if i == xlim + 1 and j == ylim - 1:
          break
    elif bot_left_corner == -1:
      #P2
      stable -= 1
      i = size - 1
      j = 0
      xlim = i
      ylim = j
      while s[i][j] == s[i][j + 1]:
        stable -= 1
        j += 1
        ylim = j
        if j == size // 2 - 1:
          break
      j = 0
      while s[i][j] == s[i - 1][j]:
        stable -= 1
        i -= 1
        xlim = i
        if i == size // 2:
          break
      i = size - 1
      j = 0
      temp = 0
      while i - 1 > xlim and j < ylim and s[i][j] == s[i - 1][j + 1]:
        i -= 1
        j += 1
        temp += 1
        stable -= 1
        while j < ylim - 1 and s[i][j] == s[i][j + 1]:
          stable -= 1
          j += 1
          if j == ylim - 1:
            break
        ylim = j
        j = temp
        while i - 1 > xlim and s[i][j] == s[i - 1][j]:
          stable -= 1
          i -= 1
          if i == xlim + 1:
            break
        xlim = i
        i = temp
        if i == xlim + 1 and j == ylim - 1:
          break
    if bot_right_corner == 1:
      #P1
      stable += 1
      i = size - 1
      j = size - 1
      xlim = i
      ylim = j
      while s[i][j] == s[i][j - 1]:
        stable += 1
        j -= 1
        ylim = j
        if j == size // 2:
          break
      j = size - 1
      while s[i][j] == s[i - 1][j]:
        stable += 1
        i -= 1
        xlim = i
        if i == size // 2:
          break
      i = size - 1
      j = size - 1
      temp = size - 1
      while i - 1 > xlim and j - 1 > ylim and s[i][j] == s[i - 1][j - 1]:
        i -= 1
        j -= 1
        temp -= 1
        stable += 1
        while j - 1 > ylim and s[i][j] == s[i][j - 1]:
          stable += 1
          j -= 1
          if j == ylim + 1:
            break
        ylim = j
        j = temp
        while i - 1 > xlim and s[i][j] == s[i - 1][j]:
          stable += 1
          i -= 1
          if i == xlim + 1:
            break
        xlim = i
        i = temp
        if i == xlim + 1 and j == ylim + 1:
          break
    elif bot_right_corner == -1:
      #P2
      stable -= 1
      i = size - 1
      j = size - 1
      xlim = i
      ylim = j
      while s[i][j] == s[i][j - 1]:
        stable -= 1
        j -= 1
        ylim = j
        if j == size // 2:
          break
      j = size - 1
      while s[i][j] == s[i - 1][j]:
        stable -= 1
        i -= 1
        xlim = i
        if i == size // 2:
          break
      i = size - 1
      j = size - 1
      temp = size - 1
      while i - 1 > xlim and j - 1 > ylim and s[i][j] == s[i - 1][j - 1]:
        i -= 1
        j -= 1
        temp -= 1
        stable -= 1
        while j - 1 > ylim and s[i][j] == s[i][j - 1]:
          stable -= 1
          j -= 1
          if j == ylim + 1:
            break
        ylim = j
        j = temp
        while i - 1 > xlim and s[i][j] == s[i - 1][j]:
          stable -= 1
          i -= 1
          if i == xlim + 1:
            break
        xlim = i
        i = temp
        if i == xlim + 1 and j == ylim + 1:
          break
    ################### STABILITY EVAL ####################
    
    if size == 6:
      c1 = 150
      c2 = 25
      c3 = 50
      c4 = 20
      c5 = 50
    elif size == 8:
      c1 = 100
      c2 = 25
      c3 = 50
      c4 = 20
      c5 = 50
    elif size == 10:
      c1 = 100
      c2 = 25
      c3 = 50
      c4 = 20
      c5 = 50
    else:
      c1 = 100
      c2 = 25
      c3 = 50
      c4 = 20
      c5 = 50

    return c1*corner + c2*wall - c3*adj + c4*mobility + c5*stable


  def cutoff(self, s, d, t):
    if t>1.95:
      return True
    size = s.shape[0]
    maxdepth = 0
    if size == 6:
      maxdepth = 5
    elif size == 8:
      maxdepth = 4
    elif size == 10:
      maxdepth = 3
    else:
      maxdepth = 3
    # if size == 6:
    #   maxdepth = 6
    # elif size == 8:
    #   maxdepth = 5
    # elif size == 10:
    #   maxdepth = 4
    # else:
    #   maxdepth = 4
    if d == maxdepth:
      return True
    else:
      return False
  
  def MaxValue(self, s, a, b, d, best_move, t):
    output = check_endgame(s, 1, 2) # check this
    if output[0]:
      if output[1] > output[2]:
        print("P1")
        return np.inf, best_move
      elif output[1] == output[2]:
        print("TIE")
        return 0, best_move
      else:
        print("P2")
        return -np.inf, best_move
    else:
      if self.cutoff(s, d, time.time() - t):
        return self.eval(s, 1, (output[1], output[2])), best_move
      successors = self.successor(s, 1)
      for board, move in successors:
        a_opt = np.max([a, self.MinValue(board, a, b, d+1, best_move, t)[0]]) # confirm this
        if a_opt > a:
          if d == 0:
            best_move = move
          a = a_opt
        if a >= b:
          return b, best_move
        

      if successors == []:
        a_opt = np.max([a, self.MinValue(s, a, b, d+1, best_move, t)[0]])
        if a_opt > a:
          if d == 0:
            best_move = move
          a = a_opt
        if a >= b:
          return b, best_move


      return a, best_move
    
  def MinValue(self, s, a, b, d, best_move, t):
    output = check_endgame(s, 2, 1) # check this
    if output[0]:
      if output[1] > output[2]:
        print("P1")
        return np.inf, best_move
      elif output[1] == output[2]:
        print("TIE")
        return 0, best_move
      else:
        print("P2")
        return -np.inf, best_move
    else:
      if self.cutoff(s, d, time.time() - t):
        return self.eval(s, 2, (output[1], output[2])), best_move
      successors = self.successor(s, 2)
      for board, move in successors:
        b_opt = np.min([b, self.MaxValue(board, a, b, d+1, best_move, t)[0]]) # confirm this
        if b_opt < b:
          if d == 0:
            best_move = move
          b = b_opt
        if a >= b:
          return a, best_move
        
      
      if successors == []:
        b_opt = np.min([b, self.MaxValue(s, a, b, d+1, best_move, t)[0]]) # confirm this
        if b_opt < b:
          if d == 0:
            best_move = move
          b = b_opt
        if a >= b:
          return a, best_move




      return b, best_move
