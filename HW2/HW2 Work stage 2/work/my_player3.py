import copy
import numpy as np


# Creating variable for the input file name. 
input_file = 'input.txt'

# Creating variable for the output file name. 
output_file = 'output.txt'

# Creating variable for storing the number of steps taken in total.
steps_num_file = 'steps_num.txt'


board_dimension = 5
not_occupied = 0
black_player = 1 #Represents the black player
white_player = 2 # Represents the white player 
komi_value = 2.5 # Komi Value to be added for the white player 

# The Liberties of the stone Right, Left, Botton and Top
X = [1, 0, -1, 0]
Y = [0, 1, 0, -1]


class Player:
    # The different states of the board and the player currently playing
    def __init__(self, party, prev_board_state, curr_board_state):
        self.party = party
        self.opponent_party = self.get_opponent_party(self.party)
        self.prev_board_state = prev_board_state
        self.curr_board_state = curr_board_state

    # Calculating the max value out of the possible values at the Max Node

    def max_value_calculate(self, board_state, party, max_depth, curr_depth, b, alpha, beta, prev_move, step_num, is_pass_2nd):
        # Check if the maximum depth is reached or the game has ended
        if max_depth == curr_depth or step_num + curr_depth == 24:
            return self.evaluate_board_state(board_state, party)

        # Check if it's a pass situation for the second player
        if is_pass_2nd:
            return self.evaluate_board_state(board_state, party)

        # Reset the pass flag
        is_pass_2nd = False

        # Initialize variables to keep track of the best move and its value
        move_of_max_value = -np.inf
        move_of_max_player = None

        # Find legal moves for the current player
        legal_moves = self.find_legal_moves(board_state, party)

        # Add a special "pass" move to the legal moves
        legal_moves.append((-1, -1))

        # Check if this is the first move (no previous move)
        if prev_move == (-1, -1):
            is_pass_2nd = True

        for legal_move in legal_moves[:b]:
            # Create a new game state
            opponent_party = self.get_opponent_party(party)
            if legal_move == (-1, -1):
                new_board_state = copy.deepcopy(board_state)
            else:
                new_board_state = self.make_move(board_state, party, legal_move)

            # Recursively calculate the minimum value for the opponent
            move_of_min_value = self.min_value_calculate(new_board_state, opponent_party, max_depth, curr_depth + 1, b, alpha, beta, legal_move, step_num, is_pass_2nd)

            # Update the best move and its value for the current player
            if move_of_max_value < move_of_min_value:
                move_of_max_value = move_of_min_value
                move_of_max_player = legal_move

            # Check if the current maximum value is greater than or equal to beta
            if move_of_max_value >= beta:
                if curr_depth == 0:
                    # If this is the initial call, return both the move and its value
                    return move_of_max_player, move_of_max_value
                else:
                    # Otherwise, return just the maximum value
                    return move_of_max_value

            # Update the alpha value
            alpha = alpha if alpha > move_of_max_value else move_of_max_value

        if curr_depth == 0:
            # If this is the initial call, return both the move and its value
            return move_of_max_player, move_of_max_value
        else:
            # Otherwise, return just the maximum value
            return move_of_max_value


    # Calculating Min Value at the Min Node

    def min_value_calculate(self, board_state, party, max_depth, curr_depth, b, alpha, beta, prev_move, step_num, is_pass_2nd):
        # Check if the maximum depth is reached
        if max_depth == curr_depth:
            return self.evaluate_board_state(board_state, party)

        # Check if the game has ended or it's a pass situation for the second player
        if step_num + curr_depth == 24 or is_pass_2nd:
            return self.evaluate_board_state(board_state, self.party)

        # Reset the pass flag
        is_pass_2nd = False

        # Initialize variables to keep track of the best move's value
        move_of_min_value = np.inf

        # Find legal moves for the current player
        legal_moves = self.find_legal_moves(board_state, party)

        # Add a special "pass" move to the legal moves
        legal_moves.append((-1, -1))

        # Check if this is the first move (no previous move)
        if prev_move == (-1, -1):
            is_pass_2nd = True

        for legal_move in legal_moves[:b]:
            # Create a new game state
            opponent_party = self.get_opponent_party(party)
            if legal_move == (-1, -1):
                new_board_state = copy.deepcopy(board_state)
            else:
                new_board_state = self.make_move(board_state, party, legal_move)

            # Recursively calculate the maximum value for the opponent
            move_of_max_value = self.max_value_calculate(new_board_state, opponent_party, max_depth, curr_depth + 1, b, alpha, beta, legal_move, step_num, is_pass_2nd)

            # Update the minimum value for the current player
            if move_of_max_value < move_of_min_value:
                move_of_min_value = move_of_max_value

            # Check if the current minimum value is less than or equal to alpha
            if move_of_min_value <= alpha:
                return move_of_min_value

            # Update the beta value
            beta = beta if beta < move_of_min_value else move_of_min_value

        return move_of_min_value

    # Evaluation Function for the Go Game
    def evaluate_board_state(self, board_state, party):
        # Determine the opponent's party.
        opponent_party = self.get_opponent_party(party)

        # Initialize counts and liberty sets for both parties.
        party_count = 0
        party_liberty = set()
        opponent_party_count = 0
        opponent_party_liberty = set()

        # Iterate through the board to calculate party counts and liberties.
        for i in range(board_dimension):
            for j in range(board_dimension):
                if board_state[i][j] == party:
                    party_count += 1
                elif board_state[i][j] == opponent_party:
                    opponent_party_count += 1
                else:
                    # Check the neighboring cells to identify liberties.
                    for index in range(len(X)):
                        i1 = i + X[index]
                        j1 = j + Y[index]
                        if 0 <= i1 < board_dimension and 0 <= j1 < board_dimension:
                            if board_state[i1][j1] == party:
                                party_liberty.add((i, j))
                            elif board_state[i1][j1] == opponent_party:
                                opponent_party_liberty.add((i, j))

        # Initialize counts for party and opponent party edges.
        party_edge_count = 0
        opponent_party_edge_count = 0

        # Check for party and opponent party edges on the board.
        for j in range(board_dimension):
            if board_state[0][j] == party or board_state[board_dimension - 1][j] == party:
                party_edge_count += 1
            if board_state[0][j] == opponent_party or board_state[board_dimension - 1][j] == opponent_party:
                opponent_party_edge_count += 1

        for j in range(1, board_dimension - 1):
            if board_state[j][0] == party or board_state[j][board_dimension - 1] == party:
                party_edge_count += 1
            if board_state[j][0] == opponent_party or board_state[j][board_dimension - 1] == opponent_party:
                opponent_party_edge_count += 1

        # Initialize count for center not occupied cells.
        center_not_occupied_count = 0

        # Count the number of not occupied cells in the center of the board.
        for i in range(1, board_dimension - 1):
            for j in range(1, board_dimension - 1):
                if board_state[i][j] == not_occupied:
                    center_not_occupied_count += 1

        # Calculate the possible score based on various board characteristics.
        possible_score = min(max((len(party_liberty) - len(opponent_party_liberty)), -8), 8) + (
            -4 * self.calculate_euler_value(board_state, party)) + (
            5 * (party_count - opponent_party_count)) - (9 * party_edge_count * (center_not_occupied_count / 9))

        # Adjust the score if the current party is the white player.
        if self.party == white_player:
            possible_score += komi_value

        # Return the calculated possible score.
        return possible_score


    # Making a Move
    def make_move(self, board_state, party, legal_move):
        
        # Create a deep copy of the current board state to prevent modifying the original.
        new_board_state = copy.deepcopy(board_state)

         # Place the player's stone at the specified legal move position.
        new_board_state[legal_move[0]][legal_move[1]] = party

        # Iterate through the adjacent positions to check for captured opponent groups.
        for index in range(len(X)):
            i1 = legal_move[0] + X[index]
            j1 = legal_move[1] + Y[index]
            
            # Ensure that the adjacent position is within the board's boundaries.
            if 0 <= i1 < board_dimension and 0 <= j1 < board_dimension:
                # Determine the opponent's party.
                opponent_party = self.get_opponent_party(party)
                if new_board_state[i1][j1] == opponent_party:
                    
                    # If the adjacent position contains an opponent's stone, initiate depth-first search (DFS)
                    dfs_stack = [(i1, j1)]
                    dfs_visited = set()
                    opponent_group_eliminated = True
                    while dfs_stack:
                        top_node = dfs_stack.pop()
                        dfs_visited.add(top_node)
                        
                        # Explore neighboring positions.
                        for index in range(len(X)):
                            i2 = top_node[0] + X[index]
                            j2 = top_node[1] + Y[index]
                            if 0 <= i2 < board_dimension and 0 <= j2 < board_dimension:
                                if (i2, j2) in dfs_visited:
                                    continue
                                elif new_board_state[i2][j2] == not_occupied:
                                    opponent_group_eliminated = False
                                    break
                                elif new_board_state[i2][j2] == opponent_party and \
                                        (i2, j2) not in dfs_visited:
                                    dfs_stack.append((i2, j2))

                    #Determine the eliminated group 
                    if opponent_group_eliminated:
                        for stone in dfs_visited:
                            new_board_state[stone[0]][stone[1]] = not_occupied
        return new_board_state

    def no_of_q1(self, board_sub_state, party):
        # This function calculates the number of "quadrant 1" (q1) cells that have the specified party's piece.

        # Check each of the four corner cells in the given board_sub_state.
        # If a corner cell has the party's piece while the other three corners don't,
        # it is considered a q1 cell.

        if (
            (board_sub_state[0][0] == party and board_sub_state[0][1] != party
                and board_sub_state[1][0] != party and board_sub_state[1][1] != party)
            or (board_sub_state[0][0] != party and board_sub_state[0][1] == party
                and board_sub_state[1][0] != party and board_sub_state[1][1] != party)
            or (board_sub_state[0][0] != party and board_sub_state[0][1] != party
                and board_sub_state[1][0] == party and board_sub_state[1][1] != party)
            or (board_sub_state[0][0] != party and board_sub_state[0][1] != party
                and board_sub_state[1][0] != party and board_sub_state[1][1] == party)
        ):
            # If any of the above conditions are met, there is 1 q1 cell with the party's piece.
            return 1
        else:
            # If none of the conditions are met, there are 0 q1 cells with the party's piece.
            return 0


    def no_of_q2(self, board_sub_state, party):
        # This function calculates the number of "quadrant 2" (q2) cells that have the specified party's piece.

        # Check the two corner cells in the top-right corner of the given board_sub_state.
        # If a corner cell has the party's piece and the other one doesn't, it is considered a q2 cell.

        if (
            (board_sub_state[0][0] == party and board_sub_state[0][1] != party
                and board_sub_state[1][0] != party and board_sub_state[1][1] == party)
            or (board_sub_state[0][0] != party and board_sub_state[0][1] == party
                and board_sub_state[1][0] == party and board_sub_state[1][1] != party)
        ):
            # If any of the above conditions are met, there is 1 q2 cell with the party's piece.
            return 1
        else:
            # If none of the conditions are met, there are 0 q2 cells with the party's piece.
            return 0


    # This function calculates the number of "quadrant 3" (q3) cells that have the specified party's piece.
    def no_of_q3(self, board_sub_state, party):
        # Check the four corner cells in the bottom-right quadrant of the given board_sub_state.
        # If a corner cell has the party's piece and the other three have a different piece, it is considered a q3 cell.

        if (
            (board_sub_state[0][0] == party and board_sub_state[0][1] == party
                and board_sub_state[1][0] == party and board_sub_state[1][1] != party)
            or (board_sub_state[0][0] != party and board_sub_state[0][1] == party
                and board_sub_state[1][0] == party and board_sub_state[1][1] == party)
            or (board_sub_state[0][0] == party and board_sub_state[0][1] != party
                and board_sub_state[1][0] == party and board_sub_state[1][1] == party)
            or (board_sub_state[0][0] != party and board_sub_state[0][1] == party
                and board_sub_state[1][0] == party and board_sub_state[1][1] == party)
        ):
            # If any of the above conditions are met, there is 1 q3 cell with the party's piece.
            return 1
        else:
            # If none of the conditions are met, there are 0 q3 cells with the party's piece.
            return 0


    # Euler function to calculate
    # This function calculates the Euler value for the current board state.
    def calculate_euler_value(self, board_state, party):
        # Determine the opponent's party.
        opponent_party = self.get_opponent_party(party)

        # Create a new board state with an additional border to simplify quadrant calculations.
        new_board_state = np.zeros((board_dimension + 2, board_dimension + 2), dtype=int)

        for i in range(board_dimension):
            for j in range(board_dimension):
                new_board_state[i + 1][j + 1] = board_state[i][j]

        # Initialize variables to count the number of stones in each quadrant for both parties.
        q1_party, q2_party, q3_party, q1_opponent_party, q2_opponent_party, q3_opponent_party = 0, 0, 0, 0, 0, 0

        # Iterate through the board and analyze each 2x2 sub-state to count stones in each quadrant.
        for i in range(board_dimension):
            for j in range(board_dimension):
                new_board_sub_state = new_board_state[i: i + 2, j: j + 2]
                q1_party += self.no_of_q1(new_board_sub_state, party)
                q2_party += self.no_of_q2(new_board_sub_state, party)
                q3_party += self.no_of_q3(new_board_sub_state, party)
                q1_opponent_party += self.no_of_q1(new_board_sub_state, opponent_party)
                q2_opponent_party += self.no_of_q2(new_board_sub_state, opponent_party)
                q3_opponent_party += self.no_of_q3(new_board_sub_state, opponent_party)

        # Calculate the Euler value using the quadrant stone counts for both parties.
        euler_value = (q1_party - q3_party + 2 * q2_party - (q1_opponent_party - q3_opponent_party + 2 * q2_opponent_party)) / 4

        return euler_value


    # Finding the Valid moves out of all possible moves

    # This function finds and categorizes legal moves for a player.
    def find_legal_moves(self, board_state, party):
        # Create dictionaries to categorize legal moves by their types.
        legal_moves = {'3S': [], '1C': [], '2R': []}

        # Iterate over the board to check each cell for possible legal moves.
        for i in range(board_dimension):
            for j in range(board_dimension):
                # Check if the cell is not occupied by any player.
                if board_state[i][j] == not_occupied:
                    # Check if there is a liberty for the group.
                    if self.lookout_for_liberty(board_state, i, j, party):
                        # Check for Ko rule before adding the move.
                        if not self._ko(i, j):
                            # Categorize the move as 3-space (3S) if it's on the edge, otherwise, 2-space (2R).
                            if i == 0 or j == 0 or i == board_dimension - 1 or j == board_dimension - 1:
                                legal_moves.get('3S').append((i, j))
                            else:
                                legal_moves.get('2R').append((i, j))
                    else:
                        # Check the neighboring cells to capture opponent's stones.
                        for index in range(len(X)):
                            i1 = i + X[index]
                            j1 = j + Y[index]
                            if 0 <= i1 < board_dimension and 0 <= j1 < board_dimension:
                                opponent_party = self.get_opponent_party(party)
                                if board_state[i1][j1] == opponent_party:
                                    # Clone the board state and place the stone to capture.
                                    new_board_state = copy.deepcopy(board_state)
                                    new_board_state[i][j] = party
                                    # Check if capturing the opponent's stones creates liberties.
                                    if not self.lookout_for_liberty(new_board_state, i1, j1, opponent_party):
                                        # Check for Ko rule before adding the move.
                                        if not self._ko(i, j):
                                            legal_moves.get('1C').append((i, j))
                                        break

        # Create a list of all legal moves for the player.
        legal_moves_list = []
        for legal_move in legal_moves.get('1C'):
            legal_moves_list.append(legal_move)
        for legal_move in legal_moves.get('2R'):
            legal_moves_list.append(legal_move)
        for legal_move in legal_moves.get('3S'):
            legal_moves_list.append(legal_move)

        return legal_moves_list


    # This function checks for liberty (an empty adjacent cell) of a group of stones at a given position.
    def lookout_for_liberty(self, board_state, i, j, party):
        # Initialize a depth-first search (DFS) stack with the starting position.
        dfs_stack = [(i, j)]
        # Create a set to keep track of visited nodes during the DFS.
        dfs_visited = set()

        # Perform a DFS to check for the presence of an empty adjacent cell (liberty).
        while dfs_stack:
            top_node = dfs_stack.pop()
            dfs_visited.add(top_node)

            # Check the neighboring cells for empty cells or stones of the same party.
            for index in range(len(X)):
                i1 = top_node[0] + X[index]
                j1 = top_node[1] + Y[index]
                if 0 <= i1 < board_dimension and 0 <= j1 < board_dimension:
                    if (i1, j1) in dfs_visited:
                        continue
                    elif board_state[i1][j1] == not_occupied:
                        return True
                    elif board_state[i1][j1] == party and (i1, j1) not in dfs_visited:
                        dfs_stack.append((i1, j1))

        # Return False if no liberties were found.
        return False


    # The opponent player color
    def get_opponent_party(self, party):
        return white_player if party == black_player else black_player

    
    # Ko rule
    def _ko(self, i, j):
        if self.prev_board_state[i][j] != self.party:
            return False
        new_board_state = copy.deepcopy(self.curr_board_state)
        new_board_state[i][j] = self.party
        opponent_i, opponent_j = self.opponent_party_move()
        for index in range(len(X)):
            i1 = i + X[index]
            j1 = j + Y[index]
            if i1 == opponent_i and j1 == opponent_j:

                if not self.lookout_for_liberty(new_board_state, i1, j1, self.opponent_party):

                    self.eliminate_group(new_board_state, i1,
                                         j1, self.opponent_party)

        return np.array_equal(new_board_state, self.prev_board_state)
    # This function identifies the move made by the opponent.
    def opponent_party_move(self):
        # Check if the current and previous board states are the same, indicating a "PASS" move.
        if np.all(self.curr_board_state == self.prev_board_state):
            return None

        # Find the indices of the cell where a change occurred (opponent's move).
        indices = np.where((self.curr_board_state != self.prev_board_state) & (
            self.curr_board_state != not_occupied))

        # If there are indices found, return the first changed cell's coordinates.
        if indices[0].size > 0:
            return indices[0][0], indices[1][0]

        # If no changes were detected, return None to indicate no opponent move.
        return None


    # opponent's move
    # This function identifies the move made by the opponent.
    def opponent_party_move(self):
        # Check if the current and previous board states are the same, indicating a "PASS" move.
        if np.all(self.curr_board_state == self.prev_board_state):
            return None

        # Find the indices of the cell where a change occurred (opponent's move).
        indices = np.where((self.curr_board_state != self.prev_board_state) & (
            self.curr_board_state != not_occupied))

        # If there are indices found, return the first changed cell's coordinates.
        if indices[0].size > 0:
            return indices[0][0], indices[1][0]

        # If no changes were detected, return None to indicate no opponent move.
        return None


    # This function eliminates a group of connected stones of the specified party on the board.
    def eliminate_group(self, board_state, i, j, party):
        # Initialize a depth-first search (DFS) stack with the starting position.
        dfs_stack = [(i, j)]
        # Create a set to keep track of visited nodes during the DFS.
        dfs_visited = set()

        # Perform a DFS to eliminate the group of stones.
        while dfs_stack:
            top_node = dfs_stack.pop()
            dfs_visited.add(top_node)
            # Set the cell to not_occupied, removing the stone.
            board_state[top_node[0]][top_node[1]] = not_occupied

            # Check the neighboring cells for more stones to eliminate.
            for index in range(len(X)):
                i1 = top_node[0] + X[index]
                j1 = top_node[1] + Y[index]
                if 0 <= i1 < board_dimension and 0 <= j1 < board_dimension:
                    if (i1, j1) in dfs_visited:
                        continue
                    elif board_state[i1][j1] == party:
                        dfs_stack.append((i1, j1))

        # Return the updated board state after eliminating the group.
        return board_state


    # This function implements the Min-Max algorithm with Alpha-Beta Pruning.
    def alpha_beta_pruning_search(self, max_depth, b, step_num):
        # Call the max_value_calculate function to find the best move and its value.
        move_of_max_player, move_of_max_value = self.max_value_calculate(
            self.curr_board_state, self.party, max_depth, 0, b, -np.inf, np.inf, None, step_num, False)

        # Write the best move into the output file.
        write_into_output_file(move_of_max_player)


# This function reads input from the specified input file or the default input file.
def read_input(input_file_name=input_file):
    # Open and read the input file.
    with open(input_file_name) as fi:
        # Read and strip each line in the file.
        fi_lines = [fi_line.strip() for fi_line in fi.readlines()]

    # Parse the party, which is an integer.
    party = int(fi_lines[0])
    
    # Parse the previous and current board states as NumPy arrays of integers.
    prev_board_state = np.array([list(map(int, line)) for line in fi_lines[1:6]], dtype=int)
    curr_board_state = np.array([list(map(int, line)) for line in fi_lines[6:11]], dtype=int)

    # Return the party and the board states.
    return party, prev_board_state, curr_board_state


# This function writes the next move into the output file.
def write_into_output_file(next_move):
    # Open the output file for writing.
    with open(output_file, 'w') as fo:
        # If the next move is None or a special "pass" move (-1, -1), write "PASS" to the file.
        if next_move is None or next_move == (-1, -1):
            fo.write('PASS')
        # Otherwise, write the coordinates of the next move to the file.
        else:
            fo.write(f'{next_move[0]},{next_move[1]}')



# This function calculates the step number based on the previous and current board states.
def calculate_step_num(prev_board_state, curr_board_state):
    # Initialize flags to check if the previous and current board states are empty.
    prev_board_state_init = True
    curr_board_state_init = True
    
    # Loop through the board's cells (excluding the last row and column).
    for i in range(board_dimension - 1):
        for j in range(board_dimension - 1):
            # Check if the cell in the previous board state is not empty.
            if prev_board_state[i][j] != not_occupied:
                prev_board_state_init = False
                curr_board_state_init = False
                break
            # Check if the cell in the current board state is not empty.
            elif curr_board_state[i][j] != not_occupied:
                curr_board_state_init = False

    # Determine the step number based on the initialization state of the board states.
    if prev_board_state_init and curr_board_state_init:
        step_num = 0
    elif prev_board_state_init and not curr_board_state_init:
        step_num = 1
    else:
        # If not in the initial state, read the step number from a file and increment it by 2.
        with open(steps_num_file) as fs:
            step_num = int(fs.readline())
            step_num += 2

    # Write the updated step number back to the file.
    with open(steps_num_file, 'w') as fs:
        fs.write(f'{step_num}')

    # Return the calculated step number.
    return step_num



# Check if this script is being run as the main program
if __name__ == '__main__':
    # Read the input, including the party, previous board state, and current board state
    party, prev_board_state, curr_board_state = read_input()
    
    # Calculate the step number (e.g., the current move number)
    step_num = calculate_step_num(prev_board_state, curr_board_state)
    
    # Create an instance of the Player class with the given party and game states
    my_player = Player(party, prev_board_state, curr_board_state)
    
    # Perform an alpha-beta pruning search with a depth of 4, a branching factor of 20, and the current step number
    my_player.alpha_beta_pruning_search(4, 20, step_num)

