import numpy as np
import random

def parse_state_observation_weights(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Extract parameters from the first line
    num_pairs, num_states, num_observations, default_weight = map(int, lines[1].split())

    # Initialize the dictionary
    state_observation_dict = {}
    observation_list=[]
    states_list =[]

    # Parse the remaining lines and populate the dictionary
    for line in lines[2:]:
        tokens = line.strip().split()

        state = tokens[0]
        states_list.append(tokens[0])
        observation = tokens[1]
        observation_list.append(tokens[1])
        weight = int(tokens[2]) if len(tokens) == 3 else default_weight


        # Check if the state is already in the dictionary, if not, add it
        if state not in state_observation_dict:
            state_observation_dict[state] = {}

        

        # Add the observation and weight to the dictionary
        state_observation_dict[state][observation] = weight
        #state_observation_updated_dict = {key.strip('"'): {inner_key.strip('"'): value for inner_key, value in inner_dict.items()}
                #for key, inner_dict in state_observation_dict.items()}
    
    
    unique_observations = list(set(observation_list))
    unique_states = list(set(states_list))

    for state, obs_dict in state_observation_dict.items():
      for element in unique_observations:
        if element not in obs_dict:
            state_observation_dict[state][element] = default_weight
          
    state_observation_updt_dict = {key.strip('"'): value for key, value in state_observation_dict.items()}
    for sub_dict in state_observation_updt_dict.values():
        for inner_key in list(sub_dict):
            sub_dict[inner_key.strip('"')] = sub_dict.pop(inner_key)
  
    sum_values ={}
    normalized_theta ={}

    for state, observations in state_observation_updt_dict.items():
      sum_values [state] =0
      
      sum_values[state] = sum(observations.values())
        # Normalize the values
      normalized_theta [state] = {key: value / sum_values[state] for key, value in observations.items()}

    return num_states, num_observations,normalized_theta


def parse_state_weights(file_path):
    state_weights = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()

    tokens = lines[1].split()
    if len(tokens) == 2:
        num_states, default_weight = tokens
        
    elif len(tokens) == 1:
        num_states = tokens[0]
    
    # Extracting the number of states and default weight
    #num_states, default_weight = map(float, lines[1].split())
    #print(num_states)
    #print(default_weight)
    states_list=[]

    for line in lines[2:]:
        # Split the line into tokens
        tokens = line.strip().split()

        # Extract the state and weight (if available)
        state = tokens[0]
        states_list.append(state)
        weight = int(tokens[1]) if len(tokens) == 2 else default_weight

        # Update the dictionary
        state_weights[state] = weight
        #state_weights = {key.strip('"'): value for key, value in state_weights.items()}

    
    #Remove double quotes 
    state_weight_updt = {key.strip('"'): value for key, value in state_weights.items()}

        
    #Normalization of the weight 
    # Calculate the sum of values
    sum_values = sum(state_weight_updt.values())
    

    # Normalize the values
    normalized_state_weights = {key: value / sum_values for key, value in state_weight_updt.items()}

    return num_states,normalized_state_weights

  
def parse_state_action_state_weights(file_path):  
  transition_dict = {}
  with open(file_path, 'r') as file:
        lines = file.readlines()
      

  # Extract parameters from the first line
  num_triples, num_states, num_actions, default_weight = map(int, lines[1].split())
  states_list =[]

  for line in lines [2:]:
          state, action, next_state, weight = line.strip().split()
          weight = int(weight)
          states_list.append(state)



          if action not in transition_dict:
              transition_dict[action] = {}

          if state not in transition_dict[action]:
              transition_dict[action][state] = {}

          transition_dict[action][state][next_state] = weight

  transition_updt_dict = {}


  
  for action, state_nextstate_dict in transition_dict.items():
    for state, nextstate_dict in state_nextstate_dict.items():
      #print("Next states transition probability:"+ "for "+ state+ "="+ str(next_state_dict))my
      for element in set(states_list):
        if element not in nextstate_dict.keys():
            #print("Next states transition probability:"+ "for "+ state+ "="+ str(next_state_dict))
            transition_dict[action][state][element] = default_weight

  #Removal of double quotes
  for action, state_dict in transition_dict.items():
      new_action = action.strip('"')
      new_state_dict = {state.strip('"'): {next_state.strip('"'): value for next_state, value in next_state_dict.items()} for state, next_state_dict in state_dict.items()}
      transition_updt_dict[new_action] = new_state_dict

   # Normalize the values
  normalized_dict = {}
  for action, state_dict in transition_updt_dict.items():
      normalized_action = {}
      for state, next_state_dict in state_dict.items():
          total_weight = sum(next_state_dict.values())
          normalized_action[state] = {next_state: value / total_weight for next_state, value in next_state_dict.items()}
      normalized_dict[action] = normalized_action
  return normalized_dict

def parse_observation_action(file_path):
  with open(file_path, 'r') as file:
      lines = file.readlines()

  # Extract the number of pairs from the first line
  num_pairs = int(lines[1])
  words =[]

  for line in lines[2:] :

    # Remove double quotes and split the string into a list of words
    words_list = line.replace('"', '').split()
    words.extend (words_list)
  #print(words)
  observation_action_list = words
  return num_pairs, observation_action_list

def calculate_alpha_1_value(pi_dict, theta_dict, observation):
  alpha_1 = {}
  
  for element in pi_dict.keys():
    alpha_1[element] = pi_dict[element]*theta_dict[element][observation]

  return alpha_1


def viterbi_algorithm2():

  #Parsing State Weights (PI values)
  num_states, pi_dict = parse_state_weights('state_weights.txt')

  #print("The Pi Values : ", pi_dict)

  #Parsing theta values (state_observation_weights)
  num_states, num_observations,theta_dict = parse_state_observation_weights('state_observation_weights.txt')
  #print("The theta value: ",theta_dict)

  
  #Observation_action list parsing
  num_pairs, observation_action_list = parse_observation_action('observation_actions.txt')
  #print ("The observation action list: ", observation_action_list)

  
  #Calculating alpha_1
  alpha_1 = calculate_alpha_1_value(pi_dict,theta_dict,observation_action_list[0])
  #print("The alpha_1 value:",alpha_1)

  #Action weights calculation 
  action_weights_dict = parse_state_action_state_weights('state_action_state_weights.txt')
  #print("The action dictionary values: ", action_weights_dict)

  #List to store the path 
  solution = []
  states =[]

  # for state, state_values in alpha_1.items():
  #   initial_path.append(state)
  # solution.append(initial_path)

  states = list(alpha_1.keys())
  solution.append(states)
  solution_temp = []

  #print("The given states: ", states)
  #print("The initial solution: ", solution)

  #alpha_1_dict = alpha_1.copy()
  alpha={0:alpha_1}
  #print("Initial alpha dicitionary: ",alpha)
  w =2
  k =1
  
  for t in range(1, num_pairs):
    solution_temp =[]
    #print("the observation iteration: ",t)
    observation = observation_action_list[w]
    action = observation_action_list[k]
    k+=2 #updating the pointer to get actions from the observation action list
    w+=2 #updating the pointer to get observation from the observation action list
    
    for next_state in states :
      #print("The next state:",next_state)
      max_prob = float('-inf')
      best_prev_state = None
      
      for prev_state in states:
        temp_prob = alpha[t-1][prev_state]*action_weights_dict[action][prev_state][next_state]*theta_dict[next_state][observation]
        #print("   prev_state: ",prev_state)
        #print("   temp_prob:",temp_prob)
        # temp_next_state.append(prev_state)
        if temp_prob > max_prob:
            max_prob = temp_prob
            #print("   max_prob:",max_prob)
            best_prev_state = prev_state
            #print("   best prev state:", best_prev_state)
      # Update alpha for the given time step and next_state
      if t not in alpha:
          alpha[t] = {}
      alpha[t][next_state] = max_prob
      #temp_next_state.append(next_state)
      if best_prev_state in states:
        index = states.index(best_prev_state)
      path = str(solution[t-1][index]+'-'+ next_state)

      #print(path)
      solution_temp.append(path)
      #print(solution_temp)
    solution.append(solution_temp)
    #print(solution)
    #print(temp_next_state)
    #print(alpha)

  path_with_max_probability_value = max(alpha[num_pairs-1], key=lambda k: alpha[num_pairs-1][k])
  path_with_max_probability_index = list(alpha[num_pairs-1].keys()).index(path_with_max_probability_value)

  Final_path = solution[-1][path_with_max_probability_index]
  #print("the final path :",solution[-1][path_with_max_probability_index])

  input_string = Final_path

  # Split the input string into a list of strings
  sequence_list = input_string.split('-')

  # Specify the file path
  file_path = "states.txt"

  # Write the length of the sequence to the file
  with open(file_path, 'w') as file:
      file.write("states\n")
      file.write(f"{len(sequence_list)}\n")

  # Append each string in the sequence to the file
  with open(file_path, 'a') as file:
      for item in sequence_list:
          file.write(f'"{item}"\n')

  
  #print(f"Sequence has been written to {file_path}.")
     
viterbi_algorithm2()