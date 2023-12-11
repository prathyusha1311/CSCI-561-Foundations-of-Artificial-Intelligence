import numpy as np
import random
import threading
#import matplotlib.pyplot as plt

import time
#import matplotlib.pyplot as plt
# Constants for the TSP problem

f = open("input.txt",'r')


N = int(f.readline())
#print(N) #No of cities
NUM_CITIES = N  # Number of cities


CITIES_COORDINATES = [] #[(x, y) for x, y in zip(np.random.randint(0, 100, NUM_CITIES), np.random.randint(0, 100, NUM_CITIES))]
for line in f:
    items = line.split(' ')
    #print(items)
    CITIES_COORDINATES.append([int(item) for item in items if(item!= '\n')])
#-----------
#Code for num_cities< = 200
# Genetic Algorithm parameters
POPULATION_SIZE = 15
NUM_GENERATIONS = 10
MUTATION_RATE = 0.5
NUM_PARENTS = 2
tournament_size = 5
costbyexecution =[]
#For the cities > 100
# Precompute distances between cities and store in a matrix
#Calculate 3D Distance
def new_calculate_3d_distance(city1, city2):
    x1, y1, z1 = city1
    x2, y2, z2 = city2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


distance_matrix = np.zeros((NUM_CITIES, NUM_CITIES))
for i in range(NUM_CITIES):
    for j in range(i + 1, NUM_CITIES):
        distance_matrix[i][j] = new_calculate_3d_distance(CITIES_COORDINATES[i], CITIES_COORDINATES[j])
        distance_matrix[j][i] = distance_matrix[i][j]

def generate_initial_population(population_size, num_cities):
    population = []
    for _ in range(population_size):
        tour = list(range(num_cities))
        random.shuffle(tour)
        population.append(tour)
    return population


#Calculate 3D Distance
def calculate_3d_distance(city1, city2):
    x1, y1, z1 = city1
    x2, y2, z2 = city2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

# Calculate the total tour distance
def calculate_tour_distance(tour):
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += calculate_3d_distance(CITIES_COORDINATES[tour[i]], CITIES_COORDINATES[tour[i+1]])
    total_distance += calculate_3d_distance(CITIES_COORDINATES[tour[-1]], CITIES_COORDINATES[tour[0]])
    return total_distance

# Fitness function (inverse of the tour distance)
def fitness(tour):
    return 1 / (calculate_tour_distance(tour) + 1e-10)

# ... (Rest of the code remains unchanged up to the genetic_algorithm function)

# Parallelized fitness calculation
def parallel_fitness(population):
    # List to store fitness scores
    fitness_scores = []

    # Worker function for each thread
    def calculate_fitness_thread(start, end):
        for i in range(start, end):
            fitness_scores.append(fitness(population[i]))

    # Create threads
    num_threads = 4  # Adjust the number of threads as needed
    threads = []

    # Divide the work among threads
    chunk_size = len(population) // num_threads
    for i in range(num_threads):
        start = i * chunk_size
        end = start + chunk_size if i < num_threads - 1 else len(population)
        thread = threading.Thread(target=calculate_fitness_thread, args=(start, end))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    return fitness_scores
#----------
# 2-opt local search for improving the tour
def two_opt(tour):
    improved = True
    while improved:
        improved = False
        for i in range(len(tour) - 1):
            for j in range(i + 2, len(tour)):
                if j - i == 1:
                    continue
                new_tour = tour[:i + 1] + tour[i + 1:j][::-1] + tour[j:]
                new_distance = calculate_tour_distance(new_tour)
                if new_distance < calculate_tour_distance(tour):
                    tour = new_tour
                    improved = True
    return tour

# Fast Order Crossover (OX) implementation
def order_crossover(parent1, parent2):
    length = len(parent1)
    start, end = sorted(random.sample(range(length), 2))
    child1 = [-1] * length
    child2 = [-1] * length
    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]

    idx_child1, idx_child2 = end, end

    for i in range(length):
        if parent2[(end + i) % length] not in child1:
            child1[idx_child1 % length] = parent2[(end + i) % length]
            idx_child1 += 1

        if parent1[(end + i) % length] not in child2:
            child2[idx_child2 % length] = parent1[(end + i) % length]
            idx_child2 += 1

    return child1, child2

# Optimized Genetic Algorithm main function
def genetic_algorithm(num_generations, population_size, num_parents, mutation_rate):

    start_time = time.time()  # Record the start time
    if NUM_CITIES <= 50:
      time_limit = 57
    elif NUM_CITIES > 50 and NUM_CITIES <=100:
      time_limit = 67
    elif NUM_CITIES>100 and NUM_CITIES <=200:
      time_limit = 106
    else:
      time_limit = 190


    population = generate_initial_population(population_size, NUM_CITIES)

    best_tour = None
    best_distance = float('inf')

    for generation in range(num_generations):
        population = generate_initial_population(population_size, NUM_CITIES)

    best_tour = None
    best_distance = float('inf')
    costbyexecution=[]
    for generation in range(num_generations):
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Check if the time limit is exceeded
        if elapsed_time >= time_limit:
          break
        fitness_scores = [fitness(tour) for tour in population]
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]

        # Elitism: Preserve the best tour from the previous generation
        if calculate_tour_distance(sorted_population[0]) < best_distance:
            best_tour = sorted_population[0]
            best_distance = calculate_tour_distance(best_tour)

        # Use Roulette Wheel Selection to select parents
        parents = roulette_wheel_selection(sorted_population, fitness_scores)

        offspring = []
        for i in range(0, num_parents, 2):
            child1, child2 = order_crossover(parents[i], parents[i + 1])
            offspring.extend([child1, child2])

        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                offspring[i] = swap_mutation(offspring[i])

            # Apply 2-opt local search to improve the offspring
            offspring[i] = two_opt(offspring[i])

        population = parents + offspring
        costbyexecution.append(best_distance)
        #print("Generation: ",generation, " Cost by execution: ",costbyexecution)
    #Plot of costbyexecution for each generation
    #plt.plot(costbyexecution)
    #plt.show()


    #plt.plot(costbyexecution)
    #plt.show()
    # Final best tour after all generations
    if calculate_tour_distance(sorted_population[0]) < best_distance:
        best_tour = sorted_population[0]
        best_distance = calculate_tour_distance(best_tour)

    return best_tour, best_distance

def custom_selection(population, probabilities):
    selected_parents = []
    for _ in range(len(population)):
        selected_parent = random.choices(population, probabilities)[0]
        selected_parents.append(selected_parent)
    return selected_parents

# Modified Roulette Wheel Selection
def roulette_wheel_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    selected_parents = custom_selection(population, probabilities)
    return selected_parents

# # Improved Tournament Selection
# def tournament_selection(population, tournament_size):
#     selected_parents = []
#     population_size = len(population)
#     for _ in range(population_size):
#         tournament_indices = random.sample(range(population_size), tournament_size)
#         tournament = [population[i] for i in tournament_indices]
#         winner = max(tournament, key=fitness)
#         selected_parents.append(winner)
#     return selected_parents


# Fast Swap Mutation
def swap_mutation(individual):
    idx1, idx2 = sorted(random.sample(range(len(individual)), 2))
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

#-----------
#Code for the new genetic algorithm with cities number >=200

# # Genetic Algorithm parameters
# POPULATION_SIZE = 20
# NUM_GENERATIONS = 100
# MUTATION_RATE = 0.5
# NUM_PARENTS = 5
# tournament_size = 5

# Generate initial population
def new_generate_initial_population(population_size, num_cities):
    population = []
    for _ in range(population_size):
        tour = list(range(num_cities))
        random.shuffle(tour)
        population.append(tour)
    return population



# Update calculate_tour_distance to use the precomputed distances
def new_calculate_tour_distance(tour):
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += distance_matrix[tour[i]][tour[i+1]]
    total_distance += distance_matrix[tour[-1]][tour[0]]
    return total_distance

# # Calculate the total tour distance
# def calculate_tour_distance(tour):
#     total_distance = 0
#     for i in range(len(tour) - 1):
#         total_distance += calculate_3d_distance(CITIES_COORDINATES[tour[i]], CITIES_COORDINATES[tour[i+1]])
#     total_distance += calculate_3d_distance(CITIES_COORDINATES[tour[-1]], CITIES_COORDINATES[tour[0]])
#     return total_distance

# Fitness function (inverse of the tour distance)
def new_fitness(tour):
    return 1 / (new_calculate_tour_distance(tour) + 1e-10)


def new_select_parents_with_probabilities(sorted_population, selection_probabilities, num_parents):
    parents = []
    for _ in range(num_parents):
        selected_parent = None
        rand_val = random.random()
        cumulative_prob = 0

        for ind, prob in zip(sorted_population, selection_probabilities):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                selected_parent = ind
                break

        parents.append(selected_parent)

    return parents

# Usage:
# parents = select_parents_with_probabilities(sorted_population, selection_probabilities, num_parents)

# Tournament Selection
def new_tournament_selection(population, tournament_size):
    selected_parents = []
    for _ in range(len(population)):
        tournament = random.sample(population, tournament_size)
        # Only calculate fitness for the winner
        winner = max(tournament, key=lambda ind: fitness(ind))
        selected_parents.append(winner)
    return selected_parents


#Roulette wheel selection 
# def new_roulette_wheel_selection(population, selection_probabilities, num_parents):
#     parents = []
#     for _ in range(num_parents):
#         rand_val = np.random.rand()
#         cumulative_prob = 0

#         for ind, prob in zip(population, selection_probabilities):
#             cumulative_prob += prob
#             if rand_val <= cumulative_prob:
#                 parents.append(ind)
#                 break

#     return parents

# Rank-based parent selection
# def rank_based_selection(population, num_parents):
#     fitness_scores = [fitness(ind) for ind in population]
#     sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
#     total_rank = sum(range(1, len(population) + 1))
#     selection_probabilities = [(i + 1) / total_rank for i in range(len(population))]
#     parents = select_parents_with_probabilities(sorted_population, selection_probabilities, num_parents)
#     return parents

# Order Crossover (OX)
def new_order_crossover(parent1, parent2):
    length = len(parent1)
    idx1, idx2 = sorted(random.sample(range(length), 2))
    child1 = [-1] * length
    child2 = [-1] * length
    child1[idx1:idx2 + 1] = parent1[idx1:idx2 + 1]
    child2[idx1:idx2 + 1] = parent2[idx1:idx2 + 1]

    idx_child1 = idx2 + 1
    idx_child2 = idx2 + 1

    for i in range(length):
        if child1[(idx2 + i) % length] == -1:
            for j in range(length):
                if parent2[(idx2 + j) % length] not in child1:
                    child1[idx_child1 % length] = parent2[(idx2 + j) % length]
                    idx_child1 += 1
                    break

        if child2[(idx2 + i) % length] == -1:
            for j in range(length):
                if parent1[(idx2 + j) % length] not in child2:
                    child2[idx_child2 % length] = parent1[(idx2 + j) % length]
                    idx_child2 += 1
                    break

    return child1, child2

# Swap Mutation
def new_swap_mutation(individual):
    idx1, idx2 = sorted(random.sample(range(len(individual)), 2))
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

# Genetic Algorithm main function
def new_genetic_algorithm(num_generations, population_size, num_parents, mutation_rate,tournament_size):
    start_time = time.time()  # Record the start time
    
    if NUM_CITIES>100 and NUM_CITIES <200:
      time_limit = 106
    elif NUM_CITIES >= 200 and NUM_CITIES <=500:
      time_limit = 115
    else:
      time_limit = 190

    population = new_generate_initial_population(population_size, NUM_CITIES)


    for generation in range(num_generations):
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Check if the time limit is exceeded
        if elapsed_time >= time_limit:
          break
        fitness_scores = [new_fitness(ind) for ind in population]
        parents = new_tournament_selection(population, tournament_size)
        # Calculate selection probabilities for roulette wheel selection
        # fitness_scores = [new_fitness(ind) for ind in population]
        # total_fitness = sum(fitness_scores)
        # selection_probabilities = [fit / total_fitness for fit in fitness_scores]

        # Select parents using roulette wheel selection
        #parents = new_roulette_wheel_selection(population, selection_probabilities, num_parents)

        offspring = []
        for i in range(0, num_parents, 2):
            child1, child2 = new_order_crossover(parents[i], parents[i + 1])
            offspring.extend([child1, child2])

        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                offspring[i] = new_swap_mutation(offspring[i])

        population = parents + offspring
        best_tour = max(population, key=fitness)
        best_distance = new_calculate_tour_distance(best_tour)
        costbyexecution.append(best_distance)
        #print("Generation: ",generation," Cost by execution: ",costbyexecution)

    best_tour = max(population, key=fitness)
    best_distance = new_calculate_tour_distance(best_tour)

    return best_tour, best_distance

#--------------


if NUM_CITIES > 170:
  # Example usage
  # Genetic Algorithm parameters
  POPULATION_SIZE = 25
  NUM_GENERATIONS = 200
  MUTATION_RATE = 0.8
  NUM_PARENTS = 10
  tournament_size = 5
  # costbyexecution =[]
  best_tour, best_distance = new_genetic_algorithm(NUM_GENERATIONS, POPULATION_SIZE, NUM_PARENTS, MUTATION_RATE,tournament_size)

else:
  # ... (Rest of the code remains unchanged after the genetic_algorithm function)
  best_tour, best_distance = genetic_algorithm(NUM_GENERATIONS, POPULATION_SIZE, NUM_PARENTS, MUTATION_RATE)

#print("Best tour distance:", round(best_distance,4))
rounded_distance = round(best_distance,3)

#for i in best_tour:
  #print(CITIES_COORDINATES[i])

for i in range(len(best_tour)):
  best_tour[i] = best_tour[i]+1


#print("Best tour:", best_tour)
#print("Best tour distance:", best_distance)



# open file in write mode
with open(r'output.txt', 'w') as fp:
    fp.write("{:.3f}\n".format(rounded_distance))
    for i in best_tour:
        # write each item on a new line
        for j in CITIES_COORDINATES[i-1]:
          # if(j == CITIES_COORDINATES[i-1][2]):
          #   fp.write("%s" %j)
          # else:
          fp.write("%s " % j)
        fp.write("\n")
    k = best_tour[0]
    for l in CITIES_COORDINATES[k-1]:
      fp.write("%s " % l)
    #print('Done')
fp.close()


