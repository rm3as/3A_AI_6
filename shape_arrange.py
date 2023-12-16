import numpy as np
import random
from scipy.spatial import distance
import matplotlib.pyplot as plt

# Set parameters
num_points = 23
pop_size = 50
num_generations = 10000  
mutation_rate = 0.1
crossover_rate = 0.1
selection_pressure = 0.2
min_symmetry = 3

# Fitness function to minimize
def fitness_function(positions):
    dist_matrix = distance.pdist(positions)
    dist_sum = np.sum(dist_matrix)
    even_distribution_score = 1 / (1 + dist_sum)
    repulsion_score = np.sum(1 / (1 + dist_matrix**2))
    return even_distribution_score + repulsion_score

# Symmetry evaluation
def evaluate_symmetry(positions, symmetry_degree, centroid, threshold=0.01):
    rotation_angle = 2 * np.pi / symmetry_degree
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                [np.sin(rotation_angle), np.cos(rotation_angle)]])
    rotated_positions = np.dot(positions - centroid, rotation_matrix) + centroid
    min_distances = np.min(distance.cdist(positions, rotated_positions), axis=1)
    return all(min_distance < threshold for min_distance in min_distances)

# Symmetry score
def find_symmetry_score(positions, min_symmetry):
    # Modified to return symmetry score as a factor, not as a hard constraint.
    centroid = np.mean(positions, axis=0)
    for symmetry_degree in range(num_points, min_symmetry - 1, -1):
        if evaluate_symmetry(positions, symmetry_degree, centroid):
            return symmetry_degree  # Return the degree of symmetry if one is found
    # If no symmetry is found, return the lowest degree (2), so some score is still given.
    return 2

def inverse_variance_of_min_distances(positions):
    min_distances = []
    for i, point in enumerate(positions):
        other_points = np.delete(positions, i, axis=0)
        distances = distance.cdist([point], other_points)
        min_distance = np.min(distances)
        min_distances.append(min_distance)
    
    variance = np.var(min_distances)  
    
    score = 1 / (1 + variance  )
    return score


# Combined fitness function
def combined_fitness_function(positions, min_symmetry):
    # Same calculation for distribution and repulsion.
    distribution_score = fitness_function(positions)
    
    # Calculate symmetry-related score.
    symmetry_degree = find_symmetry_score(positions, min_symmetry)
    symmetry_score = symmetry_degree if symmetry_degree >= min_symmetry else 0

    distance_score = inverse_variance_of_min_distances(positions)
    
    # Combine the scores with more weight on distribution and repulsion if minimum symmetry not met.
    if symmetry_score == 0:
        return (distribution_score + distance_score)
    else:
        # Give some additional weight to symmetry above the minimum.
        return (distribution_score + 10*distance_score) * (1 + 0.1 * (symmetry_degree - min_symmetry))


# Initialize population with random position vectors
def initialize_population(pop_size):
    return [np.random.rand(num_points, 2) for _ in range(pop_size)]

# Selection of individuals with higher fitness
def select(population, fitnesses):
    selected = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]
    return selected[:int(selection_pressure * len(selected))]

# Crossover to produce offspring from two parents
def crossover(parent1, parent2):
    child = np.copy(parent1)
    for i in range(num_points):
        if random.random() < crossover_rate:
            child[i, :] = parent2[i, :]
    return child

# Mutation to introduce genetic diversity
def mutate(individual):
    for i in range(num_points):
        if random.random() < mutation_rate:
            individual[i, :] = np.random.rand(2)
    return individual

# Create the next generation
def create_new_generation(selected):
    new_generation = []
    while len(new_generation) < pop_size:
        parents = random.sample(selected, 2)
        child1 = crossover(parents[0], parents[1])
        child2 = crossover(parents[1], parents[0])
        new_generation.append(mutate(child1))
        new_generation.append(mutate(child2))
    return new_generation[:pop_size]

# Actual genetic algorithm
population = initialize_population(pop_size)
best_individual = None
best_fitness = 0

for generation in range(num_generations):
    fitnesses = [combined_fitness_function(indv, min_symmetry) for indv in population]
    population = create_new_generation(select(population, fitnesses))
    # Update best individual
    generation_best_fitness = max(fitnesses)
    if generation_best_fitness > best_fitness:
        best_fitness = generation_best_fitness
        best_individual = population[fitnesses.index(best_fitness)]

#Save variables
import pickle

data = {
    'population': population,
    'best_individual': best_individual,
    'best_fitness': best_fitness
}

with open('saved_data.pkl', 'wb') as file:
    pickle.dump(data, file)



# Plot the best configuration
plt.figure(figsize=(8, 8))
plt.scatter(best_individual[:, 0], best_individual[:, 1], c='blue', edgecolors='white', s=100)
plt.title('Best Configuration with Symmetry')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.xscale('linear') 
plt.yscale('linear') 
plt.show()

best_individual, best_fitness