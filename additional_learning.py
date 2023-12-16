import pickle
from shape_arrange import combined_fitness_function
from shape_arrange import select
from shape_arrange import create_new_generation
import matplotlib.pyplot as plt
#あとで色々importする！

#変数をloadする
def load_variables(file_path):
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

file_path = 'saved_data.pkl'
loaded_variables = load_variables(file_path)

population = loaded_variables['population']
best_individual = loaded_variables['best_individual']
best_fitness = loaded_variables['best_fitness']

# 追加学習
additional_generations = 10000  # もう100世代追加したい場合

for generation in range(additional_generations):
    fitnesses = [combined_fitness_function(indv, min_symmetry=3) for indv in population]
    
    # 選択、交叉、突然変異のステップを行う
    selected = select(population, fitnesses)
    population = create_new_generation(selected)
    
    # 新しい最良個体をチェックする
    generation_best_fitness = max(fitnesses)
    if generation_best_fitness > best_fitness:
        best_fitness = generation_best_fitness
        best_individual = population[fitnesses.index(generation_best_fitness)]

# 最良の個体と適応度を記録しておく
final_best_individual = best_individual
final_best_fitness = best_fitness

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
