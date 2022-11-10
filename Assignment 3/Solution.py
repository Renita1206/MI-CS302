import numpy
import pygad
import pygad.nn
import pygad.gann

def fitness_func(solution, sol_idx):
    predictions = pygad.nn.predict(last_layer=instance.population_networks[sol_idx], data_inputs=data_inputs)
    #print(predictions)
    correct_predictions = numpy.where(predictions == data_outputs)[0].size
    #print(correct_predictions)
    solution_fitness = (correct_predictions/data_outputs.size)*100
    return solution_fitness

def callback_generation(x):
    updated_wts = pygad.gann.population_as_matrices(population_networks=instance.population_networks, population_vectors=x.population)
    instance.update_population_trained_weights(population_trained_weights=updated_wts)
    #print("Generation = {generation}".format(generation=x.generations_completed))
    #print("Accuracy   = {fitness}".format(fitness=x.best_solution()[1]))

data_inputs = numpy.array([[1, 1],[1, 0],[0, 1],[0, 0]])
data_outputs = numpy.array([0, 1, 1, 0])

instance = pygad.gann.GANN(num_solutions = 10, num_neurons_input=2,num_neurons_hidden_layers=[2],
                           num_neurons_output=2,hidden_activations=["sigmoid"],output_activation="softmax")

population_vectors = pygad.gann.population_as_vectors(population_networks=instance.population_networks)
#print(population_vectors)

g = pygad.GA(num_generations=50, num_parents_mating=3, initial_population=population_vectors.copy(),
             fitness_func=fitness_func,mutation_num_genes=1,mutation_type="random",
             on_generation=callback_generation)


g.run()
g.plot_fitness()
solution, solution_fitness, x = g.best_solution()
print(solution)
print(solution_fitness)