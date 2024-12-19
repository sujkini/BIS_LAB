import numpy as np

class GreyWolfOptimizer:
    def __init__(self, num_wolves, max_iter, num_tasks, task_durations):
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.num_tasks = num_tasks
        self.task_durations = task_durations
        self.positions = np.random.randint(0, num_tasks, (self.num_wolves, self.num_tasks))
        self.best_position = None
        self.best_fitness = float('inf')
    
    def fitness(self, position):
        # Calculate the fitness based on makespan (max time taken by the worker)
        workers = np.zeros(self.num_tasks)
        for i in range(self.num_tasks):
            workers[position[i]] += self.task_durations[i]
        makespan = np.max(workers)
        return makespan
    
    def update_position(self, wolf, a, A, C, alpha_position):
        # Update position based on Grey Wolf Optimization algorithm
        return np.clip(wolf + a * (alpha_position - wolf), 0, self.num_tasks - 1)

    def optimize(self):
        for t in range(self.max_iter):
            # Track the best fitness and position (alpha, beta, delta wolves)
            fitness_values = np.array([self.fitness(self.positions[i]) for i in range(self.num_wolves)])
            sorted_indices = np.argsort(fitness_values)
            
            self.alpha_position = self.positions[sorted_indices[0]]  # Best solution
            self.alpha_fitness = fitness_values[sorted_indices[0]]
            self.beta_position = self.positions[sorted_indices[1]]  # Second best
            self.beta_fitness = fitness_values[sorted_indices[1]]
            self.delta_position = self.positions[sorted_indices[2]]  # Third best
            self.delta_fitness = fitness_values[sorted_indices[2]]
            
            # Update the alpha wolf (best fitness)
            if self.alpha_fitness < self.best_fitness:
                self.best_fitness = self.alpha_fitness
                self.best_position = self.alpha_position.copy()

            a = 2 - t * (2 / self.max_iter)  # Decreasing factor
            A = np.random.uniform(-1, 1, size=self.positions.shape)  # Random vector A
            C = np.random.uniform(0, 2, size=self.positions.shape)  # Random vector C

            for i in range(self.num_wolves):
                if i != 0:  # Skip the alpha wolf (best solution so far)
                    self.positions[i] = self.update_position(self.positions[i], a, A[i], C[i], self.alpha_position)

            print(f"Iteration {t+1}/{self.max_iter}, Best Makespan: {self.best_fitness}")
        return self.best_position, self.best_fitness


# Parameters
num_wolves = 30
max_iter = 100
num_tasks = 10
task_durations = np.random.randint(1, 10, size=num_tasks)  # Random task durations between 1 and 10

# Run Grey Wolf Optimization for task scheduling
gwo = GreyWolfOptimizer(num_wolves, max_iter, num_tasks, task_durations)
best_schedule, best_makespan = gwo.optimize()

print("\nOptimal Task Schedule (Task Assignment to Workers):")
print(best_schedule)
print(f"Optimal Makespan: {best_makespan}")
