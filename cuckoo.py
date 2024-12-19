import numpy as np
import random

# Define the network graph as an adjacency matrix
network = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'C': 1, 'D': 5},
    'C': {'A': 2, 'B': 1, 'D': 8, 'E': 10},
    'D': {'B': 5, 'C': 8, 'E': 2, 'F': 6},
    'E': {'C': 10, 'D': 2, 'F': 3},
    'F': {'D': 6, 'E': 3}
}

# Function to calculate path delay
def path_delay(path, graph):
    delay = 0
    for i in range(len(path) - 1):
        if path[i+1] in graph[path[i]]:
            delay += graph[path[i]][path[i+1]]
        else:
            return float('inf')  # Return high value for invalid paths
    return delay

# Generate a random path
def random_path(start, end, graph):
    nodes = list(graph.keys())
    path = [start]
    while path[-1] != end:
        next_node = random.choice(nodes)
        if next_node not in path:  # Prevent cycles
            path.append(next_node)
    return path

# Perform Levy flight to generate new paths
def levy_flight(path, graph, alpha=1.5):
    nodes = list(graph.keys())
    for _ in range(random.randint(1, 2)):
        random_idx = random.randint(1, len(path) - 2)
        random_node = random.choice(nodes)
        path[random_idx] = random_node
    return path

# Cuckoo Search Algorithm
def cuckoo_search(graph, start, end, population_size=10, max_generations=100, pa=0.25):
    # Initialize population
    nests = [random_path(start, end, graph) for _ in range(population_size)]
    best_path = None
    best_delay = float('inf')
    
    for generation in range(max_generations):
        # Generate new solutions (paths) using Levy flight
        new_nests = [levy_flight(path.copy(), graph) for path in nests]
        
        # Evaluate and replace worse nests
        for i in range(population_size):
            new_delay = path_delay(new_nests[i], graph)
            if new_delay < path_delay(nests[i], graph):
                nests[i] = new_nests[i]
        
        # Find the best path in the current generation
        for path in nests:
            current_delay = path_delay(path, graph)
            if current_delay < best_delay:
                best_delay = current_delay
                best_path = path
        
        # Abandon some nests and generate new ones (controlled by pa)
        for i in range(population_size):
            if random.random() < pa:
                nests[i] = random_path(start, end, graph)
    
    return best_path, best_delay

# Define start and end routers
start_router = 'A'
end_router = 'F'

# Run Cuckoo Search Algorithm
best_path, best_delay = cuckoo_search(network, start_router, end_router)

# Display the results
print(f"Best Path: {best_path}")
print(f"Lowest Delay: {best_delay}")
