import numpy as np

class AntColonyOptimization:
    def __init__(self, graph, n_ants, n_iterations, alpha=1.0, beta=2.0, evaporation_rate=0.5, q=1.0):
        self.graph = graph
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # Influence of pheromone
        self.beta = beta  # Influence of heuristic
        self.evaporation_rate = evaporation_rate
        self.q = q  # Pheromone deposit factor
        self.n_nodes = len(graph)
        self.pheromone = np.ones((self.n_nodes, self.n_nodes))  # Initial pheromone levels

    def run(self, start_node, end_node):
        best_route = None
        best_route_length = float('inf')

        for iteration in range(self.n_iterations):
            all_routes = []
            all_route_lengths = []

            for ant in range(self.n_ants):
                route, route_length = self.construct_route(start_node, end_node)
                all_routes.append(route)
                all_route_lengths.append(route_length)

                if route_length < best_route_length:
                    best_route = route
                    best_route_length = route_length

            self.update_pheromone(all_routes, all_route_lengths)
            print(f"Iteration {iteration + 1}: Best Route Length = {best_route_length}")

        return best_route, best_route_length

    def construct_route(self, start_node, end_node):
        route = [start_node]
        visited = set(route)
        current_node = start_node
        route_length = 0

        while current_node != end_node:
            next_node = self.select_next_node(current_node, visited)
            route.append(next_node)
            visited.add(next_node)
            route_length += self.graph[current_node][next_node]
            current_node = next_node

        return route, route_length

    def select_next_node(self, current_node, visited):
        probabilities = []
        for next_node in range(self.n_nodes):
            if next_node not in visited and self.graph[current_node][next_node] > 0:
                pheromone_level = self.pheromone[current_node][next_node] ** self.alpha
                heuristic_value = (1 / self.graph[current_node][next_node]) ** self.beta
                probabilities.append(pheromone_level * heuristic_value)
            else:
                probabilities.append(0)

        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()
        return np.random.choice(range(self.n_nodes), p=probabilities)

    def update_pheromone(self, all_routes, all_route_lengths):
        self.pheromone *= (1 - self.evaporation_rate)

        for route, route_length in zip(all_routes, all_route_lengths):
            for i in range(len(route) - 1):
                self.pheromone[route[i]][route[i + 1]] += self.q / route_length

if __name__ == "__main__":
    # Example graph (adjacency matrix, 0 indicates no connection)
    graph = np.array([
        [0, 2, 0, 1, 0],
        [2, 0, 3, 2, 0],
        [0, 3, 0, 4, 5],
        [1, 2, 4, 0, 3],
        [0, 0, 5, 3, 0],
    ])

    aco = AntColonyOptimization(graph, n_ants=10, n_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5, q=1.0)
    best_route, best_route_length = aco.run(start_node=0, end_node=4)

    print("Best Route:", best_route)
    print("Best Route Length:", best_route_length)
