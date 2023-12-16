import heapq

class State:
    def __init__(self, left, boat, right):
        self.left = left
        self.boat = boat
        self.right = right
    
    def __eq__(self, other):
        return self.left == other.left and self.boat == other.boat and self.right == other.right
    
    def __lt__(self, other):
        return str(self) < str(other)
    
    def __hash__(self):
        return hash((tuple(self.left), self.boat, tuple(self.right)))
    
    def __str__(self):
        return "Left: {}, Boat: {}, Right: {}".format(self.left, self.boat, self.right)
    
    def is_goal(self):
        return len(self.left) == 0 and self.boat == "right"
    
    def is_valid_state(self):
        # (同様の検証ロジック)
        ...

def get_possible_moves(state):
    # (以前と同様の移動生成ロジック)
    ...

def heuristic(state):
    return len(state.left)

def a_star_search(initial_state):
    priority_queue = [(heuristic(initial_state), initial_state)]
    visited = set()
    node_expanded_count = 0
    expanded_states = {}
    
    while priority_queue:
        _, current_state = heapq.heappop(priority_queue)
        node_expanded_count += 1
        if current_state.is_goal():
            return current_state, node_expanded_count
        visited.add(current_state)
        for move in get_possible_moves(current_state):
            actual_cost = expanded_states.get(current_state, 0) + 1
            if move not in visited or actual_cost < expanded_states.get(move, float('inf')):
                expanded_states[move] = actual_cost
                priority = actual_cost + heuristic(move)
                heapq.heappush(priority_queue, (priority, move))
    return None, node_expanded_count

initial_state = State(['H1', 'W1', 'H2', 'W2', 'H3', 'W3'], "left", [])
a_star_solution, a_star_nodes_expanded = a_star_search(initial_state)