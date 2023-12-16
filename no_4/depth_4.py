

class State:
    def __init__(self, left, boat, right):
        # (Stateクラスの詳細は省略)
        ...

    # (get_possible_moves関数の詳細は省略)
    def get_possible_moves(state):
        ...

    def dfs(initial_state):
        stack = [initial_state]
        visited = set()
        node_expanded_count = 0
        
        while stack:
            state = stack.pop()
            node_expanded_count += 1
            
            if state.is_goal():
                return state, node_expanded_count
            
            visited.add(state)
            
            moves = get_possible_moves(state)
            for move in moves:
                if move not in visited:
                    stack.append(move)
                    
        return None, node_expanded_count

    initial_state = State(['H1', 'W1', 'H2', 'W2', 'H3', 'W3'], "left", [])
    dfs_solution, dfs_nodes_expanded = dfs(initial_state)