import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time

# =====================================
# Problem Setup (Customize These Values)
# =====================================
NUM_TASKS = 50       # e.g., IoT/AR/VR tasks
NUM_EDGE_NODES = 10  # e.g., base stations, MEC servers

# Task properties (now with locations)
tasks = [
    {'cpu': np.random.randint(1, 5), 
     'deadline': np.random.randint(10, 50), 
     'data': np.random.randint(10, 100),
     'loc': np.random.rand(2) * 100}  # Added task location
    for _ in range(NUM_TASKS)
]

# Edge node properties (CPU capacity, location (x,y), energy cost)
edge_nodes = [
    {'cpu_cap': np.random.randint(20, 40), 
     'loc': np.random.rand(2) * 100, 
     'energy_cost': np.random.uniform(0.1, 0.5)}
    for _ in range(NUM_EDGE_NODES)
]

# Hyperparameters (initialized as global variables but will be updated via UI)
POP_SIZE = 30        # Number of wolves
MAX_ITER = 100       # Optimization iterations
ALPHA = 0.5          # Weight for latency term
BETA = 0.3           # Weight for energy term
GAMMA = 0.2          # Weight for cost term
BANDWIDTH = 100      # Mbps (for transmission time)

# =====================================
# Enhanced GWO Algorithm with Visualization
# =====================================
class GWOEdgeOptimizer:
    def __init__(self, pop_size=POP_SIZE, max_iter=MAX_ITER, 
                 alpha_weight=ALPHA, beta_weight=BETA, gamma_weight=GAMMA):
        # Initialize iteration counter first
        self.iter = 0
        self.max_iter = max_iter
        self.alpha_weight = alpha_weight
        self.beta_weight = beta_weight
        self.gamma_weight = gamma_weight
        
        # Initialize population
        self.population = np.random.randint(0, NUM_EDGE_NODES, (pop_size, NUM_TASKS))
        self.fitness = np.zeros(pop_size)
        self.pop_size = pop_size
        
        # Initialize wolf positions
        self.alpha_pos = None
        self.beta_pos = None
        self.delta_pos = None
        
        # Setup visualization
        self._init_visualization()
        
        # Initialize convergence list
        self.convergence = []

    def _init_visualization(self):
        """Initialize the visualization components"""
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(18, 6))
        self.fig.suptitle('Grey Wolf Optimization - Live Visualization')
        self._plot_edge_nodes()
        self._plot_tasks()
        plt.tight_layout()

    def _compute_fitness(self, solution):
        """Calculate fitness with fixed task locations"""
        latency = energy = cost = 0
        node_loads = np.zeros(NUM_EDGE_NODES)

        for task_idx, node_idx in enumerate(solution):
            task = tasks[task_idx]
            node = edge_nodes[node_idx]

            # Processing time calculation
            proc_time = task['cpu'] / node['cpu_cap']
            tx_time = (task['data'] / BANDWIDTH) * np.linalg.norm(node['loc'] - task['loc'])
            latency += self.alpha_weight * (proc_time + tx_time)
            
            energy += self.beta_weight * (node['energy_cost'] * task['cpu'])
            cost += self.gamma_weight * (proc_time * 0.1)
            node_loads[node_idx] += task['cpu']

        overload = np.sum(np.maximum(node_loads - [n['cpu_cap'] for n in edge_nodes], 0))
        penalty = 1e3 * overload
        return 1 / (latency + energy + cost + penalty + 1e-10)

    def _update_visualization(self):
        """Update all visualization components"""
        # Clear previous frame
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()

        # Plot task allocation
        self._plot_edge_nodes()
        self._plot_tasks()
        self._plot_connections(self.alpha_pos)
        self.ax1.set_title(f'Iteration {self.iter+1}\nTask Allocation')

        # Plot convergence
        if self.convergence:
            self.ax2.plot(self.convergence, 'b-', linewidth=2)
            self.ax2.set_yscale('log')
            self.ax2.set_title('Optimization Convergence')
            self.ax2.set_xlabel('Iteration')
            self.ax2.set_ylabel('Objective Value')
            self.ax2.grid(True)

        # Plot node loads
        if self.alpha_pos is not None:
            node_loads = self._calculate_node_loads(self.alpha_pos)
            node_capacities = [n['cpu_cap'] for n in edge_nodes]
            
            # Bar plot for loads
            bars = self.ax3.bar(range(NUM_EDGE_NODES), node_loads, color='orange')
            
            # Overlay line for capacity
            self.ax3.hlines(node_capacities, range(NUM_EDGE_NODES), 
                          [i+0.8 for i in range(NUM_EDGE_NODES)], 
                          colors='red', linestyles='dashed', linewidth=2)
            
            self.ax3.set_title('Node Resource Utilization')
            self.ax3.set_xlabel('Node Index')
            self.ax3.set_ylabel('CPU Load')
            self.ax3.set_xticks(range(NUM_EDGE_NODES))
            self.ax3.set_ylim(0, max(node_capacities) * 1.1)

        # Make sure the figure layout is tight
        plt.tight_layout()

    def _plot_edge_nodes(self):
        """Plot edge nodes with their capacities"""
        for idx, node in enumerate(edge_nodes):
            self.ax1.scatter(*node['loc'], s=300, marker='s', 
                           label=f'Node {idx}' if idx < 3 else "",
                           edgecolors='black')
            self.ax1.text(*node['loc'], f'{node["cpu_cap"]}U', 
                        ha='center', va='bottom', fontsize=8)
        
        # Set limits and labels
        self.ax1.set_xlim(0, 100)
        self.ax1.set_ylim(0, 100)
        self.ax1.set_xlabel('X coordinate')
        self.ax1.set_ylabel('Y coordinate')

    def _plot_tasks(self):
        """Plot tasks without displaying deadline for each task"""
        task_locs = np.array([t['loc'] for t in tasks])
        
        if len(task_locs) > 0:  # Check if there are tasks to plot
            # Use a single color for all tasks instead of coloring by deadline
            self.ax1.scatter(task_locs[:,0], task_locs[:,1], 
                           color='blue', marker='o', s=50,
                           alpha=0.7, edgecolor='black')
            
            # Add a single legend entry for tasks instead of a colorbar
            self.ax1.scatter([], [], color='blue', marker='o', 
                          label='Tasks', edgecolor='black')
            self.ax1.legend(loc='upper right')

    def _plot_connections(self, solution):
        """Draw connections between tasks and their assigned nodes"""
        if solution is None:
            return
            
        node_loads = self._calculate_node_loads(solution)
        for task_idx, node_idx in enumerate(solution):
            task_loc = tasks[task_idx]['loc']
            node_loc = edge_nodes[node_idx]['loc']
            self.ax1.plot([task_loc[0], node_loc[0]], 
                        [task_loc[1], node_loc[1]], 
                        color='gray', alpha=0.3, linewidth=0.5)
            
        # Update node size based on load (separate loop to avoid overlapping)
        for node_idx, load in enumerate(node_loads):
            node_loc = edge_nodes[node_idx]['loc']
            size_factor = min(load / edge_nodes[node_idx]['cpu_cap'], 1.0)
            self.ax1.scatter(*node_loc, s=300 * (1 + size_factor),
                           marker='s', edgecolors='red', facecolors='none')

    def _calculate_node_loads(self, solution):
        """Calculate current node loads for a solution"""
        if solution is None:
            return np.zeros(NUM_EDGE_NODES)
            
        node_loads = np.zeros(NUM_EDGE_NODES)
        for task_idx, node_idx in enumerate(solution):
            node_loads[node_idx] += tasks[task_idx]['cpu']
        return node_loads

    def step(self):
        """Simulate one iteration of the GWO algorithm"""
        # Fitness evaluation
        self.fitness = np.array([self._compute_fitness(sol) for sol in self.population])
        
        # Find best solutions (sort by fitness in descending order)
        sorted_indices = np.argsort(self.fitness)[::-1]
        
        # Update wolf positions
        self.alpha_pos = self.population[sorted_indices[0]].copy()
        self.beta_pos = self.population[sorted_indices[1]].copy() if len(sorted_indices) > 1 else self.alpha_pos.copy()
        self.delta_pos = self.population[sorted_indices[2]].copy() if len(sorted_indices) > 2 else self.beta_pos.copy()
        
        # Record convergence (using the best fitness value)
        best_fitness = self.fitness[sorted_indices[0]]
        self.convergence.append(1 / (best_fitness + 1e-10))

        # GWO position updates
        a = 2 - (2 * self.iter) / self.max_iter  # Linear decrease from 2 to 0
        
        for i in range(self.pop_size):
            # Calculate A and C coefficients for alpha, beta, and delta
            A1 = 2 * a * np.random.rand(NUM_TASKS) - a  # Exploration/exploitation parameter for alpha
            A2 = 2 * a * np.random.rand(NUM_TASKS) - a  # For beta
            A3 = 2 * a * np.random.rand(NUM_TASKS) - a  # For delta

            C1 = 2 * np.random.rand(NUM_TASKS)  # Randomness for alpha
            C2 = 2 * np.random.rand(NUM_TASKS)  # For beta
            C3 = 2 * np.random.rand(NUM_TASKS)  # For delta

            # Move toward alpha, beta, and delta wolves
            D_alpha = np.abs(C1 * self.alpha_pos - self.population[i])
            D_beta = np.abs(C2 * self.beta_pos - self.population[i])
            D_delta = np.abs(C3 * self.delta_pos - self.population[i])

            X1 = self.alpha_pos - A1 * D_alpha
            X2 = self.beta_pos - A2 * D_beta
            X3 = self.delta_pos - A3 * D_delta

            # Update the wolf's position
            new_pos = (X1 + X2 + X3) / 3  # Average of alpha, beta, and delta positions

            # Add some randomness to the new position (smaller chance for exploration)
            if np.random.rand() < 0.1:  # 10% chance of random tweak
                new_pos += np.random.randint(-1, 2, size=NUM_TASKS)  # Small random adjustment

            # Ensure the new position is valid (within node index range)
            new_pos = np.clip(np.round(new_pos), 0, NUM_EDGE_NODES - 1).astype(int)

            # Update the population
            self.population[i] = new_pos

        # Update visualization
        self._update_visualization()
        
        # Increment iteration counter
        self.iter += 1
        
        return self.alpha_pos, 1 / (best_fitness + 1e-10)

# =====================================
# Streamlit UI
# =====================================
def main():
    st.title("Grey Wolf Optimization for Edge Resource Allocation")

    # Sidebar controls for parameters
    with st.sidebar:
        st.header("Algorithm Parameters")
        pop_size = st.slider("Population Size", 10, 100, POP_SIZE)
        max_iter = st.slider("Max Iterations", 10, 200, MAX_ITER)
        alpha_weight = st.slider("Latency Weight (ALPHA)", 0.0, 1.0, ALPHA)
        beta_weight = st.slider("Energy Weight (BETA)", 0.0, 1.0, BETA)
        gamma_weight = st.slider("Cost Weight (GAMMA)", 0.0, 1.0, GAMMA)
        
        # Normalize weights to sum to 1.0
        total = alpha_weight + beta_weight + gamma_weight
        if total > 0:
            alpha_weight = alpha_weight / total
            beta_weight = beta_weight / total
            gamma_weight = gamma_weight / total
            
        st.write(f"Normalized weights: α={alpha_weight:.2f}, β={beta_weight:.2f}, γ={gamma_weight:.2f}")

    # Initialize the optimizer with user parameters
    optimizer = GWOEdgeOptimizer(pop_size, max_iter, alpha_weight, beta_weight, gamma_weight)

    # Run optimization and update plots in real-time
    if st.button("Run Optimization"):
        # Create progress bar
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()  # Reserve space for the plot
        metrics_placeholder = st.empty()  # Reserve space for metrics

        best_solution = None
        best_fitness = 0
        
        for iter in range(max_iter):
            # Run one step of the optimizer
            solution, fitness = optimizer.step()
            
            # Track best solution
            if iter == 0 or fitness > best_fitness:
                best_solution = solution.copy()
                best_fitness = fitness
            
            # Display the updated plot
            plot_placeholder.pyplot(optimizer.fig)
            
            # Update metrics
            col1, col2, col3 = metrics_placeholder.columns(3)
            col1.metric("Current Iteration", f"{iter+1}/{max_iter}")
            col2.metric("Best Fitness", f"{1/best_fitness:.4f}")
            
            # Calculate current allocation
            node_allocation = np.bincount(solution, minlength=NUM_EDGE_NODES)
            most_loaded = np.argmax(node_allocation)
            col3.metric("Most Loaded Node", f"Node {most_loaded} ({node_allocation[most_loaded]} tasks)")
            
            # Update progress
            progress_bar.progress((iter + 1) / max_iter)
            
            # Add a small delay for smoother visualization
            time.sleep(0.1)

        st.success("Optimization complete!")

        # Final results
        st.subheader("Final Task Allocation Results")
        
        # Create a visualization of final allocation
        fig, ax = plt.subplots(figsize=(10, 6))
        node_loads = np.bincount(best_solution, minlength=NUM_EDGE_NODES)
        node_capacities = [n['cpu_cap'] for n in edge_nodes]
        
        # Create bar chart
        bar_positions = np.arange(NUM_EDGE_NODES)
        bars = ax.bar(bar_positions, node_loads, color='skyblue')
        
        # Add capacity lines
        for i, cap in enumerate(node_capacities):
            ax.hlines(cap, i-0.4, i+0.4, colors='red', linestyles='dashed', linewidth=2)
        
        # Mark overloaded nodes
        for i, (load, cap) in enumerate(zip(node_loads, node_capacities)):
            if load > cap:
                bars[i].set_color('crimson')
        
        # Add labels and formatting
        ax.set_xlabel('Edge Node')
        ax.set_ylabel('Number of Tasks')
        ax.set_title('Final Task Allocation')
        ax.set_xticks(bar_positions)
        ax.set_xticklabels([f'Node {i}' for i in range(NUM_EDGE_NODES)])
        plt.xticks(rotation=45)
        
        # Add text labels on bars
        for i, v in enumerate(node_loads):
            ax.text(i, v + 0.5, str(v), ha='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display detailed allocation
        st.subheader("Task Assignment Details")
        
        # Create a dataframe with the assignment details
        assignment_data = []
        for task_idx, node_idx in enumerate(best_solution):
            task = tasks[task_idx]
            node = edge_nodes[node_idx]
            
            # Calculate metrics
            proc_time = task['cpu'] / node['cpu_cap']
            tx_time = (task['data'] / BANDWIDTH) * np.linalg.norm(node['loc'] - task['loc'])
            energy = node['energy_cost'] * task['cpu']
            
            assignment_data.append({
                "Task ID": task_idx,
                "Assigned Node": f"Node {node_idx}",
                "CPU Req": task['cpu'],
                "Data Size": f"{task['data']} MB",
                "Deadline": task['deadline'],
                "Proc. Time": f"{proc_time:.2f}s",
                "Tx Time": f"{tx_time:.2f}s",
                "Total Time": f"{(proc_time + tx_time):.2f}s",
                "Energy": f"{energy:.2f}",
            })
        
        # Show as an expandable details section
        with st.expander("View Task Assignment Details"):
            # Convert to columns for display (pagination)
            for i in range(0, len(assignment_data), 10):
                batch = assignment_data[i:i+10]
                cols = st.columns(len(batch))
                for j, item in enumerate(batch):
                    with cols[j]:
                        st.write(f"**Task {item['Task ID']}**")
                        st.write(f"→ {item['Assigned Node']}")
                        st.write(f"CPU: {item['CPU Req']}U")
                        st.write(f"Time: {item['Total Time']}")

if __name__ == "__main__":
    main()