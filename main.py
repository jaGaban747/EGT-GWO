import numpy as np
import streamlit as st
import time
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
import base64
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go


# =====================================
# Data Persistence Utilities
# =====================================
def save_optimization_results(optimizer, key_suffix=""):
    """Save optimization results to session state"""
    key = f"optimization_results{key_suffix}"
    st.session_state[key] = {
        'convergence': optimizer.convergence.copy(),
        'latency_history': optimizer.latency_history.copy(),
        'energy_history': optimizer.energy_history.copy(),
        'response_time_history': optimizer.response_time_history.copy(),
        'node_loads_history': [loads.copy() for loads in optimizer.node_loads_history],
        'alpha_pos': optimizer.alpha_pos.copy() if optimizer.alpha_pos is not None else None,
        'beta_pos': optimizer.beta_pos.copy() if optimizer.beta_pos is not None else None,
        'delta_pos': optimizer.delta_pos.copy() if optimizer.delta_pos is not None else None,
        'population': optimizer.population.copy(),
        'fitness': optimizer.fitness.copy(),
        'iter': optimizer.iter,
        'alpha': optimizer.alpha,
        'beta': optimizer.beta,
        'gamma': optimizer.gamma,
        'pop_size': optimizer.pop_size,
        'max_iter': optimizer.max_iter,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save hybrid-specific data if available
    if hasattr(optimizer, 'equilibrium_history'):
        st.session_state[key]['equilibrium_history'] = [eq.copy() for eq in optimizer.equilibrium_history]
        st.session_state[key]['logit_strategy_probs'] = optimizer.logit_game.strategy_probs.copy()
        st.session_state[key]['logit_utility_history'] = optimizer.logit_game.utility_history.copy()

def load_optimization_results(optimizer, key_suffix=""):
    """Load optimization results from session state"""
    key = f"optimization_results{key_suffix}"
    if key not in st.session_state:
        return False
    
    data = st.session_state[key]
    
    # Restore basic optimizer state
    optimizer.convergence = data['convergence'].copy()
    optimizer.latency_history = data['latency_history'].copy()
    optimizer.energy_history = data['energy_history'].copy()
    optimizer.response_time_history = data['response_time_history'].copy()
    optimizer.node_loads_history = [loads.copy() for loads in data['node_loads_history']]
    optimizer.alpha_pos = data['alpha_pos'].copy() if data['alpha_pos'] is not None else None
    optimizer.beta_pos = data['beta_pos'].copy() if data['beta_pos'] is not None else None
    optimizer.delta_pos = data['delta_pos'].copy() if data['delta_pos'] is not None else None
    optimizer.population = data['population'].copy()
    optimizer.fitness = data['fitness'].copy()
    optimizer.iter = data['iter']
    
    # Restore hybrid-specific data if available
    if hasattr(optimizer, 'equilibrium_history') and 'equilibrium_history' in data:
        optimizer.equilibrium_history = [eq.copy() for eq in data['equilibrium_history']]
        optimizer.logit_game.strategy_probs = data['logit_strategy_probs'].copy()
        optimizer.logit_game.utility_history = data['logit_utility_history'].copy()
    
    return True

def check_parameters_changed(optimizer, alpha, beta, gamma, pop_size, max_iter):
    """Check if optimization parameters have changed"""
    return (optimizer.alpha != alpha or 
            optimizer.gamma != gamma or 
            optimizer.beta != beta or 
            optimizer.pop_size != pop_size or
            optimizer.max_iter != max_iter)

def initialize_persistent_state():
    """Initialize persistent state variables"""
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = True
        st.session_state.optimization_complete = False
        st.session_state.comparison_complete = False
        st.session_state.parameter_study_complete = False
        st.session_state.last_optimization_params = None
        st.session_state.cached_problem = None

# =====================================
# Download Utility Functions
# =====================================
def create_download_button(fig, filename, button_text="Download", key=None):
    """Create a high-quality download button for matplotlib figures"""
    buf = BytesIO()
    # Save with high DPI for quality
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    
    # Convert to base64 for download
    img_str = base64.b64encode(buf.read()).decode()
    href = f'data:image/png;base64,{img_str}'
    
    st.download_button(
        label=button_text,
        data=buf.getvalue(),
        file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        mime="image/png",
        key=key
    )

def create_pdf_download(fig, filename, button_text="Download PDF", key=None):
    """Create a PDF download for matplotlib figures"""
    buf = BytesIO()
    fig.savefig(buf, format='pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    
    st.download_button(
        label=button_text,
        data=buf.getvalue(),
        file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        key=key
    )

def create_svg_download(fig, filename, button_text="Download SVG", key=None):
    """Create a SVG download for matplotlib figures (vector format)"""
    buf = BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    
    st.download_button(
        label=button_text,
        data=buf.getvalue(),
        file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
        mime="image/svg+xml",
        key=key
    )

def create_plotly_download(fig, filename, button_text="Download HTML", key=None):
    """Create an HTML download for Plotly figures"""
    html_string = fig.to_html()
    
    st.download_button(
        label=button_text,
        data=html_string,
        file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html",
        key=key
    )

def create_data_download(data, filename, button_text="Download Data", key=None):
    """Create a CSV download for data"""
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data
    
    csv = df.to_csv(index=False)
    st.download_button(
        label=button_text,
        data=csv,
        file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key=key
    )

# =====================================
# Problem Setup (Shared across pages)
# =====================================
NUM_TASKS = 50
NUM_EDGE_NODES = 10
BETA = 0.7  # Rationality parameter for Logit Dynamics
BANDWIDTH = 100  # Fixed - was missing in original code

@st.cache_data(persist=True)
def generate_problem():
    np.random.seed(42)
    tasks = [
        {'id': i,
         'cpu': np.random.randint(1, 5),
         'deadline': np.random.randint(10, 50),
         'data': np.random.randint(10, 100),
         'loc': np.random.rand(2) * 100}
        for i in range(NUM_TASKS)
    ]
    
    edge_nodes = [
        {'id': i,
         'cpu_cap': np.random.randint(20, 40),
         'loc': np.random.rand(2) * 100,
         'energy_cost': np.random.uniform(0.1, 0.5)}
        for i in range(NUM_EDGE_NODES)
    ]
    return tasks, edge_nodes

# Logit Dynamics Class
class LogitGameTheory:
    def __init__(self, tasks, edge_nodes, alpha, beta, gamma):
        self.tasks = tasks
        self.edge_nodes = edge_nodes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.strategy_probs = np.full((NUM_TASKS, NUM_EDGE_NODES), 1 / NUM_EDGE_NODES)
        self.utility_history = []
        self.equilibrium_history = []
        self.response_times = []  # Track response times

    def calculate_utilities(self, task_idx, node_idx):
        task = self.tasks[task_idx]
        node = self.edge_nodes[node_idx]
        
        # Processing time based on CPU requirements and capacity
        proc_time = task['cpu'] / node['cpu_cap']
        
        # Transmission time based on data size, bandwidth, and distance
        distance = np.linalg.norm(np.array(node['loc']) - np.array(task['loc']))
        tx_time = (task['data'] / BANDWIDTH) * (distance / 100)
        
        # Calculate response time (processing + transmission)
        response_time = proc_time + tx_time
        
        # Utility components
        latency_util = -self.alpha * response_time
        energy_util = -self.gamma * node['energy_cost'] * task['cpu']
        
        return latency_util + energy_util, response_time

    def update_strategies(self):
        new_probs = np.zeros((NUM_TASKS, NUM_EDGE_NODES))
        all_utilities = []
        response_times = []
        
        for i in range(NUM_TASKS):
            utilities_and_times = [self.calculate_utilities(i, j) for j in range(NUM_EDGE_NODES)]
            utilities = [x[0] for x in utilities_and_times]
            times = [x[1] for x in utilities_and_times]
            
            all_utilities.append(utilities)
            response_times.append(times)
            
            # Apply logit function to utilities
            exp_utilities = np.exp(self.beta * np.array(utilities))
            new_probs[i] = exp_utilities / np.sum(exp_utilities)
        
        self.strategy_probs = new_probs
        self.utility_history.append(all_utilities)
        self.response_times.append(response_times)
        self.equilibrium_history.append(np.copy(new_probs))
        
        return new_probs

    def get_nash_equilibrium(self, max_iter=50, tol=1e-4):
        for _ in range(max_iter):
            old_probs = self.strategy_probs.copy()
            self.update_strategies()
            if np.max(np.abs(self.strategy_probs - old_probs)) < tol:
                break
        return self.strategy_probs

# GWO Optimizer Class
class GWOEdgeOptimizer:
    def __init__(self, tasks, edge_nodes, pop_size, max_iter, alpha, beta, gamma):
        self.tasks = tasks
        self.edge_nodes = edge_nodes
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.iter = 0
        
        self.population = np.random.randint(0, NUM_EDGE_NODES, (pop_size, NUM_TASKS))
        self.fitness = np.zeros(pop_size)
        self.alpha_pos = None
        self.beta_pos = None
        self.delta_pos = None
        self.convergence = []
        self.latency_history = []
        self.energy_history = []
        self.response_time_history = []
        self.node_loads_history = []
        # Initialize tracking variables
        self.last_latency = 0
        self.last_energy = 0
        self.last_response_time = 0
        self.last_node_loads = np.zeros(NUM_EDGE_NODES)

    def _compute_fitness(self, solution):
        latency = energy = response_time = 0
        node_loads = np.zeros(NUM_EDGE_NODES)
        
        # Calculate node loads first
        for task_idx, node_idx in enumerate(solution):
            node_loads[node_idx] += self.tasks[task_idx]['cpu']
        
        # Calculate metrics
        for task_idx, node_idx in enumerate(solution):
            task = self.tasks[task_idx]
            node = self.edge_nodes[node_idx]
            
            # Processing time considering current node load
            # Fixed: Calculate remaining capacity correctly by considering load without current task
            current_load = node_loads[node_idx]
            task_cpu = task['cpu']
            remaining_cap = max(0.1, node['cpu_cap'] - (current_load - task_cpu))
            proc_time = task_cpu / remaining_cap
            
            # Transmission time
            distance = np.linalg.norm(np.array(node['loc']) - np.array(task['loc']))
            tx_time = (task['data'] / BANDWIDTH) * (distance / 100)
            
            # Calculate response time
            task_response_time = proc_time + tx_time
            response_time += task_response_time
            
            latency += self.alpha * task_response_time
            energy += self.gamma * node['energy_cost'] * task_cpu
        
        # Calculate averages
        avg_response_time = response_time / NUM_TASKS
        
        # Penalize overloading nodes
        overload_penalty = np.sum(np.maximum(node_loads - np.array([n['cpu_cap'] for n in self.edge_nodes]), 0)) * 1000
        
        # Store metrics for visualization
        self.last_latency = latency / NUM_TASKS
        self.last_energy = energy / NUM_TASKS
        self.last_response_time = avg_response_time
        self.last_node_loads = node_loads
        
        return 1 / (latency + energy + overload_penalty + 1e-10)

    def optimize_step(self, iteration):
        self.iter = iteration
        # Evaluate fitness
        self.fitness = np.array([self._compute_fitness(sol) for sol in self.population])
        
        # Update alpha, beta, delta wolves
        sorted_indices = np.argsort(self.fitness)[::-1]
        self.alpha_pos = self.population[sorted_indices[0]].copy()
        self.beta_pos = self.population[sorted_indices[1]].copy()
        self.delta_pos = self.population[sorted_indices[2]].copy()
        
        # Update convergence history
        self.convergence.append(1 / self.fitness[sorted_indices[0]])
        self.latency_history.append(self.last_latency)
        self.energy_history.append(self.last_energy)
        self.response_time_history.append(self.last_response_time)
        self.node_loads_history.append(self.last_node_loads)
        
        # Update positions
        a = 2 - (2 * iteration) / self.max_iter  # Decreases linearly from 2 to 0
        
        for i in range(self.pop_size):
            A1 = 2 * a * np.random.rand(NUM_TASKS) - a
            A2 = 2 * a * np.random.rand(NUM_TASKS) - a
            A3 = 2 * a * np.random.rand(NUM_TASKS) - a
            C1 = 2 * np.random.rand(NUM_TASKS)
            C2 = 2 * np.random.rand(NUM_TASKS)
            C3 = 2 * np.random.rand(NUM_TASKS)
            
            # Fixed: properly calculate D_delta and other distances
            D_alpha = np.abs(C1 * self.alpha_pos - self.population[i])
            D_beta = np.abs(C2 * self.beta_pos - self.population[i])
            D_delta = np.abs(C3 * self.delta_pos - self.population[i])
            
            X1 = self.alpha_pos - A1 * D_alpha
            X2 = self.beta_pos - A2 * D_beta
            X3 = self.delta_pos - A3 * D_delta
            
            # Calculate new position as average of the three leader-influenced positions
            new_pos = np.round((X1 + X2 + X3) / 3).astype(int)
            # Ensure positions are within valid range
            self.population[i] = np.clip(new_pos, 0, NUM_EDGE_NODES-1)
        
        return self.alpha_pos

# Hybrid Optimizer Class
class HybridOptimizer(GWOEdgeOptimizer):
    def __init__(self, tasks, edge_nodes, pop_size, max_iter, alpha, beta, gamma):
        super().__init__(tasks, edge_nodes, pop_size, max_iter, alpha, beta, gamma)
        self.logit_game = LogitGameTheory(tasks, edge_nodes, alpha, beta, gamma)
        self.equilibrium_history = []

    def _initialize_population(self):
        # Get Nash equilibrium probabilities
        equilibrium_probs = self.logit_game.get_nash_equilibrium()
        
        # Initialize half of the population from equilibrium, half randomly
        for i in range(self.pop_size):
            if i < self.pop_size // 2:
                # Sample from equilibrium probabilities for each task
                self.population[i] = [np.random.choice(NUM_EDGE_NODES, p=equilibrium_probs[task_idx])
                                      for task_idx in range(NUM_TASKS)]
            else:
                # Random initialization for diversity
                self.population[i] = np.random.randint(0, NUM_EDGE_NODES, NUM_TASKS)

    def _compute_fitness(self, solution):
        # Get base fitness from parent class
        base_fitness = super()._compute_fitness(solution)
        
        # Add game theory penalty based on deviation from Nash equilibrium
        equilibrium_probs = self.logit_game.strategy_probs
        strategy_penalty = 0
        for task_idx, node_idx in enumerate(solution):
            chosen_prob = equilibrium_probs[task_idx, node_idx]
            strategy_penalty += -np.log(chosen_prob + 1e-10)  # Avoid log(0)
        
        # Return combined fitness
        return base_fitness / (1 + 0.2 * strategy_penalty / NUM_TASKS)

    def optimize_step(self, iteration):
        # Initialize population using game theory on first iteration
        if iteration == 0:
            self._initialize_population()
        else:
            # Update game theory strategies
            self.logit_game.update_strategies()
            self.equilibrium_history.append(np.copy(self.logit_game.strategy_probs))
        
        # Run standard GWO optimization step
        return super().optimize_step(iteration)

# =====================================
# Page 1: Optimization
# =====================================
def optimization_page():
    st.title("Hybrid GWO-Logit Optimization")
    
    tasks, edge_nodes = generate_problem()
    
    # Parameters - Now in a collapsible section in the main area
    with st.expander("Optimization Parameters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            pop_size = st.slider("Population Size", 10, 100, 30)
            max_iter = st.slider("Max Iterations", 10, 200, 100)
            alpha = st.slider("Latency Weight (α)", 0.0, 1.0, 0.5)
        with col2:
            gamma = st.slider("Energy Weight (γ)", 0.0, 1.0, 0.2)
            beta = st.slider("Rationality (β)", 0.1, 2.0, 0.7)
            animation_speed = st.slider("Animation Speed", 0.01, 1.0, 0.1)
    
    # Normalize weights - fixed to handle zero case properly
    total = alpha + gamma
    if total > 0:
        alpha, gamma = alpha / total, gamma / total
    else:
        # Provide default values if both weights are zero
        alpha, gamma = 0.5, 0.5
    
    # Initialize optimizer in session state
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = HybridOptimizer(tasks, edge_nodes, pop_size, max_iter, alpha, beta, gamma)
        st.session_state.running = False
    else:
        # Check if parameters have changed and reset if needed
        current_opt = st.session_state.optimizer
        if (current_opt.alpha != alpha or current_opt.gamma != gamma or 
            current_opt.beta != beta or current_opt.pop_size != pop_size or
            current_opt.max_iter != max_iter):
            st.session_state.optimizer = HybridOptimizer(tasks, edge_nodes, pop_size, max_iter, alpha, beta, gamma)
            st.session_state.running = False
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Start Optimization"):
            st.session_state.running = True
    with col2:
        if st.button("Pause"):
            st.session_state.running = False
    with col3:
        if st.button("Reset"):
            st.session_state.running = False
            st.session_state.optimizer = HybridOptimizer(tasks, edge_nodes, pop_size, max_iter, alpha, beta, gamma)
            st.rerun()
    
    # Create placeholders
    metrics_placeholder = st.empty()
    plot_placeholder = st.empty()
    results_placeholder = st.empty()
    final_viz_placeholder = st.empty()
    
    # Initialize plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    line1, = ax1.plot([], [], 'b-', linewidth=2)
    ax1.set_title('Convergence Progress')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Value')
    ax1.grid(True)
    
    # Initialize bars with correct number of edge nodes
    bars = ax2.bar(range(NUM_EDGE_NODES), np.zeros(NUM_EDGE_NODES), color='skyblue')
    capacity_line = ax2.plot(range(NUM_EDGE_NODES), [n['cpu_cap'] for n in edge_nodes], 'r--')[0]
    ax2.set_title('Node Utilization')
    ax2.set_xlabel('Node ID')
    ax2.set_ylabel('CPU Units')
    ax2.legend(['Capacity', 'Load'])
    ax2.grid(True)
    
    # Optimization loop
    if st.session_state.running:
        progress_bar = st.progress(0)
        optimizer = st.session_state.optimizer
        
        for iteration in range(optimizer.iter, optimizer.max_iter):
            if not st.session_state.running:
                break
            
            # Run one iteration
            best_solution = optimizer.optimize_step(iteration)
            
            # Update progress
            progress_bar.progress((iteration + 1) / optimizer.max_iter)
            
            # Update metrics
            with metrics_placeholder.container():
                st.markdown("### Current Metrics")
                cols = st.columns(5)
                cols[0].metric("Iteration", f"{iteration+1}/{optimizer.max_iter}")
                cols[1].metric("Objective Value", f"{optimizer.convergence[-1]:.4f}")
                cols[2].metric("Avg Latency", f"{optimizer.latency_history[-1]:.4f} sec")
                cols[3].metric("Avg Energy", f"{optimizer.energy_history[-1]:.4f} units")
                cols[4].metric("Avg Response Time", f"{optimizer.response_time_history[-1]:.4f} sec")
            
            # Update plots
            line1.set_data(range(iteration+1), optimizer.convergence)
            ax1.relim()
            ax1.autoscale_view()
            
            current_loads = optimizer.node_loads_history[-1]
            for i, bar in enumerate(bars):
                bar.set_height(current_loads[i])
                bar.set_color('red' if current_loads[i] > edge_nodes[i]['cpu_cap'] else 'skyblue')
            
            # Set ylim to improve visibility
            ax2.set_ylim(0, max(max(current_loads) * 1.1, max([n['cpu_cap'] for n in edge_nodes]) * 1.1))
            
            # Convert plot to image and display
            buf = BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            plot_placeholder.image(buf, use_container_width=True)
            
            time.sleep(animation_speed)
        
        # Show final results
        if st.session_state.running:
            st.session_state.running = False
            with results_placeholder.container():
                st.success("Optimization Complete!")
                st.markdown("### Final Results")
                
                # Add response time to final results
                st.markdown(f"#### Average Response Time: {optimizer.response_time_history[-1]:.4f} seconds")
                
                # Node utilization statistics
                node_stats = []
                for i in range(NUM_EDGE_NODES):
                    load = optimizer.node_loads_history[-1][i]
                    cap = edge_nodes[i]['cpu_cap']
                    util = load / cap if cap > 0 else float('inf')  # Handle division by zero
                    status = "OVERLOADED" if load > cap else "OK"
                    node_stats.append({
                        "Node ID": i,
                        "CPU Load": f"{load:.1f}",
                        "CPU Capacity": cap,
                        "Utilization": f"{util*100:.1f}%",
                        "Status": status
                    })
                
                st.dataframe(node_stats)
                
                # Download node statistics
                st.markdown("#### Download Results Data")
                col1, col2, col3 = st.columns(3)
                with col1:
                    create_data_download(node_stats, "node_utilization_stats", "Download Node Stats", "node_stats")
                with col2:
                    # Performance metrics data
                    perf_data = {
                        'Iteration': list(range(len(optimizer.convergence))),
                        'Objective_Value': optimizer.convergence,
                        'Latency': optimizer.latency_history,
                        'Energy': optimizer.energy_history,
                        'Response_Time': optimizer.response_time_history
                    }
                    create_data_download(perf_data, "performance_metrics", "Download Performance Data", "perf_data")
                with col3:
                    # Task assignment data
                    task_assign_data = {
                        'Task_ID': list(range(NUM_TASKS)),
                        'Assigned_Node': best_solution.tolist()
                    }
                    create_data_download(task_assign_data, "task_assignments", "Download Task Assignments", "task_assign")
                
                # Add task assignment summary
                task_assignment = np.bincount(best_solution, minlength=NUM_EDGE_NODES)
                st.markdown("#### Tasks per Node:")
                st.write(dict(enumerate(task_assignment)))
            
            # Final Visualizations with Download Options
            with final_viz_placeholder.container():
                st.markdown("## Final Visualizations")
                
                # Create individual downloadable plots
                st.markdown("### Individual Charts")
                
                # 1. Convergence Plot
                fig_conv, ax_conv = plt.subplots(figsize=(10, 6))
                ax_conv.plot(optimizer.convergence, 'b-', linewidth=2, marker='o', markersize=4)
                ax_conv.set_title('Convergence Progress', fontsize=14, fontweight='bold')
                ax_conv.set_xlabel('Iteration', fontsize=12)
                ax_conv.set_ylabel('Objective Value', fontsize=12)
                ax_conv.grid(True, alpha=0.7)
                ax_conv.set_facecolor('#f8f9fa')
                fig_conv.tight_layout()
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.pyplot(fig_conv)
                with col2:
                    st.markdown("**Download Options:**")
                    create_download_button(fig_conv, "convergence_plot", "📊 PNG", "conv_png")
                    create_pdf_download(fig_conv, "convergence_plot", "📄 PDF", "conv_pdf")
                    create_svg_download(fig_conv, "convergence_plot", "🎨 SVG", "conv_svg")
                
                # 2. Node Utilization Chart
                fig_util, ax_util = plt.subplots(figsize=(12, 6))
                node_loads = optimizer.node_loads_history[-1]
                node_capacities = [n['cpu_cap'] for n in edge_nodes]
                node_ids = [f"Node {i}" for i in range(NUM_EDGE_NODES)]
                
                # Calculate unused capacity
                unused_capacity = [max(0, cap - load) for load, cap in zip(node_loads, node_capacities)]
                
                # Create stacked bars with better colors
                bars1 = ax_util.bar(node_ids, node_loads, label='Used Capacity', 
                                   color='#3498db', alpha=0.8)
                bars2 = ax_util.bar(node_ids, unused_capacity, bottom=node_loads, 
                                   label='Unused Capacity', color='#ecf0f1', alpha=0.8)
                ax_util.plot(node_ids, node_capacities, 'r--', linewidth=3, 
                            label='Total Capacity', marker='s', markersize=6)
                
                ax_util.set_title('Node Capacity Utilization', fontsize=14, fontweight='bold')
                ax_util.set_xlabel('Node ID', fontsize=12)
                ax_util.set_ylabel('CPU Units', fontsize=12)
                ax_util.legend(fontsize=10)
                ax_util.tick_params(axis='x', rotation=45)
                ax_util.grid(True, alpha=0.7)
                ax_util.set_facecolor('#f8f9fa')
                
                # Add value labels on bars
                for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                    height1 = bar1.get_height()
                    height2 = bar2.get_height()
                    if height1 > 0:
                        ax_util.text(bar1.get_x() + bar1.get_width()/2., height1/2,
                                    f'{height1:.1f}', ha='center', va='center', 
                                    fontweight='bold', color='white')
                    if height2 > 0:
                        ax_util.text(bar2.get_x() + bar2.get_width()/2., 
                                    height1 + height2/2, f'{height2:.1f}',
                                    ha='center', va='center', fontweight='bold')
                
                fig_util.tight_layout()
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.pyplot(fig_util)
                with col2:
                    st.markdown("**Download Options:**")
                    create_download_button(fig_util, "node_utilization", "📊 PNG", "util_png")
                    create_pdf_download(fig_util, "node_utilization", "📄 PDF", "util_pdf")
                    create_svg_download(fig_util, "node_utilization", "🎨 SVG", "util_svg")
                
                # 3. Performance Metrics Over Time
                fig_perf, ax_perf = plt.subplots(figsize=(12, 6))
                iterations = range(len(optimizer.convergence))
                
                ax_perf.plot(iterations, optimizer.latency_history, 'b-', linewidth=2, 
                            marker='o', markersize=3, label='Latency', alpha=0.8)
                ax_perf.plot(iterations, optimizer.energy_history, 'g-', linewidth=2, 
                            marker='s', markersize=3, label='Energy', alpha=0.8)
                ax_perf.plot(iterations, optimizer.response_time_history, 'r-', linewidth=2, 
                            marker='^', markersize=3, label='Response Time', alpha=0.8)
                
                ax_perf.set_title('Performance Metrics Over Time', fontsize=14, fontweight='bold')
                ax_perf.set_xlabel('Iteration', fontsize=12)
                ax_perf.set_ylabel('Metric Value', fontsize=12)
                ax_perf.legend(fontsize=10)
                ax_perf.grid(True, alpha=0.7)
                ax_perf.set_facecolor('#f8f9fa')
                fig_perf.tight_layout()
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.pyplot(fig_perf)
                with col2:
                    st.markdown("**Download Options:**")
                    create_download_button(fig_perf, "performance_metrics", "📊 PNG", "perf_png")
                    create_pdf_download(fig_perf, "performance_metrics", "📄 PDF", "perf_pdf")
                    create_svg_download(fig_perf, "performance_metrics", "🎨 SVG", "perf_svg")
                
                # 4. Task Distribution Interactive Sunburst Chart
                st.markdown("### Task Distribution Sunburst Chart")
                
                # Create task distribution data
                task_counts = np.bincount(best_solution, minlength=NUM_EDGE_NODES)
                node_labels = [f"Node {i}" for i in range(NUM_EDGE_NODES)]
                
                # Create interactive Plotly sunburst chart
                fig_sunburst = go.Figure(go.Sunburst(
                    labels=["Total Tasks"] + node_labels,
                    parents=[""] + ["Total Tasks"] * NUM_EDGE_NODES,
                    values=[NUM_TASKS] + task_counts.tolist(),
                    branchvalues="total",
                    hovertemplate='<b>%{label}</b><br>Tasks: %{value}<br>Percentage: %{percentParent}<extra></extra>',
                    maxdepth=2,
                    insidetextorientation='radial'
                ))
                
                fig_sunburst.update_layout(
                    title="Task Distribution Across Edge Nodes",
                    font_size=12,
                    width=600,
                    height=600
                )
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.plotly_chart(fig_sunburst, use_container_width=True)
                with col2:
                    st.markdown("**Download Options:**")
                    create_plotly_download(fig_sunburst, "task_distribution_sunburst", "🌐 HTML", "sunburst_html")
                
                # 5. Combined Dashboard View
                st.markdown("### Combined Dashboard")
                
                # Create a comprehensive dashboard
                fig_dashboard = plt.figure(figsize=(16, 12))
                gs = fig_dashboard.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
                
                # Convergence plot
                ax1 = fig_dashboard.add_subplot(gs[0, :2])
                ax1.plot(optimizer.convergence, 'b-', linewidth=2, marker='o', markersize=4)
                ax1.set_title('Convergence Progress', fontweight='bold')
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Objective Value')
                ax1.grid(True, alpha=0.7)
                
                # Performance metrics
                ax2 = fig_dashboard.add_subplot(gs[1, :2])
                iterations = range(len(optimizer.convergence))
                ax2.plot(iterations, optimizer.latency_history, 'b-', label='Latency', linewidth=2)
                ax2.plot(iterations, optimizer.energy_history, 'g-', label='Energy', linewidth=2)
                ax2.plot(iterations, optimizer.response_time_history, 'r-', label='Response Time', linewidth=2)
                ax2.set_title('Performance Metrics', fontweight='bold')
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Metric Value')
                ax2.legend()
                ax2.grid(True, alpha=0.7)
                
                # Node utilization
                ax3 = fig_dashboard.add_subplot(gs[2, :2])
                node_loads = optimizer.node_loads_history[-1]
                node_capacities = [n['cpu_cap'] for n in edge_nodes]
                bars = ax3.bar(range(NUM_EDGE_NODES), node_loads, 
                              color=['red' if load > cap else 'skyblue' 
                                    for load, cap in zip(node_loads, node_capacities)])
                ax3.plot(range(NUM_EDGE_NODES), node_capacities, 'r--', linewidth=2, label='Capacity')
                ax3.set_title('Node Utilization', fontweight='bold')
                ax3.set_xlabel('Node ID')
                ax3.set_ylabel('CPU Units')
                ax3.legend()
                ax3.grid(True, alpha=0.7)
                
                # Task distribution pie chart
                ax4 = fig_dashboard.add_subplot(gs[0, 2])
                task_counts = np.bincount(best_solution, minlength=NUM_EDGE_NODES)
                # Only show nodes with tasks
                non_zero_indices = np.where(task_counts > 0)[0]
                if len(non_zero_indices) > 0:
                    ax4.pie(task_counts[non_zero_indices], 
                           labels=[f'Node {i}' for i in non_zero_indices],
                           autopct='%1.1f%%', startangle=90)
                ax4.set_title('Task Distribution', fontweight='bold')
                
                # Key metrics summary
                ax5 = fig_dashboard.add_subplot(gs[1:, 2])
                ax5.axis('off')
                
                # Calculate additional metrics
                avg_util = np.mean([load/cap for load, cap in zip(node_loads, node_capacities)])
                overloaded_nodes = sum(1 for load, cap in zip(node_loads, node_capacities) if load > cap)
                
                metrics_text = f"""
KEY METRICS

Final Objective: {optimizer.convergence[-1]:.4f}

Performance:
• Avg Latency: {optimizer.latency_history[-1]:.3f}s
• Avg Energy: {optimizer.energy_history[-1]:.3f}
• Avg Response: {optimizer.response_time_history[-1]:.3f}s

Resource Utilization:
• Avg Node Util: {avg_util*100:.1f}%
• Overloaded Nodes: {overloaded_nodes}
• Total Tasks: {NUM_TASKS}
• Total Nodes: {NUM_EDGE_NODES}

Optimization:
• Iterations: {len(optimizer.convergence)}
• Population Size: {optimizer.pop_size}
• Convergence: {'Yes' if len(optimizer.convergence) < optimizer.max_iter else 'Max Iter'}
                """
                
                ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
                
                fig_dashboard.suptitle('Edge Computing Optimization Dashboard', 
                                     fontsize=16, fontweight='bold', y=0.98)
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.pyplot(fig_dashboard)
                with col2:
                    st.markdown("**Download Dashboard:**")
                    create_download_button(fig_dashboard, "optimization_dashboard", "📊 PNG", "dash_png")
                    create_pdf_download(fig_dashboard, "optimization_dashboard", "📄 PDF", "dash_pdf")
                    create_svg_download(fig_dashboard, "optimization_dashboard", "🎨 SVG", "dash_svg")

# =====================================
# Page 2: Analysis
# =====================================
def analysis_page():
    st.title("Optimization Analysis & Comparison")
    
    tasks, edge_nodes = generate_problem()
    
    # Parameter selection for comparison
    st.markdown("### Compare Different Algorithms")
    
    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider("Latency Weight (α)", 0.0, 1.0, 0.5, key="analysis_alpha")
        gamma = st.slider("Energy Weight (γ)", 0.0, 1.0, 0.2, key="analysis_gamma")
    with col2:
        beta = st.slider("Rationality (β)", 0.1, 2.0, 0.7, key="analysis_beta")
        max_iter = st.slider("Max Iterations", 20, 100, 50, key="analysis_iter")
    
    # Normalize weights
    total = alpha + gamma
    if total > 0:
        alpha, gamma = alpha / total, gamma / total
    else:
        alpha, gamma = 0.5, 0.5
    
    if st.button("Run Comparison Analysis"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize algorithms
        algorithms = {
            'Pure GWO': GWOEdgeOptimizer(tasks, edge_nodes, 30, max_iter, alpha, beta, gamma),
            'Hybrid GWO-Logit': HybridOptimizer(tasks, edge_nodes, 30, max_iter, alpha, beta, gamma),
            'Random Assignment': None  # We'll handle this separately
        }
        
        results = {}
        
        # Run GWO
        status_text.text("Running Pure GWO...")
        gwo_optimizer = algorithms['Pure GWO']
        for i in range(max_iter):
            gwo_optimizer.optimize_step(i)
            progress_bar.progress(0.33 * (i + 1) / max_iter)
        
        results['Pure GWO'] = {
            'convergence': gwo_optimizer.convergence,
            'latency': gwo_optimizer.latency_history,
            'energy': gwo_optimizer.energy_history,
            'response_time': gwo_optimizer.response_time_history,
            'best_solution': gwo_optimizer.alpha_pos,
            'final_objective': gwo_optimizer.convergence[-1]
        }
        
        # Run Hybrid
        status_text.text("Running Hybrid GWO-Logit...")
        hybrid_optimizer = algorithms['Hybrid GWO-Logit']
        for i in range(max_iter):
            hybrid_optimizer.optimize_step(i)
            progress_bar.progress(0.33 + 0.33 * (i + 1) / max_iter)
        
        results['Hybrid GWO-Logit'] = {
            'convergence': hybrid_optimizer.convergence,
            'latency': hybrid_optimizer.latency_history,
            'energy': hybrid_optimizer.energy_history,
            'response_time': hybrid_optimizer.response_time_history,
            'best_solution': hybrid_optimizer.alpha_pos,
            'final_objective': hybrid_optimizer.convergence[-1],
            'equilibrium_history': hybrid_optimizer.equilibrium_history
        }
        
        # Random assignment baseline
        status_text.text("Generating Random Assignment Baseline...")
        random_solutions = []
        random_objectives = []
        
        for i in range(max_iter):
            random_solution = np.random.randint(0, NUM_EDGE_NODES, NUM_TASKS)
            # Calculate objective using GWO fitness function
            temp_optimizer = GWOEdgeOptimizer(tasks, edge_nodes, 1, 1, alpha, beta, gamma)
            objective = 1 / temp_optimizer._compute_fitness(random_solution)
            random_objectives.append(objective)
            random_solutions.append(random_solution)
            progress_bar.progress(0.66 + 0.34 * (i + 1) / max_iter)
        
        # Find best random solution
        best_random_idx = np.argmin(random_objectives)
        results['Random Assignment'] = {
            'convergence': random_objectives,
            'best_solution': random_solutions[best_random_idx],
            'final_objective': random_objectives[best_random_idx]
        }
        
        status_text.text("Analysis Complete!")
        progress_bar.progress(1.0)
        
        # Display results
        st.markdown("## Comparison Results")
        
        # Performance comparison table
        comparison_data = []
        for alg_name, result in results.items():
            comparison_data.append({
                'Algorithm': alg_name,
                'Final Objective': f"{result['final_objective']:.4f}",
                'Improvement over Random': f"{((results['Random Assignment']['final_objective'] - result['final_objective']) / results['Random Assignment']['final_objective'] * 100):.1f}%" if alg_name != 'Random Assignment' else 'Baseline'
            })
        
        st.dataframe(comparison_data)
        
        # Convergence comparison plot
        fig_comparison, ax = plt.subplots(figsize=(12, 6))
        
        for alg_name, result in results.items():
            if alg_name != 'Random Assignment':
                ax.plot(result['convergence'], label=alg_name, linewidth=2, marker='o', markersize=3)
            else:
                # Show random as a horizontal line with some noise
                ax.axhline(y=result['final_objective'], color='red', linestyle='--', 
                          linewidth=2, label='Random Assignment (Best)')
        
        ax.set_title('Algorithm Convergence Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.legend()
        ax.grid(True, alpha=0.7)
        ax.set_facecolor('#f8f9fa')
        fig_comparison.tight_layout()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.pyplot(fig_comparison)
        with col2:
            st.markdown("**Download Options:**")
            create_download_button(fig_comparison, "algorithm_comparison", "📊 PNG", "comp_png")
            create_pdf_download(fig_comparison, "algorithm_comparison", "📄 PDF", "comp_pdf")
            create_svg_download(fig_comparison, "algorithm_comparison", "🎨 SVG", "comp_svg")
        
        # Performance metrics comparison
        if 'latency' in results['Pure GWO'] and 'latency' in results['Hybrid GWO-Logit']:
            st.markdown("### Performance Metrics Comparison")
            
            fig_metrics, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Latency comparison
            ax1.plot(results['Pure GWO']['latency'], label='Pure GWO', linewidth=2)
            ax1.plot(results['Hybrid GWO-Logit']['latency'], label='Hybrid', linewidth=2)
            ax1.set_title('Latency Comparison')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Average Latency')
            ax1.legend()
            ax1.grid(True, alpha=0.7)
            
            # Energy comparison
            ax2.plot(results['Pure GWO']['energy'], label='Pure GWO', linewidth=2)
            ax2.plot(results['Hybrid GWO-Logit']['energy'], label='Hybrid', linewidth=2)
            ax2.set_title('Energy Comparison')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Average Energy')
            ax2.legend()
            ax2.grid(True, alpha=0.7)
            
            # Response time comparison
            ax3.plot(results['Pure GWO']['response_time'], label='Pure GWO', linewidth=2)
            ax3.plot(results['Hybrid GWO-Logit']['response_time'], label='Hybrid', linewidth=2)
            ax3.set_title('Response Time Comparison')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Average Response Time')
            ax3.legend()
            ax3.grid(True, alpha=0.7)
            
            # Final metrics bar chart
            metrics = ['Latency', 'Energy', 'Response Time']
            gwo_final = [results['Pure GWO']['latency'][-1], 
                        results['Pure GWO']['energy'][-1],
                        results['Pure GWO']['response_time'][-1]]
            hybrid_final = [results['Hybrid GWO-Logit']['latency'][-1],
                           results['Hybrid GWO-Logit']['energy'][-1],
                           results['Hybrid GWO-Logit']['response_time'][-1]]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax4.bar(x - width/2, gwo_final, width, label='Pure GWO', alpha=0.8)
            ax4.bar(x + width/2, hybrid_final, width, label='Hybrid', alpha=0.8)
            ax4.set_title('Final Performance Metrics')
            ax4.set_xlabel('Metrics')
            ax4.set_ylabel('Value')
            ax4.set_xticks(x)
            ax4.set_xticklabels(metrics)
            ax4.legend()
            ax4.grid(True, alpha=0.7)
            
            fig_metrics.tight_layout()
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.pyplot(fig_metrics)
            with col2:
                st.markdown("**Download Options:**")
                create_download_button(fig_metrics, "performance_comparison", "📊 PNG", "perf_comp_png")
                create_pdf_download(fig_metrics, "performance_comparison", "📄 PDF", "perf_comp_pdf")
                create_svg_download(fig_metrics, "performance_comparison", "🎨 SVG", "perf_comp_svg")
        
        # Download comparison data
        st.markdown("### Download Comparison Data")
        col1, col2, col3 = st.columns(3)
        with col1:
            create_data_download(comparison_data, "algorithm_comparison", "Download Comparison Table", "comp_table")
        with col2:
            # Convergence data
            conv_data = {
                'Iteration': list(range(max_iter)),
                'Pure_GWO': results['Pure GWO']['convergence'],
                'Hybrid_GWO_Logit': results['Hybrid GWO-Logit']['convergence'],
                'Random_Assignment': results['Random Assignment']['convergence']
            }
            create_data_download(conv_data, "convergence_comparison", "Download Convergence Data", "conv_data")
        with col3:
            if 'latency' in results['Pure GWO']:
                perf_comp_data = {
                    'Iteration': list(range(max_iter)),
                    'GWO_Latency': results['Pure GWO']['latency'],
                    'GWO_Energy': results['Pure GWO']['energy'],
                    'GWO_Response_Time': results['Pure GWO']['response_time'],
                    'Hybrid_Latency': results['Hybrid GWO-Logit']['latency'],
                    'Hybrid_Energy': results['Hybrid GWO-Logit']['energy'],
                    'Hybrid_Response_Time': results['Hybrid GWO-Logit']['response_time']
                }
                create_data_download(perf_comp_data, "performance_comparison_data", "Download Performance Data", "perf_comp_data")

# =====================================
# Page 3: Parameters Study
# =====================================
def parameters_study_page():
    st.title("Parameter Sensitivity Analysis")
    
    st.markdown("""
    This page allows you to study how different parameters affect the optimization performance.
    You can analyze the sensitivity of the algorithm to various parameter settings.
    """)
    
    tasks, edge_nodes = generate_problem()
    
    # Parameter study type selection
    study_type = st.selectbox(
        "Select Parameter Study Type",
        ["Weight Parameters (α, γ)", "Population Size", "Rationality Parameter (β)", "Multi-Parameter Grid Search"]
    )
    
    if study_type == "Weight Parameters (α, γ)":
        st.markdown("### Weight Parameters Sensitivity Analysis")
        
        # Parameter ranges
        alpha_range = st.slider("Alpha Range", 0.1, 1.0, (0.2, 0.8), key="alpha_range")
        gamma_range = st.slider("Gamma Range", 0.1, 1.0, (0.1, 0.5), key="gamma_range")
        num_points = st.slider("Number of Points per Parameter", 3, 8, 5)
        max_iter = st.slider("Iterations per Run", 20, 100, 30, key="param_iter")
        
        if st.button("Run Weight Parameter Study"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            alpha_values = np.linspace(alpha_range[0], alpha_range[1], num_points)
            gamma_values = np.linspace(gamma_range[0], gamma_range[1], num_points)
            
            results_grid = np.zeros((num_points, num_points))
            latency_grid = np.zeros((num_points, num_points))
            energy_grid = np.zeros((num_points, num_points))
            response_time_grid = np.zeros((num_points, num_points))
            
            total_runs = num_points * num_points
            run_count = 0
            
            for i, alpha in enumerate(alpha_values):
                for j, gamma in enumerate(gamma_values):
                    # Normalize weights
                    total_weight = alpha + gamma
                    norm_alpha = alpha / total_weight
                    norm_gamma = gamma / total_weight
                    
                    status_text.text(f"Running: α={norm_alpha:.3f}, γ={norm_gamma:.3f}")
                    
                    # Run optimization
                    optimizer = HybridOptimizer(tasks, edge_nodes, 20, max_iter, 
                                              norm_alpha, 0.7, norm_gamma)
                    
                    for iter_num in range(max_iter):
                        optimizer.optimize_step(iter_num)
                    
                    # Store results
                    results_grid[i, j] = optimizer.convergence[-1]
                    latency_grid[i, j] = optimizer.latency_history[-1]
                    energy_grid[i, j] = optimizer.energy_history[-1]
                    response_time_grid[i, j] = optimizer.response_time_history[-1]
                    
                    run_count += 1
                    progress_bar.progress(run_count / total_runs)
            
            status_text.text("Parameter study complete!")
            
            # Create heatmaps
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Objective value heatmap
            im1 = ax1.imshow(results_grid, cmap='viridis', aspect='auto')
            ax1.set_title('Objective Value Heatmap')
            ax1.set_xlabel('Gamma (γ)')
            ax1.set_ylabel('Alpha (α)')
            ax1.set_xticks(range(num_points))
            ax1.set_yticks(range(num_points))
            ax1.set_xticklabels([f'{g:.2f}' for g in gamma_values])
            ax1.set_yticklabels([f'{a:.2f}' for a in alpha_values])
            plt.colorbar(im1, ax=ax1, label='Objective Value')
            
            # Latency heatmap
            im2 = ax2.imshow(latency_grid, cmap='Reds', aspect='auto')
            ax2.set_title('Average Latency Heatmap')
            ax2.set_xlabel('Gamma (γ)')
            ax2.set_ylabel('Alpha (α)')
            ax2.set_xticks(range(num_points))
            ax2.set_yticks(range(num_points))
            ax2.set_xticklabels([f'{g:.2f}' for g in gamma_values])
            ax2.set_yticklabels([f'{a:.2f}' for a in alpha_values])
            plt.colorbar(im2, ax=ax2, label='Latency')
            
            # Energy heatmap
            im3 = ax3.imshow(energy_grid, cmap='Greens', aspect='auto')
            ax3.set_title('Average Energy Heatmap')
            ax3.set_xlabel('Gamma (γ)')
            ax3.set_ylabel('Alpha (α)')
            ax3.set_xticks(range(num_points))
            ax3.set_yticks(range(num_points))
            ax3.set_xticklabels([f'{g:.2f}' for g in gamma_values])
            ax3.set_yticklabels([f'{a:.2f}' for a in alpha_values])
            plt.colorbar(im3, ax=ax3, label='Energy')
            
            # Response time heatmap
            im4 = ax4.imshow(response_time_grid, cmap='Blues', aspect='auto')
            ax4.set_title('Average Response Time Heatmap')
            ax4.set_xlabel('Gamma (γ)')
            ax4.set_ylabel('Alpha (α)')
            ax4.set_xticks(range(num_points))
            ax4.set_yticks(range(num_points))
            ax4.set_xticklabels([f'{g:.2f}' for g in gamma_values])
            ax4.set_yticklabels([f'{a:.2f}' for a in alpha_values])
            plt.colorbar(im4, ax=ax4, label='Response Time')
            
            fig.suptitle('Parameter Sensitivity Analysis: Weight Parameters', fontsize=14, fontweight='bold')
            fig.tight_layout()
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.pyplot(fig)
            with col2:
                st.markdown("**Download Options:**")
                create_download_button(fig, "weight_parameter_study", "📊 PNG", "weight_study_png")
                create_pdf_download(fig, "weight_parameter_study", "📄 PDF", "weight_study_pdf")
                create_svg_download(fig, "weight_parameter_study", "🎨 SVG", "weight_study_svg")
            
            # Find optimal parameters
            best_idx = np.unravel_index(np.argmin(results_grid), results_grid.shape)
            best_alpha = alpha_values[best_idx[0]]
            best_gamma = gamma_values[best_idx[1]]
            
            st.markdown("### Optimal Parameters Found")
            st.success(f"Best α: {best_alpha:.3f}, Best γ: {best_gamma:.3f}")
            st.info(f"Objective Value: {results_grid[best_idx]:.4f}")
            
            # Download parameter study data
            param_study_data = {
                'Alpha': np.repeat(alpha_values, num_points),
                'Gamma': np.tile(gamma_values, num_points),
                'Objective_Value': results_grid.flatten(),
                'Latency': latency_grid.flatten(),
                'Energy': energy_grid.flatten(),
                'Response_Time': response_time_grid.flatten()
            }
            create_data_download(param_study_data, "weight_parameter_study", "Download Parameter Study Data", "param_study_data")

# =====================================
# Main App
# =====================================
def main():
    st.set_page_config(
        page_title="Edge Computing Optimizer",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🚀 Edge Computing Optimizer")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["🎯 Optimization", "📊 Analysis & Comparison", "🔬 Parameter Study"],
        index=0
    )
    
    # Sidebar information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 About")
    st.sidebar.markdown("""
    This application demonstrates:
    - **Hybrid GWO-Logit Optimization**
    - **Real-time visualization**
    - **Algorithm comparison**
    - **Parameter sensitivity analysis**
    - **High-quality exports**
    """)
    
    st.sidebar.markdown("### 🎛️ Current Problem")
    st.sidebar.info(f"""
    - **Tasks**: {NUM_TASKS}
    - **Edge Nodes**: {NUM_EDGE_NODES}
    - **Bandwidth**: {BANDWIDTH} Mbps
    - **Rationality**: {BETA}
    """)
    
    # Main content
    if page == "🎯 Optimization":
        optimization_page()
    elif page == "📊 Analysis & Comparison":
        analysis_page()
    elif page == "🔬 Parameter Study":
        parameters_study_page()

if __name__ == "__main__":
    main()