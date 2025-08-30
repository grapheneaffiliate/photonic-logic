"""
Simple fallback optimizer for when DANTE fails.
"""

import numpy as np
from typing import Dict, Any, Callable, Tuple


def simple_random_search(
    objective_func: Callable,
    bounds: Tuple[np.ndarray, np.ndarray],
    n_iterations: int = 100,
    n_samples_per_iter: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple random search optimizer as fallback.
    
    Args:
        objective_func: Function to optimize
        bounds: (lower_bounds, upper_bounds) arrays
        n_iterations: Number of iterations
        n_samples_per_iter: Samples per iteration
        
    Returns:
        (best_x, best_y) tuple
    """
    lb, ub = bounds
    dims = len(lb)
    
    # Initialize with random samples
    all_x = []
    all_y = []
    
    for i in range(n_iterations):
        # Generate random samples
        for _ in range(n_samples_per_iter):
            x = np.random.uniform(lb, ub)
            y = objective_func(x)
            all_x.append(x)
            all_y.append(y)
        
        # Print progress
        if (i + 1) % 10 == 0:
            best_idx = np.argmax(all_y)
            print(f"Iteration {i+1}/{n_iterations}: Best score = {all_y[best_idx]:.4f}")
    
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    
    return all_x, all_y


def gradient_free_optimization(
    objective_func: Callable,
    bounds: Tuple[np.ndarray, np.ndarray],
    n_iterations: int = 100,
    population_size: int = 20,
    elite_fraction: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple gradient-free optimization using cross-entropy method.
    
    Args:
        objective_func: Function to optimize
        bounds: (lower_bounds, upper_bounds) arrays
        n_iterations: Number of iterations
        population_size: Population size per iteration
        elite_fraction: Fraction of population to keep as elite
        
    Returns:
        (all_x, all_y) arrays of all evaluated points
    """
    lb, ub = bounds
    dims = len(lb)
    n_elite = max(2, int(population_size * elite_fraction))
    
    # Initialize mean and std
    mean = (lb + ub) / 2
    std = (ub - lb) / 4
    
    all_x = []
    all_y = []
    
    for i in range(n_iterations):
        # Sample population
        population = []
        scores = []
        
        for _ in range(population_size):
            # Sample from current distribution
            x = np.random.normal(mean, std)
            # Clip to bounds
            x = np.clip(x, lb, ub)
            
            # Evaluate
            score = objective_func(x)
            population.append(x)
            scores.append(score)
            
            all_x.append(x)
            all_y.append(score)
        
        population = np.array(population)
        scores = np.array(scores)
        
        # Select elite
        elite_idx = np.argsort(scores)[-n_elite:]
        elite = population[elite_idx]
        
        # Update distribution
        mean = np.mean(elite, axis=0)
        std = np.std(elite, axis=0) + 1e-5  # Add small value to prevent collapse
        
        # Decay std over time
        std *= 0.95
        
        # Print progress
        if (i + 1) % 10 == 0:
            best_score = np.max(scores)
            print(f"Iteration {i+1}/{n_iterations}: Best score = {best_score:.4f}")
    
    return np.array(all_x), np.array(all_y)
