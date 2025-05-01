import time
from multiprocessing import Pool

# Fibonnaci Function 
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

# Function to measure execution time
def measure_time(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    return result, end_time - start_time

# Sequential 
def sequential_fibonacci(numbers):
    results = []
    for num in numbers:
        results.append(fibonacci(num))
    return results

# Parallel 
def parallel_fibonacci(numbers):
    with Pool(processes=len(numbers)) as pool:
        results = pool.map(fibonacci, numbers)
    return results

if __name__ == "__main__":
  
    numbers = [20, 25, 30, 35]

    # Sequential execution
    seq_results, seq_time = measure_time(sequential_fibonacci, numbers)
    print(f"Sequential Results: {seq_results}")
    print(f"Sequential Execution Time: {seq_time:.2f} seconds")

    # Parallel execution
    par_results, par_time = measure_time(parallel_fibonacci, numbers)
    print(f"Parallel Results: {par_results}")
    print(f"Parallel Execution Time: {par_time:.2f} seconds")