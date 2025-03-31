import time
import numpy as np
import concurrent.futures
from functools import partial

# -------------------------------------------------
# Fibonacci Functions
# -------------------------------------------------
def fibonacci_sequence(n):
    """
    Returns the Fibonacci sequence as a list of n numbers.
    For n=1, returns [0]. For n>=2, starts with 0, 1.
    """
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    seq = [0, 1]
    for i in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq

def compute_fib(n):
    """
    Global function to compute Fibonacci sequence and timing.
    Returns a tuple (sequence, elapsed_time).
    """
    start = time.time()
    seq = fibonacci_sequence(n)
    elapsed = time.time() - start
    return seq, elapsed

def sequential_fibonacci(fib_numbers):
    """
    Computes Fibonacci sequences sequentially for a list of input sizes.
    Returns a dictionary with the number of terms as key and a tuple (sequence, elapsed_time) as value.
    """
    results = {}
    for num in fib_numbers:
        seq, elapsed = compute_fib(num)
        results[num] = (seq, elapsed)
    return results

def parallel_fibonacci(fib_numbers):
    """
    Computes Fibonacci sequences for a list of input sizes in parallel.
    Uses ProcessPoolExecutor to compute each sequence concurrently.
    Returns a dictionary with the number of terms as key and a tuple (sequence, elapsed_time) as value.
    """
    results = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_num = {executor.submit(compute_fib, num): num for num in fib_numbers}
        for future in concurrent.futures.as_completed(future_to_num):
            num = future_to_num[future]
            seq, elapsed = future.result()
            results[num] = (seq, elapsed)
    return results

# -------------------------------------------------
# Matrix Multiplication Functions
# -------------------------------------------------
def sequential_matrix_multiplication(A, B):
    """
    Performs matrix multiplication using numpy.dot.
    Returns the product matrix and the elapsed time.
    """
    start = time.time()
    C = np.dot(A, B)
    elapsed = time.time() - start
    return C, elapsed

def compute_row(i, A, B):
    """
    Global function to compute the i-th row of the product of matrices A and B.
    """
    return np.dot(A[i, :], B)

def parallel_matrix_multiplication(A, B, num_workers=None):
    """
    Computes the matrix product of A and B in parallel.
    Uses ProcessPoolExecutor to compute each row in parallel.
    
    Note:
    - This function computes the same result as np.dot(A, B).
    - Due to the overhead of starting processes and pickling large arrays,
      the parallel version might be slower than the sequential version,
      especially for moderately sized matrices.
    """
    start = time.time()
    # Bind matrices A and B to the compute_row function.
    compute_row_with_args = partial(compute_row, A=A, B=B)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        result_rows = list(executor.map(compute_row_with_args, range(A.shape[0])))
    C = np.array(result_rows)
    elapsed = time.time() - start
    return C, elapsed

# -------------------------------------------------
# Main function demonstrating both tasks with user input
# -------------------------------------------------
def main():
    # ---------------------------
    # Fibonacci Task: Ask user for input sequence of term lengths
    # ---------------------------
    fib_input = input("Enter Fibonacci term lengths (e.g., '10 20 30 40'): ")
    try:
        fib_numbers = list(map(int, fib_input.split()))
    except ValueError:
        print("Invalid input. Using default Fibonacci term lengths: 10, 20, 30, 40")
        fib_numbers = [10, 20, 30, 40]
    
    print("\n=== Sequential Fibonacci Computation ===")
    seq_fib_results = sequential_fibonacci(fib_numbers)
    for num, (seq, elapsed) in seq_fib_results.items():
        print(f"\nFibonacci sequence for {num} terms computed in {elapsed:.6f} seconds:")
        print(seq)
    
    print("\n=== Parallel Fibonacci Computation ===")
    par_fib_results = parallel_fibonacci(fib_numbers)
    for num, (seq, elapsed) in par_fib_results.items():
        print(f"\nFibonacci sequence for {num} terms computed in {elapsed:.6f} seconds:")
        print(seq)
    
    # ---------------------------
    # Matrix Multiplication Task: Ask user for matrix size
    # ---------------------------
    matrix_size_input = input("\nEnter matrix size (for a square matrix, e.g., 300 or 1000): ")
    try:
        size = int(matrix_size_input)
    except ValueError:
        print("Invalid input. Using default matrix size: 300")
        size = 300
        
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    print(f"\n=== Sequential Matrix Multiplication ({size}x{size}) ===")
    C_seq, time_seq = sequential_matrix_multiplication(A, B)
    print(f"Time taken: {time_seq:.6f} seconds")
    
    print(f"\n=== Parallel Matrix Multiplication ({size}x{size}) ===")
    C_par, time_par = parallel_matrix_multiplication(A, B)
    print(f"Time taken: {time_par:.6f} seconds")
    
    # Verify that both multiplication methods produce nearly identical results.
    if np.allclose(C_seq, C_par):
        print("\nVERIFICATION: The sequential and parallel matrix multiplication results match.")
    else:
        print("\nWARNING: The sequential and parallel results differ.")

    # Note: The parallel version may be slower due to process overhead,
    # especially with moderately sized matrices like the one you provided.

if __name__ == '__main__':
    main()
