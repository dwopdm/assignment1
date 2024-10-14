import time
import random

def flatten_list_loop(nested_list):
    flat_list = []
    for sublist in nested_list:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def flatten_list_comprehension(nested_list):
    return [item for sublist in nested_list for item in sublist]


def char_count_loop(s):
    count = {}
    for char in s:
        if char.islower() or char == ' ':
            if char in count:
                count[char] += 1
            else:
                count[char] = 1
    return count

def char_count_comprehension(s):
    return {char: s.count(char) for char in set(s) if char.islower() or char == ' '}




def flattening_experiment():
    sizes = [10**3, 10**4, 10**5, 10**6, 10**7]
    results = []

    for size in sizes:
        # Generate a list of lists with size elements
        nested_list = [[random.randint(1, 100) for _ in range(10)] for _ in range(size // 10)]

        # Measure the time for flatten_list_loop
        start_time = time.time()
        flatten_list_loop(nested_list)
        loop_time = time.time() - start_time

        # Measure the time for flatten_list_comprehension
        start_time = time.time()
        flatten_list_comprehension(nested_list)
        comprehension_time = time.time() - start_time

        results.append((size, loop_time, comprehension_time))
        print(f"Size: {size}, Loop Time: {loop_time:.6f}s, Comprehension Time: {comprehension_time:.6f}s")

    return results

def char_count_experiment():
    sizes = [10**3, 10**4, 10**5, 10**6, 10**7]
    results = []

    for size in sizes:
        # Generate a string of size length with lowercase letters and spaces
        s = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=size))

        # Measure the time for char_count_loop
        start_time = time.time()
        char_count_loop(s)
        loop_time = time.time() - start_time

        # Measure the time for char_count_comprehension
        start_time = time.time()
        char_count_comprehension(s)
        comprehension_time = time.time() - start_time

        results.append((size, loop_time, comprehension_time))
        print(f"Size: {size}, Loop Time: {loop_time:.6f}s, Comprehension Time: {comprehension_time:.6f}s")

    return results

results = flattening_experiment()
print(results)

results = char_count_experiment()
print(results)      