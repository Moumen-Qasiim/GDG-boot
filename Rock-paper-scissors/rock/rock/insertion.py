def insertion_sort(arr):
    print("Initial array:", arr)
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        print(f"\n>>> i = {i}, key = {key}")
        # Move elements of arr[0..i-1], that are greater than key,
        # to one position ahead of their current position
        while j >= 0 and arr[j] > key:
            print(f"    j = {j}, arr[{j}] = {arr[j]} > {key} -> shifting")
            arr[j + 1] = arr[j]
            j -= 1
            print("    Current array:", arr)
        arr[j + 1] = key
        print(f"Inserted {key} at position {j + 1}")
        print("    Array now:", arr)

# Example usage
arr = [9, 7, 6, 5, 6]
insertion_sort(arr)
