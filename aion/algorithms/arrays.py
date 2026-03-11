"""
Array and sequence processing utilities.

Flattening, chunking, windowing, deduplication, rolling sums, and related
helpers for list/sequence operations. Full package documentation:
aion/algorithms/README.md
"""
from typing import Any, Union

def flatten_array(arr: list[list[Any]]) -> list[Any]:
    """
    Flatten a list of lists into a single list.

    This function takes a nested list (a list containing sublists) and
    returns a new list that contains all elements from the sublists
    in their original order.

    Parameters
    ----------
    arr : List[List[T]]
        A list of lists containing elements of any type.

    Returns
    -------
    List[T]
        A single flattened list containing all elements from the input
        sublists.

    Notes
    -----
    - The function preserves the order of elements.
    - The time complexity is O(n), where n is the total number of elements
      across all sublists.
    - The input list is not modified.
    """

    # Initialize an empty list to store the flattened result
    flat_array = list()

    # Validate input: list must not be empty
    if not arr:
        raise ValueError("List must not be empty")

    # Flatten via direct iteration (keeps order)
    for sublist in arr:
        flat_array.extend(sublist)

    # return the flatted list
    return flat_array


def chunk_array(arr: list[Any], size: int) -> list[list[Any]]:
    """
    Split a list into smaller sublists (chunks) of a given size.

    The function divides the input list into consecutive chunks.
    The last chunk may be smaller if there are not enough elements.

    Parameters
    ----------
    arr : list[Any]
        The input list to be split into chunks.
    size : int
        The size of each chunk. Must be greater than zero.

    Returns
    -------
    list[list[Any]]
        A list of sublists (chunks).

    Raises
    ------
    ValueError
        If the input list is empty or if size is less than or equal to zero.
    """

    # List that will store the chunked result
    chunked_list = list()

    # Validate input list
    if not arr:
        raise ValueError("List must not be empty")

    # Validate chunk size
    if size <= 0:
        raise ValueError("Size must be greater than 0")

    # Create chunks using slicing (O(n))
    for start in range(0, len(arr), size):
        chunked_list.append(arr[start:start + size])

    return chunked_list


def remove_duplicates(arr: list[Any]) -> list[Any]:
    """
    Remove duplicate elements from a list while preserving order.

    The function returns a new list containing only the first
    occurrence of each element.

    Parameters
    ----------
    arr : list[Any]
        The input list that may contain duplicate values.

    Returns
    -------
    list[Any]
        A list without duplicate elements.

    Raises
    ------
    ValueError
        If the input list is empty.
    """

    # List that will store unique elements
    without_dupl = []

    # Validate input list
    if not arr:
        raise ValueError("List must not be empty")

    # Add only elements that were not seen before (O(n))
    seen = set()
    for item in arr:
        if item not in seen:
            without_dupl.append(item)
            seen.add(item)

    return without_dupl


def moving_avarage(arr: list[Union[int, float]]) -> float:
    """
    Calculate the arithmetic mean of a list of numbers.

    The function computes the average value of all numeric
    elements in the list.

    Parameters
    ----------
    arr : list[Union[int, float]]
        A list containing integer or floating-point numbers.

    Returns
    -------
    float
        The average value of the numbers in the list.

    Raises
    ------
    ValueError
        If the input list is empty.
    TypeError
        If the list contains non-numeric elements.
    """

    # Accumulator for the sum of elements
    total = 0.0

    # Validate input list
    if not arr:
        raise ValueError("List must not be empty")

    # Validate types and calculate sum
    for item in arr:
        if type(item) not in (int, float):
            raise TypeError("List must contain only int or float")
        total += item

    return total / len(arr)

def flatten_deep(arr: list[list[Any]]) -> list[Any]:
    """
    Recursively flatten a deeply nested list into a single list.

    This function takes a list that may contain other lists nested
    at any depth and returns a new flat list containing all elements
    in their original order.

    Parameters
    ----------
    arr : list[list[Any]]
        A list that may contain elements or other nested lists.

    Returns
    -------
    list[Any]
        A completely flattened list containing all values from the input.

    Raises
    ------
    ValueError
        If the input list is empty.
    """

    # Initialize an empty list to store the flattened result
    flat_array = list()

    # Validate input: list must not be empty
    if not arr:
        raise ValueError("List must not be empty")

    # Iterate over each element in the input list
    for item in arr:

        # If the current element is a list, flatten it recursively
        if isinstance(item, list):
            flat_array.extend(flatten_deep(item))

        # If the current element is not a list, append it directly
        else:
            flat_array.append(item)

    # Return the fully flattened list
    return flat_array


def sliding_window(arr: list[Any], size: int):
    """
    Generate consecutive sublists (windows) of a specified size.

    This function produces overlapping slices of the input list,
    each containing exactly `size` elements.

    Example
    -------
    arr = [1, 2, 3, 4], size = 2
    Output: [1,2], [2,3], [3,4]

    Parameters
    ----------
    arr : list[Any]
        The input list to create windows from.
    size : int
        The size of each sliding window.

    Yields
    ------
    list[Any]
        Each window (sublist) of length `size`.

    Raises
    ------
    ValueError
        If the list is empty or size is invalid.
    TypeError
        If size is not an integer.
    """

    # Validate input list: must not be empty
    if not arr:
        raise ValueError("List must not be empty")

    # Validate that size is an integer
    if not isinstance(size, int):
        raise TypeError("size must be an integer")

    # Validate that size is greater than zero
    if size <= 0:
        raise ValueError("Size must be greater than 0")

    # Validate that window size does not exceed list length
    if size > len(arr):
        raise ValueError("Size cannot be greater than array length")

    # Generate each sliding window slice
    for i in range(len(arr) - size + 1):
        yield arr[i:i+size]


def pad_array(arr: list[Any], min_len: int, item: Any) -> list[Any]:
    """
    Extend a list until it reaches a minimum required length.

    This function appends the given `item` repeatedly to the list
    until its length becomes equal to `min_len`.

    Parameters
    ----------
    arr : list[Any]
        The input list to be padded.
    min_len : int
        The minimum desired length of the list.
    item : Any
        The element to append until the list reaches the target length.

    Returns
    -------
    list[Any]
        The padded list.

    Raises
    ------
    ValueError
        If the input list is empty.
    TypeError
        If min_len is not an integer.
    """

    # Validate input list: must not be empty
    if not arr:
        raise ValueError("List must not be empty")

    # Validate that min_len is an integer
    if not isinstance(min_len, int):
        raise TypeError("Size must be an integer")

    # Append the padding item until the list reaches the required length
    while len(arr) != min_len:
        arr.append(item)

    # Return the padded list
    return arr


def rolling_sum(arr: list[int], size: int) -> list[int]:
    """
    Compute rolling (moving) sums over a list of integers.

    This function calculates the sum of each consecutive window
    of length `size` in the list.

    Example
    -------
    arr = [1, 2, 3, 4], size = 2
    Output: [3, 5, 7]

    Parameters
    ----------
    arr : list[int]
        The input list of integers.
    size : int
        The size of each window used for summation.

    Returns
    -------
    list[int]
        A list of rolling sums.

    Raises
    ------
    ValueError
        If the list is empty or size is invalid.
    """

    # Validate input list: must not be empty
    if not arr:
        raise ValueError("List must not be empty")

    # Validate window size: must be greater than zero
    if size <= 0:
        raise ValueError("Size must be greater than 0")

    # List that will store all rolling sums
    roll_sum = list()

    # Iterate through all valid window starting positions
    for i in range(len(arr) - size + 1):

        # Compute sum of the current window and append it
        roll_sum.append(sum(arr[i:i+size]))

    # Return the list of rolling sums
    return roll_sum


def pairwise(arr: list[Any]) -> list[Any]:
    """
    Create pairs of consecutive elements in a list.

    This function returns a list of tuples where each tuple contains
    two neighboring elements from the input list.

    Example
    -------
    arr = [1, 2, 3, 4]
    Output: [(1,2), (2,3), (3,4)]

    Parameters
    ----------
    arr : list[Any]
        The input list containing at least two elements.

    Returns
    -------
    list[Any]
        A list of tuples representing consecutive pairs.

    Raises
    ------
    ValueError
        If the list contains fewer than two elements.
    """

    # Validate input, must contain at least two elements
    if len(arr) < 2:
        raise ValueError("List must contain at least 2 elements")

    # List that will store consecutive pairs
    pair = list()

    # Use zip to pair each element with the next one
    pair.extend(zip(arr, arr[1:]))

    # Return the list of pairs

    return pair


def fun():
    pass