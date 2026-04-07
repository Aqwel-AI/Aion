"""
Search Algorithms
=================

This module contains classical search algorithms used for locating elements
or traversal states within ordered sequences and discrete structures. The
implementations are designed for research and educational use, emphasizing
correctness, clarity, and predictable behavior.

Scope
-----
The algorithms in this module focus on:
- Searching within sorted sequences
- Boundary search operations (lower and upper bounds)
- Graph and state-space traversal where applicable

Design principles
-----------------
- Deterministic behavior under well-defined input assumptions
- Clear and readable implementations suitable for inspection and study
- Minimal abstraction to preserve algorithmic transparency
- Stable function interfaces intended for direct use in experiments

Intended usage
--------------
These search utilities are provided to reduce boilerplate in research code
while keeping standard search logic explicit and easy to reason about.

See also: aion/algorithms/README.md for full package documentation.
"""

from typing import List, Optional, TypeVar, Union, Tuple

T = TypeVar("T")


def binary_search(arr: List[T], target: T) -> Optional[int]:
    """
    Perform binary search on a sorted sequence.

    This function searches for a target value in a sorted list using
    the binary search algorithm. If the target is found, its index is
    returned. If the target is not present, None is returned.

    Parameters
    ----------
    arr : List[T]
        A list of elements sorted in ascending order.
    target : T
        The value to search for.

    Returns
    -------
    Optional[int]
        The index of the target if found, otherwise None.

    Notes
    -----
    - The input list must be sorted in ascending order.
    - The algorithm runs in O(log n) time complexity.
    """

    # Initialize search boundaries
    left, right = 0, len(arr) - 1

    # Continue searching while the interval is valid
    while left <= right:
        # Compute middle index
        mid = (left + right) // 2

        # Check if the middle element is the target
        if arr[mid] == target:
            return mid

        # If target is greater, ignore the left half
        elif arr[mid] < target:
            left = mid + 1

        # If target is smaller, ignore the right half
        else:
            right = mid - 1

    # Target not found
    return None


def lower_bound(arr: List[T], target: T) -> int:
    """
    Find the first index at which `target` can be inserted in a sorted list.

    This function returns the index of the first element in `arr` that is
    greater than or equal to `target`. If all elements in the list are less
    than `target`, the function returns the length of the list.

    Parameters
    ----------
    arr : List[T]
        A list of elements sorted in ascending order.
    target : T
        The value to locate.

    Returns
    -------
    int
        The index of the first position where `target` can be inserted
        without violating the sort order.

    Notes
    -----
    - The input list must be sorted in ascending order.
    - The algorithm runs in O(log n) time complexity.
    - This function does not check for the presence of `target`; it only
      determines the insertion position.
    """

    # Initialize the search interval [left, right)
    left = 0
    right = len(arr)

    # Continue until the search space is empty
    while left < right:
        # Compute middle index
        mid = (left + right) // 2

        # If middle element is less than target,
        # narrow search to the right half
        if arr[mid] < target:
            left = mid + 1
        else:
            # Otherwise, target belongs to the left half
            right = mid

    # Left is the first index where arr[left] >= target
    return left


def upper_bound(arr: List[T], target: T) -> int:
    """
    Find the first index at which an element greater than `target` appears.

    This function returns the index of the first element in `arr` that is
    strictly greater than `target`. If no such element exists, the function
    returns the length of the list.

    Parameters
    ----------
    arr : List[T]
        A list of elements sorted in ascending order.
    target : T
        The value to locate.

    Returns
    -------
    int
        The index of the first position where an element greater than
        `target` can be found or inserted.

    Notes
    -----
    - The input list must be sorted in ascending order.
    - The algorithm runs in O(log n) time complexity.
    - This function is equivalent to C++ std::upper_bound.
    """

    # Initialize the search interval [left, right)
    left = 0
    right = len(arr)

    # Continue until the search space is empty
    while left < right:
        # Compute middle index
        mid = (left + right) // 2

        # If middle element is less than or equal to target,
        # narrow search to the right half
        if arr[mid] <= target:
            left = mid + 1
        else:
            # Otherwise, element greater than target is in the left half
            right = mid

    # Left is the first index where arr[left] > target
    return left


def is_sorted(ulist: List[T]) -> bool:
    """
    Check whether a list is sorted in ascending or descending order.

    The function performs a single pass to verify non-decreasing or
    non-increasing order without allocating sorted copies.

    Parameters
    ----------
    ulist : List[T]
        Input list to check.

    Returns
    -------
    bool
        True if the list is sorted ascending or descending, otherwise False.

    Raises
    ------
    TypeError
        If the input is not a list.
    """
    # Validate type
    if not isinstance(ulist, list):
        raise TypeError("Input must be a list")

    # Fast linear checks (O(n)) for ascending or descending order
    if len(ulist) < 2:
        return True

    ascending = True
    descending = True

    for i in range(1, len(ulist)):
        if ulist[i] < ulist[i - 1]:
            ascending = False
        if ulist[i] > ulist[i - 1]:
            descending = False
        if not ascending and not descending:
            return False

    return True
    
def jump_search(slist: List[T], step: int, target: T) -> Optional[int]:
    """
    Search for a target in a sorted list using jump search.

    The list is scanned in fixed-size jumps to find a block that may
    contain the target, then linearly searched within that block.

    Parameters
    ----------
    slist : List[T]
        Sorted list to search.
    step : int
        Jump size. Must be a positive integer.
    target : T
        Value to find in the list.

    Returns
    -------
    Optional[int]
        Index of the target if found, otherwise None.

    Raises
    ------
    TypeError
        If inputs are invalid types.
    ValueError
        If step is not a positive integer.
    """
    if not isinstance(slist, list):
        raise TypeError("Input must be a list")
    if not isinstance(step, int):
        raise TypeError("Step must be an integer")
    if step <= 0:
        raise ValueError("Step must be a positive integer")
    if not slist:
        return None

    # Find the block where target may be present
    n = len(slist)
    prev = 0
    curr = 0
    while curr < n and slist[curr] < target:
        prev = curr
        curr = min(curr + step, n - 1)
        if prev == curr:
            break

    # Linear search within the block
    for idx in range(prev, min(curr + 1, n)):
        if slist[idx] == target:
            return idx
        if slist[idx] > target:
            break

    return None

def find_peak_element(ulist: List[T]) -> List[T]:
    """
    Find all peak elements in a list.

    A peak element is strictly greater than its immediate neighbors.
    Boundary elements are considered peaks if they are greater than
    their single neighbor.

    Parameters
    ----------
    ulist : List[T]
        Input list to scan for peaks.

    Returns
    -------
    List[T]
        List of peak elements.

    Raises
    ------
    TypeError
        If the input is not a list.
    ValueError
        If the list has fewer than 2 elements.
    """
    # Validate input
    if not isinstance(ulist, list):
        raise TypeError("Input must be a list")

    # Require at least 2 elements for peak comparison
    if len(ulist) < 2:
        raise ValueError("List must have at least 2 elements")
    # Collect all peaks
    result = []
    # Special handling for 2 elements
    if len(ulist) == 2:
        if ulist[0] > ulist[1]:
            result.append(ulist[0])
        elif ulist[1] > ulist[0]:
            result.append(ulist[1])
        return result
    # Check boundary elements
    if ulist[0] > ulist[1]:
        result.append(ulist[0])
    if ulist[-1] > ulist[-2]:
        result.append(ulist[-1])
    # Check interior elements
    for i in range(1,len(ulist)-1):
        if ulist[i-1] < ulist[i] > ulist[i+1]:
            result.append(ulist[i])

    return result
            

def exponential_search(slist: List[T], target: T) -> Optional[int]:
    """
    Search for a target in a sorted list using exponential search.

    The function expands the search range exponentially, then performs
    a binary search within that range.

    Parameters
    ----------
    slist : List[T]
        Sorted list to search.
    target : T
        Value to find in the list.

    Returns
    -------
    Optional[int]
        Index of the target if found, otherwise None.

    Raises
    ------
    TypeError
        If inputs are invalid types.
    """
    # Validate input
    if not isinstance(slist, list):
        raise TypeError("Input must be a list")

    # Handle empty list and first-element match
    if len(slist) == 0:
        return None
    if slist[0] == target:
        return 0

    # Find range where target could be
    i = 1
    while i < len(slist) and slist[i] < target:
        i *= 2
    low = i // 2
    high = min(i, len(slist) - 1)

    # Binary search within range
    while low <= high:
        mid = (low + high) // 2
        if slist[mid] == target:
            return mid
        if slist[mid] < target:
            low = mid + 1
        if slist[mid] > target:
            high = mid - 1

    return None


def linear_search(ulist: List[Union[int, float]], target: Union[int, float]) -> str:
    '''
    A basic sequential search that iterates through the list from start to finish. It compares each element with the target until a match is found or the end of the list is reached.
    '''
    if isinstance(ulist,list):
        ulist = list(ulist)
    if not isinstance(ulist, list):
        raise TypeError
    if isinstance(target, int,):
        target = int(target)
    if isinstance(target, float,):
        target = float(target)
    if not isinstance(target, (int, float)):
        raise TypeError
    
    if target not in ulist:
        return f"There isn't a number like {target}"
    i = 0
    while (ulist[i] != target):
        i+=1

    return f"The number {target} is at index {i}"

def First_Occurrence(ulist: List[Union[int, float]], target: Union[int, float]) -> str:
    '''
    Scans the list from the beginning (index 0) to find the first instance of the target.
    '''
    if isinstance(ulist,list):
        ulist = list(ulist)
    if not isinstance(ulist, list):
        raise TypeError
    if isinstance(target, int,):
        target = int(target)
    if isinstance(target, float,):
        target = float(target)
    if not isinstance(target, (int, float)):
        raise TypeError
    
    if target not in ulist:
        return f"There isn't a number like {target}"
    i = 0
    while (ulist[i] != target):
        i+=1

    return f"First occurrence of the number {target} is at index {i}"
    
def Last_Occurrence(ulist: List[Union[int, float]], target: Union[int, float]) -> str:
    '''
    Scans the list from the end (index n-1) backwards to find the final instance of the target.
    '''
    if isinstance(ulist,list):
        ulist = list(ulist)
    if not isinstance(ulist, list):
        raise TypeError
    if isinstance(target, int,):
        target = int(target)
    if isinstance(target, float,):
        target = float(target)
    if not isinstance(target, (int, float)):
        raise TypeError
    
    if target not in ulist:
        return f"There isn't a number like {target}"
    i = len(ulist)-1
    while (ulist[i] != target):
        i-=1

    return f"Last occurrence of the number {target} is at index {i}"

def First_Last_Occurrence(ulist: List[Union[int, float]], target: Union[int, float]) -> Tuple[str, str]:
    '''
    A composite function that returns the results of both searches as a tuple.
    '''
    First = First_Occurrence(ulist,target)
    Last = Last_Occurrence(ulist,target)

    return First,Last

def roatated_search(ulist: List[Union[int, float]], target: Union[int, float]) -> str:
    '''
    An optimized algorithm for sorted lists that have been "rotated" at a pivot point (e.g., [4, 5, 6, 1, 2, 3]). It uses binary search logic to find the pivot point and then maps a standard binary search onto the rotated indices using an offset.
    '''
    if isinstance(ulist,list):
        ulist = list(ulist)
    if not isinstance(ulist, list):
        raise TypeError
    if isinstance(target, int,):
        target = int(target)
    if isinstance(target, float,):
        target = float(target)
    if not isinstance(target, (int, float)):
        raise TypeError
    

    if len(ulist) == 0:
        return "List is empty."
    if ulist[0] == target:
        return f"Number {target} is at index 0."

    low = 0
    high = len(ulist) - 1

    while low < high:
        mid = (low + high) // 2
        if ulist[mid] == target:
            return f"The number {target} is at index {mid}"
        if ulist[mid] > ulist[high]:
            low = mid + 1
        else:
            high = mid

    offset = low
    low = 0
    high = len(ulist) - 1

    while low <= high:
        mid = (low + high) // 2
        mid2 = (mid + offset) % len(ulist)
        
        if ulist[mid2] == target:
            return f"The number {target} is at index {mid2}"
        if ulist[mid2] < target:
            low = mid + 1
        else:
            high = mid - 1
    
    return f"There isn't a number like {target}"


def ternary_search(slist: List[Union[int, float]], target: Union[int, float]) -> str:
    '''
    A divide-and-conquer algorithm for sorted lists. Unlike binary search which splits the list in two, this splits the list into three equal segments using two midpoints. It reduces the search space by two-thirds in each iteration.
    '''
    if isinstance(slist,list):
        slist = list(slist)
    if not isinstance(slist, list):
        raise TypeError
    if isinstance(target, int,):
        target = int(target)
    if isinstance(target, float,):
        target = float(target)
    if not isinstance(target, (int, float)):
        raise TypeError
    
    low = 0
    high = len(slist) - 1


    if len(slist) == 0:
        return "List is empty."
    
    while low <= high:
        mid1 = low + (high - low) // 3
        mid2 = high - (high - low) // 3
        if target == slist[mid1]:
            return f"The number {target} is at index {mid1}"
        if target == slist[mid2]:
            return f"The number {target} is at index {mid2}"
        if target < slist[mid1]:
            high = mid1 - 1
        elif target > slist[mid2]:
            low = mid2 + 1
        else:
            low = mid1 + 1
            high = mid2 - 1

    return f"There isn't a number like {target}"

def interpolation_search(slist: List[Union[int, float]], target: Union[int, float]) -> str:
    '''
    A specialized search for sorted, uniformly distributed numerical data. It "estimates" the position of the target based on the values at the high and low ends of the search range, similar to how a human looks for a word in a dictionary or a name in a phonebook.
    '''
    if isinstance(slist,list):
        slist = list(slist)
    if not isinstance(slist, list):
        raise TypeError
    if isinstance(target, int,):
        target = int(target)
    if isinstance(target, float,):
        target = float(target)
    if not isinstance(target, (int, float)):
        raise TypeError
    if len(slist) == 0:
        return "List is empty."
    
    low = 0
    high = len(slist)-1
    while low <= high and target >= slist[low] and target <= slist[high]:
        if slist[high] == slist[low]:
            if slist[low] == target:
                return f"The number {target} is at index {low}"
            break

        formula = (low + (high - low) * ((target - slist[low])) // (slist[high] - slist[low]))
        if slist[formula] == target:
            return f"The number {target} is at index {formula}"
        if slist[formula] < target:
            low = formula + 1
        else:
            high = formula - 1

    return f"There isn't a number like {target}"
