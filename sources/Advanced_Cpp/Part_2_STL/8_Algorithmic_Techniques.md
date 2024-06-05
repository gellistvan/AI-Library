
\newpage
## Chapter 8: Algorithmic Techniques

Algorithmic techniques are at the heart of efficient and effective C++ programming. The Standard Template Library (STL) provides a rich set of algorithms that can be applied to various data structures to solve complex problems with minimal effort. Understanding these algorithms and how to apply them is essential for writing optimized and maintainable code.

In this chapter, we will explore a range of algorithmic techniques provided by the STL, diving deep into their usage, benefits, and trade-offs. We will start with **In-Place and Out-of-Place Algorithms**, examining the differences between algorithms that modify their input directly and those that create new copies of the input. Next, we will delve into **Non-Modifying and Modifying Algorithms**, exploring how to use algorithms that either preserve the original data or change it.

Following that, we will cover **Sorted Range Algorithms**, which are specialized algorithms designed to work efficiently on sorted data. These algorithms leverage the sorted property to perform operations faster than general-purpose algorithms. Finally, we will investigate **Partitioning and Permutation Algorithms**, which provide powerful tools for reorganizing data based on specific criteria or generating permutations.

By mastering these algorithmic techniques, you will enhance your ability to write high-performance C++ code, making your programs more efficient and effective in solving real-world problems.

### 8.1. In-Place and Out-of-Place Algorithms

In the realm of algorithmic techniques, understanding the distinction between in-place and out-of-place algorithms is crucial. These concepts define how an algorithm manages memory and modifies data during its execution. In this subchapter, we will explore the characteristics, advantages, and trade-offs of in-place and out-of-place algorithms, along with practical examples in C++.

#### In-Place Algorithms

In-place algorithms modify the input data directly without requiring significant additional memory. These algorithms are memory-efficient because they typically use a constant amount of extra space, irrespective of the input size. However, the original data is altered, which might not always be desirable.

##### Characteristics of In-Place Algorithms:
- Modify the input data directly.
- Use constant or minimal extra space.
- Often more space-efficient but can be complex to implement.
- May not preserve the original data.

##### Example: In-Place Sorting with `std::sort`

`std::sort` is a classic example of an in-place algorithm. It rearranges the elements within the container without using additional memory proportional to the input size.

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {4, 2, 5, 1, 3};

    std::sort(vec.begin(), vec.end());

    std::cout << "Sorted vector: ";
    for (const auto& val : vec) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

##### Example: In-Place Reversal with `std::reverse`

`std::reverse` is another example of an in-place algorithm that reverses the order of elements within the container.

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    std::reverse(vec.begin(), vec.end());

    std::cout << "Reversed vector: ";
    for (const auto& val : vec) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

#### Out-of-Place Algorithms

Out-of-place algorithms, in contrast, create a new copy of the data structure and perform operations on it. These algorithms preserve the original data but typically require additional memory proportional to the input size.

##### Characteristics of Out-Of-Place Algorithms:
- Create a new copy of the data structure.
- Use additional memory proportional to the input size.
- Preserve the original data.
- Often simpler to implement but less space-efficient.

##### Example: Out-Of-Place Copying with `std::copy`

`std::copy` is an example of an out-of-place algorithm. It copies elements from one range to another.

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> source = {1, 2, 3, 4, 5};
    std::vector<int> destination(source.size());

    std::copy(source.begin(), source.end(), destination.begin());

    std::cout << "Source vector: ";
    for (const auto& val : source) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    std::cout << "Destination vector: ";
    for (const auto& val : destination) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

##### Example: Out-Of-Place Transform with `std::transform`

`std::transform` is another example of an out-of-place algorithm. It applies a function to each element in the source range and writes the result to a new range.

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> source = {1, 2, 3, 4, 5};
    std::vector<int> destination(source.size());

    std::transform(source.begin(), source.end(), destination.begin(), [](int x) { return x * x; });

    std::cout << "Source vector: ";
    for (const auto& val : source) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    std::cout << "Transformed vector: ";
    for (const auto& val : destination) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

#### Trade-Offs Between In-Place and Out-Of-Place Algorithms

Choosing between in-place and out-of-place algorithms involves considering several trade-offs:

1. **Memory Usage**:
   - In-place algorithms are generally more memory-efficient because they use minimal extra space.
   - Out-of-place algorithms require additional memory to store the new data structure.

2. **Data Preservation**:
   - In-place algorithms modify the original data, which may not be acceptable in scenarios where the original data needs to be retained.
   - Out-of-place algorithms preserve the original data, making them suitable for such scenarios.

3. **Complexity**:
   - In-place algorithms can be more complex to implement due to the need to carefully manage data within the existing memory space.
   - Out-of-place algorithms are often simpler to implement since they work with a separate copy of the data.

4. **Performance**:
   - In-place algorithms can be faster because they do not involve the overhead of allocating and copying additional memory.
   - Out-of-place algorithms may incur additional performance costs due to memory allocation and copying operations.

#### Combining In-Place and Out-Of-Place Algorithms

In some cases, a combination of in-place and out-of-place techniques can be used to achieve a balance between memory efficiency and data preservation.

##### Example: Combining Techniques for Partial Processing

Consider a scenario where you need to process a large dataset but want to preserve the original data. You can use out-of-place algorithms to process and store intermediate results, then use in-place algorithms for final adjustments.

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> source = {1, 2, 3, 4, 5};
    std::vector<int> intermediate(source.size());

    // Out-of-place transformation
    std::transform(source.begin(), source.end(), intermediate.begin(), [](int x) { return x * x; });

    // In-place modification
    std::for_each(intermediate.begin(), intermediate.end(), [](int& x) { x += 10; });

    std::cout << "Source vector: ";
    for (const auto& val : source) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    std::cout << "Processed vector: ";
    for (const auto& val : intermediate) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

#### Practical Considerations

When deciding between in-place and out-of-place algorithms, consider the following practical factors:

1. **Resource Constraints**: If memory is a critical resource, in-place algorithms are preferable. Conversely, if memory usage is less of a concern, out-of-place algorithms may offer simpler and more readable solutions.
2. **Concurrency**: In multi-threaded environments, in-place algorithms may introduce complexity due to shared data. Out-of-place algorithms, which operate on separate data copies, can simplify concurrency control.
3. **Algorithm Stability**: Some in-place algorithms may not be stable (i.e., they may not preserve the relative order of equivalent elements). If stability is required, an out-of-place stable algorithm might be necessary.

#### Conclusion

Understanding in-place and out-of-place algorithms is fundamental to effective C++ programming. Each approach offers distinct advantages and trade-offs in terms of memory usage, data preservation, complexity, and performance. By mastering these concepts and learning to choose the appropriate technique for a given problem, you can write more efficient and robust C++ code. Whether optimizing for memory efficiency with in-place algorithms or preserving data integrity with out-of-place algorithms, these techniques empower you to tackle a wide range of computational challenges.

### 8.2. Non-Modifying and Modifying Algorithms

In C++ programming, the Standard Template Library (STL) provides a comprehensive set of algorithms to manipulate collections of data. These algorithms can be broadly classified into two categories: non-modifying and modifying algorithms. Understanding these categories is essential for effectively using the STL to perform a wide range of operations on data structures.

#### Non-Modifying Algorithms

Non-modifying algorithms do not alter the elements of the containers they operate on. They are used primarily for examining and querying the contents of a container. These algorithms typically return information about the elements or perform actions without changing the underlying data.

##### Common Non-Modifying Algorithms

1. **`std::for_each`**: Applies a function to each element in a range.
2. **`std::find`**: Searches for an element equal to a given value.
3. **`std::count`**: Counts the number of elements equal to a given value.
4. **`std::all_of`**, **`std::any_of`**, **`std::none_of`**: Check if all, any, or none of the elements in a range satisfy a given predicate.

##### Example: Using `std::for_each`

The `std::for_each` algorithm applies a function to each element in a range.

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

void print(int n) {
    std::cout << n << ' ';
}

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    std::cout << "Elements in vector: ";
    std::for_each(vec.begin(), vec.end(), print);
    std::cout << std::endl;

    return 0;
}
```

##### Example: Using `std::find`

The `std::find` algorithm searches for an element equal to a given value.

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    auto it = std::find(vec.begin(), vec.end(), 3);
    if (it != vec.end()) {
        std::cout << "Element found: " << *it << std::endl;
    } else {
        std::cout << "Element not found" << std::endl;
    }

    return 0;
}
```

##### Example: Using `std::count`

The `std::count` algorithm counts the number of elements equal to a given value.

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {1, 2, 2, 3, 3, 3, 4, 5};

    int count = std::count(vec.begin(), vec.end(), 3);
    std::cout << "Number of 3s: " << count << std::endl;

    return 0;
}
```

#### Modifying Algorithms

Modifying algorithms change the elements of the containers they operate on. They are used to transform, rearrange, or replace elements within the container. These algorithms typically require mutable access to the elements.

##### Common Modifying Algorithms

1. **`std::copy`**: Copies elements from one range to another.
2. **`std::transform`**: Applies a function to each element in a range and stores the result in another range.
3. **`std::replace`**: Replaces elements equal to a given value with another value.
4. **`std::fill`**: Fills a range with a specified value.
5. **`std::remove`**, **`std::remove_if`**: Removes elements that satisfy a given condition (note: these algorithms do not change the container size).

##### Example: Using `std::copy`

The `std::copy` algorithm copies elements from one range to another.

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> source = {1, 2, 3, 4, 5};
    std::vector<int> destination(source.size());

    std::copy(source.begin(), source.end(), destination.begin());

    std::cout << "Destination vector: ";
    for (const auto& val : destination) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

##### Example: Using `std::transform`

The `std::transform` algorithm applies a function to each element in a range and stores the result in another range.

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> source = {1, 2, 3, 4, 5};
    std::vector<int> destination(source.size());

    std::transform(source.begin(), source.end(), destination.begin(), [](int x) { return x * x; });

    std::cout << "Transformed vector: ";
    for (const auto& val : destination) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

##### Example: Using `std::replace`

The `std::replace` algorithm replaces elements equal to a given value with another value.

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 2, 5};

    std::replace(vec.begin(), vec.end(), 2, 9);

    std::cout << "Replaced vector: ";
    for (const auto& val : vec) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

##### Example: Using `std::fill`

The `std::fill` algorithm fills a range with a specified value.

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec(5);

    std::fill(vec.begin(), vec.end(), 7);

    std::cout << "Filled vector: ";
    for (const auto& val : vec) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

##### Example: Using `std::remove` and `std::remove_if`

The `std::remove` and `std::remove_if` algorithms remove elements that satisfy a given condition. Note that these algorithms do not change the container size; they return an iterator to the new end of the range.

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 2, 5};

    // Remove all instances of 2
    auto new_end = std::remove(vec.begin(), vec.end(), 2);
    vec.erase(new_end, vec.end());

    std::cout << "Vector after remove: ";
    for (const auto& val : vec) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    // Remove all even numbers
    vec = {1, 2, 3, 4, 2, 5};
    new_end = std::remove_if(vec.begin(), vec.end(), [](int x) { return x % 2 == 0; });
    vec.erase(new_end, vec.end());

    std::cout << "Vector after remove_if: ";
    for (const auto& val : vec) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

#### Combining Non-Modifying and Modifying Algorithms

In practice, non-modifying and modifying algorithms are often used together to achieve complex data manipulations. For example, you might use a non-modifying algorithm to find elements that meet certain criteria and then use a modifying algorithm to transform or remove those elements.

##### Example: Finding and Replacing Elements

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5, 2};

    // Find the first occurrence of 2
    auto it = std::find(vec.begin(), vec.end(), 2);
    if (it != vec.end()) {
        // Replace it with 9
        *it = 9;
    }

    std::cout << "Vector after find and replace: ";
    for (const auto& val : vec) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

##### Example: Counting and Removing Elements

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 2, 5, 2};

    // Count the number of 2s
    int count = std::count(vec.begin(), vec.end(), 2);
    std::cout << "Number of 2s: " << count << std::endl;

    // Remove all instances of 2
    auto new_end = std::remove(vec.begin(), vec.end(), 2);
    vec.erase(new_end, vec.end());

    std::cout << "Vector after remove: ";
    for (const auto& val : vec) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

#### Practical Considerations

When choosing between non-modifying and modifying algorithms, consider the following factors:

1. **Intent**: Use non-modifying algorithms when you need to query or examine data without changing it. Use modifying algorithms when you need to transform or rearrange data.
2. **Efficiency**: Non-modifying algorithms often have lower overhead because they do not alter data. Modifying algorithms may involve additional steps, such as memory allocation or data movement.
3. **Complexity**: Combining non-modifying and modifying algorithms can sometimes simplify complex operations by breaking them into manageable steps.

#### Conclusion

Non-modifying and modifying algorithms are fundamental tools in the C++ STL, each serving distinct purposes. Non-modifying algorithms are used for querying and examining data without altering it, while modifying algorithms transform, rearrange, or replace data within containers. By mastering these algorithms and understanding how to combine them effectively, you can write efficient, readable, and powerful C++ code to handle a wide range of data manipulation tasks.

### 8.3. Sorted Range Algorithms

Sorted range algorithms in C++ are specialized algorithms designed to work efficiently on ranges of data that are already sorted. By leveraging the sorted property of the data, these algorithms can achieve better performance compared to their general counterparts. In this subchapter, we will explore various sorted range algorithms, their use cases, and practical examples to demonstrate their advantages and proper usage.

#### Importance of Sorted Ranges

The key advantage of sorted ranges is that they allow algorithms to perform operations more efficiently. For instance, searching, merging, and set operations can be done faster on sorted data because the algorithms can take advantage of the ordering to reduce the number of comparisons and data movements.

#### Common Sorted Range Algorithms

1. **`std::binary_search`**
2. **`std::lower_bound`**
3. **`std::upper_bound`**
4. **`std::equal_range`**
5. **`std::merge`**
6. **`std::includes`**
7. **Set operations: `std::set_union`, `std::set_intersection`, `std::set_difference`, `std::set_symmetric_difference`**

#### 1. `std::binary_search`

`std::binary_search` checks if an element exists in a sorted range. It returns `true` if the element is found, and `false` otherwise. The algorithm performs the search in logarithmic time, O(log n), by repeatedly dividing the range in half.

##### Example: Using `std::binary_search`

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    
    int key = 3;
    bool found = std::binary_search(vec.begin(), vec.end(), key);
    
    if (found) {
        std::cout << key << " is in the vector." << std::endl;
    } else {
        std::cout << key << " is not in the vector." << std::endl;
    }

    return 0;
}
```

#### 2. `std::lower_bound`

`std::lower_bound` finds the first position in a sorted range where a given value can be inserted without violating the order. It returns an iterator to the first element that is not less than (i.e., greater than or equal to) the value.

##### Example: Using `std::lower_bound`

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {1, 2, 2, 3, 4, 5};
    
    int key = 2;
    auto it = std::lower_bound(vec.begin(), vec.end(), key);
    
    if (it != vec.end()) {
        std::cout << "Lower bound of " << key << " is at position: " << std::distance(vec.begin(), it) << std::endl;
    } else {
        std::cout << key << " is greater than all elements." << std::endl;
    }

    return 0;
}
```

#### 3. `std::upper_bound`

`std::upper_bound` finds the first position in a sorted range where a given value can be inserted without violating the order. It returns an iterator to the first element that is greater than the value.

##### Example: Using `std::upper_bound`

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {1, 2, 2, 3, 4, 5};
    
    int key = 2;
    auto it = std::upper_bound(vec.begin(), vec.end(), key);
    
    if (it != vec.end()) {
        std::cout << "Upper bound of " << key << " is at position: " << std::distance(vec.begin(), it) << std::endl;
    } else {
        std::cout << key << " is greater than all elements." << std::endl;
    }

    return 0;
}
```

#### 4. `std::equal_range`

`std::equal_range` returns a pair of iterators that denote the range of elements equal to a given value. This range is defined as the range `[lower_bound, upper_bound)`.

##### Example: Using `std::equal_range`

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {1, 2, 2, 3, 4, 5};
    
    int key = 2;
    auto range = std::equal_range(vec.begin(), vec.end(), key);
    
    std::cout << "Equal range of " << key << " is from position: " 
              << std::distance(vec.begin(), range.first) << " to position: " 
              << std::distance(vec.begin(), range.second) << std::endl;

    return 0;
}
```

#### 5. `std::merge`

`std::merge` combines two sorted ranges into a single sorted range. It copies the elements from both input ranges into an output range, maintaining the sorted order.

##### Example: Using `std::merge`

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec1 = {1, 3, 5};
    std::vector<int> vec2 = {2, 4, 6};
    std::vector<int> result(vec1.size() + vec2.size());
    
    std::merge(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), result.begin());
    
    std::cout << "Merged vector: ";
    for (const auto& val : result) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

#### 6. `std::includes`

`std::includes` checks if one sorted range contains all elements of another sorted range. It returns `true` if all elements of the second range are present in the first range, and `false` otherwise.

##### Example: Using `std::includes`

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec1 = {1, 2, 3, 4, 5};
    std::vector<int> vec2 = {2, 4};
    
    bool result = std::includes(vec1.begin(), vec1.end(), vec2.begin(), vec2.end());
    
    if (result) {
        std::cout << "vec1 includes all elements of vec2." << std::endl;
    } else {
        std::cout << "vec1 does not include all elements of vec2." << std::endl;
    }

    return 0;
}
```

#### 7. Set Operations: `std::set_union`, `std::set_intersection`, `std::set_difference`, `std::set_symmetric_difference`

These algorithms perform set operations on sorted ranges, treating the ranges as mathematical sets.

##### Example: Using `std::set_union`

`std::set_union` computes the union of two sorted ranges and stores the result in an output range.

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec1 = {1, 2, 3};
    std::vector<int> vec2 = {3, 4, 5};
    std::vector<int> result;
    
    std::set_union(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), std::back_inserter(result));
    
    std::cout << "Union of vec1 and vec2: ";
    for (const auto& val : result) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

##### Example: Using `std::set_intersection`

`std::set_intersection` computes the intersection of two sorted ranges and stores the result in an output range.

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec1 = {1, 2, 3};
    std::vector<int> vec2 = {2, 3, 4};
    std::vector<int> result;
    
    std::set_intersection(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), std::back_inserter(result));
    
    std::cout << "Intersection of vec1 and vec2: ";
    for (const auto& val : result) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

##### Example: Using `std::set_difference`

`std::set_difference` computes the difference of two sorted ranges and stores the result in an output range.

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec1 = {1, 2, 3, 4};
    std::vector<int> vec2 = {2, 4};
    std::vector<int> result;
    
    std::set_difference(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), std::back_inserter(result));
    
    std::cout << "Difference of vec1 and vec2: ";
    for (const auto& val : result) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```



##### Example: Using `std::set_symmetric_difference`

`std::set_symmetric_difference` computes the symmetric difference of two sorted ranges and stores the result in an output range.

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec1 = {1, 2, 3};
    std::vector<int> vec2 = {3, 4, 5};
    std::vector<int> result;
    
    std::set_symmetric_difference(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), std::back_inserter(result));
    
    std::cout << "Symmetric difference of vec1 and vec2: ";
    for (const auto& val : result) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

#### Practical Considerations

When using sorted range algorithms, consider the following:

1. **Sorting Requirement**: Ensure that the input ranges are sorted according to the same criteria before applying sorted range algorithms. If the input is not sorted, the results may be incorrect.
2. **Efficiency**: Sorted range algorithms often provide better performance than their unsorted counterparts. For example, `std::binary_search` performs searches in O(log n) time compared to O(n) for linear searches.
3. **Stability**: Some algorithms, such as `std::set_union`, are stable and preserve the relative order of equivalent elements from the input ranges.

#### Conclusion

Sorted range algorithms are powerful tools in the C++ STL that leverage the sorted property of data to achieve efficient performance. Understanding and utilizing these algorithms—such as `std::binary_search`, `std::lower_bound`, `std::upper_bound`, `std::equal_range`, `std::merge`, `std::includes`, and the set operations—allows you to perform complex operations on sorted data with ease. By mastering these algorithms, you can write more efficient and effective C++ programs that handle a wide range of data processing tasks.

### 8.4. Partitioning and Permutation Algorithms

Partitioning and permutation algorithms are powerful tools in the C++ Standard Template Library (STL) that allow you to reorganize elements within a range based on specific criteria. These algorithms can be used to divide data into subsets, rearrange elements to meet certain conditions, or generate all possible permutations of a sequence. Understanding these algorithms enhances your ability to manipulate and analyze data efficiently.

#### Partitioning Algorithms

Partitioning algorithms rearrange elements in a range based on a predicate, dividing the range into two parts: those that satisfy the predicate and those that do not. The main partitioning algorithms are `std::partition`, `std::stable_partition`, and `std::partition_point`.

##### 1. `std::partition`

`std::partition` reorders the elements in a range such that all elements satisfying a given predicate appear before those that do not. The relative order of the elements is not preserved.

###### Example: Using `std::partition`

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

bool is_even(int n) {
    return n % 2 == 0;
}

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    std::partition(vec.begin(), vec.end(), is_even);

    std::cout << "Partitioned vector: ";
    for (const auto& val : vec) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

##### 2. `std::stable_partition`

`std::stable_partition` is similar to `std::partition`, but it preserves the relative order of the elements within each partitioned subset.

###### Example: Using `std::stable_partition`

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

bool is_even(int n) {
    return n % 2 == 0;
}

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    std::stable_partition(vec.begin(), vec.end(), is_even);

    std::cout << "Stable partitioned vector: ";
    for (const auto& val : vec) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

##### 3. `std::partition_point`

`std::partition_point` returns an iterator to the first element in the partitioned range that does not satisfy the predicate. It assumes that the range is already partitioned according to the predicate.

###### Example: Using `std::partition_point`

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

bool is_even(int n) {
    return n % 2 == 0;
}

int main() {
    std::vector<int> vec = {2, 4, 6, 1, 3, 5};

    auto it = std::partition_point(vec.begin(), vec.end(), is_even);

    std::cout << "Partition point: " << std::distance(vec.begin(), it) << std::endl;

    return 0;
}
```

#### Permutation Algorithms

Permutation algorithms rearrange elements in all possible orders or specific orders. These algorithms include `std::next_permutation`, `std::prev_permutation`, and `std::rotate`.

##### 1. `std::next_permutation`

`std::next_permutation` rearranges the elements in a range to the next lexicographically greater permutation. If the range is already the largest possible permutation, it rearranges the elements to the smallest permutation (sorted order).

###### Example: Using `std::next_permutation`

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {1, 2, 3};

    do {
        for (const auto& val : vec) {
            std::cout << val << ' ';
        }
        std::cout << std::endl;
    } while (std::next_permutation(vec.begin(), vec.end()));

    return 0;
}
```

##### 2. `std::prev_permutation`

`std::prev_permutation` rearranges the elements in a range to the previous lexicographically smaller permutation. If the range is already the smallest possible permutation, it rearranges the elements to the largest permutation.

###### Example: Using `std::prev_permutation`

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {3, 2, 1};

    do {
        for (const auto& val : vec) {
            std::cout << val << ' ';
        }
        std::cout << std::endl;
    } while (std::prev_permutation(vec.begin(), vec.end()));

    return 0;
}
```

##### 3. `std::rotate`

`std::rotate` rotates the elements in a range such that the element pointed to by a given iterator becomes the first element of the new range. The order of the elements is preserved.

###### Example: Using `std::rotate`

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    std::rotate(vec.begin(), vec.begin() + 2, vec.end());

    std::cout << "Rotated vector: ";
    for (const auto& val : vec) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

#### Combining Partitioning and Permutation Algorithms

Partitioning and permutation algorithms can be combined to achieve complex data manipulation tasks. For example, you can partition data based on a predicate and then generate permutations within each subset.

##### Example: Partitioning and Generating Permutations

```cpp
#include <algorithm>

#include <iostream>
#include <vector>

bool is_even(int n) {
    return n % 2 == 0;
}

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5, 6};

    auto partition_point = std::stable_partition(vec.begin(), vec.end(), is_even);

    std::cout << "Even elements permutations:" << std::endl;
    do {
        for (auto it = vec.begin(); it != partition_point; ++it) {
            std::cout << *it << ' ';
        }
        std::cout << std::endl;
    } while (std::next_permutation(vec.begin(), partition_point));

    std::cout << "Odd elements permutations:" << std::endl;
    do {
        for (auto it = partition_point; it != vec.end(); ++it) {
            std::cout << *it << ' ';
        }
        std::cout << std::endl;
    } while (std::next_permutation(partition_point, vec.end()));

    return 0;
}
```

#### Practical Considerations

When using partitioning and permutation algorithms, consider the following:

1. **Efficiency**: Partitioning algorithms generally operate in linear time, O(n), making them efficient for large datasets. Permutation algorithms, however, can have factorial time complexity, O(n!), so they are best used on smaller datasets.
2. **Stability**: If the relative order of elements within each subset is important, use `std::stable_partition` instead of `std::partition`.
3. **Use Cases**: Partitioning is useful for categorizing data, filtering, and preparing data for other operations. Permutation algorithms are valuable in combinatorial problems, generating all possible arrangements, and exploring solution spaces.

#### Conclusion

Partitioning and permutation algorithms are essential tools in the C++ STL for reorganizing data based on specific criteria. Partitioning algorithms such as `std::partition`, `std::stable_partition`, and `std::partition_point` allow you to divide data into subsets efficiently. Permutation algorithms such as `std::next_permutation`, `std::prev_permutation`, and `std::rotate` enable you to generate different arrangements of data. By mastering these algorithms, you can perform complex data manipulations with ease, enhancing your ability to tackle a wide range of programming challenges.
