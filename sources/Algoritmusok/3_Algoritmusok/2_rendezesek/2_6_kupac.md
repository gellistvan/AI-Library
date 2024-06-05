\newpage

## 2.6. Heap Sort

A Heap Sort egy hatékony és megbízható rendezési algoritmus, amely a heap adatszerkezeten alapul. Az algoritmus két fő lépésből áll: először egy maximális vagy minimális heapet építünk a bemeneti adatokból, majd a heapből fokozatosan kivonjuk az elemeket, és rendezzük őket. Ez a rendezési módszer garantáltan $O(n \log n)$ időkomplexitással működik, így különösen hasznos nagy adathalmazok esetén. Ebben a fejezetben megismerkedünk a heap sort alapelveivel és implementációjával, megvizsgáljuk a heapek létrehozását és a heapify műveletek részleteit, elemezzük az algoritmus teljesítményét és komplexitását, valamint bemutatjuk a gyakorlati alkalmazásokat és konkrét példákat.

### 2.6.1. Alapelvek és implementáció

Heap Sort is a highly efficient comparison-based sorting algorithm that utilizes the properties of the heap data structure. In this section, we will delve into the fundamental principles of Heap Sort and discuss its implementation in detail.

#### Principles of Heap Sort

Heap Sort operates based on the concept of a binary heap, which is a complete binary tree where each node satisfies the heap property. There are two types of heaps: max-heaps and min-heaps.

- **Max-Heap**: In a max-heap, the key at the root must be greater than or equal to the keys of its children, and the same property must be recursively true for all nodes in the tree.
- **Min-Heap**: In a min-heap, the key at the root must be less than or equal to the keys of its children, and similarly, this property must be true for all nodes in the tree.

Heap Sort specifically uses a max-heap to sort an array in ascending order. The algorithm can be broken down into two main phases:

1. **Building the Heap**: Convert the array into a max-heap.
2. **Extracting Elements from the Heap**: Repeatedly remove the largest element from the heap and reconstruct the heap until all elements are sorted.

#### Building the Heap

Building a heap can be accomplished using a procedure known as `heapify`. The `heapify` function ensures that a subtree with a given node as the root satisfies the max-heap property. If a node violates the heap property, `heapify` will swap the node with its largest child and recursively call itself on the subtree affected by the swap.

The heap construction can be performed in an efficient manner using a bottom-up approach. Starting from the last non-leaf node and moving upwards to the root, we call `heapify` on each node. This process ensures that all subtrees satisfy the heap property by the time we reach the root.

The time complexity of building the heap is $O(n)$, which can be understood by summing the work done by `heapify` across all nodes in the heap.

#### Extracting Elements

Once the array is organized into a max-heap, the next phase is to sort the elements. The largest element, which is at the root of the heap, is swapped with the last element of the array. The heap size is then reduced by one, and the `heapify` function is called on the new root to restore the max-heap property. This process is repeated until all elements are extracted and the array is sorted.

The extraction phase consists of $n$ calls to `heapify`, each of which takes $O(\log n)$ time, leading to an overall time complexity of $O(n \log n)$ for the sorting phase.

#### Implementation in C++

Below is a detailed implementation of the Heap Sort algorithm in C++:

```cpp
#include <iostream>

#include <vector>

// Function to heapify a subtree rooted with node i, which is an index in array.
void heapify(std::vector<int>& arr, int n, int i) {
    int largest = i; // Initialize largest as root
    int left = 2 * i + 1; // left = 2*i + 1
    int right = 2 * i + 2; // right = 2*i + 2

    // If left child is larger than root
    if (left < n && arr[left] > arr[largest])
        largest = left;

    // If right child is larger than largest so far
    if (right < n && arr[right] > arr[largest])
        largest = right;

    // If largest is not root
    if (largest != i) {
        std::swap(arr[i], arr[largest]);

        // Recursively heapify the affected sub-tree
        heapify(arr, n, largest);
    }
}

// Main function to sort an array using Heap Sort
void heapSort(std::vector<int>& arr) {
    int n = arr.size();

    // Build heap (rearrange array)
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    // One by one extract an element from heap
    for (int i = n - 1; i > 0; i--) {
        // Move current root to end
        std::swap(arr[0], arr[i]);

        // Call max heapify on the reduced heap
        heapify(arr, i, 0);
    }
}

// Utility function to print an array
void printArray(const std::vector<int>& arr) {
    for (int i : arr)
        std::cout << i << " ";
    std::cout << std::endl;
}

// Driver program to test the above functions
int main() {
    std::vector<int> arr = {12, 11, 13, 5, 6, 7};
    heapSort(arr);

    std::cout << "Sorted array is: ";
    printArray(arr);

    return 0;
}
```

#### Detailed Explanation of the Code

1. **heapify Function**: This function is responsible for maintaining the max-heap property. It takes an array, the size of the heap (`n`), and the index of the root (`i`). It compares the root with its children and swaps it with the largest if necessary, then recursively calls itself on the affected subtree.

2. **heapSort Function**: This function orchestrates the Heap Sort algorithm. It first builds a max-heap from the input array by calling `heapify` on all non-leaf nodes. Then, it repeatedly extracts the largest element (root of the heap) and moves it to the end of the array, reducing the heap size each time and restoring the heap property with `heapify`.

3. **printArray Function**: This utility function is used to print the contents of the array.

4. **main Function**: The driver program that demonstrates the Heap Sort algorithm. It initializes an array, calls `heapSort` on it, and prints the sorted array.

#### Advantages and Disadvantages

**Advantages**:
- **Efficiency**: Heap Sort has a guaranteed time complexity of $O(n \log n)$, making it suitable for large datasets.
- **In-Place Sorting**: It requires only a constant amount of additional space, $O(1)$.

**Disadvantages**:
- **Not Stable**: Heap Sort is not a stable sorting algorithm, meaning it does not preserve the relative order of equal elements.
- **Cache Performance**: Due to its non-sequential memory access patterns, it may have poor cache performance compared to other algorithms like Quick Sort or Merge Sort.

#### Conclusion

Heap Sort is a robust sorting algorithm with a well-defined theoretical foundation and practical efficiency. It leverages the properties of binary heaps to achieve optimal time complexity and operates in-place with minimal additional space. Despite its drawbacks, it remains a valuable tool in the arsenal of sorting techniques, particularly when stability is not a concern. Understanding Heap Sort not only enriches one's knowledge of sorting algorithms but also deepens the comprehension of heap data structures and their applications.

### 2.6.2. Heaps Létrehozása és Heapify Műveletek

Heap data structures play a crucial role in various algorithmic processes, including Heap Sort, priority queues, and graph algorithms such as Dijkstra's and Prim's algorithms. This section delves into the creation of heaps and the `heapify` operations, which are fundamental to maintaining the heap property.

#### Types of Heaps

As previously mentioned, heaps are specialized binary trees that satisfy the heap property. There are two primary types of heaps:

- **Max-Heap**: Each parent node is greater than or equal to its children. The largest element is at the root.
- **Min-Heap**: Each parent node is less than or equal to its children. The smallest element is at the root.

The focus of this discussion will be on max-heaps, though the principles apply symmetrically to min-heaps.

#### Building a Heap

Building a heap from an unsorted array involves rearranging the array elements to satisfy the heap property. This can be efficiently achieved using the bottom-up heap construction approach, which utilizes the `heapify` function.

The process involves:
1. **Starting from the last non-leaf node**: This node is found at index $\lfloor \frac{n}{2} \rfloor - 1$ for an array of size $n$.
2. **Calling `heapify` on each node**: Move upwards to the root, ensuring that each subtree satisfies the max-heap property.

The `heapify` function is a key component in this process, ensuring that the subtree rooted at a given node maintains the heap property.

#### The `heapify` Function

The `heapify` function takes three parameters:
- The array representing the heap.
- The size of the heap.
- The index of the root node of the subtree to be heapified.

The function operates as follows:
1. **Identify the largest element** among the root and its children.
2. **Swap the root with the largest element** if necessary.
3. **Recursively apply `heapify`** to the affected subtree if a swap was made.

This ensures that the subtree rooted at the given node satisfies the max-heap property.

#### Detailed Implementation of `heapify` in C++

Here's a C++ implementation of the `heapify` function:

```cpp
#include <iostream>

#include <vector>

// Function to heapify a subtree rooted with node i, which is an index in array.
void heapify(std::vector<int>& arr, int n, int i) {
    int largest = i; // Initialize largest as root
    int left = 2 * i + 1; // left = 2*i + 1
    int right = 2 * i + 2; // right = 2*i + 2

    // If left child is larger than root
    if (left < n && arr[left] > arr[largest])
        largest = left;

    // If right child is larger than largest so far
    if (right < n && arr[right] > arr[largest])
        largest = right;

    // If largest is not root
    if (largest != i) {
        std::swap(arr[i], arr[largest]);

        // Recursively heapify the affected sub-tree
        heapify(arr, n, largest);
    }
}
```

This function is the backbone of heap operations, ensuring that any violations of the heap property are corrected.

#### Bottom-Up Heap Construction

The bottom-up approach to heap construction is efficient and straightforward. By starting from the last non-leaf node and moving upwards, each node is heapified, ensuring the entire array satisfies the heap property.

Here’s a C++ function to build a max-heap from an unsorted array:

```cpp
void buildMaxHeap(std::vector<int>& arr) {
    int n = arr.size();

    // Start from the last non-leaf node and move upwards
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
    }
}
```

This function calls `heapify` on each non-leaf node, ensuring the entire array is transformed into a max-heap.

#### Analysis of Heap Construction

The time complexity of the `heapify` function is $O(\log n)$, as it may need to traverse the height of the tree. However, when building the heap using the bottom-up approach, the total time complexity is $O(n)$. This may seem counterintuitive initially, but it can be understood by analyzing the work done at each level of the tree.

At each level of the tree, the number of nodes decreases exponentially, while the maximum height that `heapify` may need to traverse increases linearly. The combined work done across all levels results in a linear time complexity.

Mathematically, this is represented as:

$$
\sum_{i=0}^{\log n} \frac{n}{2^i} O(\log n - i) = O(n)
$$

This shows that building a heap is an efficient operation, crucial for the performance of Heap Sort and other heap-based algorithms.

#### Practical Applications of Heaps

Heaps are not just theoretical constructs; they have numerous practical applications, including:

1. **Priority Queues**: Heaps are often used to implement priority queues, where the highest (or lowest) priority element is always at the root and can be efficiently accessed.
2. **Graph Algorithms**: Algorithms like Dijkstra's shortest path and Prim's minimum spanning tree rely on heaps for efficient priority queue operations.
3. **Order Statistics**: Heaps can be used to find the k-th largest (or smallest) elements in an array efficiently.
4. **Scheduling Algorithms**: In operating systems, heaps are used in job scheduling and resource allocation algorithms to manage tasks with varying priorities.

#### Conclusion

The creation and maintenance of heaps through the `heapify` operation are fundamental to many algorithmic processes. By ensuring that a binary tree maintains the heap property, these operations enable efficient sorting, priority management, and various other computational tasks. Understanding the mechanics and efficiency of these operations provides a solid foundation for leveraging heaps in practical applications, from data structures to complex algorithms.

In summary, the detailed exploration of heap construction and `heapify` operations highlights the elegance and efficiency of this approach, making heaps a powerful tool in the realm of computer science and algorithm design.

### 2.6.3. Teljesítmény és Komplexitás Elemzése

Heap Sort is widely recognized for its efficient performance characteristics and predictable time complexity. In this section, we will conduct a comprehensive analysis of the performance and complexity of Heap Sort. We will discuss the worst-case, average-case, and best-case time complexities, as well as the space complexity. Additionally, we will compare Heap Sort with other sorting algorithms to highlight its strengths and weaknesses.

#### Time Complexity Analysis

The time complexity of Heap Sort can be broken down into two main phases: building the heap and performing the sorting by repeatedly extracting the maximum element.

1. **Building the Heap**: The process of converting an unsorted array into a max-heap.
2. **Sorting the Array**: The process of repeatedly extracting the maximum element from the heap and maintaining the heap property.

##### Building the Heap

The heap construction process, as described earlier, involves calling the `heapify` function on all non-leaf nodes of the heap. The `heapify` function, in the worst case, has a time complexity of $O(\log n)$. When building the heap from the bottom up, the total time complexity is $O(n)$. This result can be understood by considering that each level of the heap requires progressively less work:

$$
T(n) = \sum_{i=0}^{\log n} \frac{n}{2^i} O(\log n - i)
$$

Where $\frac{n}{2^i}$ represents the number of nodes at level $i$, and $O(\log n - i)$ represents the time complexity for `heapify` at that level. Summing this across all levels results in a total time complexity of $O(n)$.

##### Sorting the Array

After building the heap, the sorting phase begins. This phase involves repeatedly extracting the maximum element from the root of the heap and moving it to the end of the array. After each extraction, the `heapify` function is called to restore the heap property.

1. **Extract Maximum Element**: This involves swapping the root with the last element in the heap and reducing the heap size by one.
2. **Heapify**: This operation ensures that the remaining elements maintain the max-heap property.

Since there are $n$ elements in the array, and each extraction followed by `heapify` takes $O(\log n)$ time, the total time complexity for the sorting phase is $O(n \log n)$.

#### Overall Time Complexity

Combining the heap construction phase and the sorting phase, the overall time complexity of Heap Sort is:

$$
T(n) = O(n) + O(n \log n) = O(n \log n)
$$

This makes Heap Sort highly efficient, especially for large datasets, as its time complexity is consistently $O(n \log n)$ regardless of the input data distribution.

##### Best-Case Time Complexity

Heap Sort does not have a best-case time complexity better than $O(n \log n)$. Even if the array is already sorted, the algorithm still needs to build the heap and perform the sorting operations as described. Therefore, the best-case time complexity is also $O(n \log n)$.

##### Worst-Case Time Complexity

Similarly, the worst-case time complexity of Heap Sort is $O(n \log n)$. The nature of the heap data structure and the `heapify` operation ensures that even in the worst-case scenario, the algorithm will not exceed $O(n \log n)$ time complexity.

##### Average-Case Time Complexity

The average-case time complexity of Heap Sort remains $O(n \log n)$. Since the operations involved in heap construction and extraction are not dependent on the initial order of the elements, the time complexity is consistent across different inputs.

#### Space Complexity Analysis

One of the significant advantages of Heap Sort is its space efficiency. Heap Sort is an in-place sorting algorithm, which means it requires only a constant amount of additional space. The space complexity of Heap Sort is $O(1)$.

#### Comparisons with Other Sorting Algorithms

Heap Sort is often compared with other popular sorting algorithms such as Quick Sort, Merge Sort, and Bubble Sort. Each of these algorithms has its own strengths and weaknesses, which we will explore in this section.

##### Quick Sort

- **Time Complexity**: Quick Sort has an average-case time complexity of $O(n \log n)$ but a worst-case time complexity of $O(n^2)$. However, with good pivot selection strategies (like random pivoting), the worst-case can be avoided in practice.
- **Space Complexity**: Quick Sort is not an in-place sorting algorithm and requires $O(\log n)$ space for the recursion stack.

Comparison: Heap Sort is more predictable in terms of time complexity, always performing at $O(n \log n)$. Quick Sort, however, is often faster in practice due to better cache performance, but its worst-case $O(n^2)$ can be problematic without optimizations.

##### Merge Sort

- **Time Complexity**: Merge Sort has a guaranteed time complexity of $O(n \log n)$ in all cases.
- **Space Complexity**: Merge Sort requires $O(n)$ additional space due to the need for temporary arrays during the merge process.

Comparison: While both Heap Sort and Merge Sort have the same time complexity, Heap Sort is more space-efficient, requiring only $O(1)$ additional space compared to $O(n)$ for Merge Sort.

##### Bubble Sort

- **Time Complexity**: Bubble Sort has a time complexity of $O(n^2)$ in the average and worst cases.
- **Space Complexity**: Bubble Sort is an in-place sorting algorithm with $O(1)$ space complexity.

Comparison: Heap Sort is significantly more efficient than Bubble Sort in terms of time complexity, making it a better choice for large datasets. Both algorithms are in-place, but Heap Sort's $O(n \log n)$ time complexity makes it far superior for performance.

#### Practical Considerations

In practical applications, the choice of sorting algorithm can depend on various factors including the size of the dataset, memory constraints, and the nature of the data.

- **Large Datasets**: For large datasets, Heap Sort's $O(n \log n)$ time complexity and $O(1)$ space complexity make it a reliable choice.
- **Stability**: Heap Sort is not a stable sort, meaning it does not preserve the relative order of equal elements. If stability is required, algorithms like Merge Sort or Stable Quick Sort should be considered.
- **Memory Constraints**: In environments with limited memory, Heap Sort's in-place nature is advantageous.

#### Conclusion

Heap Sort is a robust and efficient sorting algorithm with a consistent $O(n \log n)$ time complexity across best, average, and worst cases. Its in-place nature makes it space-efficient, requiring only $O(1)$ additional space. While it may not be as fast as Quick Sort in practice due to cache performance, its predictability and space efficiency make it a valuable tool for a variety of applications. Understanding the performance and complexity of Heap Sort provides insights into its optimal use cases and advantages over other sorting algorithms.

### 2.6.4. Gyakorlati Alkalmazások és Példák

Heap Sort, along with the heap data structure, finds numerous applications across various domains in computer science and beyond. This section explores the practical applications of Heap Sort, illustrating its use through detailed examples. These applications range from priority queues and graph algorithms to order statistics and real-world scheduling problems.

#### Priority Queues

A priority queue is an abstract data type where each element has a priority associated with it. Elements are served based on their priority, with the highest priority elements served first. Priority queues are widely used in various algorithms and system processes, and heaps are an ideal data structure for implementing priority queues due to their efficient insertion and extraction operations.

**Example**: Implementing a Priority Queue using a Max-Heap

```cpp
#include <iostream>

#include <vector>

class PriorityQueue {
private:
    std::vector<int> heap;

    void heapifyUp(int index) {
        if (index == 0) return;
        int parent = (index - 1) / 2;
        if (heap[parent] < heap[index]) {
            std::swap(heap[parent], heap[index]);
            heapifyUp(parent);
        }
    }

    void heapifyDown(int index) {
        int left = 2 * index + 1;
        int right = 2 * index + 2;
        int largest = index;

        if (left < heap.size() && heap[left] > heap[largest])
            largest = left;
        if (right < heap.size() && heap[right] > heap[largest])
            largest = right;
        if (largest != index) {
            std::swap(heap[index], heap[largest]);
            heapifyDown(largest);
        }
    }

public:
    void insert(int value) {
        heap.push_back(value);
        heapifyUp(heap.size() - 1);
    }

    int extractMax() {
        if (heap.empty()) throw std::runtime_error("Heap is empty");
        int maxValue = heap.front();
        heap[0] = heap.back();
        heap.pop_back();
        heapifyDown(0);
        return maxValue;
    }

    bool isEmpty() const {
        return heap.empty();
    }
};

int main() {
    PriorityQueue pq;
    pq.insert(10);
    pq.insert(30);
    pq.insert(20);
    pq.insert(5);

    while (!pq.isEmpty()) {
        std::cout << pq.extractMax() << " ";
    }
    return 0;
}
```

In this example, a priority queue is implemented using a max-heap. The `insert` method adds a new element to the heap and maintains the heap property by calling `heapifyUp`. The `extractMax` method removes and returns the maximum element, reestablishing the heap property with `heapifyDown`.

#### Graph Algorithms

Heaps are fundamental in several graph algorithms, particularly those that involve priority queues. Two notable examples are Dijkstra's algorithm for shortest paths and Prim's algorithm for minimum spanning trees.

**Example**: Dijkstra's Algorithm using a Min-Heap

Dijkstra's algorithm finds the shortest paths from a source vertex to all other vertices in a weighted graph. It uses a priority queue to repeatedly extract the vertex with the minimum distance estimate.

```cpp
#include <iostream>

#include <vector>
#include <queue>

#include <limits>

struct Edge {
    int target, weight;
};

class Graph {
private:
    std::vector<std::vector<Edge>> adjList;

public:
    Graph(int vertices) : adjList(vertices) {}

    void addEdge(int src, int tgt, int weight) {
        adjList[src].push_back({tgt, weight});
    }

    std::vector<int> dijkstra(int src) {
        int n = adjList.size();
        std::vector<int> dist(n, std::numeric_limits<int>::max());
        dist[src] = 0;

        using pii = std::pair<int, int>;
        std::priority_queue<pii, std::vector<pii>, std::greater<pii>> pq;
        pq.push({0, src});

        while (!pq.empty()) {
            int u = pq.top().second;
            pq.pop();

            for (const Edge& edge : adjList[u]) {
                int v = edge.target;
                int weight = edge.weight;
                if (dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    pq.push({dist[v], v});
                }
            }
        }
        return dist;
    }
};

int main() {
    Graph g(5);
    g.addEdge(0, 1, 10);
    g.addEdge(0, 4, 5);
    g.addEdge(1, 2, 1);
    g.addEdge(4, 1, 3);
    g.addEdge(4, 2, 9);
    g.addEdge(4, 3, 2);
    g.addEdge(2, 3, 4);
    g.addEdge(3, 2, 6);

    std::vector<int> distances = g.dijkstra(0);
    for (int i = 0; i < distances.size(); ++i) {
        std::cout << "Distance from 0 to " << i << ": " << distances[i] << std::endl;
    }
    return 0;
}
```

In this example, Dijkstra's algorithm is implemented using a min-heap priority queue. The algorithm maintains a priority queue of vertices to be explored, with the priority determined by the current shortest path estimate.

#### Order Statistics

Heaps can efficiently solve order statistics problems, such as finding the k-th largest or smallest elements in an array. By maintaining a heap of the k largest elements encountered so far, these problems can be solved in $O(n \log k)$ time.

**Example**: Finding the k-th Largest Element

```cpp
#include <iostream>

#include <vector>
#include <queue>

int findKthLargest(std::vector<int>& nums, int k) {
    std::priority_queue<int, std::vector<int>, std::greater<int>> minHeap;

    for (int num : nums) {
        minHeap.push(num);
        if (minHeap.size() > k) {
            minHeap.pop();
        }
    }
    return minHeap.top();
}

int main() {
    std::vector<int> nums = {3, 2, 1, 5, 6, 4};
    int k = 2;
    std::cout << "The " << k << "-th largest element is " << findKthLargest(nums, k) << std::endl;
    return 0;
}
```

In this example, a min-heap is used to find the k-th largest element in an array. The min-heap maintains the k largest elements seen so far, and its top element is the k-th largest.

#### Real-World Scheduling Problems

Heaps are employed in various real-world scheduling problems, where tasks need to be scheduled based on priority or deadlines. Examples include CPU task scheduling, job scheduling in operating systems, and event simulation systems.

**Example**: CPU Task Scheduling using a Min-Heap

In a multi-tasking operating system, the CPU needs to schedule tasks based on their priorities or deadlines. A min-heap can be used to manage the tasks efficiently.

```cpp
#include <iostream>

#include <queue>
#include <vector>

struct Task {
    int id;
    int priority;
    bool operator>(const Task& other) const {
        return priority > other.priority;
    }
};

int main() {
    std::priority_queue<Task, std::vector<Task>, std::greater<Task>> taskQueue;

    taskQueue.push({1, 3});
    taskQueue.push({2, 1});
    taskQueue.push({3, 2});

    while (!taskQueue.empty()) {
        Task task = taskQueue.top();
        taskQueue.pop();
        std::cout << "Processing task " << task.id << " with priority " << task.priority << std::endl;
    }

    return 0;
}
```

In this example, tasks are scheduled based on their priorities using a min-heap priority queue. The task with the highest priority (lowest priority value) is processed first.

#### Conclusion

The practical applications of heaps and Heap Sort are extensive and varied, highlighting the versatility and efficiency of this data structure and algorithm. From implementing priority queues and solving graph problems to managing order statistics and scheduling tasks, heaps provide a robust solution for many computational challenges. By understanding and leveraging the properties of heaps, one can develop efficient and effective solutions for a wide range of problems in computer science and beyond.

