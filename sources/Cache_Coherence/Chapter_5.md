\newpage
## Chapter 5: Writing Cache-Friendly Code in C++

### 5.1 Techniques for Improving Cache Utilization

Optimizing code for cache utilization is crucial for achieving high performance in C++ applications, especially in embedded systems where resources are limited and efficiency is paramount. This section explores various techniques for improving cache utilization, providing practical examples and detailed explanations to help you write cache-friendly code.

#### **5.1.1 Understanding Cache Utilization**

Cache utilization refers to how effectively a program uses the CPU cache. Poor cache utilization leads to frequent cache misses, causing the CPU to wait for data to be fetched from slower main memory. Improving cache utilization involves structuring your code and data to maximize cache hits and minimize cache misses.

- **Example**: Consider an application that processes large datasets, such as a digital signal processing algorithm. Efficient cache utilization ensures that the data required for computations is readily available in the cache, significantly speeding up the processing.

#### **5.1.2 Data Locality**

Data locality is a key factor in improving cache utilization. It refers to accessing data elements that are close to each other in memory, which increases the likelihood that the data is already loaded in the cache.

1. **Spatial Locality**: Accessing data elements that are close together in memory. This is particularly important for arrays and other contiguous data structures.

    - **Example**: When iterating over a large array, access elements sequentially to take advantage of spatial locality.

    ```cpp
    for (int i = 0; i < size; ++i) {
        process(array[i]);
    }
    ```

2. **Temporal Locality**: Reusing recently accessed data elements. This increases the chance that the data is still in the cache when it is accessed again.

    - **Example**: In a loop, reuse variables and data that were accessed in recent iterations.

    ```cpp
    for (int i = 0; i < size; ++i) {
        process(array[i]);
        if (i > 0) {
            process(array[i - 1]); // Reusing recently accessed element
        }
    }
    ```

#### **5.1.3 Optimizing Data Structures**

Choosing and organizing data structures to improve cache utilization can significantly impact performance.

1. **Arrays vs. Linked Lists**: Arrays provide better spatial locality compared to linked lists because their elements are stored contiguously in memory.

    - **Example**: Use arrays instead of linked lists for data structures that will be traversed frequently.

    ```cpp
    std::vector<int> data;
    // Instead of
    std::list<int> data;
    ```

2. **Struct Packing and Alignment**: Organize the fields of structs to minimize padding and ensure that frequently accessed fields are close together.

    - **Example**: Reorganize struct fields to improve cache performance.

    ```cpp
    struct Optimized {
        int a;
        char b;
        // Padding
        char pad[3];
        float c;
    };
    ```

3. **SoA vs. AoS**: Structure of Arrays (SoA) can be more cache-friendly than Array of Structures (AoS) for certain types of data access patterns.

    - **Example**: Transform an AoS into a SoA for better cache utilization.

    ```cpp
    struct Point {
        float x, y, z;
    };

    // AoS
    std::vector<Point> points;

    // SoA
    struct Points {
        std::vector<float> x, y, z;
    } points;
    ```

#### **5.1.4 Loop Transformations**

Loop transformations can significantly enhance cache utilization by improving data locality and reducing cache misses.

1. **Loop Interchange**: Swap the order of nested loops to access memory in a more cache-friendly manner.

    - **Example**: Optimize matrix multiplication by interchanging loops.

    ```cpp
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    ```

2. **Loop Tiling**: Divide loops into smaller blocks or tiles to improve data locality.

    - **Example**: Apply loop tiling to a matrix multiplication loop.

    ```cpp
    const int tileSize = 32;

    for (int i = 0; i < N; i += tileSize) {
        for (int j = 0; j < N; j += tileSize) {
            for (int k = 0; k < N; k += tileSize) {
                for (int ii = i; ii < std::min(i + tileSize, N); ++ii) {
                    for (int jj = j; jj < std::min(j + tileSize, N); ++jj) {
                        for (int kk = k; kk < std::min(k + tileSize, N); ++kk) {
                            C[ii][jj] += A[ii][kk] * B[kk][jj];
                        }
                    }
                }
            }
        }
    }
    ```

3. **Loop Unrolling**: Unroll loops to reduce loop overhead and increase the amount of work done per iteration.

    - **Example**: Unroll a loop to process multiple elements per iteration.

    ```cpp
    for (int i = 0; i < size; i += 4) {
        sum += array[i] + array[i+1] + array[i+2] + array[i+3];
    }
    ```

#### **5.1.5 Prefetching**

Prefetching data into the cache before it is needed can significantly reduce cache misses.

1. **Software Prefetching**: Use compiler intrinsics to prefetch data.

    - **Example**: Prefetch data in a loop to improve performance.

    ```cpp
    for (int i = 0; i < size; ++i) {
        _mm_prefetch(reinterpret_cast<const char*>(&array[i + 16]), _MM_HINT_T0);
        process(array[i]);
    }
    ```

2. **Hardware Prefetching**: Leverage hardware prefetching features, which automatically detect and prefetch data based on access patterns.

    - **Example**: Optimize data access patterns to align with hardware prefetching capabilities.

    ```cpp
    // Access data sequentially to take advantage of hardware prefetching
    for (int i = 0; i < size; ++i) {
        process(array[i]);
    }
    ```

#### **5.1.6 Real-Life Example: Image Processing Application**

Consider an image processing application that applies a blur filter to an image. Optimizing cache utilization can significantly enhance performance.

##### **Initial Code**

```cpp
void blurImage(int width, int height, int image[height][width]) {
    int result[height][width] = {0};

    for (int i = 1; i < height - 1; ++i) {
        for (int j = 1; j < width - 1; ++j) {
            result[i][j] = (
                image[i-1][j-1] + image[i-1][j] + image[i-1][j+1] +
                image[i][j-1] + image[i][j] + image[i][j+1] +
                image[i+1][j-1] + image[i+1][j] + image[i+1][j+1]
            ) / 9;
        }
    }
}
```

##### **Optimized Code**

1. **Reorganize Data Access**: Optimize the loop order to improve cache utilization.

    ```cpp
    void blurImage(int width, int height, int image[height][width]) {
        int result[height][width] = {0};

        for (int j = 1; j < width - 1; ++j) {
            for (int i = 1; i < height - 1; ++i) {
                result[i][j] = (
                    image[i-1][j-1] + image[i-1][j] + image[i-1][j+1] +
                    image[i][j-1] + image[i][j] + image[i][j+1] +
                    image[i+1][j-1] + image[i+1][j] + image[i+1][j+1]
                ) / 9;
            }
        }
    }
    ```

2. **Apply Loop Tiling**: Divide the image into tiles to enhance data locality.

    ```cpp
    const int tileSize = 32;

    void blurImage(int width, int height, int image[height][width]) {
        int result[height][width] = {0};

        for (int jj = 1; jj < width - 1; jj += tileSize) {
            for (int ii = 1; ii < height - 1; ii += tileSize) {
                for (int j = jj; j < std::min(jj + tileSize, width - 1); ++j) {
                    for (int i = ii; i < std::min(ii + tileSize, height - 1); ++i) {
                        result[i][j] = (
                            image[i-1][j-1] + image[i-1][j] + image[i-1][j+1] +
                            image[i][j-1] + image[i][j] + image[i][j+1] +
                            image[i+1][j-1] + image[i+1][j] + image[i+1][j+1]
                        ) / 9;
                    }
                }
            }
        }
    }
    ```

#### **5.1.7 Conclusion**

Improving cache utilization is crucial for achieving high performance in C++ applications, especially in resource-constrained embedded systems. By understanding and applying techniques such as enhancing data locality, optimizing data structures, transforming loops, and prefetching data, you can significantly reduce cache misses and enhance the overall efficiency of your code. These techniques are essential tools in the arsenal of any embedded systems developer, enabling the creation of fast, reliable, and efficient applications. The following sections will delve into more advanced optimization strategies, providing a comprehensive guide to mastering cache optimization in C++.


### 5.2 Example Code: Optimizing Arrays and Pointers

Optimizing arrays and pointers is essential for enhancing cache utilization and overall performance in C++ applications. Arrays and pointers are fundamental data structures that, when used effectively, can significantly reduce cache misses and improve data access patterns. This section provides detailed examples and explanations of how to optimize arrays and pointers for better cache performance.

#### **5.2.1 Optimizing Array Access Patterns**

Arrays are stored contiguously in memory, making them naturally cache-friendly. However, the way you access array elements can greatly affect cache performance. Accessing elements in a sequential manner leverages spatial locality, ensuring that once an array element is loaded into the cache, the following elements are likely to be loaded as well.

##### **Example: Sequential Access**

Consider a simple array summation:

```cpp
void sumArraySequential(int* array, size_t size) {
    int sum = 0;
    for (size_t i = 0; i < size; ++i) {
        sum += array[i];
    }
    // Use the sum for something
}
```

In this example, accessing array elements sequentially ensures that the CPU cache is utilized efficiently, as consecutive elements are likely to be in the same cache line.

##### **Example: Strided Access**

Strided access occurs when elements are accessed at regular intervals, which can lead to inefficient cache usage if the stride is larger than the cache line size.

```cpp
void sumArrayStrided(int* array, size_t size, size_t stride) {
    int sum = 0;
    for (size_t i = 0; i < size; i += stride) {
        sum += array[i];
    }
    // Use the sum for something
}
```

Strided access can lead to cache misses if the stride is not cache-friendly. For instance, accessing every 64th element in a system with a 64-byte cache line size can result in each access missing the cache.

##### **Optimizing Strided Access**

To optimize strided access, try to minimize the stride length or use loop transformations to improve data locality.

```cpp
void sumArrayOptimizedStrided(int* array, size_t size) {
    int sum = 0;
    for (size_t i = 0; i < size; i += 4) {
        sum += array[i] + array[i + 1] + array[i + 2] + array[i + 3];
    }
    // Handle remaining elements
    for (size_t i = (size / 4) * 4; i < size; ++i) {
        sum += array[i];
    }
    // Use the sum for something
}
```

By unrolling the loop, we reduce the stride's impact, accessing more elements within the same cache line.

#### **5.2.2 Optimizing Multidimensional Arrays**

Multidimensional arrays can present challenges for cache optimization due to their row-major or column-major storage order. Accessing elements in a way that aligns with their storage order is crucial for efficient cache utilization.

##### **Example: Row-Major Access**

C++ arrays are stored in row-major order, meaning that rows are stored contiguously. Accessing elements row by row is cache-friendly.

```cpp
void processMatrixRowMajor(int** matrix, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            process(matrix[i][j]);
        }
    }
}
```

In this example, accessing elements row by row ensures that elements within the same row are loaded into the cache together, improving cache performance.

##### **Example: Column-Major Access**

Accessing elements column by column in a row-major array can lead to poor cache performance.

```cpp
void processMatrixColumnMajor(int** matrix, size_t rows, size_t cols) {
    for (size_t j = 0; j < cols; ++j) {
        for (size_t i = 0; i < rows; ++i) {
            process(matrix[i][j]);
        }
    }
}
```

To optimize column-major access, consider restructuring the data or transforming the loops.

#### **5.2.3 Optimizing Pointer-Based Data Structures**

Pointer-based data structures, such as linked lists and trees, can suffer from poor cache performance due to their non-contiguous memory layout. Optimizing these structures involves improving spatial locality and reducing pointer chasing.

##### **Example: Linked List**

A standard linked list has nodes scattered across memory, leading to cache misses during traversal.

```cpp
struct Node {
    int data;
    Node* next;
};

int sumLinkedList(Node* head) {
    int sum = 0;
    while (head != nullptr) {
        sum += head->data;
        head = head->next;
    }
    return sum;
}
```

##### **Optimizing Linked List**

To optimize a linked list, consider using a contiguous block of memory to store nodes or applying cache-friendly techniques.

```cpp
struct Node {
    int data;
    Node* next;
};

class ContiguousLinkedList {
public:
    ContiguousLinkedList(size_t size) : size(size), nodes(new Node[size]) {
        for (size_t i = 0; i < size - 1; ++i) {
            nodes[i].next = &nodes[i + 1];
        }
        nodes[size - 1].next = nullptr;
    }

    Node* head() { return nodes; }

private:
    size_t size;
    Node* nodes;
};

int sumContiguousLinkedList(Node* head) {
    int sum = 0;
    while (head != nullptr) {
        sum += head->data;
        head = head->next;
    }
    return sum;
}
```

By allocating nodes contiguously, we improve spatial locality and cache performance.

##### **Example: Binary Tree**

A binary tree can also benefit from cache-friendly techniques. Consider traversing the tree in a way that improves cache performance.

```cpp
struct TreeNode {
    int data;
    TreeNode* left;
    TreeNode* right;
};

void inorderTraversal(TreeNode* root) {
    if (root) {
        inorderTraversal(root->left);
        process(root->data);
        inorderTraversal(root->right);
    }
}
```

##### **Optimizing Binary Tree**

For a binary tree, techniques such as cache-oblivious algorithms or storing nodes in a contiguous array can help.

```cpp
void cacheFriendlyInorder(TreeNode* root) {
    if (!root) return;
    TreeNode* stack[1000]; // Example stack size
    int top = -1;
    TreeNode* current = root;

    while (current || top != -1) {
        while (current) {
            stack[++top] = current;
            current = current->left;
        }
        current = stack[top--];
        process(current->data);
        current = current->right;
    }
}
```

Using an explicit stack instead of recursion can reduce overhead and improve cache performance.

#### **5.2.4 Real-Life Example: Image Processing with Arrays and Pointers**

Consider an image processing application that applies a convolution filter to an image. Optimizing the access pattern and data structure can significantly improve performance.

##### **Initial Code**

```cpp
void applyFilter(int** image, int** result, int width, int height, int filter[3][3]) {
    for (int i = 1; i < height - 1; ++i) {
        for (int j = 1; j < width - 1; ++j) {
            int sum = 0;
            for (int fi = -1; fi <= 1; ++fi) {
                for (int fj = -1; fj <= 1; ++fj) {
                    sum += image[i + fi][j + fj] * filter[fi + 1][fj + 1];
                }
            }
            result[i][j] = sum;
        }
    }
}
```

##### **Optimized Code**

1. **Optimize Data Access Pattern**: Use a contiguous array for the image.

    ```cpp
    void applyFilter(int* image, int* result, int width, int height, int filter[3][3]) {
        for (int i = 1; i < height - 1; ++i) {
            for (int j = 1; j < width - 1; ++j) {
                int sum = 0;
                for (int fi = -1; fi <= 1; ++fi) {
                    for (int fj = -1; fj <= 1; ++fj) {
                        sum += image[(i + fi) * width + (j + fj)] * filter[fi + 1][fj + 1];
                    }
                }
                result[i * width + j] = sum;
            }
        }
    }
    ```

2. **Apply Loop Unrolling**: Unroll the inner loop to reduce overhead and improve performance.

    ```cpp
    void applyFilterUnrolled(int* image, int* result, int width, int height, int filter[3][3]) {
        for (int i = 1; i < height - 1; ++i) {
            for (int j = 1; j < width - 1; ++j) {
                int sum = 0;
                sum += image[(i - 1) * width + (j - 1)] * filter[0][0];
                sum += image[(i - 1) * width + j] * filter[0][1];
                sum += image[(i - 1) * width + (j + 1)] * filter[0][2];
                sum += image[i * width + (j - 1)] * filter[1][0];
                sum += image[i * width + j] * filter[1][1];
                sum += image[i * width + (j + 1)] * filter[1][2];
                sum += image[(i + 1) * width + (j - 1)] * filter[2][0];
                sum += image[(i + 1) * width + j] * filter[2][1];
                sum += image[(i + 1) * width + (j + 1)] * filter[2][2];
                result[i * width + j] = sum;
            }
        }
    }
    ```

By optimizing the data structure and access pattern, we significantly improve the cache utilization and performance of the image processing application.

#### **5.2.5 Conclusion**

Optimizing arrays and pointers for better cache utilization is a critical aspect of writing high-performance C++ code, especially in embedded systems. By leveraging techniques such as sequential access, minimizing stride lengths, optimizing multidimensional arrays, and restructuring pointer-based data structures, you can significantly reduce cache misses and improve data access patterns. These optimizations lead to faster, more efficient applications that make better use of the available hardware resources. The next sections will explore additional advanced techniques for writing cache-friendly code, providing a comprehensive guide to mastering performance optimization in C++.



### 5.3 Example Code: Efficient Use of Function Calls and Recursion

Function calls and recursion are fundamental constructs in C++ programming, but they can also introduce performance overhead if not used efficiently. Optimizing function calls and recursion can improve cache utilization and overall performance, especially in embedded systems where resources are limited. This section explores techniques for optimizing function calls and recursion, providing practical examples and detailed explanations.

#### **5.3.1 Reducing Function Call Overhead**

Function calls introduce overhead due to the need to save the current state, pass arguments, and transfer control to the called function. In performance-critical code, minimizing this overhead is crucial.

1. **Inlining Functions**: Using the `inline` keyword can reduce the overhead of function calls by replacing the function call with the function code itself.

    - **Example**: Consider a simple function that adds two numbers.

    ```cpp
    inline int add(int a, int b) {
        return a + b;
    }

    void process() {
        int sum = add(5, 3); // The call to add() will be inlined.
    }
    ```

   By inlining the `add` function, the function call overhead is eliminated, and the addition is performed directly in the `process` function.

2. **Using Templates**: Templates can be used to generate inline functions, particularly useful for generic programming.

    - **Example**: A templated function to find the maximum of two values.

    ```cpp
    template <typename T>
    inline T max(T a, T b) {
        return (a > b) ? a : b;
    }

    void process() {
        int maxValue = max(5, 3); // The call to max() will be inlined.
    }
    ```

3. **Avoiding Excessive Function Calls in Hot Loops**: Minimize function calls inside performance-critical loops to reduce overhead.

    - **Example**: Refactor a loop to avoid function calls within the loop body.

    ```cpp
    void processArray(int* array, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            array[i] = processElement(array[i]); // Avoid function calls inside hot loops.
        }
    }

    int processElement(int value) {
        // Process the value
        return value * 2;
    }
    ```

#### **5.3.2 Optimizing Recursion**

Recursion can be elegant and intuitive but may introduce significant overhead due to repeated function calls and stack usage. Optimizing recursion involves techniques to minimize overhead and improve performance.

1. **Tail Recursion**: Tail recursion occurs when the recursive call is the last operation in the function. Compilers can optimize tail-recursive functions to reduce overhead by reusing the current function’s stack frame.

    - **Example**: A tail-recursive function to calculate the factorial of a number.

    ```cpp
    int factorial(int n, int result = 1) {
        if (n == 0) return result;
        return factorial(n - 1, n * result); // Tail-recursive call.
    }

    void process() {
        int result = factorial(5); // Optimized by the compiler.
    }
    ```

2. **Iterative Solutions**: Converting recursive functions to iterative ones can eliminate function call overhead and reduce stack usage.

    - **Example**: An iterative version of the factorial function.

    ```cpp
    int factorialIterative(int n) {
        int result = 1;
        for (int i = 1; i <= n; ++i) {
            result *= i;
        }
        return result;
    }

    void process() {
        int result = factorialIterative(5); // No recursion overhead.
    }
    ```

3. **Memoization**: Memoization involves storing the results of expensive function calls and reusing them when the same inputs occur again, reducing redundant calculations in recursive algorithms.

    - **Example**: A memoized version of the Fibonacci function.

    ```cpp
    int fibonacci(int n, std::vector<int>& memo) {
        if (n <= 1) return n;
        if (memo[n] != -1) return memo[n]; // Return cached result if available.
        memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo); // Store result in cache.
        return memo[n];
    }

    void process() {
        int n = 10;
        std::vector<int> memo(n + 1, -1);
        int result = fibonacci(n, memo); // Efficiently computes the Fibonacci number.
    }
    ```

#### **5.3.3 Real-Life Example: Optimizing a Recursive Tree Traversal**

Consider an example where we need to traverse a binary tree. We can optimize the traversal to reduce function call overhead and improve cache utilization.

##### **Initial Code**

```cpp
struct TreeNode {
    int value;
    TreeNode* left;
    TreeNode* right;
};

void inorderTraversal(TreeNode* root) {
    if (root) {
        inorderTraversal(root->left);
        process(root->value);
        inorderTraversal(root->right);
    }
}

void processTree(TreeNode* root) {
    inorderTraversal(root); // Traverse the tree in-order.
}
```

##### **Optimized Code**

1. **Tail Recursion**: Convert the recursive function to use tail recursion where possible.

    ```cpp
    void tailInorderTraversal(TreeNode* root) {
        while (root) {
            if (root->left) {
                TreeNode* pre = root->left;
                while (pre->right && pre->right != root) {
                    pre = pre->right;
                }
                if (!pre->right) {
                    pre->right = root;
                    root = root->left;
                } else {
                    pre->right = nullptr;
                    process(root->value);
                    root = root->right;
                }
            } else {
                process(root->value);
                root = root->right;
            }
        }
    }

    void processTree(TreeNode* root) {
        tailInorderTraversal(root); // Tail-recursive traversal.
    }
    ```

2. **Iterative Solution**: Convert the recursive traversal to an iterative one using an explicit stack.

    ```cpp
    void iterativeInorderTraversal(TreeNode* root) {
        std::stack<TreeNode*> stack;
        TreeNode* current = root;

        while (!stack.empty() || current) {
            while (current) {
                stack.push(current);
                current = current->left;
            }
            current = stack.top();
            stack.pop();
            process(current->value);
            current = current->right;
        }
    }

    void processTree(TreeNode* root) {
        iterativeInorderTraversal(root); // Iterative traversal.
    }
    ```

#### **5.3.4 Reducing Recursion Depth**

For deeply recursive functions, reducing the recursion depth can prevent stack overflow and improve performance. Techniques include dividing the problem into smaller parts or using hybrid recursive-iterative approaches.

##### **Example: QuickSort with Reduced Recursion Depth**

```cpp
void quicksort(int* arr, int left, int right) {
    while (left < right) {
        int pivot = partition(arr, left, right);
        if (pivot - left < right - pivot) {
            quicksort(arr, left, pivot - 1); // Recursively sort the left part.
            left = pivot + 1; // Iteratively sort the right part.
        } else {
            quicksort(arr, pivot + 1, right); // Recursively sort the right part.
            right = pivot - 1; // Iteratively sort the left part.
        }
    }
}

void sortArray(int* arr, int size) {
    quicksort(arr, 0, size - 1); // Efficient quicksort with reduced recursion depth.
}
```

By iteratively handling the larger partition, we reduce the maximum recursion depth, preventing stack overflow and improving performance.

#### **5.3.5 Inlining Small Functions**

Inlining small, frequently called functions can eliminate function call overhead and improve cache locality. However, excessive inlining can increase code size, potentially leading to instruction cache misses.

##### **Example: Inlining a Small Function**

```cpp
inline int square(int x) {
    return x * x;
}

void processArray(int* array, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        array[i] = square(array[i]); // Inline the square function.
    }
}
```

In this example, the `square` function is inlined, eliminating the function call overhead and potentially improving performance.

#### **5.3.6 Conclusion**

Efficient use of function calls and recursion is crucial for writing high-performance C++ code, especially in embedded systems where resources are limited. By reducing function call overhead, optimizing recursion, and leveraging techniques such as inlining and memoization, you can significantly improve cache utilization and overall performance. These optimizations lead to faster, more efficient applications that make better use of the available hardware resources. The following sections will explore additional advanced techniques for writing cache-friendly code, providing a comprehensive guide to mastering performance optimization in C++.

