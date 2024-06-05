\newpage
## Chapter 6: Multithreading and Concurrency

### 6.1 Basics of Multithreading in C++

Multithreading is a powerful technique that allows a program to execute multiple threads concurrently, potentially improving performance and responsiveness, especially in modern multi-core processors. Understanding the basics of multithreading in C++ involves learning how to create, manage, and synchronize threads effectively. This section covers the fundamentals of multithreading in C++, providing practical examples and detailed explanations.

#### **6.1.1 Introduction to Multithreading**

Multithreading enables a program to perform multiple tasks simultaneously by running separate threads of execution. Each thread runs independently but shares the same address space, allowing them to access shared data. However, this also introduces challenges in managing data consistency and synchronization.

- **Example**: Consider a web server that handles multiple client requests simultaneously. Using multithreading, the server can process multiple requests in parallel, improving throughput and responsiveness.

#### **6.1.2 Creating and Managing Threads**

In C++, the `std::thread` class, introduced in C++11, provides a simple and efficient way to create and manage threads.

##### **Creating a Thread**

A thread can be created by passing a function or callable object to the `std::thread` constructor.

- **Example**: Creating a thread that executes a function.

    ```cpp
    #include <iostream>
    #include <thread>

    void printMessage(const std::string& message) {
        std::cout << message << std::endl;
    }

    int main() {
        std::string message = "Hello from the thread!";
        std::thread t(printMessage, message); // Create a thread that runs printMessage.
        t.join(); // Wait for the thread to finish.
        return 0;
    }
    ```

  In this example, a new thread is created to execute the `printMessage` function. The `join` method is called to wait for the thread to finish execution before the program continues.

##### **Using Lambda Functions**

Lambda functions can be used to create threads without defining separate functions.

- **Example**: Creating a thread with a lambda function.

    ```cpp
    int main() {
        std::thread t([]() {
            std::cout << "Hello from the lambda thread!" << std::endl;
        });
        t.join(); // Wait for the thread to finish.
        return 0;
    }
    ```

  This example demonstrates how to create a thread using a lambda function, making the code more concise.

#### **6.1.3 Thread Synchronization**

When multiple threads access shared data, synchronization mechanisms are required to ensure data consistency and prevent race conditions.

##### **Mutexes**

A mutex (`std::mutex`) is a synchronization primitive used to protect shared data by ensuring that only one thread can access the data at a time.

- **Example**: Using a mutex to protect shared data.

    ```cpp
    #include <iostream>
    #include <thread>
    #include <mutex>

    std::mutex mtx;
    int sharedCounter = 0;

    void incrementCounter() {
        for (int i = 0; i < 1000; ++i) {
            std::lock_guard<std::mutex> lock(mtx); // Lock the mutex.
            ++sharedCounter;
        }
    }

    int main() {
        std::thread t1(incrementCounter);
        std::thread t2(incrementCounter);

        t1.join();
        t2.join();

        std::cout << "Final counter value: " << sharedCounter << std::endl;
        return 0;
    }
    ```

  In this example, the `incrementCounter` function increments a shared counter. The `std::lock_guard` automatically locks the mutex when created and unlocks it when destroyed, ensuring that only one thread can modify the counter at a time.

##### **Condition Variables**

Condition variables (`std::condition_variable`) allow threads to wait for certain conditions to be met.

- **Example**: Using a condition variable for thread synchronization.

    ```cpp
    #include <iostream>
    #include <thread>
    #include <mutex>
    #include <condition_variable>

    std::mutex mtx;
    std::condition_variable cv;
    bool ready = false;

    void printMessage() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, []() { return ready; }); // Wait until ready is true.
        std::cout << "Thread is running!" << std::endl;
    }

    void setReady() {
        std::unique_lock<std::mutex> lock(mtx);
        ready = true;
        cv.notify_one(); // Notify one waiting thread.
    }

    int main() {
        std::thread t1(printMessage);
        std::thread t2(setReady);

        t1.join();
        t2.join();

        return 0;
    }
    ```

  In this example, the `printMessage` function waits for the `ready` flag to be set to `true` before printing a message. The `setReady` function sets the `ready` flag and notifies the waiting thread using the condition variable.

#### **6.1.4 Avoiding Common Multithreading Issues**

Multithreading introduces challenges such as race conditions, deadlocks, and data corruption. Understanding and avoiding these issues is crucial for writing robust multithreaded code.

##### **Race Conditions**

Race conditions occur when multiple threads access and modify shared data concurrently, leading to unpredictable results.

- **Solution**: Use mutexes or other synchronization primitives to protect shared data.

    ```cpp
    std::mutex mtx;
    int sharedCounter = 0;

    void incrementCounter() {
        std::lock_guard<std::mutex> lock(mtx);
        ++sharedCounter;
    }
    ```

##### **Deadlocks**

Deadlocks occur when two or more threads are blocked forever, each waiting for a resource held by the other.

- **Solution**: Avoid nested locks and use consistent lock ordering.

    ```cpp
    std::mutex mtx1, mtx2;

    void threadFunc1() {
        std::lock(mtx1, mtx2); // Lock both mutexes without risk of deadlock.
        std::lock_guard<std::mutex> lock1(mtx1, std::adopt_lock);
        std::lock_guard<std::mutex> lock2(mtx2, std::adopt_lock);
        // Perform operations.
    }

    void threadFunc2() {
        std::lock(mtx1, mtx2); // Lock both mutexes without risk of deadlock.
        std::lock_guard<std::mutex> lock1(mtx1, std::adopt_lock);
        std::lock_guard<std::mutex> lock2(mtx2, std::adopt_lock);
        // Perform operations.
    }
    ```

##### **Data Corruption**

Data corruption occurs when multiple threads read and write shared data without proper synchronization, leading to inconsistent or incorrect data.

- **Solution**: Use atomic operations or mutexes to ensure data integrity.

    ```cpp
    #include <atomic>

    std::atomic<int> sharedCounter(0);

    void incrementCounter() {
        ++sharedCounter; // Atomic increment.
    }
    ```

#### **6.1.5 Real-Life Example: Multithreaded File Processing**

Consider an example where a program processes multiple files concurrently. Each thread reads a file and counts the number of lines.

##### **Initial Code**

```cpp
#include <iostream>

#include <fstream>
#include <thread>

#include <vector>
#include <string>

void countLines(const std::string& filename, int& lineCount) {
    std::ifstream file(filename);
    std::string line;
    lineCount = 0;
    while (std::getline(file, line)) {
        ++lineCount;
    }
}

int main() {
    std::vector<std::string> filenames = {"file1.txt", "file2.txt", "file3.txt"};
    std::vector<int> lineCounts(filenames.size());
    std::vector<std::thread> threads;

    for (size_t i = 0; i < filenames.size(); ++i) {
        threads.emplace_back(countLines, filenames[i], std::ref(lineCounts[i]));
    }

    for (auto& t : threads) {
        t.join();
    }

    for (size_t i = 0; i < filenames.size(); ++i) {
        std::cout << filenames[i] << ": " << lineCounts[i] << " lines" << std::endl;
    }

    return 0;
}
```

In this example, multiple threads are created to count the lines in different files concurrently. The `countLines` function reads a file and counts its lines, and the results are printed after all threads have finished execution.

##### **Optimized Code with Mutex**

To ensure thread-safe access to shared data, use a mutex.

```cpp
#include <iostream>

#include <fstream>
#include <thread>

#include <vector>
#include <string>

#include <mutex>

std::mutex mtx;

void countLines(const std::string& filename, int& lineCount) {
    std::ifstream file(filename);
    std::string line;
    int count = 0;
    while (std::getline(file, line)) {
        ++count;
    }
    std::lock_guard<std::mutex> lock(mtx);
    lineCount = count;
}

int main() {
    std::vector<std::string> filenames = {"file1.txt", "file2.txt", "file3.txt"};
    std::vector<int> lineCounts(filenames.size());
    std::vector<std::thread> threads;

    for (size_t i = 0; i < filenames.size(); ++i) {
        threads.emplace_back(countLines,

 filenames[i], std::ref(lineCounts[i]));
    }

    for (auto& t : threads) {
        t.join();
    }

    for (size_t i = 0; i < filenames.size(); ++i) {
        std::cout << filenames[i] << ": " << lineCounts[i] << " lines" << std::endl;
    }

    return 0;
}
```

In this optimized example, a mutex ensures that each thread safely updates the line count without causing data corruption or race conditions.

#### **6.1.6 Conclusion**

Multithreading in C++ provides a powerful way to improve the performance and responsiveness of applications by allowing concurrent execution of tasks. Understanding the basics of creating and managing threads, using synchronization mechanisms, and avoiding common multithreading issues is crucial for writing robust and efficient multithreaded code. By leveraging these techniques, you can harness the full potential of modern multi-core processors, making your applications more efficient and responsive. The following sections will delve deeper into advanced concurrency techniques and strategies for optimizing multithreaded performance in C++.



### 6.2 Synchronization and Its Impact on Cache Coherence

Synchronization is essential in multithreaded programs to ensure that multiple threads can safely access shared resources. However, synchronization can significantly impact cache coherence, a critical aspect of system performance. This section explores synchronization mechanisms, their effects on cache coherence, and strategies to mitigate performance issues.

#### **6.2.1 Understanding Synchronization**

Synchronization mechanisms coordinate the access of multiple threads to shared resources, ensuring data consistency and preventing race conditions. Common synchronization tools in C++ include mutexes, condition variables, and atomic operations.

##### **Mutexes**

A mutex (mutual exclusion) ensures that only one thread can access a shared resource at a time.

- **Example**: Using a mutex to protect a shared counter.

    ```cpp
    #include <iostream>
    #include <thread>
    #include <mutex>

    std::mutex mtx;
    int sharedCounter = 0;

    void incrementCounter() {
        for (int i = 0; i < 1000; ++i) {
            std::lock_guard<std::mutex> lock(mtx); // Lock the mutex.
            ++sharedCounter;
        }
    }

    int main() {
        std::thread t1(incrementCounter);
        std::thread t2(incrementCounter);

        t1.join();
        t2.join();

        std::cout << "Final counter value: " << sharedCounter << std::endl;
        return 0;
    }
    ```

##### **Condition Variables**

Condition variables allow threads to wait for certain conditions to be met, facilitating more complex synchronization.

- **Example**: Using a condition variable to synchronize threads.

    ```cpp
    #include <iostream>
    #include <thread>
    #include <mutex>
    #include <condition_variable>

    std::mutex mtx;
    std::condition_variable cv;
    bool ready = false;

    void printMessage() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, []() { return ready; }); // Wait until ready is true.
        std::cout << "Thread is running!" << std::endl;
    }

    void setReady() {
        std::unique_lock<std::mutex> lock(mtx);
        ready = true;
        cv.notify_one(); // Notify one waiting thread.
    }

    int main() {
        std::thread t1(printMessage);
        std::thread t2(setReady);

        t1.join();
        t2.join();

        return 0;
    }
    ```

##### **Atomic Operations**

Atomic operations are indivisible and ensure that operations on shared data are performed without interference from other threads.

- **Example**: Using atomic operations to increment a counter.

    ```cpp
    #include <atomic>

    std::atomic<int> sharedCounter(0);

    void incrementCounter() {
        for (int i = 0; i < 1000; ++i) {
            ++sharedCounter; // Atomic increment.
        }
    }

    int main() {
        std::thread t1(incrementCounter);
        std::thread t2(incrementCounter);

        t1.join();
        t2.join();

        std::cout << "Final counter value: " << sharedCounter.load() << std::endl;
        return 0;
    }
    ```

#### **6.2.2 Cache Coherence and Synchronization**

Cache coherence ensures that all CPU cores have a consistent view of memory. When multiple threads modify shared data, maintaining cache coherence becomes challenging and can impact performance. Synchronization mechanisms play a crucial role in managing cache coherence but can also introduce overhead.

##### **Cache Coherence Protocols**

Cache coherence protocols, such as MESI (Modified, Exclusive, Shared, Invalid), ensure that all caches reflect the most recent value of shared data. When a thread modifies shared data, the protocol invalidates or updates copies of that data in other caches.

- **Example**: When one thread increments a shared counter, the cache coherence protocol ensures that other threads see the updated value by invalidating or updating their cached copies.

##### **Impact of Mutexes on Cache Coherence**

Mutexes can cause frequent cache line invalidations and transfers, impacting performance.

- **Example**: When a thread locks a mutex and modifies shared data, the cache line containing the mutex and data is invalidated in other caches. When the mutex is unlocked, other threads accessing the same data cause further cache coherence traffic.

    ```cpp
    std::mutex mtx;
    int sharedCounter = 0;

    void incrementCounter() {
        for (int i = 0; i < 1000; ++i) {
            std::lock_guard<std::mutex> lock(mtx);
            ++sharedCounter; // Causes cache line invalidations.
        }
    }
    ```

##### **Impact of Atomic Operations on Cache Coherence**

Atomic operations, while ensuring data consistency, can also lead to cache coherence overhead due to frequent cache line transfers.

- **Example**: Incrementing an atomic counter causes the cache line containing the counter to be transferred between cores.

    ```cpp
    std::atomic<int> sharedCounter(0);

    void incrementCounter() {
        for (int i = 0; i < 1000; ++i) {
            ++sharedCounter; // Causes cache line transfers.
        }
    }
    ```

#### **6.2.3 Strategies to Mitigate Synchronization Overhead**

To mitigate the performance impact of synchronization on cache coherence, consider the following strategies:

##### **Reducing Contention**

Minimize contention by reducing the frequency and duration of lock acquisitions.

- **Example**: Use finer-grained locking or lock-free data structures to reduce contention.

    ```cpp
    std::mutex mtx1, mtx2;
    int counter1 = 0, counter2 = 0;

    void incrementCounters() {
        for (int i = 0; i < 1000; ++i) {
            {
                std::lock_guard<std::mutex> lock(mtx1);
                ++counter1;
            }
            {
                std::lock_guard<std::mutex> lock(mtx2);
                ++counter2;
            }
        }
    }
    ```

##### **Using Read-Write Locks**

Read-write locks (`std::shared_mutex`) allow multiple readers but only one writer, reducing contention when reads are more frequent than writes.

- **Example**: Using a read-write lock to protect shared data.

    ```cpp
    #include <shared_mutex>
    std::shared_mutex rw_mtx;
    int sharedData = 0;

    void readData() {
        std::shared_lock<std::shared_mutex> lock(rw_mtx);
        std::cout << "Read data: " << sharedData << std::endl;
    }

    void writeData(int value) {
        std::unique_lock<std::shared_mutex> lock(rw_mtx);
        sharedData = value;
    }

    int main() {
        std::thread t1(readData);
        std::thread t2(writeData, 42);

        t1.join();
        t2.join();

        return 0;
    }
    ```

##### **Using Lock-Free Data Structures**

Lock-free data structures use atomic operations to ensure thread safety without using mutexes, reducing contention and cache coherence overhead.

- **Example**: Using a lock-free queue.

    ```cpp
    #include <atomic>
    #include <memory>
    #include <iostream>

    template <typename T>
    class LockFreeQueue {
    public:
        LockFreeQueue() : head(new Node()), tail(head.load()) {}

        void enqueue(T value) {
            Node* newNode = new Node(value);
            Node* oldTail = tail.load();
            while (!tail.compare_exchange_weak(oldTail, newNode)) {
                oldTail = tail.load();
            }
            oldTail->next.store(newNode);
        }

        bool dequeue(T& result) {
            Node* oldHead = head.load();
            Node* newHead = oldHead->next.load();
            if (newHead == nullptr) return false;
            result = newHead->value;
            head.store(newHead);
            delete oldHead;
            return true;
        }

    private:
        struct Node {
            T value;
            std::atomic<Node*> next;
            Node() : next(nullptr) {}
            Node(T val) : value(val), next(nullptr) {}
        };

        std::atomic<Node*> head;
        std::atomic<Node*> tail;
    };

    int main() {
        LockFreeQueue<int> queue;
        queue.enqueue(1);
        queue.enqueue(2);

        int value;
        if (queue.dequeue(value)) {
            std::cout << "Dequeued: " << value << std::endl;
        }

        return 0;
    }
    ```

#### **6.2.4 Real-Life Example: Multithreaded Data Processing**

Consider a multithreaded application that processes data chunks. Optimizing synchronization can improve performance and reduce cache coherence overhead.

##### **Initial Code**

```cpp
#include <iostream>

#include <thread>
#include <vector>

#include <mutex>

std::mutex mtx;
std::vector<int> data;

void processData(int start, int end) {
    for (int i = start; i < end; ++i) {
        std::lock_guard<std::mutex> lock(mtx);
        data[i] = data[i] * 2;
    }
}

int main() {
    const int dataSize = 1000;
    data.resize(dataSize, 1);

    std::thread t1(processData, 0, dataSize / 2);
    std::thread t2(processData, dataSize / 2, dataSize);

    t1.join();
   

 t2.join();

    for (int i = 0; i < dataSize; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

##### **Optimized Code**

1. **Reduce Contention**: Use finer-grained locking.

    ```cpp
    void processData(int start, int end) {
        for (int i = start; i < end; ++i) {
            data[i] = data[i] * 2; // No locking required.
        }
    }

    int main() {
        const int dataSize = 1000;
        data.resize(dataSize, 1);

        std::thread t1(processData, 0, dataSize / 2);
        std::thread t2(processData, dataSize / 2, dataSize);

        t1.join();
        t2.join();

        for (int i = 0; i < dataSize; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;

        return 0;
    }
    ```

2. **Use Atomic Operations**: Replace mutex with atomic operations if suitable.

    ```cpp
    #include <atomic>

    std::atomic<int> atomicCounter(0);

    void processData(int start, int end) {
        for (int i = start; i < end; ++i) {
            data[i] = data[i] * 2;
            atomicCounter.fetch_add(1, std::memory_order_relaxed);
        }
    }

    int main() {
        const int dataSize = 1000;
        data.resize(dataSize, 1);

        std::thread t1(processData, 0, dataSize / 2);
        std::thread t2(processData, dataSize / 2, dataSize);

        t1.join();
        t2.join();

        std::cout << "Processed elements: " << atomicCounter.load() << std::endl;
        return 0;
    }
    ```

By reducing contention and using atomic operations, we optimize synchronization, reducing cache coherence overhead and improving performance.

#### **6.2.5 Conclusion**

Synchronization is essential in multithreaded programming to ensure data consistency and prevent race conditions. However, synchronization mechanisms can significantly impact cache coherence, leading to performance overhead. Understanding the effects of synchronization on cache coherence and employing strategies to mitigate these effects are crucial for writing efficient multithreaded code. By reducing contention, using read-write locks, and leveraging lock-free data structures, you can improve performance and make better use of modern multi-core processors. The following sections will delve deeper into advanced concurrency techniques and strategies for optimizing multithreaded performance in C++.



### 6.3 Developing Cache-Aware Locking Mechanisms

Efficient locking mechanisms are crucial for ensuring data consistency in multithreaded applications. However, traditional locking techniques can introduce significant overhead and negatively impact cache performance. Developing cache-aware locking mechanisms can mitigate these issues, optimizing both synchronization and cache utilization. This section explores advanced locking techniques designed to minimize cache contention and improve overall system performance.

#### **6.3.1 Understanding Cache Contention in Locking**

Cache contention occurs when multiple threads attempt to access and modify data stored in the same cache line. This can lead to frequent cache invalidations and transfers, severely degrading performance. Traditional mutexes, when used heavily, can exacerbate this problem by causing high levels of contention on the cache lines where the lock and shared data reside.

- **Example**: Consider a scenario where multiple threads frequently update a shared counter protected by a mutex. Each lock and unlock operation results in cache invalidations, leading to significant performance degradation.

#### **6.3.2 Cache-Aware Locking Techniques**

Several advanced locking techniques can help reduce cache contention and improve performance in multithreaded applications.

##### **1. Fine-Grained Locking**

Fine-grained locking involves using multiple locks to protect different parts of shared data. This reduces contention by allowing multiple threads to operate on different data segments simultaneously.

- **Example**: Instead of using a single lock to protect an entire array, use separate locks for each segment of the array.

    ```cpp
    #include <iostream>
    #include <thread>
    #include <vector>
    #include <mutex>

    const int segmentSize = 10;
    std::vector<int> data(100, 0);
    std::vector<std::mutex> locks(data.size() / segmentSize);

    void incrementSegment(int segment) {
        for (int i = 0; i < segmentSize; ++i) {
            int index = segment * segmentSize + i;
            std::lock_guard<std::mutex> lock(locks[segment]);
            ++data[index];
        }
    }

    int main() {
        std::vector<std::thread> threads;
        for (int i = 0; i < data.size() / segmentSize; ++i) {
            threads.emplace_back(incrementSegment, i);
        }

        for (auto& t : threads) {
            t.join();
        }

        for (const auto& value : data) {
            std::cout << value << " ";
        }
        std::cout << std::endl;

        return 0;
    }
    ```

##### **2. Lock Striping**

Lock striping is a technique where a large data structure is divided into smaller stripes, each protected by its own lock. This reduces contention by spreading the locking load across multiple locks.

- **Example**: Applying lock striping to a hash table.

    ```cpp
    #include <iostream>
    #include <vector>
    #include <thread>
    #include <mutex>
    #include <unordered_map>

    const int numStripes = 10;
    std::vector<std::mutex> locks(numStripes);
    std::vector<std::unordered_map<int, int>> hashTable(numStripes);

    int getStripe(int key) {
        return key % numStripes;
    }

    void insert(int key, int value) {
        int stripe = getStripe(key);
        std::lock_guard<std::mutex> lock(locks[stripe]);
        hashTable[stripe][key] = value;
    }

    int get(int key) {
        int stripe = getStripe(key);
        std::lock_guard<std::mutex> lock(locks[stripe]);
        return hashTable[stripe][key];
    }

    int main() {
        std::vector<std::thread> threads;

        for (int i = 0; i < 100; ++i) {
            threads.emplace_back(insert, i, i * 10);
        }

        for (auto& t : threads) {
            t.join();
        }

        for (int i = 0; i < 100; ++i) {
            std::cout << "Key: " << i << " Value: " << get(i) << std::endl;
        }

        return 0;
    }
    ```

##### **3. Cache Line Padding**

Cache line padding involves adding padding to data structures to ensure that each lock resides in its own cache line. This prevents false sharing, where multiple threads inadvertently contend for the same cache line even though they are accessing different variables.

- **Example**: Using cache line padding to prevent false sharing.

    ```cpp
    #include <iostream>
    #include <thread>
    #include <atomic>
    #include <vector>

    struct PaddedAtomic {
        std::atomic<int> value;
        char padding[64 - sizeof(std::atomic<int>)]; // Assuming 64-byte cache lines
    };

    std::vector<PaddedAtomic> counters(10);

    void incrementCounter(int index) {
        for (int i = 0; i < 1000; ++i) {
            ++counters[index].value;
        }
    }

    int main() {
        std::vector<std::thread> threads;

        for (int i = 0; i < counters.size(); ++i) {
            threads.emplace_back(incrementCounter, i);
        }

        for (auto& t : threads) {
            t.join();
        }

        for (const auto& counter : counters) {
            std::cout << counter.value << " ";
        }
        std::cout << std::endl;

        return 0;
    }
    ```

##### **4. Read-Copy-Update (RCU)**

Read-Copy-Update is a synchronization mechanism that allows readers to access data concurrently without locking, while writers create a new copy of the data and update a pointer atomically.

- **Example**: Basic concept of RCU in a read-mostly data structure.

    ```cpp
    #include <iostream>
    #include <thread>
    #include <atomic>
    #include <vector>

    struct Node {
        int value;
        Node* next;
    };

    std::atomic<Node*> head(nullptr);

    void insert(int value) {
        Node* newNode = new Node{value, head.load()};
        while (!head.compare_exchange_weak(newNode->next, newNode));
    }

    void printList() {
        Node* current = head.load();
        while (current) {
            std::cout << current->value << " ";
            current = current->next;
        }
        std::cout << std::endl;
    }

    int main() {
        std::vector<std::thread> threads;

        for (int i = 0; i < 10; ++i) {
            threads.emplace_back(insert, i);
        }

        for (auto& t : threads) {
            t.join();
        }

        printList();

        return 0;
    }
    ```

#### **6.3.3 Real-Life Example: Optimizing a Multithreaded Counter**

Consider a real-life scenario where multiple threads increment a shared counter. Using traditional locking can cause significant cache contention and performance degradation.

##### **Initial Code with Traditional Locking**

```cpp
#include <iostream>

#include <thread>
#include <mutex>

#include <vector>

std::mutex mtx;
int sharedCounter = 0;

void incrementCounter() {
    for (int i = 0; i < 1000; ++i) {
        std::lock_guard<std::mutex> lock(mtx);
        ++sharedCounter;
    }
}

int main() {
    std::vector<std::thread> threads;

    for (int i = 0; i < 10; ++i) {
        threads.emplace_back(incrementCounter);
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Final counter value: " << sharedCounter << std::endl;

    return 0;
}
```

##### **Optimized Code with Cache-Aware Locking**

1. **Using Fine-Grained Locking**

    ```cpp
    #include <iostream>
    #include <thread>
    #include <vector>
    #include <mutex>

    const int numCounters = 10;
    std::vector<int> counters(numCounters, 0);
    std::vector<std::mutex> locks(numCounters);

    void incrementCounter(int index) {
        for (int i = 0; i < 1000; ++i) {
            std::lock_guard<std::mutex> lock(locks[index]);
            ++counters[index];
        }
    }

    int main() {
        std::vector<std::thread> threads;

        for (int i = 0; i < numCounters; ++i) {
            threads.emplace_back(incrementCounter, i);
        }

        for (auto& t : threads) {
            t.join();
        }

        int total = 0;
        for (const auto& count : counters) {
            total += count;
        }

        std::cout << "Final counter value: " << total << std::endl;

        return 0;
    }
    ```

2. **Using Cache Line Padding**

    ```cpp
    #include <iostream>
    #include <thread>
    #include <vector>
    #include <atomic>

    struct PaddedCounter {
        std::atomic<int> value;
        char padding[64 - sizeof(std::atomic<int>)]; // Assuming 64-byte cache lines
    };

    std::vector<PaddedCounter> counters(10);

    void incrementCounter(int index) {
        for (int i = 0; i < 1000; ++i) {   
	        ++counters[index].value;
        }
    }

    int main() {
        std::vector<std::thread> threads;

        for (int i = 0; i < counters.size(); ++i) {
            threads.emplace_back(incrementCounter, i);
        }

        for (auto& t : threads) {
            t.join();
        }

        int total = 0;
        for (const auto& counter : counters) {
            total += counter.value.load();
        }

        std::cout << "Final counter value: " << total << std::endl;

        return 0;
    }
    ```

By using fine-grained locking and cache line padding, we can significantly reduce cache contention, improving the performance and efficiency of the multithreaded counter.

#### **6.3.4 Conclusion**

Developing cache-aware locking mechanisms is essential for optimizing multithreaded applications. By reducing cache contention through techniques such as fine-grained locking, lock striping, cache line padding, and lock-free data structures, you can enhance both synchronization and cache utilization. These optimizations lead to better performance and scalability in modern multi-core systems, making your applications more efficient and responsive. The following sections will explore additional advanced concurrency techniques, providing a comprehensive guide to mastering multithreading and concurrency in C++.