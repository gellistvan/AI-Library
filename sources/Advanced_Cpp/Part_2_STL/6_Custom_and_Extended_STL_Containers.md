
\newpage
# Part II: STL
\newpage

## Chapter 6: Custom and Extended STL Containers

In this chapter, we delve into the realm of Custom and Extended Standard Template Library (STL) Containers, an essential aspect of advanced C++ programming. The STL provides a rich collection of containers such as vectors, lists, sets, and maps that cater to most general-purpose needs. However, as you progress to more complex and performance-critical applications, the need to customize these containers or create entirely new ones often arises.

Understanding how to effectively extend the STL involves mastering allocator design, iterators, and container interfaces. We'll explore how to design custom allocators that optimize memory usage and improve performance for specific use cases. You'll learn how to implement your own iterators, ensuring compatibility with STL algorithms and enhancing the versatility of your containers.

Furthermore, we'll cover the principles of creating completely new containers from scratch. This includes defining the necessary interfaces, ensuring exception safety, and achieving optimal time and space complexities. We'll also touch on integrating these custom containers with existing STL algorithms and how to leverage C++17 and C++20 features to simplify and enhance your implementations.

By the end of this chapter, you will have a deep understanding of how to extend and customize STL containers, allowing you to tackle complex programming challenges with confidence and precision. This knowledge is crucial for building high-performance applications and contributing to the development of robust and efficient C++ codebases.

### 6.1. Custom Allocators

Allocators in C++ play a crucial role in managing memory for container classes. The default allocator provided by the Standard Template Library (STL) is suitable for general-purpose use; however, for performance-critical applications or specialized memory management requirements, creating custom allocators can provide significant benefits. In this subchapter, we will explore the principles and implementation of custom allocators in C++.

#### Understanding Allocators

Allocators abstract the process of allocating and deallocating memory for containers. An allocator must fulfill specific requirements and provide a well-defined interface. The primary components of an allocator include:

1. **Type Definitions**: Define various types used by the allocator, such as `value_type`, `pointer`, `const_pointer`, `reference`, `const_reference`, `size_type`, and `difference_type`.
2. **Member Functions**: Implement member functions for memory allocation (`allocate`), deallocation (`deallocate`), and construction (`construct`) and destruction (`destroy`) of objects.

#### Basic Allocator Structure

Let’s start by defining a simple custom allocator. This allocator will use the default `new` and `delete` operators for memory management.

```cpp
#include <memory>
#include <iostream>

template <typename T>
class SimpleAllocator {
public:
    using value_type = T;

    SimpleAllocator() noexcept {}
    template <typename U>
    SimpleAllocator(const SimpleAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        std::cout << "Allocating " << n * sizeof(T) << " bytes" << std::endl;
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }

    void deallocate(T* p, std::size_t n) noexcept {
        std::cout << "Deallocating " << n * sizeof(T) << " bytes" << std::endl;
        ::operator delete(p);
    }

    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        new (p) U(std::forward<Args>(args)...);
    }

    template <typename U>
    void destroy(U* p) noexcept {
        p->~U();
    }
};

template <typename T, typename U>
bool operator==(const SimpleAllocator<T>&, const SimpleAllocator<U>&) noexcept {
    return true;
}

template <typename T, typename U>
bool operator!=(const SimpleAllocator<T>&, const SimpleAllocator<U>&) noexcept {
    return false;
}
```

#### Using the Custom Allocator with STL Containers

To use the `SimpleAllocator` with STL containers, simply specify it as the allocator type in the container definition. Here’s an example with `std::vector`:

```cpp
#include <vector>

int main() {
    std::vector<int, SimpleAllocator<int>> vec;

    for (int i = 0; i < 10; ++i) {
        vec.push_back(i);
    }

    for (const auto& val : vec) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

#### Advanced Allocator: Pool Allocator

For more sophisticated memory management, consider a pool allocator. Pool allocators preallocate a large block of memory and then dish out small chunks as needed, which can significantly reduce the overhead associated with frequent allocations and deallocations.

Here is an implementation of a simple pool allocator:

```cpp
#include <cstddef>
#include <vector>

template <typename T>
class PoolAllocator {
public:
    using value_type = T;

    PoolAllocator() noexcept : pool(nullptr), pool_size(0), free_list(nullptr) {}
    template <typename U>
    PoolAllocator(const PoolAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n != 1) throw std::bad_alloc();

        if (!free_list) {
            allocate_pool();
        }

        T* result = reinterpret_cast<T*>(free_list);
        free_list = free_list->next;

        return result;
    }

    void deallocate(T* p, std::size_t n) noexcept {
        if (n != 1) return;

        reinterpret_cast<Node*>(p)->next = free_list;
        free_list = reinterpret_cast<Node*>(p);
    }

    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        new (p) U(std::forward<Args>(args)...);
    }

    template <typename U>
    void destroy(U* p) noexcept {
        p->~U();
    }

private:
    struct Node {
        Node* next;
    };

    void allocate_pool() {
        pool_size = 1024;
        pool = ::operator new(pool_size * sizeof(T));
        free_list = reinterpret_cast<Node*>(pool);

        Node* current = free_list;
        for (std::size_t i = 1; i < pool_size; ++i) {
            current->next = reinterpret_cast<Node*>(reinterpret_cast<char*>(pool) + i * sizeof(T));
            current = current->next;
        }
        current->next = nullptr;
    }

    void* pool;
    std::size_t pool_size;
    Node* free_list;
};

template <typename T, typename U>
bool operator==(const PoolAllocator<T>&, const PoolAllocator<U>&) noexcept {
    return true;
}

template <typename T, typename U>
bool operator!=(const PoolAllocator<T>&, const PoolAllocator<U>&) noexcept {
    return false;
}
```

#### Using the Pool Allocator

You can use the `PoolAllocator` in the same way as the `SimpleAllocator`:

```cpp
int main() {
    std::vector<int, PoolAllocator<int>> vec;

    for (int i = 0; i < 10; ++i) {
        vec.push_back(i);
    }

    for (const auto& val : vec) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

#### Custom Allocators for Performance

Custom allocators can significantly improve performance, particularly in scenarios involving frequent allocations and deallocations of small objects. By tailoring memory management strategies to specific use cases, custom allocators reduce overhead and increase the efficiency of memory usage.

#### Conclusion

Custom allocators provide a powerful mechanism for optimizing memory management in C++ programs. By understanding and implementing custom allocators, you can fine-tune the performance characteristics of your applications, making them more efficient and responsive. Whether you are using a simple allocator for educational purposes or a sophisticated pool allocator for performance-critical applications, mastering custom allocators is an essential skill for advanced C++ programmers.

### 6.2. Extending STL Containers

While the Standard Template Library (STL) offers a robust set of containers that meet most general-purpose needs, there are times when the built-in functionality may fall short of specific requirements. Extending STL containers allows you to add custom behavior and functionality while still leveraging the powerful features of the STL. This subchapter will explore techniques for extending STL containers, including subclassing, traits, and custom iterators.

#### Subclassing STL Containers

One of the simplest ways to extend STL containers is by subclassing. This method allows you to inherit from an STL container and add new member functions or override existing ones. Here's an example of extending `std::vector` to add a `print` method:

```cpp
#include <vector>
#include <iostream>

template <typename T>
class ExtendedVector : public std::vector<T> {
public:
    using std::vector<T>::vector;

    void print() const {
        for (const auto& elem : *this) {
            std::cout << elem << ' ';
        }
        std::cout << std::endl;
    }
};

int main() {
    ExtendedVector<int> ev = {1, 2, 3, 4, 5};
    ev.print();
    return 0;
}
```

#### Custom Traits

Another approach to extending STL containers involves using custom traits. Traits are a compile-time mechanism to define properties and behaviors of types. For example, if you want to define a custom container that behaves differently based on whether its elements are integral or floating-point types, you can use type traits.

```cpp
#include <type_traits>
#include <iostream>

template <typename T>
class CustomContainer {
public:
    void process(const T& value) {
        if constexpr (std::is_integral_v<T>) {
            std::cout << "Processing integral value: " << value << std::endl;
        } else if constexpr (std::is_floating_point_v<T>) {
            std::cout << "Processing floating-point value: " << value << std::endl;
        } else {
            std::cout << "Processing unknown type" << std::endl;
        }
    }
};

int main() {
    CustomContainer<int> intContainer;
    intContainer.process(42);

    CustomContainer<double> doubleContainer;
    doubleContainer.process(3.14);

    return 0;
}
```

#### Custom Iterators

Custom iterators are a powerful way to extend the functionality of STL containers. By creating custom iterators, you can add new traversal mechanisms or enhance existing ones. Here's an example of a custom iterator that iterates over elements in reverse order:

```cpp
#include <iterator>
#include <vector>
#include <iostream>

template <typename T>
class ReverseIterator {
public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;

    ReverseIterator(pointer ptr) : current(ptr) {}

    reference operator*() const { return *current; }
    pointer operator->() { return current; }

    // Prefix increment
    ReverseIterator& operator++() {
        --current;
        return *this;
    }

    // Postfix increment
    ReverseIterator operator++(int) {
        ReverseIterator tmp = *this;
        --current;
        return tmp;
    }

    friend bool operator==(const ReverseIterator& a, const ReverseIterator& b) {
        return a.current == b.current;
    }

    friend bool operator!=(const ReverseIterator& a, const ReverseIterator& b) {
        return a.current != b.current;
    }

private:
    pointer current;
};

template <typename T>
class ReversibleVector : public std::vector<T> {
public:
    using std::vector<T>::vector;

    ReverseIterator<T> rbegin() {
        return ReverseIterator<T>(this->data() + this->size() - 1);
    }

    ReverseIterator<T> rend() {
        return ReverseIterator<T>(this->data() - 1);
    }
};

int main() {
    ReversibleVector<int> rv = {1, 2, 3, 4, 5};

    for (auto it = rv.rbegin(); it != rv.rend(); ++it) {
        std::cout << *it << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

#### Policy-Based Design

Policy-based design is another powerful technique for extending STL containers. It allows you to define flexible, reusable policies that can be combined to customize container behavior. For instance, you can define allocation, sorting, and access policies separately and then compose them into a custom container.

Here's an example of a policy-based design for a custom vector:

```cpp
#include <vector>
#include <iostream>

template <typename T>
class DefaultAllocPolicy {
public:
    using value_type = T;

    T* allocate(std::size_t n) {
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }

    void deallocate(T* p, std::size_t n) {
        ::operator delete(p);
    }
};

template <typename T>
class NoSortPolicy {
public:
    void sort(std::vector<T>&) {
        // No sorting performed
    }
};

template <typename T>
class AscendingSortPolicy {
public:
    void sort(std::vector<T>& container) {
        std::sort(container.begin(), container.end());
    }
};

template <typename T, template <typename> class AllocPolicy = DefaultAllocPolicy, template <typename> class SortPolicy = NoSortPolicy>
class CustomVector : private AllocPolicy<T>, private SortPolicy<T> {
public:
    using value_type = T;

    CustomVector() : data(nullptr), size(0), capacity(0) {}

    ~CustomVector() {
        clear();
        AllocPolicy<T>::deallocate(data, capacity);
    }

    void push_back(const T& value) {
        if (size == capacity) {
            resize();
        }
        data[size++] = value;
    }

    void sort() {
        SortPolicy<T>::sort(*this);
    }

    void print() const {
        for (std::size_t i = 0; i < size; ++i) {
            std::cout << data[i] << ' ';
        }
        std::cout << std::endl;
    }

private:
    T* data;
    std::size_t size;
    std::size_t capacity;

    void resize() {
        std::size_t new_capacity = capacity == 0 ? 1 : capacity * 2;
        T* new_data = AllocPolicy<T>::allocate(new_capacity);

        for (std::size_t i = 0; i < size; ++i) {
            new_data[i] = data[i];
        }

        AllocPolicy<T>::deallocate(data, capacity);
        data = new_data;
        capacity = new_capacity;
    }

    void clear() {
        for (std::size_t i = 0; i < size; ++i) {
            data[i].~T();
        }
        size = 0;
    }
};

int main() {
    CustomVector<int, DefaultAllocPolicy, AscendingSortPolicy> vec;

    vec.push_back(3);
    vec.push_back(1);
    vec.push_back(4);
    vec.push_back(1);
    vec.push_back(5);

    vec.print();
    vec.sort();
    vec.print();

    return 0;
}
```

#### Customizing Container Interfaces

Sometimes, you may need to create completely new containers with interfaces that better suit your specific needs. Here’s an example of a ring buffer, a circular queue that is efficient for fixed-size buffer implementations:

```cpp
#include <iostream>
#include <stdexcept>

template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(std::size_t capacity)
        : data(new T[capacity]), capacity(capacity), size(0), front(0), back(0) {}

    ~RingBuffer() {
        delete[] data;
    }

    void push_back(const T& value) {
        if (size == capacity) {
            throw std::overflow_error("Ring buffer overflow");
        }
        data[back] = value;
        back = (back + 1) % capacity;
        ++size;
    }

    void pop_front() {
        if (size == 0) {
            throw std::underflow_error("Ring buffer underflow");
        }
        front = (front + 1) % capacity;
        --size;
    }

    const T& front_value() const {
        if (size == 0) {
            throw std::underflow_error("Ring buffer is empty");
        }
        return data[front];
    }

    bool empty() const {
        return size == 0;
    }

    bool full() const {
        return size == capacity;
    }

    std::size_t get_size() const {
        return size;
    }

    void print() const {
        for (std::size_t i = 0; i < size; ++i) {
            std::cout << data[(front + i) % capacity] << ' ';
        }
        std::cout << std::endl;
    }

private:
    T* data;
    std::size_t capacity;
    std::size_t size;
    std::size_t front;
    std::size_t back;
};

int main() {
    RingBuffer<int> rb(5);

    rb.push_back(1);
    rb.push_back(2);
    rb.push_back(3);
    rb.push_back(4);
    rb.push_back(5);

    rb.print();

    rb.pop_front();
    rb.pop_front();

    rb.print();

    rb.push_back(6);
    rb.push_back(7);

    rb.print();

    return 0;
}
```

#### Conclusion

Extending STL containers is a powerful technique for customizing the behavior and functionality of your data structures to better meet specific requirements. Whether through subclassing, custom traits, custom iterators, policy-based design, or creating entirely new containers, mastering these techniques enables you to leverage the full power and flexibility of C++. By doing so, you can create more efficient, maintainable, and feature-rich applications.

### 6.3. Creating Custom Containers

Creating custom containers from scratch is an essential skill for advanced C++ programmers. While the STL provides a wide array of containers, there are situations where bespoke containers can offer more tailored performance, functionality, or interface guarantees. This subchapter delves into the principles of designing and implementing custom containers, covering aspects such as interface design, memory management, iterator support, and integration with STL algorithms.

#### Principles of Custom Container Design

When designing a custom container, consider the following principles:

1. **Interface Design**: Define the container’s public interface, including constructors, destructors, member functions, and operator overloads.
2. **Memory Management**: Implement efficient memory management, including allocation, deallocation, and object construction and destruction.
3. **Iterator Support**: Provide iterators to enable range-based for loops and compatibility with STL algorithms.
4. **Exception Safety**: Ensure your container handles exceptions gracefully, maintaining a consistent state even in the presence of errors.
5. **Performance**: Optimize for both time and space complexity to ensure the container meets performance requirements.

#### Example: A Simple Dynamic Array

Let's start by creating a simple dynamic array, akin to `std::vector`, but with a focus on understanding the fundamental building blocks.

##### Interface Design

First, define the interface for the dynamic array. The interface includes constructors, a destructor, and member functions for adding, removing, and accessing elements.

```cpp
#include <iostream>
#include <stdexcept>

template <typename T>
class DynamicArray {
public:
    DynamicArray();
    explicit DynamicArray(std::size_t initial_capacity);
    ~DynamicArray();

    void push_back(const T& value);
    void pop_back();
    T& operator[](std::size_t index);
    const T& operator[](std::size_t index) const;
    std::size_t size() const;
    bool empty() const;

private:
    T* data;
    std::size_t capacity;
    std::size_t length;

    void resize(std::size_t new_capacity);
};
```

##### Memory Management

Implement memory management functions, including the constructor, destructor, and resize method.

```cpp
template <typename T>
DynamicArray<T>::DynamicArray() : data(nullptr), capacity(0), length(0) {}

template <typename T>
DynamicArray<T>::DynamicArray(std::size_t initial_capacity)
    : data(new T[initial_capacity]), capacity(initial_capacity), length(0) {}

template <typename T>
DynamicArray<T>::~DynamicArray() {
    delete[] data;
}

template <typename T>
void DynamicArray<T>::resize(std::size_t new_capacity) {
    T* new_data = new T[new_capacity];
    for (std::size_t i = 0; i < length; ++i) {
        new_data[i] = std::move(data[i]);
    }
    delete[] data;
    data = new_data;
    capacity = new_capacity;
}
```

##### Adding and Removing Elements

Implement functions to add (`push_back`) and remove (`pop_back`) elements, as well as to access elements (`operator[]`).

```cpp
template <typename T>
void DynamicArray<T>::push_back(const T& value) {
    if (length == capacity) {
        resize(capacity == 0 ? 1 : capacity * 2);
    }
    data[length++] = value;
}

template <typename T>
void DynamicArray<T>::pop_back() {
    if (length == 0) {
        throw std::out_of_range("Array is empty");
    }
    --length;
}

template <typename T>
T& DynamicArray<T>::operator[](std::size_t index) {
    if (index >= length) {
        throw std::out_of_range("Index out of range");
    }
    return data[index];
}

template <typename T>
const T& DynamicArray<T>::operator[](std::size_t index) const {
    if (index >= length) {
        throw std::out_of_range("Index out of range");
    }
    return data[index];
}

template <typename T>
std::size_t DynamicArray<T>::size() const {
    return length;
}

template <typename T>
bool DynamicArray<T>::empty() const {
    return length == 0;
}
```

#### Iterators

To make the custom container compatible with STL algorithms and range-based for loops, we need to implement iterators.

```cpp
template <typename T>
class DynamicArrayIterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;

    DynamicArrayIterator(pointer ptr) : current(ptr) {}

    reference operator*() const { return *current; }
    pointer operator->() { return current; }

    // Prefix increment
    DynamicArrayIterator& operator++() {
        ++current;
        return *this;
    }

    // Postfix increment
    DynamicArrayIterator operator++(int) {
        DynamicArrayIterator tmp = *this;
        ++current;
        return tmp;
    }

    friend bool operator==(const DynamicArrayIterator& a, const DynamicArrayIterator& b) {
        return a.current == b.current;
    }

    friend bool operator!=(const DynamicArrayIterator& a, const DynamicArrayIterator& b) {
        return a.current != b.current;
    }

private:
    pointer current;
};

template <typename T>
DynamicArrayIterator<T> begin(DynamicArray<T>& array) {
    return DynamicArrayIterator<T>(array.data);
}

template <typename T>
DynamicArrayIterator<T> end(DynamicArray<T>& array) {
    return DynamicArrayIterator<T>(array.data + array.size());
}
```

#### Full Example

Combining all the pieces, here's the full implementation of the `DynamicArray` class along with its iterators.

```cpp
#include <iostream>
#include <stdexcept>
#include <iterator>

template <typename T>
class DynamicArray {
public:
    DynamicArray();
    explicit DynamicArray(std::size_t initial_capacity);
    ~DynamicArray();

    void push_back(const T& value);
    void pop_back();
    T& operator[](std::size_t index);
    const T& operator[](std::size_t index) const;
    std::size_t size() const;
    bool empty() const;

    using iterator = DynamicArrayIterator<T>;
    using const_iterator = DynamicArrayIterator<const T>;

    iterator begin() { return iterator(data); }
    iterator end() { return iterator(data + length); }
    const_iterator begin() const { return const_iterator(data); }
    const_iterator end() const { return const_iterator(data + length); }

private:
    T* data;
    std::size_t capacity;
    std::size_t length;

    void resize(std::size_t new_capacity);
};

template <typename T>
DynamicArray<T>::DynamicArray() : data(nullptr), capacity(0), length(0) {}

template <typename T>
DynamicArray<T>::DynamicArray(std::size_t initial_capacity)
    : data(new T[initial_capacity]), capacity(initial_capacity), length(0) {}

template <typename T>
DynamicArray<T>::~DynamicArray() {
    delete[] data;
}

template <typename T>
void DynamicArray<T>::resize(std::size_t new_capacity) {
    T* new_data = new T[new_capacity];
    for (std::size_t i = 0; i < length; ++i) {
        new_data[i] = std::move(data[i]);
    }
    delete[] data;
    data = new_data;
    capacity = new_capacity;
}

template <typename T>
void DynamicArray<T>::push_back(const T& value) {
    if (length == capacity) {
        resize(capacity == 0 ? 1 : capacity * 2);
    }
    data[length++] = value;
}

template <typename T>
void DynamicArray<T>::pop_back() {
    if (length == 0) {
        throw std::out_of_range("Array is empty");
    }
    --length;
}

template <typename T>
T& DynamicArray<T>::operator[](std::size_t index) {
    if (index >= length) {
        throw std::out_of_range("Index out of range");
    }
    return data[index];
}

template <typename T>
const T& DynamicArray<T>::operator[](std::size_t index) const {
    if (index >= length) {
        throw std::out_of_range("Index out of range");
    }
    return data[index];
}

template <typename T>
std::size_t DynamicArray<T>::size() const {
    return length;
}

template <typename T>
bool DynamicArray<T>::empty() const {
    return length == 0;
}

template <typename T>
class DynamicArrayIterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;

    DynamicArrayIterator(pointer ptr) : current(ptr) {}

    reference operator*() const { return *current; }
    pointer operator->() { return current; }

    DynamicArrayIterator& operator++() {
        ++current;
        return *this;
    }

    DynamicArrayIterator operator++(int) {
        DynamicArrayIterator tmp = *this;
        ++current;
        return tmp;
    }

    friend bool operator==(const DynamicArrayIterator& a, const DynamicArrayIterator& b) {
        return a.current == b.current;
    }

    friend bool operator!=(const DynamicArrayIterator& a, const DynamicArrayIterator& b) {
        return a.current != b.current;
    }

private:
    pointer current;
};

int main() {
    DynamicArray<int> da;

    for (int i = 0; i < 10; ++i) {
        da.push_back(i);
    }

    for (auto it = da.begin(); it != da.end(); ++it) {
        std::cout << *it << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

#### Advanced Custom Container: A Hash Table

To illustrate a more complex custom container, let’s implement a simple hash table. A hash table provides efficient average-case time complexity for search, insertion, and deletion operations.

##### Interface Design

Define the interface for the hash table. This includes functions for insertion, deletion, and searching for elements.

```cpp
#include <vector>
#include <list>
#include <functional>
#include <iostream>
#include <stdexcept>

template <typename Key, typename Value>
class HashTable {
public:
    explicit HashTable(std::size_t bucket_count = 16);

    void insert(const Key& key, const Value& value);
    void remove(const Key& key);
    Value& get(const Key& key);
    const Value& get(const Key& key) const;
    bool contains(const Key& key) const;

private:
    std::vector<std::list<std::pair<Key, Value>>> buckets;
    std::hash<Key> hash_function;

    std::size_t get_bucket_index(const Key& key) const;
};
```

##### Memory Management and Operations

Implement the functions to manage memory and perform operations on the hash table.

```cpp
template <typename Key, typename Value>
HashTable<Key, Value>::HashTable(std::size_t bucket_count) : buckets(bucket_count) {}

template <typename Key, typename Value>
std::size_t HashTable<Key, Value>::get_bucket_index(const Key& key) const {
    return hash_function(key) % buckets.size();
}

template <typename Key, typename Value>
void HashTable<Key, Value>::insert(const Key& key, const Value& value) {
    std::size_t bucket_index = get_bucket_index(key);
    for (auto& pair : buckets[bucket_index]) {
        if (pair.first == key) {
            pair.second = value;
            return;
        }
    }
    buckets[bucket_index].emplace_back(key, value);
}

template <typename Key, typename Value>
void HashTable<Key, Value>::remove(const Key& key) {
    std::size_t bucket_index = get_bucket_index(key);
    auto& bucket = buckets[bucket_index];
    for (auto it = bucket.begin(); it != bucket.end(); ++it) {
        if (it->first == key) {
            bucket.erase(it);
            return;
        }
    }
    throw std::out_of_range("Key not found");
}

template <typename Key, typename Value>
Value& HashTable<Key, Value>::get(const Key& key) {
    std::size_t bucket_index = get_bucket_index(key);
    for (auto& pair : buckets[bucket_index]) {
        if (pair.first == key) {
            return pair.second;
        }
    }
    throw std::out_of_range("Key not found");
}

template <typename Key, typename Value>
const Value& HashTable<Key, Value>::get(const Key& key) const {
    std::size_t bucket_index = get_bucket_index(key);
    for (const auto& pair : buckets[bucket_index]) {
        if (pair.first == key) {
            return pair.second;
        }
    }
    throw std::out_of_range("Key not found");
}

template <typename Key, typename Value>
bool HashTable<Key, Value>::contains(const Key& key) const {
    std::size_t bucket_index = get_bucket_index(key);
    for (const auto& pair : buckets[bucket_index]) {
        if (pair.first == key) {
            return true;
        }
    }
    return false;
}
```

#### Full Example

Combining all the pieces, here's the full implementation of the `HashTable` class.

```cpp
#include <vector>
#include <list>
#include <functional>
#include <iostream>
#include <stdexcept>

template <typename Key, typename Value>
class HashTable {
public:
    explicit HashTable(std::size_t bucket_count = 16);

    void insert(const Key& key, const Value& value);
    void remove(const Key& key);
    Value& get(const Key& key);
    const Value& get(const Key& key) const;
    bool contains(const Key& key) const;

private:
    std::vector<std::list<std::pair<Key, Value>>> buckets;
    std::hash<Key> hash_function;

    std::size_t get_bucket_index(const Key& key) const;
};

template <typename Key, typename Value>
HashTable<Key, Value>::HashTable(std::size_t bucket_count) : buckets(bucket_count) {}

template <typename Key, typename Value>
std::size_t HashTable<Key, Value>::get_bucket_index(const Key& key) const {
    return hash_function(key) % buckets.size();
}

template <typename Key, typename Value>
void HashTable<Key, Value>::insert(const Key& key, const Value& value) {
    std::size_t bucket_index = get_bucket_index(key);
    for (auto& pair : buckets[bucket_index]) {
        if (pair.first == key) {
            pair.second = value;
            return;
        }
    }
    buckets[bucket_index].emplace_back(key, value);
}

template <typename Key, typename Value>
void HashTable<Key, Value>::remove(const Key& key) {
    std::size_t bucket_index = get_bucket_index(key);
    auto& bucket = buckets[bucket_index];
    for (auto it = bucket.begin(); it != bucket.end(); ++it) {
        if (it->first == key) {
            bucket.erase(it);
            return;
        }
    }
    throw std::out_of_range("Key not found");
}

template <typename Key, typename Value>
Value& HashTable<Key, Value>::get(const Key& key) {
    std::size_t bucket_index = get_bucket_index(key);
    for (auto& pair : buckets[bucket_index]) {
        if (pair.first == key) {
            return pair.second;
        }
    }
    throw std::out_of_range("Key not found");
}

template <typename Key, typename Value>
const Value& HashTable<Key, Value>::get(const Key& key) const {
    std::size_t bucket_index = get_bucket_index(key);
    for (const auto& pair : buckets[bucket_index]) {
        if (pair.first == key) {
            return pair.second;
        }
    }
    throw std::out_of_range("Key not found");
}

template <typename Key, typename Value>
bool HashTable<Key, Value>::contains(const Key& key) const {
    std::size_t bucket_index = get_bucket_index(key);
    for (const auto& pair : buckets[bucket_index]) {
        if (pair.first == key) {
            return true;
        }
    }
    return false;
}

int main() {
    HashTable<std::string, int> ht;

    ht.insert("one", 1);
    ht.insert("two", 2);
    ht.insert("three", 3);

    std::cout << "one: " << ht.get("one") << std::endl;
    std::cout << "two: " << ht.get("two") << std::endl;
    std::cout << "three: " << ht.get("three") << std::endl;

    ht.remove("two");

    if (!ht.contains("two")) {
        std::cout << "Key 'two' successfully removed" << std::endl;
    }

    return 0;
}
```

#### Conclusion

Creating custom containers involves a deep understanding of C++ and its memory management, iterator, and template mechanisms. By designing custom containers, you can address specific performance, functionality, and interface needs that the STL might not cover. Through careful design, memory management, and implementation of iterators, you can create robust and efficient custom containers that integrate seamlessly with the rest of the C++ standard library. This chapter has provided you with the foundation to start developing your own custom containers, enhancing your ability to tackle complex programming challenges.
