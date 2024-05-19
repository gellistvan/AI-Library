
\newpage
## Chapter 7: Iterator Mastery

Iterators are a fundamental component of C++ programming, serving as the glue between containers and algorithms in the Standard Template Library (STL). They provide a uniform interface for traversing elements in a container, enabling the creation of generic algorithms that work with any iterable data structure. Mastering iterators is crucial for writing efficient, flexible, and reusable code.

In this chapter, we will explore the intricacies of iterators, starting with the basics and moving towards advanced concepts. We'll begin by understanding the different types of iterators—input, output, forward, bidirectional, and random access—and their respective use cases. Each type of iterator offers unique capabilities and performance characteristics, making it essential to choose the right one for your needs.

We'll then delve into custom iterators, learning how to create our own iterators to extend the functionality of existing containers or to support new data structures. This includes implementing iterator traits, ensuring compatibility with STL algorithms, and handling edge cases to maintain robustness.

Additionally, we'll cover iterator adaptors, which modify the behavior of existing iterators to provide additional functionality. Examples include reverse iterators, which traverse containers in reverse order, and insert iterators, which facilitate the insertion of elements during iteration.

By the end of this chapter, you will have a deep understanding of iterators and how to leverage them to write elegant and efficient C++ code. You'll be equipped with the skills to implement custom iterators, adapt existing ones, and integrate them seamlessly with STL algorithms, enhancing your ability to solve complex programming challenges.

### 7.1. Iterator Categories and Hierarchies

Iterators are a cornerstone of C++ programming, providing a uniform interface for traversing elements in containers. They enable the creation of generic algorithms that work with various data structures. Understanding the different categories of iterators and their hierarchical relationships is essential for effective C++ programming. This subchapter will explore the five primary iterator categories, their characteristics, and appropriate use cases.

#### Iterator Categories

The five primary iterator categories defined by the C++ standard are:

1. **Input Iterators**
2. **Output Iterators**
3. **Forward Iterators**
4. **Bidirectional Iterators**
5. **Random Access Iterators**

Each category builds upon the capabilities of the previous ones, forming a hierarchy of iterator types.

#### 1. Input Iterators

Input iterators are used for reading data from a sequence. They support single-pass algorithms, meaning that once an element has been read, it cannot be read again.

##### Characteristics:
- Can be incremented (++it or it++)
- Can be dereferenced to read the value (*it)
- Support equality and inequality comparisons (it == end, it != end)

##### Example: Reading from an Input Iterator

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    auto it = vec.begin();

    while (it != vec.end()) {
        std::cout << *it << ' ';
        ++it;
    }
    std::cout << std::endl;

    return 0;
}
```

#### 2. Output Iterators

Output iterators are used for writing data to a sequence. They also support single-pass algorithms, meaning that once an element has been written, it cannot be overwritten using the same iterator.

##### Characteristics:
- Can be incremented (++it or it++)
- Can be dereferenced to write a value (*it = value)
- Do not support reading from the iterator

##### Example: Writing to an Output Iterator

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec(5);
    auto it = vec.begin();

    for (int i = 1; i <= 5; ++i) {
        *it = i;
        ++it;
    }

    for (const auto& val : vec) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

#### 3. Forward Iterators

Forward iterators combine the capabilities of input and output iterators. They support multi-pass algorithms, allowing multiple reads and writes to the same elements.

##### Characteristics:
- Can be incremented (++it or it++)
- Can be dereferenced to read and write values (*it, *it = value)
- Support equality and inequality comparisons (it == end, it != end)
- Allow multi-pass algorithms

##### Example: Using a Forward Iterator

```cpp
#include <iostream>
#include <forward_list>

int main() {
    std::forward_list<int> flist = {1, 2, 3, 4, 5};
    auto it = flist.begin();

    while (it != flist.end()) {
        std::cout << *it << ' ';
        ++it;
    }
    std::cout << std::endl;

    return 0;
}
```

#### 4. Bidirectional Iterators

Bidirectional iterators extend forward iterators by allowing movement in both directions. They can be incremented and decremented, making them suitable for algorithms that require traversing a sequence in reverse.

##### Characteristics:
- Can be incremented (++it or it++)
- Can be decremented (--it or it--)
- Can be dereferenced to read and write values (*it, *it = value)
- Support equality and inequality comparisons (it == end, it != end)

##### Example: Using a Bidirectional Iterator

```cpp
#include <iostream>
#include <list>

int main() {
    std::list<int> lst = {1, 2, 3, 4, 5};
    auto it = lst.rbegin();  // Reverse iterator

    while (it != lst.rend()) {
        std::cout << *it << ' ';
        ++it;
    }
    std::cout << std::endl;

    return 0;
}
```

#### 5. Random Access Iterators

Random access iterators provide the most functionality. They allow movement to any position within a sequence in constant time, supporting both arithmetic and comparison operations.

##### Characteristics:
- Can be incremented (++it or it++)
- Can be decremented (--it or it--)
- Can be dereferenced to read and write values (*it, *it = value)
- Support equality and inequality comparisons (it == end, it != end)
- Support arithmetic operations (it + n, it - n, it += n, it -= n)
- Support random access (it[n])

##### Example: Using a Random Access Iterator

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // Forward iteration
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        std::cout << *it << ' ';
    }
    std::cout << std::endl;

    // Reverse iteration
    for (auto it = vec.rbegin(); it != vec.rend(); ++it) {
        std::cout << *it << ' ';
    }
    std::cout << std::endl;

    // Random access
    std::cout << "Element at index 2: " << vec[2] << std::endl;

    return 0;
}
```

#### Iterator Traits

Iterator traits provide a standardized way to access properties of iterators at compile time. They are essential for writing generic algorithms that work with different types of iterators. The `std::iterator_traits` template class provides the following type aliases:

- `difference_type`: Type used to represent the distance between iterators.
- `value_type`: Type of the elements pointed to by the iterator.
- `pointer`: Type of a pointer to an element.
- `reference`: Type of a reference to an element.
- `iterator_category`: Iterator category (e.g., input_iterator_tag, output_iterator_tag).

##### Example: Using Iterator Traits

```cpp
#include <iostream>
#include <iterator>
#include <vector>

template <typename Iterator>
void print_iterator_info(Iterator it) {
    using traits = std::iterator_traits<Iterator>;
    std::cout << "Value type: " << typeid(typename traits::value_type).name() << std::endl;
    std::cout << "Difference type: " << typeid(typename traits::difference_type).name() << std::endl;
    std::cout << "Pointer type: " << typeid(typename traits::pointer).name() << std::endl;
    std::cout << "Reference type: " << typeid(typename traits::reference).name() << std::endl;
    std::cout << "Iterator category: " << typeid(typename traits::iterator_category).name() << std::endl;
}

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    auto it = vec.begin();

    print_iterator_info(it);

    return 0;
}
```

#### Custom Iterator Example

To illustrate how to implement a custom iterator, let's create an iterator for a custom container. Here’s a simple custom container and its corresponding iterator.

##### Custom Container

```cpp
#include <iostream>

template <typename T>
class SimpleContainer {
public:
    SimpleContainer(std::initializer_list<T> init) : data(new T[init.size()]), size(init.size()) {
        std::copy(init.begin(), init.end(), data);
    }

    ~SimpleContainer() {
        delete[] data;
    }

    T* begin() { return data; }
    T* end() { return data + size; }

private:
    T* data;
    std::size_t size;
};
```

##### Custom Iterator

```cpp
template <typename T>
class SimpleIterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;

    SimpleIterator(pointer ptr) : current(ptr) {}

    reference operator*() const { return *current; }
    pointer operator->() { return current; }

    SimpleIterator& operator++() {
        ++current;
        return *this;
    }

    SimpleIterator operator++(int) {
        SimpleIterator tmp = *this;
        ++current;
        return tmp;
    }

    SimpleIterator& operator--() {
        --current;
        return *this;
    }

    SimpleIterator operator--(int) {
        SimpleIterator tmp = *this;
        --current;
        return tmp;
    }

    SimpleIterator operator+(difference_type n) const {
        return SimpleIterator(current + n);
    }

    SimpleIterator operator-(difference_type n) const {
        return SimpleIterator(current - n);
    }

    difference_type operator-(const SimpleIterator& other) const {
        return current - other.current;
    }

    bool operator==(const SimpleIterator& other) const {
        return current == other.current;
    }

    bool operator!=(const SimpleIterator& other) const {
        return current != other.current;
    }

private:
    pointer current;
};

int main() {
    SimpleContainer<int> container = {1, 2, 3, 4, 5};



    for (SimpleIterator<int> it = container.begin(); it != container.end(); ++it) {
        std::cout << *it << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

#### Conclusion

Understanding iterator categories and hierarchies is fundamental for mastering C++ programming. Each iterator category—input, output, forward, bidirectional, and random access—serves specific use cases, with increasing functionality and complexity. By leveraging iterator traits and implementing custom iterators, you can create flexible, efficient, and reusable code. This knowledge is essential for writing sophisticated algorithms and extending the capabilities of the STL, enabling you to tackle complex programming challenges with confidence.

### 7.2. Custom Iterators

Custom iterators provide a way to traverse custom containers or enhance the functionality of existing ones. By implementing custom iterators, you can integrate your containers seamlessly with the Standard Template Library (STL) algorithms and range-based for loops. This subchapter explores the design and implementation of custom iterators in C++.

#### Design Principles

When designing a custom iterator, consider the following principles:

1. **Iterator Category**: Determine the appropriate iterator category (input, output, forward, bidirectional, random access) based on the requirements of your container.
2. **Interface Compliance**: Ensure the iterator adheres to the standard interface required by its category, including type aliases, operators, and member functions.
3. **Iterator Traits**: Provide the necessary type definitions through `std::iterator_traits` to support generic programming.
4. **Efficiency**: Optimize for performance, ensuring that operations such as incrementing, dereferencing, and comparing iterators are efficient.

#### Custom Iterator Example: A Simple Container

To illustrate the process, let’s create a custom iterator for a simple container. We will implement a `SimpleContainer` that stores elements in a dynamically allocated array and provide an iterator to traverse the elements.

##### Defining the Container

First, define the `SimpleContainer` class:

```cpp
#include <iostream>
#include <algorithm>

template <typename T>
class SimpleContainer {
public:
    SimpleContainer(std::initializer_list<T> init) : data(new T[init.size()]), size(init.size()) {
        std::copy(init.begin(), init.end(), data);
    }

    ~SimpleContainer() {
        delete[] data;
    }

    // Forward declaration of the iterator class
    class Iterator;

    Iterator begin() { return Iterator(data); }
    Iterator end() { return Iterator(data + size); }

private:
    T* data;
    std::size_t size;
};
```

##### Implementing the Iterator

Next, implement the `Iterator` class inside `SimpleContainer`:

```cpp
template <typename T>
class SimpleContainer<T>::Iterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;

    Iterator(pointer ptr) : current(ptr) {}

    reference operator*() const { return *current; }
    pointer operator->() { return current; }

    Iterator& operator++() {
        ++current;
        return *this;
    }

    Iterator operator++(int) {
        Iterator tmp = *this;
        ++current;
        return tmp;
    }

    Iterator& operator--() {
        --current;
        return *this;
    }

    Iterator operator--(int) {
        Iterator tmp = *this;
        --current;
        return tmp;
    }

    Iterator operator+(difference_type n) const {
        return Iterator(current + n);
    }

    Iterator operator-(difference_type n) const {
        return Iterator(current - n);
    }

    difference_type operator-(const Iterator& other) const {
        return current - other.current;
    }

    Iterator& operator+=(difference_type n) {
        current += n;
        return *this;
    }

    Iterator& operator-=(difference_type n) {
        current -= n;
        return *this;
    }

    reference operator[](difference_type n) const {
        return current[n];
    }

    bool operator==(const Iterator& other) const {
        return current == other.current;
    }

    bool operator!=(const Iterator& other) const {
        return current != other.current;
    }

    bool operator<(const Iterator& other) const {
        return current < other.current;
    }

    bool operator>(const Iterator& other) const {
        return current > other.current;
    }

    bool operator<=(const Iterator& other) const {
        return current <= other.current;
    }

    bool operator>=(const Iterator& other) const {
        return current >= other.current;
    }

private:
    pointer current;
};
```

#### Using the Custom Iterator

Now, let's use the `SimpleContainer` and its iterator in a program:

```cpp
int main() {
    SimpleContainer<int> container = {1, 2, 3, 4, 5};

    for (auto it = container.begin(); it != container.end(); ++it) {
        std::cout << *it << ' ';
    }
    std::cout << std::endl;

    // Using reverse iteration
    for (auto it = container.end() - 1; it != container.begin() - 1; --it) {
        std::cout << *it << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

#### Advanced Custom Iterator: A Bidirectional Linked List Iterator

For a more complex example, let’s implement a bidirectional iterator for a doubly linked list. This iterator will allow traversal in both forward and backward directions.

##### Defining the Linked List

First, define the `LinkedList` class and its nodes:

```cpp
template <typename T>
class LinkedList {
public:
    struct Node {
        T data;
        Node* prev;
        Node* next;
    };

    LinkedList() : head(nullptr), tail(nullptr) {}

    ~LinkedList() {
        Node* current = head;
        while (current) {
            Node* next = current->next;
            delete current;
            current = next;
        }
    }

    void push_back(const T& value) {
        Node* new_node = new Node{value, tail, nullptr};
        if (tail) {
            tail->next = new_node;
        } else {
            head = new_node;
        }
        tail = new_node;
    }

    class Iterator;

    Iterator begin() { return Iterator(head); }
    Iterator end() { return Iterator(nullptr); }

private:
    Node* head;
    Node* tail;
};
```

##### Implementing the Bidirectional Iterator

Next, implement the `Iterator` class inside `LinkedList`:

```cpp
template <typename T>
class LinkedList<T>::Iterator {
public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;

    Iterator(Node* ptr) : current(ptr) {}

    reference operator*() const { return current->data; }
    pointer operator->() { return &current->data; }

    Iterator& operator++() {
        current = current->next;
        return *this;
    }

    Iterator operator++(int) {
        Iterator tmp = *this;
        current = current->next;
        return tmp;
    }

    Iterator& operator--() {
        current = current->prev;
        return *this;
    }

    Iterator operator--(int) {
        Iterator tmp = *this;
        current = current->prev;
        return tmp;
    }

    bool operator==(const Iterator& other) const {
        return current == other.current;
    }

    bool operator!=(const Iterator& other) const {
        return current != other.current;
    }

private:
    Node* current;
};
```

#### Using the Bidirectional Iterator

Now, let's use the `LinkedList` and its iterator in a program:

```cpp
int main() {
    LinkedList<int> list;
    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    list.push_back(4);
    list.push_back(5);

    for (auto it = list.begin(); it != list.end(); ++it) {
        std::cout << *it << ' ';
    }
    std::cout << std::endl;

    for (auto it = --list.end(); it != --list.begin(); --it) {
        std::cout << *it << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

#### Custom Reverse Iterator

Custom reverse iterators are useful for iterating containers in reverse order. Here’s how to create a custom reverse iterator for our `LinkedList`.

##### Defining the Reverse Iterator

First, define the `ReverseIterator` class inside `LinkedList`:

```cpp
template <typename T>
class LinkedList<T>::ReverseIterator {
public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;

    ReverseIterator(Node* ptr) : current(ptr) {}

    reference operator*() const { return current->data; }
    pointer operator->() { return &current->data; }

    ReverseIterator& operator++() {
        current = current->prev;
        return *this;
    }

    ReverseIterator operator++(int) {
        ReverseIterator tmp = *this;
        current = current->prev;
        return tmp;
    }

    ReverseIterator& operator--() {
        current = current->next;
        return *this;
    }

    ReverseIterator operator--(int) {
        ReverseIterator tmp = *this;
        current = current->next;
        return tmp;
    }

    bool operator==(const ReverseIterator& other) const {
        return current == other.current;
    }

    bool operator!=(const ReverseIterator& other) const {
        return current != other.current;
    }

private:
    Node* current;
};
```

#### Using the Reverse Iterator

Now, let's use the `ReverseIterator` in a program:

```cpp
int main() {
    LinkedList<int> list;
    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    list.push_back(4);
    list.push_back(5);

    std::cout << "Forward iteration: ";
    for (auto it = list.begin(); it != list.end(); ++it) {
        std::cout << *it << ' ';
    }
    std::cout << std::endl;

    std::cout << "Reverse iteration: ";
    for (auto it = LinkedList<int>::ReverseIterator(list.tail); it != LinkedList<int>::ReverseIterator(nullptr); ++it) {
        std::cout << *it << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

#### Conclusion

Custom iterators provide powerful mechanisms to traverse custom containers and enhance existing ones. By adhering to the principles of iterator design, ensuring interface compliance, and leveraging iterator traits, you can create iterators that integrate seamlessly with STL algorithms and range-based for loops. From simple containers to complex data structures like linked lists, mastering custom iterators enables you to write more flexible, efficient, and reusable C++ code, addressing specific needs that standard iterators may not cover.

### 7.3. Iterator Adapters

Iterator adapters are a powerful feature in C++ that allow you to modify or enhance the behavior of existing iterators. They provide a flexible way to create new iterator types by adapting existing ones, enabling a wide range of functionality without the need to implement new iterators from scratch. In this subchapter, we will explore various iterator adapters provided by the Standard Template Library (STL) and demonstrate how to create custom iterator adapters.

#### Overview of Iterator Adapters

The STL provides several iterator adapters, including:

1. **Reverse Iterator**
2. **Insert Iterator**
3. **Stream Iterator**

Each adapter modifies the behavior of an underlying iterator, allowing you to perform operations such as reverse traversal, element insertion during iteration, and reading from or writing to streams.

#### 1. Reverse Iterator

Reverse iterators iterate over a container in the reverse direction. They are useful when you need to process elements in reverse order.

##### Using `std::reverse_iterator`

The `std::reverse_iterator` adapter can be used with any bidirectional or random access iterator.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    std::cout << "Original vector: ";
    for (const auto& val : vec) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    std::cout << "Reversed vector: ";
    for (auto rit = vec.rbegin(); rit != vec.rend(); ++rit) {
        std::cout << *rit << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

##### Custom Reverse Iterator

Let's implement a custom reverse iterator for a simple container:

```cpp
#include <iostream>
#include <algorithm>

template <typename T>
class SimpleContainer {
public:
    SimpleContainer(std::initializer_list<T> init) : data(new T[init.size()]), size(init.size()) {
        std::copy(init.begin(), init.end(), data);
    }

    ~SimpleContainer() {
        delete[] data;
    }

    class Iterator;
    class ReverseIterator;

    Iterator begin() { return Iterator(data); }
    Iterator end() { return Iterator(data + size); }
    ReverseIterator rbegin() { return ReverseIterator(data + size - 1); }
    ReverseIterator rend() { return ReverseIterator(data - 1); }

private:
    T* data;
    std::size_t size;
};

template <typename T>
class SimpleContainer<T>::Iterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;

    Iterator(pointer ptr) : current(ptr) {}

    reference operator*() const { return *current; }
    pointer operator->() { return current; }

    Iterator& operator++() {
        ++current;
        return *this;
    }

    Iterator operator++(int) {
        Iterator tmp = *this;
        ++current;
        return tmp;
    }

    Iterator& operator--() {
        --current;
        return *this;
    }

    Iterator operator--(int) {
        Iterator tmp = *this;
        --current;
        return tmp;
    }

    bool operator==(const Iterator& other) const { return current == other.current; }
    bool operator!=(const Iterator& other) const { return current != other.current; }

private:
    pointer current;
};

template <typename T>
class SimpleContainer<T>::ReverseIterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;

    ReverseIterator(pointer ptr) : current(ptr) {}

    reference operator*() const { return *current; }
    pointer operator->() { return current; }

    ReverseIterator& operator++() {
        --current;
        return *this;
    }

    ReverseIterator operator++(int) {
        ReverseIterator tmp = *this;
        --current;
        return tmp;
    }

    ReverseIterator& operator--() {
        ++current;
        return *this;
    }

    ReverseIterator operator--(int) {
        ReverseIterator tmp = *this;
        ++current;
        return tmp;
    }

    bool operator==(const ReverseIterator& other) const { return current == other.current; }
    bool operator!=(const ReverseIterator& other) const { return current != other.current; }

private:
    pointer current;
};

int main() {
    SimpleContainer<int> container = {1, 2, 3, 4, 5};

    std::cout << "Forward iteration: ";
    for (auto it = container.begin(); it != container.end(); ++it) {
        std::cout << *it << ' ';
    }
    std::cout << std::endl;

    std::cout << "Reverse iteration: ";
    for (auto rit = container.rbegin(); rit != container.rend(); ++rit) {
        std::cout << *rit << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

#### 2. Insert Iterator

Insert iterators allow you to insert elements into a container while iterating. The STL provides three types of insert iterators: `std::back_inserter`, `std::front_inserter`, and `std::inserter`.

##### Using `std::back_inserter`

The `std::back_inserter` adapter inserts elements at the end of a container.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>

int main() {
    std::vector<int> source = {1, 2, 3, 4, 5};
    std::vector<int> destination;

    std::copy(source.begin(), source.end(), std::back_inserter(destination));

    std::cout << "Destination vector: ";
    for (const auto& val : destination) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

##### Using `std::front_inserter`

The `std::front_inserter` adapter inserts elements at the front of a container. It requires the container to support `push_front`, such as `std::list` or `std::deque`.

```cpp
#include <iostream>
#include <list>
#include <algorithm>
#include <iterator>

int main() {
    std::list<int> source = {1, 2, 3, 4, 5};
    std::list<int> destination;

    std::copy(source.begin(), source.end(), std::front_inserter(destination));

    std::cout << "Destination list: ";
    for (const auto& val : destination) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

##### Using `std::inserter`

The `std::inserter` adapter inserts elements at a specified position in a container.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>

int main() {
    std::vector<int> source = {1, 2, 3, 4, 5};
    std::vector<int> destination = {10, 20, 30};

    auto it = destination.begin();
    std::advance(it, 1); // Move iterator to the second position

    std::copy(source.begin(), source.end(), std::inserter(destination, it));

    std::cout << "Destination vector: ";
    for (const auto& val : destination) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

#### 3. Stream Iterator

Stream iterators allow you to read from or write to streams using iterators. The STL provides `std::istream_iterator` and `std::ostream_iterator`.

##### Using `std::istream_iterator`

The `std::istream_iterator` reads data from an input stream.

```cpp
#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>

int main() {
    std::vector<int> vec;

    std::cout << "Enter integers (end with Ctrl+D): ";

    std::copy(std::istream_iterator<int>(std::cin),
              std::istream_iterator<int>(),
              std::back_inserter(vec));

    std::cout << "You entered: ";
    for (const auto& val : vec) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

##### Using `std::ostream_iterator`

The `std::ostream_iterator` writes data to an output stream.

```cpp
#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    std::cout << "Vector contents: ";
    std::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    return 0;
}
```

#### Custom Iterator Adapter

Let's create a custom iterator adapter that transforms the values of an underlying iterator. This adapter will apply a transformation function to each element as it is accessed.

##### Defining the Transform Iterator

First, define the `TransformIterator` class:

```cpp
#include <iterator>
#include <functional>

template <typename

 Iterator, typename Func>
class TransformIterator {
public:
    using iterator_category = typename std::iterator_traits<Iterator>::iterator_category;
    using value_type = typename std::result_of<Func(typename std::iterator_traits<Iterator>::reference)>::type;
    using difference_type = typename std::iterator_traits<Iterator>::difference_type;
    using pointer = value_type*;
    using reference = value_type;

    TransformIterator(Iterator it, Func func) : current(it), transform_func(func) {}

    reference operator*() const { return transform_func(*current); }
    pointer operator->() const { return &transform_func(*current); }

    TransformIterator& operator++() {
        ++current;
        return *this;
    }

    TransformIterator operator++(int) {
        TransformIterator tmp = *this;
        ++current;
        return tmp;
    }

    bool operator==(const TransformIterator& other) const { return current == other.current; }
    bool operator!=(const TransformIterator& other) const { return current != other.current; }

private:
    Iterator current;
    Func transform_func;
};
```

##### Using the Transform Iterator

Now, let's use the `TransformIterator` to transform elements in a container:

```cpp
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    auto transform_func = [](int x) { return std::sqrt(x); };
    TransformIterator<std::vector<int>::iterator, decltype(transform_func)> begin(vec.begin(), transform_func);
    TransformIterator<std::vector<int>::iterator, decltype(transform_func)> end(vec.end(), transform_func);

    std::cout << "Transformed vector: ";
    for (auto it = begin; it != end; ++it) {
        std::cout << *it << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

#### Conclusion

Iterator adapters are a versatile and powerful tool in C++, allowing you to modify and extend the behavior of existing iterators. By understanding and utilizing the STL-provided adapters such as `std::reverse_iterator`, `std::back_inserter`, `std::front_inserter`, `std::inserter`, `std::istream_iterator`, and `std::ostream_iterator`, you can efficiently perform a wide range of operations. Additionally, by creating custom iterator adapters like the `TransformIterator`, you can tailor iterator functionality to meet specific requirements, making your code more flexible and reusable. This mastery of iterator adapters enhances your ability to write sophisticated and efficient C++ programs.

### 7.4. Stream Iterators and Beyond

Stream iterators bridge the gap between input/output streams and STL algorithms, enabling seamless data processing directly from streams. These iterators simplify tasks such as reading from files, writing to files, and manipulating stream data in a highly efficient manner. In this subchapter, we will explore the use of stream iterators in detail, including practical examples, and discuss advanced concepts to take your understanding "beyond" basic usage.

#### Stream Iterators

The C++ Standard Library provides two main types of stream iterators:

1. **`std::istream_iterator`**: For reading data from input streams.
2. **`std::ostream_iterator`**: For writing data to output streams.

#### 1. `std::istream_iterator`

The `std::istream_iterator` reads data from an input stream. It can be used with standard algorithms to process input data on-the-fly.

##### Basic Usage

Here’s how to use `std::istream_iterator` to read integers from standard input:

```cpp
#include <iostream>
#include <iterator>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec;

    std::cout << "Enter integers (end with Ctrl+D or Ctrl+Z): ";

    std::copy(std::istream_iterator<int>(std::cin),
              std::istream_iterator<int>(),
              std::back_inserter(vec));

    std::cout << "You entered: ";
    for (const auto& val : vec) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

##### Reading from a File

You can use `std::istream_iterator` to read data from a file:

```cpp
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <algorithm>

int main() {
    std::ifstream file("input.txt");
    if (!file) {
        std::cerr << "Unable to open file" << std::endl;
        return 1;
    }

    std::vector<int> vec;
    std::copy(std::istream_iterator<int>(file),
              std::istream_iterator<int>(),
              std::back_inserter(vec));

    std::cout << "File contents: ";
    for (const auto& val : vec) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

#### 2. `std::ostream_iterator`

The `std::ostream_iterator` writes data to an output stream. It can be used to output data using standard algorithms.

##### Basic Usage

Here’s how to use `std::ostream_iterator` to write integers to standard output:

```cpp
#include <iostream>
#include <iterator>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    std::copy(vec.begin(), vec.end(),
              std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    return 0;
}
```

##### Writing to a File

You can use `std::ostream_iterator` to write data to a file:

```cpp
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    std::ofstream file("output.txt");
    if (!file) {
        std::cerr << "Unable to open file" << std::endl;
        return 1;
    }

    std::copy(vec.begin(), vec.end(),
              std::ostream_iterator<int>(file, " "));
    file << std::endl;

    return 0;
}
```

#### Advanced Usage of Stream Iterators

Stream iterators can be combined with other STL components to perform more advanced operations.

##### Filtering Stream Data

You can filter data from a stream by combining `std::istream_iterator` with `std::copy_if`:

```cpp
#include <iostream>
#include <iterator>
#include <vector>
#include <algorithm>

bool is_even(int n) {
    return n % 2 == 0;
}

int main() {
    std::vector<int> vec;

    std::cout << "Enter integers (end with Ctrl+D or Ctrl+Z): ";
    std::copy(std::istream_iterator<int>(std::cin),
              std::istream_iterator<int>(),
              std::back_inserter(vec));

    std::cout << "Even numbers: ";
    std::copy_if(vec.begin(), vec.end(),
                 std::ostream_iterator<int>(std::cout, " "),
                 is_even);
    std::cout << std::endl;

    return 0;
}
```

##### Transforming Stream Data

You can transform data from a stream using `std::transform`:

```cpp
#include <iostream>
#include <iterator>
#include <vector>
#include <algorithm>
#include <cmath>

int main() {
    std::vector<int> vec;

    std::cout << "Enter integers (end with Ctrl+D or Ctrl+Z): ";
    std::copy(std::istream_iterator<int>(std::cin),
              std::istream_iterator<int>(),
              std::back_inserter(vec));

    std::cout << "Square roots: ";
    std::transform(vec.begin(), vec.end(),
                   std::ostream_iterator<double>(std::cout, " "),
                   [](int n) { return std::sqrt(n); });
    std::cout << std::endl;

    return 0;
}
```

#### Beyond Basic Stream Iterators

While the standard stream iterators are powerful, you can extend their functionality or create custom stream iterators for more specialized tasks.

##### Custom Stream Iterator

Let’s create a custom stream iterator that reads data from a stream and applies a transformation function to each element before returning it.

```cpp
#include <iostream>
#include <iterator>
#include <functional>

template <typename T, typename Func>
class TransformingIStreamIterator : public std::iterator<std::input_iterator_tag, T> {
public:
    TransformingIStreamIterator(std::istream& is, Func func)
        : stream(is), transform_func(func), value(), end_of_stream(false) {
        ++(*this);
    }

    TransformingIStreamIterator()
        : stream(std::cin), transform_func(nullptr), end_of_stream(true) {}

    T operator*() const { return value; }

    TransformingIStreamIterator& operator++() {
        if (stream >> value) {
            value = transform_func(value);
        } else {
            end_of_stream = true;
        }
        return *this;
    }

    bool operator==(const TransformingIStreamIterator& other) const {
        return end_of_stream == other.end_of_stream;
    }

    bool operator!=(const TransformingIStreamIterator& other) const {
        return !(*this == other);
    }

private:
    std::istream& stream;
    Func transform_func;
    T value;
    bool end_of_stream;
};

int main() {
    auto transform_func = [](int n) { return n * 2; };
    TransformingIStreamIterator<int, decltype(transform_func)> begin(std::cin, transform_func);
    TransformingIStreamIterator<int, decltype(transform_func)> end;

    std::vector<int> vec;
    std::copy(begin, end, std::back_inserter(vec));

    std::cout << "Transformed input: ";
    for (const auto& val : vec) {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```

#### Combining Stream Iterators with Other Iterators

Stream iterators can be combined with other iterator types, such as `std::reverse_iterator`, to create complex data processing pipelines.

##### Example: Reading from Stream, Reversing, and Writing to Another Stream

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>

int main() {
    std::vector<int> vec;

    std::cout << "Enter integers (end with Ctrl+D or Ctrl+Z): ";
    std::copy(std::istream_iterator<int>(std::cin),
              std::istream_iterator<int>(),
              std::back_inserter(vec));

    std::cout << "Reversed output: ";
    std::copy(vec.rbegin(), vec.rend(),
              std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    return 0;
}
```

#### Advanced Stream Manipulations

##### Counting Words from an Input Stream

You can count the number of words in an input stream using `std::istream_iterator`:

```cpp
#include <iostream>
#include <iterator>
#include <algorithm>

int main() {
    std::cout << "Enter text (end with Ctrl+D or Ctrl+Z): ";

    std::istream_iterator<std::string> begin(std::cin), end;
    std::size_t word_count = std::distance(begin, end);

    std::cout << "Number of words: " << word_count << std::endl;

    return 0;
}
```

##### Writing Formatted Data to an Output Stream

You can use `std::ostream_iterator` to write formatted data to an output stream:

```cpp
#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <iomanip>

int main() {
    std::vector<double> vec = {1.23, 4.56, 7.89};

    std::cout << "Formatted output: ";
    std::copy(vec.begin(), vec.end(),
              std::ostream_iterator<double>(std::cout << std::fixed << std::setprecision(2), " "));


    std::cout << std::endl;

    return 0;
}
```

#### Conclusion

Stream iterators are a powerful feature in C++ that allow you to seamlessly integrate streams with STL algorithms. By using `std::istream_iterator` and `std::ostream_iterator`, you can perform complex data processing tasks directly from and to streams. Advanced usage includes filtering, transforming, and combining stream iterators with other iterator types. Furthermore, creating custom stream iterators enables you to tailor stream processing to your specific needs, making your code more flexible and powerful. This mastery of stream iterators and beyond empowers you to handle a wide range of data processing challenges with efficiency and elegance.
