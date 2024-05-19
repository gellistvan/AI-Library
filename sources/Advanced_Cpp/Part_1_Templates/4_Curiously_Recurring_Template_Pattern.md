
\newpage

## Chapter 4: Curiously Recurring Template Pattern (CRTP)

The Curiously Recurring Template Pattern (CRTP) is a powerful and intriguing design pattern in C++ that leverages the capabilities of templates and inheritance to achieve a variety of advanced programming techniques. In CRTP, a class template derives from a specialization of itself, allowing the derived class to interact with the base class in a unique and flexible way.

CRTP can be used for a wide range of purposes, including:

1. **Static Polymorphism**: Achieving polymorphic behavior at compile time, avoiding the overhead of virtual functions and dynamic dispatch.
2. **Code Reuse and Mixins**: Creating reusable components and mixin classes that can add functionality to derived classes.
3. **Curious Optimization Techniques**: Implementing techniques that rely on compile-time computations and optimizations.
4. **Type Safety and Enforcement**: Enforcing certain constraints and behaviors at compile time, improving type safety and reducing runtime errors.

In this chapter, we will delve into the principles of CRTP, exploring its structure and benefits through detailed examples. We will demonstrate how CRTP can be used to implement static polymorphism, create mixins, and achieve other advanced programming goals. By understanding and applying CRTP, you will be able to write more efficient, maintainable, and type-safe C++ code. Whether you are developing high-performance systems, complex libraries, or reusable components, CRTP offers a versatile toolset to enhance your C++ programming skills.

### 4.1. Introduction and Basics

#### Understanding CRTP

The Curiously Recurring Template Pattern (CRTP) is a unique design pattern in C++ where a class template derives from a specialization of itself. This pattern leverages the power of templates and inheritance to create flexible and reusable code structures. CRTP enables compile-time polymorphism, avoiding the overhead associated with runtime polymorphism, such as virtual function calls.

The basic structure of CRTP is as follows:

```cpp
template <typename Derived>
class Base {
    // Base class code that can use Derived
};

class Derived : public Base<Derived> {
    // Derived class code
};
```

In this structure, `Derived` inherits from `Base<Derived>`. This allows the base class to use features and functions of the derived class through the template parameter.

#### Basic Example of CRTP

Let's start with a basic example to illustrate the concept. Suppose we want to create a base class that provides a common interface for derived classes. We can use CRTP to achieve this:

```cpp
#include <iostream>

template <typename Derived>
class Base {
public:
    void interface() {
        // Call the derived class implementation
        static_cast<Derived*>(this)->implementation();
    }

    // A default implementation
    void implementation() {
        std::cout << "Base implementation" << std::endl;
    }
};

class Derived : public Base<Derived> {
public:
    // Override the implementation
    void implementation() {
        std::cout << "Derived implementation" << std::endl;
    }
};

int main() {
    Derived d;
    d.interface(); // Output: Derived implementation
    return 0;
}
```

In this example, the `Base` class template takes a `Derived` class as a template parameter. The `Base` class provides an `interface` method that calls the `implementation` method of the derived class using `static_cast`. The `Derived` class inherits from `Base<Derived>` and overrides the `implementation` method.

#### Benefits of CRTP

CRTP provides several benefits over traditional inheritance and polymorphism:

1. **Static Polymorphism**: CRTP enables compile-time polymorphism, which eliminates the runtime overhead associated with virtual function calls.
2. **Code Reuse**: It allows for the creation of mixin classes that can add functionality to derived classes, promoting code reuse.
3. **Type Safety**: CRTP enforces type relationships at compile time, improving type safety and reducing runtime errors.
4. **Compile-Time Computations**: It enables the use of compile-time computations and optimizations, leading to more efficient code.

#### Static Polymorphism with CRTP

One of the key applications of CRTP is achieving static polymorphism. Unlike dynamic polymorphism, which relies on virtual functions and dynamic dispatch, static polymorphism is resolved at compile time, resulting in more efficient code.

##### Example: Static Polymorphism

```cpp
#include <iostream>

template <typename Derived>
class Shape {
public:
    void draw() const {
        static_cast<const Derived*>(this)->draw();
    }
};

class Circle : public Shape<Circle> {
public:
    void draw() const {
        std::cout << "Drawing Circle" << std::endl;
    }
};

class Square : public Shape<Square> {
public:
    void draw() const {
        std::cout << "Drawing Square" << std::endl;
    }
};

int main() {
    Circle circle;
    Square square;

    circle.draw(); // Output: Drawing Circle
    square.draw(); // Output: Drawing Square

    return 0;
}
```

In this example, the `Shape` class template provides a `draw` method that calls the `draw` method of the derived class. The `Circle` and `Square` classes inherit from `Shape<Circle>` and `Shape<Square>`, respectively, and provide their own implementations of the `draw` method. The `draw` method calls are resolved at compile time, providing static polymorphism.

#### Mixins with CRTP

Mixins are a powerful use case of CRTP. A mixin is a class that provides certain functionalities to be inherited by multiple derived classes. Mixins allow for the composition of behaviors through inheritance, enabling code reuse and modular design.

##### Example: Mixin Class

```cpp
#include <iostream>

template <typename Derived>
class Printable {
public:
    void print() const {
        static_cast<const Derived*>(this)->print();
    }
};

class Data : public Printable<Data> {
public:
    void print() const {
        std::cout << "Data contents" << std::endl;
    }
};

class Logger : public Printable<Logger> {
public:
    void print() const {
        std::cout << "Logger contents" << std::endl;
    }
};

int main() {
    Data data;
    Logger logger;

    data.print();   // Output: Data contents
    logger.print(); // Output: Logger contents

    return 0;
}
```

In this example, the `Printable` mixin class provides a `print` method that calls the `print` method of the derived class. The `Data` and `Logger` classes inherit from `Printable<Data>` and `Printable<Logger>`, respectively, and provide their own implementations of the `print` method. The mixin allows both classes to share the same interface.

#### Enforcing Interfaces with CRTP

CRTP can be used to enforce interfaces and ensure that derived classes implement certain methods. This can be useful for creating base classes that require derived classes to implement specific functionality.

##### Example: Enforcing Interfaces

```cpp
#include <iostream>

template <typename Derived>
class Interface {
public:
    void call() {
        static_cast<Derived*>(this)->requiredMethod();
    }
};

class Implementation : public Interface<Implementation> {
public:
    void requiredMethod() {
        std::cout << "Implementation of required method" << std::endl;
    }
};

int main() {
    Implementation impl;
    impl.call(); // Output: Implementation of required method

    return 0;
}
```

In this example, the `Interface` class template provides a `call` method that requires the derived class to implement the `requiredMethod`. The `Implementation` class inherits from `Interface<Implementation>` and provides the required method. If the `Implementation` class did not provide the `requiredMethod`, a compile-time error would occur.

#### Compile-Time Computations with CRTP

CRTP can be combined with other template metaprogramming techniques to perform compile-time computations and optimizations. This can lead to more efficient code by leveraging compile-time information.

##### Example: Compile-Time Computation

```cpp
#include <iostream>
#include <array>

template <typename Derived>
class ArrayWrapper {
public:
    void print() const {
        static_cast<const Derived*>(this)->print();
    }
};

template <typename T, size_t N>
class StaticArray : public ArrayWrapper<StaticArray<T, N>> {
public:
    StaticArray() : data{} {}

    void print() const {
        for (const auto& elem : data) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }

    T& operator[](size_t index) {
        return data[index];
    }

private:
    std::array<T, N> data;
};

int main() {
    StaticArray<int, 5> arr;
    for (size_t i = 0; i < 5; ++i) {
        arr[i] = static_cast<int>(i * i);
    }
    arr.print(); // Output: 0 1 4 9 16

    return 0;
}
```

In this example, the `ArrayWrapper` class template provides a `print` method that calls the `print` method of the derived class. The `StaticArray` class inherits from `ArrayWrapper<StaticArray<T, N>>` and provides an implementation of the `print` method. The `StaticArray` class leverages compile-time information (array size `N`) to optimize storage and access.

#### Conclusion

The Curiously Recurring Template Pattern (CRTP) is a powerful and versatile design pattern in C++ that enables compile-time polymorphism, code reuse, and type safety. By leveraging templates and inheritance, CRTP allows the creation of flexible and efficient code structures. This subchapter introduced the basics of CRTP, demonstrating its benefits and providing detailed examples of its applications in static polymorphism, mixins, interface enforcement, and compile-time computations. Understanding and applying CRTP will enable you to write more efficient, maintainable, and type-safe C++ code, enhancing your skills as an advanced C++ programmer.

### 4.2. Benefits and Use Cases

#### Benefits of CRTP

The Curiously Recurring Template Pattern (CRTP) offers several significant benefits that make it an invaluable tool in advanced C++ programming. These benefits include enhanced performance through static polymorphism, increased code reuse and modularity through mixins, improved type safety, and the ability to perform compile-time computations and optimizations.

##### 1. Enhanced Performance through Static Polymorphism

One of the primary advantages of CRTP is the ability to achieve polymorphic behavior at compile time, known as static polymorphism. Unlike dynamic polymorphism, which relies on virtual function calls and runtime type information, static polymorphism resolves function calls at compile time. This eliminates the overhead associated with virtual function dispatch and can lead to significant performance improvements.

###### Example: Static Polymorphism

```cpp
#include <iostream>

template <typename Derived>
class Shape {
public:
    void draw() const {
        static_cast<const Derived*>(this)->draw();
    }
};

class Circle : public Shape<Circle> {
public:
    void draw() const {
        std::cout << "Drawing Circle" << std::endl;
    }
};

class Square : public Shape<Square> {
public:
    void draw() const {
        std::cout << "Drawing Square" << std::endl;
    }
};

int main() {
    Circle circle;
    Square square;

    circle.draw(); // Output: Drawing Circle
    square.draw(); // Output: Drawing Square

    return 0;
}
```

In this example, the `Shape` class template provides a `draw` method that calls the `draw` method of the derived class using `static_cast`. This approach avoids the overhead of virtual function calls and enables the compiler to inline the function calls for better performance.

##### 2. Increased Code Reuse and Modularity through Mixins

CRTP allows for the creation of mixin classes that add functionality to derived classes. Mixins enable the composition of behaviors through inheritance, promoting code reuse and modularity. This makes it easy to create reusable components that can be combined in different ways to achieve the desired functionality.

###### Example: Mixin Class

```cpp
#include <iostream>

template <typename Derived>
class Printable {
public:
    void print() const {
        static_cast<const Derived*>(this)->print();
    }
};

class Data : public Printable<Data> {
public:
    void print() const {
        std::cout << "Data contents" << std::endl;
    }
};

class Logger : public Printable<Logger> {
public:
    void print() const {
        std::cout << "Logger contents" << std::endl;
    }
};

int main() {
    Data data;
    Logger logger;

    data.print();   // Output: Data contents
    logger.print(); // Output: Logger contents

    return 0;
}
```

In this example, the `Printable` mixin class provides a `print` method that calls the `print` method of the derived class. Both `Data` and `Logger` classes inherit from `Printable` and implement their own `print` methods. The mixin allows both classes to share the same interface, promoting code reuse and modularity.

##### 3. Improved Type Safety

CRTP enforces type relationships at compile time, improving type safety and reducing runtime errors. By using CRTP, you can ensure that certain methods are implemented by the derived class and that specific constraints are met, all checked at compile time.

###### Example: Enforcing Interfaces

```cpp
#include <iostream>

template <typename Derived>
class Interface {
public:
    void call() {
        static_cast<Derived*>(this)->requiredMethod();
    }
};

class Implementation : public Interface<Implementation> {
public:
    void requiredMethod() {
        std::cout << "Implementation of required method" << std::endl;
    }
};

int main() {
    Implementation impl;
    impl.call(); // Output: Implementation of required method

    return 0;
}
```

In this example, the `Interface` class template provides a `call` method that requires the derived class to implement the `requiredMethod`. The `Implementation` class inherits from `Interface<Implementation>` and provides the required method. If the `Implementation` class did not provide the `requiredMethod`, a compile-time error would occur, ensuring type safety.

##### 4. Compile-Time Computations and Optimizations

CRTP can be combined with other template metaprogramming techniques to perform compile-time computations and optimizations. This can lead to more efficient code by leveraging compile-time information.

###### Example: Compile-Time Computation

```cpp
#include <iostream>
#include <array>

template <typename Derived>
class ArrayWrapper {
public:
    void print() const {
        static_cast<const Derived*>(this)->print();
    }
};

template <typename T, size_t N>
class StaticArray : public ArrayWrapper<StaticArray<T, N>> {
public:
    StaticArray() : data{} {}

    void print() const {
        for (const auto& elem : data) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }

    T& operator[](size_t index) {
        return data[index];
    }

private:
    std::array<T, N> data;
};

int main() {
    StaticArray<int, 5> arr;
    for (size_t i = 0; i < 5; ++i) {
        arr[i] = static_cast<int>(i * i);
    }
    arr.print(); // Output: 0 1 4 9 16

    return 0;
}
```

In this example, the `ArrayWrapper` class template provides a `print` method that calls the `print` method of the derived class. The `StaticArray` class inherits from `ArrayWrapper<StaticArray<T, N>>` and provides an implementation of the `print` method. The `StaticArray` class leverages compile-time information (array size `N`) to optimize storage and access.

#### Use Cases of CRTP

CRTP is widely used in various domains of software development due to its flexibility and efficiency. Some notable use cases include implementing static polymorphism, creating mixins, enabling policy-based design, and optimizing compile-time computations.

##### 1. Implementing Static Polymorphism

CRTP is often used to implement static polymorphism, where polymorphic behavior is resolved at compile time. This is particularly useful in performance-critical applications where the overhead of virtual function calls is unacceptable.

###### Example: Static Polymorphism in Geometric Shapes

```cpp
#include <iostream>

template <typename Derived>
class Shape {
public:
    void draw() const {
        static_cast<const Derived*>(this)->draw();
    }
};

class Circle : public Shape<Circle> {
public:
    void draw() const {
        std::cout << "Drawing Circle" << std::endl;
    }
};

class Square : public Shape<Square> {
public:
    void draw() const {
        std::cout << "Drawing Square" << std::endl;
    }
};

int main() {
    Circle circle;
    Square square;

    circle.draw(); // Output: Drawing Circle
    square.draw(); // Output: Drawing Square

    return 0;
}
```

In this example, the `Shape` class template provides a `draw` method that calls the `draw` method of the derived class using `static_cast`. This approach avoids the overhead of virtual function calls and enables the compiler to inline the function calls for better performance.

##### 2. Creating Mixins

Mixins are a powerful use case of CRTP. A mixin is a class that provides certain functionalities to be inherited by multiple derived classes. Mixins allow for the composition of behaviors through inheritance, enabling code reuse and modular design.

###### Example: Logging Mixin

```cpp
#include <iostream>

template <typename Derived>
class Logger {
public:
    void log(const std::string& message) const {
        static_cast<const Derived*>(this)->logImpl(message);
    }
};

class FileLogger : public Logger<FileLogger> {
public:
    void logImpl(const std::string& message) const {
        std::cout << "File log: " << message << std::endl;
    }
};

class ConsoleLogger : public Logger<ConsoleLogger> {
public:
    void logImpl(const std::string& message) const {
        std::cout << "Console log: " << message << std::endl;
    }
};

int main() {
    FileLogger fileLogger;
    ConsoleLogger consoleLogger;

    fileLogger.log("This is a file log message."); // Output: File log: This is a file log message.
    consoleLogger.log("This is a console log message."); // Output: Console log: This is a console log message.

    return 0;
}
```

In this example, the `Logger` mixin class provides a `log` method that calls the `logImpl` method of the derived class. Both `FileLogger` and `ConsoleLogger` inherit from `Logger` and implement their own `logImpl` methods. The mixin allows both classes to share the same logging interface, promoting code reuse and modularity.

##### 3. Policy-Based Design

CRTP can be used to implement policy-based design, where behaviors are defined by policy classes. This allows for flexible and customizable designs, enabling developers to mix and match policies to achieve the desired behavior.

###### Example: Policy-Based Design for Sorting Algorithms

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

// Policy class for ascending order
class Ascending {
public:
    template <typename T>
    bool operator()(const T& a, const T& b) const {
        return a < b;
    }
};

// Policy class for descending order
class Descending {
public:
    template <typename T>
    bool operator()(const T& a,

 const T& b) const {
        return a > b;
    }
};

template <typename Derived, typename Policy>
class Sorter {
public:
    void sort(std::vector<int>& data) const {
        std::sort(data.begin(), data.end(), Policy());
        static_cast<const Derived*>(this)->print(data);
    }
};

class AscendingSorter : public Sorter<AscendingSorter, Ascending> {
public:
    void print(const std::vector<int>& data) const {
        std::cout << "Ascending: ";
        for (int val : data) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
};

class DescendingSorter : public Sorter<DescendingSorter, Descending> {
public:
    void print(const std::vector<int>& data) const {
        std::cout << "Descending: ";
        for (int val : data) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
};

int main() {
    std::vector<int> data = {5, 2, 9, 1, 5, 6};

    AscendingSorter ascSorter;
    DescendingSorter descSorter;

    ascSorter.sort(data); // Output: Ascending: 1 2 5 5 6 9
    descSorter.sort(data); // Output: Descending: 9 6 5 5 2 1

    return 0;
}
```

In this example, the `Sorter` class template uses a policy class to define the sorting order. The `AscendingSorter` and `DescendingSorter` classes inherit from `Sorter` with different policies, enabling flexible and customizable sorting behavior.

##### 4. Optimizing Compile-Time Computations

CRTP can be combined with other template metaprogramming techniques to perform compile-time computations and optimizations. This can lead to more efficient code by leveraging compile-time information.

###### Example: Compile-Time Factorial Calculation

```cpp
#include <iostream>

template <typename Derived>
class FactorialBase {
public:
    constexpr int calculate(int n) const {
        return static_cast<const Derived*>(this)->calculateImpl(n);
    }
};

class Factorial : public FactorialBase<Factorial> {
public:
    constexpr int calculateImpl(int n) const {
        return (n <= 1) ? 1 : (n * calculateImpl(n - 1));
    }
};

int main() {
    constexpr Factorial factorial;
    constexpr int result = factorial.calculate(5);
    std::cout << "Factorial of 5 is " << result << std::endl; // Output: Factorial of 5 is 120

    return 0;
}
```

In this example, the `FactorialBase` class template provides a `calculate` method that calls the `calculateImpl` method of the derived class. The `Factorial` class inherits from `FactorialBase<Factorial>` and provides an implementation of the `calculateImpl` method. The factorial calculation is performed at compile time, demonstrating the power of compile-time computations.

#### Conclusion

The Curiously Recurring Template Pattern (CRTP) offers numerous benefits and is widely used in various domains of software development. By enabling static polymorphism, promoting code reuse and modularity through mixins, improving type safety, and facilitating compile-time computations and optimizations, CRTP is a powerful tool in the advanced C++ programmer's toolkit. Understanding and applying CRTP will enable you to write more efficient, maintainable, and type-safe C++ code, enhancing your skills and expanding the possibilities of what you can achieve with C++.

### 4.3. Implementing CRTP for Static Polymorphism

#### Introduction

Static polymorphism is a powerful technique that allows for polymorphic behavior to be resolved at compile time rather than runtime. The Curiously Recurring Template Pattern (CRTP) is a key tool for implementing static polymorphism in C++. Unlike dynamic polymorphism, which relies on virtual functions and runtime type information, static polymorphism leverages template instantiation and compile-time binding, resulting in more efficient code.

In this subchapter, we will explore how to implement CRTP for static polymorphism. We will discuss the benefits of static polymorphism, provide detailed examples of its implementation, and compare it to dynamic polymorphism.

#### Benefits of Static Polymorphism

Static polymorphism offers several advantages over dynamic polymorphism:

1. **Performance**: Since function calls are resolved at compile time, there is no runtime overhead associated with virtual function dispatch.
2. **Inlining**: The compiler can inline function calls, leading to more efficient code.
3. **Type Safety**: Type relationships are enforced at compile time, reducing the risk of runtime errors.
4. **Simplicity**: Avoiding virtual tables and dynamic type information simplifies the code.

#### Basic Structure of CRTP for Static Polymorphism

The basic structure of CRTP for static polymorphism involves defining a base class template that takes a derived class as a template parameter. The base class provides a common interface, and the derived class implements the specific functionality.

```cpp
template <typename Derived>
class Base {
public:
    void interface() {
        static_cast<Derived*>(this)->implementation();
    }

    void implementation() {
        std::cout << "Base implementation" << std::endl;
    }
};

class Derived : public Base<Derived> {
public:
    void implementation() {
        std::cout << "Derived implementation" << std::endl;
    }
};

int main() {
    Derived d;
    d.interface(); // Output: Derived implementation
    return 0;
}
```

In this example, the `Base` class template provides an `interface` method that calls the `implementation` method of the derived class using `static_cast`. The `Derived` class inherits from `Base<Derived>` and overrides the `implementation` method.

#### Implementing a Shape Hierarchy with CRTP

Let's implement a more complex example involving a shape hierarchy. We will create a `Shape` base class template and several derived classes representing different shapes, such as `Circle` and `Square`.

##### Step 1: Define the Base Class Template

First, we define the `Shape` base class template. This class provides a common interface for drawing shapes.

```cpp
#include <iostream>

template <typename Derived>
class Shape {
public:
    void draw() const {
        static_cast<const Derived*>(this)->draw();
    }

    void resize(double factor) {
        static_cast<Derived*>(this)->resize(factor);
    }
};
```

In this `Shape` class template, we provide a `draw` method and a `resize` method, which call the corresponding methods in the derived class using `static_cast`.

##### Step 2: Define the Derived Classes

Next, we define the derived classes `Circle` and `Square`.

```cpp
class Circle : public Shape<Circle> {
public:
    void draw() const {
        std::cout << "Drawing Circle" << std::endl;
    }

    void resize(double factor) {
        std::cout << "Resizing Circle by factor " << factor << std::endl;
    }
};

class Square : public Shape<Square> {
public:
    void draw() const {
        std::cout << "Drawing Square" << std::endl;
    }

    void resize(double factor) {
        std::cout << "Resizing Square by factor " << factor << std::endl;
    }
};
```

In these classes, we implement the `draw` and `resize` methods to provide the specific functionality for each shape.

##### Step 3: Use the Shape Hierarchy

Finally, we create instances of `Circle` and `Square` and call their methods through the `Shape` interface.

```cpp
int main() {
    Circle circle;
    Square square;

    circle.draw();   // Output: Drawing Circle
    square.draw();   // Output: Drawing Square

    circle.resize(2.0); // Output: Resizing Circle by factor 2
    square.resize(3.0); // Output: Resizing Square by factor 3

    return 0;
}
```

In this example, the `Shape` interface allows us to draw and resize different shapes, with the specific behavior being resolved at compile time.

#### Comparing Static and Dynamic Polymorphism

To better understand the benefits of static polymorphism with CRTP, let's compare it to dynamic polymorphism using virtual functions.

##### Dynamic Polymorphism Example

```cpp
#include <iostream>

class Shape {
public:
    virtual ~Shape() = default;
    virtual void draw() const = 0;
    virtual void resize(double factor) = 0;
};

class Circle : public Shape {
public:
    void draw() const override {
        std::cout << "Drawing Circle" << std::endl;
    }

    void resize(double factor) override {
        std::cout << "Resizing Circle by factor " << factor << std::endl;
    }
};

class Square : public Shape {
public:
    void draw() const override {
        std::cout << "Drawing Square" << std::endl;
    }

    void resize(double factor) override {
        std::cout << "Resizing Square by factor " << factor << std::endl;
    }
};

int main() {
    Shape* shapes[] = {new Circle(), new Square()};

    for (Shape* shape : shapes) {
        shape->draw();
        shape->resize(2.0);
        delete shape;
    }

    return 0;
}
```

In this example, we use virtual functions to achieve polymorphism. The `Shape` base class defines virtual `draw` and `resize` methods, which are overridden by the `Circle` and `Square` classes. While this approach works, it involves runtime overhead for virtual function dispatch and dynamic memory management.

##### Static Polymorphism with CRTP

Using CRTP, we can achieve the same functionality with compile-time polymorphism, eliminating the runtime overhead.

```cpp
#include <iostream>

template <typename Derived>
class Shape {
public:
    void draw() const {
        static_cast<const Derived*>(this)->draw();
    }

    void resize(double factor) {
        static_cast<Derived*>(this)->resize(factor);
    }
};

class Circle : public Shape<Circle> {
public:
    void draw() const {
        std::cout << "Drawing Circle" << std::endl;
    }

    void resize(double factor) {
        std::cout << "Resizing Circle by factor " << factor << std::endl;
    }
};

class Square : public Shape<Square> {
public:
    void draw() const {
        std::cout << "Drawing Square" << std::endl;
    }

    void resize(double factor) {
        std::cout << "Resizing Square by factor " << factor << std::endl;
    }
};

int main() {
    Circle circle;
    Square square;

    circle.draw();   // Output: Drawing Circle
    square.draw();   // Output: Drawing Square

    circle.resize(2.0); // Output: Resizing Circle by factor 2
    square.resize(3.0); // Output: Resizing Square by factor 3

    return 0;
}
```

In this CRTP-based example, the polymorphic behavior is resolved at compile time, eliminating the need for virtual function calls and dynamic memory management. This results in more efficient code with improved performance.

#### Advanced Example: Static Polymorphism with Multiple Inheritance

CRTP can also be used to achieve static polymorphism with multiple inheritance. Let's extend our shape hierarchy to include additional functionality, such as coloring shapes.

##### Step 1: Define the Colorable Mixin

First, we define a `Colorable` mixin class that provides methods for setting and getting the color of a shape.

```cpp
template <typename Derived>
class Colorable {
public:
    void setColor(const std::string& color) {
        static_cast<Derived*>(this)->setColorImpl(color);
    }

    std::string getColor() const {
        return static_cast<const Derived*>(this)->getColorImpl();
    }
};
```

In this `Colorable` class template, we provide `setColor` and `getColor` methods that call the corresponding methods in the derived class using `static_cast`.

##### Step 2: Extend the Shape Classes

Next, we extend the `Circle` and `Square` classes to inherit from both `Shape` and `Colorable`.

```cpp
class Circle : public Shape<Circle>, public Colorable<Circle> {
public:
    void draw() const {
        std::cout << "Drawing Circle" << std::endl;
    }

    void resize(double factor) {
        std::cout << "Resizing Circle by factor " << factor << std::endl;
    }

    void setColorImpl(const std::string& color) {
        this->color = color;
    }

    std::string getColorImpl() const {
        return color;
    }

private:
    std::string color;
};

class Square : public Shape<Square>, public Colorable<Square> {
public:
    void draw() const {
        std::cout << "Drawing Square" << std::endl;
    }

    void resize(double factor) {
        std::cout << "Resizing Square by factor " << factor << std::endl;
    }

    void setColorImpl(const std::string& color) {
        this->color = color;
    }

    std::string getColorImpl() const {
        return color;
    }

private:
    std::string color;
};
```

In these classes, we implement the `setColorImpl` and `getColorImpl` methods to provide the specific functionality for each shape.

##### Step 3: Use the Extended Shape Classes

Finally, we create instances of `Circle` and `Square` and call their methods through both the `Shape` and `Colorable` interfaces.

```cpp
int main() {
    Circle circle;
    Square square;

    circle.draw();   // Output: Drawing Circle
    square.draw();   // Output: Drawing Square

    circle.resize(2.0); // Output: Resizing Circle by factor 2
    square.resize(3.0); // Output: Resizing Square by factor 3

    circle.setColor("red");
    square.setColor("blue");

    std::cout << "Circle color: " << circle.getColor() << std::endl; // Output: Circle color: red
    std::cout << "Square color: " << square.getColor() << std::endl; // Output: Square color: blue

    return 0;
}
```

In this example, the `Colorable` mixin class adds color-related functionality to both `Circle` and `Square`. By using CRTP, we achieve static polymorphism with multiple inheritance, allowing for flexible and reusable code.

#### Conclusion

The Curiously Recurring Template Pattern (CRTP) is a powerful tool for implementing static polymorphism in C++. By leveraging template instantiation and compile-time binding, CRTP enables polymorphic behavior without the overhead of virtual function calls and dynamic type information. This results in more efficient code with improved performance.

In this subchapter, we explored the benefits of static polymorphism, provided detailed examples of implementing CRTP for static polymorphism, and compared it to dynamic polymorphism. We also demonstrated advanced use cases, such as static polymorphism with multiple inheritance.

Understanding and applying CRTP for static polymorphism will enable you to write more efficient, maintainable, and type-safe C++ code, enhancing your skills as an advanced C++ programmer.
