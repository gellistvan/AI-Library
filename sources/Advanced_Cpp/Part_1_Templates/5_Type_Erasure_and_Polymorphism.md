
\newpage
## Chapter 5: Type Erasure and Polymorphism

Type erasure is a powerful and sophisticated technique in C++ that allows for runtime polymorphism without the need for traditional inheritance and virtual functions. By decoupling the interface from the implementation, type erasure enables the creation of flexible and generic code that can work with different types while providing a uniform interface. This approach is particularly useful in scenarios where the types involved are not known at compile time or when you want to avoid the overhead and constraints associated with inheritance hierarchies.

In this chapter, we will explore the concept of type erasure and its role in achieving polymorphism. We will discuss how type erasure works, its benefits, and its use cases. We will also compare type erasure with other forms of polymorphism, such as inheritance-based polymorphism and CRTP-based static polymorphism, highlighting the strengths and trade-offs of each approach.

Through detailed examples and practical applications, we will demonstrate how to implement type erasure in C++ using techniques like `std::any`, `std::function`, and custom type erasure classes. By the end of this chapter, you will have a deep understanding of type erasure and how to leverage it to write flexible, efficient, and maintainable C++ code that can handle a wide variety of types and interfaces dynamically. Whether you are designing complex systems, building generic libraries, or working with heterogeneous collections, type erasure provides a powerful tool to enhance your C++ programming capabilities.

### 5.1. Concepts and Importance

#### Introduction to Type Erasure

Type erasure is a sophisticated technique in C++ that provides runtime polymorphism without relying on inheritance and virtual functions. By abstracting the operations on a type away from the type itself, type erasure allows for the creation of flexible and reusable interfaces that can operate on any type conforming to a given interface. This technique decouples the interface from the implementation, enabling the handling of heterogeneous types uniformly.

#### How Type Erasure Works

At its core, type erasure involves wrapping a concrete type in an abstract wrapper that exposes a uniform interface. This wrapper hides the details of the concrete type, effectively "erasing" the type information while still allowing operations to be performed on the object. The wrapper typically contains a pointer to a base class with virtual functions or a function pointer table, providing the necessary operations for the wrapped type.

#### Key Components of Type Erasure

1. **Interface**: A set of operations that the type-erased object must support.
2. **Concrete Implementation**: The actual type that implements the interface.
3. **Wrapper**: An abstraction that erases the type of the concrete implementation while exposing the interface.

#### Importance of Type Erasure

Type erasure is important for several reasons:

1. **Flexibility**: Type erasure allows for the creation of flexible and generic interfaces that can operate on any type conforming to the interface, without the need for a common base class.
2. **Decoupling**: It decouples the interface from the implementation, enabling changes to the implementation without affecting the interface.
3. **Code Reuse**: By providing a uniform interface, type erasure promotes code reuse and simplifies the handling of heterogeneous types.
4. **Avoidance of Inheritance**: Type erasure eliminates the need for inheritance and virtual functions, avoiding the associated overhead and constraints.

#### Example: Using `std::any` for Type Erasure

The `std::any` class template, introduced in C++17, is a standard library utility that provides type erasure. It can hold an instance of any type that satisfies certain requirements and provides a type-safe way to retrieve the stored value.

```cpp
#include <any>

#include <iostream>
#include <string>

int main() {
    std::any value = 42;
    std::cout << std::any_cast<int>(value) << std::endl; // Output: 42

    value = std::string("Hello, World!");
    std::cout << std::any_cast<std::string>(value) << std::endl; // Output: Hello, World!

    try {
        std::cout << std::any_cast<int>(value) << std::endl; // Throws std::bad_any_cast
    } catch (const std::bad_any_cast& e) {
        std::cout << "Bad any cast: " << e.what() << std::endl; // Output: Bad any cast: bad any cast
    }

    return 0;
}
```

In this example, `std::any` is used to store and retrieve values of different types. The type information is erased, but the stored value can be safely retrieved using `std::any_cast`.

#### Example: Using `std::function` for Type Erasure

The `std::function` class template provides type erasure for callable objects. It can hold any callable object (functions, lambda expressions, function objects) that matches a specific signature.

```cpp
#include <functional>

#include <iostream>

void printMessage(const std::string& message) {
    std::cout << message << std::endl;
}

int main() {
    std::function<void(const std::string&)> func = printMessage;
    func("Hello, World!"); // Output: Hello, World!

    func = [](const std::string& message) {
        std::cout << "Lambda: " << message << std::endl;
    };
    func("Hello, Lambda!"); // Output: Lambda: Hello, Lambda!

    return 0;
}
```

In this example, `std::function` is used to store and invoke different callable objects with the same signature. The type of the callable object is erased, but the function can still be called with the expected arguments.

#### Custom Type Erasure

While `std::any` and `std::function` provide convenient type erasure mechanisms, there are cases where custom type erasure is necessary. Custom type erasure involves defining your own abstract interface and wrapper classes.

##### Step 1: Define the Interface

First, define an abstract interface that specifies the operations to be supported.

```cpp
class IShape {
public:
    virtual ~IShape() = default;
    virtual void draw() const = 0;
    virtual void resize(double factor) = 0;
};
```

In this `IShape` class, we define two pure virtual functions: `draw` and `resize`.

##### Step 2: Define the Concrete Implementation

Next, define concrete implementations of the interface for different shapes.

```cpp
class Circle : public IShape {
public:
    void draw() const override {
        std::cout << "Drawing Circle" << std::endl;
    }

    void resize(double factor) override {
        std::cout << "Resizing Circle by factor " << factor << std::endl;
    }
};

class Square : public IShape {
public:
    void draw() const override {
        std::cout << "Drawing Square" << std::endl;
    }

    void resize(double factor) override {
        std::cout << "Resizing Square by factor " << factor << std::endl;
    }
};
```

In these classes, we implement the `draw` and `resize` methods to provide the specific functionality for each shape.

##### Step 3: Define the Wrapper

Define a wrapper class that uses type erasure to store any concrete implementation of the interface.

```cpp
class Shape {
public:
    template <typename T>
    Shape(T shape) : impl(std::make_shared<Model<T>>(std::move(shape))) {}

    void draw() const {
        impl->draw();
    }

    void resize(double factor) {
        impl->resize(factor);
    }

private:
    struct Concept {
        virtual ~Concept() = default;
        virtual void draw() const = 0;
        virtual void resize(double factor) = 0;
    };

    template <typename T>
    struct Model : Concept {
        Model(T shape) : shape(std::move(shape)) {}

        void draw() const override {
            shape.draw();
        }

        void resize(double factor) override {
            shape.resize(factor);
        }

        T shape;
    };

    std::shared_ptr<const Concept> impl;
};
```

In this `Shape` class, we use the "type erasure idiom" to wrap any object that implements the `IShape` interface. The `Shape` class contains a pointer to a `Concept` object, which is an abstract base class with pure virtual functions. The `Model` template class derives from `Concept` and implements the interface by forwarding the calls to the wrapped object.

##### Step 4: Use the Wrapper

Finally, create instances of `Circle` and `Square`, and store them in the `Shape` wrapper.

```cpp
int main() {
    Shape circle = Circle();
    Shape square = Square();

    circle.draw();   // Output: Drawing Circle
    square.draw();   // Output: Drawing Square

    circle.resize(2.0); // Output: Resizing Circle by factor 2
    square.resize(3.0); // Output: Resizing Square by factor 3

    return 0;
}
```

In this example, the `Shape` wrapper can hold any object that implements the `IShape` interface, providing a uniform interface for drawing and resizing shapes.

#### Use Cases of Type Erasure

Type erasure is useful in various scenarios, including:

1. **Heterogeneous Collections**: Storing objects of different types in a single collection while providing a uniform interface.
2. **Generic Programming**: Writing generic algorithms that can operate on any type that conforms to a specific interface.
3. **Dynamic Interfaces**: Providing dynamic interfaces that can adapt to different implementations at runtime.
4. **Simplifying Code**: Decoupling interfaces from implementations, making the codebase easier to understand and maintain.

##### Example: Heterogeneous Collections

Type erasure allows us to store objects of different types in a single collection and operate on them through a uniform interface.

```cpp
#include <vector>

#include <memory>

int main() {
    std::vector<std::shared_ptr<IShape>> shapes;
    shapes.push_back(std::make_shared<Circle>());
    shapes.push_back(std::make_shared<Square>());

    for (const auto& shape : shapes) {
        shape->draw();
        shape->resize(1.5);
    }

    // Output:
    // Drawing Circle
    // Resizing Circle by factor 1.5
    // Drawing Square
    // Resizing Square by factor 1.5

    return 0;
}
```

In this example, we store `Circle` and `Square` objects in a `std::vector` of `std::shared_ptr<IShape>`. The type information is erased, but we can still call the `draw` and `resize` methods on the stored objects.

#### Conclusion

Type erasure is a powerful technique in C++ that provides runtime polymorphism without the need for inheritance and virtual functions. By decoupling the interface from the implementation, type erasure enables the creation of flexible and generic code that can operate on different types while providing a uniform interface. This technique is particularly useful in scenarios where the types involved are not known at compile time or when you want to avoid the overhead and constraints associated with inheritance hierarchies.

In

this subchapter, we explored the concepts and importance of type erasure, demonstrated how to implement type erasure using `std::any`, `std::function`, and custom type erasure classes, and discussed various use cases. Understanding and applying type erasure will enable you to write more flexible, efficient, and maintainable C++ code, enhancing your skills as an advanced C++ programmer.

### 5.2. `std::any` and `std::variant`

#### Introduction

Type erasure is a powerful technique in C++ that provides runtime polymorphism by decoupling the interface from the implementation. The C++ standard library includes utilities like `std::any` and `std::variant` that facilitate type erasure in a convenient and type-safe manner. In this subchapter, we will explore the concepts, usage, and differences between `std::any` and `std::variant`, and provide detailed examples to illustrate their capabilities.

#### `std::any`

`std::any` is a type-safe container introduced in C++17 that can hold an instance of any type that satisfies certain requirements. It provides a way to store and manipulate values of any type without knowing the type at compile time. The stored value can be retrieved safely using `std::any_cast`.

##### Basic Usage of `std::any`

```cpp
#include <any>

#include <iostream>
#include <string>

int main() {
    std::any value;

    // Store an int
    value = 42;
    std::cout << std::any_cast<int>(value) << std::endl; // Output: 42

    // Store a string
    value = std::string("Hello, World!");
    std::cout << std::any_cast<std::string>(value) << std::endl; // Output: Hello, World!

    // Attempt to retrieve a value of the wrong type
    try {
        std::cout << std::any_cast<int>(value) << std::endl; // Throws std::bad_any_cast
    } catch (const std::bad_any_cast& e) {
        std::cout << "Bad any cast: " << e.what() << std::endl; // Output: Bad any cast: bad any cast
    }

    return 0;
}
```

In this example, `std::any` is used to store and retrieve values of different types. The type information is erased, but the stored value can be safely retrieved using `std::any_cast`.

##### Checking the Stored Type

`std::any` provides methods to check the type of the stored value.

```cpp
#include <any>

#include <iostream>
#include <typeinfo>

int main() {
    std::any value = 42;

    if (value.type() == typeid(int)) {
        std::cout << "The stored type is int" << std::endl;
    } else {
        std::cout << "The stored type is not int" << std::endl;
    }

    value = std::string("Hello, World!");

    if (value.type() == typeid(std::string)) {
        std::cout << "The stored type is std::string" << std::endl;
    } else {
        std::cout << "The stored type is not std::string" << std::endl;
    }

    return 0;
}
```

In this example, the `type()` method is used to check the type of the stored value, which can be compared with `typeid` to determine if it matches a specific type.

#### `std::variant`

`std::variant` is another type-safe container introduced in C++17 that can hold a value from a fixed set of types. It provides an efficient way to represent a value that can be one of several types, with compile-time type safety. Unlike `std::any`, `std::variant` requires the set of possible types to be known at compile time.

##### Basic Usage of `std::variant`

```cpp
#include <variant>

#include <iostream>
#include <string>

int main() {
    std::variant<int, std::string> value;

    // Store an int
    value = 42;
    std::cout << std::get<int>(value) << std::endl; // Output: 42

    // Store a string
    value = std::string("Hello, World!");
    std::cout << std::get<std::string>(value) << std::endl; // Output: Hello, World!

    // Attempt to retrieve a value of the wrong type
    try {
        std::cout << std::get<int>(value) << std::endl; // Throws std::bad_variant_access
    } catch (const std::bad_variant_access& e) {
        std::cout << "Bad variant access: " << e.what() << std::endl; // Output: Bad variant access: bad_variant_access
    }

    return 0;
}
```

In this example, `std::variant` is used to store and retrieve values of different types from a fixed set. The stored value can be safely retrieved using `std::get`, but attempting to access the wrong type will throw `std::bad_variant_access`.

##### Visiting a `std::variant`

`std::variant` provides a way to visit the stored value using `std::visit`, which applies a visitor function to the value regardless of its type.

```cpp
#include <variant>

#include <iostream>
#include <string>

int main() {
    std::variant<int, std::string> value;

    value = 42;

    std::visit([](auto&& arg) {
        std::cout << "Value: " << arg << std::endl;
    }, value); // Output: Value: 42

    value = std::string("Hello, World!");

    std::visit([](auto&& arg) {
        std::cout << "Value: " << arg << std::endl;
    }, value); // Output: Value: Hello, World!

    return 0;
}
```

In this example, `std::visit` is used to apply a lambda function to the value stored in the `std::variant`, printing the value regardless of its type.

#### Differences Between `std::any` and `std::variant`

While both `std::any` and `std::variant` provide type erasure, they have different use cases and characteristics:

- **Type Information**: `std::any` can store any type, while `std::variant` can only store types from a predefined set.
- **Type Safety**: `std::variant` provides compile-time type safety, ensuring that only valid types are stored, while `std::any` requires runtime checks.
- **Performance**: `std::variant` is typically more efficient than `std::any` because it does not require dynamic memory allocation and type erasure for arbitrary types.
- **Use Cases**: `std::any` is useful for storing values of unknown or arbitrary types, while `std::variant` is ideal for representing a value that can be one of several known types.

#### Practical Examples and Use Cases

##### Example: Configurable Settings with `std::any`

`std::any` is useful for implementing a system where settings can be of various types, and the types are not known in advance.

```cpp
#include <any>

#include <iostream>
#include <string>

#include <unordered_map>

class Settings {
public:
    template <typename T>
    void set(const std::string& key, T value) {
        settings[key] = value;
    }

    template <typename T>
    T get(const std::string& key) const {
        try {
            return std::any_cast<T>(settings.at(key));
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Bad any cast");
        } catch (const std::out_of_range& e) {
            throw std::runtime_error("Key not found");
        }
    }

private:
    std::unordered_map<std::string, std::any> settings;
};

int main() {
    Settings settings;
    settings.set("volume", 10);
    settings.set("username", std::string("admin"));

    std::cout << "Volume: " << settings.get<int>("volume") << std::endl;         // Output: Volume: 10
    std::cout << "Username: " << settings.get<std::string>("username") << std::endl; // Output: Username: admin

    try {
        std::cout << "Brightness: " << settings.get<int>("brightness") << std::endl;
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl; // Output: Key not found
    }

    return 0;
}
```

In this example, `Settings` uses `std::any` to store settings of various types. The `set` and `get` methods provide a type-safe way to access the stored values.

##### Example: Event System with `std::variant`

`std::variant` is useful for implementing an event system where events can be of different types but belong to a known set of types.

```cpp
#include <variant>

#include <iostream>
#include <string>

#include <vector>

struct MouseEvent {
    int x, y;
};

struct KeyEvent {
    int keycode;
};

using Event = std::variant<MouseEvent, KeyEvent>;

void handleEvent(const Event& event) {
    std::visit([](auto&& e) {
        using T = std::decay_t<decltype(e)>;
        if constexpr (std::is_same_v<T, MouseEvent>) {
            std::cout << "MouseEvent at (" << e.x << ", " << e.y << ")" << std::endl;
        } else if constexpr (std::is_same_v<T, KeyEvent>) {
            std::cout << "KeyEvent with keycode " << e.keycode << std::endl;
        }
    }, event);
}

int main() {
    std::vector<Event> events = {
        MouseEvent{100, 200},
        KeyEvent{42},
        MouseEvent{150, 250},
    };

    for (const auto& event : events) {
        handleEvent(event);
    }

    // Output:
    // MouseEvent at (100, 200)
    // KeyEvent with keycode 42
    // MouseEvent at (150, 250)

    return 0;
}
```

In this example, `Event` is defined as a `std::variant` of `MouseEvent` and `KeyEvent`. The `handleEvent` function uses `std::visit` to handle each event type appropriately.

#### Conclusion

`std::any` and `std::variant` are powerful tools in the C++ standard library that facilitate type erasure and provide flexible, type-safe ways to handle heterogeneous types. `std::any` is ideal for cases where the types are not known at compile time, offering a way to store and retrieve values of any type. On the other hand, `std::variant` provides a way to handle a fixed set of types with compile-time type safety and efficiency.

Understanding and utilizing `std::any` and `std::variant` allows you to write more flexible, efficient, and maintainable C++ code, enabling you to handle a wide variety of types and interfaces dynamically. Whether you are designing complex systems, building generic libraries, or working with heterogeneous collections, these utilities provide powerful tools to enhance your C++ programming capabilities.

### 5.3. Type Erasure with `std::function`

#### Introduction

Type erasure is a powerful technique in C++ that enables runtime polymorphism without relying on inheritance and virtual functions. One of the most common and useful utilities in the C++ standard library for type erasure is `std::function`. Introduced in C++11, `std::function` is a polymorphic function wrapper that can store and invoke any callable object that matches a specific function signature. This makes it an essential tool for writing flexible, generic code that can work with a variety of callable types.

In this subchapter, we will delve into the details of `std::function`, exploring how it works, its benefits, and its practical applications. We will provide detailed examples to illustrate how `std::function` can be used to achieve type erasure and enable runtime polymorphism with callable objects.

#### Understanding `std::function`

`std::function` is a class template that provides a type-erased wrapper for callable objects. This means it can store and invoke any callable entity—such as functions, lambda expressions, function objects, and even member function pointers—that matches a given signature. The type information of the stored callable is erased, and `std::function` provides a uniform interface to call the stored function.

The basic usage of `std::function` involves specifying the desired function signature and then assigning it a callable object that matches this signature.

##### Example: Basic Usage of `std::function`

```cpp
#include <functional>

#include <iostream>
#include <string>

// A free function
void printMessage(const std::string& message) {
    std::cout << message << std::endl;
}

int main() {
    // Define a std::function with the desired signature
    std::function<void(const std::string&)> func;

    // Assign a free function to std::function
    func = printMessage;
    func("Hello, World!"); // Output: Hello, World!

    // Assign a lambda expression to std::function
    func = [](const std::string& message) {
        std::cout << "Lambda: " << message << std::endl;
    };
    func("Hello, Lambda!"); // Output: Lambda: Hello, Lambda!

    return 0;
}
```

In this example, `std::function<void(const std::string&)>` is used to define a function wrapper that can store any callable object with the signature `void(const std::string&)`. We assign both a free function (`printMessage`) and a lambda expression to this `std::function` and invoke them.

#### Benefits of `std::function`

`std::function` provides several key benefits:

1. **Flexibility**: It can store and invoke any callable object with a matching signature, including functions, lambda expressions, and function objects.
2. **Type Erasure**: The type information of the stored callable is erased, providing a uniform interface for calling the function.
3. **Copyability**: `std::function` is copyable and assignable, allowing it to be easily passed around and stored in containers.
4. **Exception Safety**: `std::function` provides strong exception safety guarantees, ensuring that it behaves predictably in the presence of exceptions.

#### Practical Applications of `std::function`

`std::function` is widely used in various scenarios, including event handling, callbacks, and implementing generic algorithms that operate on callable objects.

##### Example: Event Handling System

One common use case for `std::function` is in event handling systems, where it can be used to store and invoke callbacks.

```cpp
#include <functional>

#include <iostream>
#include <vector>

class Button {
public:
    using ClickHandler = std::function<void()>;

    void setClickHandler(ClickHandler handler) {
        clickHandler = std::move(handler);
    }

    void click() const {
        if (clickHandler) {
            clickHandler();
        }
    }

private:
    ClickHandler clickHandler;
};

int main() {
    Button button;

    // Set a click handler using a lambda expression
    button.setClickHandler([]() {
        std::cout << "Button clicked!" << std::endl;
    });

    // Simulate a button click
    button.click(); // Output: Button clicked!

    return 0;
}
```

In this example, the `Button` class uses `std::function<void()>` to store a click handler callback. The `setClickHandler` method allows setting the callback, and the `click` method invokes it if it is set.

##### Example: Generic Algorithm with `std::function`

`std::function` can be used to implement generic algorithms that operate on callable objects. Here is an example of a simple for-each function that takes a range of elements and a callable object to apply to each element.

```cpp
#include <functional>

#include <iostream>
#include <vector>

template <typename T>
void forEach(const std::vector<T>& vec, const std::function<void(const T&)>& func) {
    for (const auto& element : vec) {
        func(element);
    }
}

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};

    // Print each number using a lambda expression
    forEach(numbers, [](const int& n) {
        std::cout << n << " ";
    });
    std::cout << std::endl; // Output: 1 2 3 4 5

    // Print each number multiplied by 2 using a lambda expression
    forEach(numbers, [](const int& n) {
        std::cout << n * 2 << " ";
    });
    std::cout << std::endl; // Output: 2 4 6 8 10

    return 0;
}
```

In this example, the `forEach` function template takes a `std::vector<T>` and a `std::function<void(const T&)>` callable. It applies the callable to each element of the vector. This allows for flexible and reusable code that can operate on any callable object with the specified signature.

#### Using `std::function` with Member Functions

`std::function` can also store member function pointers, allowing it to invoke member functions on objects. This can be particularly useful for callbacks that need to call methods on specific objects.

##### Example: Storing and Invoking Member Functions

```cpp
#include <functional>

#include <iostream>
#include <string>

class Printer {
public:
    void print(const std::string& message) const {
        std::cout << "Printer: " << message << std::endl;
    }
};

int main() {
    Printer printer;

    // Store a member function pointer in std::function
    std::function<void(const Printer&, const std::string&)> func = &Printer::print;

    // Invoke the member function on the printer object
    func(printer, "Hello, Member Function!"); // Output: Printer: Hello, Member Function!

    return 0;
}
```

In this example, `std::function<void(const Printer&, const std::string&)>` is used to store a pointer to the `Printer::print` member function. We then invoke the member function on a `Printer` object using the stored function pointer.

#### Advanced Usage: Chaining Callables

`std::function` can be used to chain multiple callable objects together, creating a sequence of operations. This can be useful for building pipelines or processing stages.

##### Example: Chaining Callables

```cpp
#include <functional>

#include <iostream>
#include <string>

#include <vector>

class Processor {
public:
    using Step = std::function<std::string(const std::string&)>;

    void addStep(Step step) {
        steps.push_back(std::move(step));
    }

    std::string process(const std::string& input) const {
        std::string result = input;
        for (const auto& step : steps) {
            result = step(result);
        }
        return result;
    }

private:
    std::vector<Step> steps;
};

int main() {
    Processor processor;

    // Add steps to the processor pipeline
    processor.addStep([](const std::string& input) {
        return input + " Step 1";
    });
    processor.addStep([](const std::string& input) {
        return input + " -> Step 2";
    });
    processor.addStep([](const std::string& input) {
        return input + " -> Step 3";
    });

    // Process an input string through the pipeline
    std::string result = processor.process("Start");
    std::cout << "Result: " << result << std::endl; // Output: Result: Start Step 1 -> Step 2 -> Step 3

    return 0;
}
```

In this example, the `Processor` class uses `std::function<std::string(const std::string&)>` to store a sequence of processing steps. The `addStep` method adds a new step to the pipeline, and the `process` method applies each step in sequence to the input string.

#### Conclusion

`std::function` is a versatile and powerful tool in the C++ standard library that enables type erasure and runtime polymorphism for callable objects. By providing a uniform interface for storing and invoking functions, lambda expressions, function objects, and member function pointers, `std::function` facilitates the creation of flexible and generic code.

In this subchapter, we explored the concepts and benefits of `std::function`, provided detailed examples of its usage, and demonstrated its practical applications in event handling, generic algorithms, member function pointers, and callable chaining. Understanding and utilizing `std::function` will enable you to write more flexible, efficient, and maintainable C++ code, enhancing your ability to handle a wide variety of callable objects dynamically.

### 5.4. Implementing Type Erasure for Custom Types

#### Introduction

While the C++ standard library provides utilities like `std::any` and `std::function` for type erasure, there are scenarios where custom type erasure implementations are necessary. Custom type erasure allows you to create flexible and reusable interfaces tailored to specific requirements, enabling you to abstract away the details of various concrete types while exposing a uniform interface. This technique is particularly useful for designing libraries, frameworks, or applications that need to handle heterogeneous types in a type-safe and efficient manner.

In this subchapter, we will explore how to implement type erasure for custom types. We will provide a step-by-step guide to creating a type-erased wrapper class, discuss the underlying principles, and demonstrate practical examples to illustrate the concepts.

#### Principles of Custom Type Erasure

Implementing type erasure for custom types typically involves the following steps:

1. **Define an Abstract Interface**: Specify the operations that the type-erased object must support.
2. **Implement Concrete Classes**: Define classes that implement the abstract interface.
3. **Create a Type-Erased Wrapper**: Implement a wrapper class that erases the type information of the concrete classes while exposing the abstract interface.
4. **Store and Invoke**: Store instances of concrete classes in the type-erased wrapper and invoke the operations through the abstract interface.

#### Step-by-Step Implementation

##### Step 1: Define an Abstract Interface

First, define an abstract interface that specifies the operations to be supported by the type-erased objects.

```cpp
#include <memory>

#include <iostream>

// Abstract interface
class IShape {
public:
    virtual ~IShape() = default;
    virtual void draw() const = 0;
    virtual void resize(double factor) = 0;
};
```

In this example, the `IShape` interface defines two pure virtual functions: `draw` and `resize`.

##### Step 2: Implement Concrete Classes

Next, define concrete classes that implement the abstract interface.

```cpp
class Circle : public IShape {
public:
    void draw() const override {
        std::cout << "Drawing Circle" << std::endl;
    }

    void resize(double factor) override {
        std::cout << "Resizing Circle by factor " << factor << std::endl;
    }
};

class Square : public IShape {
public:
    void draw() const override {
        std::cout << "Drawing Square" << std::endl;
    }

    void resize(double factor) override {
        std::cout << "Resizing Square by factor " << factor << std::endl;
    }
};
```

In these classes, the `draw` and `resize` methods are implemented to provide specific functionality for each shape.

##### Step 3: Create a Type-Erased Wrapper

Now, implement a type-erased wrapper class that erases the type information of the concrete classes while exposing the abstract interface.

```cpp
class Shape {
public:
    // Constructor for any type that implements IShape
    template <typename T>
    Shape(T shape) : impl(std::make_shared<Model<T>>(std::move(shape))) {}

    // Forward calls to the type-erased implementation
    void draw() const {
        impl->draw();
    }

    void resize(double factor) {
        impl->resize(factor);
    }

private:
    // Abstract base class for the type-erased model
    struct Concept {
        virtual ~Concept() = default;
        virtual void draw() const = 0;
        virtual void resize(double factor) = 0;
    };

    // Template derived class for the type-erased model
    template <typename T>
    struct Model : Concept {
        Model(T shape) : shape(std::move(shape)) {}

        void draw() const override {
            shape.draw();
        }

        void resize(double factor) override {
            shape.resize(factor);
        }

        T shape;
    };

    std::shared_ptr<const Concept> impl;
};
```

In this `Shape` class, we use the type erasure idiom to wrap any object that implements the `IShape` interface. The `Shape` class contains a pointer to a `Concept` object, which is an abstract base class with pure virtual functions. The `Model` template class derives from `Concept` and implements the interface by forwarding the calls to the wrapped object.

##### Step 4: Store and Invoke

Finally, create instances of `Circle` and `Square`, and store them in the type-erased `Shape` wrapper. Invoke the operations through the abstract interface.

```cpp
int main() {
    Shape circle = Circle();
    Shape square = Square();

    circle.draw();   // Output: Drawing Circle
    square.draw();   // Output: Drawing Square

    circle.resize(2.0); // Output: Resizing Circle by factor 2
    square.resize(3.0); // Output: Resizing Square by factor 3

    return 0;
}
```

In this example, the `Shape` wrapper can hold any object that implements the `IShape` interface, providing a uniform interface for drawing and resizing shapes.

#### Advanced Example: Type-Erased Container

Let's extend our example to create a type-erased container that can store various shapes and perform operations on all stored shapes.

##### Step 1: Define the Container

First, define a container class that can store multiple shapes.

```cpp
#include <vector>

class ShapeContainer {
public:
    // Add a shape to the container
    template <typename T>
    void addShape(T shape) {
        shapes.emplace_back(std::make_shared<Model<T>>(std::move(shape)));
    }

    // Draw all shapes in the container
    void drawAll() const {
        for (const auto& shape : shapes) {
            shape->draw();
        }
    }

    // Resize all shapes in the container
    void resizeAll(double factor) {
        for (const auto& shape : shapes) {
            shape->resize(factor);
        }
    }

private:
    // Abstract base class for the type-erased model
    struct Concept {
        virtual ~Concept() = default;
        virtual void draw() const = 0;
        virtual void resize(double factor) = 0;
    };

    // Template derived class for the type-erased model
    template <typename T>
    struct Model : Concept {
        Model(T shape) : shape(std::move(shape)) {}

        void draw() const override {
            shape.draw();
        }

        void resize(double factor) override {
            shape.resize(factor);
        }

        T shape;
    };

    std::vector<std::shared_ptr<const Concept>> shapes;
};
```

In this `ShapeContainer` class, we use the type erasure idiom to store various shapes. The `addShape` method adds a shape to the container, while the `drawAll` and `resizeAll` methods perform operations on all stored shapes.

##### Step 2: Use the Container

Create instances of `Circle` and `Square`, add them to the container, and perform operations on all stored shapes.

```cpp
int main() {
    ShapeContainer container;

    // Add shapes to the container
    container.addShape(Circle());
    container.addShape(Square());

    // Draw all shapes
    container.drawAll();
    // Output:
    // Drawing Circle
    // Drawing Square

    // Resize all shapes
    container.resizeAll(1.5);
    // Output:
    // Resizing Circle by factor 1.5
    // Resizing Square by factor 1.5

    return 0;
}
```

In this example, the `ShapeContainer` stores `Circle` and `Square` objects in a type-erased manner, providing a uniform interface for drawing and resizing all stored shapes.

#### Conclusion

Custom type erasure is a powerful technique in C++ that enables runtime polymorphism and flexible interfaces without relying on inheritance and virtual functions. By defining an abstract interface, implementing concrete classes, creating a type-erased wrapper, and invoking operations through the abstract interface, you can design flexible and reusable code that can handle heterogeneous types in a type-safe and efficient manner.

In this subchapter, we explored the principles of custom type erasure, provided a step-by-step guide to implementing type erasure for custom types, and demonstrated practical examples, including a type-erased container for shapes. Understanding and applying custom type erasure will enable you to write more flexible, efficient, and maintainable C++ code, enhancing your ability to design and implement complex systems and libraries.

### 5.5. Dynamic Polymorphism vs Static Polymorphism

#### Introduction

Polymorphism is a fundamental concept in object-oriented programming that allows objects to be treated as instances of their base type rather than their derived type. C++ supports two primary forms of polymorphism: dynamic polymorphism and static polymorphism. Understanding the differences, benefits, and use cases of each type is crucial for designing efficient and maintainable C++ applications.

Dynamic polymorphism is typically achieved through inheritance and virtual functions, providing flexibility and runtime type resolution. Static polymorphism, on the other hand, leverages templates and compile-time mechanisms like the Curiously Recurring Template Pattern (CRTP) to achieve polymorphic behavior without the overhead of runtime type checking.

In this subchapter, we will explore the concepts of dynamic and static polymorphism in detail, compare their benefits and trade-offs, and provide comprehensive examples to illustrate their usage.

#### Dynamic Polymorphism

Dynamic polymorphism is achieved using inheritance and virtual functions. This approach allows derived classes to override base class methods, and the appropriate method is determined at runtime based on the actual object type. Dynamic polymorphism provides flexibility and is useful when the exact types of objects cannot be determined until runtime.

##### Example: Dynamic Polymorphism with Virtual Functions

```cpp
#include <iostream>

#include <vector>

// Base class with virtual functions
class Shape {
public:
    virtual ~Shape() = default;
    virtual void draw() const = 0;
    virtual void resize(double factor) = 0;
};

// Derived class Circle
class Circle : public Shape {
public:
    void draw() const override {
        std::cout << "Drawing Circle" << std::endl;
    }

    void resize(double factor) override {
        std::cout << "Resizing Circle by factor " << factor << std::endl;
    }
};

// Derived class Square
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
    std::vector<Shape*> shapes = {new Circle(), new Square()};

    for (Shape* shape : shapes) {
        shape->draw();
        shape->resize(1.5);
    }

    // Cleanup
    for (Shape* shape : shapes) {
        delete shape;
    }

    return 0;
}
```

In this example, the `Shape` base class defines pure virtual functions `draw` and `resize`. The `Circle` and `Square` classes override these functions to provide specific implementations. The `shapes` vector stores pointers to `Shape` objects, and the appropriate `draw` and `resize` methods are called based on the actual object type at runtime.

#### Benefits of Dynamic Polymorphism

1. **Flexibility**: Dynamic polymorphism allows for flexibility in handling different object types through a common interface. This is particularly useful when the exact types of objects are not known until runtime.
2. **Runtime Type Resolution**: The appropriate method implementation is determined at runtime, enabling polymorphic behavior based on the actual object type.
3. **Extensibility**: New derived classes can be added without modifying existing code, promoting extensibility and maintainability.

#### Trade-offs of Dynamic Polymorphism

1. **Runtime Overhead**: Virtual function calls introduce runtime overhead due to the use of virtual tables (vtables) and dynamic dispatch.
2. **Type Safety**: Since type information is resolved at runtime, there is a risk of runtime errors if objects are not correctly cast or used.
3. **Complexity**: Inheritance hierarchies can become complex and difficult to manage, especially in large codebases.

#### Static Polymorphism

Static polymorphism is achieved using templates and compile-time mechanisms like the Curiously Recurring Template Pattern (CRTP). This approach allows for polymorphic behavior to be determined at compile time, eliminating the runtime overhead associated with virtual function calls. Static polymorphism provides type safety and can lead to more efficient code.

##### Example: Static Polymorphism with CRTP

```cpp
#include <iostream>

// Base class template using CRTP
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

// Derived class Circle
class Circle : public Shape<Circle> {
public:
    void draw() const {
        std::cout << "Drawing Circle" << std::endl;
    }

    void resize(double factor) {
        std::cout << "Resizing Circle by factor " << factor << std::endl;
    }
};

// Derived class Square
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

In this example, the `Shape` base class template uses CRTP to achieve static polymorphism. The `Circle` and `Square` classes derive from `Shape<Circle>` and `Shape<Square>`, respectively, and implement the `draw` and `resize` methods. The appropriate method implementations are determined at compile time using `static_cast`.

#### Benefits of Static Polymorphism

1. **Performance**: Static polymorphism eliminates the runtime overhead of virtual function calls, leading to more efficient code.
2. **Type Safety**: Type relationships are enforced at compile time, reducing the risk of runtime errors.
3. **Inlining**: The compiler can inline function calls, further optimizing performance.
4. **Simpler Code**: Avoids the complexity of inheritance hierarchies and dynamic dispatch, resulting in simpler and more maintainable code.

#### Trade-offs of Static Polymorphism

1. **Code Bloat**: Template instantiation can lead to code bloat, as separate copies of the template code are generated for each type.
2. **Compile-Time Complexity**: Errors related to static polymorphism are often detected at compile time, which can result in more complex error messages and longer compile times.
3. **Limited Flexibility**: Static polymorphism requires the exact types to be known at compile time, which can limit flexibility in certain scenarios.

#### Comparing Dynamic and Static Polymorphism

To better understand the differences between dynamic and static polymorphism, let's compare their characteristics and use cases.

##### Dynamic Polymorphism

- **Flexibility**: Suitable for scenarios where the exact types of objects are not known until runtime.
- **Runtime Overhead**: Involves runtime overhead due to virtual function calls and dynamic dispatch.
- **Type Safety**: Type relationships are checked at runtime, with potential for runtime errors.
- **Extensibility**: Easily extensible by adding new derived classes without modifying existing code.
- **Usage**: Commonly used in object-oriented designs and frameworks where runtime flexibility is essential.

##### Static Polymorphism

- **Performance**: Provides better performance by eliminating the runtime overhead of virtual function calls.
- **Type Safety**: Ensures type safety at compile time, reducing the risk of runtime errors.
- **Inlining**: Allows the compiler to inline function calls, further optimizing performance.
- **Code Bloat**: Can lead to code bloat due to template instantiation.
- **Usage**: Suitable for scenarios where the exact types are known at compile time and performance is critical, such as in high-performance libraries and systems programming.

#### Practical Example: Polymorphic Container

Let's consider a practical example of a polymorphic container that can store and operate on different shapes using both dynamic and static polymorphism.

##### Dynamic Polymorphism Example

```cpp
#include <iostream>

#include <vector>
#include <memory>

// Base class with virtual functions
class Shape {
public:
    virtual ~Shape() = default;
    virtual void draw() const = 0;
    virtual void resize(double factor) = 0;
};

// Derived class Circle
class Circle : public Shape {
public:
    void draw() const override {
        std::cout << "Drawing Circle" << std::endl;
    }

    void resize(double factor) override {
        std::cout << "Resizing Circle by factor " << factor << std::endl;
    }
};

// Derived class Square
class Square : public Shape {
public:
    void draw() const override {
        std::cout << "Drawing Square" << std::endl;
    }

    void resize(double factor) override {
        std::cout << "Resizing Square by factor " << factor << std::endl;
    }
};

// Polymorphic container using dynamic polymorphism
class ShapeContainer {
public:
    void addShape(std::shared_ptr<Shape> shape) {
        shapes.push_back(std::move(shape));
    }

    void drawAll() const {
        for (const auto& shape : shapes) {
            shape->draw();
        }
    }

    void resizeAll(double factor) {
        for (const auto& shape : shapes) {
            shape->resize(factor);
        }
    }

private:
    std::vector<std::shared_ptr<Shape>> shapes;
};

int main() {
    ShapeContainer container;
    container.addShape(std::make_shared<Circle>());
    container.addShape(std::make_shared<Square>());

    container.drawAll();
    // Output:


    // Drawing Circle
    // Drawing Square

    container.resizeAll(1.5);
    // Output:
    // Resizing Circle by factor 1.5
    // Resizing Square by factor 1.5

    return 0;
}
```

In this example, the `ShapeContainer` uses dynamic polymorphism to store and operate on shapes. The `addShape` method adds shapes to the container, and the `drawAll` and `resizeAll` methods perform operations on all stored shapes using virtual function calls.

##### Static Polymorphism Example

```cpp
#include <iostream>

#include <vector>

// Base class template using CRTP
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

// Derived class Circle
class Circle : public Shape<Circle> {
public:
    void draw() const {
        std::cout << "Drawing Circle" << std::endl;
    }

    void resize(double factor) {
        std::cout << "Resizing Circle by factor " << factor << std::endl;
    }
};

// Derived class Square
class Square : public Shape<Square> {
public:
    void draw() const {
        std::cout << "Drawing Square" << std::endl;
    }

    void resize(double factor) {
        std::cout << "Resizing Square by factor " << factor << std::endl;
    }
};

// Polymorphic container using static polymorphism
template <typename ShapeType>
class ShapeContainer {
public:
    void addShape(ShapeType shape) {
        shapes.push_back(std::move(shape));
    }

    void drawAll() const {
        for (const auto& shape : shapes) {
            shape.draw();
        }
    }

    void resizeAll(double factor) {
        for (const auto& shape : shapes) {
            shape.resize(factor);
        }
    }

private:
    std::vector<ShapeType> shapes;
};

int main() {
    ShapeContainer<Circle> circleContainer;
    ShapeContainer<Square> squareContainer;

    circleContainer.addShape(Circle());
    squareContainer.addShape(Square());

    circleContainer.drawAll();
    // Output: Drawing Circle

    squareContainer.drawAll();
    // Output: Drawing Square

    circleContainer.resizeAll(1.5);
    // Output: Resizing Circle by factor 1.5

    squareContainer.resizeAll(2.0);
    // Output: Resizing Square by factor 2

    return 0;
}
```

In this example, the `ShapeContainer` template uses static polymorphism to store and operate on shapes. The `addShape` method adds shapes to the container, and the `drawAll` and `resizeAll` methods perform operations on all stored shapes using compile-time polymorphism with CRTP.

#### Conclusion

Both dynamic and static polymorphism are powerful techniques in C++ that enable polymorphic behavior and code reuse. Dynamic polymorphism, achieved through inheritance and virtual functions, offers flexibility and runtime type resolution, making it suitable for scenarios where the exact types of objects are not known until runtime. However, it comes with runtime overhead and potential complexity in managing inheritance hierarchies.

Static polymorphism, achieved through templates and compile-time mechanisms like CRTP, provides better performance and type safety by resolving polymorphic behavior at compile time. It eliminates the runtime overhead associated with virtual function calls and allows for function inlining, leading to more efficient code. However, it requires the exact types to be known at compile time and can result in code bloat due to template instantiation.

Understanding the differences, benefits, and trade-offs between dynamic and static polymorphism will enable you to choose the appropriate technique for your specific use case, leading to more efficient, maintainable, and flexible C++ code.
