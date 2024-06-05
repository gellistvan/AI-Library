
\newpage
## Chapter 2: Template Metaprogramming (TMP) and Modern C++

Template Metaprogramming (TMP) is an advanced C++ programming technique that leverages the power of templates to perform computations at compile time. With the advent of modern C++ standards, TMP has become more accessible, efficient, and powerful, thanks to features like variadic templates, constexpr, and improved type traits. This chapter delves into the intricacies of TMP, exploring how these modern features can be harnessed to write more expressive, flexible, and performant code.

We will start by revisiting the fundamentals of TMP, ensuring a solid understanding of basic concepts such as recursive templates and SFINAE (Substitution Failure Is Not An Error). From there, we will explore the enhancements brought by modern C++ standards, demonstrating how these features simplify and extend the capabilities of TMP. Key topics include variadic templates for handling arbitrary numbers of parameters, constexpr for compile-time evaluations, and type traits for type introspection and manipulation.

By the end of this chapter, you will have a deep understanding of how to use TMP to create elegant solutions to complex problems, leveraging the full potential of modern C++. Whether you are optimizing code for performance, reducing runtime overhead, or creating highly generic libraries, TMP offers powerful techniques that will elevate your C++ programming skills to the next level.

### 2.1. TMP in C++11 and Beyond

#### Introduction to TMP in Modern C++

Template Metaprogramming (TMP) has been a powerful technique in C++ for a long time, but the advent of C++11 and subsequent standards has revolutionized the way TMP is used and extended its capabilities significantly. Modern C++ introduces several new features that make TMP more expressive, easier to use, and more efficient. This subchapter will explore these features, demonstrating how they enhance TMP and provide practical examples of their use.

#### Variadic Templates

One of the most significant features introduced in C++11 is variadic templates, which allow templates to accept an arbitrary number of arguments. This capability is particularly useful in TMP, as it simplifies the creation of flexible and generic templates.

##### Example: Variadic Print Function

```cpp
#include <iostream>

// Base case: no arguments
void print() {
    std::cout << "End of arguments." << std::endl;
}

// Recursive case: one or more arguments
template<typename T, typename... Args>
void print(T first, Args... args) {
    std::cout << first << std::endl;
    print(args...); // Recursive call with remaining arguments
}

int main() {
    print(1, 2.5, "Hello", 'A');
    // Output:
    // 1
    // 2.5
    // Hello
    // A
    // End of arguments.
    return 0;
}
```

In this example, the `print` function uses variadic templates to handle an arbitrary number of arguments. The base case handles the scenario when no arguments are left, while the recursive case processes the first argument and recursively calls itself with the remaining arguments.

#### `constexpr` and Compile-Time Computations

The `constexpr` keyword, introduced in C++11 and enhanced in later standards, allows functions and variables to be evaluated at compile time. This feature is crucial for TMP, as it enables more complex compile-time computations and optimizations.

##### Example: Compile-Time Factorial

```cpp
#include <iostream>

// Constexpr function for compile-time factorial calculation
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : (n * factorial(n - 1));
}

int main() {
    constexpr int result = factorial(5);
    std::cout << "Factorial of 5 is " << result << std::endl; // Output: 120
    return 0;
}
```

In this example, the `factorial` function is marked as `constexpr`, allowing it to be evaluated at compile time. The result is computed during compilation, eliminating runtime overhead.

#### Type Traits and Type Manipulations

C++11 introduced a rich set of type traits in the `<type_traits>` header, which provide tools for querying and manipulating types at compile time. These traits are essential for TMP, enabling more sophisticated type checks and transformations.

##### Example: Type Traits and `enable_if`

```cpp
#include <iostream>

#include <type_traits>

// Function enabled only for integral types
template<typename T>
typename std::enable_if<std::is_integral<T>::value, void>::type process(T value) {
    std::cout << value << " is an integral type." << std::endl;
}

// Function enabled only for floating-point types
template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, void>::type process(T value) {
    std::cout << value << " is a floating-point type." << std::endl;
}

int main() {
    process(42);       // Output: 42 is an integral type.
    process(3.14);     // Output: 3.14 is a floating-point type.
    // process("text"); // Compilation error: no matching function to call 'process'
    return 0;
}
```

In this example, `std::enable_if` and type traits (`std::is_integral` and `std::is_floating_point`) are used to conditionally enable the `process` function for integral and floating-point types.

#### `decltype` and `auto`

C++11 introduced the `decltype` and `auto` keywords, which simplify type declarations and type deductions. These features are beneficial for TMP, making the code more concise and easier to read.

##### Example: Using `decltype` and `auto`

```cpp
#include <iostream>

#include <type_traits>

// Function to add two values
template<typename T1, typename T2>
auto add(T1 a, T2 b) -> decltype(a + b) {
    return a + b;
}

int main() {
    auto result = add(1, 2.5);
    std::cout << "Result of add(1, 2.5): " << result << std::endl; // Output: 3.5
    std::cout << "Type of result: " << typeid(result).name() << std::endl; // Output: d (double)
    return 0;
}
```

In this example, the `add` function uses `auto` and `decltype` to deduce the return type based on the types of the input parameters. This feature allows for more flexible and generic function definitions.

#### Fold Expressions

C++17 introduced fold expressions, which provide a concise way to apply binary operators to parameter packs. This feature simplifies many common TMP tasks, such as implementing variadic functions.

##### Example: Sum Function Using Fold Expressions

```cpp
#include <iostream>

// Variadic sum function using fold expression
template<typename... Args>
auto sum(Args... args) {
    return (args + ...); // Fold expression
}

int main() {
    std::cout << "Sum: " << sum(1, 2, 3, 4, 5) << std::endl; // Output: 15
    std::cout << "Sum: " << sum(1.1, 2.2, 3.3) << std::endl; // Output: 6.6
    return 0;
}
```

In this example, the `sum` function uses a fold expression to compute the sum of its arguments. The fold expression `(args + ...)` applies the `+` operator to each element in the parameter pack.

#### Concepts

C++20 introduced concepts, which provide a way to specify constraints on template parameters. Concepts improve the readability and maintainability of TMP code by making constraints explicit and providing better error messages.

##### Example: Using Concepts to Constrain Templates

```cpp
#include <iostream>

#include <concepts>

template<typename T>
concept Integral = std::is_integral_v<T>;

template<typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

// Function enabled only for integral types
void process(Integral auto value) {
    std::cout << value << " is an integral type." << std::endl;
}

// Function enabled only for floating-point types
void process(FloatingPoint auto value) {
    std::cout << value << " is a floating-point type." << std::endl;
}

int main() {
    process(42);       // Output: 42 is an integral type.
    process(3.14);     // Output: 3.14 is a floating-point type.
    // process("text"); // Compilation error: no matching function to call 'process'
    return 0;
}
```

In this example, concepts (`Integral` and `FloatingPoint`) are used to constrain the `process` function templates. This approach makes the intent clear and improves the readability of the code.

#### Practical Example: Compile-Time Matrix Multiplication

Combining these modern C++ features, let's implement a compile-time matrix multiplication algorithm.

```cpp
#include <iostream>

#include <array>

// Compile-time matrix multiplication
template<std::size_t N, std::size_t M, std::size_t P>
constexpr std::array<std::array<int, P>, N> multiply(const std::array<std::array<int, M>, N>& a, const std::array<std::array<int, P>, M>& b) {
    std::array<std::array<int, P>, N> result = {};

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < P; ++j) {
            int sum = 0;
            for (std::size_t k = 0; k < M; ++k) {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }

    return result;
}

int main() {
    constexpr std::array<std::array<int, 2>, 2> a = {{{1, 2}, {3, 4}}};
    constexpr std::array<std::array<int, 2>, 2> b = {{{5, 6}, {7, 8}}};
    constexpr auto result = multiply(a, b);

    for (const auto& row : result) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    // Output:
    // 19 22
    // 43 50

    return 0;
}
```

In this example, the `multiply` function performs matrix multiplication at compile time using `constexpr` and other modern C++ features. The result is computed during compilation, demonstrating the power of TMP in modern C++.

#### Conclusion

Template Metaprogramming (TMP) has evolved significantly with the introduction of modern C++ standards, making it more powerful and easier to use. Features like variadic templates, `constexpr`, type traits, `decltype`, fold expressions, and concepts have expanded the possibilities of TMP, enabling more expressive, flexible, and efficient code. By mastering these features, you can leverage the full potential of TMP to create sophisticated and high-performance C++ applications.

### 2.2. Leveraging `constexpr` and `consteval`

#### Introduction to `constexpr` and `consteval`

The introduction of `constexpr` in C++11 and the subsequent enhancement in C++14 and C++17, along with the addition of `consteval` in C++20, have significantly advanced the capabilities of compile-time computations in C++. These features allow functions and variables to be evaluated at compile time, providing a powerful toolset for optimizing performance and ensuring correctness. This subchapter will delve into the details of `constexpr` and `consteval`, exploring their usage, benefits, and practical applications with detailed examples.

#### `constexpr` in Modern C++

`constexpr` was introduced in C++11 to enable functions to be evaluated at compile time. A `constexpr` function or variable can be evaluated at compile time if all its arguments are compile-time constants. This feature allows for significant optimizations by embedding the results directly into the compiled binary, eliminating runtime overhead.

##### `constexpr` Functions

A `constexpr` function is a function that can be evaluated at compile time. Let's start with a simple example of a `constexpr` function that computes the factorial of a number:

```cpp
#include <iostream>

// Constexpr function for compile-time factorial calculation
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : (n * factorial(n - 1));
}

int main() {
    constexpr int result = factorial(5);
    std::cout << "Factorial of 5 is " << result << std::endl; // Output: 120
    return 0;
}
```

In this example, the `factorial` function is marked as `constexpr`, allowing it to be evaluated at compile time. The result is computed during compilation, eliminating the need for runtime computation.

##### `constexpr` Variables

`constexpr` can also be applied to variables, indicating that their value is constant and can be evaluated at compile time.

```cpp
#include <iostream>

// Constexpr variable
constexpr int max_size = 100;

int main() {
    int array[max_size]; // Valid because max_size is constexpr
    std::cout << "Array size: " << max_size << std::endl; // Output: 100
    return 0;
}
```

In this example, `max_size` is a `constexpr` variable, which allows it to be used in contexts where a constant expression is required, such as array sizes.

#### Enhancements in C++14 and C++17

C++14 and C++17 introduced enhancements to `constexpr`, making it more powerful and flexible. These enhancements include the ability to write more complex `constexpr` functions and the use of `constexpr` in more contexts.

##### C++14: Relaxed `constexpr` Restrictions

C++14 relaxed many of the restrictions on `constexpr` functions, allowing them to contain more complex control flow constructs, such as loops and multiple return statements.

```cpp
#include <iostream>

// Constexpr function with loop (C++14)
constexpr int factorial(int n) {
    int result = 1;
    for (int i = 1; i <= n; ++i) {
        result *= i;
    }
    return result;
}

int main() {
    constexpr int result = factorial(5);
    std::cout << "Factorial of 5 is " << result << std::endl; // Output: 120
    return 0;
}
```

In this example, the `factorial` function uses a loop to compute the factorial, which is allowed in C++14.

##### C++17: `constexpr` for `if` Statements and Switch Cases

C++17 further enhanced `constexpr` by allowing the use of `if` statements and `switch` cases within `constexpr` functions.

```cpp
#include <iostream>

// Constexpr function with if statements (C++17)
constexpr int fibonacci(int n) {
    if (n <= 1) {
        return n;
    } else {
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
}

int main() {
    constexpr int result = fibonacci(10);
    std::cout << "Fibonacci of 10 is " << result << std::endl; // Output: 55
    return 0;
}
```

In this example, the `fibonacci` function uses `if` statements to compute the Fibonacci sequence, which is allowed in C++17.

#### `consteval` in C++20

C++20 introduced `consteval`, a keyword that guarantees a function is evaluated at compile time. Unlike `constexpr`, which allows a function to be evaluated at compile time but does not require it, `consteval` mandates compile-time evaluation. This feature is useful for ensuring that certain computations are performed during compilation, improving performance and guaranteeing correctness.

##### `consteval` Functions

A `consteval` function must be evaluated at compile time. If it is called in a context where compile-time evaluation is not possible, the code will not compile.

```cpp
#include <iostream>

// Consteval function for compile-time square calculation (C++20)
consteval int square(int n) {
    return n * n;
}

int main() {
    constexpr int result = square(5);
    std::cout << "Square of 5 is " << result << std::endl; // Output: 25

    // int runtime_result = square(5); // Error: consteval function must be evaluated at compile time
    return 0;
}
```

In this example, the `square` function is marked as `consteval`, ensuring that it is evaluated at compile time. Attempting to call it in a runtime context results in a compilation error.

#### Practical Applications of `constexpr` and `consteval`

`constexpr` and `consteval` can be leveraged for various practical applications, including compile-time data structures, constant expressions, and compile-time validation.

##### Compile-Time Data Structures

Compile-time data structures can be implemented using `constexpr` and `consteval` to ensure that their operations are performed during compilation. Let's implement a compile-time fixed-size vector using `constexpr`.

```cpp
#include <iostream>

#include <array>

// Compile-time vector class
template<typename T, std::size_t N>
class Vector {
public:
    constexpr Vector() : data{} {}

    constexpr T& operator[](std::size_t index) {
        return data[index];
    }

    constexpr const T& operator[](std::size_t index) const {
        return data[index];
    }

    constexpr std::size_t size() const {
        return N;
    }

private:
    std::array<T, N> data;
};

int main() {
    constexpr Vector<int, 5> vec;
    static_assert(vec.size() == 5, "Size check failed");
    
    // Print elements of the compile-time vector
    for (std::size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << " "; // Default-initialized to zero
    }
    std::cout << std::endl;

    return 0;
}
```

In this example, the `Vector` class template provides a simple implementation of a fixed-size vector that can be evaluated at compile time. The `size` member function and the element access operators are marked as `constexpr` to enable compile-time evaluation.

##### Constant Expressions

Constant expressions can be used to create complex compile-time constants. Let's create a compile-time constant expression for a mathematical function.

```cpp
#include <iostream>

// Constexpr function for compile-time calculation of a polynomial
constexpr double polynomial(double x) {
    return 3 * x * x + 2 * x + 1;
}

int main() {
    constexpr double result = polynomial(5.0);
    std::cout << "Polynomial(5.0) is " << result << std::endl; // Output: 86
    return 0;
}
```

In this example, the `polynomial` function computes the value of a polynomial at compile time, demonstrating how `constexpr` can be used to create complex compile-time constants.

##### Compile-Time Validation

`consteval` can be used for compile-time validation, ensuring that certain conditions are met during compilation.

```cpp
#include <iostream>

// Consteval function for compile-time validation (C++20)
consteval int validate_positive(int n) {
    if (n <= 0) {
        throw "Value must be positive";
    }
    return n;
}

int main() {
    constexpr int value = validate_positive(10);
    std::cout << "Validated value: " << value << std::endl; // Output: 10

    // constexpr int invalid_value = validate_positive(-5); // Error: Value must be positive
    return 0;
}
```

In this example, the `validate_positive` function ensures that the input value is positive and throws an error if it is not. This validation is performed at compile time, ensuring that invalid values are caught early.

#### Conclusion

`constexpr` and `consteval` are powerful features in modern C++ that enable compile-time computations and validations. By leveraging these features, developers can create more efficient, optimized, and reliable code. `constexpr` allows for flexible compile-time evaluation, while `consteval` guarantees that functions are evaluated during compilation. Together, these features provide a robust toolset for advanced C++ programming, enabling sophisticated compile-time computations and ensuring correctness. Understanding and utilizing `constexpr` and `consteval` will significantly enhance your ability to write high-performance and reliable C++ applications.

### 2.3. Concepts and Ranges in C++20

#### Introduction to Concepts

Concepts, introduced in C++20, are a powerful feature that enhances the readability, maintainability, and safety of template metaprogramming. Concepts allow you to specify constraints on template parameters, ensuring that they meet certain requirements. This leads to clearer error messages and more expressive code. Concepts can be seen as a way to enforce compile-time requirements on template arguments, making it easier to write and understand generic code.

#### Basics of Concepts

A concept is a compile-time predicate that specifies requirements for types. You can define a concept using the `concept` keyword. Here’s a simple example that defines a concept to check if a type is integral:

```cpp
#include <iostream>

#include <type_traits>

// Define a concept to check if a type is integral
template<typename T>
concept Integral = std::is_integral_v<T>;

// Function constrained by the Integral concept
void print_integral(Integral auto value) {
    std::cout << value << " is an integral type." << std::endl;
}

int main() {
    print_integral(42);   // Output: 42 is an integral type.
    // print_integral(3.14); // Error: no matching function to call 'print_integral'
    return 0;
}
```

In this example, the `Integral` concept is defined using `std::is_integral_v`. The `print_integral` function is constrained by the `Integral` concept, ensuring that it only accepts integral types.

#### Using Concepts with Function Templates

Concepts can be used to constrain function templates, providing more precise control over template parameter requirements. Let’s see how to use concepts to constrain a function template:

```cpp
#include <iostream>

#include <concepts>

// Define a concept to check if a type is arithmetic
template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

// Function template constrained by the Arithmetic concept
template<Arithmetic T>
T add(T a, T b) {
    return a + b;
}

int main() {
    std::cout << add(3, 4) << std::endl;       // Output: 7
    std::cout << add(3.14, 2.86) << std::endl; // Output: 6
    // std::cout << add("Hello", "World") << std::endl; // Error: no matching function to call 'add'
    return 0;
}
```

In this example, the `add` function template is constrained by the `Arithmetic` concept, ensuring that it only accepts arithmetic types (integral and floating-point types).

#### Combining Concepts

You can combine multiple concepts to create more specific constraints. This can be done using logical operators such as `&&` (and), `||` (or), and `!` (not).

##### Example: Combining Concepts

```cpp
#include <iostream>

#include <concepts>

// Define concepts for integral and floating-point types
template<typename T>
concept Integral = std::is_integral_v<T>;

template<typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

// Function template constrained by combined concepts
template<typename T>
requires Integral<T> || FloatingPoint<T>
T multiply(T a, T b) {
    return a * b;
}

int main() {
    std::cout << multiply(3, 4) << std::endl;       // Output: 12
    std::cout << multiply(3.14, 2.0) << std::endl;  // Output: 6.28
    // std::cout << multiply("Hello", "World") << std::endl; // Error: no matching function to call 'multiply'
    return 0;
}
```

In this example, the `multiply` function template is constrained by a combination of the `Integral` and `FloatingPoint` concepts, ensuring that it only accepts integral or floating-point types.

#### Introduction to Ranges

Ranges, introduced in C++20, provide a new way to work with sequences of elements. The Ranges library offers a more powerful and expressive alternative to the traditional STL algorithms and iterators, making code easier to read and write. Ranges integrate seamlessly with concepts, enabling compile-time checks on the properties of the sequences being manipulated.

#### Basics of Ranges

Ranges provide a unified interface for working with sequences of elements. They are designed to work with the existing STL containers and iterators, but with a more expressive and concise syntax. Let’s start with a simple example that uses ranges to manipulate a sequence of integers:

```cpp
#include <iostream>

#include <vector>
#include <ranges>

// Basic usage of ranges
int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // Use ranges to create a view of the elements
    auto view = vec | std::ranges::views::reverse;

    // Print the elements of the view
    for (int i : view) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    // Output: 5 4 3 2 1

    return 0;
}
```

In this example, we use ranges to create a view of the elements in `vec` in reverse order. The `std::ranges::views::reverse` adaptor creates a reversed view of the original sequence, which we then iterate over and print.

#### Range Adaptors

Range adaptors are a key feature of the Ranges library. They allow you to create views of sequences that apply transformations or filters lazily. Some common range adaptors include `filter`, `transform`, and `take`.

##### Example: Using Range Adaptors

```cpp
#include <iostream>

#include <vector>
#include <ranges>

// Function to demonstrate range adaptors
int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Create a view that filters out even numbers and multiplies the remaining numbers by 2
    auto view = vec
        | std::ranges::views::filter([](int n) { return n % 2 != 0; })
        | std::ranges::views::transform([](int n) { return n * 2; });

    // Print the elements of the view
    for (int i : view) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    // Output: 2 6 10 14 18

    return 0;
}
```

In this example, we use the `filter` adaptor to create a view that contains only the odd numbers from `vec`, and then use the `transform` adaptor to multiply each remaining number by 2. The resulting view is printed to the console.

#### Range Algorithms

The Ranges library also provides a set of range-based algorithms that work seamlessly with range adaptors and views. These algorithms are similar to the traditional STL algorithms but are designed to work with the Range concept.

##### Example: Using Range Algorithms

```cpp
#include <iostream>

#include <vector>
#include <ranges>

#include <algorithm>

// Function to demonstrate range algorithms
int main() {
    std::vector<int> vec = {10, 20, 30, 40, 50};

    // Create a view that takes the first 3 elements and then sorts them in descending order
    auto view = vec
        | std::ranges::views::take(3)
        | std::ranges::views::transform([](int n) { return n + 1; });

    // Print the elements of the view
    for (int i : view) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    // Output: 11 21 31

    // Sort the original vector using range algorithms
    std::ranges::sort(vec);

    // Print the sorted vector
    for (int i : vec) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    // Output: 10 20 30 40 50

    return 0;
}
```

In this example, we use the `take` adaptor to create a view that contains the first three elements of `vec`, and then use the `transform` adaptor to increment each element by 1. We also demonstrate the use of the `std::ranges::sort` algorithm to sort the original vector.

#### Combining Concepts and Ranges

Concepts and ranges can be combined to create powerful, expressive, and type-safe code. By using concepts to constrain range-based algorithms and views, you can ensure that your code meets specific requirements at compile time.

##### Example: Combining Concepts and Ranges

```cpp
#include <iostream>

#include <vector>
#include <ranges>

#include <concepts>

// Define a concept to check if a type is a range of integral elements
template<typename R>
concept IntegralRange = std::ranges::range<R> && std::integral<std::ranges::range_value_t<R>>;

// Function to print the elements of a range that satisfies the IntegralRange concept
void print_integral_range(IntegralRange auto&& range) {
    for (auto&& value : range) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    print_integral_range(vec); // Output: 1 2 3 4 5

    // std::vector<std::string> str_vec = {"Hello", "World"};
    //

    // Error: no matching function to call 'print_integral_range'
    print_integral_range(str_vec); 

    return 0;
}
```

In this example, we define an `IntegralRange` concept that checks if a type is a range of integral elements. The `print_integral_range` function is constrained by the `IntegralRange` concept, ensuring that it only accepts ranges of integral elements.

#### Practical Applications of Concepts and Ranges

Concepts and ranges can be applied to various practical scenarios to improve code clarity, maintainability, and safety. Let’s explore some real-world examples.

##### Example: Filtering and Transforming Data

```cpp
#include <iostream>

#include <vector>
#include <ranges>

// Function to filter and transform data using ranges
void filter_and_transform(std::vector<int>& data) {
    auto view = data
        | std::ranges::views::filter([](int n) { return n % 2 == 0; })
        | std::ranges::views::transform([](int n) { return n * 10; });

    // Print the elements of the view
    for (int value : view) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    filter_and_transform(data); // Output: 20 40 60 80 100
    return 0;
}
```

In this example, we use ranges to filter and transform a vector of integers. The `filter` adaptor selects even numbers, and the `transform` adaptor multiplies each selected number by 10.

#### Conclusion

Concepts and ranges in C++20 represent a significant advancement in the language, providing tools that enhance the expressiveness, readability, and safety of generic programming. Concepts allow you to specify precise requirements for template parameters, leading to better error messages and more maintainable code. Ranges offer a unified and powerful way to work with sequences of elements, integrating seamlessly with the existing STL and enabling more expressive and concise code.

By leveraging concepts and ranges, you can write modern C++ code that is not only efficient and robust but also clear and easy to understand. These features are essential tools in the arsenal of any advanced C++ programmer, enabling the creation of sophisticated and high-performance applications.

### 2.4. Metaprogramming with C++23 and Beyond

#### Introduction to Modern Metaprogramming

Metaprogramming has always been a powerful feature of C++, enabling developers to write code that manipulates types and values at compile time. With each new standard, C++ continues to evolve, introducing new features that enhance metaprogramming capabilities. C++23 and beyond bring several exciting advancements, further simplifying and extending the power of template metaprogramming. This subchapter explores these new features, providing detailed examples of how they can be used to write more expressive, efficient, and maintainable metaprograms.

#### New Features in C++23

C++23 introduces several enhancements that significantly impact metaprogramming. These include improved constexpr capabilities, extended type traits, enhanced template syntax, and new standard library components. Let's explore these features in detail.

#### Extended `constexpr` Capabilities

C++23 continues to expand the capabilities of `constexpr`, allowing more complex computations to be performed at compile time. One notable enhancement is the ability to use dynamic memory allocation within `constexpr` functions, making it possible to create more sophisticated compile-time data structures.

##### Example: `constexpr` Vector with Dynamic Memory Allocation

```cpp
#include <iostream>

#include <vector>

// Constexpr function to calculate Fibonacci sequence
constexpr std::vector<int> fibonacci(int n) {
    std::vector<int> fib(n);
    fib[0] = 0;
    fib[1] = 1;
    for (int i = 2; i < n; ++i) {
        fib[i] = fib[i - 1] + fib[i - 2];
    }
    return fib;
}

int main() {
    constexpr auto fib_seq = fibonacci(10);
    for (int val : fib_seq) {
        std::cout << val << " ";
    }
    std::cout << std::endl; // Output: 0 1 1 2 3 5 8 13 21 34
    return 0;
}
```

In this example, the `fibonacci` function computes the Fibonacci sequence at compile time using a `constexpr` vector. This capability allows for more complex and dynamic compile-time computations.

#### Enhanced Type Traits

C++23 introduces several new type traits and improves existing ones, making type manipulations more powerful and expressive. These enhancements include traits for detecting more complex type properties and enabling more precise type transformations.

##### Example: New Type Traits in C++23

```cpp
#include <iostream>

#include <type_traits>

// Custom type trait to check if a type is a const pointer
template<typename T>
struct is_const_pointer : std::false_type {};

template<typename T>
struct is_const_pointer<const T*> : std::true_type {};

// Using new type traits in C++23
int main() {
    std::cout << std::boolalpha;
    std::cout << "is_const_pointer<int>::value: " << is_const_pointer<int>::value << std::endl; // Output: false
    std::cout << "is_const_pointer<const int*>::value: " << is_const_pointer<const int*>::value << std::endl; // Output: true
    return 0;
}
```

In this example, we define a custom type trait `is_const_pointer` that checks if a type is a const pointer. The new type traits introduced in C++23 can be used in conjunction with custom type traits to perform more sophisticated type checks and manipulations.

#### Enhanced Template Syntax

C++23 introduces enhancements to template syntax, making templates more flexible and expressive. These enhancements include improved support for template parameters and better integration with concepts.

##### Example: Improved Template Parameter Support

```cpp
#include <iostream>

#include <concepts>

// Define a concept for numeric types
template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

// Template function using improved syntax
template<Numeric T, Numeric U>
auto add(T a, U b) {
    return a + b;
}

int main() {
    std::cout << add(3, 4.5) << std::endl; // Output: 7.5
    return 0;
}
```

In this example, the `add` function template uses the `Numeric` concept to constrain its parameters. The improved template syntax in C++23 allows for more concise and readable code.

#### New Standard Library Components

C++23 introduces several new components in the standard library that enhance metaprogramming capabilities. These include new algorithms, improved utilities, and better support for compile-time computations.

##### Example: Using New Standard Library Components

```cpp
#include <iostream>

#include <ranges>
#include <algorithm>

// Function to demonstrate new standard library components
void use_new_components() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // Use ranges to create a view of the elements
    auto view = vec | std::views::filter([](int n) { return n % 2 == 0; })
                    | std::views::transform([](int n) { return n * n; });

    // Print the elements of the view
    for (int i : view) {
        std::cout << i << " ";
    }
    std::cout << std::endl; // Output: 4 16
}

int main() {
    use_new_components();
    return 0;
}
```

In this example, we use the new range adaptors introduced in C++23 to filter and transform a vector of integers. The `filter` adaptor selects even numbers, and the `transform` adaptor squares each selected number.

#### Practical Applications of Metaprogramming in C++23

Metaprogramming in C++23 and beyond opens up new possibilities for optimizing performance, improving code maintainability, and ensuring correctness. Let's explore some practical applications.

##### Example: Compile-Time Matrix Multiplication

Combining the new `constexpr` capabilities and enhanced type traits, we can implement a compile-time matrix multiplication algorithm.

```cpp
#include <iostream>

#include <array>

// Compile-time matrix multiplication
template<std::size_t N, std::size_t M, std::size_t P>
constexpr auto multiply(const std::array<std::array<int, M>, N>& a, const std::array<std::array<int, P>, M>& b) {
    std::array<std::array<int, P>, N> result = {};

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < P; ++j) {
            int sum = 0;
            for (std::size_t k = 0; k < M; ++k) {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }

    return result;
}

int main() {
    constexpr std::array<std::array<int, 2>, 2> a = {{{1, 2}, {3, 4}}};
    constexpr std::array<std::array<int, 2>, 2> b = {{{5, 6}, {7, 8}}};
    constexpr auto result = multiply(a, b);

    for (const auto& row : result) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    // Output:
    // 19 22
    // 43 50

    return 0;
}
```

In this example, the `multiply` function performs matrix multiplication at compile time using `constexpr` and the new capabilities introduced in C++23. The result is computed during compilation, demonstrating the power of modern metaprogramming.

#### Advanced Type Manipulations

C++23 and beyond provide more tools for advanced type manipulations, allowing developers to write more expressive and flexible metaprograms. Let's explore an example of advanced type manipulation using new type traits and enhanced template syntax.

##### Example: Advanced Type Manipulations with Concepts

```cpp
#include <iostream>

#include <type_traits>

// Define a concept for copyable types
template<typename T>
concept Copyable = std::is_copy_constructible_v<T> && std::is_copy_assignable_v<T>;

// Function template using the Copyable concept
template<Copyable T>
T copy_value(const T& value) {
    return value;
}

int main() {
    int x = 42;
    std::cout << "Copy of x: " << copy_value(x) << std::endl; // Output: Copy of x: 42

    // std::unique_ptr<int> ptr = std::make_unique<int>(42);
    // std::cout << "Copy of ptr: " << copy_value(ptr) << std::endl; // Error: no matching function to call 'copy_value'

    return 0;
}
```

In this example, the `Copyable` concept ensures that the `copy_value` function template only accepts types that are copy constructible and copy assignable. This constraint prevents the function from being instantiated with types that do not support copying, such as `std::unique_ptr`.

#### Conclusion

Metaprogramming in C++23 and beyond offers a wealth of new features and enhancements that make it easier and more powerful than ever before. With extended `constexpr` capabilities, enhanced type traits, improved template syntax, and new standard library components, developers can write more expressive, efficient, and maintainable metaprograms. These advancements enable more sophisticated compile-time computations, type manipulations, and optimizations, ensuring that modern C++ remains a powerful and versatile language for high-performance programming.

By mastering the new features introduced in C++23 and beyond, you can leverage the full potential of metaprogramming to create sophisticated and high-performance C++ applications. Whether you are optimizing performance, improving code maintainability, or ensuring correctness, these tools provide the foundation for advanced C++ programming in the modern era.
