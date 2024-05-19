
\newpage
## Chapter 3: Expression Templates

Expression templates are an advanced C++ programming technique used to optimize complex mathematical operations and expressions. By transforming operations into intermediate representation at compile time, expression templates eliminate the need for temporary objects and enable more efficient execution. This technique is particularly useful in domains like numerical computing, scientific simulations, and linear algebra, where performance is critical.

In this chapter, we will explore the concept of expression templates, starting with their basic principles and motivations. We will delve into the mechanics of building expression templates, demonstrating how to capture and manipulate expressions to improve performance. Key topics include operator overloading, template metaprogramming, and leveraging modern C++ features such as variadic templates and constexpr functions.

By the end of this chapter, you will have a thorough understanding of how to implement and use expression templates to optimize mathematical operations in C++. You will learn how to build efficient, type-safe, and maintainable code that leverages the full power of C++ for high-performance applications. Whether you are working on a custom mathematical library or optimizing existing code, expression templates provide a powerful toolset to enhance your C++ programming skills.

### 3.1. Concept and Applications

#### Understanding Expression Templates

Expression templates are an advanced C++ metaprogramming technique designed to optimize the performance of complex expressions, particularly in the context of numerical computations and linear algebra. The primary goal of expression templates is to eliminate the creation of temporary objects that can slow down execution and increase memory usage. By transforming operations into a template-based intermediate representation at compile time, expression templates enable the generation of highly efficient code.

#### Motivation and Basic Principles

When performing mathematical operations in C++, the naive approach often involves the creation of intermediate temporary objects. Consider the following example of vector addition:

```cpp
#include <vector>
#include <iostream>

std::vector<int> add(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

int main() {
    std::vector<int> vec1 = {1, 2, 3};
    std::vector<int> vec2 = {4, 5, 6};
    std::vector<int> vec3 = add(vec1, vec2);
    
    for (int v : vec3) {
        std::cout << v << " ";
    }
    std::cout << std::endl; // Output: 5 7 9

    return 0;
}
```

In this example, the `add` function creates a temporary `result` vector to store the sum of `vec1` and `vec2`. If we chain multiple operations together, the creation of these temporary objects can lead to significant overhead.

##### Example of Temporary Object Overhead

```cpp
std::vector<int> add(const std::vector<int>& a, const std::vector<int>& b);
std::vector<int> subtract(const std::vector<int>& a, const std::vector<int>& b);

int main() {
    std::vector<int> vec1 = {1, 2, 3};
    std::vector<int> vec2 = {4, 5, 6};
    std::vector<int> vec3 = {7, 8, 9};

    std::vector<int> result = add(add(vec1, vec2), subtract(vec2, vec3));
    
    for (int v : result) {
        std::cout << v << " ";
    }
    std::cout << std::endl; // Output: -5 -5 -5

    return 0;
}
```

In this example, two temporary vectors are created for the intermediate results of `add(vec1, vec2)` and `subtract(vec2, vec3)`, leading to unnecessary copying and memory allocation. Expression templates address this problem by representing expressions as types and evaluating them in a single pass, without creating intermediate temporaries.

#### Building Expression Templates

To build expression templates, we need to:

1. Define template classes to represent expressions.
2. Implement operator overloading to build expression templates.
3. Write functions to evaluate these expressions efficiently.

Let's start with a simple example of implementing expression templates for vector addition.

##### Step 1: Define Template Classes to Represent Expressions

We define a template class to represent vectors and another class to represent the addition of two vectors.

```cpp
#include <vector>
#include <iostream>

// Forward declaration of Vector class
template<typename T>
class Vector;

// Template class to represent vector addition
template<typename L, typename R>
class VectorAdd {
public:
    VectorAdd(const L& lhs, const R& rhs) : lhs(lhs), rhs(rhs) {}

    auto operator[](size_t i) const {
        return lhs[i] + rhs[i];
    }

    size_t size() const {
        return lhs.size();
    }

private:
    const L& lhs;
    const R& rhs;
};

// Template class to represent vectors
template<typename T>
class Vector {
public:
    Vector(std::initializer_list<T> init) : data(init) {}

    size_t size() const {
        return data.size();
    }

    T operator[](size_t i) const {
        return data[i];
    }

    // Overload + operator to create a VectorAdd expression template
    template<typename R>
    auto operator+(const Vector<R>& rhs) const {
        return VectorAdd<Vector<T>, Vector<R>>(*this, rhs);
    }

private:
    std::vector<T> data;
};
```

In this code, `VectorAdd` is a template class representing the addition of two vectors. The `Vector` class is a template class representing a vector, with an overloaded `+` operator to create a `VectorAdd` expression template.

##### Step 2: Implement Operator Overloading

Operator overloading in the `Vector` class allows us to build expression templates by chaining operations.

```cpp
template<typename T>
template<typename R>
auto Vector<T>::operator+(const Vector<R>& rhs) const {
    return VectorAdd<Vector<T>, Vector<R>>(*this, rhs);
}
```

This overloaded `+` operator returns a `VectorAdd` object, representing the addition of two vectors without performing the actual addition.

##### Step 3: Evaluate Expressions Efficiently

To evaluate the expression template, we need to traverse the expression tree and compute the result in a single pass.

```cpp
int main() {
    Vector<int> vec1 = {1, 2, 3};
    Vector<int> vec2 = {4, 5, 6};

    auto expr = vec1 + vec2;

    // Evaluate and print the result
    for (size_t i = 0; i < expr.size(); ++i) {
        std::cout << expr[i] << " ";
    }
    std::cout << std::endl; // Output: 5 7 9

    return 0;
}
```

In this example, `expr` is a `VectorAdd` object representing the expression `vec1 + vec2`. When we iterate over the elements of `expr`, the `operator[]` function of `VectorAdd` computes the sum of the corresponding elements of `vec1` and `vec2`.

#### Extending Expression Templates

Expression templates can be extended to support more complex operations, such as scalar multiplication, vector subtraction, and even more advanced linear algebra operations.

##### Example: Adding Scalar Multiplication

```cpp
// Template class to represent scalar multiplication
template<typename T, typename Scalar>
class ScalarMultiply {
public:
    ScalarMultiply(const T& vec, Scalar scalar) : vec(vec), scalar(scalar) {}

    auto operator[](size_t i) const {
        return vec[i] * scalar;
    }

    size_t size() const {
        return vec.size();
    }

private:
    const T& vec;
    Scalar scalar;
};

// Overload * operator in Vector class to create a ScalarMultiply expression template
template<typename T>
template<typename Scalar>
auto Vector<T>::operator*(Scalar scalar) const {
    return ScalarMultiply<Vector<T>, Scalar>(*this, scalar);
}

int main() {
    Vector<int> vec1 = {1, 2, 3};
    Vector<int> vec2 = {4, 5, 6};

    auto expr = (vec1 + vec2) * 2;

    // Evaluate and print the result
    for (size_t i = 0; i < expr.size(); ++i) {
        std::cout << expr[i] << " ";
    }
    std::cout << std::endl; // Output: 10 14 18

    return 0;
}
```

In this example, `ScalarMultiply` is a template class representing the multiplication of a vector by a scalar. The `operator*` is overloaded in the `Vector` class to create a `ScalarMultiply` expression template.

#### Applications of Expression Templates

Expression templates are widely used in high-performance computing, numerical libraries, and scientific computing applications. Some notable applications include:

1. **Linear Algebra Libraries**: Expression templates are used in libraries such as Eigen and Blaze to optimize matrix and vector operations.
2. **Symbolic Computation**: Expression templates facilitate symbolic manipulation of mathematical expressions, useful in computer algebra systems.
3. **Differential Equations**: Solving differential equations often involves complex mathematical expressions that can benefit from the optimization provided by expression templates.
4. **Graphics and Game Development**: Expression templates can optimize geometric computations, such as transformations and intersections, improving performance in graphics applications.

#### Conclusion

Expression templates are a powerful technique for optimizing complex mathematical expressions in C++. By representing operations as template-based intermediate representations, expression templates eliminate the need for temporary objects and enable more efficient execution. This subchapter has introduced the concept of expression templates, explored their basic principles, and demonstrated their implementation with detailed examples. By leveraging expression templates, you can write high-performance, type-safe, and maintainable code, making them an essential tool in the advanced C++ programmer's toolkit.

### 3.2. Building a Simple Expression Template Library

#### Introduction

Expression templates offer a powerful technique to optimize mathematical expressions in C++. By eliminating the creation of temporary objects and deferring the evaluation of expressions until they are needed, expression templates can significantly enhance performance. In this subchapter, we will build a simple expression template library step by step. This library will support basic operations like vector addition and scalar multiplication, demonstrating the principles and benefits of expression templates.

#### Step 1: Defining the Basic Vector Class

We start by defining a basic vector class that will serve as the foundation for our expression templates. This class will support initialization from an initializer list and provide access to its elements.

```cpp
#include <vector>
#include <iostream>

template<typename T>
class Vector {
public:
    Vector(std::initializer_list<T> init) : data(init) {}

    size_t size() const {
        return data.size();
    }

    T operator[](size_t i) const {
        return data[i];
    }

private:
    std::vector<T> data;
};

int main() {
    Vector<int> vec = {1, 2, 3};
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl; // Output: 1 2 3

    return 0;
}
```

In this code, the `Vector` class stores its elements in a `std::vector` and provides access through the `operator[]`.

#### Step 2: Creating Expression Templates for Addition

Next, we create a template class to represent the addition of two vectors. This class will not perform the addition immediately but will store references to the operands and compute the result on demand.

```cpp
template<typename L, typename R>
class VectorAdd {
public:
    VectorAdd(const L& lhs, const R& rhs) : lhs(lhs), rhs(rhs) {}

    auto operator[](size_t i) const {
        return lhs[i] + rhs[i];
    }

    size_t size() const {
        return lhs.size();
    }

private:
    const L& lhs;
    const R& rhs;
};
```

The `VectorAdd` class stores references to the left-hand side (`lhs`) and right-hand side (`rhs`) operands and defines the `operator[]` to compute the sum of the corresponding elements.

#### Step 3: Overloading the Addition Operator

To use `VectorAdd` in expressions, we overload the addition operator in the `Vector` class. This operator will create a `VectorAdd` object instead of performing the addition immediately.

```cpp
template<typename T>
class Vector {
public:
    Vector(std::initializer_list<T> init) : data(init) {}

    size_t size() const {
        return data.size();
    }

    T operator[](size_t i) const {
        return data[i];
    }

    template<typename R>
    auto operator+(const Vector<R>& rhs) const {
        return VectorAdd<Vector, Vector<R>>(*this, rhs);
    }

private:
    std::vector<T> data;
};

int main() {
    Vector<int> vec1 = {1, 2, 3};
    Vector<int> vec2 = {4, 5, 6};

    auto result = vec1 + vec2;

    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl; // Output: 5 7 9

    return 0;
}
```

In this code, the `operator+` creates a `VectorAdd` object that represents the addition of `vec1` and `vec2`. The result is computed when we access the elements of `result`.

#### Step 4: Adding Scalar Multiplication

To support scalar multiplication, we create a template class similar to `VectorAdd` but for multiplying a vector by a scalar.

```cpp
template<typename T, typename Scalar>
class ScalarMultiply {
public:
    ScalarMultiply(const T& vec, Scalar scalar) : vec(vec), scalar(scalar) {}

    auto operator[](size_t i) const {
        return vec[i] * scalar;
    }

    size_t size() const {
        return vec.size();
    }

private:
    const T& vec;
    Scalar scalar;
};
```

The `ScalarMultiply` class stores a reference to the vector and the scalar value and defines the `operator[]` to compute the product of the corresponding element and the scalar.

#### Step 5: Overloading the Multiplication Operator

We overload the multiplication operator in the `Vector` class to create a `ScalarMultiply` object.

```cpp
template<typename T>
class Vector {
public:
    Vector(std::initializer_list<T> init) : data(init) {}

    size_t size() const {
        return data.size();
    }

    T operator[](size_t i) const {
        return data[i];
    }

    template<typename R>
    auto operator+(const Vector<R>& rhs) const {
        return VectorAdd<Vector, Vector<R>>(*this, rhs);
    }

    template<typename Scalar>
    auto operator*(Scalar scalar) const {
        return ScalarMultiply<Vector, Scalar>(*this, scalar);
    }

private:
    std::vector<T> data;
};

int main() {
    Vector<int> vec1 = {1, 2, 3};
    Vector<int> vec2 = {4, 5, 6};

    auto result = (vec1 + vec2) * 2;

    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl; // Output: 10 14 18

    return 0;
}
```

In this code, the `operator*` creates a `ScalarMultiply` object representing the multiplication of a vector by a scalar. The expression `(vec1 + vec2) * 2` is evaluated lazily, with the actual computation performed when accessing the elements of `result`.

#### Step 6: Adding More Operations

To make our expression template library more versatile, we can add support for more operations, such as subtraction and division.

##### Subtraction

```cpp
template<typename L, typename R>
class VectorSubtract {
public:
    VectorSubtract(const L& lhs, const R& rhs) : lhs(lhs), rhs(rhs) {}

    auto operator[](size_t i) const {
        return lhs[i] - rhs[i];
    }

    size_t size() const {
        return lhs.size();
    }

private:
    const L& lhs;
    const R& rhs;
};

template<typename T>
class Vector {
public:
    Vector(std::initializer_list<T> init) : data(init) {}

    size_t size() const {
        return data.size();
    }

    T operator[](size_t i) const {
        return data[i];
    }

    template<typename R>
    auto operator+(const Vector<R>& rhs) const {
        return VectorAdd<Vector, Vector<R>>(*this, rhs);
    }

    template<typename R>
    auto operator-(const Vector<R>& rhs) const {
        return VectorSubtract<Vector, Vector<R>>(*this, rhs);
    }

    template<typename Scalar>
    auto operator*(Scalar scalar) const {
        return ScalarMultiply<Vector, Scalar>(*this, scalar);
    }

private:
    std::vector<T> data;
};

int main() {
    Vector<int> vec1 = {1, 2, 3};
    Vector<int> vec2 = {4, 5, 6};

    auto result = (vec1 + vec2) - vec1 * 2;

    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl; // Output: 3 5 7

    return 0;
}
```

##### Division

```cpp
template<typename T, typename Scalar>
class ScalarDivide {
public:
    ScalarDivide(const T& vec, Scalar scalar) : vec(vec), scalar(scalar) {}

    auto operator[](size_t i) const {
        return vec[i] / scalar;
    }

    size_t size() const {
        return vec.size();
    }

private:
    const T& vec;
    Scalar scalar;
};

template<typename T>
class Vector {
public:
    Vector(std::initializer_list<T> init) : data(init) {}

    size_t size() const {
        return data.size();
    }

    T operator[](size_t i) const {
        return data[i];
    }

    template<typename R>
    auto operator+(const Vector<R>& rhs) const {
        return VectorAdd<Vector, Vector<R>>(*this, rhs);
    }

    template<typename R>
    auto operator-(const Vector<R>& rhs) const {
        return VectorSubtract<Vector, Vector<R>>(*this, rhs);
    }

    template<typename Scalar>
    auto operator*(Scalar scalar) const {
        return ScalarMultiply<Vector, Scalar>(*this, scalar);
    }

    template<typename Scalar>
    auto operator/(Scalar scalar) const {
        return ScalarDivide<Vector, Scalar>(*this, scalar);
    }

private:
    std::vector<T> data;
};

int main() {
    Vector<int> vec1 = {1, 2, 3};
    Vector<int> vec2 = {4, 5, 6};

    auto result = (vec1 + vec2) / 2;

    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl; // Output: 2 3 4

    return 0;
}
```

In this extended example, we have added support for subtraction and division, further enhancing our expression template library.

#### Step 7: Optimizing Expression Evaluation

To optimize expression evaluation further, we can implement lazy evaluation and avoid unnecessary computations by combining multiple operations into a single pass.

##### Combining Multiple Operations

```cpp
template<typename L, typename R>
class VectorMultiplyAdd {
public:
    VectorMultiplyAdd(const L& lhs, const R& rhs, int scalar) : lhs(lhs), rhs(rhs), scalar(scalar) {}

    auto operator[](size_t i) const {
        return (lhs[i] + rhs[i]) * scalar;
    }

    size_t size() const {
        return lhs.size();
    }

private:
    const L& lhs;
    const R& rhs;
    int scalar;
};

template<typename T>
class Vector {
public:
    Vector(std::initializer_list<T> init) : data(init) {}

    size_t size() const {
        return data.size();
    }

    T operator[](size_t i) const {
        return data[i];
    }

    template<typename R>
    auto operator+(const Vector<R>& rhs) const {
        return VectorAdd<Vector, Vector<R>>(*this, rhs);
    }

    template<typename R>
    auto operator-(const Vector<R>& rhs) const {
        return VectorSubtract<Vector, Vector<R>>(*this, rhs);
    }

    template<typename Scalar>
    auto operator*(Scalar scalar) const {
        return ScalarMultiply<Vector, Scalar>(*this, scalar);
    }

    template<typename Scalar>
    auto operator/(Scalar scalar) const {
        return ScalarDivide<Vector, Scalar>(*this, scalar);
    }

    auto operator+(const VectorAdd<Vector, Vector<T>>& expr) const {
        return VectorMultiplyAdd<Vector, Vector<T>>(*this, expr, 1);
    }

    auto operator*(const VectorAdd<Vector, Vector<T>>& expr) const {
        return VectorMultiplyAdd<Vector, Vector<T>>(*this, expr, 1);
    }

private:
    std::vector<T> data;
};

int main() {
    Vector<int> vec1 = {1, 2, 3};
    Vector<int> vec2 = {4, 5, 6};

    auto result = vec1 + (vec2 * 2);

    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl; // Output: 9 12 15

    return 0;
}
```

In this example, we have combined the addition and multiplication operations into a single class `VectorMultiplyAdd`, further optimizing the evaluation of complex expressions.

#### Conclusion

Building an expression template library in C++ involves defining template classes to represent expressions, overloading operators to create these templates, and efficiently evaluating the expressions in a single pass. This subchapter has provided a detailed guide to constructing a simple expression template library, supporting basic operations like vector addition, scalar multiplication, subtraction, and division. By leveraging expression templates, you can eliminate the creation of temporary objects, optimize performance, and write high-performance, type-safe, and maintainable code. Understanding these principles will empower you to develop more sophisticated numerical and scientific computing applications in C++.

### 3.2. Performance Benefits and Use Cases

#### Introduction to Performance Benefits

Expression templates provide substantial performance benefits, especially in computationally intensive applications such as numerical simulations, graphics, and scientific computing. By transforming expressions into intermediate representations at compile time, expression templates eliminate the creation of temporary objects and reduce the number of redundant computations. This results in significant improvements in both execution speed and memory efficiency.

In this subchapter, we will explore the performance benefits of expression templates in detail. We will compare traditional approaches with expression templates, analyze the resulting performance improvements, and discuss various use cases where expression templates can make a substantial difference.

#### Eliminating Temporary Objects

One of the primary performance benefits of expression templates is the elimination of temporary objects. In traditional C++ code, each intermediate result in an expression creates a temporary object, leading to unnecessary memory allocations and copies. Expression templates defer the evaluation of expressions until the final result is needed, avoiding these temporary objects.

##### Example: Traditional Vector Addition

Consider the traditional approach to vector addition:

```cpp
#include <vector>
#include <iostream>

std::vector<int> add(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

int main() {
    std::vector<int> vec1 = {1, 2, 3};
    std::vector<int> vec2 = {4, 5, 6};
    std::vector<int> vec3 = add(vec1, vec2);

    for (int v : vec3) {
        std::cout << v << " ";
    }
    std::cout << std::endl; // Output: 5 7 9

    return 0;
}
```

In this example, the `add` function creates a temporary vector `result` to store the sum of `vec1` and `vec2`. This approach involves memory allocation for the `result` vector and copying the elements.

##### Example: Vector Addition with Expression Templates

Using expression templates, we can eliminate the temporary vector:

```cpp
template<typename L, typename R>
class VectorAdd {
public:
    VectorAdd(const L& lhs, const R& rhs) : lhs(lhs), rhs(rhs) {}

    auto operator[](size_t i) const {
        return lhs[i] + rhs[i];
    }

    size_t size() const {
        return lhs.size();
    }

private:
    const L& lhs;
    const R& rhs;
};

template<typename T>
class Vector {
public:
    Vector(std::initializer_list<T> init) : data(init) {}

    size_t size() const {
        return data.size();
    }

    T operator[](size_t i) const {
        return data[i];
    }

    template<typename R>
    auto operator+(const Vector<R>& rhs) const {
        return VectorAdd<Vector, Vector<R>>(*this, rhs);
    }

private:
    std::vector<T> data;
};

int main() {
    Vector<int> vec1 = {1, 2, 3};
    Vector<int> vec2 = {4, 5, 6};

    auto result = vec1 + vec2;

    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl; // Output: 5 7 9

    return 0;
}
```

In this example, the `VectorAdd` class represents the addition of two vectors without creating a temporary vector. The actual addition is performed when the elements are accessed, reducing memory allocation and copying.

#### Reducing Redundant Computations

Expression templates also reduce redundant computations by combining multiple operations into a single pass. This is particularly beneficial in complex expressions where intermediate results can be reused without recomputing them.

##### Example: Traditional Approach with Redundant Computations

Consider the following example using traditional C++ code:

```cpp
#include <vector>
#include <iostream>

std::vector<int> add(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

std::vector<int> multiply(const std::vector<int>& a, int scalar) {
    std::vector<int> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * scalar;
    }
    return result;
}

int main() {
    std::vector<int> vec1 = {1, 2, 3};
    std::vector<int> vec2 = {4, 5, 6};

    std::vector<int> temp = add(vec1, vec2);
    std::vector<int> result = multiply(temp, 2);

    for (int v : result) {
        std::cout << v << " ";
    }
    std::cout << std::endl; // Output: 10 14 18

    return 0;
}
```

In this example, the intermediate result `temp` is computed and stored, leading to redundant computations and memory allocations.

##### Example: Reducing Redundant Computations with Expression Templates

Using expression templates, we can reduce redundant computations:

```cpp
template<typename L, typename R>
class VectorAdd {
public:
    VectorAdd(const L& lhs, const R& rhs) : lhs(lhs), rhs(rhs) {}

    auto operator[](size_t i) const {
        return lhs[i] + rhs[i];
    }

    size_t size() const {
        return lhs.size();
    }

private:
    const L& lhs;
    const R& rhs;
};

template<typename T, typename Scalar>
class ScalarMultiply {
public:
    ScalarMultiply(const T& vec, Scalar scalar) : vec(vec), scalar(scalar) {}

    auto operator[](size_t i) const {
        return vec[i] * scalar;
    }

    size_t size() const {
        return vec.size();
    }

private:
    const T& vec;
    Scalar scalar;
};

template<typename T>
class Vector {
public:
    Vector(std::initializer_list<T> init) : data(init) {}

    size_t size() const {
        return data.size();
    }

    T operator[](size_t i) const {
        return data[i];
    }

    template<typename R>
    auto operator+(const Vector<R>& rhs) const {
        return VectorAdd<Vector, Vector<R>>(*this, rhs);
    }

    template<typename Scalar>
    auto operator*(Scalar scalar) const {
        return ScalarMultiply<Vector, Scalar>(*this, scalar);
    }

private:
    std::vector<T> data;
};

int main() {
    Vector<int> vec1 = {1, 2, 3};
    Vector<int> vec2 = {4, 5, 6};

    auto result = (vec1 + vec2) * 2;

    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl; // Output: 10 14 18

    return 0;
}
```

In this example, the expression `(vec1 + vec2) * 2` is represented as a combination of `VectorAdd` and `ScalarMultiply` objects. The computations are performed in a single pass, reducing redundant operations.

#### Use Cases of Expression Templates

Expression templates are particularly useful in scenarios where performance is critical and where complex mathematical expressions are common. Here are some notable use cases:

##### Linear Algebra Libraries

Expression templates are widely used in linear algebra libraries to optimize matrix and vector operations. Libraries such as Eigen and Blaze leverage expression templates to achieve high performance.

##### Example: Optimizing Matrix Multiplication

```cpp
#include <iostream>
#include <vector>

// Template class to represent matrix multiplication
template<typename L, typename R>
class MatrixMultiply {
public:
    MatrixMultiply(const L& lhs, const R& rhs) : lhs(lhs), rhs(rhs) {}

    auto operator()(size_t i, size_t j) const {
        auto sum = lhs(i, 0) * rhs(0, j);
        for (size_t k = 1; k < lhs.cols(); ++k) {
            sum += lhs(i, k) * rhs(k, j);
        }
        return sum;
    }

    size_t rows() const {
        return lhs.rows();
    }

    size_t cols() const {
        return rhs.cols();
    }

private:
    const L& lhs;
    const R& rhs;
};

// Template class to represent matrices
template<typename T>
class Matrix {
public:
    Matrix(size_t rows, size_t cols) : data(rows, std::vector<T>(cols)), rows(rows), cols(cols) {}

    size_t rows() const {
        return rows;
    }

    size_t cols() const {
        return cols;
    }

    T& operator()(size_t i, size_t j) {
        return data[i][j];
    }

    T operator()(size_t i, size_t j) const {
        return data[i][j];
    }

    template<typename R>
    auto operator*(const Matrix<R>& rhs) const {
        return MatrixMultiply<Matrix, Matrix<R>>(*this, rhs);
    }

private:
    std::vector<std::vector<T>> data;
    size_t rows, cols;
};

int main() {
    Matrix<int> mat1(2, 2);
    mat1(0, 0) = 1; mat1(0, 1) = 2;
    mat1(1, 0) = 3; mat1(1, 1) = 4;

    Matrix<int> mat2(2, 2);
    mat2(0, 0) = 5; mat2(0, 1) = 6;
    mat2(1, 0) = 7; mat2(1, 1) = 8;

    auto result = mat1 * mat2;

    for (size_t i = 0; i < result.rows(); ++i) {
        for (size_t j = 0; j < result.cols(); ++j) {
            std::cout << result(i, j) << " ";
        }
        std::cout << std::endl;
    }
    // Output:
    // 19 22
    // 43 50

    return 0;
}
```

In this example, the `MatrixMultiply` class represents the multiplication of two matrices. The computations are performed lazily, avoiding the creation of temporary matrices and optimizing the evaluation.

##### Symbolic Computation

Expression templates are used in symbolic computation to manipulate and simplify mathematical expressions. This is useful in computer algebra systems and automatic differentiation.

##### Example: Simplifying Expressions

```cpp
#include <iostream>
#include <cmath>

template<typename L, typename R>
class ExpressionAdd {
public:
    ExpressionAdd(const L& lhs, const R& rhs) : lhs(lhs), rhs(rhs) {}

    auto operator[](size_t i) const {
        return lhs[i] + rhs[i];
    }

private:
    const L& lhs;
    const R& rhs;
};

template<typename T>
class Expression {
public:
    Expression(T value) : value(value) {}

    T operator[](size_t i) const {
        return value;
    }

    template<typename R>
    auto operator+(const Expression<R>& rhs) const {
        return ExpressionAdd<Expression, Expression<R>>(*this, rhs);
    }

private:
    T value;
};

int main() {
    Expression<int> expr1(3);
    Expression<int> expr2(4);
    auto result = expr1 + expr2;

    std::cout << result[0] << std::endl; // Output: 7

    return 0;
}
```

In this example, the `Expression` and `ExpressionAdd` classes represent symbolic expressions. The expression `expr1 + expr2` is simplified at compile time, avoiding unnecessary computations.

##### Differential Equations

Solving differential equations often involves complex mathematical expressions that can benefit from the optimization provided by expression templates.

##### Example: Solving Differential Equations

```cpp
#include <iostream>
#include <vector>
#include <cmath>

template<typename Func>
void solve_ode(Func f, double y0, double t0, double t1, double h) {
    std::vector<double> t, y;
    t.push_back(t0);
    y.push_back(y0);

    while (t.back() < t1) {
        double tn = t.back();
        double yn = y.back();
        double k1 = h * f(tn, yn);
        double k2 = h * f(tn + h / 2, yn + k1 / 2);
        double k3 = h * f(tn + h / 2, yn + k2 / 2);
        double k4 = h * f(tn + h, yn + k3);
        double yn1 = yn + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
        t.push_back(tn + h);
        y.push_back(yn1);
    }

    for (size_t i = 0; i < t.size(); ++i) {
        std::cout << "t: " << t[i] << ", y: " << y[i] << std::endl;
    }
}

int main() {
    auto f = [](double t, double y) {
        return -2 * t * y;
    };

    solve_ode(f, 1.0, 0.0, 2.0, 0.1);

    return 0;
}
```

In this example, the `solve_ode` function solves an ordinary differential equation using the Runge-Kutta method. Expression templates can be used to optimize the computations involved in solving such equations.

#### Conclusion

Expression templates provide significant performance benefits by eliminating temporary objects, reducing redundant computations, and enabling more efficient evaluation of complex expressions. They are widely used in various domains, including linear algebra libraries, symbolic computation, and solving differential equations. By leveraging expression templates, developers can write high-performance, type-safe, and maintainable code, making them an essential tool in advanced C++ programming. Understanding and applying expression templates will empower you to optimize your computationally intensive applications and achieve better performance.
