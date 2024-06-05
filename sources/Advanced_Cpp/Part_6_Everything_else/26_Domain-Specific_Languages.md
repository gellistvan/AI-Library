
\newpage
## Chapter 26: Domain-Specific Languages

In modern software development, the ability to create domain-specific languages (DSLs) offers a powerful way to address specific problem domains with tailored, expressive syntax and semantics. This chapter explores the intricacies of designing and implementing DSLs in C++, providing a comprehensive guide to various techniques and tools. We begin with creating embedded DSLs, leveraging the flexibility and expressiveness of C++ to integrate domain-specific constructs seamlessly into your code. Next, we delve into parser generators, essential tools for crafting standalone DSLs with robust parsing capabilities. The chapter then introduces the concept of expression templates, a sophisticated metaprogramming technique to enable efficient and readable domain-specific expressions. Finally, we present practical examples of DSLs in C++, illustrating their application in real-world scenarios. Through these sections, you will gain the knowledge and skills to harness the full potential of DSLs in your C++ projects.

### 26.1. Creating Embedded DSLs

Creating embedded domain-specific languages (DSLs) in C++ allows you to design and implement a language that is specific to a particular domain, directly within the host language. This technique leverages the syntax and semantics of C++ to create a fluent and expressive interface for the specific problem domain. In this subchapter, we will explore the principles and techniques for creating embedded DSLs, focusing on design considerations, syntactic enhancements, and practical implementation strategies. Throughout, we will provide detailed code examples to illustrate these concepts.

#### Introduction to Embedded DSLs

Embedded DSLs are languages designed to be used within a general-purpose host language. In the context of C++, an embedded DSL integrates seamlessly into the codebase, leveraging C++’s powerful features such as operator overloading, templates, and metaprogramming. This integration allows for concise and readable domain-specific code, improving both development speed and code maintainability.

#### Design Considerations

Before diving into the implementation of an embedded DSL, it is crucial to consider the following design aspects:

1. **Domain Analysis**: Identify the specific needs and patterns of the domain. Understand the operations, data structures, and workflows that are common in this domain.
2. **Fluent Interface**: Aim for an API that reads naturally and fluently, as if it were a part of the language itself. This often involves method chaining and operator overloading.
3. **Performance**: Ensure that the abstractions introduced by the DSL do not significantly degrade performance.
4. **Error Handling**: Provide meaningful error messages and handle domain-specific errors gracefully.

#### Example: A Simple Math DSL

Let’s start with a simple example of an embedded DSL for mathematical expressions. The goal is to create a DSL that allows users to write mathematical expressions in a natural and readable way.

```cpp
#include <iostream>

#include <sstream>
#include <string>

// Forward declaration of Expression class
class Expression;

// Base class for all expressions
class Expression {
public:
    virtual ~Expression() = default;
    virtual std::string toString() const = 0;
};

// Class for literal values
class Literal : public Expression {
    double value;
public:
    explicit Literal(double value) : value(value) {}
    std::string toString() const override {
        return std::to_string(value);
    }
};

// Class for binary operations
class BinaryOperation : public Expression {
    const Expression& left;
    const Expression& right;
    char op;
public:
    BinaryOperation(const Expression& left, char op, const Expression& right)
        : left(left), op(op), right(right) {}
    std::string toString() const override {
        std::ostringstream oss;
        oss << "(" << left.toString() << " " << op << " " << right.toString() << ")";
        return oss.str();
    }
};

// Operator overloading for creating binary operations
BinaryOperation operator+(const Expression& left, const Expression& right) {
    return BinaryOperation(left, '+', right);
}

BinaryOperation operator-(const Expression& left, const Expression& right) {
    return BinaryOperation(left, '-', right);
}

BinaryOperation operator*(const Expression& left, const Expression& right) {
    return BinaryOperation(left, '*', right);
}

BinaryOperation operator/(const Expression& left, const Expression& right) {
    return BinaryOperation(left, '/', right);
}

// Utility function for creating literals
Literal lit(double value) {
    return Literal(value);
}

int main() {
    Expression* expr = new BinaryOperation(lit(5), '+', BinaryOperation(lit(3), '*', lit(2)));
    std::cout << expr->toString() << std::endl; // Outputs: (5.000000 + (3.000000 * 2.000000))
    delete expr;
    return 0;
}
```

In this example, we have defined a simple DSL for mathematical expressions. The `Literal` class represents constant values, while the `BinaryOperation` class represents binary operations such as addition and multiplication. Operator overloading is used to allow expressions to be written in a natural, mathematical way.

#### Enhancing the DSL with Method Chaining

To improve the fluency of our DSL, we can enhance it with method chaining. This approach allows us to build expressions by chaining method calls, which can make the code more readable and expressive.

```cpp
#include <iostream>

#include <sstream>
#include <memory>

#include <string>

class Expression {
public:
    virtual ~Expression() = default;
    virtual std::string toString() const = 0;
    virtual std::unique_ptr<Expression> clone() const = 0;
};

class Literal : public Expression {
    double value;
public:
    explicit Literal(double value) : value(value) {}
    std::string toString() const override {
        return std::to_string(value);
    }
    std::unique_ptr<Expression> clone() const override {
        return std::make_unique<Literal>(*this);
    }
};

class BinaryOperation : public Expression {
    std::unique_ptr<Expression> left;
    std::unique_ptr<Expression> right;
    char op;
public:
    BinaryOperation(std::unique_ptr<Expression> left, char op, std::unique_ptr<Expression> right)
        : left(std::move(left)), op(op), right(std::move(right)) {}
    std::string toString() const override {
        std::ostringstream oss;
        oss << "(" << left->toString() << " " << op << " " << right->toString() << ")";
        return oss.str();
    }
    std::unique_ptr<Expression> clone() const override {
        return std::make_unique<BinaryOperation>(left->clone(), op, right->clone());
    }
};

class ExpressionBuilder {
    std::unique_ptr<Expression> expr;
public:
    ExpressionBuilder(std::unique_ptr<Expression> expr) : expr(std::move(expr)) {}
    
    ExpressionBuilder add(std::unique_ptr<Expression> right) {
        return ExpressionBuilder(std::make_unique<BinaryOperation>(std::move(expr), '+', std::move(right)));
    }
    
    ExpressionBuilder subtract(std::unique_ptr<Expression> right) {
        return ExpressionBuilder(std::make_unique<BinaryOperation>(std::move(expr), '-', std::move(right)));
    }
    
    ExpressionBuilder multiply(std::unique_ptr<Expression> right) {
        return ExpressionBuilder(std::make_unique<BinaryOperation>(std::move(expr), '*', std::move(right)));
    }
    
    ExpressionBuilder divide(std::unique_ptr<Expression> right) {
        return ExpressionBuilder(std::make_unique<BinaryOperation>(std::move(expr), '/', std::move(right)));
    }
    
    std::string toString() const {
        return expr->toString();
    }
};

int main() {
    ExpressionBuilder exprBuilder(std::make_unique<Literal>(5));
    auto expr = exprBuilder.add(std::make_unique<BinaryOperation>(std::make_unique<Literal>(3), '*', std::make_unique<Literal>(2)));
    std::cout << expr.toString() << std::endl; // Outputs: (5.000000 + (3.000000 * 2.000000))
    return 0;
}
```

In this enhanced example, the `ExpressionBuilder` class allows us to chain method calls to build complex expressions in a more readable manner. Each method in `ExpressionBuilder` returns a new `ExpressionBuilder` object, enabling the fluent interface.

#### Advanced Features: Template Metaprogramming

For more advanced DSLs, template metaprogramming can be used to create even more powerful and flexible constructs. Here, we explore a simple example of using templates to build a type-safe mathematical DSL.

```cpp
#include <iostream>

#include <sstream>
#include <memory>

#include <string>

template<typename T>
class Expression {
public:
    virtual ~Expression() = default;
    virtual std::string toString() const = 0;
    virtual T evaluate() const = 0;
};

template<typename T>
class Literal : public Expression<T> {
    T value;
public:
    explicit Literal(T value) : value(value) {}
    std::string toString() const override {
        return std::to_string(value);
    }
    T evaluate() const override {
        return value;
    }
};

template<typename T>
class BinaryOperation : public Expression<T> {
    std::unique_ptr<Expression<T>> left;
    std::unique_ptr<Expression<T>> right;
    char op;
public:
    BinaryOperation(std::unique_ptr<Expression<T>> left, char op, std::unique_ptr<Expression<T>> right)
        : left(std::move(left)), op(op), right(std::move(right)) {}
    std::string toString() const override {
        std::ostringstream oss;
        oss << "(" << left->toString() << " " << op << " " << right->toString() << ")";
        return oss.str();
    }
    T evaluate() const override {
        if (op == '+') return left->evaluate() + right->evaluate();
        if (op == '-') return left->evaluate() - right->evaluate();
        if (op == '*') return left->evaluate() * right->evaluate();
        if (op == '/') return left->evaluate() / right->evaluate();
        throw std::runtime_error("Invalid operator");
    }
};

template<typename T>
class ExpressionBuilder {
    std::unique_ptr<Expression<T>> expr;
public:
    ExpressionBuilder(std::unique_ptr<Expression<T>> expr) : expr(std::move(expr)) {}
    
    ExpressionBuilder add(std::unique_ptr<Expression<T>> right) {
        return ExpressionBuilder(std::make_unique<BinaryOperation<T>>(std::move(expr), '+', std::move(right)));
    }
    
    ExpressionBuilder subtract(std::unique_ptr<Expression<T>> right) {
        return ExpressionBuilder(std::make_unique<BinaryOperation<T>>(std::move(expr), '-', std::move(right)));
    }
    
    ExpressionBuilder multiply(std::unique_ptr<Expression<T>> right) {
        return ExpressionBuilder(std::make_unique<BinaryOperation<T>>(std::move(expr), '*', std::move(right)));
    }
    
    ExpressionBuilder divide(std::unique_ptr<Expression<T>> right) {
        return ExpressionBuilder(std::make_unique<BinaryOperation<T>>(std::move(expr), '/', std::move(right)));
    }
    
    std::string toString() const {
        return expr->toString();
    }
    
    T evaluate() const {
        return expr->evaluate();
    }
};

int main() {
    ExpressionBuilder<double> exprBuilder(std::make_unique<Literal<double>>(5));
    auto expr = exprBuilder.add(std::make_unique<BinaryOperation<double>>(std::make_unique<Literal<double>>(3), '*', std::make_unique<Literal<double>>(2)));
    std::cout << expr.toString() << std::endl; // Outputs: (5.000000 + (3.000000 * 2.000000))
    std::cout << expr.evaluate() << std::endl; // Outputs: 11
    return 0;
}
```

In this advanced example, we use templates to create a type-safe mathematical DSL. The `Expression`, `Literal`, and `BinaryOperation` classes are all templated, allowing for different types of numeric expressions. The `ExpressionBuilder` class is also templated, ensuring type safety throughout the DSL.

#### Conclusion

Creating embedded DSLs in C++ allows you to design domain-specific solutions that are both expressive and efficient. By leveraging C++’s powerful features such as operator overloading, method chaining, and template metaprogramming, you can build DSLs that integrate seamlessly into your codebase. This subchapter has provided an overview of the principles and techniques involved in creating embedded DSLs, along with detailed examples to illustrate these concepts. With these tools and techniques, you are well-equipped to develop sophisticated DSLs tailored to your specific problem domains.

### 26.2. Parser Generators

Parser generators are powerful tools used to create parsers for domain-specific languages (DSLs) and other custom language constructs. These tools take a formal description of a language's grammar and automatically generate code for a parser that can recognize and process input in that language. In this subchapter, we will explore the fundamentals of parser generators, examine common tools and techniques, and provide detailed examples to illustrate their use in C++.

#### Introduction to Parser Generators

A parser generator automates the creation of a parser from a formal grammar. The formal grammar typically describes the syntax of the language using rules written in a format such as Backus-Naur Form (BNF) or Extended Backus-Naur Form (EBNF). The parser generator reads this grammar and produces a parser that can interpret and process input according to the specified rules.

Commonly used parser generators include:
- **Yacc/Bison**: Tools for generating parsers in C and C++.
- **ANTLR**: A powerful tool for generating parsers in multiple programming languages, including C++.
- **Boost.Spirit**: A C++ library for creating parsers directly in C++ code using template metaprogramming.

#### Using Bison for Parser Generation

Bison is a popular parser generator that is often used in conjunction with Flex, a lexical analyzer generator. Together, they provide a powerful combination for building parsers in C++.

##### Defining the Grammar

First, we need to define the grammar of our DSL. Let's consider a simple arithmetic language that supports addition, subtraction, multiplication, and division.

Create a file named `calc.y` for the grammar:

```yacc
%{
#include <cstdio>

#include <cstdlib>

void yyerror(const char *s);
int yylex();
%}

%token NUMBER
%left '+' '-'
%left '*' '/'

%%

expr:
      expr '+' expr { printf("%f\n", $1 + $3); }
    | expr '-' expr { printf("%f\n", $1 - $3); }
    | expr '*' expr { printf("%f\n", $1 * $3); }
    | expr '/' expr { printf("%f\n", $1 / $3); }
    | '(' expr ')' { $$ = $2; }
    | NUMBER { $$ = $1; }
    ;

%%

void yyerror(const char *s) {
    fprintf(stderr, "Error: %s\n", s);
}

int main() {
    printf("Enter an expression: ");
    return yyparse();
}
```

##### Lexical Analysis with Flex

Next, we need a lexical analyzer to tokenize the input. Create a file named `calc.l`:

```lex
%{
#include "calc.tab.h"
%}

%%

[0-9]+(\.[0-9]+)?  { yylval = atof(yytext); return NUMBER; }
[ \t\n]            { /* ignore whitespace */ }
"+"                { return '+'; }
"-"                { return '-'; }
"*"                { return '*'; }
"/"                { return '/'; }
"("                { return '('; }
")"                { return ')'; }

%%

int yywrap() {
    return 1;
}
```

##### Generating and Compiling the Parser

To generate the parser, use Bison and Flex as follows:

```sh
bison -d calc.y
flex calc.l
g++ calc.tab.c lex.yy.c -o calc -ll
```

This sequence of commands will generate the parser and lexical analyzer, and compile them into an executable named `calc`.

##### Running the Parser

You can now run the parser and input arithmetic expressions:

```sh
./calc
Enter an expression: 3 + 4 * 2
3 + 4 * 2
11.000000
```

#### Using ANTLR for Parser Generation

ANTLR (Another Tool for Language Recognition) is another powerful parser generator that supports multiple target languages, including C++. ANTLR provides a rich set of features for defining complex grammars and generating efficient parsers.

##### Defining the Grammar

Let's define the same arithmetic language using ANTLR. Create a file named `Calc.g4`:

```antlr
grammar Calc;

prog:   expr+ ;

expr:   expr op=('*'|'/') expr
    |   expr op=('+'|'-') expr
    |   '(' expr ')'
    |   NUMBER
    ;

NUMBER: [0-9]+ ('.' [0-9]+)? ;

WS: [ \t\n\r]+ -> skip ;
```

##### Generating the Parser

To generate the parser code, use the ANTLR tool. First, download ANTLR from [the official website](https://www.antlr.org/). Then, generate the parser code:

```sh
java -jar antlr-4.9.2-complete.jar -Dlanguage=Cpp Calc.g4
```

This will generate C++ source files for the lexer and parser.

##### Integrating the Generated Code

Create a `main.cpp` file to integrate the generated parser:

```cpp
#include <iostream>

#include "antlr4-runtime.h"
#include "CalcLexer.h"

#include "CalcParser.h"

using namespace antlr4;

int main(int argc, const char* argv[]) {
    ANTLRInputStream input(std::cin);
    CalcLexer lexer(&input);
    CommonTokenStream tokens(&lexer);

    CalcParser parser(&tokens);
    tree::ParseTree *tree = parser.prog();

    std::cout << tree->toStringTree(&parser) << std::endl;

    return 0;
}
```

Compile the program:

```sh
g++ main.cpp CalcLexer.cpp CalcParser.cpp -I/usr/local/include/antlr4-runtime -lantlr4-runtime -o calc
```

Run the program:

```sh
./calc
3 + 4 * 2
(prog (expr (expr 3) + (expr (expr 4) * (expr 2))))
```

#### Using Boost.Spirit for Parser Generation

Boost.Spirit is a C++ library for creating parsers directly in C++ code using template metaprogramming. It provides a highly expressive syntax that allows you to define grammars directly in C++.

##### Defining the Grammar

Let’s define the same arithmetic language using Boost.Spirit:

```cpp
#include <boost/spirit/include/qi.hpp>

#include <iostream>
#include <string>

namespace qi = boost::spirit::qi;

template <typename Iterator>
struct calculator : qi::grammar<Iterator, double(), qi::space_type> {
    calculator() : calculator::base_type(expression) {
        expression =
              term
              >> *(   ('+' >> term)
                  |   ('-' >> term)
                  )
              ;
        term =
              factor
              >> *(   ('*' >> factor)
                  |   ('/' >> factor)
                  )
              ;
        factor =
              qi::double_
              |   '(' >> expression >> ')'
              ;
    }

    qi::rule<Iterator, double(), qi::space_type> expression, term, factor;
};

int main() {
    std::string str;
    while (std::getline(std::cin, str)) {
        auto it = str.begin();
        calculator<std::string::iterator> calc;
        double result;

        bool r = qi::phrase_parse(it, str.end(), calc, qi::space, result);

        if (r && it == str.end()) {
            std::cout << result << std::endl;
        } else {
            std::cout << "Parsing failed\n";
        }
    }
    return 0;
}
```

##### Compiling and Running the Parser

Compile the program with Boost:

```sh
g++ -std=c++11 -o calc calc.cpp -lboost_system -lboost_filesystem -lboost_program_options -lboost_regex
```

Run the program:

```sh
./calc
3 + 4 * 2
11
```

#### Conclusion

Parser generators are invaluable tools for creating parsers for domain-specific languages and custom language constructs. By automating the parsing process, they allow you to focus on the semantics and functionality of your language. In this subchapter, we explored the use of Bison, ANTLR, and Boost.Spirit to create parsers in C++. Each tool offers unique strengths and can be chosen based on the specific requirements and constraints of your project. With these tools and techniques, you are well-equipped to develop robust and efficient parsers for your DSLs.

### 26.3. Using Expression Templates

Expression templates are an advanced metaprogramming technique in C++ that allows for the optimization of complex expressions, particularly in the context of numerical computing and domain-specific languages (DSLs). By representing expressions as types, expression templates enable the compiler to perform optimizations such as eliminating temporary objects and reducing runtime overhead. This subchapter explores the concept of expression templates, explaining their principles and illustrating their application through detailed code examples.

#### Introduction to Expression Templates

Expression templates involve representing expressions as a series of nested template types rather than evaluating them immediately. This approach allows the compiler to analyze and optimize the entire expression before generating the final code. The primary benefit of expression templates is the ability to perform operations without creating intermediate temporaries, thus improving performance and memory efficiency.

#### Basic Concept of Expression Templates

To understand expression templates, let's start with a simple example of vector arithmetic. Typically, adding two vectors involves creating temporary vectors for intermediate results. Expression templates can eliminate these temporaries.

##### Basic Vector Class

First, let's define a basic vector class without expression templates:

```cpp
#include <iostream>

#include <vector>

class Vector {
public:
    Vector(size_t size) : data(size) {}

    double& operator[](size_t index) {
        return data[index];
    }

    const double& operator[](size_t index) const {
        return data[index];
    }

    size_t size() const {
        return data.size();
    }

private:
    std::vector<double> data;
};

Vector operator+(const Vector& lhs, const Vector& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("Vectors must be of the same size");
    }

    Vector result(lhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] + rhs[i];
    }

    return result;
}

Vector operator-(const Vector& lhs, const Vector& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("Vectors must be of the same size");
    }

    Vector result(lhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] - rhs[i];
    }

    return result;
}
```

Using this class, adding vectors results in multiple temporary objects:

```cpp
int main() {
    Vector a(3), b(3), c(3);
    a[0] = 1; a[1] = 2; a[2] = 3;
    b[0] = 4; b[1] = 5; b[2] = 6;
    c[0] = 7; c[1] = 8; c[2] = 9;

    Vector result = a + b + c;  // Creates temporary vectors
    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result[i] << " ";
    }

    return 0;
}
```

In the above example, `a + b` creates a temporary vector which is then added to `c`, resulting in another temporary vector.

#### Introducing Expression Templates

Expression templates can eliminate these temporary objects. The key idea is to represent the expression as a type and evaluate it in one go.

##### Expression Template Framework

Let's define an expression template framework:

```cpp
#include <iostream>

#include <vector>

// Forward declaration of template classes
template <typename E>
class VectorExpr;

template <typename E1, typename E2>
class VectorSum;

// Base class for vector expressions
template <typename E>
class VectorExpr {
public:
    double operator[](size_t index) const {
        return static_cast<const E&>(*this)[index];
    }

    size_t size() const {
        return static_cast<const E&>(*this).size();
    }
};

// Class for actual vectors
class Vector : public VectorExpr<Vector> {
public:
    Vector(size_t size) : data(size) {}

    double& operator[](size_t index) {
        return data[index];
    }

    const double& operator[](size_t index) const {
        return data[index];
    }

    size_t size() const {
        return data.size();
    }

    // Vector addition
    VectorSum<Vector, Vector> operator+(const Vector& rhs) const;

private:
    std::vector<double> data;
};

// Class for vector addition expressions
template <typename E1, typename E2>
class VectorSum : public VectorExpr<VectorSum<E1, E2>> {
public:
    VectorSum(const E1& u, const E2& v) : u(u), v(v) {
        if (u.size() != v.size()) {
            throw std::invalid_argument("Vectors must be of the same size");
        }
    }

    double operator[](size_t index) const {
        return u[index] + v[index];
    }

    size_t size() const {
        return u.size();
    }

private:
    const E1& u;
    const E2& v;
};

// Vector addition operator
template <typename E1, typename E2>
VectorSum<E1, E2> operator+(const VectorExpr<E1>& u, const VectorExpr<E2>& v) {
    return VectorSum<E1, E2>(u, v);
}

// Implement Vector's addition operator using expression templates
VectorSum<Vector, Vector> Vector::operator+(const Vector& rhs) const {
    return VectorSum<Vector, Vector>(*this, rhs);
}
```

In this framework:
- `VectorExpr` is a base class template representing any vector expression.
- `Vector` is the actual vector class inheriting from `VectorExpr<Vector>`.
- `VectorSum` is a template class representing the sum of two vector expressions.

##### Using the Expression Templates

Now we can use the expression templates to perform vector addition without creating unnecessary temporaries:

```cpp
int main() {
    Vector a(3), b(3), c(3);
    a[0] = 1; a[1] = 2; a[2] = 3;
    b[0] = 4; b[1] = 5; b[2] = 6;
    c[0] = 7; c[1] = 8; c[2] = 9;

    auto result = a + b + c;  // No temporary vectors created
    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result[i] << " ";
    }

    return 0;
}
```

The expression `a + b + c` creates a single composite expression that is evaluated in one pass, avoiding the creation of intermediate temporaries.

#### Advanced Usage: Expression Templates with Multiple Operations

Expression templates can also be extended to support more complex expressions involving multiple operations.

##### Extending the Framework

Let's extend the framework to include subtraction and scalar multiplication:

```cpp
template <typename E1, typename E2>
class VectorDifference : public VectorExpr<VectorDifference<E1, E2>> {
public:
    VectorDifference(const E1& u, const E2& v) : u(u), v(v) {
        if (u.size() != v.size()) {
            throw std::invalid_argument("Vectors must be of the same size");
        }
    }

    double operator[](size_t index) const {
        return u[index] - v[index];
    }

    size_t size() const {
        return u.size();
    }

private:
    const E1& u;
    const E2& v;
};

template <typename E>
class ScalarMultiply : public VectorExpr<ScalarMultiply<E>> {
public:
    ScalarMultiply(double scalar, const E& v) : scalar(scalar), v(v) {}

    double operator[](size_t index) const {
        return scalar * v[index];
    }

    size_t size() const {
        return v.size();
    }

private:
    double scalar;
    const E& v;
};

template <typename E1, typename E2>
VectorDifference<E1, E2> operator-(const VectorExpr<E1>& u, const VectorExpr<E2>& v) {
    return VectorDifference<E1, E2>(u, v);
}

template <typename E>
ScalarMultiply<E> operator*(double scalar, const VectorExpr<E>& v) {
    return ScalarMultiply<E>(scalar, v);
}

template <typename E>
ScalarMultiply<E> operator*(const VectorExpr<E>& v, double scalar) {
    return ScalarMultiply<E>(scalar, v);
}
```

Now we can use the extended framework to perform complex vector operations efficiently:

```cpp
int main() {
    Vector a(3), b(3), c(3);
    a[0] = 1; a[1] = 2; a[2] = 3;
    b[0] = 4; b[1] = 5; b[2] = 6;
    c[0] = 7; c[1] = 8; c[2] = 9;

    auto result = a + b - c * 2.0;  // Composite expression with multiple operations
    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result[i] << " ";
    }

    return 0;
}
```

The expression `a + b - c * 2.0` is represented as a single composite expression, evaluated efficiently without creating temporary objects.

#### Benefits of Expression Templates

1. **Performance**: By eliminating intermediate temporaries, expression templates can significantly improve the performance of complex expressions.
2. **Memory Efficiency**:

Reduced memory usage due to the elimination of unnecessary temporaries.
3. **Readability**: Complex expressions can be written in a natural and readable way without sacrificing performance.
4. **Flexibility**: The template-based approach allows for easy extension to support additional operations and optimizations.

#### Conclusion

Expression templates are a powerful tool in C++ metaprogramming that enable efficient and flexible expression evaluation. By representing expressions as types, they allow the compiler to optimize complex expressions, eliminating unnecessary temporaries and improving performance. This subchapter has provided an in-depth exploration of expression templates, from basic concepts to advanced usage, illustrated with comprehensive code examples. With this knowledge, you can leverage expression templates to create highly efficient and expressive DSLs in C++.

### 26.4. Examples of DSLs in C++

In this subchapter, we explore practical examples of domain-specific languages (DSLs) implemented in C++. DSLs are specialized languages tailored to specific problem domains, offering expressive syntax and semantics to simplify complex tasks. We will examine several examples of DSLs, highlighting their design and implementation. These examples will demonstrate the versatility and power of C++ in creating effective DSLs.

#### Example 1: A Simple Query Language

Our first example is a DSL for querying data. This DSL allows users to construct complex queries in a readable and concise manner.

##### Defining the Query Language

We start by defining the basic constructs of our query language. We'll focus on filtering and selecting data from a collection.

```cpp
#include <iostream>

#include <vector>
#include <string>

#include <functional>

class Query {
public:
    Query(std::function<bool(const std::string&)> predicate) : predicate(predicate) {}

    Query operator&&(const Query& other) const {
        return Query([=](const std::string& s) {
            return this->predicate(s) && other.predicate(s);
        });
    }

    Query operator||(const Query& other) const {
        return Query([=](const std::string& s) {
            return this->predicate(s) || other.predicate(s);
        });
    }

    bool evaluate(const std::string& s) const {
        return predicate(s);
    }

private:
    std::function<bool(const std::string&)> predicate;
};

class DataCollection {
public:
    DataCollection(const std::vector<std::string>& data) : data(data) {}

    std::vector<std::string> filter(const Query& query) const {
        std::vector<std::string> result;
        for (const auto& item : data) {
            if (query.evaluate(item)) {
                result.push_back(item);
            }
        }
        return result;
    }

private:
    std::vector<std::string> data;
};
```

##### Using the Query Language

With the query language defined, we can create queries and apply them to a data collection.

```cpp
int main() {
    std::vector<std::string> data = {"apple", "banana", "cherry", "date", "elderberry", "fig", "grape"};

    DataCollection collection(data);

    auto starts_with_b = Query([](const std::string& s) { return s[0] == 'b'; });
    auto ends_with_e = Query([](const std::string& s) { return s.back() == 'e'; });

    auto query = starts_with_b || ends_with_e;

    auto result = collection.filter(query);

    for (const auto& item : result) {
        std::cout << item << " ";
    }

    return 0;
}
```

Output:

```
banana date
```

In this example, the DSL allows users to define complex queries using logical operators. The query `starts_with_b || ends_with_e` filters the collection to include items that start with 'b' or end with 'e'.

#### Example 2: A DSL for Building HTML

Our second example is a DSL for constructing HTML documents. This DSL provides a fluent interface for creating HTML elements and attributes.

##### Defining the HTML Builder

We define classes for HTML elements and attributes, allowing for nested structures.

```cpp
#include <iostream>

#include <string>
#include <vector>

class HTMLElement {
public:
    HTMLElement(const std::string& name) : name(name) {}

    HTMLElement& addChild(const HTMLElement& child) {
        children.push_back(child);
        return *this;
    }

    HTMLElement& setAttribute(const std::string& key, const std::string& value) {
        attributes.push_back({key, value});
        return *this;
    }

    std::string toString() const {
        std::string result = "<" + name;
        for (const auto& attr : attributes) {
            result += " " + attr.first + "=\"" + attr.second + "\"";
        }
        result += ">";
        for (const auto& child : children) {
            result += child.toString();
        }
        result += "</" + name + ">";
        return result;
    }

private:
    std::string name;
    std::vector<std::pair<std::string, std::string>> attributes;
    std::vector<HTMLElement> children;
};
```

##### Using the HTML Builder

We can now use the HTML builder to create an HTML document.

```cpp
int main() {
    HTMLElement html("html");
    HTMLElement head("head");
    HTMLElement body("body");

    head.addChild(HTMLElement("title").addChild(HTMLElement("Text")));

    body.addChild(HTMLElement("h1").addChild(HTMLElement("Hello, World!")))
        .addChild(HTMLElement("p").addChild(HTMLElement("This is a paragraph.")))
        .setAttribute("style", "color: red;");

    html.addChild(head).addChild(body);

    std::cout << html.toString() << std::endl;

    return 0;
}
```

Output:

```html
<html><head><title>Text</title></head><body style="color: red;"><h1>Hello, World!</h1><p>This is a paragraph.</p></body></html>
```

In this example, the DSL provides a fluent interface for constructing HTML elements, making the code more readable and expressive.

#### Example 3: A Matrix Computation DSL

Our third example is a DSL for matrix computations. This DSL allows for concise and efficient matrix operations.

##### Defining the Matrix Class

We start by defining a basic matrix class with support for addition, subtraction, and multiplication.

```cpp
#include <iostream>

#include <vector>
#include <stdexcept>

class Matrix {
public:
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols), data(rows, std::vector<double>(cols, 0.0)) {}

    std::vector<double>& operator[](size_t row) {
        return data[row];
    }

    const std::vector<double>& operator[](size_t row) const {
        return data[row];
    }

    size_t rowCount() const {
        return rows;
    }

    size_t colCount() const {
        return cols;
    }

    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }

        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] = data[i][j] + other[i][j];
            }
        }

        return result;
    }

    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must match for subtraction");
        }

        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] = data[i][j] - other[i][j];
            }
        }

        return result;
    }

    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions must match for multiplication");
        }

        Matrix result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                for (size_t k = 0; k < cols; ++k) {
                    result[i][j] += data[i][k] * other[k][j];
                }
            }
        }

        return result;
    }

private:
    size_t rows, cols;
    std::vector<std::vector<double>> data;
};
```

##### Using the Matrix Computation DSL

We can now perform matrix operations using the defined class.

```cpp
int main() {
    Matrix a(2, 2);
    Matrix b(2, 2);

    a[0][0] = 1; a[0][1] = 2;
    a[1][0] = 3; a[1][1] = 4;

    b[0][0] = 5; b[0][1] = 6;
    b[1][0] = 7; b[1][1] = 8;

    Matrix c = a + b;
    Matrix d = a - b;
    Matrix e = a * b;

    std::cout << "Matrix c (a + b):" << std::endl;
    for (size_t i = 0; i < c.rowCount(); ++i) {
        for (size_t j = 0; j < c.colCount(); ++j) {
            std::cout << c[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Matrix d (a - b):" << std::endl;
    for (size_t i = 0; i < d.rowCount(); ++i) {
        for (size_t j = 0; j < d.colCount(); ++j) {
            std::cout << d[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Matrix e (a * b):" << std::endl;
    for (size_t i = 0; i < e.rowCount(); ++i) {
        for (size_t j = 0; j < e.colCount(); ++j) {
            std::cout << e[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
```

Output:

```
Matrix c (a + b):
6 8 
10 12 
Matrix d (a - b):
-4 -4 
-4 -4 
Matrix e (a * b):
19 22 
43 50 
```

In this example, the DSL simplifies matrix operations, making the code more readable and maintainable.

#### Conclusion

These examples demonstrate how C++ can be used to create effective and efficient domain-specific languages. By leveraging C++'s powerful features such as operator overloading, template metaprogramming, and functional programming constructs, we can design DSLs that improve the readability, maintainability, and performance of code in specific problem domains. Whether it's querying data, building HTML, or performing matrix computations, DSLs provide a powerful tool for developers to express complex operations concisely and clearly.
