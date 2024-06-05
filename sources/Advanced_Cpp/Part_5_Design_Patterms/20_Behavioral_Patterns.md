

\newpage
## Chapter 20: Behavioral Patterns

In the realm of software design, Behavioral Patterns play a crucial role in managing the complex interactions and responsibilities between objects. These patterns define how objects communicate and collaborate to achieve cohesive functionality and flexibility. This chapter delves into four key Behavioral Patterns: the Strategy Pattern, which encapsulates algorithms to enable dynamic selection at runtime; the Observer Pattern, which fosters loose coupling by allowing objects to subscribe and react to state changes in other objects; the Visitor Pattern, which enables operations on a collection of diverse objects without altering their classes; and the Chain of Responsibility Pattern, which provides a mechanism for passing requests along a chain of potential handlers. Through detailed examples and C++ implementations, this chapter will guide you in leveraging these patterns to create more robust and maintainable software systems.

### 20.1. Strategy Pattern: Encapsulating Algorithms

The Strategy Pattern is a design pattern that enables selecting an algorithm's behavior at runtime. Instead of implementing a single algorithm directly, the code receives run-time instructions as to which in a family of algorithms to use. This pattern is particularly useful for promoting flexibility and reusable code, allowing algorithms to be interchanged without altering the client code that uses them.

#### 20.1.1. Understanding the Strategy Pattern

The Strategy Pattern involves three primary components:
1. **Strategy Interface**: An interface common to all supported algorithms. This interface makes it possible to interchange algorithms.
2. **Concrete Strategies**: Classes that implement the Strategy interface. Each class encapsulates a specific algorithm.
3. **Context**: A class that uses a Strategy object to invoke the algorithm defined by a Concrete Strategy. The Context maintains a reference to a Strategy object and delegates the algorithm execution to it.

This structure decouples the algorithm implementation from the context in which it is used, facilitating easier maintenance and expansion.

#### 20.1.2. Implementing the Strategy Pattern in C++

Let's consider an example where we need to sort a list of integers. We may have different sorting algorithms like Bubble Sort, Quick Sort, and Merge Sort. The Strategy Pattern allows us to encapsulate these sorting algorithms and interchange them dynamically.

##### 20.1.2.1. Defining the Strategy Interface

First, we define a common interface for all sorting strategies:

```cpp
// Strategy.h
#ifndef STRATEGY_H

#define STRATEGY_H

#include <vector>

class Strategy {
public:
    virtual ~Strategy() = default;
    virtual void sort(std::vector<int>& data) = 0;
};

#endif // STRATEGY_H
```

##### 20.1.2.2. Implementing Concrete Strategies

Next, we implement several sorting algorithms that conform to the `Strategy` interface:

```cpp
// BubbleSort.h
#ifndef BUBBLESORT_H

#define BUBBLESORT_H

#include "Strategy.h"

class BubbleSort : public Strategy {
public:
    void sort(std::vector<int>& data) override {
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data.size() - i - 1; ++j) {
                if (data[j] > data[j + 1]) {
                    std::swap(data[j], data[j + 1]);
                }
            }
        }
    }
};

#endif // BUBBLESORT_H
```

```cpp
// QuickSort.h
#ifndef QUICKSORT_H

#define QUICKSORT_H

#include "Strategy.h"

class QuickSort : public Strategy {
public:
    void sort(std::vector<int>& data) override {
        quickSort(data, 0, data.size() - 1);
    }

private:
    void quickSort(std::vector<int>& data, int low, int high) {
        if (low < high) {
            int pi = partition(data, low, high);
            quickSort(data, low, pi - 1);
            quickSort(data, pi + 1, high);
        }
    }

    int partition(std::vector<int>& data, int low, int high) {
        int pivot = data[high];
        int i = (low - 1);
        for (int j = low; j < high; ++j) {
            if (data[j] < pivot) {
                ++i;
                std::swap(data[i], data[j]);
            }
        }
        std::swap(data[i + 1], data[high]);
        return (i + 1);
    }
};

#endif // QUICKSORT_H
```

```cpp
// MergeSort.h
#ifndef MERGESORT_H

#define MERGESORT_H

#include "Strategy.h"

class MergeSort : public Strategy {
public:
    void sort(std::vector<int>& data) override {
        if (data.size() > 1) {
            mergeSort(data, 0, data.size() - 1);
        }
    }

private:
    void mergeSort(std::vector<int>& data, int left, int right) {
        if (left < right) {
            int mid = left + (right - left) / 2;
            mergeSort(data, left, mid);
            mergeSort(data, mid + 1, right);
            merge(data, left, mid, right);
        }
    }

    void merge(std::vector<int>& data, int left, int mid, int right) {
        int n1 = mid - left + 1;
        int n2 = right - mid;

        std::vector<int> leftArray(n1);
        std::vector<int> rightArray(n2);

        for (int i = 0; i < n1; ++i) {
            leftArray[i] = data[left + i];
        }
        for (int j = 0; j < n2; ++j) {
            rightArray[j] = data[mid + 1 + j];
        }

        int i = 0, j = 0, k = left;
        while (i < n1 && j < n2) {
            if (leftArray[i] <= rightArray[j]) {
                data[k] = leftArray[i];
                ++i;
            } else {
                data[k] = rightArray[j];
                ++j;
            }
            ++k;
        }

        while (i < n1) {
            data[k] = leftArray[i];
            ++i;
            ++k;
        }

        while (j < n2) {
            data[k] = rightArray[j];
            ++j;
            ++k;
        }
    }
};

#endif // MERGESORT_H
```

##### 20.1.2.3. Implementing the Context

The `Context` class will maintain a reference to a `Strategy` object and delegate the sorting task to it:

```cpp
// Context.h
#ifndef CONTEXT_H

#define CONTEXT_H

#include "Strategy.h"

#include <memory>

class Context {
public:
    void setStrategy(std::shared_ptr<Strategy> strategy) {
        this->strategy = strategy;
    }

    void executeStrategy(std::vector<int>& data) {
        if (strategy) {
            strategy->sort(data);
        }
    }

private:
    std::shared_ptr<Strategy> strategy;
};

#endif // CONTEXT_H
```

##### 20.1.2.4. Using the Strategy Pattern

Here's how we can use the Strategy Pattern to sort a list of integers with different algorithms dynamically:

```cpp
// main.cpp
#include <iostream>

#include "Context.h"
#include "BubbleSort.h"

#include "QuickSort.h"
#include "MergeSort.h"

int main() {
    std::vector<int> data = {34, 7, 23, 32, 5, 62};

    Context context;

    std::cout << "Original data: ";
    for (int num : data) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Using BubbleSort
    context.setStrategy(std::make_shared<BubbleSort>());
    context.executeStrategy(data);

    std::cout << "BubbleSorted data: ";
    for (int num : data) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Reset data
    data = {34, 7, 23, 32, 5, 62};

    // Using QuickSort
    context.setStrategy(std::make_shared<QuickSort>());
    context.executeStrategy(data);

    std::cout << "QuickSorted data: ";
    for (int num : data) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Reset data
    data = {34, 7, 23, 32, 5, 62};

    // Using MergeSort
    context.setStrategy(std::make_shared<MergeSort>());
    context.executeStrategy(data);

    std::cout << "MergeSorted data: ";
    for (int num : data) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

#### 20.1.3. Benefits of the Strategy Pattern

1. **Flexibility**: The Strategy Pattern allows algorithms to be selected and changed dynamically.
2. **Reusability**: By encapsulating algorithms in separate classes, they can be reused across different contexts.
3. **Maintainability**: Adding new algorithms does not affect the client code; you only need to implement the new strategy and use it with the context.

#### 20.1.4. When to Use the Strategy Pattern

- When you have multiple algorithms for a specific task and want to switch between them dynamically.
- When you want to isolate the implementation details of an algorithm from the context that uses it.
- When you want to eliminate conditional statements for selecting different algorithms.

By understanding and implementing the Strategy Pattern, you can design flexible and maintainable software systems that easily adapt to changing requirements and extend functionalities with minimal modifications to the existing codebase.

### 20.2. Observer Pattern: Promoting Loose Coupling

The Observer Pattern is a behavioral design pattern that defines a one-to-many dependency between objects. When the state of one object changes, all its dependents (observers) are notified and updated automatically. This pattern is particularly useful in scenarios where an object should notify other objects about changes in its state without knowing who these objects are, thus promoting loose coupling.

#### 20.2.1. Understanding the Observer Pattern

The Observer Pattern involves several key components:
1. **Subject**: The object that maintains a list of observers and sends notifications when its state changes.
2. **Observer**: An interface or abstract class defining the update method, which gets called when the subject's state changes.
3. **Concrete Subject**: The subject implementation, which maintains the state of interest and notifies observers of state changes.
4. **Concrete Observer**: The observer implementations that respond to the state changes of the subject.

This pattern ensures that objects are decoupled as much as possible, making the system more modular and easier to maintain and extend.

#### 20.2.2. Implementing the Observer Pattern in C++

Let's consider an example where we have a weather station that monitors temperature changes and notifies various display units (observers) about these changes. This example will illustrate how to implement the Observer Pattern in C++.

##### 20.2.2.1. Defining the Observer Interface

First, we define an abstract class for observers that declares the `update` method:

```cpp
// Observer.h
#ifndef OBSERVER_H

#define OBSERVER_H

class Observer {
public:
    virtual ~Observer() = default;
    virtual void update(float temperature) = 0;
};

#endif // OBSERVER_H
```

##### 20.2.2.2. Defining the Subject Interface

Next, we define an abstract class for the subject that declares methods for attaching, detaching, and notifying observers:

```cpp
// Subject.h
#ifndef SUBJECT_H

#define SUBJECT_H

#include <vector>

#include <memory>
#include "Observer.h"

class Subject {
public:
    virtual ~Subject() = default;

    void attach(std::shared_ptr<Observer> observer) {
        observers.push_back(observer);
    }

    void detach(std::shared_ptr<Observer> observer) {
        observers.erase(
            std::remove(observers.begin(), observers.end(), observer), 
            observers.end()
        );
    }

    void notify(float temperature) {
        for (const auto& observer : observers) {
            observer->update(temperature);
        }
    }

private:
    std::vector<std::shared_ptr<Observer>> observers;
};

#endif // SUBJECT_H
```

##### 20.2.2.3. Implementing the Concrete Subject

The `WeatherStation` class maintains the current temperature and notifies its observers whenever the temperature changes:

```cpp
// WeatherStation.h
#ifndef WEATHERSTATION_H

#define WEATHERSTATION_H

#include "Subject.h"

class WeatherStation : public Subject {
public:
    void setTemperature(float temperature) {
        this->temperature = temperature;
        notify(temperature);
    }

private:
    float temperature;
};

#endif // WEATHERSTATION_H
```

##### 20.2.2.4. Implementing Concrete Observers

Now, we implement concrete observers that display the temperature in different ways:

```cpp
// CurrentConditionsDisplay.h
#ifndef CURRENTCONDITIONSDISPLAY_H

#define CURRENTCONDITIONSDISPLAY_H

#include <iostream>

#include "Observer.h"

class CurrentConditionsDisplay : public Observer {
public:
    void update(float temperature) override {
        std::cout << "Current conditions: " << temperature << "°C" << std::endl;
    }
};

#endif // CURRENTCONDITIONSDISPLAY_H
```

```cpp
// StatisticsDisplay.h
#ifndef STATISTICSDISPLAY_H

#define STATISTICSDISPLAY_H

#include <iostream>

#include "Observer.h"

class StatisticsDisplay : public Observer {
public:
    void update(float temperature) override {
        totalTemperature += temperature;
        ++numReadings;
        float avgTemperature = totalTemperature / numReadings;
        std::cout << "Average temperature: " << avgTemperature << "°C" << std::endl;
    }

private:
    float totalTemperature = 0;
    int numReadings = 0;
};

#endif // STATISTICSDISPLAY_H
```

##### 20.2.2.5. Using the Observer Pattern

Here's how we can use the Observer Pattern to monitor and display temperature changes:

```cpp
// main.cpp
#include "WeatherStation.h"

#include "CurrentConditionsDisplay.h"
#include "StatisticsDisplay.h"

int main() {
    std::shared_ptr<WeatherStation> weatherStation = std::make_shared<WeatherStation>();

    std::shared_ptr<CurrentConditionsDisplay> currentDisplay = std::make_shared<CurrentConditionsDisplay>();
    std::shared_ptr<StatisticsDisplay> statisticsDisplay = std::make_shared<StatisticsDisplay>();

    weatherStation->attach(currentDisplay);
    weatherStation->attach(statisticsDisplay);

    weatherStation->setTemperature(25.0);
    weatherStation->setTemperature(26.5);
    weatherStation->setTemperature(27.3);

    weatherStation->detach(currentDisplay);

    weatherStation->setTemperature(28.1);

    return 0;
}
```

#### 20.2.3. Benefits of the Observer Pattern

1. **Loose Coupling**: The Observer Pattern promotes loose coupling between the subject and observers. The subject knows nothing about the observers except that they implement the Observer interface.
2. **Scalability**: New observers can be added without modifying the subject. This makes the system easy to scale and extend.
3. **Flexibility**: Observers can be attached and detached at runtime, providing a dynamic and flexible way to handle updates.

#### 20.2.4. When to Use the Observer Pattern

- When a change to one object requires changing other objects, and you do not know how many objects need to be changed.
- When an object should be able to notify other objects without making assumptions about who these objects are.
- When you want to promote loose coupling in your system, making it more modular and maintainable.

By implementing the Observer Pattern, you can design systems where objects can communicate and respond to changes without being tightly coupled, leading to more flexible and maintainable software architectures.

### 20.3. Visitor Pattern: Operations on Object Structures

The Visitor Pattern is a behavioral design pattern that allows you to add further operations to objects without modifying them. This pattern is particularly useful when you have a structure of objects (such as a composite pattern) and want to perform operations on these objects that are not central to their functionality. The Visitor Pattern promotes the open/closed principle, allowing you to add new operations without changing the existing code.

#### 20.3.1. Understanding the Visitor Pattern

The Visitor Pattern involves several key components:
1. **Visitor Interface**: An interface that declares a visit method for each type of element in the object structure.
2. **Concrete Visitor**: A class that implements the visitor interface, defining specific operations to be performed on each type of element.
3. **Element Interface**: An interface or abstract class that declares an accept method, which takes a visitor as an argument.
4. **Concrete Element**: A class that implements the element interface and defines the accept method, which calls the appropriate visit method on the visitor.

This structure decouples the operations from the object structure, allowing new operations to be added easily.

#### 20.3.2. Implementing the Visitor Pattern in C++

Let's consider an example where we have a hierarchy of shapes (such as Circle, Rectangle, and Triangle) and want to perform various operations on these shapes, such as calculating their area and drawing them. This example will illustrate how to implement the Visitor Pattern in C++.

##### 20.3.2.1. Defining the Visitor Interface

First, we define an interface for visitors that declares visit methods for each type of shape:

```cpp
// Visitor.h
#ifndef VISITOR_H

#define VISITOR_H

class Circle;
class Rectangle;
class Triangle;

class Visitor {
public:
    virtual ~Visitor() = default;
    virtual void visit(Circle& circle) = 0;
    virtual void visit(Rectangle& rectangle) = 0;
    virtual void visit(Triangle& triangle) = 0;
};

#endif // VISITOR_H
```

##### 20.3.2.2. Defining the Element Interface

Next, we define an interface for elements that declares an accept method:

```cpp
// Element.h
#ifndef ELEMENT_H

#define ELEMENT_H

#include "Visitor.h"

class Element {
public:
    virtual ~Element() = default;
    virtual void accept(Visitor& visitor) = 0;
};

#endif // ELEMENT_H
```

##### 20.3.2.3. Implementing Concrete Elements

Now, we implement the concrete shapes (Circle, Rectangle, and Triangle) that accept a visitor:

```cpp
// Circle.h
#ifndef CIRCLE_H

#define CIRCLE_H

#include "Element.h"

class Circle : public Element {
public:
    Circle(double radius) : radius(radius) {}
    
    double getRadius() const { return radius; }

    void accept(Visitor& visitor) override {
        visitor.visit(*this);
    }

private:
    double radius;
};

#endif // CIRCLE_H
```

```cpp
// Rectangle.h
#ifndef RECTANGLE_H

#define RECTANGLE_H

#include "Element.h"

class Rectangle : public Element {
public:
    Rectangle(double width, double height) : width(width), height(height) {}
    
    double getWidth() const { return width; }
    double getHeight() const { return height; }

    void accept(Visitor& visitor) override {
        visitor.visit(*this);
    }

private:
    double width;
    double height;
};

#endif // RECTANGLE_H
```

```cpp
// Triangle.h
#ifndef TRIANGLE_H

#define TRIANGLE_H

#include "Element.h"

class Triangle : public Element {
public:
    Triangle(double base, double height) : base(base), height(height) {}
    
    double getBase() const { return base; }
    double getHeight() const { return height; }

    void accept(Visitor& visitor) override {
        visitor.visit(*this);
    }

private:
    double base;
    double height;
};

#endif // TRIANGLE_H
```

##### 20.3.2.4. Implementing Concrete Visitors

Next, we implement concrete visitors that define specific operations to be performed on the shapes:

```cpp
// AreaVisitor.h
#ifndef AREAVISITOR_H

#define AREAVISITOR_H

#include <iostream>

#include "Visitor.h"
#include "Circle.h"

#include "Rectangle.h"
#include "Triangle.h"

class AreaVisitor : public Visitor {
public:
    void visit(Circle& circle) override {
        double area = 3.14159 * circle.getRadius() * circle.getRadius();
        std::cout << "Circle area: " << area << std::endl;
    }

    void visit(Rectangle& rectangle) override {
        double area = rectangle.getWidth() * rectangle.getHeight();
        std::cout << "Rectangle area: " << area << std::endl;
    }

    void visit(Triangle& triangle) override {
        double area = 0.5 * triangle.getBase() * triangle.getHeight();
        std::cout << "Triangle area: " << area << std::endl;
    }
};

#endif // AREAVISITOR_H
```

```cpp
// DrawVisitor.h
#ifndef DRAWVISITOR_H

#define DRAWVISITOR_H

#include <iostream>

#include "Visitor.h"
#include "Circle.h"

#include "Rectangle.h"
#include "Triangle.h"

class DrawVisitor : public Visitor {
public:
    void visit(Circle& circle) override {
        std::cout << "Drawing a circle with radius " << circle.getRadius() << std::endl;
    }

    void visit(Rectangle& rectangle) override {
        std::cout << "Drawing a rectangle with width " << rectangle.getWidth() << " and height " << rectangle.getHeight() << std::endl;
    }

    void visit(Triangle& triangle) override {
        std::cout << "Drawing a triangle with base " << triangle.getBase() << " and height " << triangle.getHeight() << std::endl;
    }
};

#endif // DRAWVISITOR_H
```

##### 20.3.2.5. Using the Visitor Pattern

Here's how we can use the Visitor Pattern to perform operations on a collection of shapes:

```cpp
// main.cpp
#include "Circle.h"

#include "Rectangle.h"
#include "Triangle.h"

#include "AreaVisitor.h"
#include "DrawVisitor.h"

int main() {
    std::vector<std::shared_ptr<Element>> shapes;
    shapes.push_back(std::make_shared<Circle>(5.0));
    shapes.push_back(std::make_shared<Rectangle>(4.0, 6.0));
    shapes.push_back(std::make_shared<Triangle>(3.0, 7.0));

    AreaVisitor areaVisitor;
    DrawVisitor drawVisitor;

    for (auto& shape : shapes) {
        shape->accept(areaVisitor);
        shape->accept(drawVisitor);
    }

    return 0;
}
```

#### 20.3.3. Benefits of the Visitor Pattern

1. **Separation of Concerns**: The Visitor Pattern separates algorithms from the objects on which they operate, promoting a clear division of responsibilities.
2. **Extensibility**: New operations can be added without modifying the object structure by simply creating new visitor classes.
3. **Single Responsibility**: Each visitor handles a specific operation, adhering to the single responsibility principle.

#### 20.3.4. When to Use the Visitor Pattern

- When you have an object structure, such as a composite pattern, and want to perform operations on these objects without changing their classes.
- When you want to add new operations to existing object structures without modifying their code.
- When you need to perform multiple unrelated operations on an object structure, and you want to keep these operations separate and modular.

By implementing the Visitor Pattern, you can design systems where operations on object structures are flexible and extensible, allowing new functionality to be added with minimal changes to existing code. This pattern is particularly useful in scenarios where the object structure is stable but the operations on it vary frequently.

### 20.4. Chain of Responsibility: Passing Requests Along the Chain

The Chain of Responsibility Pattern is a behavioral design pattern that allows an object to pass a request along a chain of potential handlers until the request is handled. This pattern decouples the sender and receiver of a request, providing multiple objects a chance to handle the request without the sender needing to know which object will handle it.

#### 20.4.1. Understanding the Chain of Responsibility Pattern

The Chain of Responsibility Pattern involves several key components:
1. **Handler Interface**: An interface that defines a method for handling requests and setting the next handler in the chain.
2. **Concrete Handler**: A class that implements the handler interface and processes requests or forwards them to the next handler.
3. **Client**: The object that initiates the request and forwards it to the chain.

This pattern promotes loose coupling and provides flexibility in processing requests. It is particularly useful in scenarios where multiple objects can handle a request and the handler is determined at runtime.

#### 20.4.2. Implementing the Chain of Responsibility Pattern in C++

Let's consider an example where we have a series of logging handlers (such as ConsoleLogger, FileLogger, and EmailLogger) that process log messages based on their severity. This example will illustrate how to implement the Chain of Responsibility Pattern in C++.

##### 20.4.2.1. Defining the Handler Interface

First, we define an interface for handlers that declares a method for handling requests and setting the next handler in the chain:

```cpp
// Handler.h
#ifndef HANDLER_H

#define HANDLER_H

#include <memory>

#include <string>

class Handler {
public:
    virtual ~Handler() = default;

    void setNext(std::shared_ptr<Handler> nextHandler) {
        next = nextHandler;
    }

    virtual void handleRequest(const std::string& message, int level) {
        if (next) {
            next->handleRequest(message, level);
        }
    }

protected:
    std::shared_ptr<Handler> next;
};

#endif // HANDLER_H
```

##### 20.4.2.2. Implementing Concrete Handlers

Next, we implement concrete handlers that process requests or forward them to the next handler:

```cpp
// ConsoleLogger.h
#ifndef CONSOLELOGGER_H

#define CONSOLELOGGER_H

#include "Handler.h"

#include <iostream>

class ConsoleLogger : public Handler {
public:
    ConsoleLogger(int logLevel) : level(logLevel) {}

    void handleRequest(const std::string& message, int level) override {
        if (this->level <= level) {
            std::cout << "ConsoleLogger: " << message << std::endl;
        }
        Handler::handleRequest(message, level);
    }

private:
    int level;
};

#endif // CONSOLELOGGER_H
```

```cpp
// FileLogger.h
#ifndef FILELOGGER_H

#define FILELOGGER_H

#include "Handler.h"

#include <iostream>

class FileLogger : public Handler {
public:
    FileLogger(int logLevel) : level(logLevel) {}

    void handleRequest(const std::string& message, int level) override {
        if (this->level <= level) {
            // For simplicity, we'll just print to console instead of writing to a file
            std::cout << "FileLogger: " << message << std::endl;
        }
        Handler::handleRequest(message, level);
    }

private:
    int level;
};

#endif // FILELOGGER_H
```

```cpp
// EmailLogger.h
#ifndef EMAILLOGGER_H

#define EMAILLOGGER_H

#include "Handler.h"

#include <iostream>

class EmailLogger : public Handler {
public:
    EmailLogger(int logLevel) : level(logLevel) {}

    void handleRequest(const std::string& message, int level) override {
        if (this->level <= level) {
            // For simplicity, we'll just print to console instead of sending an email
            std::cout << "EmailLogger: " << message << std::endl;
        }
        Handler::handleRequest(message, level);
    }

private:
    int level;
};

#endif // EMAILLOGGER_H
```

##### 20.4.2.3. Using the Chain of Responsibility Pattern

Here's how we can set up and use the chain of logging handlers to process log messages based on their severity:

```cpp
// main.cpp
#include "ConsoleLogger.h"

#include "FileLogger.h"
#include "EmailLogger.h"

int main() {
    std::shared_ptr<Handler> consoleLogger = std::make_shared<ConsoleLogger>(1);
    std::shared_ptr<Handler> fileLogger = std::make_shared<FileLogger>(2);
    std::shared_ptr<Handler> emailLogger = std::make_shared<EmailLogger>(3);

    consoleLogger->setNext(fileLogger);
    fileLogger->setNext(emailLogger);

    std::cout << "Sending log message with severity 1 (INFO):" << std::endl;
    consoleLogger->handleRequest("This is an information message.", 1);

    std::cout << "\nSending log message with severity 2 (WARNING):" << std::endl;
    consoleLogger->handleRequest("This is a warning message.", 2);

    std::cout << "\nSending log message with severity 3 (ERROR):" << std::endl;
    consoleLogger->handleRequest("This is an error message.", 3);

    return 0;
}
```

#### 20.4.3. Benefits of the Chain of Responsibility Pattern

1. **Decoupling**: The Chain of Responsibility Pattern decouples the sender of a request from its receivers by allowing multiple objects to handle the request.
2. **Flexibility**: Handlers can be added or removed dynamically without modifying the client code or other handlers.
3. **Scalability**: The pattern makes it easy to scale the handling mechanism by adding more handlers to the chain.

#### 20.4.4. When to Use the Chain of Responsibility Pattern

- When multiple objects can handle a request, and the handler is determined at runtime.
- When you want to decouple the sender and receiver of a request.
- When you want to simplify the client code by allowing the request to be passed along a chain of potential handlers.

By implementing the Chain of Responsibility Pattern, you can design systems where requests are processed flexibly and dynamically, allowing new handlers to be added with minimal changes to existing code. This pattern is particularly useful in scenarios where the handling mechanism needs to be flexible and extensible, such as in logging, event handling, and command processing systems.
