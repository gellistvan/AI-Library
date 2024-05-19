
\newpage
#part V: Software Design
\newpage
## Chapter 19: Structural Patterns

In the realm of software design, structural patterns play a pivotal role in defining the relationships between objects, allowing for more efficient and scalable code architecture. This chapter delves into some of the most essential structural patterns in C++ programming. We begin with the Composite Pattern, which simplifies the management of object hierarchies, enabling complex structures to be treated uniformly. Next, we explore the Flyweight Pattern, a technique that minimizes memory consumption by sharing as much data as possible. The Bridge Pattern follows, offering a robust solution to decouple abstraction from implementation, thereby enhancing flexibility and maintainability. Finally, we examine the Proxy Pattern, a powerful tool for controlling access to objects, providing an additional layer of security and functionality. Together, these patterns equip you with the strategies needed to tackle intricate design challenges and optimize your C++ applications.

### 19.1. Composite Pattern: Managing Hierarchies

The Composite Pattern is a structural pattern that enables clients to treat individual objects and compositions of objects uniformly. This pattern is particularly useful when dealing with tree structures, such as file systems, organizational charts, or any scenario where individual objects and groups of objects should be treated the same way. In C++, the Composite Pattern helps manage hierarchies by creating a common interface for both simple and composite objects, allowing for flexible and reusable code.

#### 19.1.1. The Problem

Imagine you are developing a graphics application where you need to draw shapes such as circles and squares. Some shapes might be simple (like a single circle), while others might be complex (like a group of shapes). You need a way to treat these shapes uniformly, so you can perform operations like drawing, moving, or resizing, regardless of whether the shape is simple or complex.

#### 19.1.2. The Solution

The Composite Pattern provides a solution by defining a unified interface for both simple and composite objects. In C++, this involves creating an abstract base class that declares the common operations, and then deriving both simple and composite classes from this base class.

#### 19.1.3. Implementation

Let's walk through a detailed implementation of the Composite Pattern in C++. We'll create a hierarchy of shapes that can be drawn. The hierarchy will include both simple shapes (like `Circle` and `Square`) and composite shapes (like `Group`).

##### Step 1: Define the Component Interface

First, we define an abstract base class `Shape` that declares a common interface for all shapes.

```cpp
#include <iostream>
#include <vector>
#include <memory>

// Abstract base class
class Shape {
public:
    virtual void draw() const = 0; // Pure virtual function
    virtual ~Shape() = default; // Virtual destructor
};
```

The `Shape` class declares a pure virtual function `draw()` which must be implemented by all derived classes.

##### Step 2: Implement Leaf Classes

Next, we implement the simple shapes, `Circle` and `Square`, which are leaf nodes in our hierarchy.

```cpp
class Circle : public Shape {
public:
    void draw() const override {
        std::cout << "Drawing a Circle" << std::endl;
    }
};

class Square : public Shape {
public:
    void draw() const override {
        std::cout << "Drawing a Square" << std::endl;
    }
};
```

These classes override the `draw()` function to provide specific implementations for drawing a circle and a square.

##### Step 3: Implement the Composite Class

Now, we implement the `Group` class, which can contain multiple shapes. The `Group` class is a composite that can hold both simple and composite shapes.

```cpp
class Group : public Shape {
private:
    std::vector<std::shared_ptr<Shape>> shapes; // Vector to hold child shapes
public:
    void addShape(std::shared_ptr<Shape> shape) {
        shapes.push_back(shape);
    }

    void draw() const override {
        std::cout << "Drawing a Group of Shapes:" << std::endl;
        for (const auto& shape : shapes) {
            shape->draw();
        }
    }
};
```

The `Group` class contains a vector of `shared_ptr<Shape>`, allowing it to manage its child shapes. The `addShape()` method adds a shape to the group, and the `draw()` method iterates over the child shapes and calls their `draw()` methods.

##### Step 4: Using the Composite Pattern

Let's create some shapes and a group of shapes to see how the Composite Pattern works in practice.

```cpp
int main() {
    // Create individual shapes
    std::shared_ptr<Shape> circle1 = std::make_shared<Circle>();
    std::shared_ptr<Shape> circle2 = std::make_shared<Circle>();
    std::shared_ptr<Shape> square1 = std::make_shared<Square>();

    // Create a group and add shapes to it
    std::shared_ptr<Group> group1 = std::make_shared<Group>();
    group1->addShape(circle1);
    group1->addShape(square1);

    // Create another group and add shapes to it
    std::shared_ptr<Group> group2 = std::make_shared<Group>();
    group2->addShape(circle2);
    group2->addShape(group1); // Adding a group to another group

    // Draw the groups
    group2->draw();

    return 0;
}
```

In this example, we create two circles and one square. We then create two groups, `group1` and `group2`. We add the shapes to `group1` and then add `group1` to `group2`. When we call the `draw()` method on `group2`, it draws all the shapes it contains, demonstrating the uniform treatment of individual and composite objects.

#### 19.1.4. Benefits of the Composite Pattern

1. **Simplicity**: The Composite Pattern simplifies client code that deals with tree structures by allowing clients to treat individual objects and compositions of objects uniformly.
2. **Flexibility**: It is easy to add new types of components and composites without changing existing code, adhering to the Open/Closed Principle.
3. **Maintainability**: The pattern promotes cleaner, more maintainable code by centralizing common operations in the base class.

#### 19.1.5. Potential Drawbacks

1. **Complexity**: The pattern can introduce complexity by adding more classes to the system, especially if the hierarchy is deep.
2. **Performance**: Overhead may be introduced due to the need to manage and traverse the composite structures, which can affect performance in certain scenarios.

#### Conclusion

The Composite Pattern is a powerful tool for managing hierarchies in C++. By defining a unified interface for simple and composite objects, it allows for flexible and reusable code that can handle complex tree structures with ease. Whether you are working with graphical shapes, file systems, or organizational charts, the Composite Pattern provides a robust solution for treating individual and composite objects uniformly, ultimately leading to cleaner, more maintainable code.

### 19.2. Flyweight Pattern: Reducing Memory Usage

The Flyweight Pattern is a structural design pattern aimed at minimizing memory usage by sharing as much data as possible with similar objects. This pattern is particularly useful in scenarios where many fine-grained objects are needed, but the memory cost of creating and maintaining these objects individually would be prohibitive. By sharing common parts of state between multiple objects, the Flyweight Pattern can significantly reduce the amount of memory used.

#### 19.2.1. The Problem

Consider a text editor that needs to represent each character as an object. If every character were to have its own distinct object, the memory usage would be enormous, especially for large documents. The Flyweight Pattern helps solve this problem by sharing common state (intrinsic state) among multiple objects and maintaining only the unique state (extrinsic state) separately.

#### 19.2.2. The Solution

The Flyweight Pattern involves creating a flyweight factory that manages the creation and sharing of flyweight objects. Intrinsic state, which is shared among objects, is stored within the flyweight. Extrinsic state, which varies from one object to another, is stored outside the flyweight and passed to the flyweight when necessary.

#### 19.2.3. Implementation

Let's walk through a detailed implementation of the Flyweight Pattern in C++. We'll create a system to manage characters in a text editor efficiently.

##### Step 1: Define the Flyweight Class

First, we define a `Character` class that represents a character. This class will include intrinsic state shared among all characters and methods to operate on the characters.

```cpp
#include <iostream>
#include <unordered_map>
#include <string>
#include <memory>

// Flyweight class
class Character {
private:
    char symbol; // Intrinsic state

public:
    Character(char symbol) : symbol(symbol) {}

    void display(int fontSize) const {
        std::cout << "Character: " << symbol << ", Font size: " << fontSize << std::endl;
    }

    char getSymbol() const {
        return symbol;
    }
};
```

The `Character` class contains the intrinsic state (`symbol`) and a method to display the character with an extrinsic state (`fontSize`).

##### Step 2: Implement the Flyweight Factory

Next, we implement the `CharacterFactory` class, which manages the creation and sharing of `Character` objects.

```cpp
class CharacterFactory {
private:
    std::unordered_map<char, std::shared_ptr<Character>> characters; // Cache of flyweights

public:
    std::shared_ptr<Character> getCharacter(char symbol) {
        // Check if the character is already created
        auto it = characters.find(symbol);
        if (it != characters.end()) {
            return it->second;
        }

        // Create a new character and add it to the cache
        std::shared_ptr<Character> character = std::make_shared<Character>(symbol);
        characters[symbol] = character;
        return character;
    }
};
```

The `CharacterFactory` class maintains a cache of `Character` objects. When a request for a character is made, the factory checks if the character already exists in the cache. If it does, the existing character is returned; otherwise, a new character is created, added to the cache, and then returned.

##### Step 3: Using the Flyweight Pattern

Let's create some characters and see how the Flyweight Pattern works in practice.

```cpp
int main() {
    CharacterFactory factory;

    // Create characters
    std::shared_ptr<Character> charA1 = factory.getCharacter('A');
    std::shared_ptr<Character> charA2 = factory.getCharacter('A');
    std::shared_ptr<Character> charB = factory.getCharacter('B');

    // Display characters with different font sizes
    charA1->display(12);
    charA2->display(14);
    charB->display(16);

    // Check if charA1 and charA2 are the same object
    if (charA1 == charA2) {
        std::cout << "charA1 and charA2 are the same object" << std::endl;
    } else {
        std::cout << "charA1 and charA2 are different objects" << std::endl;
    }

    return 0;
}
```

In this example, we create two 'A' characters and one 'B' character using the `CharacterFactory`. When we request the 'A' character for the second time, the factory returns the existing 'A' character from the cache, demonstrating the sharing of intrinsic state. The `display` method is called with different font sizes to show how extrinsic state can be managed.

#### 19.2.4. Benefits of the Flyweight Pattern

1. **Reduced Memory Usage**: By sharing common state among multiple objects, the Flyweight Pattern significantly reduces the amount of memory needed.
2. **Efficiency**: This pattern is highly efficient in scenarios where a large number of fine-grained objects are required, but many share common state.
3. **Scalability**: The pattern allows for the creation of a large number of objects without a corresponding increase in memory usage.

#### 19.2.5. Potential Drawbacks

1. **Complexity**: The pattern introduces complexity by separating intrinsic and extrinsic states, which can make the code harder to understand and maintain.
2. **Runtime Overhead**: The pattern can introduce runtime overhead due to the need to manage and look up shared objects in the flyweight factory.

#### 19.2.6. Practical Applications

The Flyweight Pattern is particularly useful in the following scenarios:

1. **Text Editors**: Representing characters in a document efficiently by sharing common character objects.
2. **Graphics Systems**: Managing graphical objects like shapes or icons where many instances share common properties.
3. **Game Development**: Handling a large number of similar objects, such as tiles in a game map or units in a strategy game.

#### Conclusion

The Flyweight Pattern is a powerful tool for reducing memory usage by sharing common state among multiple objects. By leveraging a flyweight factory to manage shared objects, this pattern enables the creation of a large number of fine-grained objects efficiently. While it introduces some complexity and potential runtime overhead, the benefits in terms of memory savings and scalability often outweigh these drawbacks. Whether you're developing a text editor, a graphics system, or a game, the Flyweight Pattern can help you manage memory usage effectively and keep your applications running smoothly.

### 19.3. Bridge Pattern: Decoupling Abstraction from Implementation

The Bridge Pattern is a structural design pattern that decouples an abstraction from its implementation so that the two can vary independently. This pattern is particularly useful when both the abstraction and the implementation are likely to change. The Bridge Pattern achieves this by using composition over inheritance, promoting flexibility and scalability in your code architecture.

#### 19.3.1. The Problem

In complex systems, it is common to encounter situations where an abstraction has multiple implementations. For example, consider a graphic library where shapes need to be drawn on different platforms like Windows, Linux, and macOS. Directly coupling the shapes with the platform-specific rendering code would make the system rigid and hard to maintain. Every time a new shape or platform is added, significant changes would be required across the codebase.

#### 19.3.2. The Solution

The Bridge Pattern addresses this issue by separating the abstraction (in this case, the shape) from its implementation (the platform-specific rendering code). This separation is achieved by creating two hierarchies: one for the abstraction and another for the implementation. A bridge interface connects these hierarchies, allowing them to vary independently.

#### 19.3.3. Implementation

Let's walk through a detailed implementation of the Bridge Pattern in C++. We'll create a system to draw shapes on different platforms, demonstrating how the abstraction and implementation can be decoupled.

##### Step 1: Define the Implementation Interface

First, we define an interface for the implementation that declares the platform-specific operations.

```cpp
#include <iostream>
#include <memory>

// Implementation interface
class DrawingAPI {
public:
    virtual void drawCircle(double x, double y, double radius) const = 0;
    virtual ~DrawingAPI() = default;
};
```

The `DrawingAPI` interface declares a method to draw a circle, which will be implemented by platform-specific classes.

##### Step 2: Implement Concrete Implementations

Next, we implement the platform-specific classes that inherit from the `DrawingAPI` interface.

```cpp
// Concrete implementation for Windows
class WindowsAPI : public DrawingAPI {
public:
    void drawCircle(double x, double y, double radius) const override {
        std::cout << "Drawing circle on Windows at (" << x << ", " << y << ") with radius " << radius << std::endl;
    }
};

// Concrete implementation for Linux
class LinuxAPI : public DrawingAPI {
public:
    void drawCircle(double x, double y, double radius) const override {
        std::cout << "Drawing circle on Linux at (" << x << ", " << y << ") with radius " << radius << std::endl;
    }
};
```

These classes provide platform-specific implementations for drawing a circle.

##### Step 3: Define the Abstraction Interface

We then define an abstract class for shapes that uses the `DrawingAPI` interface to perform the actual drawing.

```cpp
// Abstraction interface
class Shape {
protected:
    std::shared_ptr<DrawingAPI> drawingAPI;

public:
    Shape(std::shared_ptr<DrawingAPI> drawingAPI) : drawingAPI(drawingAPI) {}

    virtual void draw() const = 0;
    virtual void resizeByPercentage(double percent) = 0;
    virtual ~Shape() = default;
};
```

The `Shape` class holds a reference to a `DrawingAPI` object and declares methods for drawing and resizing shapes.

##### Step 4: Implement Concrete Abstractions

Next, we implement concrete shapes that inherit from the `Shape` class.

```cpp
// Concrete abstraction for Circle
class CircleShape : public Shape {
private:
    double x, y, radius;

public:
    CircleShape(double x, double y, double radius, std::shared_ptr<DrawingAPI> drawingAPI)
        : Shape(drawingAPI), x(x), y(y), radius(radius) {}

    void draw() const override {
        drawingAPI->drawCircle(x, y, radius);
    }

    void resizeByPercentage(double percent) override {
        radius *= (1 + percent / 100);
    }
};
```

The `CircleShape` class uses the `DrawingAPI` to draw itself and provides a method to resize the circle.

##### Step 5: Using the Bridge Pattern

Let's create some shapes and draw them using different platform-specific implementations.

```cpp
int main() {
    // Create platform-specific drawing APIs
    std::shared_ptr<DrawingAPI> windowsAPI = std::make_shared<WindowsAPI>();
    std::shared_ptr<DrawingAPI> linuxAPI = std::make_shared<LinuxAPI>();

    // Create shapes with different implementations
    CircleShape circle1(1, 2, 3, windowsAPI);
    CircleShape circle2(4, 5, 6, linuxAPI);

    // Draw shapes
    circle1.draw();
    circle2.draw();

    // Resize and draw again
    circle1.resizeByPercentage(50);
    circle1.draw();

    return 0;
}
```

In this example, we create two `CircleShape` objects with different `DrawingAPI` implementations. The shapes can be drawn and resized independently of the platform-specific rendering code, demonstrating the flexibility provided by the Bridge Pattern.

#### 19.3.4. Benefits of the Bridge Pattern

1. **Decoupling**: The Bridge Pattern decouples the abstraction from its implementation, allowing them to vary independently. This leads to more flexible and maintainable code.
2. **Scalability**: New abstractions and implementations can be added without modifying existing code, adhering to the Open/Closed Principle.
3. **Code Reusability**: Common functionality can be shared across multiple implementations, promoting code reuse.

#### 19.3.5. Potential Drawbacks

1. **Complexity**: The pattern introduces additional layers of abstraction, which can make the code more complex and harder to understand.
2. **Performance Overhead**: The indirection introduced by the bridge can add some runtime overhead, although this is typically negligible compared to the benefits.

#### 19.3.6. Practical Applications

The Bridge Pattern is particularly useful in the following scenarios:

1. **Cross-Platform Applications**: Applications that need to run on multiple platforms with different implementations for the same functionality.
2. **Graphics Libraries**: Libraries that need to support various rendering engines or APIs.
3. **Device Drivers**: Systems that interact with different hardware devices, where the high-level functionality remains the same but the low-level implementation varies.

#### Conclusion

The Bridge Pattern is a powerful tool for decoupling abstraction from implementation, allowing them to evolve independently. By using composition over inheritance, this pattern promotes flexibility, scalability, and code reuse. Although it introduces some complexity, the benefits in terms of maintainability and extensibility often outweigh these drawbacks. Whether you are developing cross-platform applications, graphics libraries, or device drivers, the Bridge Pattern can help you manage the complexity and variability of your codebase effectively.

### 19.4. Proxy Pattern: Controlling Access

The Proxy Pattern is a structural design pattern that provides a surrogate or placeholder for another object to control access to it. This pattern is particularly useful in scenarios where direct access to an object is either not desired or not possible. The Proxy Pattern can help with lazy initialization, access control, logging, caching, and more. By introducing a proxy object, the pattern ensures that additional functionality can be added transparently to the real object.

#### 19.4.1. The Problem

Consider a situation where you have a resource-intensive object, such as a large image or a connection to a remote server. Creating and initializing this object might be costly in terms of time and memory. You may not want to load or initialize this object until it is absolutely necessary. Additionally, you may need to control access to the object to enforce security, perform logging, or manage caching.

#### 19.4.2. The Solution

The Proxy Pattern addresses this problem by creating a proxy object that acts as an intermediary between the client and the real object. The proxy controls access to the real object, ensuring that it is created or accessed only when necessary. The proxy can also add additional behavior such as logging, access control, and caching.

#### 19.4.3. Types of Proxies

1. **Virtual Proxy**: Delays the creation and initialization of an expensive object until it is actually needed.
2. **Protection Proxy**: Controls access to the original object based on access rights.
3. **Remote Proxy**: Represents an object located in a different address space.
4. **Caching Proxy**: Provides temporary storage of results to speed up subsequent requests.
5. **Logging Proxy**: Logs requests and responses to and from the real object.

#### 19.4.4. Implementation

Let's walk through a detailed implementation of the Proxy Pattern in C++. We'll create a system to manage access to a large image file, demonstrating how the proxy can control access and delay the loading of the image.

##### Step 1: Define the Subject Interface

First, we define an interface that both the real object and the proxy will implement.

```cpp
#include <iostream>
#include <memory>
#include <string>

// Subject interface
class Image {
public:
    virtual void display() const = 0;
    virtual ~Image() = default;
};
```

The `Image` interface declares a method for displaying the image.

##### Step 2: Implement the Real Subject

Next, we implement the real object that performs the actual work.

```cpp
// Real subject
class RealImage : public Image {
private:
    std::string filename;

    void loadImageFromDisk() const {
        std::cout << "Loading image from disk: " << filename << std::endl;
    }

public:
    RealImage(const std::string& filename) : filename(filename) {
        loadImageFromDisk();
    }

    void display() const override {
        std::cout << "Displaying image: " << filename << std::endl;
    }
};
```

The `RealImage` class represents a large image that is loaded from disk. The image is loaded when the `RealImage` object is created.

##### Step 3: Implement the Proxy

We then implement the proxy that controls access to the `RealImage` object.

```cpp
// Proxy
class ProxyImage : public Image {
private:
    std::string filename;
    mutable std::shared_ptr<RealImage> realImage; // Use mutable to allow lazy initialization in const methods

public:
    ProxyImage(const std::string& filename) : filename(filename), realImage(nullptr) {}

    void display() const override {
        if (!realImage) {
            realImage = std::make_shared<RealImage>(filename);
        }
        realImage->display();
    }
};
```

The `ProxyImage` class implements the `Image` interface and holds a reference to a `RealImage` object. The `display()` method checks if the `RealImage` object has been created. If not, it creates the `RealImage` object and then delegates the display call to it.

##### Step 4: Using the Proxy Pattern

Let's use the `ProxyImage` to control access to the `RealImage`.

```cpp
int main() {
    // Create a proxy for the image
    ProxyImage proxyImage("large_image.jpg");

    // Display the image
    std::cout << "First call to display():" << std::endl;
    proxyImage.display();

    std::cout << "\nSecond call to display():" << std::endl;
    proxyImage.display();

    return 0;
}
```

In this example, we create a `ProxyImage` object for a large image file. The first call to `display()` causes the `RealImage` to be loaded from disk and displayed. The second call to `display()` uses the already loaded `RealImage`, demonstrating the lazy initialization provided by the proxy.

#### 19.4.5. Benefits of the Proxy Pattern

1. **Lazy Initialization**: The proxy can delay the creation and initialization of the real object until it is actually needed.
2. **Access Control**: The proxy can control access to the real object based on access rights or other criteria.
3. **Logging**: The proxy can log requests and responses, providing a way to monitor and debug the interactions with the real object.
4. **Caching**: The proxy can cache the results of expensive operations to speed up subsequent requests.
5. **Remote Access**: The proxy can act as a local representative for an object located in a different address space, such as on a remote server.

#### 19.4.6. Potential Drawbacks

1. **Complexity**: The Proxy Pattern introduces additional layers of abstraction, which can make the code more complex and harder to understand.
2. **Overhead**: The proxy adds an extra level of indirection, which can introduce runtime overhead, although this is often negligible compared to the benefits.

#### 19.4.7. Practical Applications

The Proxy Pattern is particularly useful in the following scenarios:

1. **Resource-Intensive Objects**: Managing access to objects that are expensive to create or initialize, such as large images, complex calculations, or network connections.
2. **Access Control**: Implementing access control mechanisms where certain users or processes are allowed or denied access to specific objects.
3. **Remote Objects**: Providing a local representative for objects that exist in different address spaces or on remote servers, such as in distributed systems or remote method invocation (RMI).
4. **Caching**: Caching the results of expensive operations to improve performance, such as in database access or web service calls.
5. **Logging and Monitoring**: Adding logging and monitoring functionality to track interactions with the real object, useful for debugging and auditing.

#### Conclusion

The Proxy Pattern is a versatile and powerful tool for controlling access to objects in your system. By introducing a proxy, you can add additional functionality such as lazy initialization, access control, logging, and caching transparently. Although it introduces some complexity and potential overhead, the benefits in terms of flexibility, maintainability, and performance often outweigh these drawbacks. Whether you are dealing with resource-intensive objects, implementing access control, or providing remote access, the Proxy Pattern can help you manage and optimize access to your objects effectively.
