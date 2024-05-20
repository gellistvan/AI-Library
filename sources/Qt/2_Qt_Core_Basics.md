## Chapter 2: Qt Core Basics
This chapter will guide students through essential classes, the object model, event management, memory management, and ﬁle handling. Here’s a detailed breakdown of the elements for Chapter 2, complete with examples and key usage details.
### 2.1: Qt Core Classes

#### Core Classes Overview
Qt provides a wide range of core classes that handle non-GUI tasks necessary for application development. These classes include data handling, ﬁle I/O, threading, and more. Each class is designed to oﬀer ﬂexibility and robust functionality to the Qt framework.
#### QString
Manages immutable Unicode character strings and provides numerous functions for string manipulation, such as slicing, concatenation, conversion, and comparison.

**Key Properties and Usage:**

* `length()`: Returns the number of characters in the string.
* `isEmpty()`: Checks if the string is empty.
* `toInt()`, `toDouble()`: Converts the string to integers or ﬂoating-point numbers.
* `split()`: Divides the string into substrings based on a delimiter.

**Example:** More complex operations with `QString`:

```cpp
QString s = "Hello, world!"; 
qDebug() << "Length:" << s.length(); 
qDebug() << "Empty?" << s.isEmpty(); 
QStringList parts = s.split(','); 
qDebug() << "Split result:" << parts.at(0).trimmed();  // Output: "Hello"

QString s = "Temperature"; 
double temp = 24.5; 
QString text = s + ": " + QString::number(temp); 
```
#### QVariant
Holds a single value of a variety of data types.

**Key Properties and Usage:**

* `isValid()`: Determines if the variant contains a valid data.
* `canConvert<T>()`: Checks whether the stored value can be converted to a speciﬁed type.

**Example**: Using `QVariant` to store diﬀerent types and retrieve values:

```cpp
QVariant v(42);  // Stores an integer
qDebug() << "Is string?" << v.canConvert<QString>(); 
qDebug() << "As string:" << v.toString(); 
v.setValue("Hello Qt"); 
qDebug() << "Now a string?" << v.canConvert<QString>(); 
```### 2.2: Object Model (QObject and QCoreApplication)

`QObject` is the base class of all Qt objects and the core of the object model. It enables object communication via signals and slots, dynamic property system, and more.

**Key Properties and Usage:**

* `setParent()`, `parent()`: Manages object hierarchy and ownership.
* `setProperty()`, `property()`: Accesses properties deﬁned via the Q_PROPERTY macro.

`QCoreApplication` manages the application's control ﬂow and main settings, and it's necessary for applications that do not use the GUI.

**Example 1**: Deeper use of `QObject` and `QCoreApplication`:

```cpp
#include <QCoreApplication>
#include <QObject> 
 
class Application : public QObject { 
    Q_OBJECT 
public: 
    Application(QObject *parent = nullptr) : QObject(parent) { 
        setProperty("Version", "1.0"); 
    } 
    void printVersion() { 

        qDebug() << "Application version:" << property("Version").toString(); 
    } 
}; 
 
int main(int argc, char *argv[]) { 
    QCoreApplication app(argc, argv); 
    Application myApp; 
    myApp.printVersion(); 
    return app.exec(); 
} 
```

**Example 2**: Creating a simple application with `QCoreApplication` and connecting a `QObject`'s signal to a
slot.

```cpp
#include <QCoreApplication>
#include <QObject> 
 
class MyObject : public QObject { 
    Q_OBJECT 
public: 
    MyObject(QObject *parent = nullptr) : QObject(parent) {} 
signals: 
    void mySignal(); 
public slots: 
    void mySlot() { qDebug() << "Slot called"; } 
}; 
 
int main(int argc, char *argv[]) { 
    QCoreApplication app(argc, argv); 
    MyObject obj; 
    QObject::connect(&obj, &MyObject::mySignal, &obj, &MyObject::mySlot); 
    emit obj.mySignal(); 
    return app.exec(); 
}
```

### 2.3: Event Loop and Signal & Slots

The event loop is a core part of Qt applications, handling all events from the window system and other sources. Signals and slots are used for communication between objects and can cross thread boundaries.

**Key Concepts:**

* `QEventLoop`: Manages a loop that processes events.
* `signals`: Methods that are declared but not deﬁned.
* `slots`: Methods that can be called in response to signals.

#### QEventLoop
The `QEventLoop` is a crucial part of the Qt framework that manages an event loop within a Qt application. An event loop is a programming construct that waits for and dispatches events or messages in a program. It's an integral part of any Qt application because it handles all events from the window system and other sources, such as timers and network sockets.
In a typical Qt application, the event loop is started using the `exec()` method of a `QEventLoop` or `QApplication` object. This loop continues running until `exit()` is called on the respective object, processing incoming events and ensuring that your application remains responsive. The `QEventLoop` can be used to create local event loops in specific parts of your application, which can be useful for handling tasks that require a focused, uninterrupted sequence of operations while still processing events.

#### Signals and Slots
Signals and slots are a mechanism in Qt used for communication between objects and are one of the key features of Qt's event-driven architecture. They make it easy to implement the Observer pattern while avoiding boilerplate code.

**Signals**
In Qt, a signal is a method that is declared but not defined by the programmer. Instead, it is automatically generated by the Meta-Object Compiler (MOC). Signals are used to announce that a specific event has occurred. For example, a button widget might emit a `clicked()` signal when it is pressed. Here's an example of declaring a signal in a class:
```cpp
class MyClass : public QObject {
	Q_OBJECT
public:
	MyClass() {}
signals:
	void mySignal();
};
```
In the above example, `mySignal()` is a signal. You do not provide a definition for this method—the MOC takes care of that.

**Slots**
A slot is a normal C++ method that can be connected to one or more signals. When a signal connected to a slot is emitted, the slot is automatically invoked by the Qt framework. This allows for a very flexible communication mechanism between different parts of your application. Here's how you might define a class with a slot:

```cpp
class MyClass : public QObject {
	Q_OBJECT
public slots:
	void mySlot() {
		// Your code here
	}
};
```

To connect a signal to a slot, Qt provides the `connect()` method, which can link signals from any object to slots of any other (or the same) object. For example:

```cpp
MyClass obj;
QPushButton button;
QObject::connect(&button, &QPushButton::clicked, &obj, &MyClass::mySlot);
```
In this example, clicking the button would automatically call `mySlot()` on `obj`.


**Example 1**: A simple timer that uses signals and slots.
```cpp
#include <QTimer>
#include <QCoreApplication>
#include <QDebug> 
 
class Timer : public QObject { 
    Q_OBJECT 
public slots: 
    void handleTimeout() { qDebug() << "Timeout occurred"; } 
}; 
 
int main(int argc, char *argv[]) { 
    QCoreApplication app(argc, argv); 
    QTimer timer; 
    Timer t; 
    QObject::connect(&timer, &QTimer::timeout, &t, &Timer::handleTimeout); 
    timer.start(1000);  // 1000 milliseconds 
    return app.exec(); 
}
```

**Example 2**: Advanced signal-slot connection:

```cpp
#include <QObject>
#include <QTimer>
#include <QDebug> 
 
class Worker : public QObject { 
    Q_OBJECT 
public slots: 
    void process() { 
        qDebug() << "Processing..."; 
        emit finished(); 
    } 
signals: 
    void finished(); 
}; 
 
int main(int argc, char *argv[]) { 
    QCoreApplication app(argc, argv); 
    QTimer timer; 
    Worker worker; 
    QObject::connect(&timer, &QTimer::timeout, &worker, &Worker::process); 
    QObject::connect(&worker, &Worker::finished, &app, &QCoreApplication::quit); 
    timer.start(1000); 
    return app.exec(); 
} 
```
### 2.4: Memory Management


Qt's approach to memory management, which centers on the parent-child hierarchy among QObject instances, is indeed crucial for effective resource management in Qt applications.

#### Parent-Child Relationship in QObject
In Qt, memory management of objects (especially QObject derived objects) can be simplified using parent-child relationships. When a QObject is created, you can specify another QObject as its parent. The parent takes responsibility for deleting its children when it itself is deleted. This is an essential feature for avoiding memory leaks, especially in large applications with complex UIs.

Here’s what happens in a parent-child relationship:
* **Ownership and Deletion**: When you assign a parent to a QObject, the parent will automatically delete its children in its destructor. This means that you don’t need to explicitly delete the child objects; they will be cleaned up when the parent is.
* **Hierarchy**: This relationship also defines an object hierarchy or a tree structure, which is useful for organizing objects in, for example, a graphical user interface.

#### Using new and delete
In C++, new and delete are used for direct memory management:

* **new**: This operator allocates memory on the heap for an object and returns a pointer to it. When you use new, you are responsible for manually managing the allocated memory and must eventually release it using delete.
* **delete**: This operator deallocates memory and calls the destructor of the object.

#### Automatic Deletion through Parent-Child Hierarchy
Qt enhances C++ memory management with the parent-child mechanism, which provides automatic memory management:
* **Automatic Deletion**: When you create a QObject with a parent, you typically use new to allocate it, but you do not need to explicitly delete it. The parent QObject will automatically call delete on it once the parent itself is being destroyed.
* **Safety in UI Components**: This is particularly useful in UI applications where widgets often have nested child widgets. By setting the parent-child relationship appropriately, you can ensure that all child widgets are deleted when the parent widget is closed, thus preventing memory leaks.
  Here's a simple example to illustrate this:

```cpp
#include <QWidget>
#include <QPushButton>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    QWidget *window = new QWidget;
    QPushButton *button = new QPushButton("Click me", window);
    // The button now has 'window' as its parent.

    window->show();
    return app.exec();
}
```
In the example above, the QPushButton is created with window as its parent. When window is closed and deleted, it will automatically delete the button as well, without any need for explicit cleanup.

**Example**: Demonstrating parent-based deletion:
```cpp
QObject *parent = new QObject; 
QObject *child1 = new QObject(parent); 
QObject *child2 = new QObject(parent); 

// child1 and child2 will be deleted when parent is deleted
qDebug() << "Children count:" << parent->children().count();  // Output: 2
delete parent; 
```
### 2.5: File Handling

`QFile` and `QDir` are crucial for handling ﬁle input/output operations and directory management.
* QFile: Manages ﬁle I/O.
* QDir: Manages directory and path information.

**Example**: Reading and writing ﬁles with error handling:

```cpp
QFile file("data.txt"); 
if (!file.open(QIODevice::ReadWrite)) { 
    qWarning("Cannot open file for reading"); 
} else { 
    QTextStream in(&file); 
    while (!in.atEnd()) { 
        QString line = in.readLine(); 
        qDebug() << line; 
    } 
    QTextStream out(&file); 
    out << "New line of text\n"; 
    file.close(); 
} 
```

```cpp
QFile file("example.txt"); 
if (file.open(QIODevice::ReadWrite)) { 
    QTextStream stream(&file); 
    stream << "Hello, Qt!"; 
    file.flush();  // Write changes to disk 
    file.seek(0);  // Go back to the start of the file 
    QString content = stream.readAll(); 
    qDebug() << content; 
    file.close(); 
} 
```