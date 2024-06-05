## Chapter 4: Advanced GUI with Qt Quick and QML 

Chapter 4 of your Qt programming course, "Advanced GUI with Qt Quick and QML," explores the dynamic capabilities of Qt Quick for developing modern and responsive GUIs. This section introduces QML, the language used with the Qt Quick framework, to build highly interactive user interfaces. We will cover the basics of QML, the integration of C++ with QML, the use of standard components, and the development of custom components.

Chapter 4 provides an in-depth look at advanced GUI development with Qt Quick and QML, focusing on both standard and custom component creation to foster a practical understanding of dynamic UI development. This structure ensures students not only learn the theory but also how to apply these techniques in real-world applications.


### 4.1: Introduction to Qt Quick

**Overview**
Qt Quick is a framework for creating ﬂuid, animated, and touch-friendly graphical user interfaces. It uses 
QML (Qt Modeling Language) for designing the UI, which is both declarative and JavaScript-based, 
making it easy to read and write.
- **QML Engine:** Manages QML components and facilitates the interaction with C++.
- **Qt Quick Components:** Provide ready-to-use UI elements.

**Key Concepts and Usage**

- **Declarative UI:** Focus on describing "what" the interface should contain, rather than "how" it is displayed. In QML, you describe the user interface in terms of its composition and design rather than through a step-by-step procedure for constructing the UI. This means you specify what components (like buttons, sliders, and text fields) the interface should contain and their relationships, but not the exact flow of control for how each element is created and displayed.
- **Performance:** Qt Quick is designed for high performance, leveraging hardware acceleration and scene graph based rendering. 
-   **Ease of Use**: This syntax simplifies UI development, making it more accessible to designers who might not have a deep programming background. It also enhances collaboration between designers and developers, as changes to the UI can be made quickly and with minimal code adjustments.

#### Performance: Qt Quick's Rendering and Hardware Acceleration

Qt Quick provides high performance for modern applications through its innovative approach to rendering and its use of hardware acceleration:

**Scene Graph-Based Rendering:**
-   **Scene Graph**: At the heart of Qt Quick's rendering engine is a scene graph, which is a data structure that arranges the graphical elements of the UI in a tree-like hierarchy. When a QML element changes, only the relevant parts of the scene graph need to be updated and redrawn, not the entire UI.
-   **Efficient Updates**: This selective updating minimizes the computational load and optimizes rendering performance, which is particularly beneficial for complex animations and dynamic content changes.

**Hardware Acceleration:**

-   **GPU Utilization**: Qt Quick leverages the underlying hardware by utilizing the Graphics Processing Unit (GPU) for rendering. This ensures that graphical operations are fast and efficient, offloading much of the rendering workload from the CPU.
-   **Smooth Animations and Transitions**: The use of the GPU helps in creating smooth animations and transitions in the UI, enhancing the user experience without sacrificing performance.

**Example:** Setting up a simple Qt Quick application.

```cpp
import QtQuick 2.15 
import QtQuick.Window 2.15 
 
Window { 
    visible: true 
    width: 360 
    height: 360 
    title: "Qt Quick Intro" 
 
    Text { 
        id: helloText 
        text: "Hello, Qt Quick!" 
        anchors.centerIn: parent 
        font.pointSize: 24 
    } 
} 
```

### 4.2: QML Basics

**Basic Elements**
QML uses a hierarchy of elements, with properties and behaviors, to create UIs.
- **Properties:** Deﬁne the characteristics of QML elements.
- **Signals and Handlers:** React to events like user interaction.

**Key Concepts and Usage**
- **Property Bindings:** Dynamically link properties to maintain consistency between values.
- **JavaScript Integration:** Use JavaScript for handling complex logic.

#### Property Bindings

Property bindings are one of the core concepts in Qt Quick, which allow properties of QML elements to be dynamically linked together. This mechanism ensures that when one property changes, any other properties that are bound to it are automatically updated to reflect the new value. This feature is crucial for maintaining consistency across the UI without needing to manually synchronize values.
-   **Automatic Updates**: When you use property bindings, changes to one property automatically propagate to any other properties that depend on it. This reduces the amount of code needed to update the UI in response to data changes.
-   **Declarative Syntax**: Property bindings are expressed declaratively. You specify the relationship between properties directly in the QML code, which makes the dependencies clear and the code easier to maintain.

**Example of Property Bindings:**
```cpp
import QtQuick 2.0

Rectangle {
    width: 200
    height: width / 2  // Binding height to be always half of width

    Text {
        text: "Width is " + parent.width  // Text updates automatically when rectangle's width changes
        anchors.centerIn: parent
    }
}
``` 

In this example, the `height` of the rectangle is bound to its `width`, ensuring that the height is always half the width, and the text displays the current width, updating automatically when the width changes.

#### JavaScript Integration

Qt Quick integrates JavaScript to handle complex logic that goes beyond simple property bindings. This allows for more sophisticated data processing, event handling, and interaction within QML applications, tapping into the full potential of a mature scripting language.
-   **Handling Complex Logic**: JavaScript can be used in QML to perform calculations, manipulate data, and handle complex decision-making processes within the UI.
-   **Event Handling**: JavaScript functions can be connected to signals in QML, providing a way to respond to user interactions or other events in a dynamic manner.

**Example of JavaScript Integration:**
```cpp
import QtQuick 2.0

Rectangle {
    id: rect
    width: 200; height: 200
    color: "blue"

    MouseArea {
        anchors.fill: parent
        onClicked: {
            rect.color = (rect.color === "blue" ? "green" : "blue")  
            // Toggle color between blue and green
        }
    }
}
```

In this example, a `MouseArea` is used to handle mouse clicks on the rectangle. The `onClicked` handler contains JavaScript code that toggles the rectangle's color between blue and green.

**Example:** Creating a dynamic UI with property bindings.

```cpp
import QtQuick 2.15 
 
Rectangle { 
    width: 200; height: 200 
    color: "blue" 
 
    Text { 
        text: "Click me!" 
        anchors.centerIn: parent 
        MouseArea { 
            anchors.fill: parent 
            onClicked: parent.color = "red" 
        } 

    } 
} 
```

```cpp
#include <QGuiApplication>
#include <QQmlApplicationEngine>

int main(int argc, char *argv[])
{
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QGuiApplication app(argc, argv);

    QQmlApplicationEngine engine;
    const QUrl url(QStringLiteral("qrc:/main.qml"));
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
                     &app, [url](QObject *obj, const QUrl &objUrl) {
        if (!obj && url == objUrl)
            QCoreApplication::exit(-1);
    }, Qt::QueuedConnection);
    engine.load(url);

    return app.exec();
}
```
### 4.3: Integrating C++ with QML

**Overview**
Combining QML with C++ enables leveraging the power of C++ for backend logic while using QML for the frontend.
- **Exposing C++ Objects to QML:** Use `QQmlContext` to make C++ objects available in QML. 
- **Calling C++ Functions from QML:** Enhance interaction capabilities.

**Key Concepts and Usage**
- **Registering Types:** Use `qmlRegisterType()` to make custom C++ types available in QML.

#### 1. Exposing C++ Objects to QML

To make C++ objects available in QML, you can use the `QQmlContext` class. This class provides the environment in which QML expressions are evaluated. By setting properties on the `QQmlContext`, you can expose C++ objects to QML, allowing QML code to interact with them.
**Example:**
Here's how you might expose a C++ object to QML:

```cpp
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QDebug>

class MyObject : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString message READ message WRITE setMessage NOTIFY messageChanged)
public:
    explicit MyObject(QObject *parent = nullptr) : QObject(parent), m_message("Hello from C++!") {}

    QString message() const { return m_message; }
    void setMessage(const QString &message) {
        if (m_message != message) {
            m_message = message;
            emit messageChanged();
        }
    }

signals:
    void messageChanged();

private:
    QString m_message;
};

int main(int argc, char *argv[]) {
    QGuiApplication app(argc, argv);

    QQmlApplicationEngine engine;
    MyObject myObject;

    // Expose the MyObject instance to QML
    engine.rootContext()->setContextProperty("myObject", &myObject);

    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));

    return app.exec();
}
``` 

In your QML file, you can now access `myObject`:
``` cpp
import QtQuick 2.0
import QtQuick.Controls 2.0

ApplicationWindow {
    visible: true
    width: 400
    height: 300
    title: qsTr("Hello World")

    Text {
        text: myObject.message
        anchors.centerIn: parent
    }
}
```

#### 2. Calling C++ Functions from QML

You can enhance the interaction capabilities of your QML UI by calling C++ functions directly from QML. This is typically done by exposing C++ methods as public slots or Q_INVOKABLE functions.

#### Example:

Modify the `MyObject` class to include a callable function:
```cpp
class MyObject : public QObject {
    Q_OBJECT
public:
    Q_INVOKABLE void updateMessage(const QString &newMessage) {
        setMessage(newMessage);
    }
    // Existing code...
};
```
Now, you can call this method from QML:
```cpp
Button {
    text: "Update Message"
    onClicked: myObject.updateMessage("Updated from QML!")
}
``` 

#### 3. Registering Custom C++ Types in QML

To use custom C++ types as QML types, you can register them using `qmlRegisterType()`. This allows you to instantiate your C++ classes as QML objects.
**Example:**
Here's how you might register a custom type and use it directly in QML:
```cpp
#include <QtQuick>

class MyCustomType : public QObject {
    Q_OBJECT
public:
    MyCustomType() {}
};

int main(int argc, char *argv[]) {
    QGuiApplication app(argc, argv);

    qmlRegisterType<MyCustomType>("com.mycompany", 1, 0, "MyCustomType");

    QQmlApplicationEngine engine;
    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    return app.exec();
}
``` 
In your QML file, you can instantiate `MyCustomType`:
```cpp
import QtQuick 2.0
import com.mycompany 1.0

MyCustomType {
    // Properties and methods of MyCustomType can be used here
}
```

**Example:** Exposing a C++ class to QML.

```cpp
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext> 
 
class Backend : public QObject { 
    Q_OBJECT 
    Q_PROPERTY(QString userName READ userName WRITE setUserName NOTIFY userNameChanged)
public: 
    QString userName() const { return m_userName; } 
    void setUserName(const QString &userName) { 
        if (m_userName == userName) 
            return; 
        m_userName = userName; 
        emit userNameChanged(); 
    } 
signals: 
    void userNameChanged(); 
private: 
    QString m_userName; 
}; 
 
int main(int argc, char *argv[]) { 
    QGuiApplication app(argc, argv); 
    QQmlApplicationEngine engine; 
 
    Backend backend; 
    engine.rootContext()->setContextProperty("backend", &backend); 
 
    const QUrl url(QStringLiteral("qrc:/main.qml")); 
    engine.load(url); 
 
    return app.exec(); 
} 
```
### 4.4: Using Standard Components

QML's standard components cover a wide range of UI elements:
-   **Interactive Elements**: Such as `Button`, `CheckBox`, `RadioButton`, and `Slider`, which allow users to perform actions and make selections.
-   **Input Fields**: Including `TextField`, `TextArea`, which are used for entering and editing text.
-   **Display Containers**: Such as `ListView`, `GridView`, and `StackView`, which organize content in various layouts.

Among these, `ListView`, `GridView`, and `Repeater` are particularly important for handling dynamic data sets.

**Model-View-Delegate Architecture**
This architecture is a cornerstone of QML's approach to displaying collections of data:
-   **Model**: This is the data source that holds the data items to be displayed. In QML, models can be simple list models provided in QML itself, or more complex models backed by C++.
-   **View**: The view presents the data from the model to the user. The view itself does not contain logic for item layout or how the data should be formatted; it merely uses the delegate to create a visual representation of each item. `ListView` and `GridView` are examples of views.
-   **Delegate**: This is a template for creating items in the view. Each item in the view is instantiated from the delegate, which defines how each data item should be displayed.

#### Example: Using ListView with a Model and Delegate

Let's consider an example where we use a `ListView` to display a list of names. We'll use a simple ListModel as our data source, and a Component to define how each item should look.

```cpp
import QtQuick 2.15
import QtQuick.Controls 2.15

ApplicationWindow {
    visible: true
    width: 400
    height: 300
    title: "ListView Example"

    ListView {
        width: 200; height: 250
        anchors.centerIn: parent

        model: ListModel {
            ListElement { name: "Alice" }
            ListElement { name: "Bob" }
            ListElement { name: "Carol" }
            ListElement { name: "Dave" }
        }

        delegate: Rectangle {
            width: 180; height: 40
            color: "lightblue"
            radius: 5
            margin: 5
            Text {
                text: name
                anchors.verticalCenter: parent.verticalCenter
                anchors.left: parent.left
                anchors.leftMargin: 10
            }
        }
    }
}
``` 

**How It Works:**
-   **ListView**: The `ListView` is set up with a fixed size and centered in the application window.
-   **Model**: The `ListModel` contains several `ListElement` items, each with a `name` property.
-   **Delegate**: The delegate is a `Rectangle` that represents each item. Inside the rectangle, a `Text` element displays the name. The delegate is styled with a light blue background and rounded corners.


### 4.5: Developing Custom Components

**Custom Components**
Creating custom components in QML allows for reusable and encapsulated UI functionality.
Creating a Custom Component: Typically involves deﬁning a new QML ﬁle.

**Key Concepts and Usage**
**Reusability:** Design components to be reusable across diﬀerent parts of applications.

**Example:** Creating a custom button component.

```cpp
// Save as MyButton.qml import QtQuick 2.15 
 
Rectangle { 
    width: 100; height: 40 
    color: "green" 
    radius: 5 
 
    Text { 
        text: "Button" 
        anchors.centerIn: parent 
        color: "white" 
    } 
 
    MouseArea { 
        anchors.fill: parent 
        onClicked: console.log("Button clicked") 
    } 
} 
```
Usage:

```cpp
import QtQuick 2.15 
import QtQuick.Controls 2.15 
 
ApplicationWindow { 
    visible: true 
    width: 400 
    height: 300 
 
    MyButton { 
        anchors.centerIn: parent 
    } 
} 
```
