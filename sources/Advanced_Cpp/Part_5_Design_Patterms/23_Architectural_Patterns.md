
\newpage
## Chapter 23: Architectural Patterns

In the rapidly evolving landscape of software development, architectural patterns play a crucial role in shaping the design and structure of applications. This chapter delves into advanced architectural patterns that enable developers to create scalable, maintainable, and flexible systems. We will explore the Microservices Architecture, which breaks down applications into smaller, manageable services to enhance scalability and resilience. Next, we will examine the Service-Oriented Architecture (SOA), focusing on the creation of reusable and interoperable services. The chapter will also cover Event-Driven Architecture, emphasizing asynchronous communication for responsive and decoupled systems. Finally, we will discuss Hexagonal Architecture, also known as Ports and Adapters, which promotes a clean separation of concerns, facilitating easier testing and maintenance. Through these architectural patterns, we aim to equip you with the knowledge to design robust and adaptable software solutions.

### 23.1. Microservices Architecture: Building Scalable Applications

Microservices architecture has emerged as a powerful approach to building scalable, maintainable, and resilient applications. By decomposing a monolithic application into a collection of loosely coupled services, each responsible for a specific business capability, developers can achieve greater flexibility and ease of management. This subchapter will explore the fundamental concepts, benefits, and challenges of microservices architecture, accompanied by detailed code examples to illustrate key principles.

#### 23.1.1. Introduction to Microservices

Microservices architecture is characterized by the following key principles:
1. **Single Responsibility**: Each microservice is responsible for a specific business function.
2. **Loose Coupling**: Services are designed to minimize dependencies on each other.
3. **Independent Deployment**: Each service can be deployed independently without affecting other services.
4. **Technology Diversity**: Different services can use different technologies best suited for their requirements.

#### 23.1.2. Benefits of Microservices

Microservices offer several advantages over traditional monolithic architectures:
- **Scalability**: Individual services can be scaled independently based on demand.
- **Resilience**: Failures in one service do not necessarily impact others.
- **Flexibility**: Teams can choose the most appropriate technology stack for each service.
- **Faster Development**: Smaller, focused teams can develop, test, and deploy services independently.

#### 23.1.3. Challenges of Microservices

Despite the benefits, microservices also introduce challenges:
- **Complexity**: Managing multiple services, each with its own lifecycle, increases complexity.
- **Data Management**: Ensuring data consistency across services can be difficult.
- **Deployment**: Orchestrating the deployment of numerous services requires sophisticated tooling.

#### 23.1.4. Implementing Microservices in C++

Let's explore a practical example of implementing microservices using C++. We will develop a simple e-commerce application with three microservices: `ProductService`, `OrderService`, and `UserService`.

##### 23.1.4.1. Setting Up the Environment

To manage our microservices, we will use Docker for containerization and Kubernetes for orchestration. Ensure you have both Docker and Kubernetes installed on your system.

##### 23.1.4.2. ProductService

**ProductService** handles product-related operations, such as listing products and retrieving product details.

```cpp
// ProductService.h
#ifndef PRODUCT_SERVICE_H
#define PRODUCT_SERVICE_H

#include <string>
#include <vector>

struct Product {
    int id;
    std::string name;
    double price;
};

class ProductService {
public:
    std::vector<Product> listProducts();
    Product getProduct(int productId);
};

#endif // PRODUCT_SERVICE_H
```

```cpp
// ProductService.cpp
#include "ProductService.h"

std::vector<Product> ProductService::listProducts() {
    return {
        {1, "Laptop", 999.99},
        {2, "Smartphone", 499.99},
        {3, "Tablet", 299.99}
    };
}

Product ProductService::getProduct(int productId) {
    auto products = listProducts();
    for (const auto& product : products) {
        if (product.id == productId) {
            return product;
        }
    }
    throw std::runtime_error("Product not found");
}
```

##### 23.1.4.3. OrderService

**OrderService** manages orders placed by users.

```cpp
// OrderService.h
#ifndef ORDER_SERVICE_H
#define ORDER_SERVICE_H

#include <string>
#include <vector>

struct Order {
    int id;
    int productId;
    int userId;
    std::string status;
};

class OrderService {
public:
    std::vector<Order> listOrders();
    void createOrder(int productId, int userId);
};

#endif // ORDER_SERVICE_H
```

```cpp
// OrderService.cpp
#include "OrderService.h"

std::vector<Order> OrderService::listOrders() {
    return {
        {1, 1, 1, "Pending"},
        {2, 2, 2, "Shipped"}
    };
}

void OrderService::createOrder(int productId, int userId) {
    // In a real application, this would persist the order to a database.
    std::cout << "Order created: Product ID " << productId << ", User ID " << userId << std::endl;
}
```

##### 23.1.4.4. UserService

**UserService** handles user-related operations.

```cpp
// UserService.h
#ifndef USER_SERVICE_H
#define USER_SERVICE_H

#include <string>
#include <vector>

struct User {
    int id;
    std::string name;
};

class UserService {
public:
    std::vector<User> listUsers();
    User getUser(int userId);
};

#endif // USER_SERVICE_H
```

```cpp
// UserService.cpp
#include "UserService.h"

std::vector<User> UserService::listUsers() {
    return {
        {1, "Alice"},
        {2, "Bob"}
    };
}

User UserService::getUser(int userId) {
    auto users = listUsers();
    for (const auto& user : users) {
        if (user.id == userId) {
            return user;
        }
    }
    throw std::runtime_error("User not found");
}
```

#### 23.1.5. Containerizing Microservices with Docker

We will create Dockerfiles for each service to containerize them.

##### 23.1.5.1. Dockerfile for ProductService

```dockerfile
# Dockerfile for ProductService
FROM gcc:latest

WORKDIR /app
COPY . .

RUN g++ -o ProductService ProductService.cpp
CMD ["./ProductService"]
```

##### 23.1.5.2. Dockerfile for OrderService

```dockerfile
# Dockerfile for OrderService
FROM gcc:latest

WORKDIR /app
COPY . .

RUN g++ -o OrderService OrderService.cpp
CMD ["./OrderService"]
```

##### 23.1.5.3. Dockerfile for UserService

```dockerfile
# Dockerfile for UserService
FROM gcc:latest

WORKDIR /app
COPY . .

RUN g++ -o UserService UserService.cpp
CMD ["./UserService"]
```

#### 23.1.6. Orchestrating with Kubernetes

Next, we will create Kubernetes manifests to deploy our microservices.

##### 23.1.6.1. Deployment for ProductService

```yaml
# product-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: product-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: product-service
  template:
    metadata:
      labels:
        app: product-service
    spec:
      containers:
      - name: product-service
        image: product-service:latest
        ports:
        - containerPort: 8080
```

##### 23.1.6.2. Deployment for OrderService

```yaml
# order-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: order-service
  template:
    metadata:
      labels:
        app: order-service
    spec:
      containers:
      - name: order-service
        image: order-service:latest
        ports:
        - containerPort: 8081
```

##### 23.1.6.3. Deployment for UserService

```yaml
# user-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: user-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: user-service
  template:
    metadata:
      labels:
        app: user-service
    spec:
      containers:
      - name: user-service
        image: user-service:latest
        ports:
        - containerPort: 8082
```

#### Conclusion

Microservices architecture offers significant advantages in terms of scalability, flexibility, and maintainability. However, it also introduces new challenges that require careful consideration and sophisticated tooling. By following the principles and practices outlined in this subchapter, you can begin to harness the power of microservices to build robust and scalable applications in C++. With containerization and orchestration tools like Docker and Kubernetes, managing and deploying microservices becomes more manageable, enabling you to focus on delivering value to your users.

### 23.2. Service-Oriented Architecture (SOA): Designing Reusable Services

Service-Oriented Architecture (SOA) is an architectural pattern that focuses on designing and developing reusable services. Unlike microservices, which emphasize independent deployment and fine-grained services, SOA typically involves more coarse-grained services that are designed to be reused across different applications and business processes. This subchapter will delve into the core concepts, benefits, and challenges of SOA, with comprehensive code examples to illustrate key principles.

#### 23.2.1. Introduction to SOA

SOA is based on the following key principles:
1. **Service Abstraction**: Services are designed with well-defined interfaces that hide the underlying implementation details.
2. **Service Reusability**: Services are designed to be reused across different contexts and applications.
3. **Service Loose Coupling**: Services interact with each other in a loosely coupled manner, typically through well-defined interfaces.
4. **Service Contract**: Services communicate based on a contract that defines the inputs, outputs, and behavior.
5. **Service Composability**: Services can be composed to form more complex services or applications.

#### 23.2.2. Benefits of SOA

SOA offers several advantages:
- **Reusability**: Services can be reused across different applications, reducing redundancy.
- **Interoperability**: Services can interact with each other irrespective of the underlying technology, promoting interoperability.
- **Scalability**: Services can be scaled independently based on their usage patterns.
- **Maintainability**: Modifying a service does not impact other services, making maintenance easier.

#### 23.2.3. Challenges of SOA

Implementing SOA comes with its own set of challenges:
- **Complexity**: Designing reusable services requires careful planning and a deep understanding of business processes.
- **Performance**: Service interactions, especially over a network, can introduce latency.
- **Governance**: Managing and enforcing standards across services can be difficult.

#### 23.2.4. Implementing SOA in C++

Let's explore a practical example of implementing SOA using C++. We will develop a basic system with three services: `CustomerService`, `ProductService`, and `OrderService`.

##### 23.2.4.1. Defining the Service Interfaces

We will start by defining interfaces for our services. These interfaces will serve as contracts for service communication.

```cpp
// ICustomerService.h
#ifndef I_CUSTOMER_SERVICE_H
#define I_CUSTOMER_SERVICE_H

#include <string>
#include <vector>

struct Customer {
    int id;
    std::string name;
};

class ICustomerService {
public:
    virtual ~ICustomerService() = default;
    virtual Customer getCustomer(int customerId) = 0;
    virtual std::vector<Customer> listCustomers() = 0;
};

#endif // I_CUSTOMER_SERVICE_H
```

```cpp
// IProductService.h
#ifndef I_PRODUCT_SERVICE_H
#define I_PRODUCT_SERVICE_H

#include <string>
#include <vector>

struct Product {
    int id;
    std::string name;
    double price;
};

class IProductService {
public:
    virtual ~IProductService() = default;
    virtual Product getProduct(int productId) = 0;
    virtual std::vector<Product> listProducts() = 0;
};

#endif // I_PRODUCT_SERVICE_H
```

```cpp
// IOrderService.h
#ifndef I_ORDER_SERVICE_H
#define I_ORDER_SERVICE_H

#include <string>
#include <vector>

struct Order {
    int id;
    int customerId;
    int productId;
    std::string status;
};

class IOrderService {
public:
    virtual ~IOrderService() = default;
    virtual Order getOrder(int orderId) = 0;
    virtual void createOrder(int customerId, int productId) = 0;
    virtual std::vector<Order> listOrders() = 0;
};

#endif // I_ORDER_SERVICE_H
```

##### 23.2.4.2. Implementing the Services

Next, we will implement the services based on the defined interfaces.

```cpp
// CustomerService.h
#ifndef CUSTOMER_SERVICE_H
#define CUSTOMER_SERVICE_H

#include "ICustomerService.h"
#include <vector>

class CustomerService : public ICustomerService {
public:
    Customer getCustomer(int customerId) override;
    std::vector<Customer> listCustomers() override;
};

#endif // CUSTOMER_SERVICE_H
```

```cpp
// CustomerService.cpp
#include "CustomerService.h"

std::vector<Customer> CustomerService::listCustomers() {
    return {
        {1, "Alice"},
        {2, "Bob"}
    };
}

Customer CustomerService::getCustomer(int customerId) {
    auto customers = listCustomers();
    for (const auto& customer : customers) {
        if (customer.id == customerId) {
            return customer;
        }
    }
    throw std::runtime_error("Customer not found");
}
```

```cpp
// ProductService.h
#ifndef PRODUCT_SERVICE_H
#define PRODUCT_SERVICE_H

#include "IProductService.h"
#include <vector>

class ProductService : public IProductService {
public:
    Product getProduct(int productId) override;
    std::vector<Product> listProducts() override;
};

#endif // PRODUCT_SERVICE_H
```

```cpp
// ProductService.cpp
#include "ProductService.h"

std::vector<Product> ProductService::listProducts() {
    return {
        {1, "Laptop", 999.99},
        {2, "Smartphone", 499.99},
        {3, "Tablet", 299.99}
    };
}

Product ProductService::getProduct(int productId) {
    auto products = listProducts();
    for (const auto& product : products) {
        if (product.id == productId) {
            return product;
        }
    }
    throw std::runtime_error("Product not found");
}
```

```cpp
// OrderService.h
#ifndef ORDER_SERVICE_H
#define ORDER_SERVICE_H

#include "IOrderService.h"
#include <vector>

class OrderService : public IOrderService {
public:
    Order getOrder(int orderId) override;
    void createOrder(int customerId, int productId) override;
    std::vector<Order> listOrders() override;
};

#endif // ORDER_SERVICE_H
```

```cpp
// OrderService.cpp
#include "OrderService.h"
#include <iostream>

std::vector<Order> OrderService::listOrders() {
    return {
        {1, 1, 1, "Pending"},
        {2, 2, 2, "Shipped"}
    };
}

Order OrderService::getOrder(int orderId) {
    auto orders = listOrders();
    for (const auto& order : orders) {
        if (order.id == orderId) {
            return order;
        }
    }
    throw std::runtime_error("Order not found");
}

void OrderService::createOrder(int customerId, int productId) {
    // In a real application, this would persist the order to a database.
    std::cout << "Order created: Customer ID " << customerId << ", Product ID " << productId << std::endl;
}
```

#### 23.2.5. Service Communication

In SOA, services typically communicate through a common messaging protocol, such as HTTP or SOAP. For simplicity, we will use RESTful HTTP communication in our examples.

##### 23.2.5.1. RESTful Communication

We will use a lightweight HTTP server library to expose our services as RESTful endpoints. For C++, one popular choice is the `cpp-httplib` library.

**Installing cpp-httplib**

First, download the library from its [GitHub repository](https://github.com/yhirose/cpp-httplib).

**Exposing CustomerService**

```cpp
// main.cpp
#include "CustomerService.h"
#include "httplib.h"

int main() {
    CustomerService customerService;
    httplib::Server svr;

    svr.Get("/customers", [&customerService](const httplib::Request& req, httplib::Response& res) {
        auto customers = customerService.listCustomers();
        std::string response;
        for (const auto& customer : customers) {
            response += "Customer ID: " + std::to_string(customer.id) + ", Name: " + customer.name + "\n";
        }
        res.set_content(response, "text/plain");
    });

    svr.Get(R"(/customers/(\d+))", [&customerService](const httplib::Request& req, httplib::Response& res) {
        int customerId = std::stoi(req.matches[1]);
        try {
            auto customer = customerService.getCustomer(customerId);
            std::string response = "Customer ID: " + std::to_string(customer.id) + ", Name: " + customer.name + "\n";
            res.set_content(response, "text/plain");
        } catch (const std::exception& e) {
            res.status = 404;
            res.set_content("Customer not found", "text/plain");
        }
    });

    svr.listen("0.0.0.0", 8080);
    return 0;
}
```

**Exposing ProductService**

```cpp
// main.cpp
#include "ProductService.h"
#include "httplib.h"

int main() {
    ProductService productService;
    httplib::Server svr;

    svr.Get("/products", [&productService](const httplib::Request& req, httplib::Response& res) {
        auto products = productService.listProducts();
        std::string response;
        for (const auto& product : products) {
            response += "Product ID: " + std::to_string(product.id) + ", Name: " + product.name + ", Price: " + std::to_string(product.price) + "\n";
        }
        res.set_content(response, "text/plain");
    });

    svr.Get(R"(/products/(\d+))", [&productService](const httplib::Request& req, httplib::Response& res) {
        int productId = std::stoi(req.matches[1]);
        try {
            auto product = productService.getProduct(productId);
            std::string response = "Product ID: " + std::to_string(product.id) + ", Name: " + product.name + ", Price: " + std::to_string(product.price) + "\n";
            res.set_content(response, "text/plain");
        } catch (const std::exception& e) {
            res.status = 404;
            res.set_content("Product not found", "text/plain");
        }
    });

    svr.listen("0.0.0.0", 8081);
    return 0;
}
```

**Exposing OrderService**

```cpp
// main.cpp
#include "OrderService.h"
#include "httplib.h"

int main() {
    OrderService orderService;
    httplib::Server svr;

    svr.Get("/orders", [&orderService](const httplib::Request& req, httplib::Response& res) {
        auto orders = orderService.listOrders();
        std::string response;
        for (const auto& order : orders) {
            response += "Order ID: " + std::to_string(order.id) + ", Customer ID: " + std::to_string(order.customerId) + ", Product ID: " + std::to_string(order.productId) + ", Status: " + order.status + "\n";
        }
        res.set_content(response, "text/plain");
    });

    svr.Post("/orders", [&orderService](const httplib::Request& req, httplib::Response& res) {
        auto customerId = std::stoi(req.get_param_value("customerId"));
        auto productId = std::stoi(req.get_param_value("productId"));
        orderService.createOrder(customerId, productId);
        res.set_content("Order created", "text/plain");
    });

    svr.listen("0.0.0.0", 8082);
    return 0;
}
```

#### 23.2.6. Composing Services

One of the key strengths of SOA is the ability to compose services to build more complex workflows. Let's create a simple service composition example where the `OrderService` interacts with `CustomerService` and `ProductService`.

```cpp
// CompositeOrderService.h
#ifndef COMPOSITE_ORDER_SERVICE_H
#define COMPOSITE_ORDER_SERVICE_H

#include "ICustomerService.h"
#include "IProductService.h"
#include "IOrderService.h"

class CompositeOrderService : public IOrderService {
    ICustomerService* customerService;
    IProductService* productService;
    IOrderService* orderService;

public:
    CompositeOrderService(ICustomerService* cs, IProductService* ps, IOrderService* os)
        : customerService(cs), productService(ps), orderService(os) {}

    Order getOrder(int orderId) override {
        return orderService->getOrder(orderId);
    }

    void createOrder(int customerId, int productId) override {
        auto customer = customerService->getCustomer(customerId);
        auto product = productService->getProduct(productId);
        orderService->createOrder(customer.id, product.id);
    }

    std::vector<Order> listOrders() override {
        return orderService->listOrders();
    }
};

#endif // COMPOSITE_ORDER_SERVICE_H
```

#### Conclusion

Service-Oriented Architecture (SOA) provides a robust framework for designing reusable and interoperable services. By adhering to principles such as service abstraction, loose coupling, and composability, SOA enables the creation of scalable and maintainable systems. Although implementing SOA can be complex and requires careful planning, the benefits in terms of reusability and flexibility are significant. Through the use of well-defined interfaces and RESTful communication, we can effectively design and integrate services in a SOA-based system.

### 23.3. Event-Driven Architecture: Asynchronous Communication

Event-Driven Architecture (EDA) is a design pattern that focuses on the production, detection, and consumption of events to enable asynchronous communication within a system. EDA is particularly useful for building highly scalable, responsive, and decoupled systems, as it allows different components to interact without direct dependencies on each other. This subchapter will explore the fundamental concepts, benefits, and challenges of EDA, accompanied by detailed code examples to illustrate key principles.

#### 23.3.1. Introduction to Event-Driven Architecture

EDA is based on the following key principles:
1. **Event Producers and Consumers**: Components in an EDA system are classified as event producers or event consumers.
2. **Event Channels**: Events are communicated through channels, which can be message queues, event buses, or other intermediaries.
3. **Asynchronous Communication**: Events are processed asynchronously, allowing systems to remain responsive even under heavy load.
4. **Decoupling**: Producers and consumers are decoupled, meaning changes to one do not directly impact the other.

#### 23.3.2. Benefits of EDA

EDA offers several advantages:
- **Scalability**: Asynchronous processing allows systems to handle high volumes of events efficiently.
- **Responsiveness**: Systems can remain responsive by offloading work to background processing.
- **Decoupling**: Producers and consumers are loosely coupled, enhancing maintainability and flexibility.
- **Real-Time Processing**: EDA supports real-time processing of events, which is essential for applications like financial trading or real-time analytics.

#### 23.3.3. Challenges of EDA

Implementing EDA comes with its own set of challenges:
- **Complexity**: Designing and managing event-driven systems can be complex, especially in terms of error handling and event ordering.
- **Debugging**: Asynchronous processes can be harder to debug compared to synchronous processes.
- **Data Consistency**: Ensuring data consistency across distributed components can be challenging.

#### 23.3.4. Implementing EDA in C++

Let's explore a practical example of implementing EDA using C++. We will develop a basic system with three components: `OrderService`, `InventoryService`, and `NotificationService`.

##### 23.3.4.1. Setting Up the Environment

To manage asynchronous communication, we will use a message queue. RabbitMQ is a popular choice for this purpose. Ensure you have RabbitMQ installed and running on your system.

##### 23.3.4.2. Defining Events

First, we will define the events that our system will use.

```cpp
// Event.h
#ifndef EVENT_H
#define EVENT_H

#include <string>

class Event {
public:
    std::string type;
    std::string data;

    Event(const std::string& type, const std::string& data)
        : type(type), data(data) {}
};

#endif // EVENT_H
```

##### 23.3.4.3. Event Producer

**OrderService** will produce events when orders are created.

```cpp
// OrderService.h
#ifndef ORDER_SERVICE_H
#define ORDER_SERVICE_H

#include "Event.h"
#include <string>

class OrderService {
public:
    void createOrder(const std::string& orderId, const std::string& productId);
};

#endif // ORDER_SERVICE_H
```

```cpp
// OrderService.cpp
#include "OrderService.h"
#include <iostream>
#include <amqpcpp.h>
#include <amqpcpp/libevent.h>

void OrderService::createOrder(const std::string& orderId, const std::string& productId) {
    // Simulate order creation logic
    std::cout << "Order created: Order ID " << orderId << ", Product ID " << productId << std::endl;

    // Produce event
    Event event("OrderCreated", "Order ID: " + orderId + ", Product ID: " + productId);

    // Publish event to RabbitMQ
    struct event_base *base = event_base_new();
    AMQP::LibEventHandler handler(base);
    AMQP::TcpConnection connection(&handler, AMQP::Address("amqp://guest:guest@localhost/"));
    AMQP::TcpChannel channel(&connection);

    channel.onReady([&]() {
        channel.publish("", "order_events", event.data);
        connection.close();
    });

    event_base_dispatch(base);
    event_base_free(base);
}
```

##### 23.3.4.4. Event Consumers

**InventoryService** and **NotificationService** will consume events.

```cpp
// InventoryService.h
#ifndef INVENTORY_SERVICE_H
#define INVENTORY_SERVICE_H

#include "Event.h"
#include <amqpcpp.h>
#include <amqpcpp/libevent.h>

class InventoryService {
public:
    void handleEvent(const Event& event);
    void start();
};

#endif // INVENTORY_SERVICE_H
```

```cpp
// InventoryService.cpp
#include "InventoryService.h"
#include <iostream>
#include <event2/event.h>

void InventoryService::handleEvent(const Event& event) {
    if (event.type == "OrderCreated") {
        std::cout << "InventoryService received event: " << event.data << std::endl;
        // Update inventory logic here
    }
}

void InventoryService::start() {
    struct event_base *base = event_base_new();
    AMQP::LibEventHandler handler(base);
    AMQP::TcpConnection connection(&handler, AMQP::Address("amqp://guest:guest@localhost/"));
    AMQP::TcpChannel channel(&connection);

    channel.declareQueue("order_events").onSuccess([&](const std::string &name, uint32_t messageCount, uint32_t consumerCount) {
        channel.consume(name, AMQP::noack).onReceived([&](const AMQP::Message &message, uint64_t deliveryTag, bool redelivered) {
            Event event("OrderCreated", message.body());
            handleEvent(event);
        });
    });

    event_base_dispatch(base);
    event_base_free(base);
}
```

```cpp
// NotificationService.h
#ifndef NOTIFICATION_SERVICE_H
#define NOTIFICATION_SERVICE_H

#include "Event.h"
#include <amqpcpp.h>
#include <amqpcpp/libevent.h>

class NotificationService {
public:
    void handleEvent(const Event& event);
    void start();
};

#endif // NOTIFICATION_SERVICE_H
```

```cpp
// NotificationService.cpp
#include "NotificationService.h"
#include <iostream>
#include <event2/event.h>

void NotificationService::handleEvent(const Event& event) {
    if (event.type == "OrderCreated") {
        std::cout << "NotificationService received event: " << event.data << std::endl;
        // Send notification logic here
    }
}

void NotificationService::start() {
    struct event_base *base = event_base_new();
    AMQP::LibEventHandler handler(base);
    AMQP::TcpConnection connection(&handler, AMQP::Address("amqp://guest:guest@localhost/"));
    AMQP::TcpChannel channel(&connection);

    channel.declareQueue("order_events").onSuccess([&](const std::string &name, uint32_t messageCount, uint32_t consumerCount) {
        channel.consume(name, AMQP::noack).onReceived([&](const AMQP::Message &message, uint64_t deliveryTag, bool redelivered) {
            Event event("OrderCreated", message.body());
            handleEvent(event);
        });
    });

    event_base_dispatch(base);
    event_base_free(base);
}
```

#### 23.3.5. Running the Example

To run this example, follow these steps:
1. Ensure RabbitMQ is installed and running.
2. Compile and run `OrderService`, `InventoryService`, and `NotificationService`.
3. Create an order using `OrderService`.

When an order is created, `OrderService` will publish an `OrderCreated` event to RabbitMQ. `InventoryService` and `NotificationService` will consume this event and perform their respective actions asynchronously.

#### 23.3.6. Advanced Topics in EDA

##### 23.3.6.1. Event Sourcing

Event sourcing is a pattern where changes to the application's state are stored as a sequence of events. This provides an audit trail and allows reconstructing the application's state at any point in time.

```cpp
// EventStore.h
#ifndef EVENT_STORE_H
#define EVENT_STORE_H

#include "Event.h"
#include <vector>

class EventStore {
    std::vector<Event> events;

public:
    void saveEvent(const Event& event) {
        events.push_back(event);
    }

    std::vector<Event> getEvents() const {
        return events;
    }
};

#endif // EVENT_STORE_H
```

##### 23.3.6.2. CQRS (Command Query Responsibility Segregation)

CQRS is a pattern that separates the read and write operations of a system. This allows optimizing the read and write sides independently and can be particularly effective in combination with event sourcing.

```cpp
// CommandHandler.h
#ifndef COMMAND_HANDLER_H
#define COMMAND_HANDLER_H

#include <string>

class CommandHandler {
public:
    void handleCommand(const std::string& command) {
        // Handle write operations
    }
};

#endif // COMMAND_HANDLER_H
```

```cpp
// QueryHandler.h
#ifndef QUERY_HANDLER_H
#define QUERY_HANDLER_H

#include <string>

class QueryHandler {
public:
    std::string handleQuery(const std::string& query) {
        // Handle read operations
        return "Query result";
    }
};

#endif // QUERY_HANDLER_H
```

#### Conclusion

Event-Driven Architecture (EDA) provides a powerful framework for building scalable, responsive, and decoupled systems. By leveraging asynchronous communication through events, EDA enables different components to interact without direct dependencies on each other. Although implementing EDA can be complex, the benefits in terms of scalability, responsiveness, and maintainability are significant. Through the use of message queues like RabbitMQ and well-defined event handling mechanisms, we can effectively design and implement event-driven systems in C++. Advanced topics like event sourcing and CQRS further enhance the capabilities of EDA, providing robust solutions for modern software applications.

### 23.4. Hexagonal Architecture: Ports and Adapters

Hexagonal Architecture, also known as the Ports and Adapters pattern, is an architectural style that aims to create loosely coupled, highly testable systems. By decoupling the core business logic from external dependencies through the use of ports and adapters, Hexagonal Architecture promotes a clean separation of concerns, facilitating easier maintenance, testing, and evolution of software. This subchapter will explore the fundamental concepts, benefits, and challenges of Hexagonal Architecture, accompanied by detailed code examples to illustrate key principles.

#### 23.4.1. Introduction to Hexagonal Architecture

Hexagonal Architecture, introduced by Alistair Cockburn, is based on the following key principles:
1. **Core Domain Logic**: The core business logic of the application, which is isolated from external systems.
2. **Ports**: Interfaces that define the communication boundaries between the core domain logic and the outside world.
3. **Adapters**: Implementations of ports that handle the interaction with external systems such as databases, web services, or user interfaces.
4. **Dependency Inversion**: The core domain depends on abstractions (ports) rather than concrete implementations (adapters).

#### 23.4.2. Benefits of Hexagonal Architecture

Hexagonal Architecture offers several advantages:
- **Testability**: The core business logic can be tested independently of external systems by using mock implementations of ports.
- **Flexibility**: Adapters can be easily replaced or modified without impacting the core domain logic.
- **Maintainability**: The separation of concerns reduces the complexity of the codebase, making it easier to maintain.
- **Decoupling**: The core domain is decoupled from external dependencies, promoting a clean and modular design.

#### 23.4.3. Challenges of Hexagonal Architecture

Despite its benefits, implementing Hexagonal Architecture can be challenging:
- **Complexity**: The additional layers of abstraction can introduce complexity, especially in smaller projects.
- **Learning Curve**: Developers may need to familiarize themselves with the concepts and patterns of Hexagonal Architecture.

#### 23.4.4. Implementing Hexagonal Architecture in C++

Let's explore a practical example of implementing Hexagonal Architecture using C++. We will develop a simple application with three main components: `ApplicationService`, `PersistenceAdapter`, and `WebAdapter`.

##### 23.4.4.1. Defining the Core Domain Logic

First, we define the core business logic and the associated interfaces (ports).

```cpp
// Product.h
#ifndef PRODUCT_H
#define PRODUCT_H

#include <string>

class Product {
public:
    int id;
    std::string name;
    double price;

    Product(int id, const std::string& name, double price)
        : id(id), name(name), price(price) {}
};

#endif // PRODUCT_H
```

```cpp
// IProductRepository.h
#ifndef I_PRODUCT_REPOSITORY_H
#define I_PRODUCT_REPOSITORY_H

#include "Product.h"
#include <vector>

class IProductRepository {
public:
    virtual ~IProductRepository() = default;
    virtual void addProduct(const Product& product) = 0;
    virtual Product getProduct(int productId) = 0;
    virtual std::vector<Product> listProducts() = 0;
};

#endif // I_PRODUCT_REPOSITORY_H
```

```cpp
// IProductService.h
#ifndef I_PRODUCT_SERVICE_H
#define I_PRODUCT_SERVICE_H

#include "Product.h"
#include <vector>

class IProductService {
public:
    virtual ~IProductService() = default;
    virtual void createProduct(const Product& product) = 0;
    virtual Product fetchProduct(int productId) = 0;
    virtual std::vector<Product> fetchAllProducts() = 0;
};

#endif // I_PRODUCT_SERVICE_H
```

##### 23.4.4.2. Implementing the Core Domain Logic

Next, we implement the core business logic, which depends on the defined interfaces (ports).

```cpp
// ProductService.h
#ifndef PRODUCT_SERVICE_H
#define PRODUCT_SERVICE_H

#include "IProductService.h"
#include "IProductRepository.h"

class ProductService : public IProductService {
    IProductRepository& productRepository;

public:
    ProductService(IProductRepository& repo) : productRepository(repo) {}

    void createProduct(const Product& product) override {
        productRepository.addProduct(product);
    }

    Product fetchProduct(int productId) override {
        return productRepository.getProduct(productId);
    }

    std::vector<Product> fetchAllProducts() override {
        return productRepository.listProducts();
    }
};

#endif // PRODUCT_SERVICE_H
```

##### 23.4.4.3. Creating Adapters

Adapters are implementations of the ports that handle the interaction with external systems.

**Persistence Adapter**

The persistence adapter interacts with a database to store and retrieve products.

```cpp
// InMemoryProductRepository.h
#ifndef IN_MEMORY_PRODUCT_REPOSITORY_H
#define IN_MEMORY_PRODUCT_REPOSITORY_H

#include "IProductRepository.h"
#include <unordered_map>

class InMemoryProductRepository : public IProductRepository {
    std::unordered_map<int, Product> products;

public:
    void addProduct(const Product& product) override {
        products[product.id] = product;
    }

    Product getProduct(int productId) override {
        if (products.find(productId) != products.end()) {
            return products[productId];
        }
        throw std::runtime_error("Product not found");
    }

    std::vector<Product> listProducts() override {
        std::vector<Product> productList;
        for (const auto& [id, product] : products) {
            productList.push_back(product);
        }
        return productList;
    }
};

#endif // IN_MEMORY_PRODUCT_REPOSITORY_H
```

**Web Adapter**

The web adapter exposes the product service as RESTful endpoints using a lightweight HTTP server library.

```cpp
// WebAdapter.h
#ifndef WEB_ADAPTER_H
#define WEB_ADAPTER_H

#include "IProductService.h"
#include <httplib.h>

class WebAdapter {
    IProductService& productService;

public:
    WebAdapter(IProductService& service) : productService(service) {}

    void start() {
        httplib::Server svr;

        svr.Post("/products", [this](const httplib::Request& req, httplib::Response& res) {
            // Parse product from request body
            auto product = parseProduct(req.body);
            productService.createProduct(product);
            res.set_content("Product created", "text/plain");
        });

        svr.Get(R"(/products/(\d+))", [this](const httplib::Request& req, httplib::Response& res) {
            int productId = std::stoi(req.matches[1]);
            try {
                auto product = productService.fetchProduct(productId);
                res.set_content(serializeProduct(product), "application/json");
            } catch (const std::exception& e) {
                res.status = 404;
                res.set_content("Product not found", "text/plain");
            }
        });

        svr.Get("/products", [this](const httplib::Request& req, httplib::Response& res) {
            auto products = productService.fetchAllProducts();
            res.set_content(serializeProducts(products), "application/json");
        });

        svr.listen("0.0.0.0", 8080);
    }

private:
    Product parseProduct(const std::string& body) {
        // Simple JSON parsing (for illustration purposes)
        // In a real application, use a proper JSON library
        int id = /* extract id from body */;
        std::string name = /* extract name from body */;
        double price = /* extract price from body */;
        return Product(id, name, price);
    }

    std::string serializeProduct(const Product& product) {
        return "{ \"id\": " + std::to_string(product.id) +
               ", \"name\": \"" + product.name +
               "\", \"price\": " + std::to_string(product.price) + " }";
    }

    std::string serializeProducts(const std::vector<Product>& products) {
        std::string result = "[";
        for (const auto& product : products) {
            result += serializeProduct(product) + ",";
        }
        if (!products.empty()) {
            result.pop_back(); // Remove trailing comma
        }
        result += "]";
        return result;
    }
};

#endif // WEB_ADAPTER_H
```

#### 23.4.5. Running the Example

To run this example, follow these steps:
1. Implement the core domain logic, persistence adapter, and web adapter.
2. Instantiate the `ProductService` and adapters in your main function.
3. Start the web server to expose the RESTful endpoints.

```cpp
// main.cpp
#include "ProductService.h"
#include "InMemoryProductRepository.h"
#include "WebAdapter.h"

int main() {
    InMemoryProductRepository productRepository;
    ProductService productService(productRepository);
    WebAdapter webAdapter(productService);

    webAdapter.start();
    return 0;
}
```

#### 23.4.6. Advanced Topics in Hexagonal Architecture

##### 23.4.6.1. Testing

One of the significant benefits of Hexagonal Architecture is the ease of testing. By using mock implementations of ports, you can test the core domain logic in isolation.

```cpp
// MockProductRepository.h
#ifndef MOCK_PRODUCT_REPOSITORY_H
#define MOCK_PRODUCT_REPOSITORY_H

#include "IProductRepository.h"
#include <unordered_map>

class MockProductRepository : public IProductRepository {
    std::unordered_map<int, Product> products;

public:
    void addProduct(const Product& product) override {
        products[product.id] = product;
    }

    Product getProduct(int productId) override {
        if (products.find(productId) != products.end()) {
            return products[productId];
        }
        throw std::runtime_error("Product not found");
    }

    std::vector<Product> listProducts() override {
        std::vector<Product> productList;
        for (const auto& [id, product] : products) {
            productList.push_back(product);
        }
        return productList;
    }
};

#endif // MOCK_PRODUCT_REPOSITORY_H
```

```cpp
// ProductServiceTest.cpp
#include "ProductService.h"
#include "MockProductRepository.h"
#include <cassert>

void testCreateProduct() {
    MockProductRepository mockRepo;
    ProductService productService(mockRepo);

    Product product(1, "Laptop", 999.99);
    productService.createProduct(product);

    assert(mockRepo.getProduct(1).name == "Laptop");
    assert(mockRepo.getProduct(1).price == 999.99);
}

void testFetchProduct() {
    MockProductRepository mockRepo;
    ProductService productService(mockRepo);

    Product product(1, "Laptop", 999.99);
    mockRepo.addProduct(product);

    assert(productService.fetchProduct(1).name == "Laptop");
    assert(productService.fetchProduct(1).price == 999.99);
}

int main() {
    testCreateProduct();
    testFetchProduct();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
```

##### 23.4.6.2. Dependency Injection

Using dependency injection frameworks can help manage the instantiation and wiring of ports and adapters.

```cpp
// DependencyInjection.cpp
#include "ProductService.h"
#include "InMemoryProductRepository.h"
#include "WebAdapter.h"

class DependencyInjection {
public:
    static ProductService createProductService() {
        static InMemoryProductRepository productRepository;
        return ProductService(productRepository);
    }

    static WebAdapter createWebAdapter() {
        static ProductService productService = createProductService();
        return WebAdapter(productService);
    }
};

int main() {
    WebAdapter webAdapter = DependencyInjection::createWebAdapter();
    webAdapter.start();
    return 0;
}
```

#### 23.4.7. Conclusion

Hexagonal Architecture, or Ports and Adapters, provides a robust framework for creating loosely coupled, highly testable systems. By decoupling the core domain logic from external dependencies through the use of ports and adapters, this architecture promotes a clean separation of concerns, facilitating easier maintenance, testing, and evolution of software. Although implementing Hexagonal Architecture can introduce complexity, the benefits in terms of testability, flexibility, and maintainability are significant. Through the use of well-defined interfaces and adapters, we can effectively design and implement hexagonal systems in C++. Advanced topics like testing and dependency injection further enhance the capabilities of Hexagonal Architecture, providing robust solutions for modern software applications.
