\newpage


## 9. Real-Time Operating Systems (RTOS) and C++
In the ever-evolving landscape of embedded systems, the integration of Real-Time Operating Systems (RTOS) with C++ has become crucial for building robust, efficient, and scalable applications. This chapter delves into the essential aspects of RTOS integration, providing techniques for seamless incorporation into your C++ projects. You will explore task management and scheduling, ensuring your code harmonizes with task schedulers to optimize performance. Furthermore, we will discuss synchronization and inter-task communication, covering the use of mutexes, semaphores, and other mechanisms to maintain data integrity and facilitate smooth inter-process interactions. Through practical insights and examples, this chapter aims to equip you with the knowledge to leverage RTOS capabilities effectively in your embedded systems programming.

### 9.1. Integrating with an RTOS

Integrating an RTOS with your C++ projects involves understanding the underlying principles of RTOS and leveraging its features to enhance the functionality and performance of your embedded system applications. This subchapter will guide you through the techniques for seamless integration, focusing on setting up your development environment, understanding the core components of an RTOS, and integrating them with C++.

#### 9.1.1. Setting Up the Development Environment

To begin with RTOS integration, you need to set up a suitable development environment. This typically involves selecting an RTOS that suits your application requirements and configuring your build system to support both the RTOS and C++ code.

**Example: Setting up FreeRTOS with CMake and GCC**

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(EmbeddedRTOS CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Include FreeRTOS source files
add_subdirectory(freertos)

# Add your application source files
add_executable(my_app main.cpp)

# Link FreeRTOS to your application
target_link_libraries(my_app PRIVATE FreeRTOS::FreeRTOS)
```

**main.cpp**

```cpp
#include <FreeRTOS.h>
#include <task.h>
#include <iostream>

void vTaskFunction(void* pvParameters) {
    while (true) {
        std::cout << "Task is running\n";
        vTaskDelay(pdMS_TO_TICKS(1000)); // Delay for 1000ms
    }
}

int main() {
    xTaskCreate(vTaskFunction, "Task 1", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);
    vTaskStartScheduler();

    // Should never reach here
    while (true);
    return 0;
}
```

#### 9.1.2. Understanding Core Components of an RTOS

An RTOS provides several core components that are crucial for real-time performance and multitasking. These include tasks, schedulers, queues, semaphores, and mutexes.

**Tasks and Schedulers**

Tasks are the basic units of execution in an RTOS. The scheduler is responsible for switching between tasks based on their priorities and states.

**Example: Creating Multiple Tasks**

```cpp
void vTask1(void* pvParameters) {
    while (true) {
        std::cout << "Task 1 is running\n";
        vTaskDelay(pdMS_TO_TICKS(500)); // Delay for 500ms
    }
}

void vTask2(void* pvParameters) {
    while (true) {
        std::cout << "Task 2 is running\n";
        vTaskDelay(pdMS_TO_TICKS(1000)); // Delay for 1000ms
    }
}

int main() {
    xTaskCreate(vTask1, "Task 1", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);
    xTaskCreate(vTask2, "Task 2", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);
    vTaskStartScheduler();

    // Should never reach here
    while (true);
    return 0;
}
```

#### 9.1.3. Task Management and Scheduling

Writing C++ code that integrates well with task schedulers involves understanding task priorities, states, and context switching.

**Task Priorities and Preemption**

In an RTOS, tasks can have different priorities. The scheduler ensures that the highest priority task runs first.

**Example: Task Priorities**

```cpp
void vHighPriorityTask(void* pvParameters) {
    while (true) {
        std::cout << "High priority task running\n";
        vTaskDelay(pdMS_TO_TICKS(200)); // Delay for 200ms
    }
}

void vLowPriorityTask(void* pvParameters) {
    while (true) {
        std::cout << "Low priority task running\n";
        vTaskDelay(pdMS_TO_TICKS(1000)); // Delay for 1000ms
    }
}

int main() {
    xTaskCreate(vHighPriorityTask, "High Priority Task", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 2, nullptr);
    xTaskCreate(vLowPriorityTask, "Low Priority Task", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);
    vTaskStartScheduler();

    // Should never reach here
    while (true);
    return 0;
}
```

**Context Switching**

Context switching involves saving the state of a currently running task and loading the state of the next task to run.

**Example: Context Switching**

```cpp
void vTaskA(void* pvParameters) {
    while (true) {
        std::cout << "Task A running\n";
        vTaskDelay(pdMS_TO_TICKS(300)); // Delay for 300ms
    }
}

void vTaskB(void* pvParameters) {
    while (true) {
        std::cout << "Task B running\n";
        vTaskDelay(pdMS_TO_TICKS(600)); // Delay for 600ms
    }
}

int main() {
    xTaskCreate(vTaskA, "Task A", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);
    xTaskCreate(vTaskB, "Task B", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);
    vTaskStartScheduler();

    // Should never reach here
    while (true);
    return 0;
}
```

#### 9.1.4. Synchronization and Inter-task Communication

In real-time systems, tasks often need to communicate and synchronize with each other. This is achieved using mechanisms like mutexes, semaphores, and queues.

**Mutexes**

Mutexes are used to ensure mutual exclusion, preventing multiple tasks from accessing shared resources simultaneously.

**Example: Using Mutexes**

```cpp
#include <semphr.h>

SemaphoreHandle_t xMutex;

void vTaskUsingMutex(void* pvParameters) {
    while (true) {
        if (xSemaphoreTake(xMutex, portMAX_DELAY) == pdTRUE) {
            std::cout << "Task has acquired the mutex\n";
            vTaskDelay(pdMS_TO_TICKS(500)); // Simulate work
            xSemaphoreGive(xMutex);
            std::cout << "Task has released the mutex\n";
        }
        vTaskDelay(pdMS_TO_TICKS(100)); // Delay to simulate other work
    }
}

int main() {
    xMutex = xSemaphoreCreateMutex();

    xTaskCreate(vTaskUsingMutex, "Task 1", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);
    xTaskCreate(vTaskUsingMutex, "Task 2", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);
    vTaskStartScheduler();

    // Should never reach here
    while (true);
    return 0;
}
```

**Semaphores**

Semaphores are used for signaling between tasks, especially for event notification.

**Example: Using Semaphores**

```cpp
SemaphoreHandle_t xBinarySemaphore;

void vTaskWaiting(void* pvParameters) {
    while (true) {
        if (xSemaphoreTake(xBinarySemaphore, portMAX_DELAY) == pdTRUE) {
            std::cout << "Semaphore taken, Task proceeding\n";
        }
    }
}

void vTaskGiving(void* pvParameters) {
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(1000)); // Simulate periodic event
        xSemaphoreGive(xBinarySemaphore);
        std::cout << "Semaphore given\n";
    }
}

int main() {
    xBinarySemaphore = xSemaphoreCreateBinary();

    xTaskCreate(vTaskWaiting, "Task Waiting", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);
    xTaskCreate(vTaskGiving, "Task Giving", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);
    vTaskStartScheduler();

    // Should never reach here
    while (true);
    return 0;
}
```

**Queues**

Queues are used for inter-task communication, allowing tasks to send and receive messages in a FIFO manner.

**Example: Using Queues**

```cpp
QueueHandle_t xQueue;

void vSenderTask(void* pvParameters) {
    int32_t lValueToSend = 100;
    while (true) {
        if (xQueueSend(xQueue, &lValueToSend, portMAX_DELAY) == pdPASS) {
            std::cout << "Value sent: " << lValueToSend << "\n";
            lValueToSend++;
        }
        vTaskDelay(pdMS_TO_TICKS(500)); // Delay to simulate work
    }
}

void vReceiverTask(void* pvParameters) {
    int32_t lReceivedValue;
    while (true) {
        if (xQueueReceive(xQueue, &lReceivedValue, portMAX_DELAY) == pdPASS) {
            std::cout << "Value received: " << lReceivedValue << "\n";
        }
    }
}

int main() {
    xQueue = xQueueCreate(10, sizeof(int32_t));

    xTaskCreate(vSenderTask, "Sender Task", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);
    xTaskCreate(vReceiverTask, "Receiver Task", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);
    vTaskStartScheduler();

    // Should never reach here
    while (true);
    return 0;
}
```

By understanding and leveraging these core components of an RTOS, you can effectively integrate it with your C++ applications, enabling the development of robust and efficient embedded systems. This foundation will pave the way for more advanced topics, such as handling real-time constraints and optimizing system performance.

### 9.2. Task Management and Scheduling

Task management and scheduling are critical components in the design and implementation of real-time embedded systems. In this subchapter, we will explore how to write C++ code that integrates effectively with task schedulers in an RTOS environment. We will cover various aspects of task management, including task creation, prioritization, state management, and context switching. Additionally, we will delve into advanced scheduling techniques to ensure your system meets real-time performance requirements.

#### 9.2.1. Task Creation and Management

Creating and managing tasks in an RTOS involves defining task functions, setting priorities, and ensuring efficient use of system resources.

**Example: Basic Task Creation**

```cpp
#include <FreeRTOS.h>
#include <task.h>
#include <iostream>

void vTaskFunction(void* pvParameters) {
    const char* taskName = static_cast<const char*>(pvParameters);
    while (true) {
        std::cout << taskName << " is running\n";
        vTaskDelay(pdMS_TO_TICKS(1000)); // Delay for 1000ms
    }
}

int main() {
    xTaskCreate(vTaskFunction, "Task 1", configMINIMAL_STACK_SIZE, (void*)"Task 1", tskIDLE_PRIORITY + 1, nullptr);
    xTaskCreate(vTaskFunction, "Task 2", configMINIMAL_STACK_SIZE, (void*)"Task 2", tskIDLE_PRIORITY + 1, nullptr);
    vTaskStartScheduler();

    // Should never reach here
    while (true);
    return 0;
}
```

In this example, we define a simple task function `vTaskFunction` that prints a message and delays for 1000 milliseconds. We create two tasks with different names and start the scheduler.

#### 9.2.2. Task Prioritization

Task prioritization is essential in real-time systems to ensure that critical tasks receive the necessary CPU time. The RTOS scheduler uses task priorities to decide which task to run next.

**Example: Task Prioritization**

```cpp
void vHighPriorityTask(void* pvParameters) {
    while (true) {
        std::cout << "High priority task running\n";
        vTaskDelay(pdMS_TO_TICKS(500)); // Delay for 500ms
    }
}

void vLowPriorityTask(void* pvParameters) {
    while (true) {
        std::cout << "Low priority task running\n";
        vTaskDelay(pdMS_TO_TICKS(1000)); // Delay for 1000ms
    }
}

int main() {
    xTaskCreate(vHighPriorityTask, "High Priority Task", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 2, nullptr);
    xTaskCreate(vLowPriorityTask, "Low Priority Task", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);
    vTaskStartScheduler();

    // Should never reach here
    while (true);
    return 0;
}
```

Here, the high-priority task runs more frequently than the low-priority task, demonstrating how the scheduler manages task execution based on priorities.

#### 9.2.3. Task States and Transitions

Tasks in an RTOS can be in various states such as running, ready, blocked, or suspended. Understanding these states and how to transition between them is crucial for effective task management.

**Example: Task States**

```cpp
void vTaskStateFunction(void* pvParameters) {
    while (true) {
        std::cout << "Task is running\n";
        vTaskDelay(pdMS_TO_TICKS(1000)); // Task enters blocked state for 1000ms
    }
}

int main() {
    TaskHandle_t xHandle = nullptr;
    xTaskCreate(vTaskStateFunction, "State Task", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, &xHandle);

    // Suspend the task
    vTaskSuspend(xHandle);
    std::cout << "Task suspended\n";
    vTaskDelay(pdMS_TO_TICKS(2000)); // Delay to simulate other work

    // Resume the task
    vTaskResume(xHandle);
    std::cout << "Task resumed\n";

    vTaskStartScheduler();

    // Should never reach here
    while (true);
    return 0;
}
```

This example shows how to suspend and resume a task, demonstrating transitions between the suspended and ready states.

#### 9.2.4. Context Switching

Context switching is the process of saving the state of a currently running task and restoring the state of the next task to be executed. Efficient context switching is vital for maintaining system performance.

**Example: Context Switching**

```cpp
void vTaskA(void* pvParameters) {
    while (true) {
        std::cout << "Task A running\n";
        vTaskDelay(pdMS_TO_TICKS(300)); // Delay for 300ms
    }
}

void vTaskB(void* pvParameters) {
    while (true) {
        std::cout << "Task B running\n";
        vTaskDelay(pdMS_TO_TICKS(600)); // Delay for 600ms
    }
}

int main() {
    xTaskCreate(vTaskA, "Task A", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);
    xTaskCreate(vTaskB, "Task B", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);
    vTaskStartScheduler();

    // Should never reach here
    while (true);
    return 0;
}
```

In this example, `Task A` and `Task B` alternate execution, demonstrating context switching managed by the RTOS scheduler.

#### 9.2.5. Advanced Scheduling Techniques

Advanced scheduling techniques, such as round-robin, time-slicing, and rate-monotonic scheduling, help ensure that tasks meet their deadlines and system performance requirements.

**Round-Robin Scheduling**

Round-robin scheduling ensures that all tasks get an equal share of CPU time.

**Example: Round-Robin Scheduling**

```cpp
void vTaskRoundRobin(void* pvParameters) {
    const char* taskName = static_cast<const char*>(pvParameters);
    while (true) {
        std::cout << taskName << " is running\n";
        vTaskDelay(pdMS_TO_TICKS(500)); // Delay for 500ms
    }
}

int main() {
    xTaskCreate(vTaskRoundRobin, "Task 1", configMINIMAL_STACK_SIZE, (void*)"Task 1", tskIDLE_PRIORITY + 1, nullptr);
    xTaskCreate(vTaskRoundRobin, "Task 2", configMINIMAL_STACK_SIZE, (void*)"Task 2", tskIDLE_PRIORITY + 1, nullptr);
    xTaskCreate(vTaskRoundRobin, "Task 3", configMINIMAL_STACK_SIZE, (void*)"Task 3", tskIDLE_PRIORITY + 1, nullptr);

    vTaskStartScheduler();

    // Should never reach here
    while (true);
    return 0;
}
```

**Time-Slicing**

Time-slicing allows multiple tasks to share CPU time within a specific period.

**Example: Time-Slicing**

```cpp
void vTimeSliceTask(void* pvParameters) {
    const char* taskName = static_cast<const char*>(pvParameters);
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xFrequency = pdMS_TO_TICKS(200); // 200ms time slice

    while (true) {
        std::cout << taskName << " is running\n";
        vTaskDelayUntil(&xLastWakeTime, xFrequency);
    }
}

int main() {
    xTaskCreate(vTimeSliceTask, "Task 1", configMINIMAL_STACK_SIZE, (void*)"Task 1", tskIDLE_PRIORITY + 1, nullptr);
    xTaskCreate(vTimeSliceTask, "Task 2", configMINIMAL_STACK_SIZE, (void*)"Task 2", tskIDLE_PRIORITY + 1, nullptr);
    xTaskCreate(vTimeSliceTask, "Task 3", configMINIMAL_STACK_SIZE, (void*)"Task 3", tskIDLE_PRIORITY + 1, nullptr);

    vTaskStartScheduler();

    // Should never reach here
    while (true);
    return 0;
}
```

**Rate-Monotonic Scheduling**

Rate-monotonic scheduling assigns higher priorities to tasks with shorter periods.

**Example: Rate-Monotonic Scheduling**

```cpp
void vFastTask(void* pvParameters) {
    while (true) {
        std::cout << "Fast task running\n";
        vTaskDelay(pdMS_TO_TICKS(100)); // Delay for 100ms
    }
}

void vSlowTask(void* pvParameters) {
    while (true) {
        std::cout << "Slow task running\n";
        vTaskDelay(pdMS_TO_TICKS(500)); // Delay for 500ms
    }
}

int main() {
    xTaskCreate(vFastTask, "Fast Task", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 2, nullptr);
    xTaskCreate(vSlowTask, "Slow Task", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);

    vTaskStartScheduler();

    // Should never reach here
    while (true);
    return 0;
}
```

In this example, the fast task has a higher priority than the slow task, ensuring it runs more frequently, consistent with rate-monotonic scheduling principles.

#### 9.2.6. Task Suspension and Deletion

Tasks may need to be suspended and resumed

based on system requirements. Proper task deletion is also essential to free system resources.

**Example: Task Suspension and Deletion**

```cpp
TaskHandle_t xHandle1 = nullptr;
TaskHandle_t xHandle2 = nullptr;

void vTask1(void* pvParameters) {
    while (true) {
        std::cout << "Task 1 running\n";
        vTaskDelay(pdMS_TO_TICKS(1000)); // Delay for 1000ms
    }
}

void vTask2(void* pvParameters) {
    while (true) {
        std::cout << "Task 2 running\n";
        vTaskDelay(pdMS_TO_TICKS(1000)); // Delay for 1000ms
    }
}

int main() {
    xTaskCreate(vTask1, "Task 1", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, &xHandle1);
    xTaskCreate(vTask2, "Task 2", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, &xHandle2);

    vTaskDelay(pdMS_TO_TICKS(5000)); // Allow tasks to run for 5 seconds

    vTaskSuspend(xHandle1); // Suspend Task 1
    std::cout << "Task 1 suspended\n";

    vTaskDelay(pdMS_TO_TICKS(5000)); // Allow Task 2 to run alone for 5 seconds

    vTaskResume(xHandle1); // Resume Task 1
    std::cout << "Task 1 resumed\n";

    vTaskDelay(pdMS_TO_TICKS(5000)); // Allow tasks to run for 5 seconds

    vTaskDelete(xHandle1); // Delete Task 1
    vTaskDelete(xHandle2); // Delete Task 2
    std::cout << "Tasks deleted\n";

    vTaskStartScheduler();

    // Should never reach here
    while (true);
    return 0;
}
```

In this example, `Task 1` is suspended and then resumed, showcasing task state transitions. Both tasks are eventually deleted to free up resources.

#### Summary

Effective task management and scheduling are pivotal for the performance and reliability of real-time embedded systems. By leveraging the techniques and examples provided in this subchapter, you can design and implement systems that meet stringent real-time requirements. Whether it’s through basic task creation, advanced scheduling techniques, or proper task state management, mastering these concepts will enhance your ability to develop robust and efficient embedded applications.

### 9.3. Synchronization and Inter-task Communication

In real-time embedded systems, tasks often need to work together, share resources, and communicate with each other to achieve common goals. Effective synchronization and inter-task communication are crucial for ensuring data consistency, preventing race conditions, and achieving reliable system behavior. In this subchapter, we will explore various synchronization mechanisms and communication techniques available in C++ when using an RTOS. We will cover mutexes, semaphores, and queues, providing detailed explanations and rich code examples for each.

#### 9.3.1. Mutexes

Mutexes (Mutual Exclusion Objects) are used to prevent multiple tasks from accessing a shared resource simultaneously, ensuring data integrity. They are essential for protecting critical sections of code.

**Example: Using Mutexes**

```cpp
#include <FreeRTOS.h>
#include <task.h>
#include <semphr.h>
#include <iostream>

SemaphoreHandle_t xMutex;

void vTaskWithMutex(void* pvParameters) {
    const char* taskName = static_cast<const char*>(pvParameters);
    while (true) {
        // Try to take the mutex
        if (xSemaphoreTake(xMutex, portMAX_DELAY) == pdTRUE) {
            std::cout << taskName << " has acquired the mutex\n";
            vTaskDelay(pdMS_TO_TICKS(500)); // Simulate task working
            xSemaphoreGive(xMutex); // Release the mutex
            std::cout << taskName << " has released the mutex\n";
        }
        vTaskDelay(pdMS_TO_TICKS(100)); // Delay to simulate other work
    }
}

int main() {
    xMutex = xSemaphoreCreateMutex();

    if (xMutex != nullptr) {
        xTaskCreate(vTaskWithMutex, "Task 1", configMINIMAL_STACK_SIZE, (void*)"Task 1", tskIDLE_PRIORITY + 1, nullptr);
        xTaskCreate(vTaskWithMutex, "Task 2", configMINIMAL_STACK_SIZE, (void*)"Task 2", tskIDLE_PRIORITY + 1, nullptr);

        vTaskStartScheduler();
    }

    // Should never reach here
    while (true);
    return 0;
}
```

In this example, two tasks attempt to access a shared resource protected by a mutex. Only one task can hold the mutex at any time, ensuring that the critical section is not accessed concurrently.

#### 9.3.2. Semaphores

Semaphores are signaling mechanisms used to manage access to shared resources and synchronize tasks. They can be binary (taking values 0 and 1) or counting (taking values within a specified range).

**Example: Binary Semaphores**

```cpp
SemaphoreHandle_t xBinarySemaphore;

void vTaskWaiting(void* pvParameters) {
    while (true) {
        if (xSemaphoreTake(xBinarySemaphore, portMAX_DELAY) == pdTRUE) {
            std::cout << "Semaphore taken, Task proceeding\n";
            // Simulate task processing
            vTaskDelay(pdMS_TO_TICKS(500));
        }
    }
}

void vTaskGiving(void* pvParameters) {
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(1000)); // Simulate periodic event
        xSemaphoreGive(xBinarySemaphore);
        std::cout << "Semaphore given\n";
    }
}

int main() {
    xBinarySemaphore = xSemaphoreCreateBinary();

    if (xBinarySemaphore != nullptr) {
        xTaskCreate(vTaskWaiting, "Task Waiting", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);
        xTaskCreate(vTaskGiving, "Task Giving", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);

        vTaskStartScheduler();
    }

    // Should never reach here
    while (true);
    return 0;
}
```

In this example, `vTaskWaiting` waits for the semaphore to be given by `vTaskGiving`. When `vTaskGiving` gives the semaphore, `vTaskWaiting` proceeds with its task.

**Example: Counting Semaphores**

```cpp
SemaphoreHandle_t xCountingSemaphore;

void vTaskProducer(void* pvParameters) {
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(500)); // Simulate item production
        xSemaphoreGive(xCountingSemaphore);
        std::cout << "Produced an item\n";
    }
}

void vTaskConsumer(void* pvParameters) {
    while (true) {
        if (xSemaphoreTake(xCountingSemaphore, portMAX_DELAY) == pdTRUE) {
            std::cout << "Consumed an item\n";
            // Simulate item consumption
            vTaskDelay(pdMS_TO_TICKS(1000));
        }
    }
}

int main() {
    xCountingSemaphore = xSemaphoreCreateCounting(10, 0); // Max count 10, initial count 0

    if (xCountingSemaphore != nullptr) {
        xTaskCreate(vTaskProducer, "Producer", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);
        xTaskCreate(vTaskConsumer, "Consumer", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);

        vTaskStartScheduler();
    }

    // Should never reach here
    while (true);
    return 0;
}
```

In this example, the counting semaphore allows the producer task to signal the availability of items, while the consumer task waits for these signals to consume items.

#### 9.3.3. Queues

Queues are used for inter-task communication, allowing tasks to send and receive messages in a FIFO manner. They are essential for passing data between tasks without shared memory.

**Example: Using Queues**

```cpp
QueueHandle_t xQueue;

void vSenderTask(void* pvParameters) {
    int32_t lValueToSend = 100;
    while (true) {
        if (xQueueSend(xQueue, &lValueToSend, portMAX_DELAY) == pdPASS) {
            std::cout << "Value sent: " << lValueToSend << "\n";
            lValueToSend++;
        }
        vTaskDelay(pdMS_TO_TICKS(500)); // Delay to simulate work
    }
}

void vReceiverTask(void* pvParameters) {
    int32_t lReceivedValue;
    while (true) {
        if (xQueueReceive(xQueue, &lReceivedValue, portMAX_DELAY) == pdPASS) {
            std::cout << "Value received: " << lReceivedValue << "\n";
        }
    }
}

int main() {
    xQueue = xQueueCreate(10, sizeof(int32_t));

    if (xQueue != nullptr) {
        xTaskCreate(vSenderTask, "Sender Task", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);
        xTaskCreate(vReceiverTask, "Receiver Task", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);

        vTaskStartScheduler();
    }

    // Should never reach here
    while (true);
    return 0;
}
```

In this example, `vSenderTask` sends values to the queue, while `vReceiverTask` receives and processes these values. The queue handles the synchronization between the sender and receiver tasks.

#### 9.3.4. Event Groups

Event groups allow multiple tasks to synchronize based on the occurrence of multiple events. Tasks can wait for specific combinations of events to be set.

**Example: Using Event Groups**

```cpp
#include <FreeRTOS.h>
#include <task.h>
#include <event_groups.h>
#include <iostream>

EventGroupHandle_t xEventGroup;
const EventBits_t BIT_0 = (1 << 0);
const EventBits_t BIT_1 = (1 << 1);

void vTaskA(void* pvParameters) {
    while (true) {
        std::cout << "Task A setting bit 0\n";
        xEventGroupSetBits(xEventGroup, BIT_0);
        vTaskDelay(pdMS_TO_TICKS(1000)); // Delay to simulate work
    }
}

void vTaskB(void* pvParameters) {
    while (true) {
        std::cout << "Task B setting bit 1\n";
        xEventGroupSetBits(xEventGroup, BIT_1);
        vTaskDelay(pdMS_TO_TICKS(1500)); // Delay to simulate work
    }
}

void vTaskC(void* pvParameters) {
    while (true) {
        EventBits_t uxBits = xEventGroupWaitBits(xEventGroup, BIT_0 | BIT_1, pdTRUE, pdTRUE, portMAX_DELAY);
        if ((uxBits & (BIT_0 | BIT_1)) == (BIT_0 | BIT_1)) {
            std::cout << "Task C received both bits\n";
        }
    }
}

int main() {
    xEventGroup = xEventGroupCreate();

    if (xEventGroup != nullptr) {
        xTaskCreate(vTaskA, "Task A", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);
        xTaskCreate(vTaskB, "Task B", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);
        xTaskCreate(vTaskC, "Task C", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);

        vTaskStartScheduler();
    }

    // Should never reach here
    while (true);
    return 0;
}
```

In this example, `Task A` and `Task B` set different bits in an event group. `Task C` waits for both bits to be set before proceeding.

#### 9.3.5. Message Buffers and Stream Buffers

Message buffers and stream buffers are used for communication between tasks, especially when variable-length messages need to be transmitted.

**Example: Using Message Buffers**

```cpp
#include <FreeRTOS.h>
#include <task.h>
#include <message_buffer.h>
#include <iostream>
#include <string>

MessageBufferHandle_t xMessageBuffer;

void vSenderTask(void* pvParameters) {
    const char* pcMessageToSend = "Hello";
    while (true) {
        xMessageBufferSend(xMessageBuffer, (void*)pcMessageToSend, strlen(pcMessageToSend), portMAX_DELAY);
        std::cout << "Sent: " << pcMessageToSend << "\n";
        vTaskDelay(pdMS_TO_TICKS(1000)); // Delay to simulate work
    }
}

void vReceiverTask(void* pvParameters) {
    char pcReceivedMessage[100];
    size_t xReceivedBytes;
    while (true) {
        xReceivedBytes = xMessageBufferReceive(xMessageBuffer, (void*)pcReceivedMessage, sizeof(pcReceivedMessage), portMAX_DELAY);
        pcReceivedMessage[xReceivedBytes] = '\0'; // Null-terminate the received message
        std::cout << "Received: " << pcReceivedMessage << "\n";
    }
}

int main() {
    xMessageBuffer = xMessageBufferCreate(100);

    if (xMessageBuffer != nullptr) {
        xTaskCreate(vSenderTask, "Sender Task", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);
        xTaskCreate(vReceiverTask, "Receiver Task", configMINIMAL_STACK_SIZE, nullptr, tskIDLE_PRIORITY + 1, nullptr);

        vTaskStartScheduler();
    }

    // Should never reach here
    while (true);
    return 0;
}
```

In this example, `vSenderTask` sends messages to a message buffer, and `vReceiverTask` receives and processes these messages. Message buffers handle the synchronization and transmission of variable-length messages.

#### Summary

Synchronization and inter-task communication are fundamental aspects of real-time embedded systems. By using mutexes, semaphores, queues, event groups, and message buffers, you can effectively manage task interactions and ensure reliable system behavior. The examples provided in this subchapter demonstrate practical implementations of these mechanisms, offering a solid foundation for developing robust and efficient embedded applications with RTOS and C++.