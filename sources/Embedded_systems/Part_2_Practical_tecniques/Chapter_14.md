\newpage


## 14. Case Studies and Examples

Chapter 14 delves into practical applications of the principles and techniques discussed throughout this book by presenting detailed case studies and examples. These real-world scenarios illustrate the development and optimization of embedded systems using C++. We begin with a comprehensive guide to developing a miniature operating system, providing a step-by-step approach to building a small-scale real-time operating system (RTOS). Next, we explore the creation of a smart sensor node, demonstrating how to integrate sensors, process data, and establish network communication. Finally, we tackle the performance optimization of an existing embedded application, showcasing strategies for enhancing efficiency and responsiveness. Through these case studies, you will gain valuable insights and hands-on experience in embedded systems development.

### 14.1. Developing a Miniature Operating System

Developing a miniature operating system (OS) for embedded systems can seem daunting, but with a structured approach, it becomes an achievable task. This subchapter provides a step-by-step guide to building a small-scale real-time operating system (RTOS) in C++. We will cover the fundamental components of an RTOS, including task scheduling, context switching, inter-task communication, and system initialization. Each section includes detailed code examples to illustrate key concepts.

#### Introduction to RTOS

A real-time operating system (RTOS) is designed to handle tasks with precise timing requirements. Unlike general-purpose operating systems, an RTOS ensures that high-priority tasks are executed predictably and within specified time constraints. The core components of an RTOS include:

- Task management and scheduling
- Context switching
- Inter-task communication
- System initialization and configuration

#### Task Management and Scheduling

Task management involves creating, deleting, and managing tasks. Scheduling determines the order in which tasks are executed. A common scheduling algorithm in RTOS is round-robin scheduling, where each task gets an equal share of the CPU time.

**Task Structure**

Each task is typically represented by a task control block (TCB), which contains information about the task's state, stack, and priority.

```cpp
#include <cstdint>
#include <vector>

enum class TaskState {
    Ready,
    Running,
    Blocked,
    Suspended
};

struct TaskControlBlock {
    void (*taskFunction)();
    TaskState state;
    uint32_t* stackPointer;
    uint32_t priority;
};

std::vector<TaskControlBlock> taskList;
```

**Task Creation**

Tasks are created by defining a function and adding a corresponding TCB to the task list.

```cpp
void createTask(void (*taskFunction)(), uint32_t* stackPointer, uint32_t priority) {
    TaskControlBlock tcb = {taskFunction, TaskState::Ready, stackPointer, priority};
    taskList.push_back(tcb);
}

void task1() {
    while (true) {
        // Task 1 code
    }
}

void task2() {
    while (true) {
        // Task 2 code
    }
}

int main() {
    uint32_t stack1[256];
    uint32_t stack2[256];

    createTask(task1, stack1, 1);
    createTask(task2, stack2, 2);

    // Initialize and start the RTOS scheduler
    startScheduler();

    return 0;
}
```

**Task Scheduling**

A simple round-robin scheduler can be implemented by iterating through the task list and selecting the next ready task to run.

```cpp
void startScheduler() {
    while (true) {
        for (auto& task : taskList) {
            if (task.state == TaskState::Ready) {
                task.state = TaskState::Running;
                task.taskFunction();
                task.state = TaskState::Ready;
            }
        }
    }
}
```

#### Context Switching

Context switching involves saving the state of the currently running task and restoring the state of the next task. This allows multiple tasks to share the CPU effectively.

**Saving and Restoring Context**

The context of a task includes the CPU registers and stack pointer. These need to be saved and restored during a context switch.

```cpp
struct Context {
    uint32_t registers[16]; // General-purpose registers
    uint32_t* stackPointer;
};

void saveContext(Context& context) {
    // Save the current task's context
    asm volatile ("MRS %0, PSP" : "=r" (context.stackPointer) : : );
    for (int i = 0; i < 16; ++i) {
        asm volatile ("STR r%0, [%1, #%2]" : : "r" (i), "r" (context.stackPointer), "r" (i * 4) : );
    }
}

void restoreContext(const Context& context) {
    // Restore the next task's context
    for (int i = 0; i < 16; ++i) {
        asm volatile ("LDR r%0, [%1, #%2]" : : "r" (i), "r" (context.stackPointer), "r" (i * 4) : );
    }
    asm volatile ("MSR PSP, %0" : : "r" (context.stackPointer) : );
}
```

**Implementing Context Switch**

A context switch is triggered by an interrupt, such as a timer interrupt. The interrupt service routine (ISR) saves the current task's context, selects the next task, and restores its context.

```cpp
extern "C" void SysTick_Handler() {
    // Save the context of the current task
    saveContext(currentTask.context);

    // Select the next task to run
    currentTask = selectNextTask();

    // Restore the context of the next task
    restoreContext(currentTask.context);
}

TaskControlBlock& selectNextTask() {
    static size_t currentIndex = 0;
    currentIndex = (currentIndex + 1) % taskList.size();
    return taskList[currentIndex];
}
```

#### Inter-Task Communication

Inter-task communication (ITC) is essential for tasks to coordinate and share data. Common ITC mechanisms in RTOS include queues, semaphores, and mutexes.

**Message Queues**

Message queues allow tasks to send and receive messages in a FIFO manner, providing a way to communicate safely between tasks.

```cpp
#include <queue>

struct Message {
    int id;
    std::string data;
};

class MessageQueue {
public:
    void send(const Message& message) {
        queue.push(message);
    }

    bool receive(Message& message) {
        if (queue.empty()) return false;
        message = queue.front();
        queue.pop();
        return true;
    }

private:
    std::queue<Message> queue;
};

MessageQueue messageQueue;

void senderTask() {
    while (true) {
        Message msg = {1, "Hello"};
        messageQueue.send(msg);
        delay(1000); // Simulate work
    }
}

void receiverTask() {
    while (true) {
        Message msg;
        if (messageQueue.receive(msg)) {
            // Process the message
        }
        delay(100); // Simulate work
    }
}
```

**Semaphores**

Semaphores are signaling mechanisms to synchronize tasks. They can be used to manage access to shared resources.

```cpp
#include <atomic>

class Semaphore {
public:
    Semaphore(int count = 0) : count(count) {}

    void signal() {
        ++count;
    }

    void wait() {
        while (count == 0) {
            // Busy-wait (can be replaced with a more efficient waiting mechanism)
        }
        --count;
    }

private:
    std::atomic<int> count;
};

Semaphore semaphore;

void taskA() {
    while (true) {
        semaphore.wait();
        // Access shared resource
        semaphore.signal();
        delay(1000);
    }
}

void taskB() {
    while (true) {
        semaphore.wait();
        // Access shared resource
        semaphore.signal();
        delay(1000);
    }
}
```

#### System Initialization

System initialization involves setting up the hardware, configuring system parameters, and initializing the RTOS components before starting task execution.

**Hardware Initialization**

Initialize hardware components such as clocks, GPIOs, and peripherals.

```cpp
void initHardware() {
    // Initialize system clock
    SystemClock_Config();

    // Initialize GPIOs
    initGPIO();
}

void SystemClock_Config() {
    // Configure system clock
}

void initGPIO() {
    // Configure GPIO pins
}
```

**RTOS Initialization**

Initialize RTOS components such as tasks, scheduler, and ITC mechanisms.

```cpp
void initRTOS() {
    // Create tasks
    uint32_t stack1[256];
    uint32_t stack2[256];
    createTask(senderTask, stack1, 1);
    createTask(receiverTask, stack2, 2);

    // Initialize scheduler
    initScheduler();

    // Start the scheduler
    startScheduler();
}

void initScheduler() {
    // Configure and start the system tick timer
    SysTick_Config(SystemCoreClock / 1000);
}

int main() {
    // Initialize hardware
    initHardware();

    // Initialize RTOS
    initRTOS();

    while (true) {
        // Main loop
    }
}
```

#### Conclusion

Building a miniature operating system for embedded systems involves understanding and implementing key RTOS components, including task management, context switching, inter-task communication, and system initialization. By following the structured approach and code examples provided in this subchapter, you can develop a small-scale RTOS that meets the real-time requirements of embedded applications. This foundational knowledge will also prepare you for more advanced RTOS development and customization in your future embedded system projects.

### 14.2. Building a Smart Sensor Node

Smart sensor nodes are integral components of modern embedded systems, particularly in the Internet of Things (IoT) landscape. They collect data from the environment, process it, and communicate the results over a network. This subchapter provides a comprehensive guide to building a smart sensor node, covering sensor integration, data processing, and network communication. Each section includes detailed code examples in C++ to illustrate key concepts.

#### Introduction to Smart Sensor Nodes

A smart sensor node typically consists of the following components:

1. **Sensors**: Devices that detect and measure physical properties (e.g., temperature, humidity, light).
2. **Microcontroller**: The brain of the node, responsible for data acquisition, processing, and communication.
3. **Communication Module**: Enables the node to send and receive data over a network (e.g., Wi-Fi, Zigbee).
4. **Power Management**: Ensures efficient power usage, crucial for battery-operated nodes.

#### Sensor Integration

Integrating sensors involves interfacing them with the microcontroller, reading sensor data, and converting it into a usable format.

**Connecting a Temperature Sensor**

We will use a common temperature sensor, the LM35, which provides an analog output proportional to the temperature.

```cpp
#include <Arduino.h>

const int sensorPin = A0; // Analog pin where the sensor is connected

void setup() {
    Serial.begin(9600); // Initialize serial communication
    pinMode(sensorPin, INPUT); // Set the sensor pin as input
}

void loop() {
    int sensorValue = analogRead(sensorPin); // Read the analog value
    float temperature = sensorValue * (5.0 / 1023.0) * 100.0; // Convert to Celsius
    Serial.print("Temperature: ");
    Serial.print(temperature);
    Serial.println(" °C");
    delay(1000); // Wait for 1 second before the next reading
}
```

In this example, we read the analog value from the LM35 sensor and convert it to a temperature in Celsius.

**Connecting a Humidity Sensor**

Similarly, we can connect a humidity sensor like the DHT11, which uses a digital interface.

```cpp
#include <DHT.h>

#define DHTPIN 2 // Digital pin where the sensor is connected
#define DHTTYPE DHT11 // DHT 11

DHT dht(DHTPIN, DHTTYPE);

void setup() {
    Serial.begin(9600); // Initialize serial communication
    dht.begin(); // Initialize the sensor
}

void loop() {
    float humidity = dht.readHumidity(); // Read humidity
    float temperature = dht.readTemperature(); // Read temperature in Celsius

    if (isnan(humidity) || isnan(temperature)) {
        Serial.println("Failed to read from DHT sensor!");
        return;
    }

    Serial.print("Humidity: ");
    Serial.print(humidity);
    Serial.print(" %\t");
    Serial.print("Temperature: ");
    Serial.print(temperature);
    Serial.println(" °C");
    delay(2000); // Wait for 2 seconds before the next reading
}
```

In this example, we use the DHT library to read humidity and temperature data from the DHT11 sensor.

#### Data Processing

Once the sensor data is acquired, it often needs to be processed before being sent over the network. This processing can include filtering, averaging, and unit conversions.

**Averaging Sensor Readings**

To reduce noise in sensor readings, we can average multiple samples.

```cpp
const int numSamples = 10;
float readAverageTemperature() {
    int total = 0;
    for (int i = 0; i < numSamples; ++i) {
        total += analogRead(sensorPin);
        delay(10); // Small delay between samples
    }
    float average = total / numSamples;
    float temperature = average * (5.0 / 1023.0) * 100.0;
    return temperature;
}

void loop() {
    float temperature = readAverageTemperature();
    Serial.print("Average Temperature: ");
    Serial.print(temperature);
    Serial.println(" °C");
    delay(1000);
}
```

This function reads the temperature sensor multiple times and calculates the average to produce a more stable reading.

**Filtering Data**

A simple moving average filter can smooth out fluctuations in sensor data.

```cpp
#include <deque>

const int windowSize = 5;
std::deque<float> temperatureWindow;

float readFilteredTemperature() {
    float temperature = analogRead(sensorPin) * (5.0 / 1023.0) * 100.0;
    if (temperatureWindow.size() >= windowSize) {
        temperatureWindow.pop_front();
    }
    temperatureWindow.push_back(temperature);

    float sum = 0.0;
    for (float temp : temperatureWindow) {
        sum += temp;
    }
    return sum / temperatureWindow.size();
}

void loop() {
    float temperature = readFilteredTemperature();
    Serial.print("Filtered Temperature: ");
    Serial.print(temperature);
    Serial.println(" °C");
    delay(1000);
}
```

This function implements a simple moving average filter to smooth the temperature readings.

#### Network Communication

Communicating sensor data over a network involves configuring a communication module and transmitting the processed data. We'll use Wi-Fi for this example.

**Configuring Wi-Fi Communication**

We'll use the ESP8266 Wi-Fi module to send data to a remote server.

```cpp
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>

const char* ssid = "your_SSID";
const char* password = "your_PASSWORD";
const char* serverUrl = "http://example.com/post-data";

void setup() {
    Serial.begin(9600);
    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.println("Connecting to WiFi...");
    }

    Serial.println("Connected to WiFi");
}

void loop() {
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.begin(serverUrl);
        http.addHeader("Content-Type", "application/json");

        float temperature = readAverageTemperature(); // Using the previously defined function
        String postData = "{\"temperature\": " + String(temperature) + "}";

        int httpResponseCode = http.POST(postData);
        if (httpResponseCode > 0) {
            String response = http.getString();
            Serial.println(httpResponseCode);
            Serial.println(response);
        } else {
            Serial.println("Error in sending POST request");
        }
        http.end();
    }
    delay(10000); // Send data every 10 seconds
}
```

In this example, we connect to a Wi-Fi network and send the average temperature data to a remote server as a JSON payload.

**Sending Data Securely**

For secure data transmission, we can use HTTPS instead of HTTP.

```cpp
#include <WiFiClientSecure.h>

const char* serverUrl = "https://example.com/post-data";
const int httpsPort = 443;
const char* fingerprint = "XX XX XX XX XX XX XX XX XX XX XX XX XX XX XX XX XX XX XX XX"; // Server certificate fingerprint

void loop() {
    if (WiFi.status() == WL_CONNECTED) {
        WiFiClientSecure client;
        client.setFingerprint(fingerprint);

        if (!client.connect(serverUrl, httpsPort)) {
            Serial.println("Connection failed");
            return;
        }

        String postData = "{\"temperature\": " + String(readAverageTemperature()) + "}";

        client.println("POST /post-data HTTP/1.1");
        client.println("Host: example.com");
        client.println("User-Agent: ESP8266");
        client.println("Content-Type: application/json");
        client.print("Content-Length: ");
        client.println(postData.length());
        client.println();
        client.println(postData);

        while (client.connected()) {
            String line = client.readStringUntil('\n');
            if (line == "\r") {
                break;
            }
        }
        String response = client.readStringUntil('\n');
        Serial.println(response);
    }
    delay(10000); // Send data every 10 seconds
}
```

In this example, we use the WiFiClientSecure library to establish a secure connection and send data over HTTPS.

#### Power Management

Efficient power management is crucial for battery-operated smart sensor nodes. Implementing sleep modes and optimizing power consumption can significantly extend battery life.

**Using Sleep Modes**

Microcontrollers often support various sleep modes to reduce power consumption when idle.

```cpp
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>

void setup() {
    Serial.begin(9600);
    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.println("Connecting to WiFi...");
    }

    Serial.println("Connected to WiFi");
}

void deepSleepSetup() {
    ESP.deepSleep(10e6); // Sleep for 10 seconds
}

void loop() {
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.begin(serverUrl);
        http.addHeader("Content-Type", "application/json");

        float temperature = readAverageTemperature();
        String postData = "{\"temperature\": " + String(temperature) + "}";

        int httpResponseCode = http.POST(postData);
        if (httpResponseCode > 0) {
            String response = http.getString();
            Serial.println(httpResponseCode);
            Serial.println(response);
        } else {
            Serial.println("Error in sending POST request");
        }
        http.end();
    }
    deepSleepSetup();
}
```

In this example, the

ESP8266 is put into deep sleep mode for 10 seconds after sending data, significantly reducing power consumption.

#### Conclusion

Building a smart sensor node involves integrating sensors, processing data, and communicating over a network while managing power efficiently. This subchapter has provided a detailed guide with code examples to help you build a functional smart sensor node. By understanding and applying these concepts, you can create robust and efficient sensor nodes for various embedded and IoT applications.

### 14.3. Performance Optimization of an Embedded Application

Performance optimization is a critical aspect of embedded systems development, ensuring that applications run efficiently within the constraints of limited resources. This subchapter explores various strategies and techniques for optimizing the performance of embedded applications, covering profiling and analysis, code optimization, memory management, and power efficiency. Detailed code examples illustrate the practical application of these techniques in C++.

#### Introduction to Performance Optimization

Optimizing the performance of an embedded application involves improving execution speed, reducing memory usage, and enhancing power efficiency. The process typically includes the following steps:

1. **Profiling and Analysis**: Identifying performance bottlenecks through profiling tools and analysis techniques.
2. **Code Optimization**: Refining code to improve execution speed and efficiency.
3. **Memory Management**: Efficiently managing memory allocation and usage.
4. **Power Efficiency**: Reducing power consumption through various optimization strategies.

#### Profiling and Analysis

Profiling helps identify parts of the code that consume the most resources. Tools like `gprof`, `Valgrind`, and built-in MCU profilers can be used to gather performance data.

**Using gprof**

To profile an embedded application using `gprof`, compile the code with profiling enabled and run the profiler.

```sh
# Compile with profiling enabled
g++ -pg -o my_app my_app.cpp

# Run the application to generate profile data
./my_app

# Analyze the profile data
gprof my_app gmon.out > analysis.txt
```

**Analyzing the Output**

The `gprof` output shows the time spent in each function, helping identify performance bottlenecks.

```plaintext
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total
 time   seconds   seconds    calls  ms/call  ms/call  name
  40.00      0.04     0.04       10     4.00     4.00  processData
  20.00      0.06     0.02      100     0.20     0.20  readSensor
  10.00      0.07     0.01       50     0.20     0.20  sendData
```

From this output, `processData` is the most time-consuming function, indicating a potential area for optimization.

#### Code Optimization

Code optimization involves refining algorithms and code structures to enhance performance. Techniques include loop unrolling, minimizing function calls, and using efficient data structures.

**Loop Unrolling**

Loop unrolling reduces the overhead of loop control by increasing the loop's body size. This technique can improve performance, especially in time-critical sections.

```cpp
// Original loop
for (int i = 0; i < 100; ++i) {
    processElement(i);
}

// Unrolled loop
for (int i = 0; i < 100; i += 4) {
    processElement(i);
    processElement(i + 1);
    processElement(i + 2);
    processElement(i + 3);
}
```

**Minimizing Function Calls**

Reducing the number of function calls, especially in deeply nested loops, can significantly improve performance.

```cpp
// Original code
for (int i = 0; i < 1000; ++i) {
    readSensor();
    processData();
    sendData();
}

// Optimized code
void readProcessSend() {
    for (int i = 0; i < 1000; ++i) {
        readSensor();
        processData();
        sendData();
    }
}

readProcessSend();
```

**Using Efficient Data Structures**

Choosing the right data structure can greatly impact performance. For example, using a `std::vector` instead of a linked list can improve cache performance and reduce overhead.

```cpp
#include <vector>

// Original code using linked list
std::list<int> dataList;
for (int i = 0; i < 1000; ++i) {
    dataList.push_back(i);
}

// Optimized code using vector
std::vector<int> dataVector;
dataVector.reserve(1000); // Reserve memory to avoid reallocations
for (int i = 0; i < 1000; ++i) {
    dataVector.push_back(i);
}
```

#### Memory Management

Efficient memory management is crucial in embedded systems to prevent fragmentation and optimize usage. Techniques include using fixed-size allocations and avoiding dynamic memory allocation where possible.

**Using Fixed-Size Allocations**

Fixed-size allocations can prevent fragmentation and make memory management more predictable.

```cpp
class FixedSizeAllocator {
public:
    FixedSizeAllocator(size_t size) : poolSize(size), pool(new char[size]), freeList(pool) {
        // Initialize free list
        for (size_t i = 0; i < poolSize - blockSize; i += blockSize) {
            *reinterpret_cast<void**>(pool + i) = pool + i + blockSize;
        }
        *reinterpret_cast<void**>(pool + poolSize - blockSize) = nullptr;
    }

    void* allocate() {
        if (!freeList) return nullptr;
        void* block = freeList;
        freeList = *reinterpret_cast<void**>(freeList);
        return block;
    }

    void deallocate(void* block) {
        *reinterpret_cast<void**>(block) = freeList;
        freeList = block;
    }

private:
    const size_t blockSize = 32;
    size_t poolSize;
    char* pool;
    void* freeList;
};

// Usage
FixedSizeAllocator allocator(1024);
void* block = allocator.allocate();
allocator.deallocate(block);
```

**Avoiding Dynamic Memory Allocation**

Minimize the use of dynamic memory allocation, which can lead to fragmentation and unpredictable behavior.

```cpp
// Avoid using dynamic allocation
int* dynamicArray = new int[100];

// Prefer static allocation
int staticArray[100];
```

#### Power Efficiency

Optimizing power efficiency is vital for battery-operated embedded systems. Techniques include using low-power modes and optimizing peripheral usage.

**Using Low-Power Modes**

Microcontrollers often have low-power modes that can significantly reduce power consumption when the system is idle.

```cpp
#include <avr/sleep.h>

void setup() {
    // Setup code
}

void loop() {
    // Enter sleep mode
    set_sleep_mode(SLEEP_MODE_PWR_DOWN);
    sleep_enable();
    sleep_mode();

    // Wake up here after an interrupt
    sleep_disable();
}
```

**Optimizing Peripheral Usage**

Disabling unused peripherals can save power. Ensure that peripherals are only powered when needed.

```cpp
void disableUnusedPeripherals() {
    PRR |= (1 << PRADC);  // Disable ADC
    PRR |= (1 << PRUSART0); // Disable USART0
}

void setup() {
    disableUnusedPeripherals();
    // Other setup code
}
```

**Optimizing Communication**

Optimizing network communication can save power by reducing the time the communication module is active.

```cpp
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>

void setup() {
    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
    }
}

void sendData() {
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.begin(serverUrl);
        http.addHeader("Content-Type", "application/json");

        String postData = "{\"temperature\": " + String(readAverageTemperature()) + "}";
        http.POST(postData);
        http.end();
    }
}

void loop() {
    sendData();
    delay(60000); // Send data every 60 seconds
}
```

In this example, the ESP8266 module connects to Wi-Fi and sends data every 60 seconds, minimizing the active communication time.

#### Conclusion

Performance optimization in embedded applications involves a combination of profiling and analysis, code refinement, efficient memory management, and power-saving techniques. By systematically identifying bottlenecks and applying the strategies discussed in this subchapter, you can enhance the performance and efficiency of your embedded systems. The detailed code examples provided serve as practical guides to implementing these optimization techniques in real-world applications.

