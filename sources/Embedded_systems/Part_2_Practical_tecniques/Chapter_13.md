\newpage

## 13: **Workshops and Labs**

In this chapter, we transition from theoretical knowledge to practical application. Workshops and labs provide an invaluable opportunity to solidify your understanding of embedded systems through interactive, hands-on experiences. We will engage in real-time coding and problem-solving sessions, allowing you to tackle real-world challenges in a collaborative environment. Additionally, the hardware labs will offer you direct experience with microcontrollers, sensors, and actuators, bridging the gap between abstract concepts and tangible implementations. This chapter is designed to enhance your skills, foster creativity, and build confidence in your ability to develop and deploy embedded systems.

### 13.1. Interactive Sessions: Real-Time Coding and Problem-Solving

Interactive sessions are an essential part of learning embedded systems, as they provide an opportunity to apply theoretical knowledge in a practical setting. These sessions involve real-time coding and problem-solving, enabling you to work through challenges, debug issues, and optimize your code on the fly. This section will guide you through a series of exercises designed to reinforce your understanding of embedded C++ programming and its applications.

#### 1. Real-Time Coding Exercises

Real-time coding exercises help you practice writing code under simulated conditions that mimic real-world scenarios. Below are a few examples to get you started:

##### Example 1: Blinking LED with Timers

This exercise demonstrates how to use hardware timers to create a precise blinking LED without using the `delay()` function. This is crucial in embedded systems where efficient use of resources is necessary.

**Setup:**
- Microcontroller: Arduino Uno
- Component: LED connected to pin 13

**Code:**
```cpp
const int ledPin = 13; // LED connected to digital pin 13
volatile bool ledState = false;

void setup() {
    pinMode(ledPin, OUTPUT);

    // Configure Timer1 for a 1Hz (1 second) interval
    noInterrupts(); // Disable interrupts during configuration
    TCCR1A = 0; // Clear Timer1 control register A
    TCCR1B = 0; // Clear Timer1 control register B
    TCNT1 = 0; // Initialize counter value to 0

    // Set compare match register for 1Hz increments
    OCR1A = 15624; // (16*10^6) / (1*1024) - 1 (must be <65536)
    TCCR1B |= (1 << WGM12); // CTC mode
    TCCR1B |= (1 << CS12) | (1 << CS10); // 1024 prescaler
    TIMSK1 |= (1 << OCIE1A); // Enable Timer1 compare interrupt
    interrupts(); // Enable interrupts
}

ISR(TIMER1_COMPA_vect) {
    ledState = !ledState; // Toggle LED state
    digitalWrite(ledPin, ledState); // Update LED
}

void loop() {
    // Main loop does nothing, all action happens in ISR
}
```

##### Explanation:
- **Timer Configuration**: The timer is configured to trigger an interrupt every second.
- **ISR (Interrupt Service Routine)**: The ISR toggles the LED state, creating a blinking effect without using blocking functions like `delay()`.

##### Example 2: Reading Analog Sensors

This exercise demonstrates how to read analog values from a sensor and process the data.

**Setup:**
- Microcontroller: Arduino Uno
- Component: Potentiometer connected to analog pin A0

**Code:**
```cpp
const int sensorPin = A0; // Potentiometer connected to analog pin A0

void setup() {
    Serial.begin(9600); // Initialize serial communication at 9600 baud rate
}

void loop() {
    int sensorValue = analogRead(sensorPin); // Read the analog value
    float voltage = sensorValue * (5.0 / 1023.0); // Convert to voltage
    Serial.print("Sensor Value: ");
    Serial.print(sensorValue);
    Serial.print(" Voltage: ");
    Serial.println(voltage);
    delay(500); // Wait for 500 milliseconds
}
```

##### Explanation:
- **Analog Read**: The analog value from the potentiometer is read and converted to a voltage.
- **Serial Communication**: The sensor value and corresponding voltage are printed to the serial monitor for real-time observation.

#### 2. Problem-Solving Sessions

Problem-solving sessions are designed to challenge your understanding and push the boundaries of your knowledge. These exercises require you to identify, diagnose, and fix issues within the code or hardware setup.

##### Problem 1: Debouncing a Button

Buttons can produce noisy signals, causing multiple triggers. This problem involves writing code to debounce a button.

**Setup:**
- Microcontroller: Arduino Uno
- Component: Push button connected to digital pin 2

**Code:**
```cpp
const int buttonPin = 2; // Button connected to digital pin 2
const int ledPin = 13; // LED connected to digital pin 13

int buttonState = LOW; // Current state of the button
int lastButtonState = LOW; // Previous state of the button
unsigned long lastDebounceTime = 0; // The last time the output pin was toggled
unsigned long debounceDelay = 50; // Debounce time, increase if necessary

void setup() {
    pinMode(buttonPin, INPUT);
    pinMode(ledPin, OUTPUT);
    digitalWrite(ledPin, LOW);
}

void loop() {
    int reading = digitalRead(buttonPin);

    // If the button state has changed (due to noise or pressing)
    if (reading != lastButtonState) {
        lastDebounceTime = millis(); // reset the debouncing timer
    }

    if ((millis() - lastDebounceTime) > debounceDelay) {
        // If the button state has been stable for longer than the debounce delay
        if (reading != buttonState) {
            buttonState = reading;
            // Only toggle the LED if the new button state is HIGH
            if (buttonState == HIGH) {
                digitalWrite(ledPin, !digitalRead(ledPin));
            }
        }
    }

    // Save the reading. Next time through the loop, it'll be the lastButtonState
    lastButtonState = reading;
}
```

##### Explanation:
- **Debouncing Logic**: The code uses a debounce delay to filter out noise from the button press.
- **State Change Detection**: It checks if the button state has changed and if the change persists beyond the debounce delay.

##### Problem 2: Implementing a Finite State Machine

Design a simple finite state machine (FSM) to control an LED sequence based on button presses.

**Setup:**
- Microcontroller: Arduino Uno
- Components: Three LEDs connected to digital pins 9, 10, and 11; Button connected to digital pin 2

**Code:**
```cpp
enum State {STATE_OFF, STATE_RED, STATE_GREEN, STATE_BLUE};
State currentState = STATE_OFF;

const int buttonPin = 2; // Button connected to digital pin 2
const int redLedPin = 9; // Red LED connected to digital pin 9
const int greenLedPin = 10; // Green LED connected to digital pin 10
const int blueLedPin = 11; // Blue LED connected to digital pin 11

int buttonState = LOW;
int lastButtonState = LOW;
unsigned long lastDebounceTime = 0;
unsigned long debounceDelay = 50;

void setup() {
    pinMode(buttonPin, INPUT);
    pinMode(redLedPin, OUTPUT);
    pinMode(greenLedPin, OUTPUT);
    pinMode(blueLedPin, OUTPUT);

    digitalWrite(redLedPin, LOW);
    digitalWrite(greenLedPin, LOW);
    digitalWrite(blueLedPin, LOW);
}

void loop() {
    int reading = digitalRead(buttonPin);

    if (reading != lastButtonState) {
        lastDebounceTime = millis();
    }

    if ((millis() - lastDebounceTime) > debounceDelay) {
        if (reading != buttonState) {
            buttonState = reading;
            if (buttonState == HIGH) {
                switch (currentState) {
                    case STATE_OFF:
                        currentState = STATE_RED;
                        break;
                    case STATE_RED:
                        currentState = STATE_GREEN;
                        break;
                    case STATE_GREEN:
                        currentState = STATE_BLUE;
                        break;
                    case STATE_BLUE:
                        currentState = STATE_OFF;
                        break;
                }
            }
        }
    }

    lastButtonState = reading;

    // Update LEDs based on the current state
    switch (currentState) {
        case STATE_OFF:
            digitalWrite(redLedPin, LOW);
            digitalWrite(greenLedPin, LOW);
            digitalWrite(blueLedPin, LOW);
            break;
        case STATE_RED:
            digitalWrite(redLedPin, HIGH);
            digitalWrite(greenLedPin, LOW);
            digitalWrite(blueLedPin, LOW);
            break;
        case STATE_GREEN:
            digitalWrite(redLedPin, LOW);
            digitalWrite(greenLedPin, HIGH);
            digitalWrite(blueLedPin, LOW);
            break;
        case STATE_BLUE:
            digitalWrite(redLedPin, LOW);
            digitalWrite(greenLedPin, LOW);
            digitalWrite(blueLedPin, HIGH);
            break;
    }
}
```

##### Explanation:
- **State Management**: The FSM manages the LED states based on button presses.
- **Debouncing**: The button input is debounced to ensure reliable state transitions.

#### Conclusion

Interactive sessions are a crucial component of learning embedded systems, providing practical experience in real-time coding and problem-solving. By engaging in these exercises, you develop a deeper understanding of how to implement and troubleshoot embedded C++ programs. The examples provided in this section serve as a foundation for more complex projects and real-world applications, enhancing your skills and confidence in embedded systems development.

### 13.2. Hardware Labs: Hands-On Experience with Microcontrollers, Sensors, and Actuators

Hardware labs provide an invaluable opportunity to gain practical experience with microcontrollers, sensors, and actuators. These hands-on sessions enable you to apply theoretical knowledge, develop hardware interfacing skills, and understand the intricacies of embedded systems. This section will guide you through several hardware lab exercises designed to help you master the integration and programming of various components.

#### 1. Introduction to Microcontrollers

Microcontrollers are the heart of embedded systems. In these labs, you will work with popular microcontroller platforms such as Arduino, ESP8266, and STM32. The focus will be on understanding pin configurations, setting up development environments, and writing basic programs.

##### Lab 1: Blinking LED

**Objective**: Learn to configure and control a digital output pin.

**Setup**:
- Microcontroller: Arduino Uno
- Component: LED connected to digital pin 13

**Code**:
```cpp
void setup() {
    pinMode(13, OUTPUT); // Set pin 13 as an output
}

void loop() {
    digitalWrite(13, HIGH); // Turn the LED on
    delay(1000); // Wait for 1 second
    digitalWrite(13, LOW); // Turn the LED off
    delay(1000); // Wait for 1 second
}
```

##### Explanation:
- **pinMode()**: Configures the specified pin to behave either as an input or an output.
- **digitalWrite()**: Sets the output voltage of a digital pin to HIGH or LOW.
- **delay()**: Pauses the program for a specified duration (milliseconds).

#### 2. Interfacing with Sensors

Sensors allow microcontrollers to interact with the physical world by measuring various parameters such as temperature, humidity, light, and motion.

##### Lab 2: Reading Temperature and Humidity

**Objective**: Interface with a DHT11 sensor to read temperature and humidity data.

**Setup**:
- Microcontroller: Arduino Uno
- Component: DHT11 sensor connected to digital pin 2

**Code**:
```cpp
#include <DHT.h>

#define DHTPIN 2 // Digital pin connected to the DHT sensor

#define DHTTYPE DHT11 // DHT11 sensor type

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
    Serial.print("%  Temperature: ");
    Serial.print(temperature);
    Serial.println("°C");

    delay(2000); // Wait for 2 seconds before next read
}
```

##### Explanation:
- **DHT Library**: A library specifically for reading from DHT sensors.
- **Serial Communication**: Used to send data to the computer for display on the serial monitor.

#### 3. Controlling Actuators

Actuators convert electrical signals into physical actions. Common actuators include motors, relays, and servos.

##### Lab 3: Controlling a Servo Motor

**Objective**: Interface with a servo motor and control its position.

**Setup**:
- Microcontroller: Arduino Uno
- Component: Servo motor connected to digital pin 9

**Code**:
```cpp
#include <Servo.h>

Servo myservo; // Create servo object

void setup() {
    myservo.attach(9); // Attach servo to pin 9
}

void loop() {
    myservo.write(0); // Set servo to 0 degrees
    delay(1000); // Wait for 1 second

    myservo.write(90); // Set servo to 90 degrees
    delay(1000); // Wait for 1 second

    myservo.write(180); // Set servo to 180 degrees
    delay(1000); // Wait for 1 second
}
```

##### Explanation:
- **Servo Library**: Provides an easy interface for controlling servo motors.
- **write() Method**: Sets the position of the servo.

#### 4. Building Integrated Systems

In this lab, you will combine sensors and actuators to build an integrated system.

##### Lab 4: Automatic Light Control

**Objective**: Build a system that turns on an LED when the ambient light level drops below a certain threshold.

**Setup**:
- Microcontroller: Arduino Uno
- Components: Photoresistor (light sensor) connected to analog pin A0, LED connected to digital pin 13

**Code**:
```cpp
const int sensorPin = A0; // Photoresistor connected to analog pin A0
const int ledPin = 13; // LED connected to digital pin 13
const int threshold = 500; // Light threshold

void setup() {
    pinMode(ledPin, OUTPUT); // Set pin 13 as an output
    Serial.begin(9600); // Initialize serial communication
}

void loop() {
    int sensorValue = analogRead(sensorPin); // Read the analog value
    Serial.print("Sensor Value: ");
    Serial.println(sensorValue);

    if (sensorValue < threshold) {
        digitalWrite(ledPin, HIGH); // Turn the LED on
    } else {
        digitalWrite(ledPin, LOW); // Turn the LED off
    }

    delay(500); // Wait for 500 milliseconds
}
```

##### Explanation:
- **Analog Read**: Reads the voltage level from the photoresistor.
- **Threshold Comparison**: Turns the LED on or off based on the light level.

#### 5. Advanced Hardware Labs

Advanced labs involve more complex integrations and use of additional hardware interfaces such as I2C, SPI, and UART.

##### Lab 5: I2C Communication with an LCD Display

**Objective**: Display sensor data on an I2C LCD display.

**Setup**:
- Microcontroller: Arduino Uno
- Components: I2C LCD display connected to SDA and SCL pins, DHT11 sensor connected to digital pin 2

**Code**:
```cpp
#include <Wire.h>

#include <LiquidCrystal_I2C.h>
#include <DHT.h>

#define DHTPIN 2

#define DHTTYPE DHT11

DHT dht(DHTPIN, DHTTYPE);
LiquidCrystal_I2C lcd(0x27, 16, 2); // Set the LCD address to 0x27 for a 16 chars and 2 line display

void setup() {
    dht.begin();
    lcd.init(); // Initialize the LCD
    lcd.backlight(); // Turn on the backlight
}

void loop() {
    float humidity = dht.readHumidity();
    float temperature = dht.readTemperature();

    if (isnan(humidity) || isnan(temperature)) {
        lcd.setCursor(0, 0);
        lcd.print("Read error");
        return;
    }

    lcd.setCursor(0, 0);
    lcd.print("Temp: ");
    lcd.print(temperature);
    lcd.print(" C");

    lcd.setCursor(0, 1);
    lcd.print("Humidity: ");
    lcd.print(humidity);
    lcd.print(" %");

    delay(2000);
}
```

##### Explanation:
- **I2C Communication**: Uses the I2C protocol to communicate with the LCD.
- **LiquidCrystal_I2C Library**: Simplifies interfacing with I2C LCD displays.

#### Conclusion

Hands-on hardware labs are crucial for mastering embedded systems. They provide practical experience with microcontrollers, sensors, and actuators, reinforcing theoretical concepts through real-world applications. The examples in this section are designed to build your confidence and proficiency in developing embedded systems, preparing you for more complex projects and professional challenges.

