\newpage

# **Part IV: Practical Applications and Projects**

## 12. **Interfacing with Hardware**

Chapter 12 delves into the practical applications of Assembly Language and ARM Architecture by exploring how to control and communicate with various hardware components. This chapter begins with an overview of Input/Output (I/O) operations, demonstrating how to interact with and control hardware peripherals such as sensors and actuators. It then progresses to Peripheral Programming, where you will learn to utilize essential peripherals like timers and Analog-to-Digital Converters (ADCs), enhancing the functionality of your embedded systems. The concept of Memory-Mapped I/O is introduced, providing a direct method to access and manipulate hardware devices. To consolidate your understanding, the chapter concludes with a comprehensive example that integrates all these elements, offering a step-by-step explanation to solidify your knowledge and practical skills in hardware interfacing using Assembly Language and ARM Architecture.

### Input/Output Operations

Interfacing with hardware components is a fundamental aspect of embedded systems programming, where the ability to control and communicate with external devices is paramount. In this subchapter, we will explore Input/Output (I/O) operations in detail, focusing on how to control hardware components using Assembly Language on ARM architecture. This will include the concepts of digital I/O, analog I/O, serial communication, and interrupt-driven I/O. Each section will provide a thorough examination of the underlying principles, programming techniques, and practical examples to solidify your understanding.

#### **1. Digital Input/Output (I/O)**

Digital I/O is the simplest form of interfacing, where the state of a pin is either high (logic 1) or low (logic 0). In ARM microcontrollers, each pin can be configured as an input or an output, and can be controlled or read accordingly.

**1.1. Configuring Digital I/O Pins**

Before using a pin for digital I/O operations, it must be configured. This involves setting the pin mode (input or output) and the state (high or low for output, read for input). ARM microcontrollers typically have dedicated registers for this purpose, such as the GPIO (General Purpose Input/Output) registers.

Example: Configuring a pin as output
```assembly
LDR R0, =0x48000000  ; Load base address of GPIO port
LDR R1, [R0, #0x00]  ; Load the current configuration
ORR R1, R1, #0x01    ; Set pin 0 as output
STR R1, [R0, #0x00]  ; Store the new configuration
```

**1.2. Writing to Digital Output Pins**

Once a pin is configured as an output, you can set its state to high or low by writing to the appropriate data register.

Example: Setting a pin high
```assembly
LDR R0, =0x48000014  ; Load address of GPIO data register
LDR R1, =0x01        ; Set pin 0 high
STR R1, [R0]         ; Write to the data register
```

**1.3. Reading from Digital Input Pins**

For input pins, you read the pin state from the appropriate data register.

Example: Reading a pin state
```assembly
LDR R0, =0x48000010  ; Load address of GPIO input data register
LDR R1, [R0]         ; Read the pin states
AND R2, R1, #0x01    ; Isolate the state of pin 0
```

#### **2. Analog Input/Output (I/O)**

Analog I/O operations involve interfacing with devices that output or accept continuous signals. This typically involves Analog-to-Digital Converters (ADC) and Digital-to-Analog Converters (DAC).

**2.1. Analog-to-Digital Conversion (ADC)**

ADCs convert analog signals into digital values. In ARM microcontrollers, the ADC peripheral must be configured before use.

**2.1.1. Configuring the ADC**

To configure the ADC, you need to set the reference voltage, resolution, and input channel.

Example: Configuring the ADC
```assembly
LDR R0, =0x50000000  ; Load base address of ADC
LDR R1, [R0, #0x00]  ; Load current configuration
ORR R1, R1, #0x01    ; Enable the ADC
STR R1, [R0, #0x00]  ; Store the new configuration
```

**2.1.2. Reading from the ADC**

After configuration, you can start a conversion and read the result from the ADC data register.

Example: Reading an analog value
```assembly
LDR R0, =0x50000004  ; Load address of ADC start register
MOV R1, #0x01        ; Start conversion on channel 0
STR R1, [R0]         ; Write to start register

LDR R2, =0x50000008  ; Load address of ADC data register
WAIT_LOOP:           ; Wait for conversion to complete
    LDR R3, [R2]
    TST R3, #0x8000  ; Check if conversion complete
    BEQ WAIT_LOOP    ; If not, wait

MOV R3, R3, LSR #16  ; Extract the ADC result
```

**2.2. Digital-to-Analog Conversion (DAC)**

DACs convert digital values into analog signals. Similar to ADCs, DACs must be configured before use.

**2.2.1. Configuring the DAC**

To configure the DAC, you need to set the reference voltage and enable the output channel.

Example: Configuring the DAC
```assembly
LDR R0, =0x50010000  ; Load base address of DAC
LDR R1, [R0, #0x00]  ; Load current configuration
ORR R1, R1, #0x01    ; Enable the DAC
STR R1, [R0, #0x00]  ; Store the new configuration
```

**2.2.2. Writing to the DAC**

After configuration, you can write a digital value to be converted to an analog signal.

Example: Writing an analog value
```assembly
LDR R0, =0x50010008  ; Load address of DAC data register
MOV R1, #0x0FFF      ; Write maximum value for 12-bit DAC
STR R1, [R0]         ; Write to the data register
```

#### **3. Serial Communication**

Serial communication involves transmitting data one bit at a time over a communication channel. Common protocols include UART (Universal Asynchronous Receiver/Transmitter), SPI (Serial Peripheral Interface), and I2C (Inter-Integrated Circuit).

**3.1. UART Communication**

UART is a widely used protocol for serial communication between devices. It requires configuration of the baud rate, data bits, parity, and stop bits.

**3.1.1. Configuring UART**

To configure UART, set the baud rate and other communication parameters.

Example: Configuring UART
```assembly
LDR R0, =0x40004000  ; Load base address of UART
LDR R1, =0x960       ; Set baud rate (assuming 16 MHz clock)
STR R1, [R0, #0x24]  ; Write to the baud rate register
LDR R1, [R0, #0x0C]  ; Load current configuration
ORR R1, R1, #0x301   ; 8 data bits, no parity, 1 stop bit
STR R1, [R0, #0x0C]  ; Store the new configuration
```

**3.1.2. Transmitting Data via UART**

To transmit data, write the data to the UART transmit register.

Example: Transmitting a character
```assembly
LDR R0, =0x40004028  ; Load address of UART data register
MOV R1, #0x41        ; ASCII code for 'A'
STR R1, [R0]         ; Write to the data register
```

**3.1.3. Receiving Data via UART**

To receive data, read from the UART receive register.

Example: Receiving a character
```assembly
LDR R0, =0x40004024  ; Load address of UART receive register
LDR R1, [R0]         ; Read received data
```

**3.2. SPI Communication**

SPI is a synchronous protocol used for high-speed communication between a master and one or more slave devices.

**3.2.1. Configuring SPI**

To configure SPI, set the clock polarity, phase, and data order.

Example: Configuring SPI
```assembly
LDR R0, =0x40013000  ; Load base address of SPI
LDR R1, [R0, #0x00]  ; Load current configuration
ORR R1, R1, #0x31    ; Set clock polarity, phase, and data order
STR R1, [R0, #0x00]  ; Store the new configuration
```

**3.2.2. Transmitting Data via SPI**

To transmit data, write to the SPI data register.

Example: Transmitting data
```assembly
LDR R0, =0x4001300C  ; Load address of SPI data register
MOV R1, #0xFF        ; Data to transmit
STR R1, [R0]         ; Write to the data register
```

**3.2.3. Receiving Data via SPI**

To receive data, read from the SPI data register.

Example: Receiving data
```assembly
LDR R0, =0x4001300C  ; Load address of SPI data register
LDR R1, [R0]         ; Read received data
```

**3.3. I2C Communication**

I2C is a multi-master, multi-slave, packet-switched, single-ended, serial communication bus.

**3.3.1. Configuring I2C**

To configure I2C, set the clock speed and address mode.

Example: Configuring I2C
```assembly
LDR R0, =0x40005400  ; Load base address of I2C
LDR R1, [R0, #0x00]  ; Load current configuration
ORR R1, R1, #0x01    ; Enable I2C
STR R1, [R0, #0x00]  ; Store the new configuration
```

**3.3.2. Transmitting Data via I2C**

To transmit data, write to the I2C data register.

Example: Transmitting data
```assembly
LDR R0, =0x40005410  ; Load address of I2C data register
MOV R1, #0xA0        ; Address of the slave device
STR R1, [R0]         ; Write address to the data register
MOV R1, #0x55        ; Data to transmit
STR R1, [R0]         ; Write data to the data register
```

**3.3.3. Receiving Data via I2C**

To receive data, read from the I2C data register.

Example: Receiving data
```assembly
LDR R0, =0x40005410  ; Load address of I2C data register
LDR R1, [R0]         ; Read received data
```

#### **4. Interrupt-Driven I/O**

Interrupt-driven I/O allows a microcontroller to respond to events from peripherals asynchronously, without the need to continuously poll the status registers.

**4.1. Configuring Interrupts**

To use interrupts, you need to enable the interrupt in the peripheral and the NVIC (Nested Vectored Interrupt Controller).

Example: Configuring an interrupt
```assembly
LDR R0, =0x40010000  ; Load base address of peripheral
LDR R1, [R0, #0x10]  ; Load current interrupt configuration
ORR R1, R1, #0x01    ; Enable the interrupt
STR R1, [R0, #0x10]  ; Store the new configuration

LDR R0, =0xE000E100  ; Load base address of NVIC
LDR R1, [R0]         ; Load current NVIC configuration
ORR R1, R1, #0x01    ; Enable the interrupt in NVIC
STR R1, [R0]         ; Store the new configuration
```

**4.2. Handling Interrupts**

Interrupt service routines (ISRs) are used to handle interrupts. The ISR should be as short as possible to minimize the time spent in the interrupt context.

Example: Interrupt Service Routine
```assembly
ISR_Handler:
    ; Save context
    PUSH {R0-R3, LR}

    ; Handle interrupt
    LDR R0, =0x40010020  ; Load address of peripheral status register
    LDR R1, [R0]         ; Read status register
    ; (Perform specific handling based on the status)

    ; Clear interrupt
    LDR R0, =0x40010024  ; Load address of interrupt clear register
    MOV R1, #0x01        ; Clear the interrupt
    STR R1, [R0]         ; Write to the clear register

    ; Restore context
    POP {R0-R3, LR}
    BX LR                ; Return from interrupt
```

#### **5. Practical Example: LED Blinking with Button Press**

To consolidate the concepts learned, let's work through a practical example. We'll create a program that blinks an LED when a button is pressed. This involves configuring a digital input for the button, a digital output for the LED, and handling the button press using interrupts.

**5.1. Configuring the Button (Input)**

Configure a GPIO pin as an input for the button.
```assembly
LDR R0, =0x48000000  ; Load base address of GPIO port
LDR R1, [R0, #0x00]  ; Load the current configuration
BIC R1, R1, #0x01    ; Set pin 0 as input
STR R1, [R0, #0x00]  ; Store the new configuration

LDR R0, =0x48000010  ; Load address of GPIO pull-up register
LDR R1, [R0, #0x00]  ; Enable pull-up resistor for the button
ORR R1, R1, #0x01    ; (Assuming pull-up resistor is bit 0)
STR R1, [R0, #0x00]  ; Store the new configuration
```

**5.2. Configuring the LED (Output)**

Configure a GPIO pin as an output for the LED.
```assembly
LDR R0, =0x48000000  ; Load base address of GPIO port
LDR R1, [R0, #0x00]  ; Load the current configuration
ORR R1, R1, #0x02    ; Set pin 1 as output
STR R1, [R0, #0x00]  ; Store the new configuration
```

**5.3. Configuring the Button Interrupt**

Enable the interrupt for the button press.
```assembly
LDR R0, =0x40010400  ; Load base address of EXTI (External Interrupt) controller
LDR R1, [R0, #0x00]  ; Load current interrupt configuration
ORR R1, R1, #0x01    ; Enable interrupt for pin 0
STR R1, [R0, #0x00]  ; Store the new configuration

LDR R0, =0xE000E100  ; Load base address of NVIC
LDR R1, [R0]         ; Load current NVIC configuration
ORR R1, R1, #0x01    ; Enable interrupt in NVIC
STR R1, [R0]         ; Store the new configuration
```

**5.4. Writing the Interrupt Service Routine**

Implement the ISR to toggle the LED.
```assembly
Button_ISR_Handler:
    ; Save context
    PUSH {R0-R3, LR}

    ; Read the button state
    LDR R0, =0x48000010  ; Load address of GPIO input data register
    LDR R1, [R0]         ; Read pin states
    TST R1, #0x01        ; Check if button is pressed
    BEQ END_ISR          ; If not pressed, exit

    ; Toggle the LED
    LDR R0, =0x48000014  ; Load address of GPIO output data register
    LDR R1, [R0]         ; Read current LED state
    EOR R1, R1, #0x02    ; Toggle pin 1
    STR R1, [R0]         ; Write new state

    ; Clear the interrupt
    LDR R0, =0x4001040C  ; Load address of EXTI interrupt clear register
    MOV R1, #0x01        ; Clear interrupt for pin 0
    STR R1, [R0]         ; Write to clear register

END_ISR:
    ; Restore context
    POP {R0-R3, LR}
    BX LR                ; Return from interrupt
```

By following these steps, you can create a simple yet effective program to interface with hardware components using ARM assembly language. This chapter has covered the essential concepts and provided detailed examples to equip you with the knowledge needed to handle various I/O operations, making your embedded systems programming more robust and versatile.

### Peripheral Programming: Timers, ADCs, and Other Peripherals

Peripheral programming is a critical aspect of embedded systems development, enabling microcontrollers to interact with various external and internal devices to perform complex tasks. This chapter will provide a comprehensive overview of programming different peripherals such as timers, Analog-to-Digital Converters (ADCs), Digital-to-Analog Converters (DACs), and other commonly used peripherals. We will explore the configuration, operation, and practical applications of these peripherals using Assembly Language on ARM architecture.

#### **1. Timers**

Timers are essential peripherals in microcontrollers, used for a wide range of applications including time delays, event counting, and generating periodic interrupts.

**1.1. Types of Timers**

1. **Basic Timers**: Used for simple time delays.
2. **General-Purpose Timers**: Capable of input capture, output compare, and PWM generation.
3. **Advanced Timers**: Feature-rich timers often used for motor control and other complex tasks.

**1.2. Configuring a Timer**

Configuring a timer typically involves setting the prescaler, auto-reload value, and enabling the timer. The prescaler divides the system clock to achieve the desired timer frequency, while the auto-reload value determines the period of the timer.

Example: Configuring a basic timer
```assembly
LDR R0, =0x40010000  ; Load base address of timer
LDR R1, =0x00FF      ; Set prescaler value
STR R1, [R0, #0x28]  ; Write to prescaler register
LDR R1, =0x0FFF      ; Set auto-reload value
STR R1, [R0, #0x2C]  ; Write to auto-reload register
LDR R1, =0x01        ; Enable the timer
STR R1, [R0, #0x00]  ; Write to control register
```

**1.3. Timer Modes**

Timers can operate in different modes such as:

1. **Upcounting Mode**: The counter counts from 0 to the auto-reload value.
2. **Downcounting Mode**: The counter counts down from the auto-reload value to 0.
3. **Center-Aligned Mode**: The counter counts up and down, useful for PWM generation.

**1.4. Generating Delays with Timers**

Timers can be used to generate precise time delays. This involves starting the timer, waiting for it to reach the desired count, and then stopping it.

Example: Generating a delay
```assembly
Delay:
    LDR R0, =0x40010000  ; Load base address of timer
    LDR R1, =0x00FF      ; Set prescaler value
    STR R1, [R0, #0x28]  ; Write to prescaler register
    LDR R1, =0x0FFF      ; Set auto-reload value
    STR R1, [R0, #0x2C]  ; Write to auto-reload register
    LDR R1, =0x01        ; Enable the timer
    STR R1, [R0, #0x00]  ; Write to control register

Wait:
    LDR R1, [R0, #0x24]  ; Read the counter value
    CMP R1, #0x0FFF      ; Compare with auto-reload value
    BNE Wait             ; Wait until the counter reaches the value

    LDR R1, =0x00        ; Disable the timer
    STR R1, [R0, #0x00]  ; Write to control register
    BX LR                ; Return from subroutine
```

**1.5. Timer Interrupts**

Timers can generate interrupts at specific intervals, allowing for periodic tasks without continuous polling.

Example: Configuring a timer interrupt
```assembly
LDR R0, =0x40010000  ; Load base address of timer
LDR R1, =0x00FF      ; Set prescaler value
STR R1, [R0, #0x28]  ; Write to prescaler register
LDR R1, =0x0FFF      ; Set auto-reload value
STR R1, [R0, #0x2C]  ; Write to auto-reload register
LDR R1, =0x01        ; Enable the timer
STR R1, [R0, #0x00]  ; Write to control register
LDR R1, [R0, #0x0C]  ; Enable update interrupt
ORR R1, R1, #0x01
STR R1, [R0, #0x0C]

LDR R0, =0xE000E100  ; Load base address of NVIC
LDR R1, [R0]         ; Enable timer interrupt in NVIC
ORR R1, R1, #0x01
STR R1, [R0]
```

**1.6. Timer Interrupt Service Routine**

Example: Timer ISR
```assembly
Timer_ISR_Handler:
    ; Save context
    PUSH {R0-R3, LR}

    ; Handle the interrupt
    ; (Your specific interrupt handling code here)

    ; Clear the interrupt flag
    LDR R0, =0x40010010  ; Load base address of timer status register
    LDR R1, [R0]         ; Read the status register
    BIC R1, R1, #0x01    ; Clear the update interrupt flag
    STR R1, [R0]         ; Write back to status register

    ; Restore context
    POP {R0-R3, LR}
    BX LR                ; Return from interrupt
```

#### **2. Analog-to-Digital Converters (ADCs)**

ADCs are used to convert analog signals to digital values, enabling microcontrollers to process analog inputs such as sensor readings.

**2.1. ADC Configuration**

Configuring an ADC involves selecting the input channel, setting the resolution, and enabling the ADC.

Example: Configuring an ADC
```assembly
LDR R0, =0x50000000  ; Load base address of ADC
LDR R1, [R0, #0x00]  ; Load current configuration
ORR R1, R1, #0x01    ; Enable the ADC
STR R1, [R0, #0x00]  ; Store the new configuration
```

**2.2. Starting an ADC Conversion**

To start an ADC conversion, write to the start conversion register.

Example: Starting a conversion
```assembly
LDR R0, =0x50000004  ; Load address of ADC start register
MOV R1, #0x01        ; Start conversion on channel 0
STR R1, [R0]         ; Write to start register
```

**2.3. Reading ADC Conversion Results**

After starting a conversion, wait for it to complete and then read the result from the data register.

Example: Reading ADC result
```assembly
LDR R0, =0x50000008  ; Load address of ADC data register
WAIT_LOOP:           ; Wait for conversion to complete
    LDR R1, [R0]
    TST R1, #0x8000  ; Check if conversion complete
    BEQ WAIT_LOOP    ; If not, wait

MOV R1, R1, LSR #16  ; Extract the ADC result
```

**2.4. ADC Interrupts**

ADCs can generate interrupts when a conversion is complete, allowing for asynchronous processing.

Example: Configuring an ADC interrupt
```assembly
LDR R0, =0x5000000C  ; Load base address of ADC interrupt enable register
LDR R1, [R0]
ORR R1, R1, #0x01    ; Enable end-of-conversion interrupt
STR R1, [R0]

LDR R0, =0xE000E100  ; Load base address of NVIC
LDR R1, [R0]
ORR R1, R1, #0x02    ; Enable ADC interrupt in NVIC
STR R1, [R0]
```

**2.5. ADC Interrupt Service Routine**

Example: ADC ISR
```assembly
ADC_ISR_Handler:
    ; Save context
    PUSH {R0-R3, LR}

    ; Handle the interrupt
    LDR R0, =0x50000008  ; Load address of ADC data register
    LDR R1, [R0]         ; Read the ADC result
    ; (Process the result)

    ; Clear the interrupt flag
    LDR R0, =0x50000010  ; Load address of ADC status register
    LDR R1, [R0]
    BIC R1, R1, #0x01    ; Clear the end-of-conversion flag
    STR R1, [R0]

    ; Restore context
    POP {R0-R3, LR}
    BX LR                ; Return from interrupt
```

#### **3. Digital-to-Analog Converters (DACs)**

DACs convert digital values to analog signals, used for generating analog outputs such as audio signals or variable voltage outputs.

**3.1. DAC Configuration**

Configuring a DAC involves setting the reference voltage and enabling the output channel.

Example: Configuring a DAC
```assembly
LDR R0, =0x50010000  ; Load base address of DAC
LDR R1, [R0, #0x00]  ; Load current configuration
ORR R1, R1, #0x01    ; Enable the DAC
STR R1, [R0, #0x00]  ; Store the new configuration
```

**3.2. Writing to the DAC**

After configuration, write a digital value to be converted to an analog signal.

Example: Writing to DAC
```assembly
LDR R0, =0x50010008  ; Load address of DAC data register
MOV R1, #0x0FFF      ; Write maximum value for 12-bit DAC
STR R1, [R0]         ; Write to the data register
```

**3.3. Generating Waveforms with DACs**

DACs can be used to generate waveforms by writing different values in a timed sequence.

Example: Generating a sine wave
```assembly
; Assume we have a lookup table for sine wave values
SineTable: .word 0x800, 0xA8C, 0xC94, 0xE14, 0xF02, 0xF74, 0xF76, 0xF02
           .word 0xE14, 0xC94, 0xA8C, 0x800, 0x574, 0x374, 0x1EC, 0x0FC
           .word 0x08A, 0x080, 0x08A, 0x0FC, 0x1EC, 0x374, 0x574, 0x800

GenerateSineWave:
    LDR R0, =SineTable   ; Load address of sine table
    LDR R1, =0x50010008  ; Load address of DAC data register

Loop:
    LDR R2, [R0], #4     ; Load next sine value and increment pointer
    STR R2, [R1]         ; Write value to DAC
    BL  Delay            ; Call delay subroutine
    B   Loop             ; Repeat
```

#### **4. Other Peripherals**

ARM microcontrollers come with a variety of other peripherals such as PWM generators, communication interfaces (I2C, SPI, UART), and more. Here, we will discuss some of these peripherals briefly.

**4.1. Pulse Width Modulation (PWM)**

PWM is used to generate a square wave with variable duty cycle, commonly used for controlling motors and LEDs.

**4.1.1. Configuring PWM**

Configuring PWM involves setting the frequency and duty cycle.

Example: Configuring PWM
```assembly
LDR R0, =0x40014000  ; Load base address of PWM
LDR R1, =0x00FF      ; Set prescaler value
STR R1, [R0, #0x28]  ; Write to prescaler register
LDR R1, =0x0FFF      ; Set auto-reload value
STR R1, [R0, #0x2C]  ; Write to auto-reload register
LDR R1, =0x0800      ; Set compare value for 50% duty cycle
STR R1, [R0, #0x34]  ; Write to compare register
LDR R1, =0x01        ; Enable the PWM
STR R1, [R0, #0x00]  ; Write to control register
```

**4.2. I2C Communication**

I2C is a two-wire communication protocol used for interfacing with sensors and other devices.

**4.2.1. Configuring I2C**

To configure I2C, set the clock speed and address mode.

Example: Configuring I2C
```assembly
LDR R0, =0x40005400  ; Load base address of I2C
LDR R1, [R0, #0x00]  ; Load current configuration
ORR R1, R1, #0x01    ; Enable I2C
STR R1, [R0, #0x00]  ; Store the new configuration
```

**4.2.2. Transmitting Data via I2C**

To transmit data, write to the I2C data register.

Example: Transmitting data
```assembly
LDR R0, =0x40005410  ; Load address of I2C data register
MOV R1, #0xA0        ; Address of the slave device
STR R1, [R0]         ; Write address to the data register
MOV R1, #0x55        ; Data to transmit
STR R1, [R0]         ; Write data to the data register
```

**4.2.3. Receiving Data via I2C**

To receive data, read from the I2C data register.

Example: Receiving data
```assembly
LDR R0, =0x40005410  ; Load address of I2C data register
LDR R1, [R0]         ; Read received data
```

**4.3. SPI Communication**

SPI is a high-speed communication protocol used for interfacing with devices such as flash memory and sensors.

**4.3.1. Configuring SPI**

To configure SPI, set the clock polarity, phase, and data order.

Example: Configuring SPI
```assembly
LDR R0, =0x40013000  ; Load base address of SPI
LDR R1, [R0, #0x00]  ; Load current configuration
ORR R1, R1, #0x31    ; Set clock polarity, phase, and data order
STR R1, [R0, #0x00]  ; Store the new configuration
```

**4.3.2. Transmitting Data via SPI**

To transmit data, write to the SPI data register.

Example: Transmitting data
```assembly
LDR R0, =0x4001300C  ; Load address of SPI data register
MOV R1, #0xFF        ; Data to transmit
STR R1, [R0]         ; Write to the data register
```

**4.3.3. Receiving Data via SPI**

To receive data, read from the SPI data register.

Example: Receiving data
```assembly
LDR R0, =0x4001300C  ; Load address of SPI data register
LDR R1, [R0]         ; Read received data
```

#### **5. Practical Example: Temperature Monitoring System**

To illustrate the concepts learned, we will create a temperature monitoring system using an ADC to read a temperature sensor, a DAC to control a fan speed based on temperature, and a timer to periodically update the system.

**5.1. Configuring the ADC for Temperature Sensor**

Configure the ADC to read from the temperature sensor.

Example: Configuring the ADC
```assembly
LDR R0, =0x50000000  ; Load base address of ADC
LDR R1, [R0, #0x00]  ; Load current configuration
ORR R1, R1, #0x01    ; Enable the ADC
STR R1, [R0, #0x00]  ; Store the new configuration
```

**5.2. Configuring the DAC for Fan Control**

Configure the DAC to control the fan speed.

Example: Configuring the DAC
```assembly
LDR R0, =0x50010000  ; Load base address of DAC
LDR R1, [R0, #0x00]  ; Load current configuration
ORR R1, R1, #0x01    ; Enable the DAC
STR R1, [R0, #0x00]  ; Store the new configuration
```

**5.3. Configuring the Timer for Periodic Updates**

Configure a timer to generate periodic interrupts for system updates.

Example: Configuring the timer
```assembly
LDR R0, =0x40010000  ; Load base address of timer
LDR R1, =0x00FF      ; Set prescaler value
STR R1, [R0, #0x28]  ; Write to prescaler register
LDR R1, =0x0FFF      ; Set auto-reload value
STR R1, [R0, #0x2C]  ; Write to auto-reload register
LDR R1, =0x01        ; Enable the timer
STR R1, [R0, #0x00]  ; Write to control register
```

**5.4. Timer Interrupt Service Routine**

Implement the ISR to read the temperature sensor and control the fan.

Example: Timer ISR
```assembly
Timer_ISR_Handler:
    ; Save context
    PUSH {R0-R3, LR}

    ; Read the temperature sensor
    LDR R0, =0x50000008  ; Load address of ADC data register
    WAIT_LOOP:           ; Wait for conversion to complete
        LDR R1, [R0]
        TST R1, #0x8000  ; Check if conversion complete
        BEQ WAIT_LOOP    ; If not, wait
    MOV R1, R1, LSR #16  ; Extract the ADC result

    ; Control the fan speed
    LDR R2, =0x50010008  ; Load address of DAC data register
    STR R1, [R2]         ; Write the temperature value to DAC

    ; Clear the interrupt flag
    LDR R0, =0x40010010  ; Load base address of timer status register
    LDR R1, [R0]
    BIC R1, R1, #0x01    ; Clear the update interrupt flag
    STR R1, [R0]

    ; Restore context
    POP {R0-R3, LR}
    BX LR                ; Return from interrupt
```

By following these detailed steps, you can effectively program and utilize various peripherals on an ARM microcontroller. This chapter has provided an exhaustive overview of working with timers, ADCs, DACs, and other peripherals, equipping you with the necessary knowledge to handle complex tasks in embedded systems development.

### Memory-Mapped I/O

Memory-Mapped I/O (MMIO) is a method used in computer systems to communicate with hardware devices by mapping their registers into the same address space as the program memory. This approach allows software to interact with peripherals using standard memory access instructions. In this chapter, we will delve into the intricacies of MMIO, exploring its advantages, how it works, and providing detailed examples to illustrate its use. This will include addressing, register access, synchronization issues, and practical applications.

#### **1. Introduction to Memory-Mapped I/O**

Memory-Mapped I/O involves assigning specific memory addresses to hardware device registers. Instead of using separate I/O instructions, the processor uses regular load and store instructions to access these addresses, thereby controlling the peripherals.

**1.1. Advantages of Memory-Mapped I/O**

1. **Unified Address Space**: Simplifies the processor design as both memory and I/O devices share the same address space.
2. **Simplified Programming Model**: Allows the use of standard memory access instructions to control I/O devices, making the programming model consistent.
3. **Efficient Access**: Enables faster access to I/O devices since no special instructions are needed.

**1.2. How Memory-Mapped I/O Works**

In MMIO, each peripheral device is assigned a block of addresses. These addresses correspond to the device's registers. By reading from or writing to these addresses, the processor can control the peripheral.

Example: Suppose an LED is controlled by a register located at address 0x40021000. Writing a 1 to this address turns the LED on, and writing a 0 turns it off.

#### **2. Addressing in Memory-Mapped I/O**

Addressing is a critical aspect of MMIO. Each peripheral has a base address, and its registers are located at offsets from this base address.

**2.1. Base Address and Offset**

The base address is the starting address of a peripheral's register block. Each register within the block is accessed using an offset from the base address.

Example: For a GPIO peripheral with a base address of 0x40020000:
- Mode Register (MODER) might be at offset 0x00.
- Output Data Register (ODR) might be at offset 0x14.

To access the ODR:
```assembly
LDR R0, =0x40020014  ; Base address + offset
LDR R1, [R0]         ; Read the register
```

**2.2. Register Access**

Accessing a register involves calculating its address and using load/store instructions.

Example: Configuring a GPIO pin as output
```assembly
LDR R0, =0x40020000  ; Load base address of GPIO
LDR R1, [R0, #0x00]  ; Load current MODER register value
ORR R1, R1, #0x01    ; Set pin 0 as output
STR R1, [R0, #0x00]  ; Store the new value
```

#### **3. Synchronization and Atomicity**

When dealing with MMIO, synchronization and atomicity are important considerations to prevent race conditions and ensure data integrity.

**3.1. Volatile Keyword**

In high-level languages like C, the `volatile` keyword is used to inform the compiler that a variable may change at any time, preventing the compiler from optimizing out necessary reads and writes.

Example:
```c
volatile uint32_t *gpio_odr = (uint32_t *)0x40020014;
*gpio_odr = 0x01;  // Set pin 0 high
```

**3.2. Read-Modify-Write Operations**

Read-modify-write operations must be performed atomically to avoid race conditions.

Example: Toggling a GPIO pin atomically
```assembly
LDR R0, =0x40020014  ; Load address of ODR
LDR R1, [R0]         ; Read current value
EOR R1, R1, #0x01    ; Toggle pin 0
STR R1, [R0]         ; Write back the value
```

#### **4. Practical Examples of Memory-Mapped I/O**

**4.1. GPIO Control**

Let's consider a practical example of configuring and using GPIO pins via MMIO.

**4.1.1. Configuring GPIO as Output**

Example: Configuring pin 0 as output
```assembly
LDR R0, =0x40020000  ; Load base address of GPIO
LDR R1, [R0, #0x00]  ; Load current MODER register value
ORR R1, R1, #0x01    ; Set pin 0 as output
STR R1, [R0, #0x00]  ; Store the new value
```

**4.1.2. Setting and Clearing GPIO Pins**

Example: Setting pin 0 high
```assembly
LDR R0, =0x40020014  ; Load address of ODR
LDR R1, [R0]         ; Read current value
ORR R1, R1, #0x01    ; Set pin 0 high
STR R1, [R0]         ; Write back the value
```

Example: Clearing pin 0
```assembly
LDR R0, =0x40020014  ; Load address of ODR
LDR R1, [R0]         ; Read current value
BIC R1, R1, #0x01    ; Clear pin 0
STR R1, [R0]         ; Write back the value
```

**4.2. UART Communication**

UART (Universal Asynchronous Receiver/Transmitter) is a common peripheral used for serial communication.

**4.2.1. Configuring UART**

Example: Configuring UART with a specific baud rate
```assembly
LDR R0, =0x40004000  ; Load base address of UART
LDR R1, =0x960       ; Set baud rate (assuming 16 MHz clock)
STR R1, [R0, #0x24]  ; Write to the baud rate register
LDR R1, [R0, #0x0C]  ; Load current configuration
ORR R1, R1, #0x301   ; 8 data bits, no parity, 1 stop bit
STR R1, [R0, #0x0C]  ; Store the new configuration
```

**4.2.2. Transmitting Data via UART**

Example: Transmitting a character
```assembly
LDR R0, =0x40004028  ; Load address of UART data register
MOV R1, #0x41        ; ASCII code for 'A'
STR R1, [R0]         ; Write to the data register
```

**4.2.3. Receiving Data via UART**

Example: Receiving a character
```assembly
LDR R0, =0x40004024  ; Load address of UART receive register
LDR R1, [R0]         ; Read received data
```

#### **5. Advanced Topics in Memory-Mapped I/O**

**5.1. Direct Memory Access (DMA)**

DMA is a feature that allows peripherals to directly read from or write to memory without CPU intervention, enhancing data transfer efficiency.

**5.1.1. Configuring DMA**

Example: Configuring DMA for memory-to-memory transfer
```assembly
LDR R0, =0x40026000  ; Load base address of DMA
LDR R1, [R0, #0x00]  ; Load current configuration
ORR R1, R1, #0x01    ; Enable DMA
STR R1, [R0, #0x00]  ; Store the new configuration
LDR R2, =0x20000000  ; Source address
LDR R3, =0x20001000  ; Destination address
LDR R4, =0x100       ; Number of bytes to transfer
STR R2, [R0, #0x04]  ; Write source address
STR R3, [R0, #0x08]  ; Write destination address
STR R4, [R0, #0x0C]  ; Write transfer size
LDR R1, [R0, #0x10]  ; Start the transfer
ORR R1, R1, #0x01
STR R1, [R0, #0x10]
```

**5.1.2. DMA Interrupts**

DMA can generate interrupts on transfer completion, error, etc.

Example: Configuring DMA interrupt
```assembly
LDR R0, =0x4002601C  ; Load address of DMA interrupt enable register
LDR R1, [R0]
ORR R1, R1, #0x01    ; Enable transfer complete interrupt
STR R1, [R0]

LDR R0, =0xE000E100  ; Load base address of NVIC
LDR R1, [R0]
ORR R1, R1, #0x04    ; Enable DMA interrupt in NVIC
STR R1, [R0]
```

**5.2. Memory-Mapped I/O in Multicore Systems**

In multicore systems, memory-mapped I/O must handle potential conflicts and synchronization issues between cores.

**5.2.1. Synchronization Mechanisms**

Mechanisms such as mutexes, semaphores, and memory barriers ensure proper synchronization.

Example: Using a memory barrier in ARM assembly
```assembly
DMB    ; Data Memory Barrier
LDR R0, =0x40020000
LDR R1, [R0]
; Perform operations
DMB    ; Data Memory Barrier
```

**5.2.2. Inter-Processor Communication (IPC)**

IPC mechanisms such as message passing, shared memory, and interrupts enable communication between cores.

Example: Sending a message via shared memory
```assembly
LDR R0, =0x20002000  ; Shared memory address
MOV R1, #0x12345678  ; Message to send
STR R1, [R0]         ; Write message
```
