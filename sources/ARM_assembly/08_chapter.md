\newpage

## 8. **Memory and Addressing Modes**

In this chapter, we delve into the intricate workings of memory and addressing modes in ARM architecture, providing a crucial foundation for effective programming and system optimization. We'll explore how ARM processors manage memory through their sophisticated memory architecture, ensuring efficient data handling and processing. Next, we’ll dissect various addressing modes, including immediate, register, and indexed addressing, to demonstrate how they enable precise and flexible data access within the system. Furthermore, we’ll cover stack and heap management, essential for both static and dynamic memory allocation, ensuring robust and scalable code. To solidify your understanding, we will conclude with a comprehensive example that integrates these concepts, offering a detailed explanation to bridge theory and practical application.

### Memory Architecture

Understanding the memory architecture of ARM processors is fundamental for writing efficient and optimized code. ARM processors, widely known for their low power consumption and high performance, utilize a sophisticated memory architecture to handle data storage and retrieval effectively. This chapter provides an exhaustive exploration of how ARM processors manage memory, covering the essential components, their interactions, and the underlying principles that govern memory operations.

#### Overview of ARM Memory Architecture

ARM processors employ a Harvard architecture, where the instruction and data caches are separate. This allows simultaneous access to instructions and data, significantly enhancing performance. The memory architecture can be broadly divided into the following components:

1. **Memory Types and Hierarchy**:
    - **Registers**: The fastest type of memory, used for immediate data processing.
    - **Cache Memory**: Small, high-speed memory layers (L1, L2, and sometimes L3) that store frequently accessed data and instructions to speed up processing.
    - **Main Memory (RAM)**: The primary workspace for the processor, where active data and programs are stored.
    - **Secondary Storage**: Non-volatile memory such as SSDs and HDDs used for long-term data storage.

2. **Memory Access Methods**:
    - **Load/Store Architecture**: ARM employs a load/store architecture, meaning operations are performed only on registers, and data must be loaded into registers from memory before processing.
    - **Endianness**: ARM processors support both little-endian and big-endian formats, allowing flexibility in how data is stored and interpreted.

#### Registers and Their Roles

Registers are the topmost layer in the ARM memory hierarchy, providing the fastest access to data. ARM processors typically have a set of 16 general-purpose registers (R0-R15), including the Program Counter (PC), Stack Pointer (SP), Link Register (LR), and the Current Program Status Register (CPSR).

- **Program Counter (PC)**: Holds the address of the next instruction to be executed.
- **Stack Pointer (SP)**: Points to the current position in the stack.
- **Link Register (LR)**: Stores the return address for function calls.
- **Current Program Status Register (CPSR)**: Contains flags and status bits that affect the processor state.

#### Cache Memory

Caches are critical for bridging the speed gap between the processor and main memory. ARM processors typically have a multi-level cache hierarchy:

1. **L1 Cache**: Split into Instruction Cache (I-Cache) and Data Cache (D-Cache). It is the smallest but fastest cache.
2. **L2 Cache**: Unified cache that stores both instructions and data, larger and slightly slower than L1.
3. **L3 Cache**: Not always present, but when available, it provides a larger but slower cache layer compared to L1 and L2.

Caches use various techniques to improve performance, such as:

- **Cache Coherency**: Ensures that multiple caches in a multi-core system have consistent data.
- **Write-Back vs. Write-Through**: Determines how and when data is written from the cache to the main memory.
- **Replacement Policies**: Strategies like LRU (Least Recently Used) to decide which cache lines to evict when new data is loaded.

#### Main Memory (RAM)

Main memory is the primary workspace for ARM processors, where data and instructions that are currently being used are stored. The characteristics of main memory include:

- **Volatile Nature**: RAM is volatile, meaning data is lost when power is turned off.
- **Access Time**: Slower than cache but provides larger storage capacity.

#### Secondary Storage

Secondary storage provides non-volatile, long-term storage for data and programs. Examples include Solid-State Drives (SSDs) and Hard Disk Drives (HDDs). Although much slower than RAM, secondary storage is essential for retaining data across power cycles.

#### Memory Access Techniques

ARM processors employ various techniques to access and manage memory efficiently:

1. **Load/Store Instructions**: ARM's load/store architecture mandates that all data processing occurs in registers. Data must be loaded from memory into registers, processed, and then stored back into memory if needed.
2. **Prefetching**: ARM processors use prefetching to load instructions and data into the cache before they are needed, reducing wait times.
3. **Memory-Mapped I/O**: Peripherals are mapped into the same address space as program memory and data, allowing the CPU to read from and write to hardware devices using standard load/store instructions.

#### Memory Protection and Management

Memory management is critical for ensuring system stability and security. ARM processors include several features to manage and protect memory:

1. **Memory Protection Unit (MPU)**: The MPU controls access permissions for different memory regions, preventing unauthorized access and protecting critical data.
2. **Virtual Memory and MMU**: The Memory Management Unit (MMU) translates virtual addresses to physical addresses, enabling features like paging and segmentation, which enhance memory utilization and provide isolation between processes.
3. **Address Space Layout Randomization (ASLR)**: ASLR randomizes the memory addresses used by system and application processes, making it harder for attackers to predict the location of specific functions or data structures.

#### Memory Access Models

ARM processors support different memory access models, including:

1. **Strongly Ordered**: Ensures strict ordering of memory operations.
2. **Device**: Allows reordering of operations to improve performance while maintaining consistency for device accesses.
3. **Normal**: Used for general-purpose memory, where caching and speculative access are permitted.
4. **Shared vs. Non-Shared**: Indicates whether memory is shared between multiple processors or used exclusively by one.

#### Endianness

ARM processors support both little-endian and big-endian memory formats. Little-endian format stores the least significant byte at the smallest address, while big-endian format stores the most significant byte at the smallest address. This flexibility allows ARM processors to interface with various systems and peripherals seamlessly.

#### Memory Allocation Techniques

Efficient memory allocation is vital for optimal performance. ARM processors use several techniques to manage memory allocation:

1. **Static Allocation**: Memory is allocated at compile time, with fixed sizes and locations.
2. **Dynamic Allocation**: Memory is allocated at runtime using functions like `malloc` and `free` in C, allowing flexible and efficient use of memory resources.
3. **Stack Allocation**: Memory for local variables is allocated on the stack, which grows and shrinks dynamically with function calls and returns.
4. **Heap Allocation**: Used for dynamic memory allocation, managed through functions like `malloc` and `free`, allowing programs to request and release memory as needed.

#### Stack and Heap Management

Effective management of the stack and heap is crucial for program stability and performance:

1. **Stack Management**: The stack is used for local variables, function parameters, and return addresses. ARM processors use the Stack Pointer (SP) to keep track of the top of the stack. Functions push data onto the stack when called and pop data when returning.
2. **Heap Management**: The heap is used for dynamic memory allocation. ARM processors rely on software routines to manage heap memory, ensuring efficient allocation and deallocation to avoid fragmentation and memory leaks.

#### A Combined Example with Explanation

To illustrate the concepts discussed, let's consider a comprehensive example that integrates various aspects of ARM memory architecture:

```assembly
    .data
message:
    .asciz "Hello, ARM!\n"

    .text
    .global _start

_start:
    LDR R0, =message   // Load the address of the message into R0
    BL print_string    // Call the print_string function
    B exit             // Branch to exit

print_string:
    PUSH {LR}          // Save the Link Register
    LDR R1, [R0]       // Load the first byte of the message
print_loop:
    CMP R1, #0         // Compare the byte with null terminator
    BEQ end_print      // If null, end the print loop
    BL putchar         // Call putchar function
    ADD R0, R0, #1     // Move to the next byte
    LDR R1, [R0]       // Load the next byte
    B print_loop       // Repeat the loop
end_print:
    POP {LR}           // Restore the Link Register
    BX LR              // Return from function

putchar:
    // Implementation of putchar that writes a character to stdout
    // For simplicity, let's assume a system call is used
    PUSH {R7}          // Save the syscall number register
    MOV R7, #4         // Syscall number for write
    MOV R2, #1         // Write 1 byte
    MOV R1, R0         // Character to write
    MOV R0, #1         // File descriptor (stdout)
    SWI 0              // Software interrupt to make the syscall
    POP {R7}           // Restore the syscall number register
    BX LR              // Return from function

exit:
    MOV R7, #1         // Syscall number for exit
    MOV R0, #0         // Exit status
    SWI 0              // Software interrupt to make the syscall
```

**Explanation**:

1. **Data Section**: The `message` variable is stored in the data section with a null terminator.
2. **Text Section**: Contains the main program and function definitions.
3. **Main Program**: Loads the address of the message into register R0 and calls the `print_string` function.
4. **Print String Function**:
    - Saves the Link Register (LR) on the stack.
    - Loads each byte of the message in a loop and calls `putchar` to print each character.
    - Uses the stack to save and restore the LR, demonstrating stack management.
5. **Putchar Function**:
    - Uses a system call to write a character to stdout.
    - Saves and restores the syscall number register R7, showcasing register usage.
6. **Exit**: Terminates the program using a system call.

This example illustrates how ARM processors handle memory through register manipulation, stack management, and efficient memory access techniques, providing a practical application of the theoretical concepts discussed in this chapter.

By understanding the intricate details of ARM memory architecture, developers can write more efficient, reliable, and optimized code, harnessing the full potential of ARM processors.

### Addressing Modes

Addressing modes are a critical aspect of ARM architecture, determining how the processor accesses data stored in memory or registers. Efficient use of addressing modes can greatly enhance the performance and flexibility of your code. In this chapter, we will delve deeply into the three primary addressing modes in ARM: immediate, register, and indexed addressing. We will cover the theoretical foundations, practical implementations, and provide comprehensive examples to illustrate each mode's usage.

#### Overview of Addressing Modes

Addressing modes define how the operand of an instruction is specified. ARM architecture supports several addressing modes to facilitate various programming needs. The primary addressing modes we will explore are:

1. **Immediate Addressing**: The operand is a constant value encoded within the instruction.
2. **Register Addressing**: The operand is stored in a register.
3. **Indexed Addressing**: The operand’s address is calculated using a base register and an offset.

#### Immediate Addressing

Immediate addressing involves encoding a constant value directly within the instruction. This allows the processor to access the operand quickly, as no additional memory access is required. Immediate addressing is particularly useful for operations involving constants, such as setting register values or performing arithmetic operations.

##### Characteristics of Immediate Addressing

- **Simplicity**: The operand is directly available in the instruction, leading to faster execution.
- **Limited Range**: The size of the immediate value is constrained by the instruction format, typically allowing only small constants.
- **Usage**: Commonly used for initializing registers, comparing values, and simple arithmetic operations.

##### ARM Syntax for Immediate Addressing

In ARM assembly language, immediate values are specified using the `#` symbol followed by the constant value. For example:

```assembly
MOV R0, #10   // Move the immediate value 10 into register R0
ADD R1, R1, #5  // Add the immediate value 5 to the value in R1 and store the result in R1
```

##### Example of Immediate Addressing

Consider a scenario where we need to initialize several registers with constant values and perform arithmetic operations:

```assembly
    MOV R0, #20      // Initialize R0 with 20
    MOV R1, #10      // Initialize R1 with 10
    ADD R2, R0, #5   // Add 5 to the value in R0 and store the result in R2
    SUB R3, R1, #3   // Subtract 3 from the value in R1 and store the result in R3
    CMP R0, #20      // Compare the value in R0 with 20
    BEQ equal        // Branch to the label 'equal' if R0 equals 20
equal:
    // Code to execute if R0 is equal to 20
```

In this example, immediate addressing is used to initialize registers and perform arithmetic and comparison operations efficiently.

#### Register Addressing

Register addressing uses registers to hold the operand. This mode provides fast access to data as registers are located within the CPU, allowing quick read and write operations. Register addressing is fundamental to ARM's load/store architecture, where most data manipulations are performed on registers rather than directly on memory.

##### Characteristics of Register Addressing

- **Speed**: Accessing data in registers is significantly faster than accessing data in memory.
- **Flexibility**: Registers can be used for various purposes, such as holding data, addresses, or temporary values.
- **Usage**: Widely used in data processing instructions, loops, and function calls.

##### ARM Syntax for Register Addressing

In ARM assembly language, register operands are specified by the register names (e.g., R0, R1, R2). For example:

```assembly
    MOV R0, R1   // Move the value in R1 to R0
    ADD R2, R0, R1  // Add the values in R0 and R1 and store the result in R2
    LDR R3, [R0]   // Load the value from the memory address contained in R0 into R3
```

##### Example of Register Addressing

Consider a scenario where we perform a series of arithmetic operations on register values:

```assembly
    MOV R0, #15      // Initialize R0 with 15
    MOV R1, #25      // Initialize R1 with 25
    ADD R2, R0, R1   // Add the values in R0 and R1, store the result in R2
    SUB R3, R1, R0   // Subtract the value in R0 from R1, store the result in R3
    MUL R4, R2, R3   // Multiply the values in R2 and R3, store the result in R4
    MOV R5, R4       // Move the result from R4 to R5
```

In this example, register addressing is used to perform arithmetic operations directly on the values stored in registers, showcasing the speed and efficiency of this addressing mode.

#### Indexed Addressing

Indexed addressing involves calculating the effective address of the operand by combining a base register with an offset. This mode is particularly useful for accessing elements of arrays, structures, and other data structures where the location of data can be determined relative to a base address.

##### Characteristics of Indexed Addressing

- **Flexibility**: Allows access to data at variable offsets from a base address, useful for array and structure manipulation.
- **Complexity**: Requires additional computation to determine the effective address, which may involve adding or subtracting offsets.
- **Usage**: Commonly used for accessing elements in arrays, data structures, and for pointer arithmetic.

##### ARM Syntax for Indexed Addressing

In ARM assembly language, indexed addressing can be specified in various ways, including pre-indexed and post-indexed addressing:

- **Pre-indexed Addressing**: The effective address is calculated before the memory access.
- **Post-indexed Addressing**: The effective address is calculated after the memory access.

Examples:

```assembly
    LDR R0, [R1, #4]       // Pre-indexed: Load the value from the address (R1 + 4) into R0
    STR R2, [R3, #-8]!     // Pre-indexed with write-back: Store the value in R2 to the address (R3 - 8) and update R3
    LDR R4, [R5], #12      // Post-indexed: Load the value from the address in R5 into R4, then increment R5 by 12
    STR R6, [R7], #-4      // Post-indexed: Store the value in R6 to the address in R7, then decrement R7 by 4
```

##### Example of Indexed Addressing

Consider a scenario where we need to access elements of an array stored in memory:

```assembly
    // Assuming the base address of the array is stored in R0
    MOV R1, #0           // Initialize index register R1 to 0
    LDR R2, [R0, R1]     // Load the first element of the array into R2
    ADD R1, R1, #4       // Increment the index by 4 (assuming 32-bit elements)
    LDR R3, [R0, R1]     // Load the second element of the array into R3
    ADD R1, R1, #4       // Increment the index by 4
    LDR R4, [R0, R1]     // Load the third element of the array into R4
```

In this example, indexed addressing is used to access elements of an array by calculating the effective address using a base address and an offset. This demonstrates how indexed addressing can simplify the process of iterating through arrays and other data structures.

#### Combining Addressing Modes

In practice, addressing modes are often combined to achieve more complex memory access patterns. ARM instructions support various combinations of immediate, register, and indexed addressing to provide flexibility and efficiency.

##### Example of Combined Addressing Modes

Consider a scenario where we need to copy elements from one array to another:

```assembly
    // Assuming the base address of the source array is in R0
    // and the base address of the destination array is in R1
    MOV R2, #0           // Initialize index register R2 to 0
copy_loop:
    LDR R3, [R0, R2]     // Load the element from the source array into R3
    STR R3, [R1, R2]     // Store the element into the destination array
    ADD R2, R2, #4       // Increment the index by 4
    CMP R2, #40          // Compare the index with the array length (10 elements * 4 bytes)
    BLT copy_loop        // If the index is less than 40, repeat the loop
```

In this example, immediate addressing is used to initialize the index register, register addressing is used for arithmetic operations, and indexed addressing is used to access and manipulate elements of the arrays. This combination demonstrates the power and flexibility of ARM addressing modes in real-world applications.

#### Advanced Topics in Addressing Modes

##### Load-Store Multiple Instructions

ARM architecture includes load-store multiple instructions (LDM/STM) that allow multiple registers to be loaded from or stored to memory in a single instruction. These instructions can use various addressing modes to specify the base address and offset.

Example:

```assembly
    // Load multiple registers from memory
    LDMIA R0!, {R1-R4}    // Load R1, R2, R3, and R4 from memory starting at the address in R0, then increment R0

    // Store multiple registers to memory
    STMDB R0!, {R1-R4}    // Store R1, R2, R3, and R4 to memory starting at the address in R0, then decrement R0
```

##### Scaled Register Indexing

Scaled register indexing allows the offset to be scaled by a factor, providing more flexibility in accessing data structures.

Example:

```assembly
    // Assuming R0 contains the base address and R1 contains the index
    LDR R2, [R0, R1, LSL #2]   // Load the value from the address (R0 + R1 * 4) into R2
```

In this example, the offset in R1 is scaled by 4 (using the logical shift left operation) before being added to the base address in R0. This is particularly useful for accessing elements in an array where each element is 4 bytes (32 bits) wide.

### Stack and Heap Management

Efficient memory management is a cornerstone of effective programming, particularly in systems programming and applications that demand high performance. ARM processors, renowned for their power efficiency and versatility, provide robust mechanisms for managing both stack and heap memory. This chapter provides an exhaustive exploration of stack and heap management, detailing their roles, implementations, and best practices in ARM architecture.

#### Overview of Memory Management

Memory management involves the allocation, use, and deallocation of memory resources in a program. Two primary types of memory allocation are:

1. **Stack Memory**: Used for static memory allocation, which includes local variables, function parameters, and control data. It operates in a Last-In-First-Out (LIFO) manner.
2. **Heap Memory**: Used for dynamic memory allocation, allowing memory to be allocated and freed at runtime.

Each type of memory has its own management techniques and characteristics, which we will explore in detail.

#### The Stack

The stack is a special region of memory that stores temporary data such as function parameters, local variables, and return addresses. It grows and shrinks dynamically as functions are called and return. The stack is critical for maintaining function call hierarchies and local scopes in a program.

##### Characteristics of the Stack

- **LIFO Structure**: The stack follows a Last-In-First-Out (LIFO) principle, meaning the last item pushed onto the stack is the first one to be popped off.
- **Automatic Allocation**: Memory allocation and deallocation for the stack are managed automatically by the processor.
- **Limited Size**: The stack size is typically limited by the system, which can lead to stack overflow if exceeded.

##### Stack Operations

Common stack operations include:

1. **Push**: Adding an element to the top of the stack.
2. **Pop**: Removing the top element from the stack.
3. **Peek**: Viewing the top element without removing it.

##### ARM Stack Management

In ARM architecture, the stack is managed using the Stack Pointer (SP) register, which points to the top of the stack. ARM processors use two primary stack-related instructions:

1. **PUSH**: Saves registers onto the stack.
2. **POP**: Restores registers from the stack.

##### Example of Stack Usage

Consider a simple function call where local variables and parameters are pushed onto the stack:

```assembly
    // Function prologue
    PUSH {LR}            // Save the Link Register (return address)
    PUSH {R0, R1}        // Save parameters R0 and R1 on the stack

    // Function body
    MOV R0, #10          // Initialize local variable
    ADD R1, R0, R1       // Perform some operation

    // Function epilogue
    POP {R0, R1}         // Restore parameters
    POP {LR}             // Restore return address
    BX LR                // Return from function
```

In this example, the `PUSH` instruction saves the return address and parameters on the stack, while the `POP` instruction restores them, maintaining the function call integrity.

##### Function Call Stack

The function call stack, or call stack, is a stack data structure that stores information about the active subroutines of a computer program. Each entry in the call stack is called a "stack frame" or "activation record". A stack frame typically contains:

- Return address
- Function parameters
- Local variables
- Saved registers

##### Example of Nested Function Calls

Consider nested function calls to demonstrate stack frame management:

```assembly
    // Main function
main:
    PUSH {LR}             // Save return address
    BL functionA          // Call functionA
    POP {LR}              // Restore return address
    BX LR                 // Return from main

// Function A
functionA:
    PUSH {LR}             // Save return address
    BL functionB          // Call functionB
    POP {LR}              // Restore return address
    BX LR                 // Return from functionA

// Function B
functionB:
    PUSH {LR}             // Save return address
    // Function B body
    POP {LR}              // Restore return address
    BX LR                 // Return from functionB
```

Each function call pushes a new stack frame onto the stack, and returning from a function pops the corresponding stack frame, maintaining the integrity of nested function calls.

#### The Heap

The heap is a region of memory used for dynamic memory allocation. Unlike the stack, which follows a strict LIFO order, the heap allows memory to be allocated and freed in any order, providing greater flexibility.

##### Characteristics of the Heap

- **Dynamic Allocation**: Memory is allocated and freed at runtime, allowing for flexible memory usage.
- **Fragmentation**: Frequent allocation and deallocation can lead to memory fragmentation, where free memory is scattered in small blocks.
- **Managed by the Programmer**: Memory management on the heap is typically done explicitly by the programmer, using functions like `malloc` and `free` in C.

##### Dynamic Memory Allocation

Dynamic memory allocation involves requesting memory from the heap at runtime and releasing it when no longer needed. Common functions for dynamic memory allocation in C include:

1. **malloc**: Allocates a specified number of bytes and returns a pointer to the allocated memory.
2. **free**: Frees previously allocated memory, making it available for future allocations.
3. **realloc**: Changes the size of previously allocated memory.

##### Example of Heap Usage in C

Consider a C program that uses dynamic memory allocation to manage an array:

```c
#include <stdlib.h>

#include <stdio.h>

int main() {
    int *array;
    int size = 10;

    // Allocate memory for an array of 10 integers
    array = (int *)malloc(size * sizeof(int));
    if (array == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize the array
    for (int i = 0; i < size; i++) {
        array[i] = i * 10;
    }

    // Print the array
    for (int i = 0; i < size; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");

    // Free the allocated memory
    free(array);

    return 0;
}
```

In this example, `malloc` is used to allocate memory for an array, which is then initialized and printed. Finally, `free` is used to release the allocated memory.

##### Memory Fragmentation

Memory fragmentation occurs when free memory is divided into small, non-contiguous blocks. This can lead to inefficient memory usage and allocation failures, even when there is enough total free memory. Fragmentation can be mitigated through strategies such as:

- **Memory Pooling**: Pre-allocating fixed-size blocks of memory to reduce fragmentation.
- **Garbage Collection**: Automatically reclaiming memory that is no longer in use, typically used in languages with automatic memory management like Java.

#### Combined Example of Stack and Heap Usage

Consider a more complex example that combines stack and heap usage in an ARM assembly program:

```assembly
    .data
array_size:
    .word 10

    .text
    .global _start

_start:
    // Allocate memory for an array on the heap
    LDR R0, =array_size      // Load the address of array_size into R0
    LDR R1, [R0]             // Load the value of array_size into R1
    MOV R2, #4               // Each element is 4 bytes (32 bits)
    MUL R1, R1, R2           // Calculate the total size in bytes
    BL malloc                // Allocate memory and store the pointer in R0

    CMP R0, #0               // Check if allocation was successful
    BEQ allocation_failed    // Branch if allocation failed

    // Initialize the array
    MOV R1, #0               // Initialize index register R1 to 0
initialize_loop:
    CMP R1, #10              // Compare index with array size
    BEQ initialization_done  // Branch if all elements are initialized
    STR R1, [R0, R1, LSL #2] // Store the value of R1 at address (R0 + R1 * 4)
    ADD R1, R1, #1           // Increment index
    B initialize_loop        // Repeat the loop
initialization_done:

    // Function call example (using stack)
    BL print_array           // Call print_array function
    B cleanup                // Branch to cleanup

print_array:
    PUSH {LR}                // Save return address
    MOV R1, #0               // Initialize index register R1 to 0
print_loop:
    CMP R1, #10              // Compare index with array size
    BEQ print_done           // Branch if all elements are printed
    LDR R2, [R0, R1, LSL #2] // Load the value from address (R0 + R1 * 4)
    BL print_int             // Call print_int function
    ADD R1, R1, #1           // Increment index
    B print_loop             // Repeat the loop
print_done:
    POP {LR}                 // Restore return address
    BX LR                    // Return from function

print_int:
    PUSH {LR}                // Save return address
    // Code to print integer (implementation depends on system)
    POP {LR}                 // Restore return address
    BX LR                    // Return from function

cleanup:
    // Free the allocated memory
    BL free                  // Free the memory allocated by malloc

allocation_failed:
    // Handle allocation failure (implementation depends on system)

    // Exit program (implementation depends on system)


```

In this example:

1. **Heap Allocation**: Memory for an array is dynamically allocated using `malloc`.
2. **Stack Usage**: The `print_array` function demonstrates stack usage by saving and restoring the return address and using the stack for function calls.
3. **Array Initialization and Printing**: The array is initialized and printed using indexed addressing.

#### Advanced Topics in Memory Management

##### Stack Overflow and Underflow

- **Stack Overflow**: Occurs when the stack exceeds its allocated limit, potentially overwriting adjacent memory and causing undefined behavior. It is typically caused by deep recursion or excessive local variable usage.
- **Stack Underflow**: Occurs when there are more `POP` operations than `PUSH` operations, leading to attempts to pop from an empty stack, causing undefined behavior.

##### Heap Memory Leaks

Memory leaks occur when dynamically allocated memory is not properly freed, leading to a gradual increase in memory usage and potentially exhausting available memory. Tools like `valgrind` can help detect memory leaks.

##### Memory Alignment

Memory alignment refers to the arrangement of data in memory at address boundaries that match the data's size. Proper alignment can improve performance by allowing faster memory access. ARM processors typically require data to be aligned on 4-byte boundaries for 32-bit data.

