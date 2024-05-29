\newpage

# Part III: Advanced Assembly Programming Techniques

## 9. **Subroutines and Functions**

Chapter 9 delves into the essential concepts of creating reusable code blocks in Assembly Language and ARM Architecture. This chapter begins by exploring the fundamentals of defining and calling subroutines, which are critical for structuring efficient and maintainable code. It further examines various methods for parameter passing, ensuring that arguments are correctly conveyed to subroutines. Additionally, the chapter covers the intricate details of managing stack frames and local variables, providing a comprehensive understanding of how function calls and local data are handled. To solidify these concepts, a combined example with a detailed explanation is included, demonstrating practical implementation and reinforcing the chapter's teachings.

### Defining and Calling Subroutines

Subroutines, also known as procedures or functions, are fundamental to programming, providing a way to organize code into manageable, reusable blocks. In the context of Assembly Language and ARM Architecture, understanding how to define and call subroutines is crucial for writing efficient and maintainable code. This chapter delves deeply into the intricacies of subroutines, covering their definition, calling conventions, and practical applications.

#### **1. Introduction to Subroutines**

A subroutine is a self-contained sequence of instructions that performs a specific task. Once defined, a subroutine can be invoked, or called, from various points within a program. This modular approach enhances code readability, reduces redundancy, and simplifies debugging and maintenance.

#### **2. Defining Subroutines**

To define a subroutine in ARM Assembly Language, you create a labeled block of code. The label serves as the entry point for the subroutine. Here is the basic structure of a subroutine:

```assembly
my_subroutine:
    ; Subroutine code goes here
    bx lr ; Return from subroutine
```

- **Label**: `my_subroutine` is the name of the subroutine.
- **Instructions**: The body of the subroutine contains the instructions that perform the task.
- **Return Instruction**: `bx lr` (branch and exchange to the link register) is used to return to the calling code.

#### **3. Calling Subroutines**

Subroutines are called using the `bl` (branch with link) instruction, which branches to the subroutine's address and stores the return address in the link register (`lr`). For example:

```assembly
bl my_subroutine ; Call my_subroutine
```

When `bl my_subroutine` is executed, the processor jumps to the address labeled `my_subroutine` and continues executing from there. After the subroutine finishes, it returns to the instruction following the `bl` call.

#### **4. Parameter Passing**

Subroutines often need to operate on data provided by the caller. There are several methods to pass parameters to a subroutine:

1. **Registers**: The most common and efficient method. ARM calling conventions typically use registers `r0` to `r3` for the first four parameters.
2. **Stack**: When more than four parameters are needed or if the parameters do not fit in registers, they are passed via the stack.

Here is an example of passing parameters through registers:

```assembly
mov r0, #10    ; First parameter
mov r1, #20    ; Second parameter
bl add_numbers ; Call subroutine
```

And the corresponding subroutine definition:

```assembly
add_numbers:
    add r0, r0, r1 ; Add the parameters
    bx lr          ; Return with the result in r0
```

#### **5. Stack Frames and Local Variables**

Subroutines often require local variables, which can be managed using the stack. This involves creating a stack frame, which is a section of the stack that stores local variables, parameters, and return addresses.

##### **5.1 Creating a Stack Frame**

A typical stack frame creation involves:

- **Prologue**: Code at the beginning of the subroutine that sets up the stack frame.
- **Epilogue**: Code at the end that cleans up the stack frame.

Here is an example of a subroutine with a stack frame:

```assembly
my_subroutine:
    push {lr}        ; Save the return address
    sub sp, sp, #8   ; Allocate space for local variables
    ; Subroutine code
    add sp, sp, #8   ; Deallocate local variables
    pop {lr}         ; Restore the return address
    bx lr            ; Return from subroutine
```

##### **5.2 Local Variables**

Local variables are stored in the allocated stack space. Accessing these variables involves calculating their offsets from the stack pointer (`sp`).

For instance, if a local variable is stored at `sp-4`, you can access it as follows:

```assembly
my_subroutine:
    push {lr}
    sub sp, sp, #8
    str r0, [sp, #4]  ; Store r0 at sp+4 (first local variable)
    ldr r1, [sp, #4]  ; Load the first local variable into r1
    add sp, sp, #8
    pop {lr}
    bx lr
```

#### **6. Managing Function Calls and Local Data**

Efficiently managing function calls and local data is essential for writing optimized ARM Assembly code. This involves:

1. **Minimizing Register Usage**: Use registers efficiently to avoid excessive stack operations.
2. **Proper Stack Management**: Ensure that the stack is correctly maintained to prevent corruption and ensure proper return address handling.
3. **Using Calling Conventions**: Adhere to ARM's calling conventions to ensure compatibility with other code and libraries.

##### **6.1 ARM Calling Conventions**

ARM's Procedure Call Standard (AAPCS) defines the conventions for passing arguments, returning values, and using registers. Key points include:

- **Registers r0-r3**: Used for parameter passing and return values.
- **Registers r4-r11**: Callee-saved registers, which must be preserved by the called function.
- **Registers r12-r15**: Special-purpose registers, where `r13` is the stack pointer (`sp`), `r14` is the link register (`lr`), and `r15` is the program counter (`pc`).

Following these conventions ensures that your code is interoperable with other functions and libraries.

#### **7. Combined Example with Explanation**

To illustrate the concepts discussed, let's look at a combined example:

```assembly
.global main

main:
    mov r0, #5        ; First parameter
    mov r1, #10       ; Second parameter
    bl multiply       ; Call multiply subroutine
    b done            ; End of main

multiply:
    push {lr}         ; Save return address
    mul r0, r0, r1    ; Multiply r0 by r1, result in r0
    pop {lr}          ; Restore return address
    bx lr             ; Return to caller

done:
    b done            ; Infinite loop to end program
```

**Explanation**:

1. **Main Routine**:
    - Sets up the parameters `r0` and `r1` with values `5` and `10`, respectively.
    - Calls the `multiply` subroutine.

2. **Multiply Subroutine**:
    - Saves the return address by pushing `lr` onto the stack.
    - Performs the multiplication `r0 = r0 * r1`, storing the result in `r0`.
    - Restores the return address by popping `lr` from the stack.
    - Returns to the caller using `bx lr`.

3. **End of Main**:
    - Enters an infinite loop to signal the end of the program.

This example showcases the fundamental steps in defining and calling subroutines, parameter passing, and managing the stack. By adhering to these principles, you can create efficient, modular, and maintainable ARM Assembly programs.

### Parameter Passing

Parameter passing is a critical concept in programming, allowing data to be provided to subroutines or functions for processing. In Assembly Language and ARM Architecture, there are several methods for passing parameters to subroutines, each with its own advantages and use cases. This chapter explores these methods in detail, including passing parameters through registers, the stack, and memory, along with best practices and examples.

#### **1. Introduction to Parameter Passing**

When a subroutine is called, it often needs to operate on data supplied by the caller. This data, known as parameters or arguments, can be passed using various methods. Efficient parameter passing is essential for optimizing performance and ensuring the correct operation of the subroutine.

#### **2. Passing Parameters Through Registers**

Passing parameters through registers is the most efficient method, as it involves direct access to the CPU's fastest storage. ARM architecture, following the ARM Procedure Call Standard (AAPCS), uses registers `r0` to `r3` for passing the first four arguments to a subroutine.

##### **2.1 Register Usage Conventions**

- **Registers r0-r3**: Used for passing arguments and returning values. These are caller-saved registers.
- **Registers r4-r11**: Callee-saved registers, used for preserving values across subroutine calls.

Here's an example of passing parameters through registers:

```assembly
.global main

main:
    mov r0, #5        ; First parameter
    mov r1, #10       ; Second parameter
    bl add            ; Call add subroutine
    b done            ; End of main

add:
    add r0, r0, r1    ; Add r0 and r1, result in r0
    bx lr             ; Return with result in r0

done:
    b done            ; Infinite loop to end program
```

In this example:
- The `main` routine sets up two parameters in `r0` and `r1`.
- The `add` subroutine adds these parameters and returns the result in `r0`.

#### **3. Passing Parameters Through the Stack**

When more than four parameters are needed or when parameters do not fit in registers, they can be passed through the stack. This method involves pushing parameters onto the stack before calling the subroutine and popping them off inside the subroutine.

##### **3.1 Stack Management**

Using the stack for parameter passing requires careful management to ensure data integrity and proper execution flow.

- **Prologue**: The code at the beginning of the subroutine that sets up the stack frame.
- **Epilogue**: The code at the end that cleans up the stack frame.

Example of passing parameters through the stack:

```assembly
.global main

main:
    mov r0, #5         ; First parameter
    mov r1, #10        ; Second parameter
    push {r0, r1}      ; Push parameters onto stack
    bl multiply        ; Call multiply subroutine
    add sp, sp, #8     ; Clean up stack
    b done             ; End of main

multiply:
    push {lr}          ; Save return address
    ldr r0, [sp, #4]   ; Load first parameter
    ldr r1, [sp, #0]   ; Load second parameter
    mul r0, r0, r1     ; Multiply parameters, result in r0
    pop {lr}           ; Restore return address
    bx lr              ; Return with result in r0

done:
    b done             ; Infinite loop to end program
```

In this example:
- The `main` routine pushes parameters onto the stack.
- The `multiply` subroutine retrieves the parameters from the stack, performs the multiplication, and returns the result.

#### **4. Passing Parameters Through Memory**

In some cases, parameters may be passed through memory, particularly when dealing with large data structures like arrays or structures. This method involves passing the address (pointer) of the data rather than the data itself.

##### **4.1 Memory Addressing**

Passing parameters through memory typically involves loading the address of the data into a register and passing that register to the subroutine.

Example of passing parameters through memory:

```assembly
.data
array:
    .word 1, 2, 3, 4, 5

.text
.global main

main:
    ldr r0, =array     ; Load address of array into r0
    mov r1, #5         ; Length of array
    bl sum_array       ; Call sum_array subroutine
    b done             ; End of main

sum_array:
    push {lr}          ; Save return address
    mov r2, #0         ; Initialize sum to 0
    mov r3, #0         ; Initialize index to 0

loop:
    ldr r4, [r0, r3, lsl #2] ; Load array element
    add r2, r2, r4           ; Add element to sum
    add r3, r3, #1           ; Increment index
    cmp r3, r1               ; Compare index with length
    blt loop                 ; Repeat if index < length

    mov r0, r2               ; Move sum to r0
    pop {lr}                 ; Restore return address
    bx lr                    ; Return with sum in r0

done:
    b done                   ; Infinite loop to end program
```

In this example:
- The `main` routine loads the address of the array into `r0` and the length into `r1`.
- The `sum_array` subroutine calculates the sum of the array elements and returns the result.

#### **5. Combining Methods for Complex Parameter Passing**

In real-world applications, it is common to combine different methods of parameter passing to handle complex data structures and multiple parameters efficiently. For example, you might pass the first few parameters through registers and additional parameters through the stack or memory.

Example of combined parameter passing:

```assembly
.data
array:
    .word 1, 2, 3, 4, 5

.text
.global main

main:
    ldr r0, =array     ; Load address of array into r0
    mov r1, #5         ; Length of array
    mov r2, #10        ; Multiplier
    bl process_array   ; Call process_array subroutine
    b done             ; End of main

process_array:
    push {r4-r6, lr}   ; Save registers and return address
    mov r4, r2         ; Move multiplier to r4
    mov r5, #0         ; Initialize sum to 0
    mov r6, #0         ; Initialize index to 0

loop:
    ldr r3, [r0, r6, lsl #2] ; Load array element
    mul r3, r3, r4           ; Multiply element by multiplier
    add r5, r5, r3           ; Add to sum
    add r6, r6, #1           ; Increment index
    cmp r6, r1               ; Compare index with length
    blt loop                 ; Repeat if index < length

    mov r0, r5               ; Move sum to r0
    pop {r4-r6, lr}          ; Restore registers and return address
    bx lr                    ; Return with sum in r0

done:
    b done                   ; Infinite loop to end program
```

In this example:
- The `main` routine passes the array address and length through registers and the multiplier through a register.
- The `process_array` subroutine uses a combination of register and memory-based parameter passing to process the array.

#### **6. Best Practices for Parameter Passing**

To ensure efficient and maintainable code, consider the following best practices for parameter passing in ARM Assembly:

1. **Use Registers Whenever Possible**: Registers provide the fastest access and are the preferred method for passing a small number of parameters.
2. **Minimize Stack Usage**: Use the stack for additional parameters or when dealing with larger data structures to avoid register overflow.
3. **Follow Calling Conventions**: Adhering to ARM's calling conventions ensures compatibility with other code and libraries.
4. **Manage Stack Frames Properly**: Ensure that the stack is correctly maintained to prevent corruption and ensure proper return address handling.
5. **Optimize Memory Access**: When passing large data structures, pass pointers to memory rather than copying data, reducing memory footprint and improving performance.

#### **7. Advanced Techniques in Parameter Passing**

For advanced applications, additional techniques such as parameter passing through global variables, inline assembly, and interworking with high-level languages can be utilized.

##### **7.1 Global Variables**

In some scenarios, global variables can be used to pass parameters between subroutines, especially when dealing with data that needs to be accessed by multiple parts of the program.

Example:

```assembly
.data
global_var:
    .word 0

.text
.global main

main:
    ldr r0, =global_var
    mov r1, #123
    str r1, [r0]
    bl read_global
    b done

read_global:
    ldr r0, =global_var
    ldr r1, [r0]
    bx lr

done:
    b done
```

##### **7.2 Inline Assembly**

Inline assembly allows mixing assembly code within high-level languages like C, providing flexibility in parameter passing and leveraging the strengths of both languages.

Example (in C with ARM assembly):

```c
#include <stdio.h>

int multiply(int a, int b) {
    int result;
    asm ("mul %0, %1, %2"
         : "=r" (result)
         : "r" (a), "r" (b));
    return result;
}

int main() {
    int a = 5, b = 10;
    printf("Result: %d\n", multiply(a, b));
    return 0;
}
```

##### **7.3 Interworking with High-Level Languages**

Interworking allows ARM assembly routines to be called from high-level languages and vice versa. This is particularly useful for performance-critical code sections.

Example (calling ARM assembly from C):

```assembly
.global add

add:
    add r0, r0, r1
    bx lr
```

```c
#include <stdio.h>

extern int add(int a, int b);

int main() {
    int result = add(5, 10);
    printf("Result: %d\n", result);
    return 0;
}
```

### Stack Frames and Local Variables

In Assembly Language and ARM Architecture, managing function calls and local data efficiently is essential for writing robust and maintainable code. This involves understanding the concepts of stack frames and local variables. Stack frames allow us to manage the function call process, including parameter passing, return addresses, and local variables. This chapter provides an exhaustive exploration of these concepts, focusing on the ARM architecture.

#### **1. Introduction to Stack Frames**

A stack frame is a block of memory on the stack that contains all the information required for a single function call. This typically includes:

- **Return Address**: The address to return to after the function completes.
- **Saved Registers**: Registers that need to be preserved across function calls.
- **Local Variables**: Variables that are local to the function.
- **Parameters**: Arguments passed to the function (when more than the register can hold).

#### **2. Anatomy of a Stack Frame**

When a function is called, a new stack frame is created. The stack grows downwards (towards lower memory addresses) on ARM architectures. Let's break down a typical stack frame:

```assembly
------------------------
| Saved registers      |
|----------------------|
| Return address (LR)  |
|----------------------|
| Local variables      |
|----------------------|
| Parameters           |
|----------------------|
| Previous stack frame |
------------------------
```

##### **2.1 Stack Pointer (SP) and Frame Pointer (FP)**

- **Stack Pointer (SP)**: Points to the top of the stack. It is updated as data is pushed to or popped from the stack.
- **Frame Pointer (FP)**: Also known as the base pointer (BP), it points to a fixed location within the stack frame, making it easier to access local variables and parameters.

##### **2.2 Creating and Destroying Stack Frames**

Creating a stack frame involves the following steps:

1. **Save the return address**: Push the link register (LR) onto the stack.
2. **Save the current frame pointer**: Push the current frame pointer (FP) onto the stack.
3. **Set the new frame pointer**: Update the frame pointer to the current stack pointer.
4. **Allocate space for local variables**: Adjust the stack pointer to allocate space.

Destroying a stack frame involves the reverse process:

1. **Deallocate space for local variables**: Adjust the stack pointer.
2. **Restore the frame pointer**: Pop the old frame pointer from the stack.
3. **Restore the return address**: Pop the link register (LR) from the stack.

Here is an example of creating and destroying a stack frame in ARM Assembly:

```assembly
function:
    push {fp, lr}       ; Save the frame pointer and return address
    add fp, sp, #4      ; Set the new frame pointer
    sub sp, sp, #16     ; Allocate space for local variables (e.g., 4 words)
    ; Function body
    add sp, sp, #16     ; Deallocate space for local variables
    pop {fp, lr}        ; Restore the frame pointer and return address
    bx lr               ; Return from function
```

#### **3. Managing Local Variables**

Local variables are stored within the stack frame of the function. This ensures that each function call has its own set of local variables, preventing conflicts between calls.

##### **3.1 Accessing Local Variables**

Local variables are accessed using offsets from the frame pointer. For example, if the frame pointer is at `fp`, and we have allocated 16 bytes (4 words) for local variables, the first local variable might be at `[fp, #-4]`, the second at `[fp, #-8]`, and so on.

Example:

```assembly
function:
    push {fp, lr}
    add fp, sp, #4
    sub sp, sp, #16     ; Allocate space for 4 local variables

    mov r0, #10
    str r0, [fp, #-4]   ; Store 10 in the first local variable
    ldr r1, [fp, #-4]   ; Load the first local variable into r1

    add sp, sp, #16
    pop {fp, lr}
    bx lr
```

In this example:
- `str r0, [fp, #-4]` stores the value in `r0` into the first local variable.
- `ldr r1, [fp, #-4]` loads the value of the first local variable into `r1`.

#### **4. Managing Parameters**

When more parameters are passed to a function than can be held in the registers `r0-r3`, they are typically passed on the stack. These parameters are also accessed using offsets from the frame pointer.

##### **4.1 Accessing Parameters on the Stack**

Example:

```assembly
function:
    push {fp, lr}
    add fp, sp, #4
    sub sp, sp, #16     ; Allocate space for local variables

    ldr r0, [fp, #8]    ; Load the first parameter (passed on the stack) into r0
    ldr r1, [fp, #12]   ; Load the second parameter into r1

    add sp, sp, #16
    pop {fp, lr}
    bx lr
```

In this example:
- Parameters are accessed using positive offsets from `fp`. The exact offset depends on the number of saved registers and local variables.

#### **5. Practical Example with Stack Frames and Local Variables**

To illustrate the concepts discussed, let's consider a more comprehensive example:

```assembly
.global main

main:
    mov r0, #5        ; First parameter
    mov r1, #10       ; Second parameter
    bl compute        ; Call compute subroutine
    b done            ; End of main

compute:
    push {fp, lr}     ; Save the frame pointer and return address
    add fp, sp, #4    ; Set the new frame pointer
    sub sp, sp, #16   ; Allocate space for local variables

    str r0, [fp, #-4] ; Store the first parameter in the first local variable
    str r1, [fp, #-8] ; Store the second parameter in the second local variable

    ldr r0, [fp, #-4] ; Load the first local variable into r0
    ldr r1, [fp, #-8] ; Load the second local variable into r1

    add r2, r0, r1    ; Perform some operation (e.g., addition)
    str r2, [fp, #-12]; Store the result in the third local variable

    ldr r0, [fp, #-12]; Load the result into r0 to return it

    add sp, sp, #16   ; Deallocate space for local variables
    pop {fp, lr}      ; Restore the frame pointer and return address
    bx lr             ; Return from subroutine

done:
    b done            ; Infinite loop to end program
```

In this example:
- The `main` routine sets up parameters and calls the `compute` subroutine.
- The `compute` subroutine creates a stack frame, stores parameters in local variables, performs an operation, and returns the result.

#### **6. Advanced Techniques for Stack Frame Management**

Advanced techniques include optimizing stack frame usage, using frame pointer elimination, and handling dynamic memory allocation within functions.

##### **6.1 Frame Pointer Elimination**

In some cases, the frame pointer can be eliminated to save a register and reduce overhead. This technique is known as frame pointer omission (FPO) or frame pointer elimination (FPE).

Example without a frame pointer:

```assembly
function:
    push {lr}
    sub sp, sp, #16     ; Allocate space for local variables

    ; Function body

    add sp, sp, #16     ; Deallocate space for local variables
    pop {lr}            ; Restore the return address
    bx lr               ; Return from function
```

##### **6.2 Dynamic Memory Allocation**

When dealing with dynamic data, functions may need to allocate and deallocate memory at runtime. This involves using system calls or runtime libraries to manage heap memory.

Example using `malloc` and `free` in ARM Assembly (assuming an ARM system with C runtime):

```assembly
.extern malloc
.extern free

.global main

main:
    mov r0, #100       ; Size of memory to allocate
    bl malloc          ; Call malloc
    mov r4, r0         ; Save the allocated memory address in r4

    ; Use the allocated memory (r4 points to the allocated block)

    mov r0, r4         ; Prepare the pointer for free
    bl free            ; Call free to deallocate memory
    b done             ; End of main

done:
    b done             ; Infinite loop to end program
```

In this example:
- `malloc` is used to allocate 100 bytes of memory, and the address is saved in `r4`.
- `free` is called to deallocate the memory.

#### **7. Interworking with High-Level Languages**

ARM Assembly functions can interwork with high-level languages such as C. This requires adherence to calling conventions and proper management of the stack and registers.

Example of calling an ARM Assembly function from C:

```assembly
.global add

add:
    add r0, r0, r1
    bx lr
```

C code:

```c
#include <stdio.h>

extern int add(int a, int b);

int main() {
    int result = add(5, 10);
    printf("Result: %d\n", result);
    return 0;
}
```

In this example:
- The `add` function is defined in ARM Assembly and follows the ARM calling conventions.
- The C code calls the `add` function and prints the result.

#### **8. Debugging and Optimization**

Efficient stack frame management is crucial for debugging and optimization. Tools such as debuggers and profilers can help identify issues and optimize function calls and stack usage.

##### **8.1 Debugging with Stack Frames**

Debuggers use stack frames to provide a call stack trace, helping to diagnose issues such as stack overflows, invalid memory access, and incorrect function calls.

##### **8.2 Optimizing Stack Usage**

- **Minimize Local Variables**: Reduce the number of local variables to decrease stack frame size.
- **Use Registers Efficiently**: Utilize registers for temporary storage and intermediate calculations.
- **Inline Functions**: Inline small functions to eliminate the overhead of function calls and stack frame management.

