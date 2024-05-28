\newpage

## 7. **Working with Registers**

Chapter 7 delves into the fundamental components of the ARM architecture, focusing on the various types of registers and their operations. This chapter begins with an overview of the General Purpose Registers (GPRs), which are crucial for holding temporary data and performing arithmetic and logic operations. It then explores Special Purpose Registers, including the Program Counter (PC), Stack Pointer (SP), and status registers, which play essential roles in managing program flow and system states. The chapter progresses to cover essential register operations such as loading, storing, and manipulating data, providing the foundational skills necessary for effective programming in assembly language. Finally, a comprehensive example ties together these concepts, demonstrating their practical application and reinforcing the learning through detailed explanations.

### General Purpose Registers: Overview of ARM Registers

General Purpose Registers (GPRs) are integral to the ARM architecture, serving as the primary means for data storage, manipulation, and computation within the CPU. Understanding these registers is essential for anyone aiming to master ARM assembly language, as they are the workhorses of the processor, involved in virtually every instruction executed by the CPU.

#### 1. **Introduction to ARM Registers**

ARM processors have a register-based architecture, meaning most operations are performed on data stored in registers rather than directly in memory. This design choice results in faster processing times because accessing data in registers is significantly quicker than accessing data in memory. ARM architecture defines a set of 16 to 32 general purpose registers, depending on the specific processor model and mode of operation.

#### 2. **Register Naming and Numbering Conventions**

ARM registers are typically named R0 through R15 in the basic user mode. Here's a breakdown of these registers:

- **R0-R7**: Low registers, generally used for holding temporary data and for passing function parameters.
- **R8-R12**: High registers, also used for temporary data but often have specific uses in certain conventions or compilers.
- **R13 (SP)**: Stack Pointer, which points to the top of the stack.
- **R14 (LR)**: Link Register, which holds the return address for function calls.
- **R15 (PC)**: Program Counter, which holds the address of the next instruction to be executed.

In more advanced modes, such as FIQ (Fast Interrupt Request) mode, additional banked registers are available (R8_fiq to R14_fiq), providing separate register sets to improve interrupt handling performance by reducing the need to save and restore registers.

#### 3. **Register Functions and Roles**

Each general-purpose register has a specific role, though their usage can be quite flexible depending on the programmer's needs.

##### 3.1 **R0-R3: Argument and Result Registers**

These registers are primarily used to pass arguments to functions and to return results from functions. In many ARM calling conventions, the first four arguments to a function are passed in R0 to R3. If a function returns a result, it typically places the result in R0.

##### 3.2 **R4-R11: Callee-Saved Registers**

Registers R4 to R11 are callee-saved registers, meaning that if a function uses these registers, it must save their original values and restore them before returning control to the caller. This convention helps maintain stability and predictability across function calls.

##### 3.3 **R12 (IP): Intra-Procedure-call Scratch Register**

R12, also known as the Intra-Procedure-call scratch register (IP), is often used as a scratch register that is not preserved across function calls. This register can be used by compilers for temporary storage during function prologues and epilogues.

##### 3.4 **R13 (SP): Stack Pointer**

The Stack Pointer (SP) is a special-purpose register used to manage the stack, which is a region of memory used for dynamic storage allocation during program execution. The SP points to the top of the stack, and its value changes as data is pushed onto or popped off the stack. The stack is crucial for managing function calls, local variables, and context switching.

##### 3.5 **R14 (LR): Link Register**

The Link Register (LR) holds the return address for function calls. When a function is called using the `BL` (Branch with Link) instruction, the address of the next instruction (i.e., the return address) is stored in LR. When the function completes, it typically uses the `BX LR` instruction to return to the caller.

##### 3.6 **R15 (PC): Program Counter**

The Program Counter (PC) holds the address of the currently executing instruction. It is automatically updated to point to the next instruction as each instruction is executed. Direct manipulation of the PC allows for implementing control flow changes such as branches, jumps, and function calls.

#### 4. **Register Operations**

Understanding the various operations that can be performed on registers is critical for effective ARM assembly programming.

##### 4.1 **Data Movement**

- **MOV**: Moves data from one register to another.
    - Syntax: `MOV Rd, Rn`
    - Example: `MOV R1, R2` copies the value in R2 to R1.

- **LDR/STR**: Load and Store data between registers and memory.
    - Syntax: `LDR Rd, [Rn, #offset]`
    - Example: `LDR R1, [R2, #4]` loads the value from memory at address `R2 + 4` into R1.

##### 4.2 **Arithmetic Operations**

- **ADD/SUB**: Perform addition and subtraction.
    - Syntax: `ADD Rd, Rn, Rm`
    - Example: `ADD R1, R2, R3` adds the values in R2 and R3, storing the result in R1.

- **MUL**: Multiply values.
    - Syntax: `MUL Rd, Rn, Rm`
    - Example: `MUL R1, R2, R3` multiplies the values in R2 and R3, storing the result in R1.

##### 4.3 **Logical Operations**

- **AND/ORR/EOR**: Perform bitwise logical operations.
    - Syntax: `AND Rd, Rn, Rm`
    - Example: `AND R1, R2, R3` performs a bitwise AND on the values in R2 and R3, storing the result in R1.

- **BIC**: Bit clear operation.
    - Syntax: `BIC Rd, Rn, Rm`
    - Example: `BIC R1, R2, R3` clears the bits in R2 that are set in R3, storing the result in R1.

##### 4.4 **Shift Operations**

- **LSL/LSR**: Logical shift left and right.
    - Syntax: `LSL Rd, Rn, #shift`
    - Example: `LSL R1, R2, #2` logically shifts the value in R2 left by 2 bits, storing the result in R1.

- **ASR**: Arithmetic shift right.
    - Syntax: `ASR Rd, Rn, #shift`
    - Example: `ASR R1, R2, #2` arithmetically shifts the value in R2 right by 2 bits, storing the result in R1.

#### 5. **Register Usage Conventions**

Different operating systems and application binary interfaces (ABIs) define conventions for register usage to ensure compatibility and predictability in function calls and system operations.

##### 5.1 **AAPCS (ARM Architecture Procedure Call Standard)**

The AAPCS defines the usage of registers in function calls:

- **Argument Passing**: The first four arguments to a function are passed in R0-R3. Additional arguments are passed on the stack.
- **Return Values**: The result of a function is returned in R0. If a function returns a 64-bit value, it is returned in R0 and R1.
- **Callee-Saved Registers**: R4-R11 and the stack pointer (R13) must be preserved by the callee. If a function uses these registers, it must save and restore their original values.

##### 5.2 **Stack Usage**

The stack is used for storing local variables, function parameters, and return addresses. The stack grows downward, meaning it starts at a high memory address and grows towards lower memory addresses as data is pushed onto it.

#### 6. **Combined Example with Explanation**

Consider a simple function that calculates the sum of two integers and returns the result:

```assembly
.global sum

sum:
    ; Function prologue
    PUSH {LR}        ; Save the Link Register

    ; Function body
    ADD R0, R0, R1   ; Add the values in R0 and R1, store the result in R0

    ; Function epilogue
    POP {LR}         ; Restore the Link Register
    BX LR            ; Return to the caller
```

In this example:

- The function `sum` is declared globally using `.global sum`.
- The function prologue saves the return address (LR) on the stack using `PUSH {LR}`.
- The body of the function adds the two input arguments (stored in R0 and R1) and stores the result in R0 using `ADD R0, R0, R1`.
- The function epilogue restores the return address from the stack using `POP {LR}` and returns to the caller using `BX LR`.

This example demonstrates the use of GPRs for passing function arguments, performing arithmetic operations, and managing the stack for function calls. It highlights the importance of following conventions to ensure that register values are preserved across function calls and that the program operates correctly and predictably.

### Special Purpose Registers: Program Counter, Stack Pointer, and Status Registers

In ARM architecture, Special Purpose Registers (SPRs) play critical roles in managing program execution, controlling the flow of data, and maintaining the state of the processor. Unlike General Purpose Registers (GPRs), which are mainly used for temporary data storage and computation, SPRs are designed for specific control functions. This chapter will provide an exhaustive and detailed examination of the key SPRs in ARM processors: the Program Counter (PC), the Stack Pointer (SP), and the Status Registers. These registers are fundamental to understanding how ARM processors execute instructions, manage memory, and handle various system states.

#### 1. **Program Counter (PC)**

The Program Counter (PC) is one of the most critical registers in any processor architecture, and ARM is no exception. The PC holds the address of the next instruction to be executed, thus guiding the flow of program execution.

##### 1.1 **Function and Role**

- **Instruction Fetching**: The PC points to the memory location of the instruction that the CPU will fetch and execute next. After fetching an instruction, the PC is automatically updated to point to the subsequent instruction.
- **Control Flow**: The PC is directly manipulated by branch instructions, function calls, and interrupts to alter the flow of execution. For instance, a branch instruction updates the PC to point to a different memory address, effectively jumping to a new part of the code.

##### 1.2 **Manipulation of the PC**

- **Branch Instructions**: Instructions like `B` (Branch) and `BL` (Branch with Link) modify the PC to implement jumps and function calls.
    - Example: `B label` sets the PC to the address of `label`.
    - Example: `BL func` sets the PC to the address of `func` and stores the return address in the Link Register (LR).
- **Direct Assignment**: The PC can be directly loaded with a value using the `MOV` instruction or other data movement instructions.
    - Example: `MOV PC, R0` sets the PC to the value in R0.

##### 1.3 **Pipeline Effects**

- **Prefetching**: ARM processors use pipelining, where multiple instructions are fetched, decoded, and executed in parallel. This affects the apparent value of the PC when viewed within a program. Typically, the PC points two instructions ahead of the currently executing instruction in a three-stage pipeline.

#### 2. **Stack Pointer (SP)**

The Stack Pointer (SP) is a special-purpose register used to manage the stack, which is a crucial component for function calls, local variable storage, and interrupt handling.

##### 2.1 **Function and Role**

- **Stack Management**: The SP points to the top of the stack, a contiguous block of memory used for dynamic storage allocation. The stack operates in a Last In, First Out (LIFO) manner.
- **Function Calls**: During function calls, the stack is used to store return addresses, function parameters, and local variables.
- **Interrupt Handling**: The stack is also used to save the state of the processor during interrupts, ensuring that execution can resume correctly after the interrupt is handled.

##### 2.2 **Manipulation of the SP**

- **PUSH and POP**: These pseudo-instructions are used to save and restore register values to and from the stack.
    - Example: `PUSH {R0-R3, LR}` saves R0 through R3 and the Link Register onto the stack.
    - Example: `POP {R0-R3, PC}` restores R0 through R3 and sets the PC to the saved value, effectively returning from a function.
- **Direct Modification**: The SP can be directly modified using arithmetic operations to allocate or deallocate stack space.
    - Example: `SUB SP, SP, #4` allocates 4 bytes on the stack by decrementing the SP.
    - Example: `ADD SP, SP, #4` deallocates 4 bytes from the stack by incrementing the SP.

##### 2.3 **Stack Growth Direction**

- **Downward Growth**: In ARM architecture, the stack typically grows downward, meaning the SP is decremented to allocate new stack space and incremented to deallocate stack space. This is reflected in the usage of `SUB` and `ADD` instructions to manipulate the SP.

#### 3. **Status Registers**

Status Registers in ARM architecture hold crucial information about the state of the processor and the results of various operations. The two main status registers are the Current Program Status Register (CPSR) and the Saved Program Status Register (SPSR).

##### 3.1 **Current Program Status Register (CPSR)**

The CPSR contains several fields that reflect the current state of the processor, including condition flags, interrupt masks, and processor mode bits.

- **Condition Flags**: These flags indicate the results of arithmetic and logical operations and are used for conditional execution of instructions.
    - **N (Negative)**: Set if the result of an operation is negative.
    - **Z (Zero)**: Set if the result of an operation is zero.
    - **C (Carry)**: Set if an operation results in a carry out or borrow.
    - **V (Overflow)**: Set if an operation results in an overflow.

- **Interrupt Masks**: These bits control the enabling and disabling of interrupts.
    - **I (IRQ disable)**: When set, normal interrupts are disabled.
    - **F (FIQ disable)**: When set, fast interrupts are disabled.

- **Processor Mode Bits**: These bits determine the current mode of the processor, such as User mode, Supervisor mode, or Interrupt modes (FIQ, IRQ).
    - Example: `M[4:0]` bits in the CPSR indicate the current processor mode.

##### 3.2 **Saved Program Status Register (SPSR)**

The SPSR is used to save the state of the CPSR when an exception occurs, allowing the processor to restore the original state when returning from the exception.

- **Exception Handling**: When an exception occurs, the CPSR is copied to the SPSR, and the CPSR is modified to reflect the new state required to handle the exception.
    - Example: When an IRQ occurs, the CPSR is saved to the SPSR_irq, and the CPSR is modified to disable further IRQs and switch to IRQ mode.

##### 3.3 **Manipulation of Status Registers**

- **MRS and MSR Instructions**: These instructions are used to read from and write to the CPSR and SPSR.
    - **MRS**: Move status register to register.
        - Syntax: `MRS Rd, CPSR`
        - Example: `MRS R0, CPSR` copies the CPSR to R0.
    - **MSR**: Move register to status register.
        - Syntax: `MSR CPSR_fsxc, Rn`
        - Example: `MSR CPSR_c, R0` updates the condition flags in the CPSR with the value in R0.

#### 4. **Usage and Implications of Special Purpose Registers**

##### 4.1 **Program Control and Flow**

SPRs are integral to controlling program flow and managing execution states. The PC ensures sequential execution and allows for conditional branching and function calls. The SP manages the stack, essential for nested function calls and interrupt handling, providing a mechanism for dynamic memory allocation within functions.

##### 4.2 **System State and Interrupts**

Status registers like the CPSR and SPSR are vital for maintaining system state, especially during context switches and interrupt handling. The CPSR's condition flags enable conditional execution of instructions, enhancing performance by reducing branch instructions. The interrupt masks and processor mode bits within the CPSR and SPSR facilitate efficient and predictable handling of interrupts, ensuring system stability and responsiveness.

##### 4.3 **Optimizations and Performance**

Understanding and effectively using SPRs can lead to significant optimizations in ARM assembly programming. Efficient use of the PC for branching, optimal management of the SP for stack operations, and precise control of the CPSR for condition checks and interrupts can enhance the performance and reliability of ARM-based applications.

#### 5. **Combined Example with Explanation**

Consider an example demonstrating the use of the PC, SP, and CPSR in a simple interrupt handler:

```assembly
.global main
.global irq_handler

main:
    ; Initialize stack pointer
    LDR SP, =stack_top

    ; Enable interrupts
    CPSIE I

    ; Main loop
main_loop:
    NOP
    B main_loop

irq_handler:
    ; Save context
    SUB SP, SP, #16
    STMIA SP!, {R0-R3}
    MRS R0, CPSR
    STMIA SP!, {R0}

    ; Handle interrupt
    ; (Interrupt handling code goes here)

    ; Restore context
    LDMIA SP!, {R0}
    MSR CPSR_c, R0
    LDMIA SP!, {R0-R3}
    ADD SP, SP, #16

    ; Return from interrupt
    SUBS PC, LR, #4

stack_top:
    .word 0x8000
```

In this example:

- The `main` function initializes the SP and enables interrupts using the `CPSIE I` instruction.
- The `main_loop` represents the main program loop, continuously executing a no-operation (`NOP`) instruction.
- The `irq_handler` demonstrates an interrupt handler that saves the current processor state onto the stack, handles the interrupt, and then restores the processor state before returning to the main program.
    - The context is saved by pushing R0-R3 and the CPSR onto the stack.
    - After handling the interrupt, the context is restored by popping the saved values back into the registers and the CPSR.
- The `SUBS PC, LR, #4` instruction ensures the correct return to the interrupted code by adjusting the PC.

### Register Operations: Loading, Storing, and Manipulating Data in Registers

Register operations are the cornerstone of ARM assembly programming, involving a range of instructions for loading data into registers, storing data from registers to memory, and manipulating data within registers. These operations are fundamental for executing any meaningful computation or control flow in a program. This chapter provides an exhaustive and detailed exploration of the various types of register operations in ARM architecture, including loading, storing, and manipulating data.

#### 1. **Loading Data into Registers**

Loading data into registers is a crucial operation that transfers data from memory or immediate values directly into the registers for further processing.

##### 1.1 **MOV (Move) Instruction**

The `MOV` instruction copies a value into a register. This value can be an immediate value or the content of another register.

- **Syntax**: `MOV Rd, Operand`
- **Operands**:
    - `Rd`: Destination register
    - `Operand`: Immediate value or register

- **Examples**:
    - `MOV R1, #10` : Moves the immediate value `10` into register `R1`.
    - `MOV R2, R3` : Copies the value in register `R3` into register `R2`.

##### 1.2 **MVN (Move Not) Instruction**

The `MVN` instruction moves the bitwise NOT of an operand into a register.

- **Syntax**: `MVN Rd, Operand`
- **Operands**:
    - `Rd`: Destination register
    - `Operand`: Immediate value or register

- **Example**:
    - `MVN R1, #0xFF` : Moves the bitwise NOT of `0xFF` (which is `0xFFFFFF00` in a 32-bit register) into register `R1`.

##### 1.3 **LDR (Load Register) Instruction**

The `LDR` instruction loads a word from memory into a register.

- **Syntax**: `LDR Rd, [Rn, Offset]`
- **Operands**:
    - `Rd`: Destination register
    - `Rn`: Base register containing the base address
    - `Offset`: Immediate value or register that specifies the offset from the base address

- **Examples**:
    - `LDR R1, [R2, #4]` : Loads the word at the address `R2 + 4` into register `R1`.
    - `LDR R3, [R4, R5]` : Loads the word at the address `R4 + R5` into register `R3`.

##### 1.4 **LDRH (Load Register Halfword) and LDRB (Load Register Byte) Instructions**

These instructions load halfwords (16 bits) and bytes (8 bits) from memory into registers.

- **Syntax**:
    - `LDRH Rd, [Rn, Offset]`
    - `LDRB Rd, [Rn, Offset]`

- **Examples**:
    - `LDRH R1, [R2, #2]` : Loads the halfword at `R2 + 2` into the lower 16 bits of `R1`.
    - `LDRB R1, [R2, #1]` : Loads the byte at `R2 + 1` into the lower 8 bits of `R1`.

#### 2. **Storing Data from Registers to Memory**

Storing data from registers to memory is essential for preserving the state of a program, especially for passing data between functions and for working with data structures.

##### 2.1 **STR (Store Register) Instruction**

The `STR` instruction stores a word from a register into memory.

- **Syntax**: `STR Rd, [Rn, Offset]`
- **Operands**:
    - `Rd`: Source register
    - `Rn`: Base register containing the base address
    - `Offset`: Immediate value or register that specifies the offset from the base address

- **Examples**:
    - `STR R1, [R2, #4]` : Stores the word in `R1` at the address `R2 + 4`.
    - `STR R3, [R4, R5]` : Stores the word in `R3` at the address `R4 + R5`.

##### 2.2 **STRH (Store Register Halfword) and STRB (Store Register Byte) Instructions**

These instructions store halfwords (16 bits) and bytes (8 bits) from registers to memory.

- **Syntax**:
    - `STRH Rd, [Rn, Offset]`
    - `STRB Rd, [Rn, Offset]`

- **Examples**:
    - `STRH R1, [R2, #2]` : Stores the lower 16 bits of `R1` at `R2 + 2`.
    - `STRB R1, [R2, #1]` : Stores the lower 8 bits of `R1` at `R2 + 1`.

#### 3. **Manipulating Data in Registers**

Manipulating data within registers is the core of computational tasks in ARM assembly programming. These operations include arithmetic, logical, and shift operations.

##### 3.1 **Arithmetic Operations**

###### 3.1.1 **ADD (Add) and SUB (Subtract) Instructions**

- **ADD Syntax**: `ADD Rd, Rn, Operand`
- **SUB Syntax**: `SUB Rd, Rn, Operand`
- **Operands**:
    - `Rd`: Destination register
    - `Rn`: First operand register
    - `Operand`: Second operand, which can be an immediate value or a register

- **Examples**:
    - `ADD R1, R2, #5` : Adds `5` to the value in `R2` and stores the result in `R1`.
    - `SUB R3, R4, R5` : Subtracts the value in `R5` from `R4` and stores the result in `R3`.

###### 3.1.2 **ADC (Add with Carry) and SBC (Subtract with Carry) Instructions**

- **ADC Syntax**: `ADC Rd, Rn, Operand`
- **SBC Syntax**: `SBC Rd, Rn, Operand`
- **Operands**:
    - `Rd`: Destination register
    - `Rn`: First operand register
    - `Operand`: Second operand, which can be an immediate value or a register

- **Examples**:
    - `ADC R1, R2, #5` : Adds `5`, the value in `R2`, and the carry flag, then stores the result in `R1`.
    - `SBC R3, R4, R5` : Subtracts the value in `R5` and the carry flag from `R4` and stores the result in `R3`.

###### 3.1.3 **MUL (Multiply) and MLA (Multiply Accumulate) Instructions**

- **MUL Syntax**: `MUL Rd, Rn, Rm`
- **MLA Syntax**: `MLA Rd, Rn, Rm, Ra`
- **Operands**:
    - `Rd`: Destination register
    - `Rn`, `Rm`: Operand registers
    - `Ra`: Accumulate register (for MLA only)

- **Examples**:
    - `MUL R1, R2, R3` : Multiplies the values in `R2` and `R3` and stores the result in `R1`.
    - `MLA R4, R5, R6, R7` : Multiplies the values in `R5` and `R6`, adds the result to the value in `R7`, and stores the result in `R4`.

##### 3.2 **Logical Operations**

###### 3.2.1 **AND, ORR, and EOR (XOR) Instructions**

- **AND Syntax**: `AND Rd, Rn, Operand`
- **ORR Syntax**: `ORR Rd, Rn, Operand`
- **EOR Syntax**: `EOR Rd, Rn, Operand`
- **Operands**:
    - `Rd`: Destination register
    - `Rn`: First operand register
    - `Operand`: Second operand, which can be an immediate value or a register

- **Examples**:
    - `AND R1, R2, #0xFF` : Performs a bitwise AND between the value in `R2` and `0xFF`, storing the result in `R1`.
    - `ORR R3, R4, R5` : Performs a bitwise OR between the values in `R4` and `R5`, storing the result in `R3`.
    - `EOR R6, R7, #0x1` : Performs a bitwise XOR between the value in `R7` and `0x1`, storing the result in `R6`.

###### 3.2.2 **BIC (Bit Clear) Instruction**

The `BIC` instruction performs a bitwise AND of a register with the bitwise NOT of an operand.

- **Syntax**: `BIC Rd, Rn, Operand`
- **Operands**:
    - `Rd`: Destination register
    - `Rn`: First operand register
    - `Operand`: Second operand, which can be an immediate value or a register

- **Example**:
    - `BIC R1, R2, #0xFF` : Clears the lower 8 bits of `R2` and stores the result in `R1`.

##### 3.3 **Shift and Rotate Operations**

Shift and rotate operations move the bits within a register to the left or right.

###### 3.3.1 **LSL (Logical Shift Left) and LSR (Logical Shift Right) Instructions**

- **LSL Syntax**: `LSL Rd, Rn, #Shift`
- **LSR Syntax**: `LSR Rd, Rn, #Shift`
- **Operands**:
    - `Rd`: Destination register
    - `Rn`: Operand register
    - `#Shift`: Number of bit positions to shift

- **Examples**:
    - `LSL R1, R2, #2` : Shifts the bits in `R2` left by 2 positions, filling the rightmost bits with zeros, and stores the result in `R1`.
    - `LSR R3, R4, #3` : Shifts the bits in `R4` right by 3 positions, filling the leftmost bits with zeros, and stores the result in `R3`.

###### 3.3.2 **ASR (Arithmetic Shift Right) Instruction**

The `ASR` instruction performs a right shift, preserving the sign bit (the leftmost bit).

- **Syntax**: `ASR Rd, Rn, #Shift`
- **Operands**:
    - `Rd`: Destination register
    - `Rn`: Operand register
    - `#Shift`: Number of bit positions to shift

- **Example**:
    - `ASR R1, R2, #1` : Shifts the bits in `R2` right by 1 position, preserving the sign bit, and stores the result in `R1`.

###### 3.3.3 **ROR (Rotate Right) Instruction**

The `ROR` instruction rotates the bits in a register to the right.

- **Syntax**: `ROR Rd, Rn, #Shift`
- **Operands**:
    - `Rd`: Destination register
    - `Rn`: Operand register
    - `#Shift`: Number of bit positions to rotate

- **Example**:
    - `ROR R1, R2, #4` : Rotates the bits in `R2` right by 4 positions and stores the result in `R1`.

#### 4. **Combined Example with Explanation**

Consider an example that demonstrates loading, storing, and manipulating data in registers:

```assembly
.global main

main:
    ; Initialize values
    MOV R0, #10       ; R0 = 10
    MOV R1, #20       ; R1 = 20
    MOV R2, #0xFF     ; R2 = 0xFF
    LDR R3, =0x1000   ; Load address 0x1000 into R3

    ; Store values in memory
    STR R0, [R3]      ; Store R0 at address 0x1000
    STR R1, [R3, #4]  ; Store R1 at address 0x1004

    ; Load values from memory
    LDR R4, [R3]      ; Load the value at 0x1000 into R4
    LDR R5, [R3, #4]  ; Load the value at 0x1004 into R5

    ; Perform arithmetic operations
    ADD R6, R4, R5    ; R6 = R4 + R5
    SUB R7, R5, R4    ; R7 = R5 - R4

    ; Perform logical operations
    AND R8, R2, R4    ; R8 = R2 AND R4
    ORR R9, R2, R5    ; R9 = R2 OR R5
    EOR R10, R2, R6   ; R10 = R2 XOR R6
    BIC R11, R2, R7   ; R11 = R2 AND NOT R7

    ; Perform shift operations
    LSL R12, R4, #1   ; R12 = R4 << 1
    LSR R13, R5, #2   ; R13 = R5 >> 2
    ASR R14, R6, #1   ; R14 = R6 >> 1 (arithmetic shift)
    ROR R15, R7, #4   ; R15 = R7 rotated right by 4

    ; Infinite loop
    B .               ; Loop indefinitely
```

In this example:

- **Loading Data**:
    - Immediate values are loaded into registers R0, R1, and R2 using the `MOV` instruction.
    - An address is loaded into R3 using the `LDR` pseudo-instruction with an immediate value.
- **Storing Data**:
    - Values from R0 and R1 are stored into memory at the addresses specified by R3 and `R3 + 4` using the `STR` instruction.
- **Loading from Memory**:
    - Values are loaded from memory into R4 and R5 using the `LDR` instruction.
- **Arithmetic Operations**:
    - Addition and subtraction are performed using `ADD` and `SUB`, with results stored in R6 and R7.
- **Logical Operations**:
    - Bitwise operations are performed using `AND`, `ORR`, `EOR`, and `BIC`, with results stored in R8 to R11.
- **Shift Operations**:
    - Shift operations are performed using `LSL`, `LSR`, `ASR`, and `ROR`, with results stored in R12 to R15.
- **Control Flow**:
    - An infinite loop is created using the `B .` instruction, causing the program to continuously branch to itself.

