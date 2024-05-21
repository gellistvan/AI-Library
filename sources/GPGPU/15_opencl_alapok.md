\newpage

## 15 OpenCL Programozási Alapok

Az OpenCL (Open Computing Language) lehetővé teszi a párhuzamos számítások végrehajtását különböző hardvereszközökön, mint például CPU-kon, GPU-kon és más típusú processzorokon. Ebben a fejezetben az OpenCL programozás alapjait fogjuk megvizsgálni, beleértve az OpenCL program struktúráját, a memória kezelését és a kernel írásának és futtatásának folyamatát. Részletesen bemutatjuk, hogyan választhatunk platformot és eszközt, hogyan hozhatunk létre kontextust és kezelhetjük a parancs sort, valamint hogyan allokálhatunk memóriát, és hogyan vihetünk át adatokat a host és az eszköz között. Ezenkívül bemutatjuk, hogyan írhatunk és fordíthatunk kernel forráskódot, valamint hogyan futtathatjuk azt hatékonyan.

### 15.1 OpenCL program struktúrája

Az OpenCL programok alapvető struktúrája több lépésből áll. Először ki kell választani a platformot és az eszközt, majd létre kell hozni a kontextust és a parancs sort. Ezek az alapvető lépések biztosítják, hogy az OpenCL program megfelelően tudjon kommunikálni a hardverrel és végrehajtani a szükséges számításokat.

#### Platform és eszköz kiválasztása

Az OpenCL programok futtatása előtt ki kell választani a megfelelő platformot és eszközt. Az OpenCL API biztosítja a szükséges függvényeket a rendelkezésre álló platformok és eszközök listázására és kiválasztására.

```c
#include <CL/cl.h>
#include <stdio.h>

int main() {
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_platforms;
    cl_uint ret_num_devices;
    cl_int ret;

    // Get Platform Info
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (ret != CL_SUCCESS) {
        printf("Failed to find an OpenCL platform.\n");
        return 1;
    }

    // Get Device Info
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    if (ret != CL_SUCCESS) {
        printf("Failed to find an OpenCL device.\n");
        return 1;
    }

    printf("Platform and device selected successfully.\n");
    return 0;
}
```

#### Kontextus létrehozása és parancs sor kezelése

A platform és eszköz kiválasztása után létre kell hoznunk egy kontextust és egy parancs sort. A kontextus egy olyan környezet, amelyben az OpenCL objektumok (például memóriakönyvtárak, programok és kernel-ek) élnek. A parancs sor lehetővé teszi a feladatok végrehajtását az OpenCL eszközön.

```c
int main() {
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_uint ret_num_platforms;
    cl_uint ret_num_devices;
    cl_int ret;

    // Get Platform and Device Info
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    // Create OpenCL Context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS) {
        printf("Failed to create OpenCL context.\n");
        return 1;
    }

    // Create Command Queue
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    if (ret != CL_SUCCESS) {
        printf("Failed to create command queue.\n");
        return 1;
    }

    printf("Context and command queue created successfully.\n");

    // Cleanup
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}
```

### 15.2 Memória kezelés

Az OpenCL programokban a memória kezelés kulcsfontosságú szerepet játszik. Az adatok host és eszköz közötti átvitele, valamint a memória allokációja és felszabadítása elengedhetetlen a hatékony párhuzamos számításokhoz.

#### Memória allokáció (clCreateBuffer)

Az OpenCL memóriaobjektumokat a `clCreateBuffer` függvénnyel lehet létrehozni. Ezek a memóriaobjektumok használhatók az adatok tárolására és átadására a host és az eszköz között.

```c
int main() {
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem memobj = NULL;
    cl_uint ret_num_platforms;
    cl_uint ret_num_devices;
    cl_int ret;

    // Get Platform and Device Info
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    // Create OpenCL Context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create Command Queue
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create Memory Buffer
    memobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * sizeof(float), NULL, &ret);
    if (ret != CL_SUCCESS) {
        printf("Failed to create memory buffer.\n");
        return 1;
    }

    printf("Memory buffer created successfully.\n");

    // Cleanup
    clReleaseMemObject(memobj);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}
```

#### Adatok átvitele host és eszköz között (clEnqueueWriteBuffer, clEnqueueReadBuffer)

Az adatok host és eszköz közötti átviteléhez a `clEnqueueWriteBuffer` és `clEnqueueReadBuffer` függvényeket használhatjuk. Ezek a függvények biztosítják, hogy az adatok megfelelően átkerüljenek a memóriába és vissza.

```c
int main() {
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem memobj = NULL;
    cl_uint ret_num_platforms;
    cl_uint ret_num_devices;
    cl_int ret;

    float data[1024];
    for (int i = 0; i < 1024; i++) {
        data[i] = i * 1.0f;
    }

    // Get Platform and Device Info
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    // Create OpenCL Context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create Command Queue
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create Memory Buffer
    memobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * sizeof(float), NULL, &ret);

    // Write Data to Memory Buffer
    ret = clEnqueueWriteBuffer(command_queue, memobj, CL_TRUE, 0, 1024 * sizeof(float), data, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf("Failed to write data to buffer.\n");
        return 1;
    }

    // Read Data from Memory Buffer
    float result[1024];
    ret = clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0, 1024 * sizeof(float), result, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf("Failed to read data from buffer.\n");
        return 1;
    }

    printf("Data transferred successfully.\n");

    // Cleanup
    clReleaseMemObject(memobj);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}
```

### 15.3 Kernel írás és futtatás

A kernel az OpenCL programok alapvető számítási egysége. A kernel írása, fordítása és futtatása az OpenCL programozás egyik legfontosabb része. A következőkben bemutatjuk, hogyan írhatunk, fordíthatunk és futtathatunk egy OpenCL kernel-t.

#### Kernel forráskód írása és fordítása

A kernel forráskódját C-szerű szintaxissal írjuk, és a `clCreateProgramWithSource` és `clBuildProgram` függvények segítségével fordítjuk.

```c
const char *kernelSource = "__kernel void vec_add(__global float* a, __global float* b, __global float* c) { int id = get_global_id(0); c[id] = a[id] + b[id]; }";

int main() {
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem memobj_a = NULL;
    cl_mem memobj_b = NULL;
    cl_mem memobj_c = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_uint ret_num_platforms;
    cl_uint ret_num_devices;
    cl_int ret;

    float a[1024], b[1024], c[1024];
    for (int i = 0; i < 1024; i++) {
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }

    // Get Platform and Device Info
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    // Create OpenCL Context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create Command Queue
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create Memory Buffers
    memobj_a = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * sizeof(float), NULL, &ret);
    memobj_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * sizeof(float), NULL, &ret);
    memobj_c = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * sizeof(float), NULL, &ret);

    // Write Data to Memory Buffers
    ret = clEnqueueWriteBuffer(command_queue, memobj_a, CL_TRUE, 0, 1024 * sizeof(float), a, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, memobj_b, CL_TRUE, 0, 1024 * sizeof(float), b, 0, NULL, NULL);

    // Create Kernel Program from the source
    program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &ret);

    // Build Kernel Program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Error in kernel: %s\n", log);
        free(log);
        return 1;
    }

    // Create OpenCL Kernel
    kernel = clCreateKernel(program, "vec_add", &ret);
    if (ret != CL_SUCCESS) {
        printf("Failed to create kernel.\n");
        return 1;
    }

    // Set OpenCL Kernel Arguments
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj_a);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memobj_b);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&memobj_c);

    // Execute OpenCL Kernel
    size_t global_item_size = 1024;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf("Failed to execute kernel.\n");
        return 1;
    }

    // Read Data from Memory Buffer
    ret = clEnqueueReadBuffer(command_queue, memobj_c, CL_TRUE, 0, 1024 * sizeof(float), c, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf("Failed to read data from buffer.\n");
        return 1;
    }

    // Display Result
    for (int i = 0; i < 10; i++) {
        printf("%f + %f = %f\n", a[i], b[i], c[i]);
    }

    // Cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(memobj_a);
    clReleaseMemObject(memobj_b);
    clReleaseMemObject(memobj_c);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}
```

Ebben a fejezetben bemutattuk az OpenCL programozás alapjait, beleértve az OpenCL program struktúráját, a memória kezelését, valamint a kernel írásának és futtatásának folyamatát. Az OpenCL programok fejlesztése során ezek az alapvető lépések biztosítják, hogy a program hatékonyan tudja kihasználni a rendelkezésre álló hardver erőforrásokat, és lehetővé teszik a párhuzamos számítási feladatok végrehajtását különböző platformokon. Az alábbi példakódok és leírások segítségével könnyen megérthetjük és alkalmazhatjuk az OpenCL programozás alapvető technikáit saját projektjeinkben.