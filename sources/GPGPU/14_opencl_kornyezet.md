\newpage
## 14 OpenCL Környezet Beállítása

Az OpenCL (Open Computing Language) lehetővé teszi, hogy heterogén rendszerekben, például CPU-kon, GPU-kon és más típusú processzorokon egyaránt hatékonyan végezzünk párhuzamos számításokat. Ebben a fejezetben megismerkedünk az OpenCL környezet beállításának alapjaival, hogy előkészítsük a fejlesztési folyamatot. Először áttekintjük, hogyan telepíthetjük az OpenCL SDK-t különböző operációs rendszerekre, mint a Windows, Linux és MacOS. Ezt követően bemutatjuk a legnépszerűbb fejlesztői eszközök konfigurálását, mint például az Eclipse és a Visual Studio Code. Végül pedig olyan hasznos eszközökkel és bővítményekkel ismerkedünk meg, mint a CodeXL és a gDEBugger, amelyek segítenek az OpenCL programok hatékony fejlesztésében és hibakeresésében.

### 14.1 OpenCL SDK telepítése

Az OpenCL fejlesztés megkezdéséhez először telepítenünk kell az OpenCL SDK-t (Software Development Kit). Az SDK biztosítja azokat az eszközöket és könyvtárakat, amelyek szükségesek az OpenCL programok fejlesztéséhez és futtatásához. Az alábbiakban bemutatjuk, hogyan telepíthetjük az OpenCL SDK-t a három leggyakoribb operációs rendszerre: Windows, Linux és MacOS.

#### OpenCL SDK letöltése és telepítése Windows-ra

1. **Intel OpenCL SDK Telepítése**
    - Nyissuk meg a [Intel OpenCL SDK letöltési oldalát](https://software.intel.com/content/www/us/en/develop/tools/opencl-sdk.html).
    - Töltsük le a legfrissebb telepítőt.
    - Futtassuk a letöltött fájlt, és kövessük a telepítési utasításokat.

2. **NVIDIA CUDA Toolkit Telepítése**
    - Látogassuk meg a [NVIDIA CUDA Toolkit letöltési oldalát](https://developer.nvidia.com/cuda-downloads).
    - Válasszuk ki a megfelelő operációs rendszert, és töltsük le a CUDA Toolkit-et.
    - Telepítsük a CUDA Toolkit-et, amely tartalmazza az NVIDIA OpenCL SDK-t is.

3. **AMD APP SDK Telepítése**
    - Nyissuk meg az [AMD APP SDK letöltési oldalát](https://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/).
    - Töltsük le az SDK-t és telepítsük a letöltött csomagot.

#### OpenCL SDK letöltése és telepítése Linux-ra

1. **Intel OpenCL SDK Telepítése**
    - Töltsük le az Intel OpenCL SDK csomagot az [Intel letöltési oldaláról](https://software.intel.com/content/www/us/en/develop/tools/opencl-sdk.html).
    - Telepítsük a csomagot az alábbi parancsokkal:
      ```bash
      tar -xvf opencl-sdk-linux.tar.gz
      cd opencl-sdk-linux
      sudo ./install.sh
      ```

2. **NVIDIA CUDA Toolkit Telepítése**
    - Látogassuk meg a [NVIDIA CUDA Toolkit letöltési oldalát](https://developer.nvidia.com/cuda-downloads).
    - Válasszuk ki a megfelelő Linux disztribúciót, és töltsük le a csomagot.
    - Telepítsük a CUDA Toolkit-et:
      ```bash
      sudo dpkg -i cuda-repo-<distro>_<version>_amd64.deb
      sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/<distro>/x86_64/7fa2af80.pub
      sudo apt-get update
      sudo apt-get install cuda
      ```

3. **AMD APP SDK Telepítése**
    - Töltsük le az AMD APP SDK-t az [AMD letöltési oldaláról](https://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/).
    - Telepítsük az SDK-t az alábbi parancsokkal:
      ```bash
      tar -xvf AMD-APP-SDK-linux.tar.bz2
      cd AMD-APP-SDK-linux
      sudo ./install.sh
      ```

#### OpenCL SDK letöltése és telepítése MacOS-re

1. **Apple OpenCL Framework**
    - MacOS rendszer esetén az OpenCL támogatás be van építve az operációs rendszerbe.
    - A Xcode telepítése (az Apple fejlesztői környezete) tartalmazza az OpenCL keretrendszert.
    - Töltsük le és telepítsük a Xcode-ot az [App Store-ból](https://apps.apple.com/us/app/xcode/id497799835?mt=12).

### Fejlesztői eszközök konfigurálása

A megfelelő fejlesztői eszközök beállítása elengedhetetlen az OpenCL fejlesztési folyamatának megkönnyítéséhez. Az alábbiakban bemutatjuk, hogyan lehet konfigurálni két népszerű fejlesztőkörnyezetet, az Eclipse-et és a Visual Studio Code-ot.

#### Eclipse beállítása

1. **Eclipse IDE telepítése**
    - Töltsük le az Eclipse IDE-t az [Eclipse letöltési oldaláról](https://www.eclipse.org/downloads/).
    - Telepítsük az Eclipse-et a letöltött csomag futtatásával.

2. **CDT plugin telepítése**
    - Indítsuk el az Eclipse-et, és válasszuk a `Help -> Eclipse Marketplace` menüpontot.
    - Keressük meg a "CDT" (C/C++ Development Tools) plugint, és telepítsük.

3. **OpenCL projekt létrehozása**
    - Indítsunk egy új C/C++ projektet az `File -> New -> C/C++ Project` menüpont alatt.
    - Válasszuk ki a megfelelő toolchain-t (pl. GCC).
    - Adjuk hozzá az OpenCL fejléceket és könyvtárakat a projekt beállításainál.

4. **OpenCL példa kód**
    - Hozzunk létre egy új C fájlt, és írjuk be a következő OpenCL példakódot:
      ```c
      #include <CL/cl.h>
      #include <stdio.h>
      #include <stdlib.h>
 
      const char *kernelSource = "__kernel void hello(__global char* string) { string[0] = 'H'; }";
 
      int main() {
          cl_platform_id platform_id = NULL;
          cl_device_id device_id = NULL;
          cl_context context = NULL;
          cl_command_queue command_queue = NULL;
          cl_mem memobj = NULL;
          cl_program program = NULL;
          cl_kernel kernel = NULL;
          cl_uint ret_num_devices;
          cl_uint ret_num_platforms;
          cl_int ret;
 
          char string[16];
 
          // Get Platform and Device Info
          ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
          ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
 
          // Create OpenCL Context
          context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
 
          // Create Command Queue
          command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
 
          // Create Memory Buffer
          memobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 16 * sizeof(char), NULL, &ret);
 
          // Create Kernel Program from the source
          program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &ret);
 
          // Build Kernel Program
          ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
 
          // Create OpenCL Kernel
          kernel = clCreateKernel(program, "hello", &ret);
 
          // Set OpenCL Kernel Arguments
          ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj);
 
          // Execute OpenCL Kernel
          ret = clEnqueueTask(command_queue, kernel, 0, NULL, NULL);
 
          // Copy results from the memory buffer
          ret = clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0, 16 * sizeof(char), string, 0, NULL, NULL);
 
          // Display Result
          printf("%s\n", string);
 
          // Finalization
          ret = clFlush(command_queue);
          ret = clFinish(command_queue);
          ret = clReleaseKernel(kernel);
          ret = clReleaseProgram(program);
          ret = clReleaseMemObject(memobj);
          ret = clReleaseCommandQueue(command_queue);
          ret = clReleaseContext(context);
 
          return 0;
      }
      ```

#### Visual Studio Code beállítása

1. **Visual Studio Code telepítése**
    - Töltsük le a Visual Studio Code-ot az [Visual Studio Code letöltési oldaláról](https://code.visualstudio.com/).
    - Telepítsük a letöltött csomagot.

2. **C/C++ Extension telepítése**
    - Indítsuk el a Visual Studio Code-ot, és válasszuk az `Extensions` ikont a bal oldali sávban.
    - Keressük meg a "C/C++" extension-t, és telepítsük.

3. **OpenCL projekt létrehozása**
    - Hozzunk létre egy új mappát a projekthez, és nyissuk meg a Visual Studio Code-ban.
    - Hozzunk létre egy új C fájlt (pl. `main.c`), és írjuk be a korábban bemutatott OpenCL példakódot.

4. **Tasks és Launch konfiguráció beállítása**
    - Hozzunk létre egy `.vscode` mappát a projekt gyökerében, és hozzuk létre a `tasks.json` és `launch.json` fájlokat az alábbi tartalommal:

      `tasks.json`:
      ```json
      {
          "version": "2.0.0",
          "tasks": [
              {
                  "label": "build",
                  "type": "shell",
                  "command": "gcc",
                  "args": [
                      "-o",
                      "main",
                      "main.c",
                      "-lOpenCL"
                  ],
                  "group": {
                      "kind": "build",
                      "isDefault": true
                  },
                  "problemMatcher": ["$gcc"],
                  "detail": "Generated task by Visual Studio Code"
              }
          ]
      }
      ```

      `launch.json`:
      ```json
      {
          "version": "0.2.0",
          "configurations": [
              {
                  "name": "C/C++: gcc build and debug active file",
                  "type": "cppdbg",
                  "request": "launch",
                  "program": "${workspaceFolder}/main",
                  "args": [],
                  "stopAtEntry": false,
                  "cwd": "${workspaceFolder}",
                  "environment": [],
                  "externalConsole": false,
                  "MIMode": "gdb",
                  "setupCommands": [
                      {
                          "description": "Enable pretty-printing for gdb",
                          "text": "-enable-pretty-printing",
                          "ignoreFailures": true
                      }
                  ],
                  "preLaunchTask": "build",
                  "miDebuggerPath": "/usr/bin/gdb",
                  "logging": {
                      "trace": true,
                      "traceResponse": true,
                      "engineLogging": true,
                      "programOutput": true,
                      "exceptions": true
                  },
                  "launchCompleteCommand": "exec-run",
                  "linux": {
                      "MIMode": "gdb"
                  },
                  "osx": {
                      "MIMode": "lldb"
                  },
                  "windows": {
                      "MIMode": "gdb"
                  }
              }
          ]
      }
      ```

### 14.2 OpenCL fejlesztőkörnyezet és eszközök

Az OpenCL programok fejlesztésének és optimalizálásának folyamata során számos hasznos eszköz és bővítmény áll rendelkezésre. Ezek az eszközök segítenek a kód hatékonyságának növelésében, a hibák felderítésében és a teljesítmény optimalizálásában.

#### Hasznos eszközök és bővítmények

1. **CodeXL**
    - A CodeXL egy AMD által fejlesztett nyílt forráskódú eszköz, amely segít az OpenCL programok teljesítményének elemzésében és hibakeresésében.
    - Telepítés:
        - Látogassuk meg a [CodeXL letöltési oldalát](https://github.com/GPUOpen-Tools/CodeXL).
        - Töltsük le a megfelelő verziót, és kövessük a telepítési utasításokat.

    - Használat:
        - Indítsuk el a CodeXL-t, és hozzunk létre egy új projektet.
        - Futtassuk az OpenCL alkalmazást a CodeXL-ben, és használjuk a profilozó és hibakereső eszközöket a teljesítmény és a hibák elemzéséhez.

2. **gDEBugger**
    - A gDEBugger egy NVIDIA által fejlesztett eszköz, amely segít az OpenCL programok hibakeresésében és teljesítményének optimalizálásában.
    - Telepítés:
        - Látogassuk meg a [gDEBugger letöltési oldalát](https://developer.nvidia.com/gdebugger).
        - Töltsük le a megfelelő verziót, és telepítsük az eszközt.

    - Használat:
        - Indítsuk el a gDEBugger-t, és nyissuk meg az OpenCL projektet.
        - Futtassuk az alkalmazást a gDEBugger-ben, és használjuk a rendelkezésre álló eszközöket a hibák és a teljesítmény elemzéséhez.

#### Példakód a gDEBugger használatához

```c
#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>

const char *kernelSource = "__kernel void vector_add(__global int* a, __global int* b, __global int* c) { int id = get_global_id(0); c[id] = a[id] + b[id]; }";

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
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    int a[10], b[10], c[10];
    for (int i = 0; i < 10; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Get Platform and Device Info
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    // Create OpenCL Context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create Command Queue
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create Memory Buffers
    memobj_a = clCreateBuffer(context, CL_MEM_READ_WRITE, 10 * sizeof(int), NULL, &ret);
    memobj_b = clCreateBuffer(context, CL_MEM_READ_WRITE, 10 * sizeof(int), NULL, &ret);
    memobj_c = clCreateBuffer(context, CL_MEM_READ_WRITE, 10 * sizeof(int), NULL, &ret);

    // Copy lists to Memory Buffers
    ret = clEnqueueWriteBuffer(command_queue, memobj_a, CL_TRUE, 0, 10 * sizeof(int), a, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, memobj_b, CL_TRUE, 0, 10 * sizeof(int), b, 0, NULL, NULL);

    // Create Kernel Program from the source
    program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &ret);

    // Build Kernel Program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create OpenCL Kernel
    kernel = clCreateKernel(program, "vector_add", &ret);

    // Set OpenCL Kernel Arguments
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj_a);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memobj_b);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&memobj_c);

    // Execute OpenCL Kernel
    size_t global_item_size = 10;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);

    // Copy results from the memory buffer
    ret = clEnqueueReadBuffer(command_queue, memobj_c, CL_TRUE, 0, 10 * sizeof(int), c, 0, NULL, NULL);

    // Display Result
    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // Finalization
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(memobj_a);
    ret = clReleaseMemObject(memobj_b);
    ret = clReleaseMemObject(memobj_c);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    return 0;
}
```

Ebben a fejezetben áttekintettük az OpenCL környezet beállításának alapjait, beleértve az SDK-k telepítését különböző operációs rendszerekre és a fejlesztői eszközök konfigurálását. Ezenkívül bemutattuk a CodeXL és a gDEBugger használatát az OpenCL programok teljesítményének optimalizálására és hibakeresésére. A megfelelő fejlesztői környezet és eszközök használata jelentősen megkönnyíti az OpenCL programok fejlesztését és karbantartását, lehetővé téve a hatékony és gyors párhuzamos számítási megoldások létrehozását.