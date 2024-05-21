\newpage

Rendben, bővítem a második függeléket részletesebb szójegyzékkel és rövidítésekkel.

### Függelékek

#### 1. függelék: Hasznos linkek és dokumentációk

Ebben a függelékben összegyűjtöttük a GPU programozás világában hasznos linkeket és dokumentációkat, amelyek tovább segíthetnek a tanulásban és a fejlődésben.

##### Általános GPU programozás
- [NVIDIA Developer](https://developer.nvidia.com): Hivatalos oldala az NVIDIA fejlesztői eszközöknek és dokumentációknak.
- [AMD Developer](https://developer.amd.com): Hivatalos oldala az AMD fejlesztői eszközöknek és dokumentációknak.

##### CUDA
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/): A CUDA Toolkit hivatalos dokumentációja.
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html): Részletes útmutató a CUDA programozáshoz.
- [CUDA Sample Code](https://github.com/NVIDIA/cuda-samples): Példakódok és projektek a CUDA világából.

##### OpenCL
- [Khronos Group OpenCL](https://www.khronos.org/opencl/): A Khronos Group hivatalos oldala az OpenCL szabványhoz.
- [OpenCL Specification](https://www.khronos.org/registry/OpenCL/specs/): Az OpenCL specifikáció hivatalos dokumentuma.
- [OpenCL Programming Guide](https://www.khronos.org/registry/OpenCL/sdk/2.2/docs/man/html/): Részletes útmutató az OpenCL programozáshoz.

##### Egyéb GPU programozási eszközök
- [Vulkan](https://www.khronos.org/vulkan/): A Vulkan API hivatalos oldala és dokumentációi.
- [DirectCompute](https://learn.microsoft.com/en-us/windows/win32/direct3d11/directcompute-overview): A DirectCompute hivatalos oldala és dokumentációi.

#### 2. függelék: Szójegyzék és rövidítések

Ebben a függelékben összegyűjtöttük a könyvben használt legfontosabb szakkifejezéseket és rövidítéseket.

##### Szójegyzék
- **Atomic Operation**: Egy olyan művelet, amely megszakíthatatlanul és oszthatatlanul hajtódik végre.
- **Bandwidth**: Az az adatmennyiség, amelyet egy rendszer egy adott idő alatt képes feldolgozni.
- **Block**: A CUDA programozásban egy szálcsoport, amely közösen hajt végre egy kernel függvényt.
- **Buffer**: Egy memória terület, amely adatokat tárol az átvitel vagy feldolgozás közben.
- **Context**: Az erőforrások halmaza, amelyeket egy GPU program futtatása közben használ.
- **Device**: A GPU maga, amely a párhuzamos számításokat végzi.
- **Host**: Az a számítógép, amely a GPU-t vezérli és a programot futtatja.
- **Kernel**: A CUDA vagy OpenCL kód egy függvénye, amelyet a GPU hajt végre.
- **Latency**: Az az idő, amely alatt egy adat eljut egyik helyről a másikra a rendszerben.
- **Occupancy**: A szálak száma, amelyek párhuzamosan futnak egy multiprocesszoron belül a GPU-n.
- **Profiling**: A program teljesítményének elemzése és optimalizálása.
- **Scheduler**: Az a rendszerkomponens, amely a feladatok végrehajtási sorrendjét határozza meg.
- **SIMD (Single Instruction, Multiple Data)**: Egyutas, többszörös adatos feldolgozási modell.
- **SIMT (Single Instruction, Multiple Threads)**: Egyutas, többszálas feldolgozási modell.
- **Stream**: Egy olyan sorozat, amely aszinkron műveleteket hajt végre a GPU-n.
- **Synchronization**: A folyamat, amely biztosítja, hogy a szálak megfelelő sorrendben hajtsák végre a műveleteket.

##### Rövidítések
- **API (Application Programming Interface)**: Alkalmazásprogramozási felület.
- **CPU (Central Processing Unit)**: Központi feldolgozóegység.
- **CUDA (Compute Unified Device Architecture)**: Az NVIDIA által kifejlesztett párhuzamos számítási platform és programozási modell.
- **FP32 (32-bit Floating Point)**: 32 bites lebegőpontos számformátum.
- **FP64 (64-bit Floating Point)**: 64 bites lebegőpontos számformátum.
- **GPGPU (General-Purpose Computing on Graphics Processing Units)**: Általános célú számítások GPU-n.
- **IDE (Integrated Development Environment)**: Integrált fejlesztőkörnyezet.
- **L1, L2 Cache**: Az első- és másodszintű gyorsítótár a memóriahierarchiában.
- **PCIe (Peripheral Component Interconnect Express)**: Egy nagy sebességű interfész a számítógép és a GPU között.
- **SDK (Software Development Kit)**: Szoftverfejlesztői készlet.
- **SM (Streaming Multiprocessor)**: A GPU egyik komponense, amely több szál párhuzamos végrehajtásáért felelős.
- **SP (Streaming Processor)**: Az SM egy komponense, amely az egyedi szálak végrehajtásáért felelős.

Ez a részletes függelék hasznos referenciapontként szolgálhat a könyv tartalmának jobb megértéséhez és a GPU programozásban való elmélyüléshez.