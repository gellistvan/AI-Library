\newpage

## 9. Haladó CUDA Programozás

Ahogy mélyebbre merülünk a CUDA programozás világába, a Haladó CUDA Programozás című fejezet célja, hogy kibővítse az alapvető ismereteket, és bemutassa azokat a technikákat, amelyekkel maximalizálhatjuk a GPU-k teljesítményét komplex számítási feladatok során. Ebben a fejezetben először a stream-ek és aszinkron műveletek kerülnek górcső alá, ahol részletesen tárgyaljuk a cudaStreamCreate és cudaStreamSynchronize funkciókat, amelyek lehetővé teszik az adatátvitel és számítás párhuzamosítását a GPU-n. Ezt követően a dinamikus memóriaallokáció technikái kerülnek bemutatásra a kernelen belül, kiemelve a cudaMallocManaged használatának jelentőségét. Végezetül, a Unified Memory és Managed Memory koncepcióival foglalkozunk, feltárva azok előnyeit és korlátait, hogy a fejlesztők hatékonyabban kezelhessék az erőforrásokat és optimalizálhassák alkalmazásaik teljesítményét a heterogén számítási környezetekben.


### 9.1 Stream-ek és aszinkron műveletek

A CUDA programozás egyik legfontosabb és leghasznosabb eszköze a stream-ek és aszinkron műveletek alkalmazása. Ezek segítségével párhuzamosan futtathatunk különböző számításokat és adatmozgatásokat, így növelve az alkalmazások hatékonyságát és teljesítményét. Ebben a fejezetben részletesen bemutatjuk, hogyan használhatók a CUDA stream-ek és aszinkron műveletek, valamint számos példakóddal illusztráljuk azok alkalmazását.

#### Stream-ek alapjai

A CUDA stream-ek lehetővé teszik, hogy különböző műveleteket párhuzamosan hajtsunk végre a GPU-n. Alapértelmezés szerint minden CUDA művelet a nulladik stream-ben fut, amely implicit szinkronizálva van, vagyis egy művelet csak akkor kezdődhet el, ha az előző már befejeződött. Azonban további stream-ek létrehozásával és használatával lehetőségünk van arra, hogy több műveletet párhuzamosan hajtsunk végre, így jobb kihasználtságot érhetünk el.

Stream létrehozása a `cudaStreamCreate` függvénnyel történik:

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);
```

Miután létrehoztunk egy stream-et, a különböző CUDA műveleteket (mint például kernel hívások vagy adatmozgatások) hozzárendelhetjük ehhez a stream-hez. Például:

```cpp
kernel<<<blocks, threads, 0, stream>>>(params);
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
```

Ezek a műveletek most párhuzamosan futnak a megadott stream-ben, anélkül hogy várnának a nulladik stream műveleteire.

#### Stream-ek és aszinkron műveletek

A stream-ek lehetővé teszik az aszinkron műveletek végrehajtását is. Az aszinkron műveletek elindulnak, de nem várják meg a befejeződésüket, mielőtt a vezérlés visszatérne a CPU-ra. Ez lehetővé teszi a CPU és a GPU közötti jobb munkamegosztást, hiszen a CPU folytathatja más feladatok végrehajtását, miközben a GPU dolgozik.

Például az aszinkron memóriaátvitelt a `cudaMemcpyAsync` függvénnyel végezhetjük:

```cpp
cudaMemcpyAsync(devicePtr, hostPtr, size, cudaMemcpyHostToDevice, stream);
```

Ez a függvény azonnal visszatér, és az adatátvitel a háttérben történik. A kernel hívások is futtathatók aszinkron módon:

```cpp
kernel<<<blocks, threads, 0, stream>>>(params);
```

A `cudaStreamSynchronize` függvénnyel szinkronizálhatunk egy adott stream-re, vagyis megvárhatjuk, hogy az összes hozzá tartozó művelet befejeződjön:

```cpp
cudaStreamSynchronize(stream);
```

Ez hasznos lehet akkor, amikor biztosítani szeretnénk, hogy az összes művelet befejeződött egy stream-ben, mielőtt további műveleteket végeznénk.

#### Példakódok

##### Egyszerű stream használat

Az alábbi példában egy egyszerű CUDA alkalmazást mutatunk be, amely két különböző stream-ben hajt végre műveleteket:

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void simpleKernel(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] += 1;
}

int main() {
    const int size = 1024;
    int *deviceData1, *deviceData2;
    int *hostData = new int[size];

    // Allocate device memory
    cudaMalloc(&deviceData1, size * sizeof(int));
    cudaMalloc(&deviceData2, size * sizeof(int));

    // Create streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Initialize host data
    for (int i = 0; i < size; ++i) {
        hostData[i] = i;
    }

    // Copy data to device asynchronously in different streams
    cudaMemcpyAsync(deviceData1, hostData, size * sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(deviceData2, hostData, size * sizeof(int), cudaMemcpyHostToDevice, stream2);

    // Launch kernels in different streams
    simpleKernel<<<size / 256, 256, 0, stream1>>>(deviceData1);
    simpleKernel<<<size / 256, 256, 0, stream2>>>(deviceData2);

    // Copy results back to host asynchronously
    cudaMemcpyAsync(hostData, deviceData1, size * sizeof(int), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(hostData, deviceData2, size * sizeof(int), cudaMemcpyDeviceToHost, stream2);

    // Synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Clean up
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(deviceData1);
    cudaFree(deviceData2);
    delete[] hostData;

    return 0;
}
```

Ez a példakód két különböző stream-et használ, hogy párhuzamosan másolja az adatokat a host-ról a device-ra, végrehajt két kernel hívást, majd visszamásolja az eredményeket a host-ra. Mindkét stream szinkronizálása után a műveletek befejeződnek.

##### Több stream hatékony kezelése

A következő példában egy összetettebb alkalmazást mutatunk be, amely több stream-et kezel dinamikusan:

```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void computeKernel(int *data, int offset) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx + offset] += 1;
}

void processWithStreams(int *hostData, int dataSize, int numStreams) {
    int *deviceData;
    cudaMalloc(&deviceData, dataSize * sizeof(int));

    // Create streams
    std::vector<cudaStream_t> streams(numStreams);
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Process data in chunks
    int chunkSize = dataSize / numStreams;
    for (int i = 0; i < numStreams; ++i) {
        int offset = i * chunkSize;
        cudaMemcpyAsync(&deviceData[offset], &hostData[offset], chunkSize * sizeof(int), cudaMemcpyHostToDevice, streams[i]);
        computeKernel<<<chunkSize / 256, 256, 0, streams[i]>>>(&deviceData[offset], offset);
        cudaMemcpyAsync(&hostData[offset], &deviceData[offset], chunkSize * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
    }

    // Synchronize streams
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(deviceData);
}

int main() {
    const int dataSize = 1024;
    int *hostData = new int[dataSize];

    // Initialize host data
    for (int i = 0; i < dataSize; ++i) {
        hostData[i] = i;
    }

    // Process data with streams
    int numStreams = 4;
    processWithStreams(hostData, dataSize, numStreams);

    // Output results
    for (int i = 0; i < dataSize; ++i) {
        std::cout << hostData[i] << " ";
    }
    std::cout << std::endl;

    delete[] hostData;
    return 0;
}
```

Ebben a példában az adatokat több stream-ben dolgozzuk fel. Az adatokat kisebb darabokra bontjuk, és minden darabot külön stream-ben kezelünk. Ez a megközelítés lehetővé teszi a párhuzamos feldolgozást és az erőforrások hatékonyabb kihasználását.

#### Szinkronizáció és függőségek kezelése

A stream-ek használata során fontos, hogy megfelelően kezeljük a függőségeket a különböző műveletek között. Az aszinkron műveletek indításával biztosítanunk kell, hogy a szükséges műveletek megfelelő sorrendben fejeződjenek be.

Az `cudaEvent` objektumok segítségével finomhangolhatjuk a szinkronizációt. Egy eseményt egy adott stream-hez rendelhetünk, és az esemény teljesülése után más stream-ekben is végrehajthatunk műveleteket. Például:

```cpp
cudaEvent_t event;
cudaEventCreate(&event);

// Launch kernel in stream1
simpleKernel<<<blocks, threads, 0, stream1>>>(deviceData1);

// Record event in stream1
cudaEventRecord(event, stream1);

// Wait for event in stream2
cudaStreamWaitEvent(stream2, event, 0);

// Launch kernel in stream2
simpleKernel<<<blocks, threads, 0, stream2>>>(deviceData2);
```

Ebben a példában az `cudaEventRecord` eseményt rögzítjük az első stream-ben, majd a második stream vár erre az eseményre az `cudaStreamWaitEvent` segítségével. Így biztosítjuk, hogy a második kernel hívás csak akkor induljon el, ha az első már befejeződött.

#### Összefoglalás

A stream-ek és aszinkron műveletek használata lehetővé teszi a CUDA alkalmazások párhuzamosítását és optimalizálását. A `cudaStreamCreate`, `cudaStreamSynchronize`, `cudaMemcpyAsync` és más hasonló függvények segítségével hatékonyan kihasználhatjuk a GPU teljesítményét, miközben a CPU és GPU közötti munkamegosztást is javíthatjuk. A megfelelő szinkronizációs technikák alkalmazásával biztosíthatjuk a helyes működést és a függőségek kezelését a különböző műveletek között.

### 9.2 Dinamikus memóriaallokáció kernelen belül

A CUDA programozásban a dinamikus memóriaallokáció lehetőséget biztosít arra, hogy a kernel futása közben igény szerint allokáljunk memóriát. Ez különösen hasznos, amikor a szükséges memória mérete előre nem ismert, vagy amikor a memóriahasználat hatékonyságát szeretnénk növelni. Ebben a fejezetben részletesen bemutatjuk, hogyan használhatjuk a dinamikus memóriaallokációt a CUDA kernelen belül, és hogyan kezelhetjük az ezzel járó kihívásokat.

#### Dinamikus memóriaallokáció a kernelen belül

A CUDA 6.0 óta a `cudaMalloc` és `cudaFree` függvények nemcsak a host oldalon, hanem a GPU kernelen belül is elérhetők. Ez lehetővé teszi, hogy a kernel futása közben allokáljunk és szabadítsunk fel memóriát. Az alábbiakban bemutatjuk a dinamikus memóriaallokáció alapvető használatát egy egyszerű példán keresztül.

##### Példakód: Dinamikus memóriaallokáció kernelen belül

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void dynamicAllocKernel(int n) {
    // Dinamikus memóriaallokáció kernelen belül
    int *data = (int*)malloc(n * sizeof(int));

    // Ellenőrizzük, hogy a memóriaallokáció sikeres volt-e
    if (data != nullptr) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            data[idx] = idx * idx;
        }

        // Szinkronizáljuk a szálakat
        __syncthreads();

        // Csak az első szál írja ki az eredményeket
        if (idx == 0) {
            for (int i = 0; i < n; ++i) {
                printf("data[%d] = %d\n", i, data[i]);
            }
        }

        // Felszabadítjuk a memóriát
        free(data);
    } else {
        printf("Memory allocation failed\n");
    }
}

int main() {
    const int n = 10;

    // Kernel indítása
    dynamicAllocKernel<<<1, n>>>(n);

    // Szinkronizálás a GPU-val
    cudaDeviceSynchronize();

    return 0;
}
```

Ebben a példában a `dynamicAllocKernel` kernel futása közben dinamikusan allokál egy `data` tömböt, amelynek mérete a kernel argumentumában megadott `n` értéktől függ. Az allokáció sikerességét ellenőrizzük, majd az első szál kiírja az eredményeket. Végül felszabadítjuk a memóriát.

#### Teljesítmény és hatékonyság

A dinamikus memóriaallokáció kernelen belül rugalmasságot biztosít, ugyanakkor teljesítménybeli kihívásokkal is járhat. A memóriaallokáció és felszabadítás időigényes műveletek lehetnek, amelyek befolyásolhatják a kernel futási idejét. Ezért fontos, hogy a dinamikus memóriaallokációt körültekintően használjuk, és csak akkor alkalmazzuk, ha valóban szükséges.

Egy alternatív megoldás a `cudaMallocManaged` használata, amely unified memory-t biztosít, és automatikusan kezeli a memória helyét a CPU és GPU között. Ez egyszerűsítheti a memória kezelést, ugyanakkor bizonyos esetekben csökkentheti a teljesítményt.

##### Példakód: Dinamikus memóriaallokáció managed memóriával

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void managedMemoryKernel(int *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] = idx * idx;
    }
}

int main() {
    const int n = 10;
    int *data;

    // Managed memória allokálása
    cudaMallocManaged(&data, n * sizeof(int));

    // Kernel indítása
    managedMemoryKernel<<<1, n>>>(data, n);

    // Szinkronizálás a GPU-val
    cudaDeviceSynchronize();

    // Eredmények kiírása
    for (int i = 0; i < n; ++i) {
        std::cout << "data[" << i << "] = " << data[i] << std::endl;
    }

    // Memória felszabadítása
    cudaFree(data);

    return 0;
}
```

Ebben a példában a `cudaMallocManaged` segítségével allokálunk memóriát, amelyet a `managedMemoryKernel` kernel használ. A managed memória automatikusan szinkronizálja az adatokat a CPU és GPU között, egyszerűsítve a memória kezelést.

#### Gyakorlati alkalmazások

A dinamikus memóriaallokáció hasznos lehet számos alkalmazásban, például:

1. **Adatszerkezetek dinamikus kezelése**: Olyan adatszerkezetek létrehozása, amelyek mérete előre nem ismert, például listák, gráfok vagy hash táblák.
2. **Adaptív algoritmusok**: Olyan algoritmusok, amelyek futás közben adaptálódnak az adatok méretéhez vagy más paraméterekhez, például adaptív rácsok vagy adaptív sugárkövetés.
3. **Memóriahatékonyság növelése**: A memóriahasználat optimalizálása olyan esetekben, amikor az adatok mérete dinamikusan változik, és nem akarjuk előre allokálni a maximális memóriát.

##### Példakód: Adaptív rácsok kezelése

```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void adaptiveGridKernel(float *data, int *sizes, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        int size = sizes[idx];
        float *localData = (float*)malloc(size * sizeof(float));
        if (localData != nullptr) {
            for (int i = 0; i < size; ++i) {
                localData[i] = data[idx] * i;
            }
            for (int i = 0; i < size; ++i) {
                printf("Thread %d: localData[%d] = %f\n", idx, i, localData[i]);
            }
            free(localData);
        } else {
            printf("Thread %d: Memory allocation failed\n", idx);
        }
    }
}

int main() {
    const int n = 5;
    float hostData[n] = {1.0, 2.0, 3.0, 4.0, 5.0};
    int hostSizes[n] = {10, 20, 30, 40, 50};

    float *deviceData;
    int *deviceSizes;
    
    // Memória allokálása
    cudaMalloc(&deviceData, n * sizeof(float));
    cudaMalloc(&deviceSizes, n * sizeof(int));

    // Adatok másolása a GPU-ra
    cudaMemcpy(deviceData, hostData, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSizes, hostSizes, n * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel indítása
    adaptiveGridKernel<<<1, n>>>(deviceData, deviceSizes, n);

    // Szinkronizálás a GPU-val
    cudaDeviceSynchronize();

    // Memória felszabadítása
    cudaFree(deviceData);
    cudaFree(deviceSizes);

    return 0;
}
```

Ebben a példában az `adaptiveGridKernel` kernel minden szála dinamikusan allokál egy `localData` tömböt, amelynek mérete a `sizes` tömbből származik. Az allokáció sikerességét ellenőrizzük, és ha sikeres, a szálak kiírják az eredményeket. Az adaptív rácsok használatával a memóriahasználat hatékonysága növelhető, mivel csak a szükséges mennyiségű memóriát allokáljuk minden szál számára.

#### Összefoglalás

A dinamikus memóriaallokáció kernelen belül rugalmasságot biztosít a CUDA programozásban, lehetővé téve az adatszerkezetek dinamikus kezelését és az adaptív algoritmusok megvalósítását. A `malloc` és `free` függvények használatával memória allokálható és felszabadítható a kernel futása közben, míg a `cudaMallocManaged` segítségével unified memory használható, amely automatikusan kezeli a memória helyét a CPU és GPU között. A megfelelő memória kezeléssel és szinkronizációval hatékony és skálázható CUDA alkalmazásokat készíthetünk, amelyek kihasználják a dinamikus memóriaallokáció előnyeit.

### 9.2 Dinamikus memóriaallokáció kernelen belül

A CUDA programozásban a dinamikus memóriaallokáció lehetőséget ad arra, hogy a kernel futása közben igény szerint allokáljunk memóriát. Ez különösen hasznos lehet, amikor a szükséges memória mérete előre nem ismert, vagy amikor a memóriahatékonyságot szeretnénk növelni az adatszerkezetek dinamikus kezelésével. Ebben az alfejezetben részletesen bemutatjuk a dinamikus memóriaallokáció használatát a CUDA kernelen belül, és megvizsgáljuk a kapcsolódó technikákat, kihívásokat és legjobb gyakorlatokat.

#### Dinamikus memóriaallokáció alapjai

A CUDA 6.0 verziótól kezdve a `malloc` és `free` függvények elérhetők a GPU kernelen belül is. Ezek a függvények lehetővé teszik, hogy a GPU memóriát dinamikusan osszuk ki és szabadítsuk fel, hasonlóan a CPU oldali használathoz. Az alábbiakban egy alapvető példát mutatunk be, amely bemutatja a dinamikus memóriaallokáció használatát egy egyszerű kernelben.

##### Példakód: Alapvető dinamikus memóriaallokáció

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void dynamicAllocKernel(int n) {
    // Dinamikus memória allokálása kernelen belül
    int *data = (int*)malloc(n * sizeof(int));

    // Ellenőrizzük, hogy a memóriaallokáció sikeres volt-e
    if (data != nullptr) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            data[idx] = idx * idx;
        }

        // Szinkronizáljuk a szálakat
        __syncthreads();

        // Csak az első szál írja ki az eredményeket
        if (idx == 0) {
            for (int i = 0; i < n; ++i) {
                printf("data[%d] = %d\n", i, data[i]);
            }
        }

        // Felszabadítjuk a memóriát
        free(data);
    } else {
        printf("Memory allocation failed\n");
    }
}

int main() {
    const int n = 10;

    // Kernel indítása
    dynamicAllocKernel<<<1, n>>>(n);

    // Szinkronizálás a GPU-val
    cudaDeviceSynchronize();

    return 0;
}
```

Ebben a példában a `dynamicAllocKernel` kernel futása közben dinamikusan allokál egy `data` tömböt, amelynek mérete a kernel argumentumában megadott `n` értéktől függ. Az allokáció sikerességét ellenőrizzük, majd az első szál kiírja az eredményeket. Végül felszabadítjuk a memóriát.

#### Teljesítmény és hatékonyság

A dinamikus memóriaallokáció használata rugalmasságot biztosít, ugyanakkor teljesítménybeli kihívásokat is jelent. A memóriaallokáció és felszabadítás időigényes műveletek, amelyek befolyásolhatják a kernel futási idejét. Ezért fontos, hogy a dinamikus memóriaallokációt körültekintően használjuk, és csak akkor alkalmazzuk, ha valóban szükséges.

A teljesítmény optimalizálása érdekében érdemes megfontolni a következő szempontokat:
- **Allokáció minimalizálása**: Minimalizáljuk a dinamikus allokációk számát a kernel futása során.
- **Memória újrafelhasználása**: Ha lehetséges, használjuk újra a már allokált memóriát, ahelyett hogy újra és újra allokálnánk és felszabadítanánk.
- **Együttműködés**: Több szál együttműködése az allokációk és felszabadítások koordinálásában segíthet csökkenteni a teljesítményveszteséget.

#### Példakód: Dinamikus memória újrafelhasználása

Az alábbi példában bemutatjuk, hogyan használhatjuk újra a dinamikusan allokált memóriát, hogy minimalizáljuk az allokációk számát és javítsuk a teljesítményt.

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void reuseDynamicAllocKernel(int *data, int n, int iterations) {
    // Dinamikus memória allokálása kernelen belül
    int *tempData = (int*)malloc(n * sizeof(int));

    // Ellenőrizzük, hogy a memóriaallokáció sikeres volt-e
    if (tempData != nullptr) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        for (int iter = 0; iter < iterations; ++iter) {
            if (idx < n) {
                tempData[idx] = data[idx] + iter;
            }

            // Szinkronizáljuk a szálakat
            __syncthreads();

            // Csak az első szál írja ki az eredményeket
            if (idx == 0) {
                printf("Iteration %d: tempData[0] = %d\n", iter, tempData[0]);
            }

            // Szinkronizáljuk a szálakat
            __syncthreads();
        }

        // Felszabadítjuk a memóriát
        free(tempData);
    } else {
        printf("Memory allocation failed\n");
    }
}

int main() {
    const int n = 10;
    const int iterations = 5;
    int hostData[n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int *deviceData;

    // Memória allokálása és másolása a GPU-ra
    cudaMalloc(&deviceData, n * sizeof(int));
    cudaMemcpy(deviceData, hostData, n * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel indítása
    reuseDynamicAllocKernel<<<1, n>>>(deviceData, n, iterations);

    // Szinkronizálás a GPU-val
    cudaDeviceSynchronize();

    // Memória felszabadítása
    cudaFree(deviceData);

    return 0;
}
```

Ebben a példában a `reuseDynamicAllocKernel` kernel több iterációban használja ugyanazt a dinamikusan allokált `tempData` tömböt, így minimalizálva az allokációk számát és javítva a teljesítményt.

#### Unified Memory használata

A `cudaMallocManaged` függvény segítségével unified memory-t is használhatunk, amely automatikusan kezeli a memória helyét a CPU és GPU között. Ez egyszerűsítheti a memória kezelést, ugyanakkor bizonyos esetekben csökkentheti a teljesítményt a memória mozgatásának költsége miatt.

##### Példakód: Dinamikus memóriaallokáció unified memory-val

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void managedMemoryKernel(int *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] = idx * idx;
    }
}

int main() {
    const int n = 10;
    int *data;

    // Managed memória allokálása
    cudaMallocManaged(&data, n * sizeof(int));

    // Kernel indítása
    managedMemoryKernel<<<1, n>>>(data, n);

    // Szinkronizálás a GPU-val
    cudaDeviceSynchronize();

    // Eredmények kiírása
    for (int i = 0; i < n; ++i) {
        std::cout << "data[" << i << "] = " << data[i] << std::endl;
    }

    // Memória felszabadítása
    cudaFree(data);

    return 0;
}
```

Ebben a példában a `cudaMallocManaged` segítségével allokálunk memóriát, amelyet a `managedMemoryKernel` kernel használ. A managed memória automatikusan szinkronizálja az adatokat a CPU és GPU között, egyszerűsítve a memória kezelést.

#### Gyakorlati alkalmazások

A dinamikus memóriaallokáció számos gyakorlati alkalmazásban hasznos lehet, például:

1. **Adatszerkezetek dinamikus kezelése**: Olyan adatszerkezetek létrehozása, amelyek mérete előre nem ismert, például listák, gráfok vagy hash táblák.
2. **Adaptív algoritmusok**: Olyan algoritmusok, amelyek futás közben adaptálódnak az adatok méretéhez vagy más paraméterekhez, például adaptív rácsok vagy adaptív sugárkövetés.
3. **Memóriahatékonyság növelése**: A memóriahasználat optimalizálása olyan esetekben, amikor az adatok mérete dinamikusan változik, és nem akarjuk előre allokálni a maximális memóriát.

##### Példakód: Adaptív rácsok kezelése

```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void adaptiveGridKernel(float *data, int *sizes, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        int size = sizes[idx];
        float *localData = (float*)malloc(size * sizeof(float));
        if (localData != nullptr) {
            for (int i = 0; i < size; ++i) {
                localData[i] = data[idx] * i;
            }
            for (int i = 0; i < size; ++i) {
                printf("Thread %d: localData[%d] = %f\n", idx, i, localData[i]);
            }
            free(localData);
        } else {
            printf("Thread %d: Memory allocation failed\n", idx);
        }
    }
}

int main() {
    const int n = 5;
    float hostData[n] = {1.0, 2.0, 3.0, 4.0, 5.0};
    int hostSizes[n] = {10, 20, 30, 40, 50};

    float *deviceData;
    int *deviceSizes;
    
    // Memória allokálása
    cudaMalloc(&deviceData, n * sizeof(float));
    cudaMalloc(&deviceSizes, n * sizeof(int));

    // Adatok másolása a GPU-ra
    cudaMemcpy(deviceData, hostData, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSizes, hostSizes, n * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel indítása
    adaptiveGridKernel<<<1, n>>>(deviceData, deviceSizes, n);

    // Szinkronizálás a GPU-val
    cudaDeviceSynchronize();

    // Memória felszabadítása
    cudaFree(deviceData);
    cudaFree(deviceSizes);

    return 0;
}
```

Ebben a példában az `adaptiveGridKernel` kernel minden szála dinamikusan allokál egy `localData` tömböt, amelynek mérete a `sizes` tömbből származik. Az allokáció sikerességét ellenőrizzük, és ha sikeres, a szálak kiírják az eredményeket. Az adaptív rácsok használatával a memóriahasználat hatékonysága növelhető, mivel csak a szükséges mennyiségű memóriát allokáljuk minden szál számára.

#### Összefoglalás

A dinamikus memóriaallokáció kernelen belül rugalmasságot biztosít a CUDA programozásban, lehetővé téve az adatszerkezetek dinamikus kezelését és az adaptív algoritmusok megvalósítását. A `malloc` és `free` függvények használatával memória allokálható és felszabadítható a kernel futása közben, míg a `cudaMallocManaged` segítségével unified memory használható, amely automatikusan kezeli a memória helyét a CPU és GPU között. A megfelelő memória kezeléssel és szinkronizációval hatékony és skálázható CUDA alkalmazásokat készíthetünk, amelyek kihasználják a dinamikus memóriaallokáció előnyeit.

### 9.3 Unified Memory és Managed Memory

Az NVIDIA CUDA ökoszisztéma egyre fejlettebb eszközöket kínál a programozók számára, hogy a GPU-kat a lehető legjobban kihasználhassák. Az egyik ilyen eszköz a Unified Memory (egységes memória), amely az alkalmazások számára egyszerűsíti a memória kezelést a CPU és a GPU között. A Unified Memory használatával a memória automatikusan megosztható és szinkronizálható a CPU és a GPU között, anélkül hogy explicit adatmozgatásra lenne szükség. Ebben az alfejezetben részletesen bemutatjuk a Unified Memory és Managed Memory koncepcióit, előnyeit, korlátait és gyakorlati alkalmazásait.

#### Unified Memory és Managed Memory áttekintése

A Unified Memory bevezetésével a fejlesztők egy egységes címteret használhatnak a CPU és a GPU számára. A `cudaMallocManaged` függvénnyel allokált memória automatikusan elérhető mind a CPU, mind a GPU számára, és az adatokat szükség szerint áthelyezi a rendszer. Ez nagyban leegyszerűsíti a programozást, mivel nem kell explicit adatmozgatást végezni a két memória tér között.

##### Példakód: Egyszerű Managed Memory használat

Az alábbi példában bemutatjuk, hogyan használhatjuk a Managed Memory-t egy egyszerű CUDA alkalmazásban:

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void incrementKernel(int *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] += 1;
    }
}

int main() {
    const int size = 1000;
    int *data;

    // Managed memória allokálása
    cudaMallocManaged(&data, size * sizeof(int));

    // Adatok inicializálása a CPU oldalon
    for (int i = 0; i < size; ++i) {
        data[i] = i;
    }

    // Kernel indítása
    incrementKernel<<<(size + 255) / 256, 256>>>(data, size);

    // Szinkronizálás a GPU-val
    cudaDeviceSynchronize();

    // Eredmények kiírása
    for (int i = 0; i < size; ++i) {
        std::cout << "data[" << i << "] = " << data[i] << std::endl;
    }

    // Memória felszabadítása
    cudaFree(data);

    return 0;
}
```

Ebben a példában a `cudaMallocManaged` függvénnyel allokálunk egy `data` tömböt, amely automatikusan elérhető mind a CPU, mind a GPU számára. Az adatok inicializálása a CPU oldalon történik, majd a `incrementKernel` kernel minden elem értékét megnöveli egy egységgel. Végül az eredményeket kiírjuk, és felszabadítjuk a memóriát.

#### Előnyök és korlátok

A Unified Memory számos előnnyel jár, de vannak bizonyos korlátai is, amelyeket figyelembe kell venni a használata során.

##### Előnyök

1. **Egyszerűsített memória kezelés**: A fejlesztőknek nem kell explicit módon kezelniük az adatmozgatást a CPU és a GPU között, ami leegyszerűsíti a kódot és csökkenti a hibalehetőségeket.
2. **Egységes címterület**: Az adatok egyetlen címterületen találhatók, amely megkönnyíti a fejlesztést és a hibaelhárítást.
3. **Automatikus adatmozgatás**: Az NVIDIA runtime automatikusan áthelyezi az adatokat a CPU és a GPU között, amikor szükséges, optimalizálva az adatmozgatási műveleteket.

##### Korlatok

1. **Teljesítmény**: Az automatikus adatmozgatás többletköltséggel járhat, és bizonyos esetekben csökkentheti a teljesítményt az explicit memória másoláshoz képest.
2. **Kompatibilitás**: A Unified Memory támogatása függ a hardver és a CUDA verziótól. Régebbi GPU-k és CUDA verziók nem támogatják teljes mértékben a Unified Memory-t.
3. **Kontroll hiánya**: Az automatikus memória kezelés kevesebb kontrollt biztosít a fejlesztőknek az adatmozgatások felett, ami bizonyos esetekben nem optimális.

#### Gyakorlati alkalmazások

A Unified Memory különösen hasznos lehet olyan alkalmazásokban, ahol az adatszerkezetek mérete dinamikusan változik, vagy ahol az explicit adatmozgatás jelentős komplexitást okozna. Az alábbi példákban különböző gyakorlati alkalmazásokat mutatunk be a Unified Memory használatával.

##### Példakód: Nagyméretű adatok kezelése

Az alábbi példában egy nagyméretű adatsorozatot dolgozunk fel a GPU-n, majd az eredményeket visszaolvassuk a CPU-ra.

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void processLargeDataKernel(float *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = sqrt(data[idx]);
    }
}

int main() {
    const int size = 1 << 20; // 1 millió elem
    float *data;

    // Managed memória allokálása
    cudaMallocManaged(&data, size * sizeof(float));

    // Adatok inicializálása a CPU oldalon
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i);
    }

    // Kernel indítása
    processLargeDataKernel<<<(size + 255) / 256, 256>>>(data, size);

    // Szinkronizálás a GPU-val
    cudaDeviceSynchronize();

    // Eredmények ellenőrzése
    for (int i = 0; i < 10; ++i) {
        std::cout << "data[" << i << "] = " << data[i] << std::endl;
    }

    // Memória felszabadítása
    cudaFree(data);

    return 0;
}
```

Ebben a példában a `processLargeDataKernel` kernel nagyméretű adatsorozatot dolgoz fel, ahol minden elem négyzetgyökét számítja ki. A Managed Memory használatával egyszerűsítjük az adatmozgatást, mivel az adatok automatikusan elérhetők mind a CPU, mind a GPU számára.

##### Példakód: Dinamikus adatszerkezetek kezelése

A következő példában egy dinamikus adatszerkezet, például egy lista, kezelését mutatjuk be a Unified Memory segítségével.

```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void processDynamicListKernel(int **list, int *sizes, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        for (int i = 0; i < sizes[idx]; ++i) {
            list[idx][i] += 1;
        }
    }
}

int main() {
    const int numLists = 5;
    std::vector<int> listSizes = {10, 20, 30, 40, 50};
    std::vector<int*> hostList(numLists);

    // Managed memória allokálása a listák számára
    int **deviceList;
    int *deviceSizes;
    cudaMallocManaged(&deviceList, numLists * sizeof(int*));
    cudaMallocManaged(&deviceSizes, numLists * sizeof(int));

    // Listák inicializálása és másolása a GPU-ra
    for (int i = 0; i < numLists; ++i) {
        cudaMallocManaged(&hostList[i], listSizes[i] * sizeof(int));
        for (int j = 0; j < listSizes[i]; ++j) {
            hostList[i][j] = j;
        }
        deviceList[i] = hostList[i];
        deviceSizes[i] = listSizes[i];
    }

    // Kernel indítása
    processDynamicListKernel<<<(numLists + 255) / 256, 256>>>(deviceList, deviceSizes, numLists);

    // Szinkronizálás a GPU-val
    cudaDeviceSynchronize();

    // Eredmények ellenőrzése
    for (int i = 0; i < numLists; ++i) {
        for (int j = 0; j < listSizes[i]; ++j) {
            std::cout << "list[" << i << "][" << j << "] = " << hostList[i][j] << std::endl;
        }
    }

    // Memória felszabadítása


    for (int i = 0; i < numLists; ++i) {
        cudaFree(hostList[i]);
    }
    cudaFree(deviceList);
    cudaFree(deviceSizes);

    return 0;
}
```

Ebben a példában dinamikusan allokálunk több különböző méretű listát, és a `processDynamicListKernel` kernel minden elem értékét megnöveli egy egységgel. A Managed Memory segítségével a listák automatikusan elérhetők mind a CPU, mind a GPU számára, egyszerűsítve a memória kezelést.

#### Teljesítmény optimalizálás

Bár a Unified Memory használata egyszerűsíti a memória kezelést, a teljesítmény optimalizálása érdekében figyelembe kell venni néhány szempontot:

1. **Memória hozzáférési minta**: Optimalizáljuk a memória hozzáférési mintákat, hogy minimalizáljuk az adatmozgatások számát a CPU és a GPU között.
2. **Prefetch**: Használjuk a `cudaMemPrefetchAsync` függvényt az adatok előzetes áthelyezésére a CPU és a GPU között, hogy csökkentsük a runtime alatt bekövetkező adatmozgatásokat.
3. **Memória szinkronizáció**: Biztosítsuk, hogy a memória szinkronizálva legyen, mielőtt a CPU vagy a GPU hozzáfér az adatokhoz, hogy elkerüljük az inkonzisztens állapotokat.

##### Példakód: Prefetch használata

Az alábbi példában bemutatjuk, hogyan használhatjuk a `cudaMemPrefetchAsync` függvényt az adatok előzetes áthelyezésére a GPU-ra.

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void prefetchKernel(float *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = sqrt(data[idx]);
    }
}

int main() {
    const int size = 1 << 20; // 1 millió elem
    float *data;

    // Managed memória allokálása
    cudaMallocManaged(&data, size * sizeof(float));

    // Adatok inicializálása a CPU oldalon
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i);
    }

    // Adatok előzetes áthelyezése a GPU-ra
    cudaMemPrefetchAsync(data, size * sizeof(float), 0);

    // Kernel indítása
    prefetchKernel<<<(size + 255) / 256, 256>>>(data, size);

    // Szinkronizálás a GPU-val
    cudaDeviceSynchronize();

    // Eredmények ellenőrzése
    for (int i = 0; i < 10; ++i) {
        std::cout << "data[" << i << "] = " << data[i] << std::endl;
    }

    // Memória felszabadítása
    cudaFree(data);

    return 0;
}
```

Ebben a példában a `cudaMemPrefetchAsync` függvényt használjuk, hogy az adatokat előzetesen áthelyezzük a GPU-ra, mielőtt a kernel futtatása megkezdődik. Ez segít minimalizálni a runtime alatt bekövetkező adatmozgatásokat, javítva a teljesítményt.

#### Összefoglalás

A Unified Memory és Managed Memory használata jelentősen egyszerűsíti a CUDA programozást, mivel nem kell explicit módon kezelni az adatmozgatást a CPU és a GPU között. Bár ez a megközelítés bizonyos esetekben csökkentheti a teljesítményt, a megfelelő optimalizálási technikák alkalmazásával hatékony és könnyen karbantartható kódot készíthetünk. A Unified Memory különösen hasznos lehet olyan alkalmazásokban, ahol az adatszerkezetek mérete dinamikusan változik, vagy ahol az explicit adatmozgatás jelentős komplexitást okozna. A Managed Memory segítségével az adatok automatikusan elérhetők mind a CPU, mind a GPU számára, lehetővé téve a fejlesztők számára, hogy a számítási feladatokra koncentráljanak, nem pedig a memória kezelésére.