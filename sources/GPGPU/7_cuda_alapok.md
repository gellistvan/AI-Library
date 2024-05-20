\newpage

## 7. CUDA Programozási Alapok

A CUDA (Compute Unified Device Architecture) programozás alapjai kulcsfontosságúak a párhuzamos számítások világában, különösen, ha a grafikus feldolgozó egységek (GPU-k) erejét szeretnénk kihasználni az általános célú számításokhoz (GPGPU). Ebben a fejezetben megismerkedünk a CUDA programozás alapvető elemeivel, kezdve a kernelfüggvények írásától a szálak kiosztásán és szinkronizálásán át a memória allokáció és adatmásolás kérdéseiig. A fejezet végén néhány egyszerű CUDA program példáját is bemutatjuk, amelyek segítenek megérteni, hogyan lehet a gyakorlatban alkalmazni a tanultakat. Ezek az alapok elengedhetetlenek ahhoz, hogy hatékonyan kihasználhassuk a GPU-k párhuzamos feldolgozási képességeit, és jelentős teljesítménynövekedést érjünk el a számítási feladatokban.

### 7.1 Kernelfüggvények írása

A CUDA programozás alapjainak elsajátítása érdekében először meg kell értenünk a kernelfüggvények írásának módszertanát. A kernelfüggvények a CUDA programozás alapvető építőkövei, melyek lehetővé teszik, hogy a párhuzamos feladatokat a GPU szálain futtassuk. Ebben az alfejezetben bemutatjuk a kernelfüggvények szintaxisát és struktúráját, valamint néhány alapvető példát is.

#### Kernel szintaxis és struktúra

A kernelfüggvények olyan speciális függvények, amelyeket a GPU-n futtatunk, és a `__global__` kulcsszóval deklarálunk. Ez a kulcsszó jelzi, hogy a függvény a host kódból (CPU) hívható, de a device-on (GPU) fut. A kernelfüggvény szintaxisa a következő:

```cpp
__global__ void kernelFunction(parameters) {
    // Kernel kód
}
```

A kernelfüggvény hívása speciális szintaxist igényel, amely meghatározza a szálak és blokkok elrendezését. Ezt a <<< >>> operátorral adjuk meg:

```cpp
kernelFunction<<<numBlocks, numThreads>>>(parameters);
```

Itt a `numBlocks` a blokkok számát, a `numThreads` pedig az egy blokkban lévő szálak számát határozza meg. A szálak és blokkok elrendezése kulcsfontosságú a teljesítmény szempontjából, mivel a GPU párhuzamos számítási képességeit így tudjuk hatékonyan kihasználni.

#### Példakód: Egyszerű kernelfüggvény

Nézzünk meg egy egyszerű kernelfüggvényt, amely egy tömb elemeinek négyzetét számolja ki:

```cpp
__global__ void squareArray(float *a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        a[idx] = a[idx] * a[idx];
    }
}

int main() {
    int N = 1000;
    float *d_a;
    size_t size = N * sizeof(float);

    cudaMalloc(&d_a, size);

    float h_a[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
    }
    
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    squareArray<<<blocksPerGrid, threadsPerBlock>>>(d_a, N);

    cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);

    return 0;
}
```

Ebben a példában egy tömb minden elemének négyzetét számoljuk ki. A `squareArray` kernelfüggvény a GPU-n fut, ahol minden szál egy-egy elemet dolgoz fel.

#### Részletes magyarázat

1. **Kernelfüggvény deklarációja**: A `__global__` kulcsszóval jelöljük a `squareArray` függvényt, amely a GPU-n fog futni.
2. **Index kiszámítása**: Az `idx` változó kiszámítja a globális indexet, amely a szálak és blokkok elrendezéséből adódik össze. Ez az index határozza meg, melyik tömbelemhez fér hozzá a szál.
3. **Feltételes vizsgálat**: Az `if` feltétel biztosítja, hogy csak a tömb határain belüli elemeket módosítsuk.
4. **Memória allokáció**: A `cudaMalloc` függvénnyel memóriát foglalunk a GPU-n, majd a `cudaMemcpy` függvénnyel átmásoljuk a CPU memóriájában lévő adatokat a GPU memóriájába.
5. **Kernelfüggvény hívása**: A kernelfüggvény hívása során meghatározzuk a blokkok és szálak számát. Ebben az esetben a `threadsPerBlock` értéke 256, ami egy általánosan jó választás a legtöbb GPU számára.
6. **Eredmények visszamásolása**: A számítások után a `cudaMemcpy` függvénnyel visszamásoljuk az eredményeket a GPU memóriájából a CPU memóriájába.
7. **Memória felszabadítása**: A `cudaFree` függvénnyel felszabadítjuk a GPU memóriáját.

#### Példakód: Többdimenziós tömb kezelése

A következő példában egy kétdimenziós tömb elemeit adjuk össze egy kernelfüggvény segítségével:

```cpp
__global__ void addMatrices(float *a, float *b, float *c, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;
    
    if (x < width && y < height) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int width = 1024;
    int height = 1024;
    int size = width * height * sizeof(float);

    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    for (int i = 0; i < width * height; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    addMatrices<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, width, height);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```

#### Részletes magyarázat

1. **Kernelfüggvény deklarációja**: Az `addMatrices` függvény kétdimenziós szálak és blokkok elrendezésével dolgozik.
2. **Indexek kiszámítása**: Az `x` és `y` koordináták kiszámítják a szál pozícióját a kétdimenziós térben. Az `index` változó segítségével egy lineáris indexet kapunk.
3. **Feltételes vizsgálat**: Az `if` feltétel biztosítja, hogy csak a tömb határain belüli elemeket dolgozzunk fel.
4. **Memória allokáció és adatmásolás**: A host (CPU) memóriájában lévő adatokat a GPU memóriájába másoljuk a `cudaMemcpy` függvénnyel.
5. **Blokk és szál elrendezés**: A `dim3` típus segítségével kétdimenziós elrendezést határozunk meg. Ebben a példában minden blokk 16x16 szálból áll.
6. **Kernelfüggvény hívása**: Az addMatrices függvényt a megfelelő blokkok és szálak számával hívjuk meg.
7. **Eredmények visszamásolása és memória felszabadítása**: A számítások után az eredményeket visszamásoljuk a CPU memóriájába, majd felszabadítjuk a GPU memóriáját.

#### Összegzés

A kernelfüggvények írása és megértése alapvető fontosságú a CUDA programozásban. A szálak és blokkok elrendezése, a memória allokáció és az adatmásolás mind kritikus elemei annak, hogy hatékonyan kihasználjuk a GPU párhuzamos számítási képességeit. A bemutatott példák és magyarázatok remélhetőleg segítenek megérteni a kernelfüggvények működését és alkalmazását különböző számítási feladatokban.


### 7.2 Szálkiosztás és szinkronizáció

A CUDA programozás egyik legkritikusabb aspektusa a szálak (threads) és blokkok (blocks) hatékony kiosztása és a szinkronizáció kezelése. A GPU-k alapvetően nagy mennyiségű szál párhuzamos végrehajtására optimalizáltak, ezért a megfelelő szálkiosztás és szinkronizáció elengedhetetlen a teljesítmény maximalizálásához és a helyes működés biztosításához. Ebben az alfejezetben részletesen megvizsgáljuk, hogyan definiáljuk és kezeljük a szálakat és blokkokat, valamint bemutatjuk a szinkronizációs technikákat a CUDA programozásban.

#### Szál és blokk dimenziók

A CUDA programozás során a szálakat blokkokba szervezzük, és ezek a blokkok egy háromdimenziós rácsban helyezkednek el. Minden blokk tartalmazhat egy, két vagy háromdimenziós szálcsoportokat. A szálak és blokkok elrendezésének meghatározása kritikus, mivel a GPU architektúrája erősen párhuzamos, és a helyes elrendezés maximalizálhatja a teljesítményt.

A szálak és blokkok elrendezését a következő módon definiáljuk:

```cpp
dim3 threadsPerBlock(x, y, z);
dim3 blocksPerGrid(x, y, z);
```

Példaként nézzünk egy egyszerű kernelfüggvényt, amely egy kétdimenziós tömb minden elemét egy konstans értékkel növeli:

```cpp
__global__ void addConstant(float *array, float constant, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;

    if (x < width && y < height) {
        array[index] += constant;
    }
}

int main() {
    int width = 1024;
    int height = 1024;
    int size = width * height * sizeof(float);

    float *h_array = (float *)malloc(size);
    for (int i = 0; i < width * height; i++) {
        h_array[i] = i;
    }

    float *d_array;
    cudaMalloc(&d_array, size);
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    addConstant<<<blocksPerGrid, threadsPerBlock>>>(d_array, 5.0f, width, height);

    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

    cudaFree(d_array);
    free(h_array);

    return 0;
}
```

#### Szálkiosztás és teljesítmény

A megfelelő szálkiosztás jelentős hatással lehet a program teljesítményére. Az optimális szálkiosztás biztosítja, hogy a GPU minden számítási erőforrását hatékonyan kihasználjuk. Általános szabály, hogy a blokk méretének (szálak száma blokkban) többszörösének kell lennie a warp méretének, amely a legtöbb modern NVIDIA GPU esetében 32. Például a 256 szál blokk méret gyakran jó választás, mivel ez 8 warp.

#### Szinkronizáció __syncthreads() használata

A szálak közötti szinkronizáció elengedhetetlen, ha több szál együttműködik és adatokat oszt meg egymással. A CUDA egy `__syncthreads()` függvényt biztosít, amely blokkon belüli szinkronizációra szolgál. Minden szálnak el kell érnie ezt a pontot, mielőtt bármelyik tovább lépne, így biztosítva, hogy az összes szál befejezte a jelenlegi munkáját, mielőtt a következő lépésbe lépnének.

Példaként nézzünk egy kernelfüggvényt, amely egy blokkon belüli redukciós műveletet végez, azaz egy tömb elemeinek összegét számítja ki:

```cpp
__global__ void sumReduction(float *input, float *output, int n) {
    extern __shared__ float sharedData[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        sharedData[tid] = input[index];
    } else {
        sharedData[tid] = 0.0f;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

int main() {
    int n = 1024;
    int size = n * sizeof(float);

    float *h_input = (float *)malloc(size);
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    cudaMalloc(&d_output, blocksPerGrid * sizeof(float));

    sumReduction<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_output, n);

    float *h_output = (float *)malloc(blocksPerGrid * sizeof(float));
    cudaMemcpy(h_output, d_output, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    float finalSum = 0.0f;
    for (int i = 0; i < blocksPerGrid; i++) {
        finalSum += h_output[i];
    }

    printf("Sum: %f\n", finalSum);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
```

#### Részletes magyarázat

1. **Kernelfüggvény deklarációja**: A `sumReduction` függvény egy blokkon belüli redukciós műveletet végez.
2. **Külső megosztott memória használata**: Az `extern __shared__ float sharedData[]` deklaráció dinamikusan allokált megosztott memóriát használ a blokkon belüli szálak közötti adatok tárolására.
3. **Adatok másolása megosztott memóriába**: Az input tömb elemeit a megosztott memóriába másoljuk.
4. **Szinkronizáció**: A `__syncthreads()` függvényt használjuk a szálak szinkronizálására minden iteráció után.
5. **Redukciós ciklus**: A redukciós művelet során a tömb elemeit folyamatosan összeadjuk, amíg egyetlen értéket nem kapunk blokkonként.
6. **Eredmény tárolása**: A blokkon belüli összeg eredményét az output tömbbe mentjük.

#### Összegzés

A szálkiosztás és szinkronizáció a CUDA programozás alapvető elemei, amelyek lehetővé teszik a GPU párhuzamos számítási képességeinek hatékony kihasználását. A megfelelő szál- és blokkdimenziók kiválasztása, valamint a szinkronizációs technikák alkalmazása elengedhetetlen a teljesítmény optimalizálása és a helyes működés biztosítása érdekében. A bemutatott példák és részletes magyarázatok segítenek megérteni ezen technikák alkalmazását különböző számítási feladatokban, és előkészítik az utat a komplexebb CUDA programok fejlesztéséhez.

### 7.3 Memória allokáció és adatmásolás

A GPU-n történő párhuzamos számítások egyik legfontosabb aspektusa a memória kezelése. A CUDA programozás során a memória allokáció és az adatmásolás hatékony kezelése kulcsfontosságú a teljesítmény és a helyes működés biztosításához. Ebben az alfejezetben részletesen bemutatjuk a CUDA memóriakezelési módszereit, különös tekintettel a `cudaMalloc`, `cudaMemcpy` és `cudaFree` függvényekre, valamint gyakorlati példákkal illusztráljuk ezek használatát.

#### Memória allokáció a GPU-n: `cudaMalloc`

A GPU memóriájának (device memory) allokálása a `cudaMalloc` függvénnyel történik. Ez a függvény hasonló a C nyelv `malloc` függvényéhez, de a GPU memóriájára vonatkozik. A `cudaMalloc` szintaxisa a következő:

```c
cudaError_t cudaMalloc(void **devPtr, size_t size);
```

- `devPtr`: A pointer, amely a GPU memóriájában lefoglalt helyet fogja mutatni.
- `size`: Az allokálni kívánt memória mérete bájtokban.

Például, egy `float` típusú tömb GPU memóriájának allokálása a következőképpen történik:

```cpp
float *d_array;
size_t size = N * sizeof(float);
cudaMalloc(&d_array, size);
```

#### Adatmásolás: `cudaMemcpy`

Az adatokat a CPU memóriájából (host memory) a GPU memóriájába és vissza a `cudaMemcpy` függvénnyel másolhatjuk. Ennek szintaxisa:

```c
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
```

- `dst`: A cél memória cím.
- `src`: A forrás memória cím.
- `count`: Az átmásolandó adatok mérete bájtokban.
- `kind`: Az adatmásolás iránya, amely lehet:
    - `cudaMemcpyHostToDevice`: Host -> Device
    - `cudaMemcpyDeviceToHost`: Device -> Host
    - `cudaMemcpyDeviceToDevice`: Device -> Device
    - `cudaMemcpyHostToHost`: Host -> Host

Példa a CPU-ról a GPU-ra történő adatmásolásra:

```cpp
float *h_array = (float *)malloc(size);
// Töltsük fel h_array elemeit adatokkal
cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);
```

#### Memória felszabadítása: `cudaFree`

A GPU-n lefoglalt memória felszabadítása a `cudaFree` függvénnyel történik, amely a `cudaMalloc` párja. Ennek szintaxisa:

```c
cudaError_t cudaFree(void *devPtr);
```

Példa a GPU memória felszabadítására:

```cpp
cudaFree(d_array);
```

#### Teljes példa: Vektorszorzás

Nézzünk egy teljes példát, amely bemutatja a memória allokáció, adatmásolás és felszabadítás folyamatát egy egyszerű vektorszorzás (dot product) művelet végrehajtásán keresztül:

```cpp
__global__ void dotProductKernel(float *a, float *b, float *result, int N) {
    __shared__ float temp[256];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    temp[tid] = (index < N) ? a[index] * b[index] : 0.0f;

    __syncthreads();

    // Reduction in shared memory
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            sum += temp[i];
        }
        atomicAdd(result, sum);
    }
}

int main() {
    int N = 1000;
    size_t size = N * sizeof(float);

    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float h_result = 0.0f;

    for (int i = 0; i < N; i++) {
        h_a[i] = i + 1.0f;
        h_b[i] = i + 1.0f;
    }

    float *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_result, sizeof(float));

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    dotProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, N);

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Dot Product: %f\n", h_result);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    free(h_a);
    free(h_b);

    return 0;
}
```

#### Részletes magyarázat

1. **Memória allokáció**: A `cudaMalloc` függvényekkel a vektorok és az eredmény számára memóriát foglalunk a GPU-n.
2. **Adatmásolás**: A `cudaMemcpy` függvényekkel a vektorokat a CPU memóriájából a GPU memóriájába másoljuk, és az eredmény tárolóját is inicializáljuk.
3. **Kernelfüggvény**: A `dotProductKernel` függvény párhuzamosan számolja ki a vektorok elemeinek szorzatát és tárolja az eredményeket egy megosztott memóriában, majd redukciós műveletet végez.
4. **Eredmények másolása vissza**: A `cudaMemcpy` segítségével az eredmény visszakerül a CPU memóriájába.
5. **Memória felszabadítása**: A `cudaFree` függvényekkel felszabadítjuk a GPU memóriáját.

#### Összetettebb példák

A CUDA lehetővé teszi a fejlettebb memória kezelési technikák alkalmazását, mint például a kétoldalas memóriakezelés (pinned memory) és az egyidejű adatmásolás és végrehajtás (asynchronous memcpy). Ezek a technikák tovább növelhetik a teljesítményt.

##### Kétoldalas memória (pinned memory)

A kétoldalas memória (más néven rögzített vagy pinned memória) lehetővé teszi az adatmásolás sebességének növelését, mivel az ilyen memória nem cserélhető ki a CPU memóriájából. A következő példában bemutatjuk, hogyan lehet kétoldalas memóriát allokálni és használni:

```cpp
float *h_a, *h_b;
cudaMallocHost((void**)&h_a, size); // Allokáció kétoldalas memóriában
cudaMallocHost((void**)&h_b, size);

// Adatok inicializálása
for (int i = 0; i < N; i++) {
    h_a[i] = i + 1.0f;
    h_b[i] = i + 1.0f;
}

cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

// Kernelfüggvény futtatása

cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);
cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);

cudaFreeHost(h_a); // Kétoldalas memória felszabadítása
cudaFreeHost(h_b);
```

#### Egyidejű adatmásolás és végrehajtás

A CUDA lehetőséget biztosít az egyidejű adatmásolásra és végrehajtásra, ami tovább javíthatja a teljesítményt. Az alábbi példa bemutatja az aszinkron adatmásolás és kernel futtatás használatát CUDA streamek segítségével:

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream);
cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream);

dotProductKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_a, d_b, d_result, N);

cudaMemcpyAsync(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost, stream);

cudaStreamSynchronize(stream);
cudaStream

Destroy(stream);
```

#### Összegzés

A memória allokáció és adatmásolás a CUDA programozás alapvető része. A hatékony memória kezelése elengedhetetlen a GPU teljesítményének maximalizálása érdekében. A `cudaMalloc`, `cudaMemcpy` és `cudaFree` függvények használata mellett a fejlettebb technikák, mint a kétoldalas memória és az aszinkron adatmásolás, további teljesítményjavulást eredményezhetnek. A bemutatott példák és magyarázatok remélhetőleg segítenek megérteni a memória kezelésének fontosságát és alkalmazását a CUDA programozásban.

### 7.4 Egyszerű CUDA programok példái

A CUDA programozás elsajátítása során hasznos lehet néhány alapvető példaprogramot megvizsgálni, amelyek bemutatják a GPU-n végzett párhuzamos számítások alapelveit. Ebben az alfejezetben két gyakran használt számítási feladatot vizsgálunk meg: a vektorszorzást és a mátrixösszeadást. Mindkét feladat alapvető fontosságú a lineáris algebrai műveletek és a tudományos számítások szempontjából, és jól illusztrálják a CUDA párhuzamosítási képességeit.

#### Vektorszorzás

A vektorszorzás (dot product) az egyik legegyszerűbb és leggyakrabban használt lineáris algebrai művelet. Két vektor elemenkénti szorzatainak összegét adja meg. Az alábbiakban bemutatunk egy CUDA programot, amely párhuzamosan számolja ki a vektorszorzást.

##### Kernelfüggvény

```cpp
__global__ void dotProductKernel(float *a, float *b, float *result, int N) {
    __shared__ float temp[256];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    if (index < N) {
        temp[tid] = a[index] * b[index];
    } else {
        temp[tid] = 0.0f;
    }

    __syncthreads();

    // Reduction in shared memory
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            sum += temp[i];
        }
        atomicAdd(result, sum);
    }
}
```

##### Host kód

```cpp
int main() {
    int N = 1000;
    size_t size = N * sizeof(float);

    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float h_result = 0.0f;

    // Vektorok inicializálása
    for (int i = 0; i < N; i++) {
        h_a[i] = i + 1.0f;
        h_b[i] = i + 1.0f;
    }

    float *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_result, sizeof(float));

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    dotProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, N);

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Dot Product: %f\n", h_result);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    free(h_a);
    free(h_b);

    return 0;
}
```

##### Magyarázat

1. **Kernelfüggvény**: A `dotProductKernel` egy blokkon belül kiszámolja a vektorok elemenkénti szorzatát, majd egy redukciós műveletet végez a blokkon belül, és az eredményt atomikusan hozzáadja az eredményhez.
2. **Host kód**: A host kód allokálja a memóriát a CPU és a GPU számára, inicializálja a vektorokat, másolja az adatokat a GPU-ra, elindítja a kernelfüggvényt, majd visszamásolja az eredményt a CPU-ra és felszabadítja a memóriát.

#### Mátrixösszeadás

A mátrixösszeadás egy másik alapvető lineáris algebrai művelet, amely két mátrix megfelelő elemeinek összeadását jelenti. Az alábbiakban bemutatunk egy CUDA programot, amely párhuzamosan hajtja végre a mátrixösszeadást.

##### Kernelfüggvény

```cpp
__global__ void addMatricesKernel(float *a, float *b, float *c, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;

    if (x < width && y < height) {
        c[index] = a[index] + b[index];
    }
}
```

##### Host kód

```cpp
int main() {
    int width = 1024;
    int height = 1024;
    int size = width * height * sizeof(float);

    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    // Mátrixok inicializálása
    for (int i = 0; i < width * height; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    addMatricesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, width, height);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Eredmény ellenőrzése
    for (int i = 0; i < width * height; i++) {
        if (h_c[i] != 3.0f) {
            printf("Error at element %d: %f\n", i, h_c[i]);
            break;
        }
    }

    printf("Matrix addition successful.\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```

##### Magyarázat

1. **Kernelfüggvény**: Az `addMatricesKernel` függvény kiszámolja a mátrixok megfelelő elemeinek összegét. A két dimenziós grid és blokk elrendezés lehetővé teszi, hogy a GPU szálai párhuzamosan végezzék el az összeadást.
2. **Host kód**: A host kód inicializálja a mátrixokat, memóriát allokál a GPU-n, majd átmásolja az adatokat a GPU-ra. Ezután elindítja a kernelfüggvényt, amely kiszámolja az eredményt, és végül visszamásolja az eredményeket a CPU memóriájába, ahol ellenőrzi az eredmény helyességét.

#### Összegzés

A vektorszorzás és a mátrixösszeadás példái jól szemléltetik, hogyan lehet egyszerű numerikus műveleteket párhuzamosan végrehajtani a CUDA segítségével. Mindkét példa bemutatja a memória allokáció, adatmásolás, kernelfüggvények írása és szinkronizáció alapvető lépéseit. A CUDA programozás során ezen technikák és módszerek elsajátítása elengedhetetlen a hatékony és gyors párhuzamos számítások végrehajtásához a GPU-kon.
