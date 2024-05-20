\newpage
## 8. Memória Kezelés és Optimalizálás

A hatékony memória kezelés és optimalizálás kulcsfontosságú tényezők a GPU-alapú számításokban, hiszen a memória elérési minták és a memóriahierarchia megfelelő kihasználása jelentős teljesítményjavulást eredményezhet. Ebben a fejezetben a különböző memória típusok kezelését és optimalizálását vesszük górcső alá, kezdve a globális memória koaleszált hozzáférésének fontosságával, amely lehetővé teszi a párhuzamos szálak számára, hogy hatékonyan és gyorsan érjék el az adatokat. Továbbá megvizsgáljuk a megosztott memória használatát, különös tekintettel a bankütközések elkerülésére, melyek komoly akadályt jelenthetnek a teljesítmény szempontjából. Végül kitérünk a konstans és textúra memória alkalmazásának előnyeire, bemutatva, hogyan deklarálhatjuk és érhetjük el ezeket a memória területeket, hogy a lehető legjobban kihasználhassuk a GPU erőforrásait.


### 8.1 Globális memória használata

A GPU globális memóriája az egyik legfontosabb és egyben legnagyobb kapacitású memória típusa, amelyet a párhuzamos számítások során használhatunk. A globális memória elérése azonban viszonylag lassú, így a hozzáférési minták optimalizálása kritikus szerepet játszik a teljesítmény maximalizálásában. Ebben az alfejezetben részletesen megvizsgáljuk a globális memória használatának legjobb gyakorlatait, különös tekintettel a koaleszált memóriahozzáférésre, amely lehetővé teszi a memóriaelérések hatékonyabb végrehajtását.

#### Koaleszált memóriahozzáférés

A koaleszált memóriahozzáférés egy olyan technika, amelynek segítségével a GPU szálai egyidejűleg hozzáférhetnek a globális memóriához, minimalizálva a memóriahozzáférési késleltetéseket. A memóriahozzáférések koaleszálása akkor lehetséges, ha a szálak memóriahozzáférései rendezett módon, azaz egymást követő címeken történnek. Ez lehetővé teszi a memória vezérlő számára, hogy egyetlen nagyobb adatátvitelt hajtson végre több kisebb helyett, így csökkentve a memóriahozzáférés idejét.

##### Koaleszált memóriahozzáférés példák

Nézzünk meg egy egyszerű példát, ahol koaleszált memóriahozzáférést valósítunk meg egy vektormásolási műveleten keresztül. A következő kódrészlet egy CUDA kernel, amely egy forrás vektor adatait másolja egy célvektorba.

```cpp
__global__ void vectorCopy(float *d_out, const float *d_in, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        d_out[idx] = d_in[idx];
    }
}
```

Ebben a kernelben minden szál egyedi indexet kap, amelyet a `threadIdx.x` és `blockIdx.x * blockDim.x` összege határoz meg. Ha az összes szál egymást követő indexeket használ, a memóriahozzáférés koaleszálódik, így a memória vezérlő egyetlen, nagyobb adatátvitelt tud végrehajtani.

##### Koaleszálatlan memóriahozzáférés elkerülése

Az alábbi példában egy rosszul megtervezett memóriahozzáférést mutatunk be, ahol a szálak nem egymást követő memória címekhez férnek hozzá. Ennek eredményeként a memóriahozzáférések nem koaleszálódnak, ami jelentős teljesítménycsökkenést eredményezhet.

```cpp
__global__ void uncoalescedAccess(float *d_out, const float *d_in, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        d_out[idx] = d_in[idx * 2];  // Nem koaleszált hozzáférés
    }
}
```

Ebben a példában az `idx * 2` használata miatt a szálak minden második memória címet érnek el, így a memóriahozzáférés nem koaleszálódik.

#### A koaleszálás optimalizálása

A koaleszálás optimalizálásához néhány alapelvet kell követni:

1. **Egymást követő memória címek használata**: A szálaknak olyan memória címekhez kell hozzáférniük, amelyek egymást követők. Ez biztosítja, hogy a memória vezérlő egyetlen adatátvitelt hajtson végre több kisebb helyett.

2. **Szálak számának megfelelő igazítása**: Az optimális teljesítmény érdekében a blokkon belüli szálak számát (threads per block) és a memóriaelérési mintát megfelelően kell igazítani. A CUDA architektúra 32 szálból álló warpot használ, ezért a legjobb teljesítmény érdekében a memóriahozzáférésnek ezekhez az egységekhez igazodnia kell.

3. **Memória igazítás**: Az adatok memóriában való elhelyezésének is igazodnia kell a memória vezérlő igényeihez. Például, ha az adatokat 64 bájtos szegmensekben helyezzük el, a memóriahozzáférés optimalizálható.

##### Optimalizált koaleszált memóriahozzáférés példa

Az alábbi példa bemutatja, hogyan érhetjük el a koaleszált memóriahozzáférést egy mátrix transzponálás során. A cél az, hogy az input mátrixot transzponáljuk úgy, hogy a memóriahozzáférés koaleszálódjon.

```cpp
__global__ void transpose(float *out, const float *in, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int in_idx = y * width + x;
        int out_idx = x * height + y;
        out[out_idx] = in[in_idx];
    }
}
```

Ebben a kernelben minden szál egyedi `x` és `y` koordinátát kap, amelyek alapján kiszámítják a be- és kimeneti indexeket. Ha a szálak egymás mellett helyezkednek el, a memóriahozzáférés koaleszált lesz.

#### Gyakorlati tanácsok

A koaleszált memóriahozzáférés biztosítása érdekében az alábbi gyakorlati tanácsokat érdemes követni:

1. **Elemek átrendezése**: Amikor lehetséges, rendezze át az elemeket a memóriában úgy, hogy azok egymást követő címeken legyenek.

2. **Használjon megosztott memóriát**: A megosztott memória gyorsabb elérést biztosít, mint a globális memória. Az adatok előzetes betöltése a megosztott memóriába és onnan való elérése jelentős teljesítményjavulást eredményezhet.

3. **Optimalizálja a blokkméretet**: Az optimális blokkméret kiválasztása fontos a koaleszálás maximalizálása érdekében. Általában a 32 szálból álló warppal történő igazítás a legjobb megoldás.

Összefoglalva, a koaleszált memóriahozzáférés elérése kritikus a GPU-alapú számítások optimalizálása szempontjából. A szálak megfelelő igazítása, az egymást követő memória címek használata, valamint a memóriaelérési minták optimalizálása jelentős teljesítményjavulást eredményezhet. Az itt bemutatott példák és gyakorlati tanácsok segítenek abban, hogy hatékonyan használjuk ki a globális memória erőforrásait és elérjük a maximális számítási teljesítményt.

### 8.2 Megosztott memória használata

A megosztott memória a CUDA architektúrában egy különösen gyors hozzáférésű memória típus, amelyet a szálblokk minden szála közösen használhat. A globális memóriával ellentétben a megosztott memória hozzáférési ideje jóval alacsonyabb, mivel a memória a multiprocesszoron belül található. Az optimális teljesítmény eléréséhez fontos a megosztott memória hatékony kihasználása és a bankütközések elkerülése.

#### A megosztott memória alapjai

A megosztott memória a CUDA programozásban egy gyorsan elérhető tárhely, amelyet a programozó deklarál és kezel. A szálblokk minden szála hozzáférhet a megosztott memóriához, ami különösen hasznos lehet az adatok lokális újraszervezése és a globális memóriahozzáférések minimalizálása érdekében.

##### Megosztott memória deklarálása

A megosztott memória deklarálása a CUDA C/C++ nyelvben egyszerű, és a `__shared__` kulcsszóval történik. Például egy egyszerű szálblokk szintű adatmegosztás deklarációja így néz ki:

```cpp
__global__ void exampleKernel() {
    __shared__ float sharedData[256];
    // Kernel kód
}
```

Ebben a példában a `sharedData` tömb minden szál számára elérhető lesz a szálblokkban. A megosztott memória használatának előnyeit különösen akkor tapasztalhatjuk, ha a globális memória hozzáférések számát jelentősen csökkenthetjük.

#### Bankütközések és elkerülésük

A megosztott memória több bankra oszlik, és ezek a bankok egyidejű hozzáférést tesznek lehetővé különböző szálak számára. Azonban ha több szál ugyanahhoz a bankhoz próbál hozzáférni egy időben, bankütközés következik be, ami jelentős teljesítménycsökkenést okozhat.

##### Bankütközések magyarázata

A CUDA architektúra 32 bankot használ a megosztott memóriában, és minden bankhoz egy memória cím tartozik. Ha két vagy több szál ugyanahhoz a bankhoz fér hozzá, a hozzáférések sorba rendeződnek, és ez késleltetést eredményez.

##### Bankütközések elkerülése

A bankütközések elkerülésének egyik módja, hogy az adatok elrendezését úgy módosítjuk, hogy a szálak különböző bankokhoz férjenek hozzá. Például, ha a megosztott memóriában lévő adatokhoz való hozzáférést módosítjuk úgy, hogy a szálak eltolt címeket használnak, elkerülhetjük a bankütközéseket.

##### Példakód bankütközések elkerülésére

Az alábbi példában bemutatjuk, hogyan lehet elkerülni a bank

### 8.2 Megosztott memória használata

A megosztott memória a CUDA architektúrában egy különösen gyors hozzáférésű memória típus, amelyet a szálblokk minden szála közösen használhat. A globális memóriával ellentétben a megosztott memória hozzáférési ideje jóval alacsonyabb, mivel a memória a multiprocesszoron belül található. Az optimális teljesítmény eléréséhez fontos a megosztott memória hatékony kihasználása és a bankütközések elkerülése.

#### A megosztott memória alapjai

A megosztott memória a CUDA programozásban egy gyorsan elérhető tárhely, amelyet a programozó deklarál és kezel. A szálblokk minden szála hozzáférhet a megosztott memóriához, ami különösen hasznos lehet az adatok lokális újraszervezése és a globális memóriahozzáférések minimalizálása érdekében.

##### Megosztott memória deklarálása

A megosztott memória deklarálása a CUDA C/C++ nyelvben egyszerű, és a `__shared__` kulcsszóval történik. Például egy egyszerű szálblokk szintű adatmegosztás deklarációja így néz ki:

```cpp
__global__ void exampleKernel() {
    __shared__ float sharedData[256];
    // Kernel kód
}
```

Ebben a példában a `sharedData` tömb minden szál számára elérhető lesz a szálblokkban. A megosztott memória használatának előnyeit különösen akkor tapasztalhatjuk, ha a globális memória hozzáférések számát jelentősen csökkenthetjük.

#### Bankütközések és elkerülésük

A megosztott memória több bankra oszlik, és ezek a bankok egyidejű hozzáférést tesznek lehetővé különböző szálak számára. Azonban ha több szál ugyanahhoz a bankhoz próbál hozzáférni egy időben, bankütközés következik be, ami jelentős teljesítménycsökkenést okozhat.

##### Bankütközések magyarázata

A CUDA architektúra 32 bankot használ a megosztott memóriában, és minden bankhoz egy memória cím tartozik. Ha két vagy több szál ugyanahhoz a bankhoz fér hozzá, a hozzáférések sorba rendeződnek, és ez késleltetést eredményez.

##### Bankütközések elkerülése

A bankütközések elkerülésének egyik módja, hogy az adatok elrendezését úgy módosítjuk, hogy a szálak különböző bankokhoz férjenek hozzá. Például, ha a megosztott memóriában lévő adatokhoz való hozzáférést módosítjuk úgy, hogy a szálak eltolt címeket használnak, elkerülhetjük a bankütközéseket.

##### Példakód bankütközések elkerülésére

Az alábbi példában bemutatjuk, hogyan lehet elkerülni a bankütközéseket egy egyszerű mátrix transzponálás során, amely megosztott memóriát használ:

```cpp
__global__ void transposeNoBankConflicts(float *odata, const float *idata, int width) {
    __shared__ float tile[32][33];

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    if (x < width && y < width) {
        tile[threadIdx.y][threadIdx.x] = idata[y * width + x];
    }

    __syncthreads();

    x = blockIdx.y * 32 + threadIdx.x;  // Transposed block offset
    y = blockIdx.x * 32 + threadIdx.y;

    if (x < width && y < width) {
        odata[y * width + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

Ebben a példában a `tile` nevű megosztott memória tömböt használjuk, amelynek mérete 32x33. Ez a +1 eltérés a második dimenzióban biztosítja, hogy minden sor különböző bankokba essen, így elkerülve a bankütközéseket.

#### Megosztott memória alkalmazása gyakorlati példákon keresztül

##### Példa 1: Redukció

A redukció egy gyakori művelet a párhuzamos számításokban, ahol egy nagy adat tömböt kell összegezni. A megosztott memória használatával jelentős teljesítményjavulást érhetünk el.

```cpp
__global__ void reduce(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}
```

Ebben a példában az `sdata` megosztott memória tömb tárolja az egyes szálak által beolvasott adatokat. A redukciós művelet során a szálak egymást követően összeadják az értékeket, és az eredményt a blokkonkénti `g_odata` tömbbe írják.

##### Példa 2: Mátrix szorzás

A mátrix szorzás egy másik fontos művelet, ahol a megosztott memória használata jelentős teljesítménynövekedést eredményezhet.

```cpp
#define TILE_WIDTH 16

__global__ void matrixMulShared(float *C, const float *A, const float *B, int width) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Cvalue = 0;

    for (int m = 0; m < width / TILE_WIDTH; ++m) {
        As[ty][tx] = A[row * width + (m * TILE_WIDTH + tx)];
        Bs[ty][tx] = B[(m * TILE_WIDTH + ty) * width + col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    C[row * width + col] = Cvalue;
}
```

Ebben a példában az `As` és `Bs` megosztott memória tömbök tárolják az A és B mátrix csempéit. A csempék beolvasása után a szálak párhuzamosan kiszámítják a C mátrix megfelelő elemeit, majd az eredményt a globális memóriába írják.

#### Összefoglalás

A megosztott memória használata jelentős teljesítményjavulást eredményezhet a CUDA programozásban, különösen akkor, ha a globális memória hozzáférések számát minimalizáljuk és a bankütközéseket elkerüljük. Az itt bemutatott példák és technikák segítenek abban, hogy hatékonyan kihasználjuk a megosztott memória előnyeit, és ezáltal növeljük a párhuzamos számítások teljesítményét.

### 8.3 Konstans és textúra memória alkalmazása

A CUDA architektúrában a konstans és textúra memória speciális memória típusok, amelyek különböző célokra használhatók, hogy optimalizáljuk a GPU-alapú számításokat. Ezek a memória típusok kifejezetten arra szolgálnak, hogy bizonyos feladatokat hatékonyabban hajtsanak végre, mint a hagyományos globális memória. Ebben az alfejezetben részletesen megvizsgáljuk a konstans és textúra memória alkalmazását, előnyeit és használatának legjobb gyakorlatait, különös tekintettel arra, hogyan deklarálhatjuk és érhetjük el ezeket a memóriákat.

#### Konstans memória

A konstans memória egy kis kapacitású, de gyorsan elérhető memória típus, amelyet főként állandó értékek tárolására használunk. Ez a memória olvasásra optimalizált, és ideális olyan adatok tárolására, amelyeket a kernel futása során nem módosítunk.

##### Konstans memória deklarálása és elérése

A konstans memória deklarálása a `__constant__` kulcsszóval történik. A deklarált konstans változók a globális memória térben helyezkednek el, de különállóan kezelhetők.

###### Példa: Konstans memória deklarálása

```cpp
__constant__ float constData[256];
```

A fenti példában a `constData` nevű konstans memória tömböt deklaráljuk. Ezt a tömböt a host kód segítségével tölthetjük fel adatokkal.

###### Példa: Adatok másolása konstans memóriába

```cpp
float h_constData[256] = { /* adatok inicializálása */ };
cudaMemcpyToSymbol(constData, h_constData, sizeof(float) * 256);
```

A `cudaMemcpyToSymbol` függvény segítségével a host memóriából átmásolhatjuk az adatokat a konstans memóriába. Ez a művelet biztosítja, hogy a konstans memória tartalma megfelelő legyen a kernel futása során.

##### Konstans memória használata a kernelben

A konstans memória elérése a kernelből nagyon egyszerű. Csak hivatkoznunk kell a konstans változóra a kernel kódjában.

###### Példa: Konstans memória használata

```cpp
__global__ void kernelUsingConstantMemory(float *output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    output[idx] = constData[idx % 256];
}
```

Ebben a példában minden szál olvassa a konstans memóriából az `constData` tömb megfelelő elemét, és az eredményt az `output` tömbbe írja.

#### Textúra memória

A textúra memória egy speciális, optimalizált memória típus, amely elsősorban képek és más nagy adatstruktúrák kezelésére szolgál. A textúra memória cache-elésre kerül, és kifejezetten a 2D és 3D adatok gyors elérésére van optimalizálva. Emellett különböző interpolációs módokat is támogat, amelyek hasznosak lehetnek képfeldolgozási feladatok során.

##### Textúra memória deklarálása és elérése

A textúra memória deklarálása bonyolultabb, mint a konstans memóriaé, mivel speciális textúra referenciákat és textúra objektumokat kell használni.

###### Példa: Textúra memória deklarálása

```cpp
texture<float, cudaTextureType1D, cudaReadModeElementType> texRef;
```

A fenti példában egy 1D textúra referenciát deklarálunk, amely `float` típusú elemeket tartalmaz. A textúra referenciát a host kódban kell inicializálni és kötni a megfelelő adatforráshoz.

##### Textúra adat kötése

A textúra adatot a `cudaBindTexture` függvénnyel köthetjük a textúra referenciához.

###### Példa: Textúra adat kötése

```cpp
float h_data[256] = { /* adatok inicializálása */ };
float *d_data;
cudaMalloc((void**)&d_data, sizeof(float) * 256);
cudaMemcpy(d_data, h_data, sizeof(float) * 256, cudaMemcpyHostToDevice);

cudaBindTexture(0, texRef, d_data, sizeof(float) * 256);
```

A fenti példában a `h_data` nevű host oldali adatokat átmásoljuk a GPU globális memóriájába, majd ezt az adatot kötjük a `texRef` textúra referenciához.

##### Textúra memória használata a kernelben

A textúra memória használata a kernelben szintén egyszerű. A `tex1Dfetch` függvény segítségével érhetjük el a textúra adatokat.

###### Példa: Textúra memória használata

```cpp
__global__ void kernelUsingTextureMemory(float *output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    output[idx] = tex1Dfetch(texRef, idx);
}
```

Ebben a példában minden szál olvassa a textúra memóriából az `texRef` megfelelő elemét, és az eredményt az `output` tömbbe írja.

#### Előnyök és alkalmazási területek

A konstans és textúra memória különböző előnyöket kínál, és különböző alkalmazási területeken használhatók hatékonyan.

##### Konstans memória előnyei

1. **Gyors hozzáférés**: A konstans memória gyorsan elérhető a szálak számára, különösen akkor, ha a hozzáférés koherens, azaz minden szál ugyanazt az értéket olvassa.
2. **Egyszerű használat**: A konstans memória deklarálása és elérése egyszerű, és különösen hasznos olyan adatok tárolására, amelyek nem változnak a kernel futása során.

##### Textúra memória előnyei

1. **Cache-elés**: A textúra memória cache-elve van, ami gyors hozzáférést biztosít az adatokhoz.
2. **Interpoláció**: A textúra memória támogatja az interpolációt, amely hasznos lehet képfeldolgozási feladatok során.
3. **Speciális hozzáférési módok**: A textúra memória különböző hozzáférési módokat támogat, amelyek optimalizálják a 2D és 3D adatok kezelését.

##### Alkalmazási területek

1. **Képfeldolgozás**: A textúra memória különösen hasznos a képfeldolgozási feladatokban, ahol nagy mennyiségű képadatot kell gyorsan elérni és manipulálni.
2. **Állandó adatok**: A konstans memória ideális állandó adatok tárolására, amelyek a kernel futása során nem változnak.
3. **Számítási feladatok**: Mindkét memória típus használható különböző számítási feladatokban, ahol a gyors memória hozzáférés és az adatok optimalizált kezelése kritikus a teljesítmény szempontjából.

#### Összefoglalás

A konstans és textúra memória használata jelentős teljesítményjavulást eredményezhet a CUDA programozásban. A konstans memória gyorsan elérhető és ideális állandó értékek tárolására, míg a textúra memória cache-elve van és kifejezetten a 2D és 3D adatok kezelésére optimalizált. Az itt bemutatott példák és technikák segítenek abban, hogy hatékonyan használjuk ki ezeket a speciális memória típusokat, és ezáltal növeljük a párhuzamos számítások teljesítményét.

