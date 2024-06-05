\newpage

## 2.13.   Párhuzamos és elosztott rendezési algoritmusok

Az adatszerkezetek és algoritmusok világában a rendezési eljárások kiemelkedő szerepet játszanak. A hatékony rendezési algoritmusok alapvető fontosságúak számos alkalmazási területen, beleértve az adatbázis-kezelést, a keresési algoritmusokat és a nagy mennyiségű adat feldolgozását. Ahogy az adatméretek és az igények növekednek, úgy válik egyre fontosabbá a rendezési folyamatok gyorsítása párhuzamos és elosztott rendszerek segítségével. Ebben a fejezetben áttekintjük a párhuzamos és elosztott rendezési algoritmusokat, beleértve a párhuzamos quicksort és mergesort, a bitonic sort és parallel bucket sort technikákat, valamint az MPI és GPU alapú rendezési módszereket. Célunk bemutatni, hogyan használhatók ezek az algoritmusok a számítási teljesítmény maximalizálására és a nagy adathalmazok hatékony rendezésére különböző párhuzamos környezetekben.

### 2.13.1. Párhuzamos quicksort és mergesort

A párhuzamos rendezési algoritmusok terén a quicksort és a mergesort két kiemelkedő példa, amelyek hatékonyan kihasználják a modern számítógépes architektúrák többmagos képességeit. Mindkét algoritmus jelentős kutatási területet képvisel a párhuzamos feldolgozásban, és számos alkalmazásban sikeresen használják őket. Ebben az alfejezetben részletesen tárgyaljuk a párhuzamos quicksort és mergesort algoritmusok működését, implementációját és előnyeit.

#### Párhuzamos Quicksort

A quicksort egy oszd meg és uralkodj típusú algoritmus, amely egy pivot elemet választ ki, majd az elemeket két alhalmazra osztja: az egyik alhalmazba kerülnek a pivotnál kisebb, a másikba pedig a pivotnál nagyobb elemek. A párhuzamos quicksort ezt az elvet követi, de az egyes alhalmazok rendezését párhuzamosan végzi.

##### Algoritmus leírása

1. **Pivot kiválasztása**: A pivot elemet általában véletlenszerűen választják ki, bár más módszerek is használhatók, például a median-of-three.
2. **Partícionálás**: Az elemeket két alhalmazra osztják a pivot alapján.
3. **Rekurzív rendezés**: A két alhalmazt párhuzamosan rendezik.
4. **Összeillesztés**: Mivel a quicksort in-place rendező algoritmus, nincs szükség különösebb összeillesztési lépésre.

##### Implementáció C++ nyelven

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <future>
#include <thread>

// Partícionáló függvény
int partition(std::vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; ++j) {
        if (arr[j] < pivot) {
            std::swap(arr[++i], arr[j]);
        }
    }
    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}

// Párhuzamos quicksort függvény
void parallel_quicksort(std::vector<int>& arr, int low, int high, int depth) {
    if (low < high) {
        int pi = partition(arr, low, high);

        if (depth > 0) {
            auto left_future = std::async(std::launch::async, parallel_quicksort, std::ref(arr), low, pi - 1, depth - 1);
            auto right_future = std::async(std::launch::async, parallel_quicksort, std::ref(arr), pi + 1, high, depth - 1);
            left_future.get();
            right_future.get();
        } else {
            parallel_quicksort(arr, low, pi - 1, depth);
            parallel_quicksort(arr, pi + 1, high, depth);
        }
    }
}

int main() {
    std::vector<int> data = {38, 27, 43, 3, 9, 82, 10};
    int max_depth = std::thread::hardware_concurrency();
    parallel_quicksort(data, 0, data.size() - 1, max_depth);
    
    for (int i : data) {
        std::cout << i << " ";
    }
    return 0;
}
```

##### Előnyök és kihívások

A párhuzamos quicksort egyik legnagyobb előnye a gyors működési idő, különösen nagy adathalmazok esetén. A rekurzív természet miatt a munkaterhelés egyenletesen elosztható a különböző processzor magok között. Azonban kihívást jelenthet a partícionálási lépés hatékony végrehajtása, mivel az in-place partícionálás során jelentős szinkronizációs költségek merülhetnek fel.

#### Párhuzamos Mergesort

A mergesort szintén egy oszd meg és uralkodj típusú algoritmus, amely két részre osztja a listát, majd a részeket külön-külön rendezi, végül pedig összefűzi a rendezett részeket. A párhuzamos mergesort hasonlóan működik, de a részek rendezését és összefűzését párhuzamosan végzi.

##### Algoritmus leírása

1. **Felbontás**: Az eredeti listát két egyenlő részre osztják.
2. **Rekurzív rendezés**: A két részt párhuzamosan rendezik.
3. **Összefűzés**: A rendezett részeket összeillesztik, ami párhuzamosan is végezhető, ha megfelelő adatstruktúrákat használnak.

##### Implementáció C++ nyelven

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <future>
#include <thread>

// Merge függvény
void merge(std::vector<int>& arr, int left, int mid, int right) {
    std::vector<int> left_sub(arr.begin() + left, arr.begin() + mid + 1);
    std::vector<int> right_sub(arr.begin() + mid + 1, arr.begin() + right + 1);

    int i = 0, j = 0, k = left;
    while (i < left_sub.size() && j < right_sub.size()) {
        if (left_sub[i] <= right_sub[j]) {
            arr[k++] = left_sub[i++];
        } else {
            arr[k++] = right_sub[j++];
        }
    }
    while (i < left_sub.size()) {
        arr[k++] = left_sub[i++];
    }
    while (j < right_sub.size()) {
        arr[k++] = right_sub[j++];
    }
}

// Párhuzamos mergesort függvény
void parallel_mergesort(std::vector<int>& arr, int left, int right, int depth) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        if (depth > 0) {
            auto left_future = std::async(std::launch::async, parallel_mergesort, std::ref(arr), left, mid, depth - 1);
            auto right_future = std::async(std::launch::async, parallel_mergesort, std::ref(arr), mid + 1, right, depth - 1);
            left_future.get();
            right_future.get();
        } else {
            parallel_mergesort(arr, left, mid, depth);
            parallel_mergesort(arr, mid + 1, right, depth);
        }

        merge(arr, left, mid, right);
    }
}

int main() {
    std::vector<int> data = {38, 27, 43, 3, 9, 82, 10};
    int max_depth = std::thread::hardware_concurrency();
    parallel_mergesort(data, 0, data.size() - 1, max_depth);
    
    for (int i : data) {
        std::cout << i << " ";
    }
    return 0;
}
```

##### Előnyök és kihívások

A párhuzamos mergesort egyik legnagyobb előnye a stabilitása és kiszámítható futási ideje, amely O(n log n) minden esetben. A mergesort jól működik nagy, rendezett adathalmazok esetén is, mivel a rendezés közben nem változik az elemek relatív sorrendje. Azonban az algoritmus hátránya, hogy jelentős mennyiségű memóriát igényel, mivel a merge lépés során különálló részeket kell tárolni és kezelni. A párhuzamos mergesort hatékonyságát tovább növelheti az adatstruktúrák és a memóriaelrendezés optimalizálása.

#### Összegzés

A párhuzamos quicksort és mergesort algoritmusok jelentős előrelépést jelentenek a nagy adathalmazok rendezésében. Mindkét algoritmus különböző előnyökkel és kihívásokkal rendelkezik, amelyek befolyásolják a választást az alkalmazás specifikus követelményei szerint. A párhuzamos feldolgozásban rejlő lehetőségek kihasználása kulcsfontosságú a modern számítástechnikai feladatok hatékony megoldásában, és ezen algoritmusok alkalmazása jelentős teljesítménynövekedést eredményezhet.

### 2.13.2. Bitonikus rendezés és párhuzamos vödör rendezés

Ebben a részben két speciális, párhuzamos számítási környezetekre optimalizált rendezési algoritmust tárgyalunk: a bitonikus rendezést és a párhuzamos vödör rendezést. Ezek az algoritmusok úgy lettek tervezve, hogy kihasználják a modern számítási architektúrák párhuzamos feldolgozási képességeit a gyorsabb rendezési idők elérése érdekében.

#### Bitonikus rendezés

A bitonikus rendezés egy összehasonlításon alapuló rendezési algoritmus, amely különösen hatékony a párhuzamos feldolgozásban. Ken Batcher 1968-ban bemutatott algoritmusa egy olyan sorozat összehasonlításokon alapul, amelyek hatékonyan oszthatók meg több processzoron.

**Algoritmus áttekintése**

A bitonikus rendezési folyamat egy bitonikus sorozat létrehozásával és annak rendezésével jár. Egy bitonikus sorozat olyan számsorozat, amely először növekszik, majd csökken, vagy először csökken, majd növekszik. A bitonikus rendezés alapelve, hogy az egész sorozatot kisebb bitonikus sorozatokra osztja, majd ezeket úgy egyesíti, hogy az eredményül kapott sorozat bitonikusan rendezett legyen. Ezt az egyesítési folyamatot bitonikus egyesítésnek nevezzük.

**A bitonikus rendezés lépései**

1. **Felosztás**: Oszd meg a tömböt két részre - az első felét növekvő, a második felét csökkenő sorrendben.
2. **Bitonikus egyesítés**: Ez a lépés egy rekurzív művelet, ahol a tömb minden fele bitonikus egyesítésen megy keresztül, ami bitonikus sorozatot képez, amelyet tovább osztanak és rekurzívan rendeznek, amíg az egész tömb rendezetté nem válik.

**Párhuzamosítás**

A bitonikus rendezés kiválóan alkalmas párhuzamosításra, mivel a bitonikus egyesítés során végzett összehasonlítások és csere műveletek függetlenül és párhuzamosan hajthatók végre. Ez a tulajdonság rendkívül alkalmas a többmagos processzorokon, GPU-kon vagy elosztott számítási platformokon történő implementálásra.

**Példa implementáció C++ nyelven**

```cpp
#include <algorithm>
#include <vector>
#include <omp.h>

void compareAndSwap(int &a, int &b, bool ascending) {
    if ((ascending && a > b) || (!ascending && a < b)) {
        std::swap(a, b);
    }
}

void bitonicMerge(std::vector<int>& arr, int low, int cnt, bool ascending) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++) {
            compareAndSwap(arr[i], arr[i + k], ascending);
        }
        bitonicMerge(arr, low, k, ascending);
        bitonicMerge(arr, low + k, k, ascending);
    }
}

void bitonicSort(std::vector<int>& arr, int low, int cnt, bool ascending) {
    if (cnt > 1) {
        int k = cnt / 2;
        bitonicSort(arr, low, k, true);
        bitonicSort(arr, low + k, k, false);
        bitonicMerge(arr, low, cnt, ascending);
    }
}

void parallelBitonicSort(std::vector<int>& arr) {
    #pragma omp parallel
    {
        #pragma omp single
        bitonicSort(arr, 0, arr.size(), true);
    }
}
```

#### Párhuzamos vödör rendezés

A párhuzamos vödör rendezés a klasszikus vödör rendezés algoritmusának kiterjesztése, amelyet úgy terveztek, hogy kihasználja a párhuzamos feldolgozási képességeket a nagy adatkészletek hatékony rendezéséhez.

**Algoritmus áttekintése**

A párhuzamos vödör rendezésnél az adatokat először több vödörbe osztják. Minden egyes vödröt aztán függetlenül rendeznek. Végül a vödrök tartalmát összefűzik egy rendezett tömbbé. Az adatok vödrökbe osztását úgy végzik, hogy az adatok tartományát egyenletesen osztják szét, így minden vödör egy adott adatainak altartományát kezeli.

**A párhuzamos vödör rendezés lépései**

1. **Elosztás**: Oszd meg az adatokat vödrökbe egy hash függvény vagy egy tartományképlet alapján.
2. **Helyi Rendezés**: Rendezd az adatokat minden egyes vödörben. Ez a lépés párhuzamosan végezhető az összes vödörön.
3. **Összefűzés**: Fűzd össze a vödröket egyetlen rendezett tömbbé.

**Párhuzamosítás**

A párhuzamos vödör rendezés legnagyobb előnye, hogy a vödrök rendezése párhuzamosan végezhető, jelentősen csökkentve az összrendezési időt. Ez különösen hatékony, amikor nagy tömböket kell rendezni, amelyek különböző csomópontokra vannak szétosztva egy klaszterben.

**Példa implementáció C++ nyelven**

```cpp
#include <vector>
#include <algorithm>
#include <omp.h>

void parallelBucketSort(std::vector<int>& arr) {
    const int n = arr.size();
    const int bucketCount = omp_get_max_threads();
    std::vector<std::vector<int>> buckets(bucketCount);

    int minValue = *min_element(arr.begin(), arr.end());
    int maxValue = *max_element(arr.begin(), arr.end());
    int range = maxValue - minValue + 1;
    int rangePerBucket = range / bucketCount;

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int bucketIndex = (arr[i] - minValue) / rangePerBucket;
        if (bucketIndex >= bucketCount) bucketIndex = bucketCount - 1;
        buckets[bucketIndex].push_back(arr[i]);
    }

    #pragma omp parallel for
    for (int i = 0; i < bucketCount; i++) {
        std::sort(buckets[i].begin(), buckets[i].end());
    }

    int index = 0;
    for (int i = 0; i < bucketCount; i++) {
        for (int num : buckets[i]) {
            arr[index++] = num;
        }
    }
}
```

### 2.13.3. MPI és GPU alapú rendezési technikák

Ebben a részben a modern párhuzamosítási technológiák, mint az Üzenetátviteli Felület (MPI) és a Grafikus Feldolgozó Egységek (GPU) alapú rendezési technikákat vizsgáljuk meg. Ezek az eszközök lehetővé teszik az adatrendezési műveletek nagymértékű gyorsítását nagy adatkészleteken keresztül, különösen elosztott rendszerek és magas teljesítményű számítógépes környezetek esetén.

#### MPI alapú rendezési technikák

Az MPI egy szabványosított és hordozható üzenetátviteli szoftver interfész, amelyet kifejezetten elosztott memóriás rendszerek számára terveztek. Ebben a modellben az egyes folyamatok önállóan futnak, és üzeneteket küldenek/kapnak egymás között az adatok megosztása érdekében.

**Működési Elve**

Az MPI-vel megvalósított rendezési technikák gyakran használnak megosztott adatmodelleket, ahol az adatokat több folyamat között osztják szét. A rendezési műveleteket ezután párhuzamosan hajtják végre az egyes folyamatokon, majd egy összegző lépés során egyesítik az eredményeket.

**MPI Rendezési Algoritmusok**

- **Többszintű Összehasonlítás és Csere (Multi-level Compare-and-Swap)**: Ez az algoritmus az adatokat folyamatok között osztja szét, ahol minden folyamat a saját adathalmazát rendezheti egy helyi rendezési algoritmus (pl. gyorsrendezés) segítségével. Ezután az adatokat összegyűjtik és globálisan összehasonlítják.
- **Mintaalapú Rendezés (Sample Sort)**: Ez a módszer egy központi mintavételi lépést használ az adatok partícióinak meghatározására, amely alapján a folyamatok partíciókat rendeznek, majd a rendezett partíciók összefésülésre kerülnek.

#### GPU alapú rendezési technikák

A GPU-kat eredetileg grafikus adatok feldolgozására tervezték, de kiválóan alkalmasak általános célú számítási feladatokra is (GPGPU), különösen az adatintenzív és párhuzamosítható feladatokra, mint amilyen a rendezés.

**Működési Elve**

A GPU-k több száz vagy ezer kisebb számítási egységből állnak, amelyek képesek nagyszámú művelet gyors párhuzamos végrehajtására. Az adatokat a GPU memóriájába töltik, ahol a számítási egységek egyszerre több adatelemet dolgoznak fel.

**GPU Rendezési Algoritmusok**

- **Bitonikus Rendezés**: A GPU-kon történő bitonikus rendezés az adatokat egy olyan struktúrába rendezheti, ahol az adatok bitonikus sorrendben vannak, majd a párhuzamos bitonikus egyesítési lépésekkel rendezik a teljes adatkészletet.
- **Radix Rendezés**: A radix rendezés a digitális rendezés egy formája, amely különösen hatékony a GPU-n, mivel a radix rendezés jól párhuzamosítható a bináris digitális szintek mentén történő adatfeldolgozás révén.

**Példa Bitonikus Rendezés Implementációja CUDA-val**

```cpp
__global__ void bitonicSortKernel(int *deviceValues, int j, int k) {
    unsigned int i, ixj;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i^j;

    if ((ixj)>i) {
        if ((i&k) == 0) {
            if (deviceValues[i] > deviceValues[ixj]) {
                int temp = deviceValues[i];
                deviceValues[i] = deviceValues[ixj];
                deviceValues[ixj] = temp;
            }
        }
        if ((i&k) != 0) {
            if (deviceValues[i] < deviceValues[ixj]) {
                int temp = deviceValues[i];
                deviceValues[i] = deviceValues[ixj];
                deviceValues[ixj] = temp;
            }
        }
    }
}

void bitonicSort(int *values, const int size) {
    int *deviceValues;
    cudaMalloc(&deviceValues, size * sizeof(int));
    cudaMemcpy(deviceValues, values, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blocks(128);
    dim3 threads(128);

    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k >> 1; j>0; j >>= 1) {
            bitonicSortKernel<<<blocks, threads>>>(deviceValues, j, k);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(values, deviceValues, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(deviceValues);
}
```

Ez a kód a CUDA programozási modellt használja a bitonikus rendezés implementálásához, kihasználva a GPU párhuzamos feldolgozási képességeit a gyors és hatékony adatrendezés érdekében. Az implementáció demonstrálja, hogy a párhuzamos algoritmusok, mint a bitonikus rendezés, jelentősen gyorsíthatják az adatfeldolgozást, kihasználva a modern számítástechnikai architektúrákat.
