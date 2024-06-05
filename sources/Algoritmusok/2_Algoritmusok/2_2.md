\newpage
## 2.2. Kiválasztásos rendezés (Selection Sort)

A kiválasztásos rendezés egy egyszerű, mégis hatékony algoritmus, amelyet széles körben alkalmaznak különböző informatikai problémák megoldására. Alapelve, hogy az algoritmus minden egyes lépésben kiválasztja a legkisebb elemet a rendezendő tömb nem rendezett részéből, majd kicseréli azt a soron következő pozícióval. Ez a módszer egyszerűsége miatt könnyen érthető és implementálható, ezért gyakran tanítják bevezető programozási kurzusokon. A fejezet részletesen bemutatja a kiválasztásos rendezés alapelveit és implementációját, optimalizált változatait, valamint elemzi a teljesítményt és komplexitást. Emellett gyakorlati alkalmazásokkal és példákkal is illusztráljuk az algoritmus használatát, hogy az olvasók átfogó képet kapjanak a kiválasztásos rendezés gyakorlati jelentőségéről.

### 2.2.1. Alapelvek és implementáció

A kiválasztásos rendezés (Selection Sort) az egyik legegyszerűbb és legközismertebb rendezési algoritmus, amelyet számos különböző típusú adat rendezésére használnak. Az algoritmus fő elve, hogy a rendezendő tömb elemeit ismétlődően átvizsgálva, minden egyes iteráció során kiválasztja a legkisebb elemet, és azt a megfelelő helyre cseréli. Ez az egyszerű módszer hatékonyan rendez bármilyen típusú numerikus vagy nem numerikus adatot, és széles körben alkalmazzák oktatási célokra az algoritmusok és adatszerkezetek oktatása során.

#### Algoritmus alapelve

A kiválasztásos rendezés működési elve a következő lépésekben foglalható össze:

1. **Kezdeti állapot**: Kezdjük azzal, hogy a tömb első eleme a rendezett része a tömbnek, és a maradék elem a rendezetlen rész.
2. **Legkisebb elem keresése**: Az aktuális helyzetből indulva keressük meg a legkisebb elemet a tömb rendezetlen részében.
3. **Csere**: Cseréljük ki a legkisebb elemet a rendezetlen rész első elemével.
4. **Lépés az új pozícióra**: Mozgassuk a határt a rendezett és a rendezetlen rész között egy elemmel jobbra.
5. **Ismétlés**: Ismételjük a folyamatot addig, amíg a tömb minden eleme rendezetté válik.

Az algoritmus pseudokódja a következőképpen néz ki:

```
for i = 0 to n-1 do
    min_index = i
    for j = i+1 to n do
        if array[j] < array[min_index] then
            min_index = j
    swap(array[i], array[min_index])
end for
```

A fenti pseudokódban az algoritmus végigmegy a tömb összes elemén, minden iterációban megkeresi a legkisebb elemet a rendezetlen részben, majd kicseréli azt a rendezetlen rész első elemével.

#### Részletes implementáció C++ nyelven

Az alábbiakban bemutatjuk a kiválasztásos rendezés algoritmusának C++ nyelvű implementációját. Ez az implementáció illusztrálja az algoritmus egyszerűségét és hatékonyságát.

```cpp
#include <iostream>
#include <vector>

void selectionSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        int min_index = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_index]) {
                min_index = j;
            }
        }
        std::swap(arr[i], arr[min_index]);
    }
}

void printArray(const std::vector<int>& arr) {
    for (int num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::vector<int> data = {64, 25, 12, 22, 11};
    std::cout << "Unsorted array: ";
    printArray(data);

    selectionSort(data);

    std::cout << "Sorted array: ";
    printArray(data);

    return 0;
}
```

Ez az egyszerű program bemutatja, hogyan rendezhetünk egy integer tömböt a kiválasztásos rendezés algoritmusával. A program elején deklarálunk egy `selectionSort` nevű függvényt, amely végrehajtja a rendezést. A `main` függvény tartalmaz egy példatömböt, amelyet rendezünk, és kiírjuk a rendezés előtti és utáni állapotát.

#### Algoritmus hatékonysága és komplexitása

A kiválasztásos rendezés algoritmusának hatékonyságát több szempontból is érdemes megvizsgálni. Először is, nézzük meg a legrosszabb, legjobb és átlagos eset komplexitását.

1. **Legrosszabb eset (Worst-case complexity)**: Az algoritmus legrosszabb esetben $O(n^2)$ időkomplexitással működik, mivel minden egyes elemet meg kell vizsgálni és összehasonlítani a tömb többi elemével. Ez akkor következik be, ha a tömb elemei fordított sorrendben vannak.

2. **Legjobb eset (Best-case complexity)**: A legjobb eset is $O(n^2)$ időkomplexitással rendelkezik, mivel az algoritmus mindig végrehajtja a belső ciklust, függetlenül attól, hogy a tömb elemei már rendezettek-e vagy sem.

3. **Átlagos eset (Average-case complexity)**: Az átlagos eset időkomplexitása szintén $O(n^2)$, mivel a tömb elemeinek összehasonlítása és cseréje minden esetben szükséges.

4. **Térbeli komplexitás (Space complexity)**: A kiválasztásos rendezés egy in-place algoritmus, ami azt jelenti, hogy nem igényel további memóriahelyet a bemeneti tömbön kívül. Ezért a térbeli komplexitása $O(1)$.

#### Az algoritmus előnyei és hátrányai

**Előnyök**:

- Egyszerű és könnyen érthető, ami ideálissá teszi oktatási célokra.
- Nem igényel extra memóriát, mivel in-place működik.
- Stabilnak tekinthető, ha megfelelő módosításokat hajtanak végre az egyenlő elemek kezelésére.

**Hátrányok**:

- Nem hatékony nagyobb tömbök rendezésére, mivel a legrosszabb és átlagos eset időkomplexitása $O(n^2)$.
- Nem adaptív, ami azt jelenti, hogy a már részben rendezett tömbök esetében sem gyorsul fel.

#### Összegzés

A kiválasztásos rendezés alapelvei és implementációja egyszerűek és könnyen érthetőek, így ideálisak az algoritmusok tanulásának kezdeti szakaszában. Bár nem a leghatékonyabb rendezési algoritmus, bizonyos körülmények között, különösen kis adathalmazok esetén, mégis megfelelően teljesít. Az in-place működés és az alacsony térbeli komplexitás miatt számos gyakorlati alkalmazásban használható, ahol a memóriahasználat minimalizálása fontos szempont.

### 2.2.2. Optimalizált változatok

A kiválasztásos rendezés (Selection Sort) alapvető algoritmusa egyszerű és jól érthető, azonban a gyakorlati alkalmazások során számos optimalizálási lehetőség kínálkozik, amelyek révén javítható az algoritmus teljesítménye bizonyos szempontok szerint. Az optimalizálás során főként az idő- és térbeli komplexitás, valamint a stabilitás kerülnek előtérbe. Ebben az alfejezetben részletesen tárgyaljuk a kiválasztásos rendezés különböző optimalizálási technikáit és azok hatásait.

#### 1. Tömb csökkentett átvizsgálása

Az egyik egyszerű, de hatékony optimalizálási módszer a tömb csökkentett átvizsgálása. Alapvetően a kiválasztásos rendezés minden iterációban a tömb egyre kisebb részét vizsgálja át, azonban az algoritmus optimalizálható úgy, hogy ne végezzen felesleges összehasonlításokat.

##### Optimalizált Pseudokód

```
for i = 0 to n-1 do
    min_index = i
    for j = i+1 to n do
        if array[j] < array[min_index] then
            min_index = j
    if min_index != i then
        swap(array[i], array[min_index])
end for
```

##### Hatás

Ez az optimalizálás csökkenti a felesleges csere műveletek számát, ezáltal javítva az algoritmus futási idejét bizonyos esetekben. Ha a minimális elem már a megfelelő helyen van, akkor nincs szükség a cserére, ami kisebb időmegtakarítást eredményezhet.

#### 2. Kettős kiválasztásos rendezés

A kettős kiválasztásos rendezés (Double Selection Sort) egy másik optimalizálási módszer, amely egyszerre keresi a legkisebb és a legnagyobb elemet a tömb rendezetlen részében, és ezeket a megfelelő helyre cseréli. Ezzel a módszerrel az algoritmus futási ideje jelentősen javítható, mivel egy iteráció során két elemet helyezünk a végső helyére.

##### Optimalizált Pseudokód

```
for i = 0 to n/2 do
    min_index = i
    max_index = i
    for j = i+1 to n-i-1 do
        if array[j] < array[min_index] then
            min_index = j
        if array[j] > array[max_index] then
            max_index = j
    if min_index != i then
        swap(array[i], array[min_index])
    if max_index != n-i-1 then
        swap(array[n-i-1], array[max_index])
end for
```

##### Hatás

Ez az optimalizálás lehetővé teszi, hogy az algoritmus párhuzamosan rendezze a tömb elejét és végét, így csökkentve az iterációk számát. Bár az időkomplexitás továbbra is $O(n^2)$ marad, az iterációk száma és a szükséges összehasonlítások száma csökken, ami gyakorlati esetekben időmegtakarítást eredményezhet.

#### 3. Stabil kiválasztásos rendezés

A kiválasztásos rendezés alapvetően nem stabil, ami azt jelenti, hogy az egyenlő értékű elemek sorrendje változhat a rendezés során. A stabilitás biztosítása érdekében az algoritmus módosítható úgy, hogy a kiválasztott elemet nem cseréli ki, hanem beszúrja a megfelelő helyre.

##### Optimalizált Pseudokód

```
for i = 0 to n-1 do
    min_index = i
    for j = i+1 to n do
        if array[j] < array[min_index] then
            min_index = j
    key = array[min_index]
    while min_index > i do
        array[min_index] = array[min_index-1]
        min_index = min_index - 1
    array[i] = key
end for
```

##### Hatás

Ez az optimalizálás biztosítja a stabilitást, mivel az egyenlő elemek sorrendje nem változik meg a rendezés során. Azonban ennek az az ára, hogy az algoritmus futási ideje megnövekedhet a beszúrási műveletek miatt.

#### 4. Párhuzamos kiválasztásos rendezés

A párhuzamos feldolgozás alkalmazása szintén egy hatékony optimalizálási módszer lehet, különösen nagy adathalmazok esetén. A párhuzamos kiválasztásos rendezés során a tömb több részre osztható, és az egyes részek párhuzamosan rendezhetők különböző processzorokon vagy szálakon.

##### Optimalizált Implementáció

A párhuzamos kiválasztásos rendezés implementálása összetettebb feladat, amely speciális könyvtárakat és technikákat igényel, mint például az OpenMP vagy a C++11 Thread Library. Az alábbi példa C++ nyelven szemlélteti a párhuzamos feldolgozás alapelveit.

```cpp
#include <iostream>
#include <vector>
#include <thread>

void selectionSort(std::vector<int>& arr, int start, int end) {
    for (int i = start; i < end - 1; i++) {
        int min_index = i;
        for (int j = i + 1; j < end; j++) {
            if (arr[j] < arr[min_index]) {
                min_index = j;
            }
        }
        std::swap(arr[i], arr[min_index]);
    }
}

void parallelSelectionSort(std::vector<int>& arr) {
    int n = arr.size();
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    int chunk_size = n / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? n : start + chunk_size;
        threads.emplace_back(selectionSort, std::ref(arr), start, end);
    }

    for (auto& th : threads) {
        th.join();
    }
    
    // Merging sorted chunks (could be further optimized)
    for (int i = 0; i < num_threads - 1; ++i) {
        int mid = (i + 1) * chunk_size;
        std::inplace_merge(arr.begin(), arr.begin() + mid, arr.begin() + (i + 2) * chunk_size);
    }
}

int main() {
    std::vector<int> data = {64, 25, 12, 22, 11, 90, 55, 32, 76, 30};
    std::cout << "Unsorted array: ";
    for (int num : data) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    parallelSelectionSort(data);

    std::cout << "Sorted array: ";
    for (int num : data) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

##### Hatás

A párhuzamos kiválasztásos rendezés jelentősen csökkentheti a futási időt nagy adathalmazok esetén, különösen modern többmagos processzorokon. Azonban a párhuzamosítás komplexitása és a szinkronizációs költségek miatt ez a megközelítés csak akkor hatékony, ha a párhuzamosítás által nyert teljesítménynövekedés meghaladja a párhuzamosítás overhead-jét.

#### Összegzés

A kiválasztásos rendezés optimalizált változatai különböző szempontok szerint javíthatják az algoritmus teljesítményét. Az egyes optimalizálási módszerek, mint például a csökkentett átvizsgálás, a kettős kiválasztásos rendezés, a stabil rendezés és a párhuzamos feldolgozás, különböző helyzetekben és különböző típusú adathalmazok esetén nyújtanak előnyöket. Az optimalizált változatok alkalmazásával a kiválasztásos rendezés hatékonyabbá és rugalmasabbá válik, lehetővé téve a nagyobb és összetettebb adathalmazok rendezését is.

### 2.2.3. Teljesítmény és komplexitás elemzése

A kiválasztásos rendezés (Selection Sort) algoritmus teljesítményének és komplexitásának elemzése fontos szempont az algoritmus megértése és alkalmazása szempontjából. Ebben az alfejezetben részletesen tárgyaljuk az idő- és térbeli komplexitást, valamint az algoritmus gyakorlati teljesítményét különböző körülmények között. Emellett kitérünk az összehasonlítások és cserék számának elemzésére, és megvizsgáljuk az algoritmus stabilitását és adaptivitását.

#### Időkomplexitás

Az időkomplexitás az egyik legfontosabb szempont, amely meghatározza egy algoritmus hatékonyságát. A kiválasztásos rendezés időkomplexitása három esetben vizsgálható: legrosszabb eset, legjobb eset és átlagos eset.

1. **Legrosszabb eset (Worst-case complexity)**:
   A legrosszabb eset időkomplexitása akkor következik be, amikor a tömb elemei fordított sorrendben vannak rendezve. Ebben az esetben az algoritmus minden egyes iterációban az összes elemet összehasonlítja, mielőtt kiválasztaná a legkisebb elemet.
    - Külső ciklus: $n$ iteráció
    - Belső ciklus: az első iterációban $n-1$ összehasonlítás, a második iterációban $n-2$ összehasonlítás, és így tovább.

   Az összes összehasonlítás száma:
   $$
   \sum_{i=1}^{n-1} (n - i) = \frac{n(n-1)}{2} = O(n^2)
   $$

2. **Legjobb eset (Best-case complexity)**:
   A legjobb eset időkomplexitása akkor következik be, amikor a tömb elemei már rendezettek. A kiválasztásos rendezés esetében a legjobb eset időkomplexitása megegyezik a legrosszabb eset időkomplexitásával, mivel az algoritmus minden egyes iterációban végrehajtja az összehasonlításokat.
   $$
   O(n^2)
   $$

3. **Átlagos eset (Average-case complexity)**:
   Az átlagos eset időkomplexitása szintén $O(n^2)$, mivel az összehasonlítások száma független a bemeneti tömb kezdeti rendezési állapotától. Az átlagos esetben is minden egyes iterációban az összes elem összehasonlítása szükséges.

#### Térbeli komplexitás

A térbeli komplexitás azt méri, hogy mennyi extra memória szükséges az algoritmus futása során. A kiválasztásos rendezés in-place algoritmus, ami azt jelenti, hogy a rendezést a bemeneti tömbben hajtja végre, és nem igényel extra memóriahelyet a bemeneti tömbön kívül.

- **Térbeli komplexitás**:
  $$
  O(1)
  $$

Ez azt jelenti, hogy a kiválasztásos rendezés nagyon hatékonyan használja a memóriát, mivel csak néhány segédváltozót használ az indexek és a csere műveletek kezelésére.

#### Összehasonlítások és cserék száma

A kiválasztásos rendezés során végrehajtott összehasonlítások és cserék száma szintén fontos tényező a teljesítmény szempontjából. Ezek az értékek meghatározzák az algoritmus hatékonyságát és befolyásolják a futási időt.

1. **Összehasonlítások száma**:
   Mint korábban említettük, az összehasonlítások száma mindhárom esetben $O(n^2)$, mivel az algoritmus minden iterációban végrehajtja az összehasonlításokat.

2. **Cserék száma**:
   A cserék száma azonban jelentősen kevesebb lehet. Minden iterációban legfeljebb egy csere történik, amikor a legkisebb elemet kicseréljük a jelenlegi elemmel.
   $$
   O(n)
   $$

Ez azt jelenti, hogy a kiválasztásos rendezés viszonylag hatékony a cserék szempontjából, mivel kevesebb cserét hajt végre, mint az összehasonlítások száma.

#### Stabilitás

A rendezési algoritmusok stabilitása azt jelenti, hogy az egyenlő értékű elemek sorrendje nem változik meg a rendezés során. Az alapvető kiválasztásos rendezés nem stabil, mivel a csere műveletek során az egyenlő értékű elemek sorrendje megváltozhat.

Példa:

Bemeneti tömb: [4a, 5, 3, 4b, 1]
Rendezett tömb: [1, 3, 4b, 4a, 5]

Azonban a korábbi alfejezetben bemutatott stabil kiválasztásos rendezés változata biztosítja a stabilitást azáltal, hogy az elemeket nem cseréli ki, hanem beszúrja a megfelelő helyre.

#### Adaptivitás

Az adaptivitás azt jelenti, hogy az algoritmus gyorsabban fut részben rendezett vagy rendezett adathalmazok esetén. A kiválasztásos rendezés nem adaptív algoritmus, mivel a legjobb esetben is $O(n^2)$ időkomplexitással rendelkezik, függetlenül attól, hogy a bemeneti tömb rendezett vagy részben rendezett.

#### Gyakorlati teljesítmény

A kiválasztásos rendezés gyakorlati teljesítményét számos tényező befolyásolja, beleértve a bemeneti adat méretét, az adatok kezdeti rendezési állapotát, valamint a hardver és szoftver környezetet. Bár az algoritmus időkomplexitása $O(n^2)$, kis adathalmazok esetén gyakran megfelelő teljesítményt nyújt, és egyszerű implementációja miatt népszerű választás oktatási célokra és egyszerűbb feladatok megoldására.

Az optimalizált változatok, mint például a kettős kiválasztásos rendezés és a párhuzamos kiválasztásos rendezés, javíthatják az algoritmus teljesítményét nagyobb adathalmazok esetén. Azonban a gyakorlati alkalmazások során fontos figyelembe venni az algoritmusok közötti különbségeket és a specifikus alkalmazási környezetet.

#### Összegzés

A kiválasztásos rendezés teljesítményének és komplexitásának elemzése rámutat arra, hogy bár az algoritmus egyszerű és könnyen érthető, nem a leghatékonyabb választás nagy adathalmazok rendezésére. Időkomplexitása mindhárom esetben $O(n^2)$, ami jelentős hátrányt jelent más, hatékonyabb algoritmusokkal szemben. Ugyanakkor a kiválasztásos rendezés in-place működése és alacsony térbeli komplexitása miatt bizonyos esetekben mégis előnyös lehet. Az optimalizált változatok és a gyakorlati teljesítmény szempontjainak figyelembevételével a kiválasztásos rendezés továbbra is releváns és hasznos eszköz marad az algoritmusok és adatszerkezetek tanulmányozása során.

### 2.2.4. Gyakorlati alkalmazások és példák

A kiválasztásos rendezés (Selection Sort) algoritmus egyszerűsége és in-place működése miatt számos gyakorlati alkalmazási területen és példában megjelenhet. Bár a nagyobb adathalmazok esetében az időkomplexitása miatt nem mindig a leghatékonyabb választás, bizonyos speciális helyzetekben mégis előnyös lehet. Ebben az alfejezetben részletesen megvizsgáljuk a kiválasztásos rendezés gyakorlati alkalmazásait és konkrét példákat mutatunk be, amelyek szemléltetik az algoritmus hasznosságát.

#### 1. Oktatási célok és alapvető algoritmusok tanítása

A kiválasztásos rendezés az egyik leggyakrabban tanított rendezési algoritmus a számítástudomány és a programozás bevezető kurzusain. Az egyszerűségéből adódóan kiválóan alkalmas arra, hogy a diákok megértsék az alapvető rendezési elveket és az algoritmusok működését. Az algoritmus könnyen megérthető és implementálható, így az oktatók és a diákok egyaránt értékelik a didaktikai értékét.

##### Példa:

Egy egyetemi programozási kurzus során a diákok megtanulják a kiválasztásos rendezés alapelveit és implementációját C++ nyelven. Az alábbi példa szemlélteti a kiválasztásos rendezés egyszerű implementációját.

```cpp
#include <iostream>
#include <vector>

void selectionSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        int min_index = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_index]) {
                min_index = j;
            }
        }
        std::swap(arr[i], arr[min_index]);
    }
}

int main() {
    std::vector<int> data = {64, 25, 12, 22, 11};
    std::cout << "Unsorted array: ";
    for (int num : data) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    selectionSort(data);

    std::cout << "Sorted array: ";
    for (int num : data) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

Ez az egyszerű program segít a diákoknak megérteni a rendezési algoritmusok alapvető működését és a kiválasztásos rendezés alapelveit.

#### 2. Kis méretű adathalmazok rendezése

A kiválasztásos rendezés gyakran használt kis méretű adathalmazok rendezésére, ahol az algoritmus időkomplexitása nem jelentős hátrány. Mivel az algoritmus in-place működik, nem igényel extra memóriahasználatot, ami előnyös lehet olyan alkalmazásokban, ahol a memória korlátozott.

##### Példa:

Egy beágyazott rendszer, például egy mikrovezérlő, amelynek korlátozott memória- és számítási kapacitása van, kis méretű adathalmazokat rendezhet a kiválasztásos rendezés algoritmusával. Az alábbi példa egy beágyazott rendszerben alkalmazott kiválasztásos rendezést mutat be.

```cpp
#include <iostream>
#include <array>

void selectionSort(std::array<int, 5>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        int min_index = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_index]) {
                min_index = j;
            }
        }
        std::swap(arr[i], arr[min_index]);
    }
}

int main() {
    std::array<int, 5> data = {64, 25, 12, 22, 11};
    std::cout << "Unsorted array: ";
    for (int num : data) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    selectionSort(data);

    std::cout << "Sorted array: ";
    for (int num : data) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

Ebben az esetben a kiválasztásos rendezés egyszerűsége és alacsony memóriahasználata teszi ideálissá az algoritmust.

#### 3. Tömbök rendezése hardveres korlátok között

A hardveres korlátok közé tartozik a memória- és processzor korlátozások, amelyek esetében az in-place rendezési algoritmusok, mint a kiválasztásos rendezés, különösen hasznosak. Az alábbiakban bemutatunk egy példát egy beágyazott rendszeren futó alkalmazásra, amely egy szenzor adatainak rendezésére használja a kiválasztásos rendezést.

##### Példa:

Egy szenzor adatainak feldolgozása és rendezése egy beágyazott rendszeren.

```cpp
#include <iostream>
#include <array>

void selectionSort(std::array<float, 10>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        int min_index = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_index]) {
                min_index = j;
            }
        }
        std::swap(arr[i], arr[min_index]);
    }
}

int main() {
    std::array<float, 10> sensorData = {5.5, 2.2, 9.8, 4.4, 7.7, 1.1, 6.6, 3.3, 8.8, 0.0};
    std::cout << "Unsorted sensor data: ";
    for (float num : sensorData) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    selectionSort(sensorData);

    std::cout << "Sorted sensor data: ";
    for (float num : sensorData) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

Ez a példaprogram egy tíz elemű tömböt rendez, amely egy szenzor adatait tartalmazza. Az in-place rendezés előnye, hogy nem igényel extra memóriát, ami kritikus szempont a beágyazott rendszerekben.

#### 4. Prioritási sorrendek kialakítása

A kiválasztásos rendezés használható prioritási sorrendek kialakítására is. Például, ha egy alkalmazásban különböző fontossági sorrendet kell felállítani, a kiválasztásos rendezés egyszerű megoldást nyújthat.

##### Példa:

Egy ügyfélszolgálati rendszerben a beérkező kérések fontossági sorrendjének meghatározása.

```cpp
#include <iostream>
#include <vector>
#include <string>

struct Request {
    std::string description;
    int priority;
};

void selectionSort(std::vector<Request>& requests) {
    int n = requests.size();
    for (int i = 0; i < n - 1; i++) {
        int min_index = i;
        for (int j = i + 1; j < n; j++) {
            if (requests[j].priority < requests[min_index].priority) {
                min_index = j;
            }
        }
        std::swap(requests[i], requests[min_index]);
    }
}

int main() {
    std::vector<Request> requests = {
        {"Request A", 3},
        {"Request B", 1},
        {"Request C", 2},
        {"Request D", 5},
        {"Request E", 4}
    };

    std::cout << "Unsorted requests:" << std::endl;
    for (const auto& request : requests) {
        std::cout << "Description: " << request.description << ", Priority: " << request.priority << std::endl;
    }

    selectionSort(requests);

    std::cout << "\nSorted requests by priority:" << std::endl;
    for (const auto& request : requests) {
        std::cout << "Description: " << request.description << ", Priority: " << request.priority << std::endl;
    }

    return 0;
}
```

Ez a példaprogram bemutatja, hogyan rendezhetők az ügyfélszolgálati kérések fontossági sorrend szerint a kiválasztásos rendezés algoritmusával.

#### 5. Beágyazott rendszerek és IoT alkalmazások

A beágyazott rendszerek és IoT (Internet of Things) alkalmazások gyakran korlátozott számítási és memória erőforrásokkal rendelkeznek. A kiválasztásos rendezés in-place működése és egyszerűsége miatt ideális választás lehet ezekben a környezetekben, ahol a memóriahasználat minimalizálása és az egyszerű algoritmusok alkalmazása a cél.

##### Példa:

Egy IoT szenzorhálózat, amely különböző érzékelőktől származó adatokat gyűjt és rendez.

```cpp
#include <iostream>
#include <vector>

struct SensorData {
    int sensorId;
    float value;
};

void selectionSort(std::vector<SensorData>& data) {
    int n = data.size();
    for (int i = 0; i < n - 1; i++) {
        int min_index = i;
        for (int j = i + 1; j < n; j++) {
            if (data[j].value < data[min_index].value) {
                min_index = j;
            }
        }
        std::swap(data[i], data[min_index]);
    }
}

int main() {
    std::vector<SensorData> sensorData = {
        {1, 25.4},
        {2, 23.8},
        {3, 26.1},
        {4, 24.3},
        {5, 22.5}
    };

    std::cout << "Unsorted sensor data:" << std::endl;
    for (const auto& data : sensorData) {
        std::cout << "Sensor ID: " << data.sensorId << ", Value: " << data.value << std::endl;
    }

    selectionSort(sensorData);

    std::cout << "\nSorted sensor data:" << std::endl;
    for (const auto& data : sensorData) {
        std::cout << "Sensor ID: " << data.sensorId << ", Value: " << data.value << std::endl;
    }

    return 0;
}
```

Ebben a példaprogramban az IoT szenzorhálózatból származó adatokat rendezik a kiválasztásos rendezés algoritmusával, amely minimalizálja a memóriahasználatot és egyszerűen implementálható.

#### Összegzés

A kiválasztásos rendezés algoritmus számos gyakorlati alkalmazási területen használható egyszerűsége és in-place működése miatt. Bár az időkomplexitása miatt nem mindig a leghatékonyabb választás nagy adathalmazok esetén, számos speciális helyzetben előnyös lehet. Az oktatási céloktól kezdve a kis méretű adathalmazok rendezésén át a beágyazott rendszerek és IoT alkalmazásokig a kiválasztásos rendezés hasznos eszköz lehet a különböző feladatok megoldására. A fenti példák szemléltetik, hogyan alkalmazható az algoritmus különböző gyakorlati helyzetekben, és rávilágítanak az előnyeire és korlátaira.

