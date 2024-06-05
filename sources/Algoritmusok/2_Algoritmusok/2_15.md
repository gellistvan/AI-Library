\newpage

## 2.15.   Stabil és instabil rendezési algoritmusok

A rendezési algoritmusok központi szerepet játszanak az adatszerkezetek és algoritmusok tanulmányozásában. Az adatok rendezése nem csupán a hatékonyság szempontjából kritikus, hanem az algoritmusok viselkedése is fontos tényező. Ebben a fejezetben a stabil és instabil rendezési algoritmusokkal foglalkozunk. Megvizsgáljuk, hogy mit jelent a stabilitás a rendezési folyamatban, miért lényeges a gyakorlatban, és hogyan határozza meg az algoritmusok teljesítményét. Bemutatunk néhány stabil rendezési algoritmust, például a stable quicksortot, és kitekintünk arra is, hogyan lehet az instabil algoritmusokat optimalizálni, hogy javítsuk a teljesítményüket és alkalmazkodóképességüket különböző adatszerkezetekhez és feladatokhoz.

### 2.15.1. Stabilitás jelentősége

A rendezési algoritmusok stabilitása alapvető fontosságú tényező az algoritmusok hatékonyságának és alkalmazási területeinek megértésében. A stabilitás egy rendezési algoritmus azon tulajdonsága, hogy két egyenlő értékű elem esetén a rendezés után is megőrzi az eredeti sorrendet. Ez a tulajdonság különösen fontos, ha az elemek több kritérium alapján vannak rendezve, vagy ha az adatok olyan struktúrában vannak, ahol az elemek közötti viszonyokat meg kell őrizni.

#### Stabilitás fontossága

A stabilitás jelentősége számos területen megmutatkozik, különösen akkor, ha az adatok rendezését több lépésben végezzük. Például egy összetett rendezés esetén, ahol először egy másodlagos kulcs szerint rendezünk, majd az eredményt egy elsődleges kulcs szerint, a stabil rendezési algoritmus biztosítja, hogy a másodlagos kulcs szerint azonos elemek közötti sorrend megmarad.

#### Példák a stabilitás fontosságára

Tegyük fel, hogy egy névjegyzéket szeretnénk rendezni először vezetéknév, majd keresztnév szerint. Ha a rendezési algoritmus stabil, akkor azok a bejegyzések, amelyek azonos vezetéknevűek, a keresztnév szerinti sorrendben maradnak, ahogyan azt az első rendezésben meghatároztuk.

#### Stabil és instabil algoritmusok összehasonlítása

A stabilitás szempontjából a rendezési algoritmusokat két fő csoportba sorolhatjuk: stabil és instabil algoritmusok. Az alábbiakban részletezzük néhány ismert rendezési algoritmus stabilitását:

1. **Bubble Sort (buborékrendezés)**: Ez egy stabil rendezési algoritmus. Az egyenlő elemek sorrendje nem változik meg a rendezés során, mivel az algoritmus csak akkor cserél elemeket, ha azok nincsenek megfelelő sorrendben.

2. **Insertion Sort (beszúró rendezés)**: Ez szintén egy stabil rendezési algoritmus. Az elemek beszúrása során az azonos kulcsú elemek eredeti sorrendje megmarad.

3. **Merge Sort (összefésülő rendezés)**: Az összefésülő rendezés is stabil, feltéve hogy az összefésülési lépés során megőrizzük az egyenlő elemek sorrendjét.

4. **Quicksort (gyors rendezés)**: A gyors rendezés hagyományos formája instabil. Azonban léteznek stabil változatai is, amelyek megőrzik az egyenlő elemek sorrendjét.

5. **Heapsort (kupacrendezés)**: A kupacrendezés egy instabil rendezési algoritmus, mivel a heap struktúra építése és rendezése során az egyenlő elemek sorrendje megváltozhat.

#### Stabil rendezési algoritmusok implementálása

Az alábbiakban bemutatjuk a stabil rendezési algoritmusok implementálásának néhány alapvető technikáját, különösen a stable quicksort algoritmusra fókuszálva.

**Stable Quicksort Implementáció C++ nyelven**

A stable quicksort algoritmus egyik módja, hogy az egyenlő elemek esetén nem cserélünk pozíciót. Ez biztosítja a stabilitást, ugyanakkor némileg megnöveli az algoritmus komplexitását.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

template <typename T>
void stableQuickSort(std::vector<T>& arr, int left, int right) {
    if (left < right) {
        int pivotIndex = partition(arr, left, right);
        stableQuickSort(arr, left, pivotIndex - 1);
        stableQuickSort(arr, pivotIndex + 1, right);
    }
}

template <typename T>
int partition(std::vector<T>& arr, int left, int right) {
    T pivot = arr[right];
    int i = left - 1;
    
    for (int j = left; j < right; ++j) {
        if (arr[j] < pivot) {
            std::swap(arr[++i], arr[j]);
        } else if (arr[j] == pivot) {
            if (i < j) {
                arr.insert(arr.begin() + i + 1, arr[j]);
                arr.erase(arr.begin() + j + 1);
                ++i;
            }
        }
    }
    std::swap(arr[++i], arr[right]);
    return i;
}

int main() {
    std::vector<int> data = {4, 2, 4, 3, 1, 2};
    stableQuickSort(data, 0, data.size() - 1);
    
    for (const auto& elem : data) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

#### Instabil rendezési algoritmusok stabilitásának elérése

Bizonyos esetekben az instabil rendezési algoritmusokat is stabilizálhatjuk. Ennek egyik módja, hogy az elemek eredeti pozícióját is figyelembe vesszük a rendezés során. Például az egyenlő elemek esetén az eredeti index alapján döntünk a sorrendről.

#### Stabilitás hatása az algoritmusok teljesítményére

A stabil rendezési algoritmusok alkalmazása nem csupán az adatok rendezésének minőségét javítja, hanem bizonyos esetekben a teljesítményre is pozitív hatással lehet. Az algoritmusok stabilitása különösen fontos a következő esetekben:

1. **Adatbázis rendezés**: Amikor nagy adatbázisokat rendezünk több kulcs alapján, a stabil rendezési algoritmusok biztosítják, hogy az előző rendezési lépések során kialakult sorrend ne vesszen el.

2. **Felhasználói felület rendezés**: Felhasználói felületek rendezésekor, például táblázatokban, a stabil rendezés biztosítja, hogy az egyenlő értékek ne "ugráljanak" a rendezési műveletek során, ami jobb felhasználói élményt eredményez.

3. **Kombinált rendezések**: Összetett algoritmusokban, ahol több lépésben, különböző szempontok alapján történik a rendezés, a stabilitás megőrzése kulcsfontosságú a helyes eredmény eléréséhez.

Összegzésképpen elmondható, hogy a stabilitás egy alapvető jellemzője a rendezési algoritmusoknak, amely jelentős hatással van az algoritmusok alkalmazhatóságára és a rendezés minőségére. A stabil rendezési algoritmusok, mint a stable quicksort vagy a merge sort, számos esetben előnyösek lehetnek, míg az instabil algoritmusokat szükség esetén stabilizálhatjuk, hogy megfeleljenek a speciális igényeknek.


### 2.15.2. Stabil változatok (pl. stable quicksort)

A stabil rendezési algoritmusok kulcsfontosságú szerepet játszanak az adatkezelésben, különösen amikor az adatok többszörös kulcs alapján vannak rendezve, vagy amikor az adatok struktúrája megköveteli az elemek közötti eredeti sorrend megőrzését. Ebben az alfejezetben részletesen megvizsgáljuk a stabil rendezési algoritmusok különböző változatait, kiemelve a stable quicksortot. Részletezzük a stabil rendezési algoritmusok működési elvét, és bemutatjuk a stabil quicksort algoritmus implementációját és működését.

#### Stabil Rendezési Algoritmusok Áttekintése

A stabil rendezési algoritmusok lényege, hogy azonos értékű elemek sorrendjét az eredeti adatszerkezetben megőrzik. Ez a tulajdonság különösen fontos, ha az adatokat több szempont alapján kell rendezni, vagy ha az adatok összetett szerkezetűek, és az elemek közötti eredeti viszonyokat fenn kell tartani.

**Példák Stabil Rendezési Algoritmusokra:**

1. **Bubble Sort (buborékrendezés)**: Ez az egyszerű rendezési algoritmus stabil, mivel az elemek cseréje csak akkor történik, ha azok nincsenek megfelelő sorrendben.

2. **Insertion Sort (beszúró rendezés)**: Az insertion sort szintén stabil, mivel az elemeket úgy szúrja be a megfelelő helyre, hogy az egyenlő elemek sorrendje nem változik meg.

3. **Merge Sort (összefésülő rendezés)**: Az összefésülő rendezés stabil, amennyiben az összefésülés során az egyenlő elemek sorrendjét megőrizzük.

4. **Tim Sort**: A Tim Sort, amely a Python és Java nyelvek alapértelmezett rendezési algoritmusa, szintén stabil. A Tim Sort az insertion sort és a merge sort kombinációjával éri el a stabilitást és a hatékonyságot.

#### Stable Quicksort

A quicksort algoritmus hagyományos formájában instabil, mivel az elemek cseréje során az egyenlő értékű elemek sorrendje megváltozhat. Azonban léteznek stabil változatai is, amelyek biztosítják, hogy az egyenlő elemek sorrendje megmaradjon. Az alábbiakban részletesen bemutatjuk a stable quicksort működését.

**Stable Quicksort Elve:**

A stable quicksort egy olyan változata a quicksort algoritmusnak, amely biztosítja az egyenlő elemek eredeti sorrendjének megőrzését. Ezt úgy érhetjük el, hogy a partícionálás során az egyenlő elemeket nem cseréljük meg, vagy az eredeti indexük alapján döntjük el a sorrendet.

**Stable Quicksort Algoritmus:**

1. **Pivot Kiválasztása**: A stabil quicksortnál is fontos a megfelelő pivot elem kiválasztása. Általában a median-of-three módszert vagy random pivot kiválasztást alkalmazunk a legrosszabb eset elkerülése érdekében.

2. **Partícionálás**: A partícionálás során az elemeket úgy helyezzük át, hogy a pivot elemnél kisebbek kerüljenek balra, a nagyobbak jobbra, és az egyenlő elemek az eredeti sorrendjüket megőrizve maradjanak a megfelelő oldalon.

3. **Rekurzív Rendezés**: Az algoritmus rekurzívan meghívja önmagát a bal és jobb oldali partíciókra, amíg az egész tömb rendezve nem lesz.

**Stable Quicksort Implementáció C++ nyelven:**

Az alábbiakban bemutatunk egy stabil quicksort implementációt C++ nyelven. Ez az implementáció megőrzi az egyenlő elemek sorrendjét az eredeti adatszerkezetben.

```cpp
#include <iostream>
#include <vector>
#include <tuple>

// Helper function to swap two elements
template <typename T>
void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

// Partition function for stable quicksort
template <typename T>
int stablePartition(std::vector<std::pair<T, int>>& arr, int low, int high) {
    T pivot = arr[high].first;
    int i = low - 1;

    for (int j = low; j < high; ++j) {
        if (arr[j].first < pivot || (arr[j].first == pivot && arr[j].second < arr[high].second)) {
            ++i;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

// Stable quicksort function
template <typename T>
void stableQuickSort(std::vector<std::pair<T, int>>& arr, int low, int high) {
    if (low < high) {
        int pi = stablePartition(arr, low, high);
        stableQuickSort(arr, low, pi - 1);
        stableQuickSort(arr, pi + 1, high);
    }
}

// Function to prepare the array for stable quicksort
template <typename T>
void sort(std::vector<T>& arr) {
    std::vector<std::pair<T, int>> indexedArr;
    for (int i = 0; i < arr.size(); ++i) {
        indexedArr.emplace_back(arr[i], i);
    }
    stableQuickSort(indexedArr, 0, indexedArr.size() - 1);
    for (int i = 0; i < arr.size(); ++i) {
        arr[i] = indexedArr[i].first;
    }
}

int main() {
    std::vector<int> data = {4, 2, 4, 3, 1, 2};
    sort(data);
    for (const auto& elem : data) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

#### Egyéb Stabil Rendezési Algoritmusok

A stable quicksort mellett számos más stabil rendezési algoritmus is létezik, amelyeket különböző alkalmazásokban használnak.

1. **Merge Sort**: Az összefésülő rendezés stabil és hatékony algoritmus, amely a divide-and-conquer elvet követi. A merge sort stabilitását az összefésülési lépés során éri el, ahol az egyenlő elemek sorrendjét megőrizzük.

2. **Tim Sort**: A Tim Sort a Python és Java nyelvek alapértelmezett rendezési algoritmusa. Ez az algoritmus az insertion sort és a merge sort kombinációjával éri el a stabilitást és a hatékonyságot. Különösen jól teljesít részben rendezett adatokon.

3. **Insertion Sort**: Az insertion sort egyszerű és stabil rendezési algoritmus, amely kis méretű vagy részben rendezett adathalmazokra ideális.

4. **Bubble Sort**: A bubble sort szintén stabil, bár kevésbé hatékony algoritmus, amelyet oktatási célokra és kis méretű adathalmazokra használnak.

#### Stabilitás Biztosítása

Az instabil rendezési algoritmusok stabilitása is biztosítható különböző technikákkal. Az egyik legelterjedtebb módszer az eredeti indexek használata a rendezés során. Ezt a technikát alkalmaztuk a stable quicksort implementációjában is.

**Általános Technika a Stabilitás Biztosítására:**

1. **Indexelés**: Az elemek eredeti pozíciójának megőrzése érdekében minden elemet párosítsunk az eredeti indexével.
2. **Rendezés**: Az algoritmus során az egyenlő elemek esetén az eredeti index alapján döntünk a sorrendről.
3. **Elemek Visszaállítása**: A rendezés befejezése után az elemeket visszaállítjuk az eredeti struktúrába, az eredeti indexek alapján.

#### Stabil Rendezési Algoritmusok Előnyei

A stabil rendezési algoritmusok számos előnyt kínálnak:

1. **Többszörös Kulcsú Rendezés**: Stabil algoritmusok használata lehetővé teszi az adatok többszörös kulcs alapján történő rendezését anélkül, hogy az előző rendezési szempontok sorrendje elvész.

2. **Felhasználói Élmény**: A stabil rendezés jobb felhasználói élményt nyújt, különösen felhasználói felületeken, ahol a rendezett adatok megjelenítése fontos.

3. **Adatkonzisztencia**: A stabil rendezés biztosítja az adatok konzisztenciáját és megbízhatóságát, különösen összetett adatstruktúrák esetén.

Összegzésképpen elmondható, hogy a stabil rendezési algoritmusok alapvető fontosságúak az adatkezelés különböző területein. A stable quicksort egy hatékony és stabil változata a hagyományos quicksort algoritmusnak, amely megőrzi az egyenlő elemek sorrendjét. A stabil rendezési algoritmusok használata számos előnyt kínál, különösen akkor, amikor az adatok többszörös kulcs alapján történő rendezése, vagy az adatok eredeti sorrendjének megőrzése fontos.

### 2.15.3. Instabil algoritmusok optimalizálása

Az instabil rendezési algoritmusok optimalizálása fontos kutatási és fejlesztési terület az informatikában. Az instabil algoritmusok, mint például a gyors rendezés (quicksort), a kupacrendezés (heapsort) és a kiválasztásos rendezés (selection sort), rendkívül hatékonyak lehetnek, azonban nem garantálják az azonos értékű elemek eredeti sorrendjének megőrzését. Az optimalizálási eljárások célja ezen algoritmusok teljesítményének javítása, stabilitás biztosítása vagy az adott problémához való jobb alkalmazkodás elérése.

#### Instabil Algoritmusok Jellemzői

Az instabil rendezési algoritmusok általában hatékonyabbak, mint stabil társaik, különösen nagy adatbázisok rendezése esetén. Az instabilitás oka, hogy az algoritmusok cserélik az elemeket anélkül, hogy figyelembe vennék azok eredeti sorrendjét. Az alábbiakban részletezünk néhány népszerű instabil algoritmust és azok jellemzőit.

1. **Quicksort (gyors rendezés)**: Az egyik leggyakrabban használt instabil rendezési algoritmus. A quicksort az oszd meg és uralkodj (divide-and-conquer) elvet követi, és a pivot elem köré partícionálja az elemeket. Bár hatékony és átlagosan $O(n \log n)$ futási idejű, a legrosszabb esetben $O(n^2)$ időt vehet igénybe.

2. **Heapsort (kupacrendezés)**: Az algoritmus egy bináris kupacot (heap) épít az adatokból, majd a kupacban lévő legnagyobb vagy legkisebb elemet kivéve rendezi az adatokat. A heapsort $O(n \log n)$ futási idejű, de instabil, mivel a kupac szerkezetében végrehajtott cserék nem őrzik meg az eredeti sorrendet.

3. **Selection Sort (kiválasztásos rendezés)**: Egyszerű, de kevésbé hatékony algoritmus, amely az elemeket iteratívan választja ki a rendezett és rendezetlen részhalmazokból. A selection sort $O(n^2)$ futási idejű és instabil, mivel az elemek cseréje során nem veszi figyelembe azok eredeti sorrendjét.

#### Instabil Algoritmusok Optimalizálási Technikái

Az instabil algoritmusok optimalizálása különböző technikákkal érhető el. Ezek a technikák javíthatják az algoritmusok teljesítményét, stabilitását és alkalmazhatóságát. Az alábbiakban részletesen tárgyaljuk ezeket a technikákat.

**1. Pivot Kiválasztási Módszerek a Quicksortnál**

A quicksort teljesítménye nagymértékben függ a pivot elem kiválasztásának módjától. Az optimális pivot kiválasztás segíthet elkerülni a legrosszabb esetet és javítani az algoritmus átlagos teljesítményét.

- **Median-of-Three**: A median-of-three módszer három elem (például az első, középső és utolsó elem) mediánjának kiválasztásával hatékonyabb pivot kiválasztást biztosít. Ez csökkenti a legrosszabb eset előfordulásának valószínűségét.
- **Random Pivot**: A véletlenszerűen kiválasztott pivot segít elkerülni a legrosszabb esetet, különösen akkor, ha az adatokat már részben rendezették.
- **IntroSort**: Az introspective sort vagy introsort a quicksort és a heapsort kombinációjával működik. Ha a partícionálási mélység meghalad egy bizonyos küszöbértéket, az algoritmus átvált heapsortra, hogy elkerülje a legrosszabb eset futási idejét.

**2. Stabilitás Biztosítása**

Az instabil algoritmusok stabilizálása érdekében különböző technikákat alkalmazhatunk. Az egyik legelterjedtebb módszer az elemek eredeti indexének megőrzése a rendezés során.

**Quicksort Stabilizálása:**

A quicksort stabilizálásának egyik módja, hogy az egyenlő elemek esetén az eredeti indexek alapján döntjük el a sorrendet. Az alábbiakban bemutatunk egy stabil quicksort implementációt C++ nyelven.

```cpp
#include <iostream>
#include <vector>
#include <tuple>

template <typename T>
void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

template <typename T>
int stablePartition(std::vector<std::pair<T, int>>& arr, int low, int high) {
    T pivot = arr[high].first;
    int i = low - 1;

    for (int j = low; j < high; ++j) {
        if (arr[j].first < pivot || (arr[j].first == pivot && arr[j].second < arr[high].second)) {
            ++i;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

template <typename T>
void stableQuickSort(std::vector<std::pair<T, int>>& arr, int low, int high) {
    if (low < high) {
        int pi = stablePartition(arr, low, high);
        stableQuickSort(arr, low, pi - 1);
        stableQuickSort(arr, pi + 1, high);
    }
}

template <typename T>
void sort(std::vector<T>& arr) {
    std::vector<std::pair<T, int>> indexedArr;
    for (int i = 0; i < arr.size(); ++i) {
        indexedArr.emplace_back(arr[i], i);
    }
    stableQuickSort(indexedArr, 0, indexedArr.size() - 1);
    for (int i = 0; i < arr.size(); ++i) {
        arr[i] = indexedArr[i].first;
    }
}

int main() {
    std::vector<int> data = {4, 2, 4, 3, 1, 2};
    sort(data);
    for (const auto& elem : data) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

**3. Kupacrendezés Optimalizálása**

A kupacrendezés optimalizálása többféle módszerrel is elérhető. Ezek közül néhány a következő:

- **Floyd's Heap Construction**: Floyd módszere a kupac gyors felépítésére hatékonyabb, mint a sorozatos beszúrásos módszer. Ez az algoritmus alsó szinttől felfelé építi fel a kupacot, csökkentve a szükséges lépések számát.
- **Optimális Swap Stratégia**: Az elemek cseréjének optimalizálása csökkentheti a kupac építésének idejét és javíthatja az algoritmus teljesítményét.
- **Kúprendezés Variánsai**: A különböző variánsok, mint például a k-heap, javíthatják a kupacrendezés hatékonyságát nagy adatkészleteken.

**4. Kiválasztásos Rendezés Optimalizálása**

A kiválasztásos rendezés alapvetően $O(n^2)$ időbonyolultságú, azonban bizonyos optimalizálási technikákkal javítható.

- **Minimum-Megkeresés Kettős Megközelítése**: A kiválasztásos rendezés optimalizálása érdekében egyszerre keressük a minimum és maximum elemet, csökkentve a szükséges összehasonlítások számát.
- **Többmenetes Kiválasztás**: A rendezési folyamat többmenetes végrehajtása, ahol minden lépésben több elemet választunk ki, hatékonyabbá teheti az algoritmust nagy adatkészleteken.

#### Összegzés

Az instabil rendezési algoritmusok, mint a quicksort, a heapsort és a selection sort, számos előnnyel rendelkeznek, különösen nagy adatkészletek rendezése esetén. Az optimalizálási technikák alkalmazásával ezek az algoritmusok még hatékonyabbá tehetők, és egyes esetekben stabilizálhatók. Az optimális pivot kiválasztás, a stabilizálási technikák, a kupacrendezés optimalizálása és a kiválasztásos rendezés fejlesztése mind hozzájárulnak ahhoz, hogy ezek az algoritmusok széles körben alkalmazhatók és hatékonyak legyenek különböző adatkezelési feladatokban.
