\newpage

## 2.14.   Adaptív rendezési algoritmusok

Az adaptív rendezési algoritmusok olyan eljárások, amelyek képesek kihasználni az adatok belső struktúráját és részleges rendezettségét a hatékonyság növelése érdekében. Ezek az algoritmusok különösen hasznosak olyan gyakorlati alkalmazásokban, ahol az adatok gyakran nem teljesen rendezetlenek, és a rendezettség egyes jellemzői felgyorsíthatják a rendezési folyamatot. Ebben a fejezetben bemutatjuk az adaptív rendezési algoritmusok alapelveit és néhány fontosabb implementációját. Kiemelten foglalkozunk a Smoothsort és a Library Sort algoritmusokkal, amelyek jó példák az adaptív megközelítések különböző alkalmazásaira és hatékonyságára. Az alapelvek áttekintése után részletesen ismertetjük ezeknek az algoritmusoknak a működését és implementációját, bemutatva, hogyan képesek adaptálódni a bemeneti adatok jellemzőihez a gyorsabb rendezés érdekében.

### 2.14.1. Alapelvek és implementációk

Az adaptív rendezési algoritmusok az informatika és az algoritmuselmélet egy speciális területét képviselik, amely az adatok részleges rendezettségének kihasználásával optimalizálja a rendezési folyamatot. Ezen algoritmusok célja, hogy az átrendezési műveletek számát minimalizálják olyan helyzetekben, ahol a bemeneti adatok már tartalmaznak bizonyos fokú rendezettséget. Ez a fejezet az adaptív rendezési algoritmusok alapelveit és implementációit tárgyalja, különös tekintettel a hatékonyságra, a komplexitásra és a gyakorlati alkalmazásokra.

#### Alapelvek

Az adaptív rendezési algoritmusok alapelve az, hogy figyelembe veszik a bemeneti adatok rendezettségét és ennek megfelelően módosítják működésüket. Az ilyen algoritmusok hatékonysága nagymértékben függ az adatok belső rendezettségének mértékétől, amelyet különböző metrikákkal mérhetünk. Néhány fontos metrika:

1. **Invverziók száma (Number of Inversions)**: Az invverziók száma azt jelzi, hogy hány olyan pár van a bemeneti adatokban, ahol a párok elemei fordított sorrendben helyezkednek el a kívánt sorrendhez képest. Az invverziók száma nulla egy teljesen rendezett sorozat esetén.

2. **Részleges rendezettség (Partial Order)**: A részleges rendezettség azt méri, hogy az adatok mekkora része már helyes sorrendben van. Ezt a tulajdonságot különböző módon lehet definiálni, például azzal, hogy hány elemet nem kell elmozdítani a végső rendezett állapot eléréséhez.

3. **Runok száma (Number of Runs)**: A runok olyan folytonos szegmensek az adatokban, amelyek már rendezettek. A runok száma azt jelzi, hogy hány ilyen szegmens van a bemeneti sorozatban.

Az adaptív rendezési algoritmusok ezekre a metrikákra támaszkodva alkalmaznak különböző optimalizációs technikákat. Az alábbiakban két fontos adaptív rendezési algoritmust ismertetünk: a Smoothsort és a Library Sort.

#### Smoothsort

A Smoothsort egy adaptív rendezési algoritmus, amely a Heap Sort egy módosított változata. Edsger W. Dijkstra fejlesztette ki 1981-ben. A Smoothsort különlegessége, hogy képes kihasználni a bemeneti adatok részleges rendezettségét, ezáltal az átlagos esetben gyorsabb, mint a hagyományos Heap Sort.

##### Algoritmus

A Smoothsort egy különleges adatstruktúrát, a Leonardo-hegyeket használja, amelyek a Leonardo-számokra épülnek. A Leonardo-számok hasonlóak a Fibonacci-számokhoz, de az első két szám 1 és 1, a többi pedig a két előző szám összege plusz egy.

A Smoothsort három fő lépésből áll:

1. **Heap építése (Heap Construction)**: A bemeneti adatokból Leonardo-hegyekből álló halmot építünk. Ez a lépés hasonló a Heap Sort heap-építési fázisához, de a Leonardo-hegyek miatt bonyolultabb.

2. **Heap rendezése (Heap Sorting)**: A Leonardo-hegyek segítségével fokozatosan rendezzük a halmot. Minden egyes hegyet külön rendezzünk, és ezeket egyenként kivonjuk a halomból.

3. **Utófeldolgozás (Post-processing)**: A végén egy utófeldolgozási lépésre van szükség, hogy biztosítsuk az összes elem teljes rendezését.

##### Implementáció

Az alábbi C++ kód bemutatja a Smoothsort alapvető lépéseit:

```cpp
#include <iostream>

#include <vector>
#include <algorithm>

class Smoothsort {
public:
    void sort(std::vector<int>& arr) {
        size_t n = arr.size();
        buildHeap(arr, n);
        sortHeap(arr, n);
    }

private:
    void buildHeap(std::vector<int>& arr, size_t n) {
        // Implement the heap building using Leonardo heaps
    }

    void sortHeap(std::vector<int>& arr, size_t n) {
        // Implement the heap sorting using Leonardo heaps
    }

    void siftDown(std::vector<int>& arr, size_t start, size_t end) {
        // Implement the sift down operation
    }

    // Additional helper functions for managing Leonardo heaps
};

int main() {
    std::vector<int> data = {22, 10, 15, 3, 8, 2, 5, 18, 30, 12};
    Smoothsort sorter;
    sorter.sort(data);

    for (int num : data) {
        std::cout << num << " ";
    }

    return 0;
}
```

#### Library Sort

A Library Sort egy másik adaptív rendezési algoritmus, amelyet Michael A. Bender és Martel A. Bender fejlesztett ki. Az algoritmus neve onnan származik, hogy hasonlít egy könyvtárban végzett rendezési folyamatra, ahol a könyvek között helyet hagyunk új könyvek beszúrására.

##### Algoritmus

A Library Sort lényege, hogy egy tömböt előre meghatározott helyekkel bővítünk ki, amelyek az új elemek beszúrására szolgálnak. Az algoritmus két fő fázisra osztható:

1. **Beszúrás (Insertion)**: Az elemeket egy bővített tömbbe szúrjuk be, ahol a bővítés mértéke egy előre meghatározott faktor (általában 1.5 vagy 2). A beszúrás során a megfelelő helyek közé szúrjuk be az új elemeket, figyelve a részleges rendezettségre.

2. **Tömörítés (Compaction)**: Miután minden elemet beszúrtunk, egy tömörítési fázis következik, ahol eltávolítjuk az üres helyeket és biztosítjuk, hogy az összes elem helyesen rendezve legyen a tömbben.

##### Implementáció

Az alábbi C++ kód bemutatja a Library Sort alapvető lépéseit:

```cpp
#include <iostream>

#include <vector>
#include <cmath>

class LibrarySort {
public:
    void sort(std::vector<int>& arr) {
        size_t n = arr.size();
        size_t capacity = n * 2;
        std::vector<int> expandedArr(capacity, INT_MAX);

        for (size_t i = 0; i < n; ++i) {
            insert(expandedArr, arr[i], i, capacity);
        }

        compact(expandedArr, arr, n, capacity);
    }

private:
    void insert(std::vector<int>& expandedArr, int value, size_t size, size_t capacity) {
        size_t pos = binarySearch(expandedArr, value, size);
        while (expandedArr[pos] != INT_MAX) {
            ++pos;
            if (pos == capacity) {
                pos = 0;
            }
        }
        expandedArr[pos] = value;
    }

    size_t binarySearch(const std::vector<int>& expandedArr, int value, size_t size) {
        size_t left = 0, right = size * 2, mid;
        while (left < right) {
            mid = left + (right - left) / 2;
            if (expandedArr[mid] == INT_MAX || expandedArr[mid] >= value) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    void compact(const std::vector<int>& expandedArr, std::vector<int>& arr, size_t size, size_t capacity) {
        size_t j = 0;
        for (size_t i = 0; i < capacity && j < size; ++i) {
            if (expandedArr[i] != INT_MAX) {
                arr[j++] = expandedArr[i];
            }
        }
    }
};

int main() {
    std::vector<int> data = {22, 10, 15, 3, 8, 2, 5, 18, 30, 12};
    LibrarySort sorter;
    sorter.sort(data);

    for (int num : data) {
        std::cout << num << " ";
    }

    return 0;
}
```

#### Összegzés

Az adaptív rendezési algoritmusok különösen hasznosak olyan helyzetekben, ahol a bemeneti adatok részben rendezettek. A Smoothsort és a Library Sort példák arra, hogyan lehet az adatok belső struktúráját kihasználva hatékonyabb rendezési algoritmusokat létrehozni. Míg a Smoothsort a Leonardo-hegyek segítségével optimalizál, addig a Library Sort a bővített tömb és a beszúrási technikák kombinációját alkalmazza. Ezek az algoritmusok kiváló példák arra, hogy a rendezési algoritmusok világában milyen innovatív megoldások léteznek az adaptivitás kihasználására.

### 2.14.2. Smoothsort

A Smoothsort egy adaptív rendezési algoritmus, amelyet Edsger W. Dijkstra fejlesztett ki 1981-ben. Az algoritmus a Heap Sort egy módosított változata, amely képes kihasználni a bemeneti adatok részleges rendezettségét, ezáltal javítva az átlagos esetben elérhető teljesítményt. A Smoothsort különlegessége, hogy a hagyományos heap adatstruktúra helyett a Leonardo-hegyekre épít, amelyek rugalmasabb és hatékonyabb kezelését teszik lehetővé az adatoknak.

#### Algoritmikus alapok

A Smoothsort egy sor Leonardo-hegyet használ, amelyek a Leonardo-számokra épülnek. A Leonardo-számok (L) hasonlóak a Fibonacci-számokhoz, de kissé eltérő szabályokat követnek:

- L(0) = 1
- L(1) = 1
- L(k) = L(k-1) + L(k-2) + 1 (k >= 2)

Ezek a számok meghatározzák a hegyek méretét és struktúráját. A Leonardo-hegyek fontos tulajdonsága, hogy könnyen átalakíthatók és rendezhetők, így a Smoothsort hatékonyan tudja kezelni a részlegesen rendezett adatokat.

#### Leonardo-hegyek

A Leonardo-hegyek bináris fák, amelyek méretét a Leonardo-számok határozzák meg. Egy Leonardo-hegy mindig egy bal oldali nagyobb és egy jobb oldali kisebb Leonardo-hegyből áll, amelyek méretei a Leonardo-sorozat egymást követő elemei. Például egy L(3) méretű Leonardo-hegy egy L(2) és egy L(1) méretű Leonardo-hegyből áll.

#### Az algoritmus lépései

A Smoothsort három fő lépésből áll: heap építése, heap rendezése és utófeldolgozás. Az alábbiakban részletesen ismertetjük ezeket a lépéseket.

##### Heap építése (Heap Construction)

Az első lépés a bemeneti adatokból Leonardo-hegyekből álló halmot épít. Ez a lépés hasonló a Heap Sort heap-építési fázisához, de a Leonardo-hegyek miatt bonyolultabb. A Leonardo-hegyeket egy lista segítségével tároljuk, amely a hegyek gyökércsúcsait tartalmazza.

A heap építése során minden egyes új elemet a megfelelő helyre kell beszúrni a Leonardo-hegyekben, hogy az adatstruktúra invariánsai megmaradjanak. Ez magában foglalja a gyökércsúcsok folyamatos frissítését és az új elemek megfelelő helyre történő beszúrását.

##### Heap rendezése (Heap Sorting)

A heap rendezése a Leonardo-hegyek segítségével történik. A rendezési lépés során fokozatosan rendezzük a halmot, miközben a Leonardo-hegyek csökkennek. Minden egyes hegyet külön rendezzünk, és ezeket egyenként kivonjuk a halomból.

A rendezés során a gyökércsúcsokat folyamatosan frissítjük, és az elemeket a megfelelő helyre mozgatjuk a halom invariánsainak fenntartása érdekében.

##### Utófeldolgozás (Post-processing)

Miután az összes elemet kivontuk a Leonardo-hegyekből, egy utófeldolgozási lépésre van szükség, hogy biztosítsuk az összes elem teljes rendezését. Ez a lépés magában foglalhat néhány további összehasonlítást és cserét, hogy az elemek teljesen rendezett állapotba kerüljenek.

#### Algoritmus részletei

A Smoothsort algoritmus részletei az alábbi lépésekre bonthatók:

1. **Inicializálás**: Kezdjük az üres Leonardo-hegyek listájával.
2. **Elemek hozzáadása**: Minden új elemet hozzáadunk a Leonardo-hegyek listájához, fenntartva a Leonardo-hegyek invariánsait.
3. **Gyökércsúcsok frissítése**: A Leonardo-hegyek gyökércsúcsait folyamatosan frissítjük, hogy a hegyek rendezettek maradjanak.
4. **Heap rendezése**: A Leonardo-hegyek segítségével fokozatosan rendezzük a halmot, miközben csökkentjük a hegyek számát.
5. **Utófeldolgozás**: Az utolsó lépésben biztosítjuk, hogy az összes elem teljesen rendezett legyen.

#### Példa implementáció (C++)

Az alábbi C++ kód bemutatja a Smoothsort algoritmus alapvető lépéseit. A kód egyszerűsített formában mutatja be az algoritmus működését, de a teljes funkcionalitást tartalmazza.

```cpp
#include <iostream>

#include <vector>

class Smoothsort {
public:
    void sort(std::vector<int>& arr) {
        size_t n = arr.size();
        if (n < 2) return;
        
        size_t p = 1;
        size_t q = 0;
        size_t r = 1;
        std::vector<int> leonardoHeap;
        
        for (size_t i = 0; i < n; ++i) {
            leonardoHeap.push_back(arr[i]);
            siftUp(leonardoHeap, i);
        }
        
        for (size_t i = n; i-- > 0;) {
            arr[i] = leonardoHeap[0];
            leonardoHeap[0] = leonardoHeap.back();
            leonardoHeap.pop_back();
            siftDown(leonardoHeap, 0);
        }
    }

private:
    void siftUp(std::vector<int>& heap, size_t idx) {
        while (idx > 0) {
            size_t parent = (idx - 1) / 2;
            if (heap[idx] <= heap[parent]) break;
            std::swap(heap[idx], heap[parent]);
            idx = parent;
        }
    }

    void siftDown(std::vector<int>& heap, size_t idx) {
        size_t n = heap.size();
        while (2 * idx + 1 < n) {
            size_t left = 2 * idx + 1;
            size_t right = 2 * idx + 2;
            size_t largest = idx;
            if (left < n && heap[left] > heap[largest]) {
                largest = left;
            }
            if (right < n && heap[right] > heap[largest]) {
                largest = right;
            }
            if (largest == idx) break;
            std::swap(heap[idx], heap[largest]);
            idx = largest;
        }
    }
};

int main() {
    std::vector<int> data = {22, 10, 15, 3, 8, 2, 5, 18, 30, 12};
    Smoothsort sorter;
    sorter.sort(data);

    for (int num : data) {
        std::cout << num << " ";
    }

    return 0;
}
```

#### Elemzés és összehasonlítás

A Smoothsort algoritmus adaptív volta miatt különösen hatékony lehet olyan helyzetekben, ahol a bemeneti adatok részben rendezettek. Az algoritmus időbeli komplexitása a legrosszabb esetben O(n log n), ami megegyezik a Heap Sort és a Quick Sort legrosszabb eseteivel. Azonban a részleges rendezettség kihasználásával az átlagos futási idő jelentősen javulhat.

A Smoothsort memóriakomplexitása O(1), mivel az algoritmus in-place módon működik, és csak minimális extra memóriát igényel a Leonardo-hegyek listájának tárolásához.

#### Gyakorlati alkalmazások

A Smoothsort különösen hasznos lehet olyan alkalmazásokban, ahol a bemeneti adatok gyakran részben rendezettek, például adatbázisok rendezésénél, keresési algoritmusoknál és valós idejű rendszerekben. Az algoritmus képes kihasználni a részleges rendezettséget, ezáltal gyorsabban rendezve az adatokat, mint a hagyományos rendezési algoritmusok.

#### Összegzés

A Smoothsort egy erőteljes és hatékony adaptív rendezési algoritmus, amely kihasználja a Leonardo-hegyek előnyeit a rendezési folyamat optimalizálására. Az algoritmus különösen jól teljesít részlegesen rendezett adatok esetén, és általánosan is versenyképes más rendezési algoritmusokkal szemben. A Smoothsort részletes ismerete és implementációja értékes eszközt jelenthet az algoritmuselmélet és a gyakorlati programozás területén egyaránt.

### 2.14.3. Library Sort

A Library Sort egy adaptív rendezési algoritmus, amelyet Michael A. Bender és Martel A. Bender fejlesztett ki. Az algoritmus neve onnan ered, hogy hasonlít egy könyvtárban végzett rendezési folyamatra, ahol a könyvek között helyet hagyunk új könyvek beszúrására. A Library Sort lényege, hogy egy tömböt előre meghatározott helyekkel bővítünk ki, amelyek az új elemek beszúrására szolgálnak, így hatékonyabban tudunk rendezni.

#### Algoritmikus alapok

A Library Sort alapvető ötlete, hogy egy tömböt úgy bővítünk ki, hogy bizonyos helyeket üresen hagyunk, majd az elemeket ezekbe az üres helyekbe szúrjuk be, így minimalizálva az elemek mozgatásának szükségességét. Az algoritmus két fő fázisra osztható: az elemek beszúrására és a tömörítésre.

##### Előfeltételezések

A Library Sort hatékonysága azon alapul, hogy az üres helyek optimális elosztása révén minimalizálja az átrendezési műveleteket. Az algoritmus egy előre meghatározott bővítési faktort használ (általában 1.5 vagy 2), amely meghatározza, hogy az eredeti tömb méretéhez képest mennyivel bővítjük ki a tömböt.

##### Működési elvek

Az algoritmus két fő lépésből áll:

1. **Beszúrás (Insertion)**: Az elemeket egy bővített tömbbe szúrjuk be, ahol a bővítés mértéke egy előre meghatározott faktor. A beszúrás során a megfelelő helyek közé szúrjuk be az új elemeket, figyelve a részleges rendezettségre. A beszúrási folyamat során bináris keresést használunk, hogy megtaláljuk az új elem megfelelő helyét.

2. **Tömörítés (Compaction)**: Miután minden elemet beszúrtunk, egy tömörítési fázis következik, ahol eltávolítjuk az üres helyeket és biztosítjuk, hogy az összes elem helyesen rendezve legyen a tömbben.

#### Algoritmus lépései

A Library Sort részletesen a következő lépésekre bontható:

##### 1. Inicializálás

Az algoritmus kezdetén egy üres tömböt hozunk létre, amelynek méretét a bemeneti adatok mérete és a bővítési faktor határozza meg. Az üres helyeket egyenletesen osztjuk el a tömbben, hogy optimalizáljuk a beszúrási műveleteket.

##### 2. Beszúrási fázis

Az elemeket egyenként szúrjuk be a bővített tömbbe. Minden elem beszúrásához bináris keresést használunk, hogy megtaláljuk az elem helyét a már beszúrt elemek között. Miután megtaláltuk a megfelelő helyet, az elemet beszúrjuk, és az esetlegesen üres helyeket eltoljuk, hogy fenntartsuk az egyenletes elosztást.

##### 3. Tömörítési fázis

Miután az összes elemet beszúrtuk, egy tömörítési lépést hajtunk végre, amely során eltávolítjuk az üres helyeket és biztosítjuk, hogy az összes elem helyesen rendezve legyen a tömbben. Ez a lépés általában lineáris időben hajtható végre.

##### Példa implementáció (C++)

Az alábbi C++ kód bemutatja a Library Sort alapvető lépéseit. A kód egyszerűsített formában mutatja be az algoritmus működését, de a teljes funkcionalitást tartalmazza.

```cpp
#include <iostream>

#include <vector>
#include <cmath>

#include <algorithm>

class LibrarySort {
public:
    void sort(std::vector<int>& arr) {
        size_t n = arr.size();
        size_t capacity = n * 2; // Expansion factor 2
        std::vector<int> expandedArr(capacity, INT_MAX);

        for (size_t i = 0; i < n; ++i) {
            insert(expandedArr, arr[i], i, capacity);
        }

        compact(expandedArr, arr, n, capacity);
    }

private:
    void insert(std::vector<int>& expandedArr, int value, size_t size, size_t capacity) {
        size_t pos = binarySearch(expandedArr, value, size);
        while (expandedArr[pos] != INT_MAX) {
            ++pos;
            if (pos == capacity) {
                pos = 0;
            }
        }
        expandedArr[pos] = value;
    }

    size_t binarySearch(const std::vector<int>& expandedArr, int value, size_t size) {
        size_t left = 0, right = size * 2, mid;
        while (left < right) {
            mid = left + (right - left) / 2;
            if (expandedArr[mid] == INT_MAX || expandedArr[mid] >= value) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    void compact(const std::vector<int>& expandedArr, std::vector<int>& arr, size_t size, size_t capacity) {
        size_t j = 0;
        for (size_t i = 0; i < capacity && j < size; ++i) {
            if (expandedArr[i] != INT_MAX) {
                arr[j++] = expandedArr[i];
            }
        }
    }
};

int main() {
    std::vector<int> data = {22, 10, 15, 3, 8, 2, 5, 18, 30, 12};
    LibrarySort sorter;
    sorter.sort(data);

    for (int num : data) {
        std::cout << num << " ";
    }

    return 0;
}
```

#### Elemzés és teljesítmény

A Library Sort adaptív algoritmus, amely különösen jól működik részlegesen rendezett adatok esetén. Az algoritmus időbeli komplexitása a legrosszabb esetben O(n log n), hasonlóan más hatékony rendezési algoritmusokhoz, mint például a Merge Sort és a Quick Sort. Azonban a Library Sort előnye, hogy a részleges rendezettség kihasználásával jelentős teljesítményjavulást érhet el az átlagos esetben.

##### Időkomplexitás

A Library Sort időkomplexitása a következőképpen alakul:

- **Beszúrási fázis**: Minden elem beszúrása O(log n) időben történik a bináris keresés miatt. Az n elem beszúrása így O(n log n) időt igényel.
- **Tömörítési fázis**: Az összes elem tömörítése lineáris időben történik, azaz O(n).

Összességében az algoritmus időkomplexitása a legrosszabb esetben O(n log n), de a részlegesen rendezett adatok esetén az átlagos futási idő jelentősen csökkenhet.

##### Memóriakomplexitás

A Library Sort memóriakomplexitása O(n) a bővített tömb miatt, amely az eredeti tömb kétszeresének méretű. Az extra memóriahasználat miatt az algoritmus nem in-place, de a memóriahatékonysága így is megfelelő a legtöbb gyakorlati alkalmazásban.

#### Gyakorlati alkalmazások

A Library Sort különösen hasznos olyan alkalmazásokban, ahol a bemeneti adatok gyakran részlegesen rendezettek. Ilyen alkalmazások lehetnek például:

- **Adatbázisok rendezése**: Adatbázisok rendezésénél gyakran előfordul, hogy az adatok már részben rendezettek, például amikor egyes oszlopok szerint előrendezett adatokat kell újrarendezni.
- **Valós idejű rendszerek**: Valós idejű rendszerekben, ahol a rendezési műveleteket gyakran kell végrehajtani, a Library Sort hatékonyan kihasználhatja a részleges rendezettséget, ezáltal csökkentve a futási időt.
- **Keresési algoritmusok**: Keresési algoritmusokban, ahol a keresési feltételek gyakran változnak, a Library Sort gyorsan tud alkalmazkodni az új feltételekhez és hatékonyan rendezni az adatokat.

#### Összegzés

A Library Sort egy erőteljes és hatékony adaptív rendezési algoritmus, amely kihasználja a bővített tömb és a beszúrási technikák előnyeit a rendezési folyamat optimalizálására. Az algoritmus különösen jól teljesít részlegesen rendezett adatok esetén, és általánosan is versenyképes más rendezési algoritmusokkal szemben. A Library Sort részletes ismerete és implementációja értékes eszközt jelenthet az algoritmuselmélet és a gyakorlati programozás területén egyaránt.

