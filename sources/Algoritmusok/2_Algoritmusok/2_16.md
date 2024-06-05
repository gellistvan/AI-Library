\newpage

## 2.16.   Rendezési algoritmusok összehasonlítása

A rendezési algoritmusok a számítástechnika alapvető eszközei közé tartoznak, amelyek nélkülözhetetlenek a hatékony adatkezeléshez. Az adatok rendezése nem csak a keresési műveleteket gyorsítja fel, hanem számos más algoritmus alapját is képezi. Ebben a fejezetben részletesen megvizsgáljuk a rendezési algoritmusok különböző típusait, összehasonlítva azok elméleti és gyakorlati teljesítményét. Kitérünk az egyes algoritmusok előnyeire és hátrányaira különböző környezetekben, valamint bemutatjuk, melyik algoritmus mikor és milyen alkalmazási területen nyújt optimális megoldást. Az olvasó átfogó képet kap a rendezési algoritmusok világáról, segítve ezzel a legmegfelelőbb módszer kiválasztását a konkrét feladatokhoz.

### 2.16.1. Elméleti és gyakorlati teljesítmény összehasonlítás

A rendezési algoritmusok teljesítményének értékelése kritikus fontosságú a hatékony programozás és adatkezelés szempontjából. Az algoritmusok teljesítményét több szempontból is vizsgálhatjuk, beleértve az időbeli és térbeli komplexitást, valamint a gyakorlati implementáció során felmerülő egyéb tényezőket, mint például a stabilitás és az adaptivitás. Ebben az alfejezetben részletesen bemutatjuk a leggyakoribb rendezési algoritmusok elméleti és gyakorlati teljesítményét, összehasonlítva azok erősségeit és gyengeségeit.

#### Időbeli komplexitás

Az időbeli komplexitás az algoritmus futási idejének függvénye az input méretével szemben. Az időbeli komplexitás elemzéséhez gyakran használjuk a Big O notációt, amely az algoritmus legrosszabb esetbeni futási idejét jelzi.

- **Bubble Sort**: Az egyik legegyszerűbb rendezési algoritmus, amelynek időbeli komplexitása $O(n^2)$. Az algoritmus összehasonlítja és cseréli az egymás melletti elemeket, amíg az egész sorozat rendezett nem lesz.

- **Insertion Sort**: Az algoritmus elemenként illeszti be a sorozat elemeit a már rendezett részsorozatba. Az időbeli komplexitása szintén $O(n^2)$ a legrosszabb esetben, de $O(n)$ lehet, ha az input már majdnem rendezett.

- **Selection Sort**: Az algoritmus kiválasztja a legkisebb elemet és a sorozat elejére helyezi, ismételve ezt a folyamatot a maradék sorozattal. Az időbeli komplexitása $O(n^2)$.

- **Merge Sort**: Egy rekurzív algoritmus, amely felosztja a sorozatot kisebb részekre, rendezi azokat, majd összefűzi a rendezett részsorozatokat. Az időbeli komplexitása $O(n \log n)$, ami jelentősen jobb a nagyobb input méretek esetében.

- **Quick Sort**: Egy rekurzív algoritmus, amely kiválaszt egy pivot elemet, és az elemeket úgy rendezi, hogy a pivotnál kisebbek balra, a nagyobbak jobbra kerülnek. Az átlagos időbeli komplexitása $O(n \log n)$, de a legrosszabb esetben $O(n^2)$ lehet, ha a pivot nem megfelelően választott.

- **Heap Sort**: Az algoritmus egy heap adatstruktúrát használ a sorozat rendezéséhez. Az időbeli komplexitása $O(n \log n)$.

#### Térbeli komplexitás

A térbeli komplexitás az algoritmus által igényelt extra memória mennyiségét jelenti az input méretéhez képest.

- **Bubble Sort, Insertion Sort, Selection Sort**: Ezek az algoritmusok in-place rendezési algoritmusok, amelyek térbeli komplexitása $O(1)$, mivel nem igényelnek extra memóriát az inputon kívül.

- **Merge Sort**: Az algoritmus térbeli komplexitása $O(n)$, mivel további memóriát igényel az egyesített részsorozatok tárolásához.

- **Quick Sort**: Az algoritmus átlagos térbeli komplexitása $O(\log n)$, mivel a rekurzív hívásokhoz szükséges memória mérete logaritmikusan nő az input méretével.

- **Heap Sort**: Az algoritmus térbeli komplexitása $O(1)$, mivel in-place rendezést alkalmaz, hasonlóan a Bubble Sorthoz és társaihoz.

#### Stabilitás

Az algoritmus stabilitása azt jelenti, hogy a rendezés során megőrzi-e az azonos értékű elemek eredeti sorrendjét.

- **Bubble Sort, Insertion Sort, Merge Sort**: Ezek stabil rendezési algoritmusok, amelyek megőrzik az azonos értékű elemek eredeti sorrendjét.

- **Selection Sort, Quick Sort, Heap Sort**: Ezek az algoritmusok általában nem stabilak, mivel az elemek cseréje során nem garantált az eredeti sorrend megőrzése.

#### Gyakorlati implementáció és optimalizáció

Az elméleti teljesítmény mellett fontos megvizsgálni az algoritmusok gyakorlati implementációját és optimalizációját is, mivel az implementáció során felmerülő tényezők jelentős hatással lehetnek az algoritmus hatékonyságára.

##### Példakódok C++ nyelven

**Bubble Sort**

```cpp
void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                std::swap(arr[j], arr[j+1]);
            }
        }
    }
}
```

**Insertion Sort**

```cpp
void insertionSort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}
```

**Selection Sort**

```cpp
void selectionSort(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        int min_idx = i;
        for (int j = i+1; j < n; j++) {
            if (arr[j] < arr[min_idx]) {
                min_idx = j;
            }
        }
        std::swap(arr[min_idx], arr[i]);
    }
}
```

**Merge Sort**

```cpp
void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    int L[n1], R[n2];

    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0;
    int k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);

        merge(arr, l, m, r);
    }
}
```

**Quick Sort**

```cpp
int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }
    std::swap(arr[i + 1], arr[high]);
    return (i + 1);
}

void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}
```

**Heap Sort**

```cpp
void heapify(int arr[], int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left] > arr[largest])
        largest = left;

    if (right < n && arr[right] > arr[largest])
        largest = right;

    if (largest != i) {
        std::swap(arr[i], arr[largest]);
        heapify(arr, n, largest);
    }
}

void heapSort(int arr[], int n) {
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    for (int i = n - 1; i > 0; i--) {
        std::swap(arr[0], arr[i]);
        heapify(arr, i, 0);
    }
}
```

#### Gyakorlati szempontok

Az algoritmusok gyakorlati alkalmazása során figyelembe kell venni a konkrét helyzet sajátosságait. Például:

- **Adatmennyiség**: Nagy adatmennyiség esetén a $O(n \log n)$ komplexitású algoritmusok, mint a Merge Sort és a Quick Sort, általában jobbak.
- **Memória korlátok**: Ha a memória korlátozott, az in-place algoritmusok, mint a Quick Sort és a Heap Sort előnyösebbek.
- **Stabilitás szükségessége**: Ha az azonos értékű elemek sorrendjének megőrzése fontos, akkor stabil algoritmusokat, mint a Merge Sort, érdemes választani.
- **Rendezett adatok**: Ha az adatok már részben rendezettek, az adaptív algoritmusok, mint az Insertion Sort, hatékonyabbak lehetnek.

Összefoglalva, a rendezési algoritmusok teljesítményének értékelése során számos tényezőt kell figyelembe venni, és az optimális választás gyakran a konkrét alkalmazási környezettől függ. Az elméleti komplexitás elemzése fontos kiindulópont, de a gyakorlati szempontok, mint a memóriahasználat, a stabilitás és az input adatstruktúra, szintén kritikusak az algoritmus kiválasztásában.

### 2.16.2. Az algoritmusok előnyei és hátrányai különböző scenáriókban

A rendezési algoritmusok kiválasztása és alkalmazása során elengedhetetlen, hogy alaposan megértsük az egyes algoritmusok előnyeit és hátrányait különböző scenáriókban. Ebben az alfejezetben részletesen tárgyaljuk a leggyakoribb rendezési algoritmusok tulajdonságait, és bemutatjuk, hogy mely körülmények között melyik algoritmus lehet a legmegfelelőbb választás.

#### Bubble Sort


**Előnyök:**

1. **Egyszerűség**: Az algoritmus rendkívül egyszerű, könnyen érthető és implementálható.
2. **In-place**: Nem igényel extra memóriát az inputon kívül, így a térbeli komplexitása $O(1)$.
3. **Stabilitás**: Az azonos értékű elemek eredeti sorrendjét megőrzi, ami előnyös lehet bizonyos alkalmazásokban.

**Hátrányok:**

1. **Időbeli komplexitás**: A legrosszabb és az átlagos esetben is $O(n^2)$ a futási ideje, ami nagy inputméretek esetén nagyon lassúvá teszi.
2. **Hatékonyság**: Más, összetettebb algoritmusokhoz képest jelentősen lassabb, különösen nagyobb adatmennyiségek esetén.

#### Insertion Sort


**Előnyök:**

1. **Egyszerűség**: Könnyen érthető és implementálható.
2. **In-place**: Nem igényel extra memóriát, térbeli komplexitása $O(1)$.
3. **Adaptivitás**: Kiválóan teljesít már majdnem rendezett adatok esetén, ahol az időbeli komplexitása $O(n)$.
4. **Stabilitás**: Megőrzi az azonos értékű elemek eredeti sorrendjét.

**Hátrányok:**

1. **Időbeli komplexitás**: A legrosszabb esetben $O(n^2)$, ami nagy inputméretek esetén korlátozza a hatékonyságát.
2. **Nagy inputméretek**: Nagyobb adathalmazok esetén más algoritmusok, mint például a Merge Sort vagy a Quick Sort, hatékonyabbak.

#### Selection Sort


**Előnyök:**

1. **Egyszerűség**: Egyszerűen érthető és implementálható.
2. **In-place**: Az algoritmus nem igényel extra memóriát, térbeli komplexitása $O(1)$.
3. **Minimális csere**: Csak $n-1$ cserét végez, ami előnyös lehet akkor, ha a csere művelet költséges.

**Hátrányok:**

1. **Időbeli komplexitás**: Mind a legrosszabb, mind az átlagos esetben $O(n^2)$, ami nagy inputméretek esetén lassúvá teszi.
2. **Stabilitás hiánya**: Nem stabil, ami azt jelenti, hogy az azonos értékű elemek sorrendje megváltozhat.

#### Merge Sort


**Előnyök:**

1. **Hatékonyság**: A legrosszabb esetben is $O(n \log n)$ a futási ideje, ami nagy inputméretek esetén is hatékony.
2. **Stabilitás**: Megőrzi az azonos értékű elemek eredeti sorrendjét.
3. **Külső rendezés**: Kiválóan alkalmazható nagy adatbázisok rendezésére, ahol az adatokat nem lehet teljesen memóriába tölteni.

**Hátrányok:**

1. **Térbeli komplexitás**: Extra memóriát igényel az adatok tárolásához, térbeli komplexitása $O(n)$.
2. **Implementáció bonyolultsága**: Az algoritmus bonyolultabb az egyszerűbb algoritmusoknál, mint például a Bubble Sort vagy az Insertion Sort.

#### Quick Sort


**Előnyök:**

1. **Hatékonyság**: Átlagosan $O(n \log n)$ a futási ideje, ami nagyon hatékony nagy inputméretek esetén.
2. **In-place**: Nem igényel extra memóriát, térbeli komplexitása $O(\log n)$.
3. **Rugalmas**: Különböző pivot kiválasztási stratégiákkal optimalizálható.

**Hátrányok:**

1. **Legrosszabb eset**: Ha a pivot választás nem optimális, a legrosszabb esetben az időbeli komplexitás $O(n^2)$ lehet.
2. **Stabilitás hiánya**: Az algoritmus nem stabil, ami azt jelenti, hogy az azonos értékű elemek sorrendje megváltozhat.
3. **Bonyolultság**: Az optimális pivot választás és a különböző változatok miatt bonyolultabb lehet az implementációja és a megértése.

#### Heap Sort


**Előnyök:**

1. **Hatékonyság**: Időbeli komplexitása $O(n \log n)$, ami nagy inputméretek esetén is hatékony.
2. **In-place**: Nem igényel extra memóriát, térbeli komplexitása $O(1)$.
3. **Általános használat**: Stabil teljesítményt nyújt különböző adathalmazok esetén is.

**Hátrányok:**

1. **Stabilitás hiánya**: Nem stabil, az azonos értékű elemek sorrendje megváltozhat.
2. **Nem adaptív**: Nem használja ki az esetlegesen már részben rendezett adatokat.

#### Specifikus scenáriók elemzése

##### Nagyméretű adathalmazok rendezése

Nagyméretű adathalmazok esetén az időbeli komplexitás a legfontosabb szempont. Az olyan algoritmusok, mint a Merge Sort és a Quick Sort, amelyek $O(n \log n)$ időbeli komplexitással rendelkeznek, általában előnyösebbek. A Quick Sort esetén fontos a megfelelő pivot választás, hogy elkerüljük a legrosszabb esetet. A Heap Sort szintén hatékony lehet, különösen akkor, ha a memóriahasználat korlátozott.

##### Külső rendezés

Amikor az adatokat nem lehet teljesen memóriába tölteni, mint például nagyon nagy adatbázisok esetén, a Merge Sort a legjobb választás, mivel hatékonyan kezel nagy mennyiségű adatot külső tárolóeszközökről is.

##### Már majdnem rendezett adatok

Ha az adatok már majdnem rendezettek, az Insertion Sort az egyik legjobb választás, mivel ilyen esetekben $O(n)$ időbeli komplexitással rendelkezik. Az algoritmus adaptivitása lehetővé teszi, hogy gyorsabban rendezze az ilyen típusú adatokat, mint a többi algoritmus.

##### Stabilitás megőrzése

Bizonyos alkalmazások esetén, például amikor az adatok több kulcs szerint vannak rendezve, fontos, hogy a rendezési algoritmus stabil legyen. Ilyen esetekben a Merge Sort és az Insertion Sort előnyösebb, mivel megőrzik az azonos értékű elemek eredeti sorrendjét.

##### Memória korlátok

Ha a rendelkezésre álló memória korlátozott, az in-place algoritmusok, mint a Quick Sort és a Heap Sort, jobbak. Ezek az algoritmusok nem igényelnek extra memóriát, így hatékonyabban használják a rendelkezésre álló erőforrásokat.

##### Számítási erőforrások korlátozottsága

Ha a számítási erőforrások korlátozottak, az egyszerűbb algoritmusok, mint a Bubble Sort, Insertion Sort vagy Selection Sort, előnyösebbek lehetnek kisebb adathalmazok esetén, mivel könnyen implementálhatók és kevésbé érzékenyek a hardver specifikációira.

#### Összegzés

A rendezési algoritmusok különböző előnyei és hátrányai számos tényezőtől függnek, beleértve az adathalmaz méretét, a memória korlátokat, az adaptivitást, a stabilitást és a számítási erőforrások rendelkezésre állását. Az algoritmusok megfelelő kiválasztása kritikus fontosságú a hatékony adatkezelés és programozás szempontjából. Az alapos megértésük és a különböző scenáriókban való alkalmazásuk segíti a fejlesztőket abban, hogy optimális megoldásokat találjanak a specifikus feladatokhoz.

### 2.16.3. Rendezési algoritmusok kiválasztása különböző alkalmazási területeken

A rendezési algoritmusok megfelelő kiválasztása különböző alkalmazási területeken kritikus fontosságú a hatékony adatfeldolgozás és programozás szempontjából. Ebben az alfejezetben részletesen megvizsgáljuk, hogy melyik rendezési algoritmus a legmegfelelőbb bizonyos specifikus alkalmazási területeken, figyelembe véve a teljesítményt, stabilitást, adaptivitást, memóriahasználatot és az implementáció bonyolultságát.

#### Nagy adathalmazok rendezése adatbázisokban

Az adatbázisokban gyakran szükséges hatalmas mennyiségű adat rendezése. Ilyen esetekben az időbeli és térbeli hatékonyság a legfontosabb tényezők.

**Merge Sort**:

* **Előnyök**: Stabil és $O(n \log n)$ időbeli komplexitású. Különösen hasznos nagy adathalmazok esetén, amelyek nem férnek el a memóriában, mivel jól működik külső tárolóeszközökkel is.
* **Hátrányok**: Extra memóriát igényel $O(n)$, ami korlátozó tényező lehet memória-intenzív környezetben.

**Heap Sort**:

* **Előnyök**: In-place algoritmus, $O(n \log n)$ időbeli komplexitással. Nem igényel extra memóriát, ami előnyös nagy adathalmazok esetén.
* **Hátrányok**: Nem stabil, ami hátrány lehet, ha a stabilitás kritikus fontosságú.

**Quick Sort**:

* **Előnyök**: Átlagosan $O(n \log n)$ időbeli komplexitású és in-place. Gyorsan működik nagy adathalmazok esetén is, és adaptálható különböző pivot választási módszerekkel.
* **Hátrányok**: A legrosszabb esetben $O(n^2)$ lehet, de ez ritka a megfelelő pivot választással. Nem stabil, de különböző változatok léteznek, amelyek javítják a stabilitást.

#### Valós idejű rendszerek

Valós idejű rendszerekben, ahol a válaszidő kritikus, a rendezési algoritmusok kiválasztásakor az időbeli komplexitás mellett az adaptivitás is fontos szerepet játszik.

**Insertion Sort**:

* **Előnyök**: Rendkívül hatékony kisebb és majdnem rendezett adathalmazok esetén, $O(n)$ időbeli komplexitással ilyen helyzetekben. In-place és stabil.
* **Hátrányok**: Nagy és rendezetlen adathalmazok esetén $O(n^2)$ időbeli komplexitású, ami nem ideális.

**Quick Sort**:

* **Előnyök**: Átlagosan $O(n \log n)$ időbeli komplexitású, és nagyon gyors a gyakorlatban. Különböző pivot választási módszerekkel adaptálható.
* **Hátrányok**: A legrosszabb esetben $O(n^2)$ lehet, de megfelelő pivot választással ez elkerülhető. Nem stabil.

#### Beágyazott rendszerek és korlátozott erőforrások

Beágyazott rendszerekben, ahol a memória és a számítási kapacitás korlátozott, az in-place algoritmusok és az egyszerű implementáció előnyt jelenthet.

**Selection Sort**:

* **Előnyök**: Egyszerűen implementálható és in-place algoritmus, minimális cserét igényel, ami előnyös, ha a csere költséges.
* **Hátrányok**: $O(n^2)$ időbeli komplexitású, ami nagy adathalmazok esetén lassú.

**Insertion Sort**:

* **Előnyök**: Egyszerűen implementálható, in-place és stabil. Hatékony kisebb és majdnem rendezett adathalmazok esetén.
* **Hátrányok**: Nagy és rendezetlen adathalmazok esetén $O(n^2)$ időbeli komplexitású.

#### Grafikai alkalmazások és animációk

Grafikai alkalmazásokban és animációkban a vizuális hatékonyság és a valós idejű teljesítmény fontos. Az egyszerűbb algoritmusok könnyen animálhatók és vizualizálhatók.

**Bubble Sort**:

* **Előnyök**: Nagyon egyszerű és könnyen animálható, mivel az elemek folyamatosan mozognak a helyükre. Stabil és in-place.
* **Hátrányok**: $O(n^2)$ időbeli komplexitású, ami nagy adathalmazok esetén lassú.

**Insertion Sort**:

* **Előnyök**: Egyszerű és könnyen animálható, mivel az elemek folyamatosan illeszkednek a helyükre. Stabil és in-place.
* **Hátrányok**: $O(n^2)$ időbeli komplexitású nagy adathalmazok esetén.

#### Pénzügyi alkalmazások

Pénzügyi alkalmazásokban, ahol az adatokat gyakran kell rendezni és a stabilitás fontos lehet (pl. tranzakciók időrendi sorrendje), a stabil algoritmusok előnyösek.

**Merge Sort**:

* **Előnyök**: Stabil és $O(n \log n)$ időbeli komplexitású, ami nagy adathalmazok esetén is hatékony. Jó választás, ha az adatokat gyakran kell rendezni.
* **Hátrányok**: Extra memóriát igényel $O(n)$, ami korlátozó tényező lehet.

**Insertion Sort**:

* **Előnyök**: Stabil és in-place. Hatékony kisebb és majdnem rendezett adathalmazok esetén.
* **Hátrányok**: Nagy és rendezetlen adathalmazok esetén $O(n^2)$ időbeli komplexitású.

#### Tudományos számítások és szimulációk

Tudományos számítások és szimulációk során gyakran nagy mennyiségű adatot kell rendezni. Ilyen esetekben a hatékony időbeli komplexitás és a párhuzamosíthatóság fontos szempontok.

**Merge Sort**:

* **Előnyök**: Stabil és $O(n \log n)$ időbeli komplexitású. Jól párhuzamosítható, ami előnyös nagy adathalmazok esetén.
* **Hátrányok**: Extra memóriát igényel $O(n)$.

**Quick Sort**:

* **Előnyök**: Átlagosan $O(n \log n)$ időbeli komplexitású, és nagyon gyors a gyakorlatban. Párhuzamosítható, különböző pivot választási módszerekkel optimalizálható.
* **Hátrányok**: A legrosszabb esetben $O(n^2)$ lehet, de megfelelő pivot választással ez elkerülhető. Nem stabil.

##### Példakódok C++ nyelven

**Merge Sort C++ példakód**

```cpp
#include <iostream>
#include <vector>

void merge(std::vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    std::vector<int> L(n1);
    std::vector<int> R(n2);

    for (int i = 0; i < n1; ++i)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; ++j)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            ++i;
        } else {
            arr[k] = R[j];
            ++j;
        }
        ++k;
    }

    while (i < n1) {
        arr[k] = L[i];
        ++i;
        ++k;
    }

    while (j < n2) {
        arr[k] = R[j];
        ++j;
        ++k;
    }
}

void mergeSort(std::vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

int main() {
    std::vector<int> arr = {12, 11, 13, 5, 6, 7};
    int arr_size = arr.size();

    std::cout << "Given array is \n";
    for (int i = 0; i < arr_size; ++i)
        std::cout << arr[i] << " ";
    std::cout << "\n";

    mergeSort(arr, 0, arr_size - 1);

    std::cout << "\nSorted array is \n";
    for (int i = 0; i < arr_size; ++i)
        std::cout << arr[i] << " ";
    std::cout << "\n";
    return 0;
}
```

**Quick Sort C++ példakód**

```cpp
#include <iostream>
#include <vector>

int partition(std::vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; ++j) {
        if (arr[j] < pivot) {
            ++i;
            std::swap(arr[i], arr[j]);
        }
    }
    std::swap(arr[i + 1], arr[high]);
    return (i + 1);
}

void quickSort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int main() {
    std::vector<int> arr = {10, 7, 8, 9, 1, 5};
    int n = arr.size();

    std::cout << "Given array is \n";
    for (int i = 0; i < n; ++i)
        std::cout << arr[i] << " ";
    std::cout << "\n";

    quickSort(arr, 0, n - 1);

    std::cout << "\nSorted array is \n";
    for (int i = 0; i < n; ++i)
        std::cout << arr[i] << " ";
    std::cout << "\n";
    return 0;
}
```

#### Összegzés

A rendezési algoritmusok kiválasztása különböző alkalmazási területeken számos tényezőtől függ, beleértve az adathalmaz méretét, a memóriahasználatot, az adaptivitást, a stabilitást és a specifikus alkalmazási követelményeket. Az egyes algoritmusok előnyeinek és hátrányainak alapos ismerete elengedhetetlen a megfelelő algoritmus kiválasztásához és a hatékony adatfeldolgozáshoz. Az ebben az alfejezetben bemutatott részletes elemzések és példák segítenek a fejlesztőknek és a mérnököknek a legjobb rendezési algoritmus kiválasztásában a különböző alkalmazási területeken.
