\newpage

## 6.2. Merge Sort

A Merge Sort az egyik legismertebb és leggyakrabban alkalmazott oszd-meg-és-uralkodj algoritmus. Ennek az algoritmusnak az alapelve az, hogy egy nagyobb adathalmazt kisebb részekre bont, majd ezeket a részeket rekurzív módon rendezi, és végül összefűzi őket egy rendezett egészbe. Ez a megközelítés stabil és hatékony rendesort eredményez, amely különösen nagy mennyiségű adat kezelésére alkalmas. Ebben a fejezetben részletesen bemutatjuk a Merge Sort algoritmust, annak lépéseit és implementációját, illetve kielemezzük a teljesítményét más rendezési algoritmusokhoz képest. Az algoritmus alapjainak megértése nemcsak a teoretikus tudásodat bővíti, hanem gyakorlati programozási készségekhez is hozzájárul, hiszen a Merge Sort különleges előnyei számos alkalmazási területen hasznosíthatók.

### 6.2.1 Algoritmus és implementáció

#### Bevezetés

A Merge Sort egy hatékony, stabil és összehasonlításos rendezési algoritmus, amely az oszd-meg-és-uralkodj (divide-and-conquer) stratégiát alkalmazza. John von Neumann által 1945-ben kifejlesztett algoritmus erőt ad a ma leggyakrabban használt rendezési technikáknak köszönhetően az alacsony worst-case idő- és helykomplexitása miatt. Ebben a szakaszban az algoritmus részletesen kerül bemutatásra, az implementációval, működésével és a mögötte rejlő elmélettel együtt.

#### Az Algoritmus Lépései

A Merge Sort algoritmus két fő lépésből áll: a listák szétválasztása kisebb részekre, majd az összeolvasztásuk (merge), miközben rendezjük őket. Az algoritmus rekurszív jellege miatt a szétválasztás és az összeolvasztás iteratív folyamata is fontos.

##### 1. Szétválasztás

Az initializáló szakaszban a bemeneti listát két egyenlő részre osztjuk. Ezt a folyamatot addig folytatjuk, amíg az alrészek hossza egy vagy nullává nem válik, mivel egy elemű vagy üres listák nyilvánvalóan rendezetten értelmezhetők.

Például, ha az eredeti listánk `[38, 27, 43, 3, 9, 82, 10]`, akkor először két részre osztjuk:

- Bal lista: `[38, 27, 43]`
- Jobb lista: `[3, 9, 82, 10]`

Ezt a folyamatot rekurszívan folytatjuk:

Bal lista `[38, 27, 43]`:

- Bal lista: `[38]`
- Jobb lista: `[27, 43]`

Jobb lista `[27, 43]`:

- Bal lista: `[27]`
- Jobb lista: `[43]`

Jobb lista `[3, 9, 82, 10]`:

- Bal lista: `[3, 9]`
- Jobb lista: `[82, 10]`

Bal lista `[3, 9]`:

- Bal lista: `[3]`
- Jobb lista: `[9]`

Jobb lista `[82, 10]`:

- Bal lista: `[82]`
- Jobb lista: `[10]`

##### 2. Összeolvasztás

Az összeolvasztási szakaszban két rendezett alrészt kombinálunk egy rendezett listává. Az eljárás során az elemeket az alrészekből összehasonlítjuk és azokat a növekvő sorrendben egy új listába illesztjük.

Az összeolvasztási folyamat egy alacsony szintű, bottom-up megközelítést követ. Vegyük példának a következőt:

- `[38]` és `[27, 43]` rendezett összeolvasztása: `[27, 38, 43]`
    - először: `[27]` és `[38]`, az eredmény: `[27, 38]`
    - majd az: `[27, 38]` és a `[43]`, az eredmény: `[27, 38, 43]`

- `[3]` és `[9]`
- `[82]` és `[10]`: `[10, 82]`

További összeolvasztása a listáknak hasonló módon történik, amíg a teljes lista rendezett listává nem válik.

#### Implementáció

Az alábbiakban bemutatjuk a Merge Sort algoritmus implementációját C++ nyelven:

```cpp
#include <iostream>
#include <vector>

void merge(std::vector<int>& array, int const left, int const mid, int const right) {
    int const subArrayOne = mid - left + 1;
    int const subArrayTwo = right - mid;

    // Create temp arrays
    std::vector<int> leftArray(subArrayOne), rightArray(subArrayTwo);

    // Copy data to temp arrays leftArray[] and rightArray[]
    for (int i = 0; i < subArrayOne; i++)
        leftArray[i] = array[left + i];
    for (int j = 0; j < subArrayTwo; j++)
        rightArray[j] = array[mid + 1 + j];

    int indexOfSubArrayOne = 0, // Initial index of first sub-array
        indexOfSubArrayTwo = 0; // Initial index of second sub-array
    int indexOfMergedArray = left; // Initial index of merged array

    // Merge the temp arrays back into array[]
    while (indexOfSubArrayOne < subArrayOne && indexOfSubArrayTwo < subArrayTwo) {
        if (leftArray[indexOfSubArrayOne] <= rightArray[indexOfSubArrayTwo]) {
            array[indexOfMergedArray] = leftArray[indexOfSubArrayOne];
            indexOfSubArrayOne++;
        } else {
            array[indexOfMergedArray] = rightArray[indexOfSubArrayTwo];
            indexOfSubArrayTwo++;
        }
        indexOfMergedArray++;
    }
    // Copy the remaining elements of left[], if there are any
    while (indexOfSubArrayOne < subArrayOne) {
        array[indexOfMergedArray] = leftArray[indexOfSubArrayOne];
        indexOfSubArrayOne++;
        indexOfMergedArray++;
    }
    // Copy the remaining elements of right[], if there are any
    while (indexOfSubArrayTwo < subArrayTwo) {
        array[indexOfMergedArray] = rightArray[indexOfSubArrayTwo];
        indexOfSubArrayTwo++;
        indexOfMergedArray++;
    }
}

void mergeSort(std::vector<int>& array, int const start, int const end) {
    if (start >= end)
        return; // Returns recursively

    int mid = start + (end - start) / 2;
    mergeSort(array, start, mid);
    mergeSort(array, mid + 1, end);
    merge(array, start, mid, end);
}

void printArray(std::vector<int> const& array) {
    for (auto const& element : array) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::vector<int> array = { 38, 27, 43, 3, 9, 82, 10 };
    std::cout << "Given array is \n";
    printArray(array);

    mergeSort(array, 0, array.size() - 1);

    std::cout << "\nSorted array is \n";
    printArray(array);
    return 0;
}
```

Az algoritmus a `mergeSort` függvényt hívja meg, mely a listát először két részre bontja, majd a `merge` funkció hívásával kombinálja a részeket. A kód egyben egy `printArray` funkciót is tartalmaz, amely segít a lista jelenlegi állapotának megjelenítésében.

#### Algoritmus Működése

A merge sort működése a rekurszió elvét követi, vakon felosztva a listát kisebb és kisebb részekre, mígnem a legkisebb részek elérik az egyjegyű hosszúságot. Minden egyes szakasz visszafelé történő olvasztási folyamatán keresztül egy rendezett lista jön létre.

A fenti C++ implementációban, az algoritmus alapvető működésének megértése céljából különös figyelmet érdemel a listák kombinálásának folyamata. Ez biztosítja, hogy a rendezési részfolyamatok helyesek és hogy a körülbelül O(n*log(n)) időkomplexitás biztosított.

#### Végeredmény

A Merge Sort algoritmus erőssége a nagy hatékonysága lehetővé teszi, hogy sok különböző típusú adatsoron (például nagy, részlegesen rendezett, szinte teljesen rendezett stb.) jól működjön. A stabilitása garantálja, hogy az egyenlő elemek eredeti sorrendje megmarad, ami különösen fontos bizonyos alkalmazásokban (például adatbázisműveleteknél).

Ezen túlmenően, a Merge Sort algoritmus további előnnyel bír paralelizáció szempontjából. A szétválasztási lépések egyidőben történhetnek, további optimalizációkat és a gyorsaság növelését eredményezve nagy számú processzorra. Ez különösen fontos szempont a modern terjesztett rendszerekben és cloud computing környezetben.

### 6.2.2 Teljesítményelemzés és összehasonlítás

A Merge Sort az egyik legismertebb és leggyakrabban tanított oszd-meg-és-uralkodj algoritmus, amely egy hatékony és stabil rendezési megoldást kínál. Ebben a fejezetben részletesen tárgyaljuk a Merge Sort algoritmus teljesítményét és összehasonlítjuk más ismert rendezési algoritmusokkal, figyelembe véve a futási idő komplexitását, a tárhelyigényt és egyéb releváns szempontokat.

#### 6.2.2.1 Futási idő komplexitása

A Merge Sort algoritmus futási ideje `O(n log n)`, ami az összes általános célú rendezési algoritmus között az egyik legjobb. Az oszd-meg-és-uralkodj stratégia újjadefiniál ergonómikus tömbműveletekkel tartja kézben az adatok rendezését. A továbbiakban részletesen is megnézzük, miért kapunk ezt az eredményt.

##### Rekurzív bontás és fúzió

A Merge Sort esetében a bemeneti tömböt mindaddig felezzük, amíg egyedi elemekhez nem érünk:

- Az osztási lépésben a tömböt két egyenlő részre bonjuk. Ez `O(log n)` lépés, mert minden felezés elosztja a tömböt két kisebb részre.
- Az egyesített rész összeállítása `O(n)` művelet, mert minden egyes lépésben át kell menni a tömb mindegyik elemén, hogy az összeolvadás művelete végbemenjen.

Ez az elméleti háttér biztosít minket arról, hogy a művelet futási ideje valóban `O(n log n)` lesz.

##### Tétel bizonyítása

Formalizáljuk a rekurrencia-elemzés segítségével a Merge Sort futási idejét. Az algoritmust a következő rekurraciával írhatjuk le:

$$
T(n) = 2T\left(\frac{n}{2}\right) + O(n)
$$

Itt a `T(n)` az `n` elem rendezésének költsége. Az egyenlet szerint két részproblémát oldunk meg, mindegyiket `n/2` méretű részhalmazzal, és összeolvasztjuk őket, ami `O(n)` költséggel jár.

A Master-tétel alapján:

$$
T(n) = a T\left(\frac{n}{b}\right) + f(n),
$$

ahol `a = 2`, `b = 2`, és `f(n) = O(n)`. A Master-tétel három esetének egyike itt alkalmazható:

- Ha `f(n) = O(n^c)` és `c < \log_b a`, akkor `T(n) = O(n^{\log_b a})`.
- Ha `f(n) = O(n^{\log_b a})`, akkor `T(n) = O(n^{\log_b a} \log n)`.
- Ha `f(n) = \Omega(n^c)` és `c > \log_b a`, akkor `T(n) = O(f(n))`.

Ebben az esetben:

$$
\log_b a = \log_2 2 = 1
$$

és `f(n) = O(n)` pontosan `n^{\log_b a}` alakú, így a futási idő:

$$
T(n) = O(n \log n).
$$

#### 6.2.2.2 Tárhelyigény

Merge Sort egyik figyelemre méltó jellemzője az extra tárterület iránti igénye. Ez azért van, mert az összeolvasztási lépéshez egy további tömböt használunk az adatok átmeneti tárolásához. Ez a tömb ugyanolyan nagyságú, mint az eredeti tömb, azaz a tárhelyigény `O(n)`.

Ez a szempont jelentős hátrány lehet, különösen nagy adatkészletek esetében. Összehasonlításként az in-place algoritmusok (mint például a Quicksort) nagyobb előnyt nyújtanak, mivel nem követelnek ilyen jelentős extra tárterületet. Az in-place algoritmusok esetében a tárhelyigény általában `O(log n)`, ami főként a rekurzív hívások veremének használatából adódik.

Az egymástól különböző rendezési algoritmusokat a következőképpen hasonlíthatjuk össze tárhelyigény szempontjából:

|Algorithm   | Time Complexity | Space Complexity|
|------------|------------------|------------------|
| Merge Sort | O(n log n)       | O(n)             |
| Quick Sort | O(n log n)       | O(log n)         |
| Heap Sort  | O(n log n)       | O(1)             |
| Bubble Sort| O(n^2)           | O(1)             |
| Selection Sort | O(n^2)       | O(1)             |

#### 6.2.2.3 Stabilitás és adaptivitás

A Merge Sort algoritmus stabil, ami azt jelenti, hogy a hasonló rendezetlen elem sorrendje nem változik az algoritmus futása során. Ez kritikus lehet, ha az adatok elsődleges, valamint másodlagos rendezési feltételekkel rendelkeznek.

Az adaptivitás tekintetében azonban a Merge Sort nem különösebben hatékony. Míg például az Insertion Sort `O(n)` futási idővel rendelkezik egy majdnem rendezett tömb esetén, addig a Merge Sort ugyanúgy `O(n log n)` futási idővel dolgozik minden bemenetre vonatkozóan. Ez azt jelenti, hogy Merge Sort nem használható adaptív algoritmusként a márna (majdnem rendezett) adatok esetében.

#### 6.2.2.4 Összehasonlítás más algoritmusokkal

**Quick Sort:**
Quicksort, ha jól van implementálva, általában gyorsabb, mint Merge Sort gyakorlatban. Ez részben az in-place jellegének köszönhető, mint ahogy az is, hogy a Quicksort a bemeneti struktúra sajátosságaiból is profitálhat. A Merge Sort viszont előre meghatározott futási időt biztosít.

**Tárkompleksitás:**
Merge Sort `O(n)` extra tárterületet igényel, szemben a Quicksort `O(log n)` helyigényével. Így nagyobb adatoknál a Quicksort kompaktabb, még akkor is, ha nem stabil.

**Stabilitás:**
Míg a Merge Sort stabil, a Quicksort eredetileg nem. Ez különösen hatásos, ha az algoritmus másodlagos rendezési kritériumokat is figyelembe kell venni.

**Heap Sort:**
Heapsort szintén `O(n log n)` futási idővel rendelkezik, jobb esetben azonban in-place algoritmus. Azonban, eltérően a Merge Sort-tól, Heapsort nem stabil, így nem alkalmas esetenként a second-key vagy multi-key rendezési feltételekhez.

**Belső és Külső rendezés:**
Amikor belső rendezést kell végezni, ahol a teljes adatkészlet memóriában van, az in-place algoritmusok (például a Quicksort) jobban teljesítenek. Azonban, külső rendezés esetén, ahol az adatok merevlemezről jönnek, a Merge Sort előnye, hogy ad hoc bájtműveletekkel rendelkezik, és rugalmasan kezel nagy mennyiségű adatot.

#### 6.2.2.5 Merge Sort Implementáció

Következzen egy C++ nyelven készült Merge Sort implementáció:

```cpp
#include <iostream>
#include <vector>

void merge(std::vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    std::vector<int> L(n1);
    std::vector<int> R(n2);

    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;
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

void mergeSort(std::vector<int>& arr, int l, int r) {
    if (l >= r) return;

    int m = l + (r - l) / 2;
    mergeSort(arr, l, m);
    mergeSort(arr, m + 1, r);
    merge(arr, l, m, r);
}

int main() {
    std::vector<int> arr = {12, 11, 13, 5, 6, 7};
    int arr_size = arr.size();
    
    std::cout << "Given array is \n";
    for (auto val : arr) std::cout << val << " ";
    std::cout << std::endl;
    
    mergeSort(arr, 0, arr_size - 1);
    
    std::cout << "Sorted array is \n";
    for (auto val : arr) std::cout << val << " ";
    std::cout << std::endl;
    
    return 0;
}
```
Ez az implementáció klasszikus példája a Merge Sort algoritmusnak, amely szemlélteti a stabilitást és a `O(n log n)` időbeli komplexitást. Az extra tárolóterületet felhasználva a vegyítési lépés végrehajtása pontos, átlátható eredményeket biztosít.
