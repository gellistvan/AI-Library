\newpage

## 2.12.   Tim Sort

A Tim Sort egy adaptív, stabil rendezési algoritmus, amely az insertion sort és a merge sort erősségeit egyesíti. Ezt az algoritmust Tim Peters fejlesztette ki 2002-ben, kifejezetten a Python programozási nyelv hatékony rendezési módszereként. Azóta számos modern programozási nyelvben és könyvtárban alkalmazzák. A Tim Sort különösen hatékony vegyes adatkészleteknél és valódi alkalmazásoknál, ahol a bemeneti adatok részben rendezettek lehetnek. Ez a fejezet részletesen bemutatja a Tim Sort alapelveit és implementációját, megvizsgálja, hogyan kombinálja az insertion sort és merge sort algoritmusokat, valamint bemutatja alkalmazását különböző modern programozási nyelvekben, mint például a Python, Java és C++.

### 2.12.1. Alapelvek és implementáció

A Tim Sort egy adaptív, stabil rendezési algoritmus, amelyet Tim Peters fejlesztett ki 2002-ben, és először a Python programozási nyelvben használták. Az algoritmus az insertion sort és a merge sort kombinációja, amelyeket stratégiailag használ a hatékonyság növelésére, különösen akkor, ha az adatok részben már rendezettek. Ebben a szakaszban részletesen bemutatjuk a Tim Sort alapelveit, működését és implementációját.

#### Alapelvek

A Tim Sort alapja az a megfigyelés, hogy a valódi alkalmazásokban az adatok gyakran részben rendezettek. Ezt az információt kihasználva a Tim Sort adaptívan kezeli a különböző rendezési feladatokat. Az algoritmus két fő részből áll: a futások (runs) azonosításából és a futások összefésüléséből (merging).

1. **Futások (Runs) azonosítása**:
   A Tim Sort az adatsorozatot kisebb, részben rendezett szegmensekre bontja, amelyeket futásoknak nevezünk. Ezek a futások lehetnek növekvők vagy csökkenők. A csökkenő futásokat az algoritmus először megfordítja, hogy növekvő futásokat kapjon. Ezután a futások hosszát egy minimális értékre (minrun) bővíti, amelyet az adatméret függvényében határoz meg.

2. **Insertion Sort alkalmazása**:
   A futások létrehozásakor a Tim Sort alkalmazza az insertion sortot a kisebb futások rendezésére. Az insertion sort hatékony a kisebb adathalmazok rendezésére, különösen akkor, ha azok részben már rendezettek. Ez növeli az algoritmus adaptív jellegét és csökkenti az időkomplexitást.

3. **Futások összefésülése (Merging)**:
   Miután az összes futás létrejött és rendezve van, a Tim Sort a merge sortot alkalmazza a futások összefésülésére. A merge sort hatékonyan kezeli a nagyobb adathalmazokat és garantálja a stabilitást, vagyis az egyenlő értékek sorrendje megmarad a rendezés után is.

#### Implementáció

Az implementáció során a Tim Sort különböző paramétereket használ, például a minrun értéket, amely meghatározza a futások minimális hosszát. Ez a következőképpen számítható ki:

```cpp
int minRunLength(int n) {
    int r = 0; 
    while (n >= 64) {
        r |= n & 1;
        n >>= 1;
    }
    return n + r;
}
```

Ez a függvény a bemenet (n) hosszától függően határozza meg a minrun értékét. Ha az adathalmaz hossza nagyobb vagy egyenlő 64-gyel, akkor a legkisebb bináris számjegyet hozzáadja a maradékhoz (r), majd jobbra tolja a bemenetet egy bittel. A végső érték a minrun lesz.

A futások azonosítása és rendezése insertion sorttal:

```cpp
void insertionSort(vector<int>& arr, int left, int right) {
    for (int i = left + 1; i <= right; i++) {
        int temp = arr[i];
        int j = i - 1;
        while (j >= left && arr[j] > temp) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = temp;
    }
}

vector<vector<int>> identifyRuns(vector<int>& arr) {
    vector<vector<int>> runs;
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        int run_start = i;
        while (i < n - 1 && arr[i] <= arr[i + 1]) {
            i++;
        }
        runs.push_back(vector<int>(arr.begin() + run_start, arr.begin() + i + 1));
    }
    return runs;
}
```

A futások összefésülése merge sorttal:

```cpp
void merge(vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;
    
    vector<int> L(n1), R(n2);
    
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

void timSort(vector<int>& arr) {
    int n = arr.size();
    int minRun = minRunLength(n);
    
    for (int i = 0; i < n; i += minRun) {
        insertionSort(arr, i, min((i + minRun - 1), (n - 1)));
    }
    
    for (int size = minRun; size < n; size = 2 * size) {
        for (int left = 0; left < n; left += 2 * size) {
            int mid = left + size - 1;
            int right = min((left + 2 * size - 1), (n - 1));
            if (mid < right)
                merge(arr, left, mid, right);
        }
    }
}
```

#### Az algoritmus hatékonysága

A Tim Sort időkomplexitása legrosszabb esetben O(n log n), ami megegyezik a merge sort időkomplexitásával. Azonban a legjobb esetben, amikor az adatok már rendezettek vagy részben rendezettek, az időkomplexitás közelít az O(n) értékhez, ami az insertion sort hatékonyságával egyenlő. Ez az adaptív jelleg a Tim Sortot rendkívül hatékonnyá teszi valódi alkalmazásokban, ahol az adatok gyakran nem teljesen rendezetlenek.

Az algoritmus stabilitása és a részben rendezett adatokra való adaptivitása teszi a Tim Sortot a Python és más modern programozási nyelvek alapértelmezett rendezési algoritmusává. A Tim Sort alkalmazása jelentős teljesítményjavulást eredményezhet, különösen nagy adathalmazok esetén, amelyeket gyakran találunk valódi alkalmazásokban.

### 2.12.2. Keverék az insertion sort és merge sort között

A Tim Sort különlegessége abban rejlik, hogy két jól ismert rendezési algoritmus, az insertion sort és a merge sort kombinációját alkalmazza. Ez a kombináció lehetővé teszi az algoritmus számára, hogy kihasználja mindkét módszer előnyeit, miközben minimalizálja azok hátrányait. A következőkben részletesen megvizsgáljuk, hogyan működik ez a keverék, és miért válik a Tim Sort olyan hatékonnyá és adaptívvá.

#### Insertion Sort

Az insertion sort egy egyszerű, de hatékony rendezési algoritmus kisebb és részben rendezett adathalmazok esetén. Az alapötlet az, hogy az adatsorozat elemeit egyesével átvizsgáljuk, és mindegyiket a megfelelő helyre illesztjük be a már rendezett részbe. Az insertion sort időkomplexitása legrosszabb esetben O(n^2), de részben rendezett adatok esetén közelíthet az O(n) értékhez. Ez az adaptivitás teszi az insertion sortot ideálissá a Tim Sort futásainak rendezéséhez.

Az insertion sort alapvető lépései a következők:

1. **Kezdeti állapot**: Kezdjük az első elemmel, amely önmagában már rendezett.
2. **Beszúrás**: Vegyük a következő elemet, és hasonlítsuk össze a rendezett rész elemeivel.
3. **Mozgatás**: Mozgassuk a nagyobb elemeket jobbra, hogy helyet biztosítsunk az új elemnek.
4. **Illesztés**: Illesszük be az új elemet a megfelelő helyre.
5. **Ismétlés**: Folytassuk a folyamatot a sorozat végéig.

Az insertion sort előnyei közé tartozik az egyszerűség, a kevés memóriahasználat és a hatékonyság kisebb adathalmazok esetén.

#### Merge Sort

A merge sort egy oszd meg és uralkodj (divide and conquer) típusú algoritmus, amely hatékonyan rendezi az adathalmazokat két részre bontással és azok egyesítésével. A merge sort időkomplexitása mindig O(n log n), ami garantálja a stabilitást és hatékonyságot nagyobb adathalmazok esetén is.

A merge sort alapvető lépései a következők:

1. **Felbontás**: Oszd az adathalmazt két egyenlő részre.
2. **Rendezés**: Rendezze mindkét részt rekurzívan.
3. **Összefésülés**: Fésüld össze a két rendezett részt egy rendezett sorozattá.

A merge sort stabil, ami azt jelenti, hogy az egyenlő értékek sorrendje megmarad a rendezés után is. Ez különösen fontos olyan alkalmazásokban, ahol a rendezési stabilitás követelmény.

#### Tim Sort Keverék

A Tim Sort ezen két algoritmus erősségeit egyesíti a következő módon:

1. **Futások Azonosítása és Bővítése**:
   A Tim Sort az adathalmazban található részben rendezett szegmenseket (futásokat) azonosítja. Ezek lehetnek növekvők vagy csökkenők, utóbbiakat megfordítja, hogy növekvő futásokat kapjon. Az azonosított futások hosszát egy minimális értékre (minrun) bővíti, amely az adathalmaz méretétől függ. Ez a folyamat az insertion sort hatékonyságát és adaptivitását használja ki.

2. **Futások Rendezése Insertion Sorttal**:
   Miután az összes futás azonosítva van, azokat az insertion sort segítségével rendezi. Ez a lépés garantálja, hogy minden futás teljesen rendezett legyen, mielőtt a merge sortot alkalmaznánk.

3. **Futások Összefésülése Merge Sorttal**:
   Az utolsó lépés a rendezett futások összefésülése merge sorttal. Ez a lépés biztosítja a végső rendezett adathalmazt, kihasználva a merge sort stabilitását és hatékonyságát.

A Tim Sort keverékének előnyei közé tartozik:

1. **Adaptivitás**: Az insertion sort részben rendezett adatok esetén közel O(n) időkomplexitású, ami jelentős teljesítménynövekedést eredményezhet.
2. **Stabilitás**: A merge sort stabilitása biztosítja, hogy az egyenlő értékek sorrendje megmaradjon, ami fontos bizonyos alkalmazásoknál.
3. **Hatékonyság**: Az algoritmus O(n log n) időkomplexitása nagy adathalmazok esetén is garantálja a hatékonyságot.

#### Alkalmazás

A Tim Sortot széles körben használják modern programozási nyelvek és könyvtárak alapértelmezett rendezési algoritmusaként. Például a Python beépített `sort()` függvénye és az `Arrays.sort()` függvény Java-ban is Tim Sort alapú. Ez a széleskörű alkalmazás a Tim Sort kiemelkedő teljesítményét és megbízhatóságát tükrözi.

#### Implementáció Részletek

Az implementáció során fontos figyelembe venni a futások hosszát és az optimális minrun értéket. Az optimális minrun érték kiválasztása biztosítja, hogy a futások elegendően nagyok legyenek a merge sort hatékonyságának kihasználásához, de elég kicsik ahhoz, hogy az insertion sort adaptivitását is ki lehessen használni.

Az alábbi C++ kód demonstrálja a Tim Sort működését:

```cpp
#include <vector>

#include <iostream>
#include <algorithm>

int minRunLength(int n) {
    int r = 0;
    while (n >= 64) {
        r |= n & 1;
        n >>= 1;
    }
    return n + r;
}

void insertionSort(std::vector<int>& arr, int left, int right) {
    for (int i = left + 1; i <= right; i++) {
        int temp = arr[i];
        int j = i - 1;
        while (j >= left && arr[j] > temp) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = temp;
    }
}

void merge(std::vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    std::vector<int> L(n1), R(n2);

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

void timSort(std::vector<int>& arr) {
    int n = arr.size();
    int minRun = minRunLength(n);

    for (int i = 0; i < n; i += minRun) {
        insertionSort(arr, i, std::min((i + minRun - 1), (n - 1)));
    }

    for (int size = minRun; size < n; size = 2 * size) {
        for (int left = 0; left < n; left += 2 * size) {
            int mid = left + size - 1;
            int right = std::min((left + 2 * size - 1), (n - 1));
            if (mid < right)
                merge(arr, left, mid, right);
        }
    }
}
```

Ez a C++ kód részletesen bemutatja a Tim Sort működését, az insertion sort alkalmazását a futások rendezésére, valamint a merge sort használatát a futások összefésülésére. Az algoritmus hatékonyságának és stabilitásának köszönhetően a Tim Sort széles körben alkalmazható különböző programozási nyelvekben és alkalmazásokban, így biztosítva a gyors és megbízható rendezést.


### 2.12.3. Alkalmazások modern programozási nyelvekben (pl. Python, Java, C++)

A Tim Sort egy rendkívül hatékony és adaptív rendezési algoritmus, amelyet számos modern programozási nyelv és könyvtár használ alapértelmezett rendezési módszerként. Ennek az algoritmusnak az előnyei közé tartozik a stabilitás, a hatékonyság, különösen részben rendezett adathalmazok esetén, valamint a rugalmasság, amely lehetővé teszi különböző adattípusok és méretek kezelését. Ebben a fejezetben részletesen megvizsgáljuk a Tim Sort alkalmazását és implementációját különböző modern programozási nyelvekben, beleértve a Python, Java és C++ nyelveket.

#### Python

A Python programozási nyelvben a Tim Sort az alapértelmezett rendezési algoritmus, amelyet a beépített `sort()` metódus és az `sorted()` függvény is használ. A Python Tim Sort implementációját Tim Peters fejlesztette ki, és 2002 óta része a Python standard könyvtárának.

##### Alkalmazás

A Pythonban a Tim Sort alkalmazása rendkívül egyszerű. A beépített rendezési függvények automatikusan használják ezt az algoritmust, így a felhasználónak nem kell külön konfigurálnia vagy implementálnia. Például:

```python
# List sorting in Python using Tim Sort

arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
arr.sort()  # This uses Tim Sort internally
print(arr)
```

Az alapértelmezett rendezési funkciók stabilak, és optimálisak különböző adathalmazok esetén is. A Python Tim Sort implementációja kihasználja az algoritmus adaptív jellegét, így gyorsan és hatékonyan kezeli a részben rendezett adatsorokat is.

##### Részletek

A Python Tim Sort implementációjának főbb lépései közé tartozik a futások azonosítása, az insertion sort alkalmazása a futások rendezésére, és a merge sort használata a futások összefésülésére. Az algoritmus a `listobject.c` fájlban található, amely a Python forráskód részét képezi.

#### Java

A Java programozási nyelvben a Tim Sort szintén alapértelmezett rendezési algoritmus, amelyet az `Arrays.sort()` és a `Collections.sort()` metódusok is használnak. A Java Tim Sort implementációját Joshua Bloch és Guy Steele fejlesztette ki, és a JDK 1.7 verziójától kezdve használatos.

##### Alkalmazás

A Java-ban a Tim Sort használata szintén rendkívül egyszerű, mivel a beépített rendezési metódusok automatikusan alkalmazzák az algoritmust:

```java
import java.util.Arrays;

public class TimSortExample {
    public static void main(String[] args) {
        int[] arr = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
        Arrays.sort(arr);  // This uses Tim Sort internally
        System.out.println(Arrays.toString(arr));
    }
}
```

A Java implementáció stabil és hatékony, nagy hangsúlyt fektetve a részben rendezett adathalmazok gyors kezelésére.

##### Részletek

A Java Tim Sort implementációja az `java.util.Arrays` osztályban található, és a `Dual-Pivot Quicksort` helyett használatos, amelyet korábban az alapértelmezett rendezési algoritmusként használtak. Az implementáció részletei a `TimSort.java` fájlban találhatók, amely a JDK forráskód része.

#### C++

A C++ programozási nyelvben a Tim Sort nem része az alapértelmezett standard könyvtárnak, de számos könyvtár és implementáció elérhető a felhasználók számára. A C++-ban a felhasználók saját maguk is implementálhatják a Tim Sortot, vagy használhatják a különböző nyílt forráskódú könyvtárakban elérhető megoldásokat.

##### Implementáció

A C++-ban a Tim Sort implementálása hasonló lépéseket követ, mint más nyelvekben, beleértve a futások azonosítását, az insertion sort alkalmazását a futások rendezésére, és a merge sort használatát a futások összefésülésére.

Az alábbiakban egy példakódot találunk a Tim Sort C++-ban történő implementálására:

```cpp
#include <vector>

#include <iostream>
#include <algorithm>

int minRunLength(int n) {
    int r = 0;
    while (n >= 64) {
        r |= n & 1;
        n >>= 1;
    }
    return n + r;
}

void insertionSort(std::vector<int>& arr, int left, int right) {
    for (int i = left + 1; i <= right; i++) {
        int temp = arr[i];
        int j = i - 1;
        while (j >= left && arr[j] > temp) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = temp;
    }
}

void merge(std::vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    std::vector<int> L(n1), R(n2);

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

void timSort(std::vector<int>& arr) {
    int n = arr.size();
    int minRun = minRunLength(n);

    for (int i = 0; i < n; i += minRun) {
        insertionSort(arr, i, std::min((i + minRun - 1), (n - 1)));
    }

    for (int size = minRun; size < n; size = 2 * size) {
        for (int left = 0; left < n; left += 2 * size) {
            int mid = left + size - 1;
            int right = std::min((left + 2 * size - 1), (n - 1));
            if (mid < right)
                merge(arr, left, mid, right);
        }
    }
}

int main() {
    std::vector<int> arr = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    timSort(arr);
    for (int i : arr) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

##### Részletek

A fenti kód részletesen bemutatja a Tim Sort implementálását C++ nyelven. Az `insertionSort` függvény a kisebb futások rendezésére szolgál, míg a `merge` függvény a futások összefésülésére használatos. A `timSort` függvény az egész rendezési folyamatot vezérli, beleértve a futások azonosítását, rendezését és összefésülését.

#### Összegzés

A Tim Sort egy rendkívül hatékony és adaptív rendezési algoritmus, amelyet széles körben használnak modern programozási nyelvekben. A Python és Java nyelvekben alapértelmezett rendezési módszerként alkalmazzák, míg a C++ nyelvben külön implementációkat találhatunk. Az algoritmus erősségei közé tartozik a stabilitás, az adaptivitás részben rendezett adathalmazok esetén, valamint a hatékonyság nagyobb adathalmazok esetén. Ezek az előnyök teszik a Tim Sortot az egyik legkedveltebb rendezési algoritmussá a modern programozásban.

