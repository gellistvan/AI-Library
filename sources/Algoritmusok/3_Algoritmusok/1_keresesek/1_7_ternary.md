\newpage

## 1.7. Ternary Search (Hármas keresés)

A ternary search egy hatékony keresési algoritmus, amelyet rendezett tömbökben alkalmaznak. Az algoritmus alapelve hasonló a bináris kereséshez, de itt nem két részre osztjuk az intervallumot, hanem három részre. Ez lehetővé teszi a keresési folyamat párhuzamosítását és optimalizálását bizonyos esetekben, különösen akkor, ha a keresett elemek száma rendkívül nagy. Ebben a fejezetben megvizsgáljuk a ternary search alapelveit és feltételeit, bemutatjuk az implementációját és az alkalmazási területeit, valamint összehasonlítjuk a bináris kereséssel, hogy jobban megérthessük a két algoritmus közötti különbségeket és hasonlóságokat.

### 1.7.1. Alapelvek és feltételek

A ternary search, vagy hármas keresés, egy keresési algoritmus, amely a bináris keresés elvén alapul, de három részre osztja az aktuális keresési intervallumot, nem pedig kettőre. Ez az algoritmus akkor hasznos, ha az adatstruktúra rendezett, és bizonyos feltételek mellett előnyösebb lehet a bináris keresésnél. Ebben a fejezetben részletesen tárgyaljuk a ternary search alapelveit, a működését meghatározó feltételeket, és azt, hogy mikor érdemes ezt az algoritmust alkalmazni.

#### Az algoritmus alapelve

A ternary search algoritmus alapelve az, hogy egy rendezett tömböt három részre oszt, és két pivot elemet választ ki, amelyek elválasztják ezeket a részeket. Az algoritmus ezután összehasonlítja a keresett értéket ezekkel a pivot elemekkel, és a következőképpen jár el:

1. **Három részre osztás**: Az aktuális keresési intervallumot három részre osztjuk úgy, hogy két pivot elemet választunk ki, amelyek a harmadolási pontokon helyezkednek el.
2. **Összehasonlítás**: A keresett értéket először az első pivot elemmel, majd szükség esetén a második pivot elemmel hasonlítjuk össze.
3. **Keresési intervallum szűkítése**: Az összehasonlítás eredményétől függően három lehetséges új keresési intervallum közül választunk:
    - Ha a keresett érték kisebb, mint az első pivot elem, akkor a keresést az első harmadban folytatjuk.
    - Ha a keresett érték nagyobb, mint az első pivot elem, de kisebb, mint a második pivot elem, akkor a keresést a középső harmadban folytatjuk.
    - Ha a keresett érték nagyobb, mint a második pivot elem, akkor a keresést az utolsó harmadban folytatjuk.
4. **Rekurzió vagy iteráció**: Az algoritmust rekurzívan vagy iteratívan alkalmazzuk a kiválasztott új keresési intervallumra, amíg meg nem találjuk a keresett elemet, vagy a keresési intervallum nullára nem csökken.

#### Feltételek és követelmények

A ternary search alkalmazásához az alábbi feltételeknek kell teljesülniük:

1. **Rendezett tömb**: Az algoritmus csak rendezett tömbökben működik helyesen, mivel a harmadolási és összehasonlítási lépések rendezett elemeket feltételeznek.
2. **Statikus adatszerkezet**: Az adatszerkezetnek statikusnak kell lennie, vagyis a keresés során nem változhatnak az elemek pozíciói.
3. **Kereshető elem típusa**: Az algoritmus bármilyen típusú elemmel működik, amelyre meghatározhatók a szükséges összehasonlítási műveletek.

#### A ternary search előnyei és hátrányai

**Előnyök**:

1. **Hatékonyság bizonyos esetekben**: A ternary search a keresési intervallumot gyorsabban szűkíti le, mint a bináris keresés, különösen akkor, ha a keresési intervallum kezdetben nagyon nagy.
2. **Párhuzamosítás lehetősége**: Az algoritmus párhuzamosítható, mivel a három részre osztott keresési intervallumok függetlenek egymástól, és külön szálakon is feldolgozhatók.

**Hátrányok**:

1. **Több összehasonlítás**: Az algoritmus több összehasonlítást végez minden iterációban, mint a bináris keresés, ami bizonyos esetekben lassabbá teheti.
2. **Összetettebb megvalósítás**: A ternary search megvalósítása bonyolultabb, mint a bináris keresésé, mivel három különböző keresési intervallumot kell kezelni.

#### Ternary search implementáció

Az alábbiakban bemutatjuk a ternary search algoritmus egy lehetséges implementációját C++ nyelven:

```cpp
#include <iostream>

#include <vector>

int ternarySearch(const std::vector<int>& arr, int left, int right, int key) {
    if (right >= left) {
        // Két harmadolási pont meghatározása
        int mid1 = left + (right - left) / 3;
        int mid2 = right - (right - left) / 3;

        // Keresett érték összehasonlítása az első harmadolási ponttal
        if (arr[mid1] == key) {
            return mid1;
        }

        // Keresett érték összehasonlítása a második harmadolási ponttal
        if (arr[mid2] == key) {
            return mid2;
        }

        // Keresés a megfelelő intervallumban
        if (key < arr[mid1]) {
            return ternarySearch(arr, left, mid1 - 1, key);
        } else if (key > arr[mid2]) {
            return ternarySearch(arr, mid2 + 1, right, key);
        } else {
            return ternarySearch(arr, mid1 + 1, mid2 - 1, key);
        }
    }

    // Ha a keresett érték nincs a tömbben
    return -1;
}

int main() {
    std::vector<int> arr = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int key = 5;
    int index = ternarySearch(arr, 0, arr.size() - 1, key);

    if (index != -1) {
        std::cout << "Element found at index " << index << std::endl;
    } else {
        std::cout << "Element not found" << std::endl;
    }

    return 0;
}
```

#### Összegzés

A ternary search egy hatékony keresési algoritmus, amely három részre osztja a keresési intervallumot, és bizonyos esetekben előnyösebb lehet a bináris keresésnél. Az algoritmus alkalmazása előtt azonban fontos mérlegelni a rendezett adatszerkezet és a keresési feltételek meglétét, valamint az algoritmus előnyeit és hátrányait. Az implementáció összetettebb, mint a bináris keresésé, de megfelelően alkalmazva gyorsabb keresési eredményeket biztosíthat nagy adathalmazok esetén.

### 1.7.2. Implementáció és alkalmazások

A ternary search (hármas keresés) implementálása és alkalmazása fontos szerepet játszik az algoritmus hatékonyságának és használhatóságának megértésében. Ebben a fejezetben részletesen bemutatjuk a ternary search algoritmus implementációját, majd különböző alkalmazási területeket és gyakorlati példákat tárgyalunk, ahol az algoritmus kiemelkedő előnyöket biztosíthat.

#### Az algoritmus implementációja

A ternary search algoritmus megvalósítása C++ nyelven jól szemlélteti az elméleti alapelvek gyakorlati alkalmazását. Az alábbiakban bemutatjuk az algoritmus rekurzív és iteratív változatait, valamint kiegészítő függvényeket a hatékonyabb keresési folyamat érdekében.

##### Rekurzív megvalósítás

A rekurzív megközelítés a keresési intervallumot három részre osztja minden egyes rekurziós lépésben, és a megfelelő intervallumban folytatja a keresést. A kód világosan mutatja a három részre osztást és az összehasonlítási műveleteket.

```cpp
#include <iostream>

#include <vector>

int ternarySearchRecursive(const std::vector<int>& arr, int left, int right, int key) {
    if (right >= left) {
        int mid1 = left + (right - left) / 3;
        int mid2 = right - (right - left) / 3;

        if (arr[mid1] == key) {
            return mid1;
        }
        if (arr[mid2] == key) {
            return mid2;
        }

        if (key < arr[mid1]) {
            return ternarySearchRecursive(arr, left, mid1 - 1, key);
        } else if (key > arr[mid2]) {
            return ternarySearchRecursive(arr, mid2 + 1, right, key);
        } else {
            return ternarySearchRecursive(arr, mid1 + 1, mid2 - 1, key);
        }
    }
    return -1;
}
```

##### Iteratív megvalósítás

Az iteratív megközelítés a rekurzív változattal azonos alapelveken nyugszik, de ciklusokat használ a rekurzió helyett. Ez a módszer gyakran előnyösebb, mivel elkerüli a rekurzív hívásokból adódó memóriahasználatot és potenciális stack overflow problémákat.

```cpp
int ternarySearchIterative(const std::vector<int>& arr, int key) {
    int left = 0;
    int right = arr.size() - 1;

    while (right >= left) {
        int mid1 = left + (right - left) / 3;
        int mid2 = right - (right - left) / 3;

        if (arr[mid1] == key) {
            return mid1;
        }
        if (arr[mid2] == key) {
            return mid2;
        }

        if (key < arr[mid1]) {
            right = mid1 - 1;
        } else if (key > arr[mid2]) {
            left = mid2 + 1;
        } else {
            left = mid1 + 1;
            right = mid2 - 1;
        }
    }
    return -1;
}
```

#### Kiegészítő függvények

A hatékony keresési folyamat érdekében különböző kiegészítő függvényeket is implementálhatunk, mint például a rendezés vagy az adatok előfeldolgozása, amelyek biztosítják, hogy az adatszerkezet megfeleljen a ternary search előfeltételeinek.

```cpp
void sortArray(std::vector<int>& arr) {
    std::sort(arr.begin(), arr.end());
}
```

#### Alkalmazások

A ternary search algoritmus számos gyakorlati alkalmazással bír különböző területeken, ahol nagy adathalmazok hatékony keresése szükséges. Az alábbiakban néhány kiemelkedő alkalmazási területet tárgyalunk.

##### Nagy adathalmazok keresése

A ternary search különösen hatékony nagy adathalmazok esetén, ahol a keresési intervallum gyors szűkítése jelentős teljesítménynövekedést eredményezhet. Az algoritmus párhuzamosíthatósága további előnyt biztosít a nagy adatbázisok és adatközpontok kezelésében.

##### Játékfejlesztés

A játékfejlesztésben gyakran szükséges gyors keresési algoritmusok alkalmazása, például pályák vagy karakterek kereséséhez egy nagy adatbázisban. A ternary search lehetővé teszi a játékok gyorsabb és hatékonyabb futását, különösen akkor, ha a keresések párhuzamosan végezhetők.

##### Hálózati protokollok és kommunikáció

A hálózati protokollok és kommunikációs rendszerek gyakran igényelnek gyors keresési algoritmusokat a csomagok vagy adatok gyors megtalálása érdekében. A ternary search itt is hasznos lehet, mivel csökkentheti a keresési időt és növelheti a rendszer hatékonyságát.

##### Keresőmotorok és információkeresés

A keresőmotorok alapvető működési elve az adatok gyors és hatékony keresése. A ternary search alkalmazása segíthet a keresési folyamatok optimalizálásában, különösen nagy adatbázisokban és komplex keresési feltételek esetén.

#### Összefoglalás

A ternary search egy hatékony keresési algoritmus, amely három részre osztja a keresési intervallumot, és számos alkalmazási területen előnyös lehet. Az algoritmus rekurzív és iteratív változatainak megértése és implementálása elengedhetetlen a hatékony használathoz. A nagy adathalmazok kezelése, a játékfejlesztés, a hálózati protokollok és a keresőmotorok mind olyan területek, ahol a ternary search jelentős teljesítménynövekedést biztosíthat. Az algoritmus előnyei mellett fontos megérteni a feltételeket és a megvalósítás bonyolultságát, hogy a lehető legjobban kihasználhassuk annak potenciálját.

### 1.7.3. Összehasonlítás bináris kereséssel

A ternary search (hármas keresés) és a binary search (bináris keresés) két hatékony algoritmus, amelyek rendezett tömbökben történő keresésre szolgálnak. Mindkét algoritmus a megosztás és uralkodás (divide and conquer) elvén alapul, de különböző módon osztják fel a keresési intervallumot. Ebben a fejezetben részletesen összehasonlítjuk a két algoritmust, bemutatva azok előnyeit, hátrányait, és különböző alkalmazási területeit. Az összehasonlítás során figyelembe vesszük az időbeli és térbeli komplexitást, a gyakorlati teljesítményt, valamint a párhuzamosíthatóságot.

#### Alapelvek és működés

##### Bináris keresés

A bináris keresés egy rendezett tömbben keres egy adott elemet az alábbi lépések végrehajtásával:

1. **Keresési intervallum felezése**: A keresési intervallumot két részre osztja a középső elemen keresztül.
2. **Összehasonlítás**: Összehasonlítja a keresett elemet a középső elemmel.
3. **Intervallum szűkítése**: Az összehasonlítás eredményétől függően a keresést az intervallum bal vagy jobb felében folytatja.
4. **Rekurzió vagy iteráció**: A keresési folyamatot rekurzívan vagy iteratívan folytatja, amíg meg nem találja az elemet, vagy az intervallum üressé nem válik.

A bináris keresés pseudokódja a következő:

```cpp
int binarySearch(const std::vector<int>& arr, int left, int right, int key) {
    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == key) {
            return mid;
        } else if (arr[mid] < key) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;
}
```

##### Ternary keresés

A ternary search három részre osztja a keresési intervallumot, és két pivot elemet választ ki:

1. **Keresési intervallum harmadolása**: Az intervallumot három részre osztja két pivot elem segítségével.
2. **Összehasonlítás**: Összehasonlítja a keresett elemet a két pivot elemmel.
3. **Intervallum szűkítése**: A keresést a három intervallum közül a megfelelőben folytatja.
4. **Rekurzió vagy iteráció**: A keresési folyamatot rekurzívan vagy iteratívan folytatja, amíg meg nem találja az elemet, vagy az intervallum üressé nem válik.

A ternary search pseudokódja a következő:

```cpp
int ternarySearch(const std::vector<int>& arr, int left, int right, int key) {
    while (right >= left) {
        int mid1 = left + (right - left) / 3;
        int mid2 = right - (right - left) / 3;

        if (arr[mid1] == key) {
            return mid1;
        } else if (arr[mid2] == key) {
            return mid2;
        } else if (key < arr[mid1]) {
            right = mid1 - 1;
        } else if (key > arr[mid2]) {
            left = mid2 + 1;
        } else {
            left = mid1 + 1;
            right = mid2 - 1;
        }
    }
    return -1;
}
```

#### Időbeli komplexitás

A bináris keresés és a ternary keresés időbeli komplexitása különböző. Az időbeli komplexitás meghatározza, hogy az algoritmus hány lépésben találja meg a keresett elemet a legrosszabb esetben.

##### Bináris keresés

A bináris keresés időbeli komplexitása O(log N), ahol N a tömb mérete. Mivel minden lépésben a keresési intervallumot felére csökkenti, az algoritmus logaritmikus időben fut.

##### Ternary keresés

A ternary keresés időbeli komplexitása O(log3 N) (logaritmus alapja 3). Bár ez kisebb, mint a bináris keresés logaritmikus alapja (2), a ternary keresés minden lépésben két összehasonlítást végez, szemben a bináris keresés egyetlen összehasonlításával. Ez azt jelenti, hogy a ternary keresés gyakorlati teljesítménye nem feltétlenül jobb a bináris keresésénél, különösen kisebb adathalmazok esetén.

#### Térbeli komplexitás

Mindkét algoritmus térbeli komplexitása O(1), mivel csak néhány kiegészítő változót használnak a keresési folyamat során. A memóriaterhelés szempontjából nincs jelentős különbség a két algoritmus között.

#### Gyakorlati teljesítmény

A gyakorlati teljesítmény szempontjából a bináris keresés és a ternary keresés közötti választás több tényezőtől függ, beleértve a tömb méretét, a környezetet (pl. cache hatékonyság), és az implementáció részleteit.

##### Kis adathalmazok

Kisebb adathalmazok esetén a bináris keresés gyakran gyorsabb, mivel kevesebb összehasonlítást végez minden lépésben. A ternary keresés extra összehasonlításai miatt a teljesítményhátrány nagyobb lehet, mint a keresési intervallum gyorsabb szűkítése.

##### Nagy adathalmazok

Nagy adathalmazok esetén a ternary keresés előnyösebb lehet, mivel a keresési intervallumot gyorsabban szűkíti. Azonban a két algoritmus közötti teljesítménykülönbség nem mindig jelentős, és a bináris keresés egyszerűsége miatt gyakran preferált.

#### Párhuzamosíthatóság

A ternary keresés egyik előnye, hogy párhuzamosítható. Mivel három részre osztja a keresési intervallumot, a három rész külön szálakon párhuzamosan is feldolgozható. Ez különösen előnyös lehet nagy adathalmazok és többmagos processzorok esetén. A bináris keresés párhuzamosítása bonyolultabb, mivel csak két részre osztja a keresési intervallumot.

#### Alkalmazási területek

##### Bináris keresés

A bináris keresés széles körben alkalmazott algoritmus különböző területeken, mint például:
- **Adatbázis-kezelés**: Rendezett rekordok keresése adatbázisokban.
- **Szövegszerkesztők**: Szavak keresése rendezett szótárakban.
- **Rendezett fájlok keresése**: Gyors keresés nagy, rendezett fájlokban.

##### Ternary keresés

A ternary keresés speciális alkalmazási területeken hasznos, ahol a keresési intervallum gyors szűkítése és a párhuzamosíthatóság előnyt jelenthet:
- **Nagy adatbázisok**: Nagy adatbázisok gyors keresése, különösen többmagos rendszereken.
- **Játékfejlesztés**: Játékokban gyors keresés nagy adatstruktúrákban, például térképeken vagy karakteradatbázisokban.
- **Hálózati rendszerek**: Csomagok gyors keresése nagy hálózati táblákban.

#### Összegzés

A bináris keresés és a ternary keresés két hatékony algoritmus rendezett tömbök keresésére. Bár a bináris keresés egyszerűbb és kisebb adathalmazok esetén gyorsabb, a ternary keresés nagy adathalmazok és párhuzamos feldolgozás esetén előnyös lehet. Az algoritmusok időbeli és térbeli komplexitása, gyakorlati teljesítménye és alkalmazási területei alapján a felhasználási kontextustól függően választhatunk a két algoritmus között. A tudományos és gyakorlati szempontok figyelembevételével a megfelelő keresési algoritmus kiválasztása jelentős teljesítményjavulást eredményezhet különböző alkalmazásokban.