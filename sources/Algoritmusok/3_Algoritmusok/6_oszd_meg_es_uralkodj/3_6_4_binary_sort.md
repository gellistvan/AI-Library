\newpage

## 6.4. Binary Search

Az oszd-meg-és-uralkodj algoritmusok egyik legismertebb alkalmazása a bináris keresés, amely hatékony módot kínál rendezett listákban történő elemek gyors megtalálására. A bináris keresés alapelve egyszerű, mégis rendkívül erőteljes: egy felosztással két részre csökkenti a keresési tér nagyságát, így exponenciálisan növelve a keresés sebességét. Ebben a fejezetben részletesen megvizsgáljuk a bináris keresés algoritmusát és annak implementációját, majd alapos teljesítményelemzést végzünk annak érdekében, hogy megértsük, miért nyújt kiemelkedő hatékonyságot. Emellett bemutatunk néhány optimalizációs technikát is, amelyek tovább javíthatják a keresés sebességét és alkalmazhatóságát a gyakorlatban.

### 6.4.1 Algoritmus és implementáció

A bináris keresés (Binary Search) egy hatékony algoritmus az elemek rendezett listájában történő keresésre. Az algoritmus alapja a "Oszd-meg-és-uralkodj" technika, amely azzal a céllal bontja kisebb részekre a problémát, hogy csökkentse a keresési térfogatot a bajt felére csökkentve minden lépésben.

#### Bináris keresés alapelve

A bináris keresés feltételezi, hogy a bemeneti lista rendezett. Ha az elemek nem rendezettek, először rendezni kell őket, mivel a bináris keresés csak ekkor működik helyesen. Az algoritmus lényege, hogy lépésről lépésre csökkentsük a keresési intervallum méretét, amíg meg nem találjuk a kívánt elemet vagy a keresési intervallum nullára nem csökken.

A bináris keresés működése:

1. **Kezdőállapot:** Meghatározunk egy bal és egy jobb határt (kezdetben a lista elején és végén).
2. **Középpont meghatározása:** Az aktuális keresési intervallum középső elemének indexét kiszámítjuk.
3. **Középső elem ellenőrzése:** Összehasonlítjuk a középső elemet a keresett értékkel.
    - Ha a középső elem megegyezik a keresett értékkel, akkor megtaláltuk az elemet.
    - Ha a keresett érték kisebb, mint a középső elem, akkor a keresési intervallum bal oldalára szűkítjük.
    - Ha a keresett érték nagyobb, mint a középső elem, akkor a keresési intervallum jobb oldalára szűkítjük.
4. **Ismétlés:** Az 2-3. lépéseket ismételjük, amíg a keresési intervallum mérete nem lesz 0 (vagyis a bal határ nem haladja meg a jobb határt).

#### Bináris keresés pseudokód

Íme a bináris keresés alapvető pseudokódja:

```
binarySearch(array, value):
    low = 0
    high = length(array) - 1
    
    while low <= high:
        mid = low + (high - low) / 2
        if array[mid] == value:
            return mid
        else if array[mid] < value:
            low = mid + 1
        else:
            high = mid - 1
    
    return -1  // Value not found
```

#### Bináris keresés implementáció C++ nyelven

Az alábbiakban bemutatjuk a bináris keresés C++ nyelvű implementációját:

```cpp
#include <iostream>
#include <vector>

int binarySearch(const std::vector<int>& array, int value) {
    int low = 0;
    int high = array.size() - 1;

    while (low <= high) {
        int mid = low + (high - low) / 2;

        if (array[mid] == value) {
            return mid;
        } 
        else if (array[mid] < value) {
            low = mid + 1;
        } 
        else {
            high = mid - 1;
        }
    }

    return -1; // Value not found
}

int main() {
    std::vector<int> sortedArray = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    int valueToFind = 7;

    int result = binarySearch(sortedArray, valueToFind);

    if (result != -1) {
        std::cout << "Value found at index: " << result << std::endl;
    } else {
        std::cout << "Value not found in array." << std::endl;
    }
    
    return 0;
}
```

#### Bináris keresés működésének részletei

A bináris keresés egy iteratív algoritmus, így viszonylag egyszerűen megérthető. Azonban érdemes megvizsgálni néhány részletet, amelyek fontosak az algoritmus hatékonysága és korrekt működése szempontjából.

1. **Középpont kiszámítása:**
   A középpont indexének kiszámítása során gyakran a `(low + high) / 2` kifejezést használják. Azonban ez a módszer túlcsordulást eredményezhet nagy értékek esetén. Ehelyett a `mid = low + (high - low) / 2` formula biztonságosabb, mivel elkerüli a túlcsordulást.

2. **Üres lista kezelése:**
   Az algoritmus első lépése a keresési intervallum inicializálása. Amennyiben a lista üres, azonnal visszaadja a -1-es értéket, jelezve, hogy az elem nem található a listában.

3. **Hatékonyság és optimalizálás:**
   Az algoritmus futási ideje O(log n), mert minden iteráció felezi a keresési intervallum méretét. Ez jelentősen gyorsabbá teszi, mint a lineáris keresést, amely O(n) időt igényel.

4. **Rekurzív változat:**
   A bináris keresést rekurzív módon is meg lehet valósítani. A rekurzív megközelítés sokszor egyszerűbb és természetesebbnek tűnik, de figyelni kell a rekurzív hívások mélységére (stack overflow veszélye). Íme, egy rekurzív C++ implementáció:

```cpp
#include <iostream>
#include <vector>

int binarySearchRecursive(const std::vector<int>& array, int low, int high, int value) {
    if (low > high) {
        return -1; // Value not found
    }

    int mid = low + (high - low) / 2;

    if (array[mid] == value) {
        return mid;
    } 
    else if (array[mid] < value) {
        return binarySearchRecursive(array, mid + 1, high, value);
    } 
    else {
        return binarySearchRecursive(array, low, mid - 1, value);
    }
}

int main() {
    std::vector<int> sortedArray = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    int valueToFind = 7;

    int result = binarySearchRecursive(sortedArray, 0, sortedArray.size() - 1, valueToFind);

    if (result != -1) {
        std::cout << "Value found at index: " << result << std::endl;
    } else {
        std::cout << "Value not found in array." << std::endl;
    }

    return 0;
}
```

#### Bináris keresés alkalmazási területei

A bináris keresés széles körben alkalmazható a számítástechnika különböző területein:
- **Adatbáziskezelés:** Gyors keresés a rendezett adatrekordok között.
- **Számítógépes grafika:** Gyorsított geometriai elemek keresése.
- **Gyors szótárkeresés:** Gyors keresés rendezett szótárakban vagy lexikonokban.
- **Hálózati forgalom:** Gyors keresés IP-tartományok között.

### 6.4.2 Teljesítmény elemzés és optimalizálás

A bináris keresés (binary search) egy kiemelkedően hatékony algoritmus, amely speciálisan rendezett listák átvizsgálásához készült. Ezen fejezet célja részletesen feltárni a bináris keresési algoritmus teljesítményének különböző aspektusait, beleértve a legjobb, legrosszabb és áltagos esetek időkomplexitását, valamint néhány gyakori optimalizációs módszert is bemutatni.

#### 6.4.2.1 Időkomplexitás

Az időkomplexitás elemzéséhez vegyük figyelembe az algoritmus iteratív és rekurzív változatait is.

**Iteratív bináris keresés**
Az iteratív bináris keresés azon az elven alapul, hogy a keresési intervallumot felezi meg mindaddig, amíg a keresett elem vagy megtalálásra kerül, vagy a keresési intervallum üres nem lesz.

```cpp
int binarySearch(int arr[], int size, int key) {
    int left = 0, right = size - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == key)
            return mid;

        if (arr[mid] < key)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return -1;
}
```

Az iteratív algoritmus legrosszabb esetben $O(\log n)$ időkomplexitású, ahol $n$ a lista hossza. Minden egyes iteráció során a keresési intervallum mérete felére csökken, tehát legrosszabb esetben $\log_2(n)$ iteráció szükséges.

**Rekurzív bináris keresés**
A rekurzív bináris keresés hasonló elveken alapul, mint az iteratív változat, de egyértelműbb és természetesebb megvalósítást tesz lehetővé a rekurzió használatával.

```cpp
int binarySearchRec(int arr[], int left, int right, int key) {
    if (right >= left) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == key)
            return mid;

        if (arr[mid] > key)
            return binarySearchRec(arr, left, mid - 1, key);

        return binarySearchRec(arr, mid + 1, right, key);
    }
    return -1;
}
```

A rekurzív változat szintén $O(\log n)$ időkomplexitású, és az esetek száma ugyanúgy logaritmikus mértékben csökken minden függvényhívás során.

**Legjobb eset**
A legjobb eset akkor valósul meg, ha a keresett elem éppen a középső elem, azaz egyszeri összehasonlítással megtaláljuk az elemet. Ennek időkomplexitása $O(1)$.

**Átlagos eset**
Az átlagos eset elemzésekor feltételezzük, hogy a keresett elem minden pozícióban egyenlő valószínűséggel lehet. Ilyen körülmények között az átlagos időkomplexitás szintén $O(\log n)$, bár az elemzése kissé bonyolultabb, és valószínűségszámítást igényel.

#### 6.4.2.2 Térkomplexitás

Az iteratív bináris keresés térkomplexitása $O(1)$, mivel csupán néhány segédváltozót használ a keresés során (left, right, mid). Ezzel szemben a rekurzív változat $O(\log n)$ térkomplexitású a függvényhívások veremtárigénye miatt.

#### 6.4.2.3 Konkretizált optimalizálási technikák

**Túlindexelési optimalizáció**
A legelső javaslott optimalizálási technika az, hogy a középső index kiszámításakor használjunk $left + (right - left) / 2$ ahelyett, hogy direkt $(left + right) / 2$-t használnánk. Ez kiküszöböli az összeg túlcsordulásának kockázatát, különösen nagy tömbméretek esetén.

**A duplaszámolás csökkentése**
Egy további technika, amely hozzájárulhat az optimalizáláshoz, hogy duplaszámolás esetén (amikor az új intervallum kezdő értékét után azonnal ellenőrizzük) egyszerű feltételalapú korrekcióval minimalizálhatjuk az ismétlő számolási lépések számát.

**Beszúrási keresés**
Ha feltételezzük, hogy a tömb rendezve van, és az elemek viszonylag egyenletes eloszlást mutatnak, az intervallum középpontja meghatározható a következő képlettel:
$$
pos = left + \frac{(right-left)}{(arr[right]-arr[left])}(key-arr[left])
$$
Ez talán felgyorsítja az algoritmust bizonyos speciális eloszlásoknál.

#### 6.4.2.4 Gyakorlati alkalmazások és korlátok

**Rendezési feltételek**
A bináris keresési algoritmus csak rendezett tömbök esetén használható; ha az adatok nincsenek rendezve, a rendezés időkomplexitása is bekapcsolhat, például $O(n \log n)$ (gyorsrendezés esetén).

**Keresési kulcs dinamikus változása**
Ha az adathalmaz dinamikusan változik, vagy gyakran módosítjuk azt (beszúrások, törlések), lehet érdemes más adatszerkezetet választani, mint például egy szelf-balan sítható keresési fa (AVL fa) vagy egy hash-tábla, amelyek jobb teljesítményt nyújthatnak ezekben az esetekben.

**Paralelizáció**
Paralel struktúrával (különösen nagy adathalmazok esetében) tovább növelhetjük az algoritmus hatékonyságát. Bár az egyes dekompozíciók és memória-hozzáférési konfliktusok kezelése külön kihívásokat jelenthet.


