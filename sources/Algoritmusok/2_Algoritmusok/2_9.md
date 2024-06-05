\newpage
## 2.9.   Radix Sort

A Radix Sort egy nem-komparatív rendezési algoritmus, amely hatékonyan képes rendezni adatokat előre meghatározott kulcsok szerint. Ez az algoritmus különösen hasznos, amikor a rendezendő elemek kulcsai fix hosszúságúak vagy meghatározott számjegyekkel rendelkeznek, például egész számok vagy sztringek esetében. A Radix Sort két változata ismert: az LSD (Least Significant Digit) és az MSD (Most Significant Digit), amelyek különböző sorrendben dolgozzák fel a kulcsok számjegyeit. E fejezet célja, hogy bemutassa a Radix Sort alapelveit és implementációját, részletesen elemezze annak teljesítményét és komplexitását, valamint gyakorlati alkalmazásokat és példákat ismertessen az algoritmus hatékony felhasználására.

### 2.9.1. Alapelvek és implementáció (LSD, MSD változatok)

A Radix Sort, vagy magyarul helyiérték rendezés, egy nem-komparatív rendezési algoritmus, amely több lépésben rendezi az adatokat a kulcsaik helyiértékei alapján. Az algoritmus a legkisebb helyiértéktől (LSD - Least Significant Digit) vagy a legnagyobb helyiértéktől (MSD - Most Significant Digit) kezdve dolgozza fel a számjegyeket. Az alábbiakban részletesen bemutatjuk mindkét változat alapelveit és implementációs lépéseit.

#### LSD Radix Sort (Least Significant Digit)

Az LSD Radix Sort az algoritmus azon változata, amely a legkisebb helyiértéktől kezdi a rendezést, és fokozatosan halad a legnagyobb helyiérték felé. Az eljárás során minden helyiértéket külön rendezünk egy stabil rendezési algoritmus segítségével, mint például a Counting Sort vagy a Bucket Sort. A stabilitás itt kritikus jelentőségű, mert biztosítja, hogy az azonos helyiértékkel rendelkező elemek sorrendje megmaradjon.

##### LSD Radix Sort Algoritmus Lépései

1. **Helyiérték meghatározása:** Az összes elem legkisebb helyiértékétől indulva, fokozatosan haladunk a legnagyobb helyiérték felé.
2. **Stabil rendezés:** Minden helyiértékre külön-külön stabil rendezést végzünk.
3. **Iteráció:** Az összes helyiérték feldolgozása után az eredmény rendezett lesz.

Például, ha egész számokat rendezünk, a legkisebb helyiérték az egyesek helye, majd a tízesek, százak stb.

##### LSD Radix Sort Példa C++ Kóddal

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

// Segédfüggvény a maximális érték megtalálásához
int getMax(const std::vector<int>& arr) {
    return *std::max_element(arr.begin(), arr.end());
}

// Counting sort segédfüggvény a Radix Sort-hoz
void countingSort(std::vector<int>& arr, int exp) {
    int n = arr.size();
    std::vector<int> output(n);
    int count[10] = {0};

    // Számjegyek gyakoriságának számlálása
    for (int i = 0; i < n; i++) {
        count[(arr[i] / exp) % 10]++;
    }

    // Count[i] most már tartalmazza a helyes pozíciókat a kimeneti tömbben
    for (int i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }

    // Kimeneti tömb építése
    for (int i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }

    // Kimenet másolása az eredeti tömbbe
    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }
}

// LSD Radix Sort fő függvénye
void radixSort(std::vector<int>& arr) {
    int m = getMax(arr);

    // Szorzat (exp) növelése 10-szeresével minden számjegyért
    for (int exp = 1; m / exp > 0; exp *= 10) {
        countingSort(arr, exp);
    }
}

int main() {
    std::vector<int> arr = {170, 45, 75, 90, 802, 24, 2, 66};
    radixSort(arr);
    for (int num : arr) {
        std::cout << num << " ";
    }
    return 0;
}
```

#### MSD Radix Sort (Most Significant Digit)

Az MSD Radix Sort az algoritmus azon változata, amely a legnagyobb helyiértéktől kezdi a rendezést, és fokozatosan halad a legkisebb helyiérték felé. Az algoritmus rekurzív jellegű, és különösen hasznos lehet változó hosszúságú kulcsok esetén, mint például karakterláncok rendezésekor.

##### MSD Radix Sort Algoritmus Lépései

1. **Helyiérték meghatározása:** Kezdjük a legnagyobb helyiértékkel.
2. **Elosztás:** Az elemeket helyiérték alapján külön vödrökbe (bucket) osztjuk.
3. **Rekurzió:** Minden vödröt külön rendezünk rekurzívan, kisebb helyiértékre lépve.
4. **Összefűzés:** Az összes vödör tartalmát összefűzzük, hogy megkapjuk a végső rendezett listát.

##### MSD Radix Sort Példa C++ Kóddal

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

// Karakter pozíciójának visszaadása
int charAt(const std::string& s, int d) {
    if (d < s.length()) return s[d];
    else return -1; // Sentinel érték a rövidebb sztringekhez
}

// MSD Radix Sort segédfüggvény
void msdSort(std::vector<std::string>& arr, int lo, int hi, int d) {
    if (hi <= lo) return;

    const int R = 256; // ASCII karakterek száma
    std::vector<int> count(R + 2, 0);
    std::vector<std::string> aux(hi - lo + 1);

    // Számlálás
    for (int i = lo; i <= hi; i++) {
        count[charAt(arr[i], d) + 2]++;
    }

    // Gyűjtés
    for (int r = 0; r < R + 1; r++) {
        count[r + 1] += count[r];
    }

    // Átrendezés
    for (int i = lo; i <= hi; i++) {
        aux[count[charAt(arr[i], d) + 1]++] = arr[i];
    }

    // Másolás vissza
    for (int i = lo; i <= hi; i++) {
        arr[i] = aux[i - lo];
    }

    // Rekurzív rendezés
    for (int r = 0; r < R; r++) {
        msdSort(arr, lo + count[r], lo + count[r + 1] - 1, d + 1);
    }
}

// MSD Radix Sort fő függvénye
void radixSortMSD(std::vector<std::string>& arr) {
    msdSort(arr, 0, arr.size() - 1, 0);
}

int main() {
    std::vector<std::string> arr = {"she", "sells", "seashells", "by", "the", "seashore"};
    radixSortMSD(arr);
    for (const std::string& str : arr) {
        std::cout << str << " ";
    }
    return 0;
}
```

#### Az LSD és MSD Összehasonlítása

Az LSD és MSD Radix Sort algoritmusok közötti választás számos tényezőtől függ, beleértve az adatok típusát és a specifikus alkalmazási követelményeket. Az LSD Radix Sort általában jobban teljesít fix hosszúságú kulcsokkal, míg az MSD Radix Sort előnyös lehet változó hosszúságú kulcsok esetén.

##### Teljesítmény

Az LSD Radix Sort időbeli komplexitása $O(w \cdot (n + k))$, ahol $n$ az elemek száma, $w$ a legnagyobb kulcs hosszúsága, és $k$ a számjegyek (karakterek) száma. Az MSD Radix Sort hasonló időbeli komplexitással rendelkezik, de gyakran hatékonyabb változó hosszúságú kulcsok esetén, mivel képes a rövidebb kulcsokat korábban elkülöníteni.

##### Stabilitás és Memóriahasználat

A stabilitás és memóriahasználat tekintetében mindkét változat stabil rendezést biztosít, amennyiben stabil rendezési algoritmust alkalmazunk a helyiértékek rendezésére. Az LSD Radix Sort általában kevesebb memóriát igényel, mivel nem használ rekurziót, míg az MSD Radix Sort rekurzív jellege miatt több memóriahasználatot eredményezhet.

##### Gyakorlati Alkalmazások

Az LSD Radix Sort gyakran alkalmazható numerikus adatok rendezésére, például bankszámlaszámok, telefonszámok vagy más fix hosszúságú azonosítók esetén. Az MSD Radix Sort különösen hasznos változó hosszúságú szöveges adatok, például szavak vagy fájlnevek rendezésére.

#### Összegzés

A Radix Sort egy hatékony és sokoldalú nem-komparatív rendezési algoritmus, amely különböző helyiérték-alapú megközelítésekkel dolgozik. Az LSD és MSD változatok eltérő előnyöket kínálnak különböző típusú adatok esetén, és megfelelően alkalmazva jelentős teljesítményjavulást eredményezhetnek a hagyományos komparatív rendezési algoritmusokkal szemben.

### 2.9.2. Teljesítmény és komplexitás elemzése

A Radix Sort teljesítménye és komplexitása különbözik a hagyományos komparatív rendezési algoritmusoktól, mint például a quicksort vagy mergesort, mivel a Radix Sort nem az elemek közötti összehasonlításokra alapoz. Az alábbiakban részletesen elemezzük a Radix Sort különböző aspektusait, beleértve az időbeli komplexitást, a térbeli komplexitást, valamint az algoritmus viselkedését különböző típusú adatokkal és gyakorlati alkalmazási körülmények között.

#### Időbeli Komplexitás

A Radix Sort időbeli komplexitása az adatok kulcsainak hosszától és az alkalmazott stabil rendezési algoritmus hatékonyságától függ. Az LSD (Least Significant Digit) és az MSD (Most Significant Digit) Radix Sort algoritmusok időbeli komplexitása különbözőképpen alakul.

##### LSD Radix Sort

Az LSD Radix Sort esetén minden helyiértéket külön rendezzük egy stabil rendezési algoritmussal, például a Counting Sorttal. A teljes időbeli komplexitás a következőképpen határozható meg:

- $n$: az elemek száma
- $d$: a legnagyobb kulcs helyiértékeinek száma (a kulcsok hosszúsága)
- $k$: a helyiértékek értéktartománya (például 0-9 közötti számjegyek esetén $k = 10$)

Az LSD Radix Sort minden helyiértékre egy stabil rendezést végez, amelynek időkomplexitása $O(n + k)$. Mivel $d$ ilyen rendezést végzünk, a teljes időbeli komplexitás:

$$
O(d \cdot (n + k))
$$

##### MSD Radix Sort

Az MSD Radix Sort rekurzív algoritmus, amely a legnagyobb helyiértékkel kezd és fokozatosan halad a kisebb helyiértékek felé. A teljes időbeli komplexitás szintén függ a kulcsok hosszától és az értéktartománytól:

- $n$: az elemek száma
- $d$: a legnagyobb kulcs helyiértékeinek száma
- $k$: a helyiértékek értéktartománya

Az MSD Radix Sort minden rekurzív lépésben a helyiértékek alapján vödrökbe (bucket) osztja az elemeket, majd újra alkalmazza a rendezést az egyes vödrökben. Az időbeli komplexitás így:

$$
O(d \cdot (n + k))
$$

Mivel minden helyiérték rendezése lineáris időben történik, az összes helyiérték feldolgozása összességében $d$-szor történik meg.

#### Térbeli Komplexitás

A térbeli komplexitás az algoritmus által használt kiegészítő memória mennyiségét jelenti.

##### LSD Radix Sort

Az LSD Radix Sort esetében az egyes helyiértékek rendezése során kiegészítő memóriát használunk a stabil rendezéshez:

- Az eredeti tömb méretével megegyező méretű kimeneti tömb.
- Számláló tömb, amely a helyiértékek gyakoriságát tartalmazza (mérete $k$).

Ezért a teljes térbeli komplexitás:

$$
O(n + k)
$$

##### MSD Radix Sort

Az MSD Radix Sort esetében a rekurzív természetből adódóan több kiegészítő memóriát használhatunk:

- Minden rekurzív lépéshez szükség van a vödrök és a kimeneti tömbök kezelésére.
- A rekurzió mélysége függ a kulcsok hosszúságától ($d$).

Ezért a teljes térbeli komplexitás:

$$
O(n + k + d)
$$

#### Stabilitás

A stabilitás egy rendezési algoritmus azon képessége, hogy az azonos kulcsú elemek eredeti sorrendjét megtartsa a rendezés során. Mind az LSD, mind az MSD Radix Sort stabil rendezési algoritmusok, feltéve, hogy stabil rendezési módszert alkalmazunk a helyiértékek rendezéséhez (például Counting Sort).

#### Gyakorlati Teljesítmény és Alkalmazások

A Radix Sort különböző gyakorlati alkalmazásokban használható, különösen ott, ahol a rendezendő elemek kulcsai numerikusak vagy karakterláncok. Néhány gyakorlati alkalmazás:

- **Számítógépes grafika:** Nagy számú geometriai objektum rendezése előre meghatározott kulcsok alapján.
- **Adatbázis-kezelés:** Rekordok rendezése több kulcs szerint, például számlaszámok, telefonszámok.
- **Telekommunikáció:** Telefonszámok vagy IP-címek rendezése.

#### Példák és Tesztelés

A Radix Sort gyakorlati teljesítményének tesztelése és összehasonlítása más rendezési algoritmusokkal kritikus fontosságú a választott módszer hatékonyságának meghatározásához. Az alábbiakban bemutatunk néhány példát és tesztelési eredményt.

##### Tesztelési Környezet

A tesztelési környezet beállításai a következők:

- Adatok típusa: véletlenszerű egész számok, sztringek.
- Adatmennyiség: különböző nagyságrendű adathalmazok (például $10^3$, $10^5$, $10^7$ elem).
- Mérés: futási idő, memóriahasználat.

##### Eredmények

Az alábbiakban bemutatunk néhány hipotetikus eredményt különböző adathalmazokra:

| Adatmennyiség | Algoritmus         | Futási idő (ms) | Memóriahasználat (MB) |
|---------------|---------------------|-----------------|-----------------------|
| $10^3$        | LSD Radix Sort      | 5               | 1                     |
|               | MSD Radix Sort      | 6               | 1                     |
| $10^5$        | LSD Radix Sort      | 50              | 10                    |
|               | MSD Radix Sort      | 55              | 12                    |
| $10^7$        | LSD Radix Sort      | 500             | 100                   |
|               | MSD Radix Sort      | 550             | 120                   |

##### Értékelés

Az eredmények azt mutatják, hogy a Radix Sort algoritmusok jól teljesítenek nagy mennyiségű adatok rendezésekor, különösen akkor, ha az adatok fix hosszúságú kulcsokkal rendelkeznek. Az LSD Radix Sort némileg jobb teljesítményt mutat a tesztek alapján, különösen nagyobb adathalmazok esetén, köszönhetően egyszerűbb memóriahasználatának.

#### Összefoglalás

A Radix Sort egy hatékony és sokoldalú nem-komparatív rendezési algoritmus, amely jelentős előnyökkel rendelkezik bizonyos típusú adatok rendezésekor. Az algoritmus időbeli és térbeli komplexitása kedvező, különösen nagy adathalmazok esetén. A Radix Sort stabilitása és gyakorlati alkalmazásai széleskörűvé teszik az algoritmust különböző területeken. Az LSD és MSD változatok különböző előnyöket kínálnak, így a választás az adott alkalmazási körülményektől és az adatok specifikus tulajdonságaitól függ.

### 2.9.3. Gyakorlati alkalmazások és példák

A Radix Sort algoritmus hatékonyságának és sokoldalúságának köszönhetően számos gyakorlati alkalmazásban használják, ahol nagy mennyiségű adat gyors és hatékony rendezése szükséges. Ebben az alfejezetben részletesen bemutatjuk a Radix Sort néhány gyakorlati alkalmazását és példákat adunk az algoritmus használatára különböző kontextusokban.

#### Numerikus Adatok Rendezése

A Radix Sort egyik leggyakoribb alkalmazási területe a numerikus adatok rendezése. A numerikus adatok esetében a kulcsok helyiértékei egyértelműek, ami lehetővé teszi a Radix Sort hatékony működését.

##### Példa: Banki Adatok Rendezése

Képzeljük el, hogy egy bank számlaszámokat szeretne rendezni. A számlaszámok általában fix hosszúságú numerikus értékek, ami ideális eset a Radix Sort számára. A következő C++ kódrészlet egy egyszerű példát mutat be, ahol számlaszámokat rendezünk LSD Radix Sort segítségével.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

// Segédfüggvény a maximális érték megtalálásához
int getMax(const std::vector<int>& arr) {
    return *std::max_element(arr.begin(), arr.end());
}

// Counting sort segédfüggvény a Radix Sort-hoz
void countingSort(std::vector<int>& arr, int exp) {
    int n = arr.size();
    std::vector<int> output(n);
    int count[10] = {0};

    // Számjegyek gyakoriságának számlálása
    for (int i = 0; i < n; i++) {
        count[(arr[i] / exp) % 10]++;
    }

    // Count[i] most már tartalmazza a helyes pozíciókat a kimeneti tömbben
    for (int i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }

    // Kimeneti tömb építése
    for (int i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }

    // Kimenet másolása az eredeti tömbbe
    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }
}

// LSD Radix Sort fő függvénye
void radixSort(std::vector<int>& arr) {
    int m = getMax(arr);

    // Szorzat (exp) növelése 10-szeresével minden számjegyért
    for (int exp = 1; m / exp > 0; exp *= 10) {
        countingSort(arr, exp);
    }
}

int main() {
    std::vector<int> accountNumbers = {170, 45, 75, 90, 802, 24, 2, 66};
    radixSort(accountNumbers);
    for (int num : accountNumbers) {
        std::cout << num << " ";
    }
    return 0;
}
```

#### Karakterláncok Rendezése

A Radix Sort nem csak numerikus adatokra alkalmazható, hanem szöveges adatok rendezésére is, különösen ha a szöveges adatok fix vagy változó hosszúságúak. Az MSD Radix Sort különösen hatékony változó hosszúságú szöveges adatok esetén.

##### Példa: Szavak Rendezése

Képzeljük el, hogy egy szótárban lévő szavakat szeretnénk rendezni ábécé sorrendbe. Az alábbi C++ példa az MSD Radix Sort használatát mutatja be szavak rendezésére.

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

// Karakter pozíciójának visszaadása
int charAt(const std::string& s, int d) {
    if (d < s.length()) return s[d];
    else return -1; // Sentinel érték a rövidebb sztringekhez
}

// MSD Radix Sort segédfüggvény
void msdSort(std::vector<std::string>& arr, int lo, int hi, int d) {
    if (hi <= lo) return;

    const int R = 256; // ASCII karakterek száma
    std::vector<int> count(R + 2, 0);
    std::vector<std::string> aux(hi - lo + 1);

    // Számlálás
    for (int i = lo; i <= hi; i++) {
        count[charAt(arr[i], d) + 2]++;
    }

    // Gyűjtés
    for (int r = 0; r < R + 1; r++) {
        count[r + 1] += count[r];
    }

    // Átrendezés
    for (int i = lo; i <= hi; i++) {
        aux[count[charAt(arr[i], d) + 1]++] = arr[i];
    }

    // Másolás vissza
    for (int i = lo; i <= hi; i++) {
        arr[i] = aux[i - lo];
    }

    // Rekurzív rendezés
    for (int r = 0; r < R; r++) {
        msdSort(arr, lo + count[r], lo + count[r + 1] - 1, d + 1);
    }
}

// MSD Radix Sort fő függvénye
void radixSortMSD(std::vector<std::string>& arr) {
    msdSort(arr, 0, arr.size() - 1, 0);
}

int main() {
    std::vector<std::string> words = {"she", "sells", "seashells", "by", "the", "seashore"};
    radixSortMSD(words);
    for (const std::string& str : words) {
        std::cout << str << " ";
    }
    return 0;
}
```

#### Nagy Mennyiségű Adat Rendezése

A Radix Sort különösen hasznos nagy mennyiségű adat rendezésénél, mivel az időbeli komplexitása $O(d \cdot (n + k))$ általában kedvezőbb, mint a hagyományos komparatív rendezési algoritmusoké, különösen fix hosszúságú kulcsok esetén.

##### Példa: Adatbázisok Rekordjainak Rendezése

Egy adatbázisban lévő rekordok gyakran több mezőből állnak, és a rendezés egyik kulcsfontosságú művelet lehet a gyors keresés és lekérdezés érdekében. Például, ha egy adatbázisban az alkalmazottak adatait szeretnénk rendezni azonosítójuk vagy nevük alapján, a Radix Sort hatékonyan használható.

#### Képfeldolgozás

A Radix Sort alkalmazható képfeldolgozásban is, ahol a képek pixeleit intenzitásuk alapján kell rendezni. Ez hasznos lehet például a hisztogram kiegyenlítésében, amely egy gyakori előfeldolgozási lépés a képfeldolgozásban.

#### Telekommunikáció és IP-címek Rendezése

A Radix Sort különösen hatékony IP-címek rendezésénél, amelyek fix hosszúságúak és numerikus formátumúak. Az IP-címek gyors rendezése fontos lehet különböző hálózati alkalmazásoknál, mint például a forgalomirányítás vagy a naplózási rendszerek.

#### Radix Sort és Parallelizmus

A Radix Sort jól illeszkedik a párhuzamos feldolgozáshoz is. Mivel az egyes helyiértékek alapján végzett rendezés független egymástól, az egyes helyiértékek feldolgozása párhuzamosítható. Ez különösen hasznos lehet nagy adathalmazok esetén, ahol a párhuzamos feldolgozás jelentős teljesítményjavulást eredményezhet.

##### Párhuzamos Radix Sort Példa

A párhuzamos Radix Sort implementációja különféle párhuzamos programozási technikákkal valósítható meg, például a C++11-ben bevezetett `std::thread` vagy a modernebb párhuzamos programozási könyvtárak segítségével. Az alábbi példa egy egyszerű illusztrációt mutat be a párhuzamos feldolgozásra C++-ban.

```cpp
#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>

void countingSortParallel(std::vector<int>& arr, int exp, int start, int end) {
    int n = end - start;
    std::vector<int> output(n);
    int count[10] = {0};

    // Számjegyek gyakoriságának számlálása
    for (int i = start; i < end; i++) {
        count[(arr[i] / exp) % 10]++;
    }

    // Count[i] most már tartalmazza a helyes pozíciókat a kimeneti tömbben
    for (int i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }

    // Kimeneti tömb építése
    for (int i = end - 1; i >= start; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }

    // Kimenet másolása az eredeti tömbbe
    for (int i = start; i < end; i++) {
        arr[i] = output[i - start];
    }
}

void radixSortParallel(std::vector<int>& arr) {
    int m = *std::max_element(arr.begin(), arr.end());
    int n = arr.size();
    int numThreads = std::thread::hardware_concurrency();
    int chunkSize = (n + numThreads - 1) / numThreads;
    std::vector<std::thread> threads;

    for (int exp = 1; m / exp > 0; exp *= 10) {
        threads.clear();
        for (int i = 0; i < numThreads; i++) {
            int start = i * chunkSize;
            int end = std::min(start + chunkSize, n);
            if (start < n) {
                threads.emplace_back(countingSortParallel, std::ref(arr), exp, start, end);
            }
        }
        for (auto& t : threads) {
            t.join();
        }
    }
}

int main() {
    std::vector<int> arr = {170, 45, 75, 90, 802, 24, 2, 66};
    radixSortParallel(arr);
    for (int num : arr) {
        std::cout << num << " ";
    }
    return 0;
}
```

#### Összefoglalás

A Radix Sort egy rendkívül hatékony és sokoldalú rendezési algoritmus, amely különböző gyakorlati alkalmazásokban használható. Az algoritmus különösen jól teljesít numerikus adatok, karakterláncok és nagy mennyiségű adat rendezésénél. A párhuzamos feldolgozásra való alkalmassága további előnyt jelent nagy adathalmazok rendezésénél. A fejezetben bemutatott példák és alkalmazások szemléltetik a Radix Sort széleskörű felhasználási lehetőségeit és hatékonyságát.

