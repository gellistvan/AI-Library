\newpage

## 3.4. Leghosszabb növekvő részsorozat (Longest Increasing Subsequence)

A leghosszabb növekvő részsorozat (LIS) problémája egy alapvető feladat a számítástudomány és az algoritmusok területén. Lényege, hogy egy adott számsorozatban meg kell találni a leghosszabb olyan részsorozatot, amelyben az elemek növekvő sorrendben követik egymást. Ez a probléma számos alkalmazási területen felmerül, például a bioinformatikában, ahol génszekvenciák elemzésére használják, vagy a pénzügyi elemzések során, ahol piaci trendek azonosítására szolgál. A LIS probléma megoldása különböző megközelítéseket igényelhet, beleértve a rekurzív algoritmusokat és a dinamikus programozást. Ebben a fejezetben bemutatjuk a LIS problémájának definícióját, alkalmazásait, majd részletesen ismertetjük a megoldás rekurzív és dinamikus programozási módszereit, beleértve az algoritmusok implementációját is.

### 3.4.1. Definíció és alkalmazások

#### Definíció

A leghosszabb növekvő részsorozat (Longest Increasing Subsequence, LIS) egy alapvető probléma a kombinatorikus optimalizálás területén. Tekintsünk egy $A = [a_1, a_2, \ldots, a_n]$ véges hosszúságú számsorozatot, ahol $a_i \in \mathbb{R}$. Egy $S = [a_{i_1}, a_{i_2}, \ldots, a_{i_k}]$ sorozat az $A$ részsorozata, ha $1 \leq i_1 < i_2 < \ldots < i_k \leq n$. Az $S$ sorozat növekvő, ha minden $1 \leq j < k$ esetén $a_{i_j} < a_{i_{j+1}}$. A LIS probléma célja megtalálni az $A$ leghosszabb olyan részsorozatát, amely növekvő.

Formálisan, a LIS probléma a következőképpen írható le:

**Bemenet:** Egy $A = [a_1, a_2, \ldots, a_n]$ számsorozat.
**Kimenet:** Egy leghosszabb $S = [a_{i_1}, a_{i_2}, \ldots, a_{i_k}]$ részsorozat, amely növekvő.

#### Példák

Például, ha az input sorozat $A = [10, 22, 9, 33, 21, 50, 41, 60, 80]$, akkor a leghosszabb növekvő részsorozat az $[10, 22, 33, 50, 60, 80]$, amelynek hossza 6. Megjegyzendő, hogy a leghosszabb növekvő részsorozat nem feltétlenül egyértelmű; egy másik lehetséges LIS lehetne $[10, 22, 33, 41, 60, 80]$.

#### Matematikai Megfogalmazás

Tekintsük az $LIS(A)$ függvényt, amely megadja a $A$ sorozat leghosszabb növekvő részsorozatának hosszát. A probléma megoldása során olyan $S$ sorozatot keresünk, amely maximalizálja a hosszúságát, azaz:

$$
LIS(A) = \max\{|S| \mid S \text{ növekvő részsorozat } A \text{-ban}\}
$$

#### Alkalmazások

A LIS probléma számos gyakorlati alkalmazással bír különböző területeken:

1. **Bioinformatika:**
    - **Génszekvenciák Elemzése:** A génszekvenciák elemzésében gyakran szükség van arra, hogy megtaláljuk a leghosszabb növekvő részsorozatokat, amelyek segítenek azonosítani bizonyos mintákat vagy evolúciós kapcsolódásokat a DNS szekvenciákban.
    - **Fehérjeszerkezet:** A fehérjék aminosav-sorrendjeinek vizsgálata során is hasznos lehet a LIS, például a fehérjék hajtogatódási folyamatainak megértésében.

2. **Pénzügyi Elemzés:**
    - **Piaci Trendek:** A részvényárfolyamok vagy más pénzügyi adatok vizsgálatakor a LIS segíthet azonosítani a hosszú távú növekvő trendeket, amelyek alapul szolgálhatnak befektetési döntésekhez.
    - **Gazdasági Adatok:** Gazdasági idősorok elemzésében, például GDP vagy inflációs adatok esetén, a LIS alkalmazható a hosszú távú gazdasági növekedés trendjeinek azonosítására.

3. **Számítógépes Vízió:**
    - **Objektumkövetés:** A képfeldolgozás és videóelemzés területén a LIS használható objektumok mozgásának követésére, ahol az objektumok pozícióinak időbeli változását növekvő sorozatként modellezhetjük.
    - **Formafelismerés:** Az alakzatok és minták felismerése során a LIS segíthet a képeken vagy videókon megjelenő objektumok szerkezetének azonosításában.

4. **Adatbányászat:**
    - **Sorozatok Klaszterezése:** Az adatbányászatban és gépi tanulásban a LIS alkalmazható időbeli adatsorok klaszterezésére, ahol a hasonló növekvő mintázatokkal rendelkező sorozatok csoportosíthatók.
    - **Jellemzők Kivonása:** A hosszú növekvő részsorozatok azonosítása segíthet az adatok jellemzőinek kivonásában, amelyeket később különböző prediktív modellekben használhatunk.

#### Matematikai Alkalmazások

1. **Számelmélet:**
    - **Rendteremtési Problémák:** A LIS kapcsolódik olyan számelméleti problémákhoz, mint például a rendteremtési problémák, ahol egy adott permutációban a leghosszabb rendezett részsorozatot keressük.
    - **Young-diagramok:** A LIS hasznos eszköz a Young-diagramokkal kapcsolatos kutatásokban, amelyek fontos szerepet játszanak a kombinatorikus optimalizálásban és a szimmetrikus csoportok elméletében.

2. **Kombinatorikus Optimalizálás:**
    - **Gyártási Ütemezés:** A gyártási folyamatok optimalizálásában a LIS segíthet a legoptimálisabb termelési sorrendek meghatározásában, amely minimalizálja az átfutási időt és maximalizálja a termelékenységet.
    - **Hálózati Tervezés:** A hálózati rendszerek optimalizálásában a LIS alkalmazható a leghatékonyabb útvonalak vagy kapcsolatok azonosítására, amelyek maximalizálják a hálózat teljesítményét.

A LIS probléma megoldása számos algoritmikus megközelítést igényelhet, amelyeket a következő alfejezetekben részletesen ismertetünk, beleértve a rekurzív és a dinamikus programozási módszereket is. 

### 3.4.2. Megoldás rekurzióval

A leghosszabb növekvő részsorozat (LIS) problémájának rekurzív megközelítése egy egyszerű, mégis hatékony módja a probléma megoldásának, különösen kisebb méretű bemenetek esetén. A rekurzió alapötlete, hogy a probléma megoldását kisebb részproblémákra bontjuk, és ezek megoldásával érjük el a teljes probléma megoldását.

#### Rekurzív Megközelítés Alapjai

A rekurzív megközelítés során a LIS probléma úgy oldható meg, hogy minden egyes elemre meghatározzuk a leghosszabb növekvő részsorozat hosszát, amely az adott elemmel végződik. Az általános gondolatmenet a következő:

1. **Alapötlet:**
   - Minden egyes elemre kiszámítjuk, hogy milyen hosszú a leghosszabb növekvő részsorozat, amely az adott elemmel végződik.
   - Az egész sorozatra nézve a LIS a legnagyobb ilyen értékek közül kerül ki.

2. **Rekurzív Függvény:**
   - Legyen $L(i)$ a leghosszabb növekvő részsorozat hossza, amely az $i$-edik elemmel végződik.
   - Ahhoz, hogy $L(i)$-t kiszámítsuk, minden $j < i$ elemre meghatározzuk $L(j)$-t, és ha $a[j] < a[i]$, akkor $L(i) = \max(L(i), L(j) + 1)$.

3. **Alap eset:**
   - Minden egyes elemnél a legrövidebb LIS hossz legalább 1, mivel az elem önmagában is egy részsorozat.

#### Rekurzív Algoritmus

Az alábbiakban részletesen bemutatjuk a rekurzív algoritmus működését, amely a fent említett elvek alapján működik.

**1. lépés: A függvény meghatározása**

A leghosszabb növekvő részsorozat hosszának meghatározása érdekében definiáljunk egy segédfüggvényt, amely kiszámítja a LIS hosszát egy adott elemre nézve. Ezt a függvényt nevezzük `lis_ending_at`-nak.

```cpp
int lis_ending_at(const std::vector<int>& arr, int i) {
    // Base case: LIS ending at the first element is 1
    if (i == 0) {
        return 1;
    }

    // Initialize the maximum length of LIS ending at i
    int max_len = 1;

    // Check all previous elements
    for (int j = 0; j < i; j++) {
        // If arr[j] is less than arr[i], it can be part of LIS ending at i
        if (arr[j] < arr[i]) {
            // Recursively find the LIS ending at j and update max_len
            max_len = std::max(max_len, lis_ending_at(arr, j) + 1);
        }
    }

    return max_len;
}
```

**2. lépés: A teljes LIS hosszának meghatározása**

Miután definiáltuk a segédfüggvényt, használhatjuk azt a teljes sorozat LIS hosszának meghatározására.

```cpp
int lis(const std::vector<int>& arr) {
    int n = arr.size();
    int max_len = 1;

    // Check LIS ending at each element
    for (int i = 0; i < n; i++) {
        max_len = std::max(max_len, lis_ending_at(arr, i));
    }

    return max_len;
}
```

**3. lépés: A teljes algoritmus**

A teljes algoritmus a fenti két részből áll össze. A `lis` függvény hívja meg a `lis_ending_at` segédfüggvényt minden egyes elemre, és meghatározza a leghosszabb növekvő részsorozat hosszát az egész sorozatban.

```cpp
#include <iostream>

#include <vector>
#include <algorithm>

// Helper function to find LIS ending at index i
int lis_ending_at(const std::vector<int>& arr, int i) {
    if (i == 0) {
        return 1;
    }

    int max_len = 1;

    for (int j = 0; j < i; j++) {
        if (arr[j] < arr[i]) {
            max_len = std::max(max_len, lis_ending_at(arr, j) + 1);
        }
    }

    return max_len;
}

// Function to find the length of LIS in arr
int lis(const std::vector<int>& arr) {
    int n = arr.size();
    int max_len = 1;

    for (int i = 0; i < n; i++) {
        max_len = std::max(max_len, lis_ending_at(arr, i));
    }

    return max_len;
}

int main() {
    std::vector<int> arr = {10, 22, 9, 33, 21, 50, 41, 60, 80};
    std::cout << "Length of LIS is " << lis(arr) << std::endl;
    return 0;
}
```

#### Rekurzív Megoldás Elemzése

A rekurzív megoldás érthető és könnyen megvalósítható, de több szempontból is figyelmet igényel, különösen nagyobb méretű bemenetek esetén.

1. **Időbeli Komplexitás:**
   - A rekurzív megközelítés időbeli komplexitása a visszahívások száma miatt exponenciális lehet. Az egyes elemekre történő visszahívások száma miatt az algoritmus időbeli komplexitása $O(2^n)$ lehet, ahol $n$ a sorozat hossza.

2. **Helybeli Komplexitás:**
   - A rekurzív algoritmus helybeli komplexitása $O(n)$ a veremmemória használata miatt, ahol $n$ a sorozat hossza. Minden rekurzív hívás újabb szintet ad a veremhez, ami helyigényes lehet nagyobb bemenetek esetén.

3. **Optimalizációs Lehetőségek:**
   - A rekurzív megoldás hatékonysága javítható memoizációval, amely során a korábban kiszámított eredményeket tároljuk és újra felhasználjuk. Ez jelentősen csökkentheti az ismétlődő számítások számát, és ezáltal javíthatja az algoritmus időbeli komplexitását.

#### Rekurzív Megoldás Memoizációval

A memoizáció egy olyan technika, amely során a korábban kiszámított eredményeket tároljuk egy adatstruktúrában (pl. vektorban vagy térképen), és újra felhasználjuk azokat a későbbi hívások során. Ezzel jelentősen csökkenthetjük az ismétlődő számítások számát és javíthatjuk az algoritmus hatékonyságát.

Az alábbiakban bemutatjuk a memoizációval kiegészített rekurzív megoldást:

```cpp
#include <iostream>

#include <vector>
#include <algorithm>

// Helper function to find LIS ending at index i with memoization
int lis_ending_at(const std::vector<int>& arr, int i, std::vector<int>& memo) {
    if (memo[i] != -1) {
        return memo[i];
    }

    int max_len = 1;

    for (int j = 0; j < i; j++) {
        if (arr[j] < arr[i]) {
            max_len = std::max(max_len, lis_ending_at(arr, j, memo) + 1);
        }
    }

    memo[i] = max_len;
    return memo[i];
}

// Function to find the length of LIS in arr with memoization
int lis(const std::vector<int>& arr) {
    int n = arr.size();
    std::vector<int> memo(n, -1);

    int max_len = 1;

    for (int i = 0; i < n; i++) {
        max_len = std::max(max_len, lis_ending_at(arr, i, memo));
    }

    return max_len;
}

int main() {
    std::vector<int> arr = {10, 22, 9, 33, 21, 50, 41, 60, 80};
    std::cout << "Length of LIS is " << lis(arr) << std::endl;
    return 0;
}
```

A memoizációval kiegészített rekurzív megoldás időbeli komplexitása $O(n^2)$, mivel minden egyes elemre egyszer kiszámítjuk a LIS hosszát, és az eredményeket újra felhasználjuk. Ez jelentős javulás a tisztán rekurzív megoldás exponenciális időbeli komplexitásához képest.

#### Összegzés

A rekurzív megközelítés egy intuitív és egyszerű módja a leghosszabb növekvő részsorozat problémájának megoldására, különösen kisebb méretű bemenetek esetén. Azonban nagyobb bemenetek esetén a tisztán rekurzív megoldás nem hatékony az exponenciális időbeli komplexitás miatt. A memoizáció alkalmazásával jelentősen javíthatjuk az algoritmus hatékonyságát, csökkentve az időbeli komplexitást $O(n^2)$-re. A következő alfejezetben bemutatjuk a dinamikus programozási megközelítést, amely további optimalizálási lehetőségeket kínál a LIS probléma megoldására.

### 3.4.3. Megoldás dinamikus programozással

A dinamikus programozás egy hatékony technika, amely számos optimalizálási problémára alkalmazható, különösen akkor, ha a probléma kisebb részproblémákra bontható, amelyek megoldásai újra felhasználhatók. A leghosszabb növekvő részsorozat (LIS) problémája egy tipikus példa arra, ahol a dinamikus programozás jelentős hatékonyságjavulást eredményezhet a rekurzív megközelítésekhez képest.

#### Alapötlet

A dinamikus programozás alapja az, hogy a problémát kisebb, átfedő részproblémákra bontjuk, és azokat egy táblázatban (vagy tömbben) tároljuk, hogy elkerüljük az ismétlődő számításokat. A LIS esetében a következő megközelítést alkalmazzuk:

1. Definiáljuk a `dp[i]` tömböt, ahol `dp[i]` a leghosszabb növekvő részsorozat hossza, amely az `i`-edik elemmel végződik.
2. Az `i`-edik elemre nézve a leghosszabb növekvő részsorozat hosszát úgy határozzuk meg, hogy megnézzük az összes korábbi elemet (j < i), és ha azok kisebbek, mint az `i`-edik elem, akkor frissítjük `dp[i]` értékét.

#### Algoritmus Leírása

A dinamikus programozási algoritmus a következő lépésekből áll:

1. Inicializáljuk a `dp` tömböt, ahol minden elem értéke kezdetben 1, mert minden egyes elem önmagában egy 1 hosszúságú növekvő részsorozatot alkot.
2. Iteráljunk végig a tömbön, és minden egyes elemre (i) nézve iteráljunk végig az összes korábbi elemen (j < i).
3. Ha az `arr[j] < arr[i]`, akkor frissítsük a `dp[i]` értékét a következőképpen: `dp[i] = max(dp[i], dp[j] + 1)`.
4. Végül a leghosszabb növekvő részsorozat hossza a `dp` tömb maximális értéke lesz.

#### Részletes Példa

Tekintsük az alábbi bemeneti sorozatot: `A = [10, 22, 9, 33, 21, 50, 41, 60, 80]`.

1. Inicializálás:

   ```
   dp = [1, 1, 1, 1, 1, 1, 1, 1, 1]
   ```

2. Iterálás és frissítés:

   - Az első elem (10) esetén nincs korábbi elem, így `dp[0] = 1`.
   - A második elem (22) esetén:
      - 10 < 22, így `dp[1] = max(dp[1], dp[0] + 1) = 2`.
   - A harmadik elem (9) esetén:
      - Nincs kisebb elem előtte, így `dp[2] = 1`.
   - A negyedik elem (33) esetén:
      - 10 < 33, így `dp[3] = max(dp[3], dp[0] + 1) = 2`.
      - 22 < 33, így `dp[3] = max(dp[3], dp[1] + 1) = 3`.
      - 9 < 33, így `dp[3] = max(dp[3], dp[2] + 1) = 3`.
   - És így tovább minden egyes elemre.

3. Az iteráció végén a `dp` tömb a következő lesz:

   ```
   dp = [1, 2, 1, 3, 2, 4, 4, 5, 6]
   ```

4. A leghosszabb növekvő részsorozat hossza a `dp` tömb maximális értéke, azaz 6.

#### Algoritmus Implementáció C++ Nyelven

Az alábbiakban bemutatjuk a dinamikus programozási megközelítés C++ nyelvű implementációját:

```cpp
#include <iostream>

#include <vector>
#include <algorithm>

// Function to find the length of the longest increasing subsequence
int lis(const std::vector<int>& arr) {
    int n = arr.size();
    if (n == 0) return 0;

    // Create a dp array to store the length of LIS ending at each index
    std::vector<int> dp(n, 1);

    // Fill the dp array according to the dp relation
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (arr[j] < arr[i]) {
                dp[i] = std::max(dp[i], dp[j] + 1);
            }
        }
    }

    // The length of the longest increasing subsequence is the maximum value in dp array
    return *std::max_element(dp.begin(), dp.end());
}

int main() {
    std::vector<int> arr = {10, 22, 9, 33, 21, 50, 41, 60, 80};
    std::cout << "Length of LIS is " << lis(arr) << std::endl;
    return 0;
}
```

#### Elemzés

1. **Időbeli Komplexitás:**
   - Az algoritmus két beágyazott ciklust használ az `i` és `j` indexekhez, ami $O(n^2)$ időbeli komplexitást eredményez, ahol $n$ a bemeneti sorozat hossza.

2. **Helybeli Komplexitás:**
   - A `dp` tömb használata $O(n)$ helybeli komplexitást eredményez, mivel a `dp` tömb mérete megegyezik a bemeneti sorozat hosszával.

3. **Hatékonyság:**
   - A dinamikus programozási megközelítés jelentős hatékonyságjavulást eredményez a tisztán rekurzív megoldáshoz képest, mivel elkerüli az ismétlődő számításokat és kihasználja a részproblémák megoldásait.

#### Optimalizálási Lehetőségek

A dinamikus programozási megközelítés további optimalizálása is lehetséges, például a bináris keresés alkalmazásával. Egy másik megközelítés a "patience sorting" technika, amely $O(n \log n)$ időbeli komplexitást eredményez.

**Patience Sorting Megközelítés:**

Ez a megközelítés a "patience sorting" technikán alapul, ahol a kártyák rendezéséhez hasonlóan járunk el, és a leghosszabb növekvő részsorozat hosszát egy bináris keresési technikával határozzuk meg.

1. **Kártyapakli Analógia:**
   - Minden egyes elemnél keressük meg a megfelelő helyet egy rendezett kártyapakliban.
   - Ha az elem nagyobb, mint a kártyapakli tetején lévő kártyák, akkor új kártyapaklit kezdünk.
   - Ellenkező esetben helyezzük az elemet a megfelelő kártyapaklira.

2. **Bináris Keresés:**
   - Az elem megfelelő helyének megtalálásához bináris keresést használunk, ami $O(\log n)$ időbeli komplexitást eredményez minden egyes elemre nézve.

Az alábbiakban bemutatjuk a "patience sorting" technikán alapuló megoldást:

```cpp
#include <iostream>

#include <vector>
#include <algorithm>

// Function to find the length of the longest increasing subsequence using patience sorting
int lis(const std::vector<int>& arr) {
    if (arr.empty()) return 0;

    std::vector<int> piles;

    for (int num : arr) {
        auto it = std::lower_bound(piles.begin(), piles.end(), num);
        if (it == piles.end()) {
            piles.push_back(num);
        } else {
            *it = num;
        }
    }

    return piles.size();
}

int main() {
    std::vector<int> arr = {10, 22, 9, 33, 21, 50, 41, 60, 80};
    std::cout << "Length of LIS is " << lis(arr) << std::endl;
    return 0;
}
```

#### Összegzés

A dinamikus programozás hatékony megoldást nyújt a leghosszabb növekvő részsorozat problémájára, különösen nagyobb méretű bemenetek esetén. A $O(n^2)$ időbeli komplexitású algoritmus jelentős javulást eredményez a rekurzív megközelítéshez képest, és további optimalizálásokkal, mint például a "patience sorting" technika, tovább csökkenthető az időbeli komplexitás $O(n \log n)$-re. A dinamikus programozás ereje abban rejlik, hogy képes hatékonyan újra felhasználni a korábban kiszámított részproblémák eredményeit, így minimalizálva az ismétlődő számításokat és maximalizálva a hatékonyságot.