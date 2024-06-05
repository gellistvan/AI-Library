\newpage

## 2.8. Számláló rendezés

A számláló rendezés (Counting Sort) egy hatékony és egyszerű rendezési algoritmus, amely ideális bizonyos típusú adathalmazok gyors rendezésére. Ez az algoritmus különösen akkor hasznos, ha az adatok szűk tartományban helyezkednek el, mivel kihasználja az elemek értékeinek gyakorisági eloszlását. A számláló rendezés alapelvei és implementációja eltér a hagyományos összehasonlítás alapú rendezési módszerektől, mint például a gyorsrendezés vagy a buborékrendezés, mivel közvetlenül az elemek értékeit használja a rendezéshez. Ebben a fejezetben bemutatjuk a számláló rendezés működési elvét, lépésről lépésre végigvezetünk az algoritmus implementációján, megvizsgáljuk a módszer korlátait és alkalmazhatóságát különböző kontextusokban, valamint elemezzük a teljesítményét és komplexitását. Végül gyakorlati példákon keresztül mutatjuk be, hogyan alkalmazható a számláló rendezés valós problémák megoldására.

### 2.8.1. Alapelvek és implementáció

A számláló rendezés (Counting Sort) egy olyan rendezési algoritmus, amely az elemek értékeit felhasználva rendezi azokat. Ez az algoritmus különösen hatékony, ha az elemek értékei viszonylag kis tartományban helyezkednek el. Az alábbiakban részletesen bemutatjuk a számláló rendezés alapelveit és annak implementációját.

#### Alapelvek

A számláló rendezés nem használ összehasonlítást a rendezés során, ezért nem tartozik a klasszikus összehasonlítás alapú rendezési algoritmusok közé. Ehelyett az algoritmus az alábbi lépéseket követi:

1. **Gyakorisági számítás:** Az algoritmus először megszámolja, hogy az egyes értékek hányszor fordulnak elő az input tömbben. Ehhez egy segédtömböt használ, amelynek mérete az értékek tartományától függ.
2. **Felhalmozott gyakoriság:** A gyakorisági tömb segítségével kiszámítja az elemek felhalmozott gyakoriságát, ami azt jelenti, hogy minden értéknél tárolja az addig előforduló elemek számát. Ez lehetővé teszi az elemek végső pozíciójának meghatározását a rendezett tömbben.
3. **Elemek rendezése:** Végül az algoritmus az eredeti tömb elemeit a felhalmozott gyakorisági tömb alapján helyezi el a megfelelő pozícióba egy új, rendezett tömbben.

#### Az algoritmus lépései

1. **Gyakorisági tömb létrehozása:**
   Az algoritmus egy `count` nevű tömböt hoz létre, amelynek mérete a lehetséges értékek tartományától függ. Ebben a tömbben minden index az adott érték előfordulásainak számát tartalmazza.

2. **Felhalmozott gyakoriság kiszámítása:**
   A `count` tömb segítségével kiszámítjuk az elemek felhalmozott gyakoriságát. Ez a lépés biztosítja, hogy minden érték végső pozícióját meghatározhassuk a rendezett tömbben.

3. **Elemek elhelyezése a rendezett tömbben:**
   Az eredeti tömb elemeit végigjárva, azokat a felhalmozott gyakorisági tömb alapján helyezzük el egy új, rendezett tömbben.

#### Az algoritmus implementációja

Az alábbiakban egy C++ nyelvű példa található, amely bemutatja a számláló rendezés implementációját:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

// A segédfüggvény a maximum érték meghatározásához
int findMax(const std::vector<int>& arr) {
    return *std::max_element(arr.begin(), arr.end());
}

void countingSort(std::vector<int>& arr) {
    int max_val = findMax(arr); // A legnagyobb elem meghatározása
    std::vector<int> count(max_val + 1, 0); // Gyakorisági tömb létrehozása

    // Gyakorisági tömb feltöltése
    for (int num : arr) {
        count[num]++;
    }

    // Felhalmozott gyakorisági tömb létrehozása
    for (int i = 1; i <= max_val; i++) {
        count[i] += count[i - 1];
    }

    // Rendezett tömb létrehozása
    std::vector<int> output(arr.size());
    for (int i = arr.size() - 1; i >= 0; i--) {
        output[count[arr[i]] - 1] = arr[i];
        count[arr[i]]--;
    }

    // Az eredeti tömb frissítése a rendezett elemekkel
    arr = output;
}

int main() {
    std::vector<int> arr = {4, 2, 2, 8, 3, 3, 1};
    countingSort(arr);

    std::cout << "Rendezett tömb: ";
    for (int num : arr) {
        std::cout << num << " ";
    }
    return 0;
}
```

#### Az algoritmus elemzése

**Időkomplexitás:**
A számláló rendezés időkomplexitása O(n + k), ahol n az elemek száma, k pedig az értékek tartománya. Az algoritmus három fő lépést hajt végre:
1. A gyakorisági tömb feltöltése, amely O(n) időt vesz igénybe.
2. A felhalmozott gyakorisági tömb létrehozása, amely O(k) időt igényel.
3. Az elemek elhelyezése a rendezett tömbben, amely O(n) időt vesz igénybe.

Mivel ezek a lépések egymás után hajtódnak végre, az algoritmus összesített időkomplexitása O(n + k).

**Térkomplexitás:**
A számláló rendezés térkomplexitása O(n + k). Az algoritmusnak szüksége van egy `count` tömbre, amely a tartomány méretétől függ (O(k)), valamint egy kimeneti tömbre, amely az elemek számától függ (O(n)). Így a teljes térkomplexitás O(n + k).

**Előnyök és hátrányok:**
A számláló rendezés egyik fő előnye, hogy nagyon gyors, ha az értékek tartománya viszonylag kicsi. Továbbá, stabil rendezési algoritmus, azaz a hasonló értékek eredeti sorrendje megmarad a rendezett tömbben. Azonban az algoritmus nem hatékony, ha az értékek tartománya nagyon nagy, mivel ilyenkor a szükséges memória mennyisége jelentősen megnő.

#### Példa a működésre

Tegyük fel, hogy az alábbi tömböt szeretnénk rendezni:

```
Input: [4, 2, 2, 8, 3, 3, 1]
```

1. **Gyakorisági tömb létrehozása:**

```
Érték:    0  1  2  3  4  5  6  7  8
Gyakoriság: [0, 1, 2, 2, 1, 0, 0, 0, 1]
```

2. **Felhalmozott gyakorisági tömb:**

```
Érték:    0  1  2  3  4  5  6  7  8
Felhalmozott: [0, 1, 3, 5, 6, 6, 6, 6, 7]
```

3. **Elemek elhelyezése a rendezett tömbben:**

Az eredeti tömb elemeit a felhalmozott gyakorisági tömb alapján helyezzük el a rendezett tömbben:

```
Output: [1, 2, 2, 3, 3, 4, 8]
```

Ez a részletes bemutatás remélhetőleg segít megérteni a számláló rendezés alapelveit és implementációját, valamint az algoritmus hatékonyságának és korlátainak tudományos elemzését.


### 2.8.2. Korlátozások és alkalmazhatóság

A számláló rendezés (Counting Sort) egy hatékony rendezési algoritmus, amely bizonyos feltételek mellett kiemelkedően jól teljesít. Azonban, mint minden algoritmusnak, ennek is megvannak a maga korlátai és specifikus alkalmazási területei. Ebben a fejezetben részletesen tárgyaljuk a számláló rendezés korlátait, valamint azt, hogy mely helyzetekben érdemes ezt az algoritmust alkalmazni.

#### Korlátozások

1. **Értéktartomány:**
   A számláló rendezés hatékonysága nagymértékben függ az elemek értéktartományától. Ha a tartomány nagyon nagy, az algoritmus memóriaigénye jelentősen megnő, mivel a `count` tömb mérete közvetlenül az értékek maximális és minimális értéke közötti különbségtől függ. Például, ha az elemek értékei 1 és 1,000,000 között vannak, akkor egy 1,000,000 elemet tartalmazó `count` tömbre van szükség, ami jelentős memóriafogyasztással jár.

2. **Adat típusok:**
   A számláló rendezés csak egész számok vagy olyan értékek rendezésére alkalmas, amelyek egy diszkrét, véges tartományba esnek. Lebegőpontos számok, szövegek vagy más összetett típusok rendezése nem lehetséges ezzel az algoritmussal anélkül, hogy azokat először egész számokká alakítanánk. Ez az átalakítás azonban bonyolult és nem mindig praktikus.

3. **Stabilitás fenntartása:**
   Bár a számláló rendezés stabil algoritmus, azaz megőrzi az egyenlő elemek eredeti sorrendjét, a stabilitás fenntartása érdekében a kimeneti tömbbe történő elemek elhelyezése során gondosan kell eljárni. Az algoritmus implementációjának bonyolultsága növekedhet, ha a stabilitást biztosítani kell, különösen nagy adathalmazok esetén.

4. **Nem alkalmas online rendezésre:**
   A számláló rendezés nem alkalmas online rendezésre, ami azt jelenti, hogy nem képes dinamikusan kezelni az adathalmazhoz hozzáadott új elemeket anélkül, hogy teljesen újra ne rendezné az egész halmazt. Az algoritmus teljes adathalmazt igényel a kezdeti fázisban, így nem használható olyan alkalmazásokban, ahol az adatok folyamatosan érkeznek és azonnal rendezni kell őket.

#### Alkalmazhatóság

1. **Kis értéktartományú adatok:**
   A számláló rendezés különösen jól alkalmazható olyan helyzetekben, ahol az adatok értéktartománya viszonylag kicsi. Például egy vizsga pontszámok rendezése esetén, ahol a pontszámok 0 és 100 között mozognak, a számláló rendezés rendkívül hatékony lehet. Az algoritmus gyorsasága és egyszerűsége ilyenkor jól kihasználható.

2. **Pozitív egész számok:**
   Olyan adathalmazok rendezése, ahol az elemek pozitív egész számok, ideális eset a számláló rendezés számára. Az algoritmus ilyenkor könnyen alkalmazható, és a memóriaigénye is kezelhető marad.

3. **Speciális alkalmazások:**
   A számláló rendezést gyakran használják speciális alkalmazásokban, például histogram készítésében, ahol az adatok gyakorisági eloszlását kell gyorsan meghatározni. Továbbá, a számláló rendezés alapötlete más algoritmusok, például a radix sort alapjául is szolgálhat, ahol az egyes lépésekben a számláló rendezés technikáját alkalmazzák.

4. **Valós idejű rendszerek:**
   Valós idejű rendszerekben, ahol a gyorsaság kulcsfontosságú, a számláló rendezés előnyös lehet, feltéve, hogy az értéktartomány korlátozott. Például beágyazott rendszerekben vagy valós idejű adatfeldolgozási feladatokban, ahol a késleltetés minimalizálása kritikus, a számláló rendezés gyors rendezési képessége kihasználható.

#### Összefoglalás

A számláló rendezés egy hatékony és egyszerű algoritmus, amely bizonyos feltételek mellett kiemelkedően jól teljesít. Az algoritmus legnagyobb előnye a gyorsasága és egyszerűsége, ha az adatok értéktartománya kicsi és az elemek pozitív egész számok. Ugyanakkor a számláló rendezés alkalmazhatósága korlátozott, ha az értéktartomány nagy, az adatok nem egész számok, vagy dinamikus rendezésre van szükség. Az algoritmus alkalmazása előtt mindig mérlegelni kell az adott probléma sajátosságait és a számláló rendezés korlátait. Ennek a mérlegelésnek az eredményeként lehet meghatározni, hogy a számláló rendezés megfelelő választás-e az adott rendezési feladathoz.

Az alábbiakban egy példa bemutatja, hogyan használható a számláló rendezés kis értéktartományú adatok rendezésére:

```cpp
#include <iostream>
#include <vector>

// Counting Sort function
void countingSort(std::vector<int>& arr, int max_val) {
    std::vector<int> count(max_val + 1, 0); // Initialize count array
    std::vector<int> output(arr.size());   // Output array

    // Count the occurrences of each element
    for (int num : arr) {
        count[num]++;
    }

    // Update count array to hold the actual positions
    for (int i = 1; i <= max_val; i++) {
        count[i] += count[i - 1];
    }

    // Build the output array
    for (int i = arr.size() - 1; i >= 0; i--) {
        output[count[arr[i]] - 1] = arr[i];
        count[arr[i]]--;
    }

    // Copy the sorted elements back into the original array
    for (int i = 0; i < arr.size(); i++) {
        arr[i] = output[i];
    }
}

int main() {
    std::vector<int> arr = {4, 2, 2, 8, 3, 3, 1};
    int max_val = 8; // The maximum value in the array

    countingSort(arr, max_val);

    std::cout << "Sorted array: ";
    for (int num : arr) {
        std::cout << num << " ";
    }
    return 0;
}
```

Ebben a példában a `countingSort` függvény a `max_val` paraméter segítségével határozza meg az értéktartomány felső határát. Ez a megközelítés biztosítja, hogy a `count` tömb mérete megfelelő legyen, és a rendezési folyamat hatékonyan végrehajtható legyen. A fenti kódrészlet jól szemlélteti, hogy a számláló rendezés hogyan alkalmazható kis értéktartományú adatok rendezésére, és hogyan érhető el gyors és hatékony rendezési eredmény.

### 2.8.3. Teljesítmény és komplexitás elemzése

A számláló rendezés (Counting Sort) hatékonysága és teljesítménye számos tényezőtől függ, beleértve az adatok értéktartományát és az adathalmaz méretét. Ebben az alfejezetben részletesen megvizsgáljuk a számláló rendezés idő- és térbeli komplexitását, valamint az algoritmus viselkedését különböző körülmények között.

#### Időkomplexitás

A számláló rendezés időkomplexitása az alábbi három fő lépésből áll:

1. **Gyakorisági tömb feltöltése:**
   Az algoritmus végigjárja az eredeti tömböt, és megszámolja az egyes elemek előfordulásait. Ez a lépés O(n) időt igényel, ahol n az elemek száma.

2. **Felhalmozott gyakorisági tömb létrehozása:**
   A gyakorisági tömbből létrehozza a felhalmozott gyakorisági tömböt, amely megadja az elemek végső pozícióját a rendezett tömbben. Ez a lépés O(k) időt vesz igénybe, ahol k az elemek értéktartománya.

3. **Elemek elhelyezése a rendezett tömbben:**
   Az algoritmus végigjárja az eredeti tömböt, és az elemeket a megfelelő helyre helyezi el a rendezett tömbben a felhalmozott gyakorisági tömb alapján. Ez a lépés szintén O(n) időt igényel.

Az időkomplexitás összességében tehát O(n + k), ahol n az elemek száma, k pedig az értéktartomány.

#### Térkomplexitás

A számláló rendezés térkomplexitása az alábbi tényezőkből tevődik össze:

1. **Gyakorisági tömb:**
   A gyakorisági tömb mérete az értéktartománytól függ, így O(k) memória szükséges a gyakorisági tömb tárolásához.

2. **Kimeneti tömb:**
   Az algoritmus egy kimeneti tömböt használ az elemek rendezéséhez, amelynek mérete O(n).

3. **Eredeti tömb:**
   Az eredeti tömb memóriája, amely O(n) méretű.

A teljes térkomplexitás tehát O(n + k), ahol n az elemek száma, k pedig az értéktartomány.

#### Teljesítmény analízis

A számláló rendezés teljesítménye különböző körülmények között változhat. Az alábbiakban néhány specifikus esetet vizsgálunk meg:

1. **Kis értéktartomány:**
   Ha az értéktartomány viszonylag kicsi, a számláló rendezés nagyon hatékony, mivel a gyakorisági tömb mérete korlátozott. Ebben az esetben az algoritmus időkomplexitása közel lineáris, azaz O(n).

2. **Nagy értéktartomány:**
   Ha az értéktartomány nagyon nagy, a számláló rendezés memóriaigénye jelentősen megnő, ami a gyakorisági tömb méretének növekedésével jár. Ebben az esetben az algoritmus idő- és térkomplexitása is O(n + k), ami kevésbé hatékony nagy értéktartomány esetén.

3. **Sűrű és ritka adatok:**
   A számláló rendezés jól teljesít sűrű adatok esetén, ahol az értékek egy szűk tartományban koncentrálódnak. Ritka adatok esetén, ahol az értékek széles tartományban szóródnak, az algoritmus hatékonysága csökkenhet.

4. **Stabilitás fenntartása:**
   A számláló rendezés stabil algoritmus, ami azt jelenti, hogy az azonos értékű elemek eredeti sorrendje megmarad. Ez előnyös bizonyos alkalmazásokban, például amikor rendezett adatokra van szükség, amelyekben az eredeti sorrend megőrzése fontos.

#### Gyakorlati alkalmazások

A számláló rendezés bizonyos körülmények között különösen hasznos:

1. **Histogramok készítése:**
   A számláló rendezést gyakran használják histogramok készítéséhez, ahol gyorsan meg kell határozni az értékek gyakoriságát egy adathalmazban.

2. **Radix sort alapjaként:**
   A számláló rendezés a radix sort alapjaként is szolgálhat, ahol az egyes lépésekben a számláló rendezés technikáját alkalmazzák. Ez lehetővé teszi a radix sort számára, hogy hatékonyan rendezzen nagyobb értéktartományú adatokat is.

3. **Adatok előfeldolgozása:**
   Olyan alkalmazásokban, ahol az adatok előfeldolgozását igénylik, a számláló rendezés gyorsan és hatékonyan használható az adatok rendezésére és a gyakorisági eloszlások meghatározására.

#### Példa a teljesítmény analízisre

Az alábbiakban egy példát mutatunk be a számláló rendezés teljesítményének elemzésére különböző értéktartományok esetén. Tegyük fel, hogy egy tömböt szeretnénk rendezni, amely 10,000 elemet tartalmaz.

1. **Kis értéktartomány (0-100):**
    - Időkomplexitás: O(10,000 + 100) $\approx$ O(10,000)
    - Térkomplexitás: O(10,000 + 100) $\approx$ O(10,000)
    - A számláló rendezés hatékony és gyors, mivel a gyakorisági tömb mérete kicsi.

2. **Nagy értéktartomány (0-1,000,000):**
    - Időkomplexitás: O(10,000 + 1,000,000) $\approx$ O(1,000,000)
    - Térkomplexitás: O(10,000 + 1,000,000) $\approx$ O(1,000,000)
    - A számláló rendezés memóriaigénye jelentős, és az algoritmus hatékonysága csökken.

#### Kód példa C++ nyelven

Az alábbi kódrészlet bemutatja a számláló rendezés implementációját C++ nyelven, amely különböző értéktartományú adatok rendezésére használható:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

// Function to find the maximum value in the array
int findMax(const std::vector<int>& arr) {
    return *std::max_element(arr.begin(), arr.end());
}

// Counting Sort function
void countingSort(std::vector<int>& arr) {
    int max_val = findMax(arr); // Find the maximum value in the array
    std::vector<int> count(max_val + 1, 0); // Initialize count array

    // Count the occurrences of each element
    for (int num : arr) {
        count[num]++;
    }

    // Update count array to hold the actual positions of the elements
    for (int i = 1; i <= max_val; i++) {
        count[i] += count[i - 1];
    }

    // Build the output array
    std::vector<int> output(arr.size());
    for (int i = arr.size() - 1; i >= 0; i--) {
        output[count[arr[i]] - 1] = arr[i];
        count[arr[i]]--;
    }

    // Copy the sorted elements back into the original array
    for (int i = 0; i < arr.size(); i++) {
        arr[i] = output[i];
    }
}

int main() {
    std::vector<int> arr = {4, 2, 2, 8, 3, 3, 1};
    countingSort(arr);

    std::cout << "Sorted array: ";
    for (int num : arr) {
        std::cout << num << " ";
    }
    return 0;
}
```

Ez a kódrészlet jól illusztrálja a számláló rendezés alapvető működését és teljesítményét különböző értéktartományok esetén. Az implementáció során a maximális érték meghatározása, a gyakorisági tömb létrehozása és feltöltése, a felhalmozott gyakoriság kiszámítása, valamint az elemek rendezett tömbbe történő elhelyezése mind hozzájárulnak az algoritmus hatékony működéséhez.

#### Összefoglalás

A számláló rendezés teljesítményének és komplexitásának elemzése rávilágít az algoritmus erősségeire és korlátaira. Az algoritmus időkomplexitása O(n + k), amely gyors rendezést tesz lehetővé kis értéktartományú adatok esetén. Ugyanakkor a nagy értéktartomány és a nem egész számok kezelése komoly kihívást jelenthet, ami az algoritmus memóriaigényének és összetettségének növekedéséhez vezet. Az algoritmus stabilitása és egyszerűsége előnyt jelent bizonyos alkalmazásokban, mint például histogram készítés vagy előfeldolgozási feladatok. A teljesítmény elemzés során figyelembe kell venni az adatok értéktartományát és típusát, hogy a számláló rendezés valóban hatékonyan alkalmazható legyen az adott probléma megoldására.

### 2.8.4. Gyakorlati alkalmazások és példák

A számláló rendezés (Counting Sort) nem csupán egy elméleti koncepció, hanem számos gyakorlati alkalmazása is van, különösen akkor, ha az adatok viszonylag szűk tartományba esnek és gyors rendezést igényelnek. Ebben az alfejezetben részletesen bemutatjuk a számláló rendezés gyakorlati alkalmazásait és konkrét példákat adunk annak használatára.

#### Gyakorlati alkalmazások

1. **Vizsga pontszámok rendezése:**
   Az oktatási intézményekben gyakran szükség van a diákok vizsga pontszámainak rendezésére. Mivel a pontszámok egy jól meghatározott tartományba (pl. 0 és 100 közé) esnek, a számláló rendezés ideális választás lehet. Az algoritmus gyorsan és hatékonyan képes rendezni a pontszámokat, lehetővé téve a gyors kiértékelést és rangsorolást.

2. **Digitális képfeldolgozás:**
   A képfeldolgozásban gyakran előfordul, hogy a pixelértékek eloszlását kell gyorsan meghatározni. A számláló rendezést használhatják például hisztogramok készítésére, ahol minden egyes pixel intenzitásának gyakoriságát számolják meg. Ez az információ felhasználható különböző képfeldolgozási műveletek, például kontrasztjavítás vagy szegmentálás során.

3. **Radix sort alapjaként:**
   A számláló rendezés gyakran szolgál alapként a radix sort algoritmusban, különösen akkor, amikor az egyes helyi értékek rendezésére van szükség. A radix sort a számláló rendezés gyorsaságát és egyszerűségét használja ki, hogy nagyobb értéktartományú adatokat is hatékonyan rendezhessen.

4. **Szociális hálózatok adatainak elemzése:**
   A szociális hálózatokban gyakran szükséges az adatok gyors rendezése és kiértékelése, például a felhasználók aktivitási gyakoriságának vagy a posztok népszerűségének meghatározása. A számláló rendezés hatékonyan alkalmazható az ilyen típusú adatfeldolgozási feladatokban.

5. **Egészségügyi adatok elemzése:**
   Az egészségügyi szektorban, különösen a járványok idején, fontos lehet a betegek száma, a fertőzési arányok és más egészségügyi mutatók gyors elemzése. A számláló rendezés gyorsan képes ezeket az adatokat feldolgozni, segítve a döntéshozatalt és az erőforrások allokálását.

#### Konkrét példák

1. **Vizsga pontszámok rendezése**

Képzeljük el, hogy egy osztály 30 diákjának vizsga pontszámait szeretnénk rendezni. A pontszámok 0 és 100 között változnak, és a cél az, hogy a diákokat pontszámuk alapján rendezzük sorba. Az alábbi példában bemutatjuk, hogyan lehet ezt a számláló rendezés segítségével megvalósítani:

```cpp
#include <iostream>
#include <vector>

// Function to perform counting sort on exam scores
void countingSort(std::vector<int>& scores, int maxScore) {
    std::vector<int> count(maxScore + 1, 0);
    std::vector<int> sortedScores(scores.size());

    // Count the occurrences of each score
    for (int score : scores) {
        count[score]++;
    }

    // Update count array to hold the actual positions
    for (int i = 1; i <= maxScore; i++) {
        count[i] += count[i - 1];
    }

    // Build the sorted scores array
    for (int i = scores.size() - 1; i >= 0; i--) {
        sortedScores[count[scores[i]] - 1] = scores[i];
        count[scores[i]]--;
    }

    // Copy the sorted scores back to the original array
    for (int i = 0; i < scores.size(); i++) {
        scores[i] = sortedScores[i];
    }
}

int main() {
    std::vector<int> scores = {55, 72, 88, 33, 45, 89, 90, 55, 72, 88};
    int maxScore = 100;

    countingSort(scores, maxScore);

    std::cout << "Sorted scores: ";
    for (int score : scores) {
        std::cout << score << " ";
    }
    return 0;
}
```

2. **Képfeldolgozás - hisztogram készítése**

A digitális képek hisztogramjának készítése során a pixelértékek gyakorisági eloszlását határozzuk meg. Az alábbi példában bemutatjuk, hogyan lehet a számláló rendezést használni egy kép hisztogramjának elkészítésére:

```cpp
#include <iostream>
#include <vector>

// Function to create histogram of image pixel values
std::vector<int> createHistogram(const std::vector<int>& image, int maxPixelValue) {
    std::vector<int> histogram(maxPixelValue + 1, 0);

    // Count the occurrences of each pixel value
    for (int pixel : image) {
        histogram[pixel]++;
    }

    return histogram;
}

int main() {
    std::vector<int> image = {0, 1, 2, 1, 2, 3, 3, 2, 1, 0, 0, 1, 3, 3, 2};
    int maxPixelValue = 3;

    std::vector<int> histogram = createHistogram(image, maxPixelValue);

    std::cout << "Pixel value histogram: ";
    for (int value = 0; value <= maxPixelValue; value++) {
        std::cout << "Value " << value << ": " << histogram[value] << " ";
    }
    return 0;
}
```

3. **Radix sort alapjaként**

A radix sort egy összetettebb rendezési algoritmus, amely több lépésben rendez nagyobb értéktartományú adatokat. Minden egyes lépésben a számláló rendezést alkalmazza a részrendezéshez. Az alábbiakban bemutatjuk, hogyan használható a számláló rendezés a radix sort részeként:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

// Function to perform counting sort on digits represented by exp
void countingSortForRadix(std::vector<int>& arr, int exp) {
    int n = arr.size();
    std::vector<int> output(n);
    std::vector<int> count(10, 0);

    // Count occurrences of digits
    for (int i = 0; i < n; i++) {
        count[(arr[i] / exp) % 10]++;
    }

    // Update count array to hold the actual positions
    for (int i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }

    // Build the output array
    for (int i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }

    // Copy the output array to arr
    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }
}

// Function to perform radix sort
void radixSort(std::vector<int>& arr) {
    int max_val = *std::max_element(arr.begin(), arr.end());

    // Apply counting sort to each digit
    for (int exp = 1; max_val / exp > 0; exp *= 10) {
        countingSortForRadix(arr, exp);
    }
}

int main() {
    std::vector<int> arr = {170, 45, 75, 90, 802, 24, 2, 66};
    radixSort(arr);

    std::cout << "Sorted array: ";
    for (int num : arr) {
        std::cout << num << " ";
    }
    return 0;
}
```

#### Elemzés és teljesítmény

A gyakorlati alkalmazásokban a számláló rendezés teljesítménye jelentős mértékben függ az adatok jellegétől. Ahogy korábban említettük, a kis értéktartományú adatok esetén az algoritmus közel lineáris időkomplexitást érhet el, ami rendkívül hatékonnyá teszi az alkalmazást.

Az elemzések és példák alapján a számláló rendezés különösen hasznos lehet olyan helyzetekben, ahol az adatok értéktartománya szűk, és gyors, hatékony rendezés szükséges. Az algoritmus egyszerűsége és hatékonysága miatt számos területen alkalmazható, beleértve az oktatást, a digitális képfeldolgozást, a komplex rendezési algoritmusok alapjaként való felhasználást, valamint a szociális hálózatok és az egészségügy adatelemzését.

#### Összefoglalás

A számláló rendezés egy sokoldalú és hatékony rendezési algoritmus, amely számos gyakorlati alkalmazásban jól teljesít. Az algoritmus egyszerűsége és gyorsasága különösen előnyös kis értéktartományú adatok esetén, míg nagyobb értéktartomány esetén is alkalmazható, ha megfelelő előfeldolgozást végeznek. A bemutatott példák szemléltetik, hogy a számláló rendezés hogyan használható különböző gyakorlati problémák megoldására, és hogyan érhető el gyors és hatékony rendezési eredmény.