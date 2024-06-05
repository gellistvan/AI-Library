\newpage

## 3.8. Intervallum ütemezés (Interval Scheduling)

Az intervallum ütemezés egy klasszikus probléma az algoritmusok és operációkutatás területén, amely számos valós alkalmazással rendelkezik, beleértve a feladatütemezést, az erőforrás-allokációt és az időbeosztás optimalizálását. Az intervallum ütemezési problémában adott egy sor intervallum, ahol minden intervallum egy adott időszakot reprezentál. A cél az, hogy kiválasszuk a lehető legtöbb, egymást nem átfedő intervallumot. Ez a probléma különösen fontos a számítástechnikai és mérnöki területeken, ahol gyakran kell maximalizálni a kihasználtságot vagy minimalizálni az átfedéseket különböző feladatok és tevékenységek között. A fejezet további részeiben megvizsgáljuk a maximális számú nem átfedő intervallum kiválasztásának módszereit, először rekurzív megközelítéssel, majd dinamikus programozással, bemutatva mindkét módszer hatékonyságát és alkalmazhatóságát.

### 3.8.1. Maximális számú nem átfedő intervallum kiválasztása

Az intervallum ütemezési probléma központi kérdése, hogy egy adott intervallumkészletből hogyan választhatjuk ki a maximális számú egymást nem átfedő intervallumot. Ez a probléma számos valós alkalmazási területtel rendelkezik, beleértve a feladatütemezést, a konferenciatermek foglalását, a hálózati kapcsolatok sávszélességének optimalizálását és sok más területet, ahol az erőforrásokat hatékonyan kell kezelni.

#### Probléma meghatározása

Az intervallum ütemezési probléma formálisan az alábbiak szerint definiálható:
- Legyen adott egy sor $\{(s_1, f_1), (s_2, f_2), \ldots, (s_n, f_n)\}$ intervallum, ahol $s_i$ az $i$-edik intervallum kezdőpontja és $f_i$ az $i$-edik intervallum végpontja.
- A cél az, hogy kiválasszuk a lehető legtöbb olyan intervallumot, hogy azok ne fedjék át egymást, azaz ha $(s_i, f_i)$ és $(s_j, f_j)$ két kiválasztott intervallum, akkor $s_j \geq f_i$ vagy $s_i \geq f_j$.

#### Általános megközelítés

Az intervallum ütemezési probléma megoldására több módszer is létezik, beleértve a rekurzív és a dinamikus programozási megközelítéseket. Az egyik legismertebb és leghatékonyabb módszer a mohó algoritmus, amely az alábbi elven működik:

1. **Rendezés végpontok szerint:** Az intervallumokat a befejezési időpontjuk szerint rendezzük növekvő sorrendbe.
2. **Intervallum kiválasztása:** A legkorábban befejeződő intervallumot választjuk ki először, majd azokat az intervallumokat vesszük figyelembe, amelyek nem fedik át az utoljára kiválasztott intervallumot.
3. **Iteráció:** Ezt a folyamatot addig folytatjuk, amíg az összes intervallumot meg nem vizsgáltuk.

A következő lépések részletesen bemutatják a mohó algoritmus működését.

#### Mohó algoritmus lépései

1. **Intervallumok rendezése:** Először rendezzük az intervallumokat a végpontjaik szerint. Ez az előfeldolgozási lépés biztosítja, hogy mindig a lehető legkorábban befejeződő intervallumot válasszuk ki, minimalizálva ezzel az átfedések esélyét.

   Például, legyenek az intervallumok: $\{(1, 4), (2, 6), (5, 7), (3, 8), (8, 9)\}$. Ezeket rendezzük a végpontjaik szerint: $\{(1, 4), (2, 6), (5, 7), (3, 8), (8, 9)\}$.

2. **Intervallum kiválasztása:** Kezdjük az első intervallummal, és vegyük fel a kiválasztott intervallumok halmazába. Ezután iteráljunk végig az intervallumokon, és minden egyes intervallum esetében ellenőrizzük, hogy átfedi-e az utoljára kiválasztott intervallumot. Ha nem fedik át egymást, akkor adjuk hozzá a kiválasztott intervallumok halmazához.

   A rendezett példából kiindulva:
    - Kezdjük az első intervallummal $(1, 4)$.
    - A következő intervallum $(2, 6)$, amely átfedi az $(1, 4)$ intervallumot, így ezt kihagyjuk.
    - A következő intervallum $(5, 7)$ nem átfedő, így hozzáadjuk.
    - A következő intervallum $(3, 8)$ átfedi a $(5, 7)$ intervallumot, így ezt kihagyjuk.
    - Az utolsó intervallum $(8, 9)$ nem átfedő, így hozzáadjuk.

   A kiválasztott intervallumok: $\{(1, 4), (5, 7), (8, 9)\}$.

#### Mohó algoritmus bizonyítása

A mohó algoritmus helyességét indukcióval bizonyíthatjuk. A bizonyítás alapja az, hogy az algoritmus minden lépésben a legkorábban befejeződő intervallumot választja ki, amely maximalizálja a hátralévő időintervallumokat.

1. **Alap eset:** Tegyük fel, hogy csak egy intervallum van, ebben az esetben nyilvánvalóan a mohó választás helyes.
2. **Indukciós lépés:** Tegyük fel, hogy az algoritmus helyes az első $k$ intervallumra. Most bizonyítsuk be, hogy a $k+1$-edik intervallum esetében is helyes.
    - Tegyük fel, hogy az algoritmus a $k+1$-edik lépésben az $i$-edik intervallumot választja ki, amely a legkorábban befejeződő intervallum az összes lehetséges $k+1$ intervallum közül.
    - Ha lenne egy másik intervallum, amely előbb befejeződik, akkor azt már korábban kiválasztottuk volna, mivel az algoritmus mindig a legkorábban befejeződőt választja ki.

Ez a bizonyítás mutatja, hogy a mohó algoritmus minden lépésben helyes választást hoz, és így garantálja a helyes megoldást.

#### Mohó algoritmus C++ implementációja

Az alábbi C++ kód egy példa a mohó algoritmus implementálására az intervallum ütemezési probléma megoldására:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

// Definiáljuk az intervallumot
struct Interval {
    int start;
    int end;
};

// Összehasonlító függvény az intervallumok rendezéséhez a végpontjuk szerint
bool compareInterval(Interval i1, Interval i2) {
    return (i1.end < i2.end);
}

// Mohó algoritmus az intervallum ütemezési probléma megoldására
std::vector<Interval> intervalScheduling(std::vector<Interval>& intervals) {
    // Intervallumok rendezése a végpontjuk szerint
    std::sort(intervals.begin(), intervals.end(), compareInterval);

    std::vector<Interval> selectedIntervals;
    int lastEnd = -1;

    // Intervallumok kiválasztása
    for (const auto& interval : intervals) {
        if (interval.start >= lastEnd) {
            selectedIntervals.push_back(interval);
            lastEnd = interval.end;
        }
    }

    return selectedIntervals;
}

int main() {
    std::vector<Interval> intervals = {{1, 4}, {2, 6}, {5, 7}, {3, 8}, {8, 9}};
    std::vector<Interval> result = intervalScheduling(intervals);

    std::cout << "Selected intervals: ";
    for (const auto& interval : result) {
        std::cout << "{" << interval.start << ", " << interval.end << "} ";
    }
    std::cout << std::endl;

    return 0;
}
```

#### Idő- és térkomplexitás

A mohó algoritmus időkomplexitása az intervallumok rendezése miatt $O(n \log n)$, ahol $n$ az intervallumok száma. Az intervallumok rendezése után az algoritmus lineáris időben $O(n)$ iterál végig az intervallumokon, hogy kiválassza a nem átfedő intervallumokat. Ezért a teljes időkomplexitás $O(n \log n)$.

A térkomplexitás $O(n)$, mivel szükséges egy tömb az intervallumok tárolásához és egy további tömb a kiválasztott intervallumok tárolásához.

#### Gyakorlati alkalmazások

Az intervallum ütemezési probléma számos gyakorlati alkalmazással rendelkezik:
- **Feladatütemezés:** Feladatok ütemezése korlátozott erőforrásokkal rendelkező rendszerekben, például CPU időosztás.
- **Időbeosztás:** Konferencia termek, előadótermek vagy egyéb erőforrások időbeosztásának optimalizálása.
- **Hálózati kapcsolatok:** Adatátvitel optimalizálása hálózatokban, ahol a sávszélesség hatékony kihasználása a cél.

#### Következtetés

A mohó algoritmus hatékony és egyszerű megoldást kínál a maximális számú nem átfedő intervallum kiválasztására az intervallum ütemezési problémában. A rendezés és az iteratív kiválasztás kombinációja garantálja, hogy az algoritmus mindig helyes és optimális megoldást talál. A következő alfejezetekben részletesebben megvizsgáljuk a probléma rekurzív és dinamikus programozási megközelítéseit, amelyek további betekintést nyújtanak a probléma komplexitásába és a különböző megoldási módszerek hatékonyságába.

### 3.8.2. Megoldás rekurzióval

Az intervallum ütemezési probléma rekurzív megközelítése egy alapvető és intuitív módszer, amely mélyebb megértést nyújt a probléma struktúrájáról és annak megoldási lehetőségeiről. A rekurzió természetes módon illeszkedik az intervallum ütemezési problémákhoz, mivel lehetőséget ad a probléma kisebb részekre bontására, amelyek függetlenül megoldhatók. Ebben a fejezetben részletesen bemutatjuk a rekurzív megközelítést az intervallum ütemezési probléma megoldására, beleértve a probléma rekurzív felbontását, a rekurzív függvények tervezését, valamint a módszer előnyeit és hátrányait.

#### Probléma meghatározása és felbontása

Az intervallum ütemezési probléma rekurzív megközelítése során a cél az, hogy kiválasszuk a lehető legtöbb, egymást nem átfedő intervallumot. A probléma megoldása érdekében a rekurzió használatával folyamatosan felosztjuk a problémát kisebb részekre, amelyek egyre egyszerűbbek és könnyebben kezelhetők.

A rekurzív megközelítés alapja az, hogy minden egyes intervallum kiválasztásakor vagy elutasításakor a probléma kisebb, hasonló struktúrájú részproblémákra bontható.

#### Rekurzív algoritmus tervezése

A rekurzív algoritmus tervezése során az alábbi lépéseket követjük:

1. **Rendezés:** Először rendezzük az intervallumokat a kezdési időpontjuk szerint, hogy biztosítsuk az intervallumok rendezett feldolgozását.
2. **Alap eset:** Definiáljuk az alap esetet, amely egyszerűen megoldható. Például, ha nincs több intervallum, amit meg kell vizsgálni, a kiválasztott intervallumok száma nulla.
3. **Rekurzív hívás:** Minden egyes intervallum kiválasztása vagy elutasítása esetén rekurzívan hívjuk a függvényt a maradék intervallumokra, majd összehasonlítjuk a különböző választási lehetőségek eredményeit a maximális számú nem átfedő intervallum megtalálása érdekében.

Az alábbi lépések részletesen bemutatják a rekurzív algoritmus működését:

1. **Rendezés:** Az intervallumokat rendezzük a kezdési időpontjuk szerint.

2. **Rekurzív függvény:** A rekurzív függvény feladata a maximális számú nem átfedő intervallum megtalálása. A függvény két fő ágat tartalmaz:
    - **Kiválasztás ága:** Ha az aktuális intervallumot kiválasztjuk, a függvényt újra meghívjuk a maradék intervallumokra, amelyek nem átfedők az aktuálisan kiválasztott intervallummal.
    - **Elutasítás ága:** Ha az aktuális intervallumot elutasítjuk, a függvényt újra meghívjuk a maradék intervallumokra az aktuális intervallum kihagyásával.

3. **Alap eset:** Ha nincs több intervallum a listában, a kiválasztott intervallumok száma nulla.

Az alábbi C++ kód szemlélteti a rekurzív megközelítést:

```cpp
#include <vector>
#include <algorithm>
#include <iostream>

struct Interval {
    int start;
    int end;
};

bool compareInterval(Interval i1, Interval i2) {
    return (i1.start < i2.start);
}

int intervalSchedulingRec(const std::vector<Interval>& intervals, int index, int lastEnd) {
    if (index == intervals.size()) {
        return 0;
    }

    // Kihagyjuk az aktuális intervallumot
    int exclude = intervalSchedulingRec(intervals, index + 1, lastEnd);

    // Kiválasztjuk az aktuális intervallumot, ha nem átfedő
    int include = 0;
    if (intervals[index].start >= lastEnd) {
        include = 1 + intervalSchedulingRec(intervals, index + 1, intervals[index].end);
    }

    // Visszaadjuk a két lehetőség közül a maximumot
    return std::max(include, exclude);
}

int intervalScheduling(const std::vector<Interval>& intervals) {
    std::vector<Interval> sortedIntervals = intervals;
    std::sort(sortedIntervals.begin(), sortedIntervals.end(), compareInterval);
    return intervalSchedulingRec(sortedIntervals, 0, 0);
}

int main() {
    std::vector<Interval> intervals = {{1, 4}, {2, 6}, {5, 7}, {3, 8}, {8, 9}};
    int result = intervalScheduling(intervals);
    std::cout << "Maximum number of non-overlapping intervals: " << result << std::endl;
    return 0;
}
```

#### Részletek és elemzés

A rekurzív megközelítés az alábbi lépésekből áll:

1. **Kezdés és rendezés:** Az intervallumokat először rendezni kell a kezdési időpontjuk szerint. Ez az előfeldolgozási lépés $O(n \log n)$ időkomplexitású, ahol $n$ az intervallumok száma.

2. **Rekurzív hívások:** Minden egyes intervallum esetén két fő ágat vizsgálunk:
    - **Kiválasztás ága:** Ha az aktuális intervallumot kiválasztjuk, a rekurzív függvényt meghívjuk a maradék intervallumokra, figyelembe véve az aktuális intervallum végpontját.
    - **Elutasítás ága:** Ha az aktuális intervallumot elutasítjuk, a rekurzív függvényt meghívjuk a maradék intervallumokra, figyelmen kívül hagyva az aktuális intervallumot.

3. **Alap eset:** Ha nincs több intervallum a listában, a visszatérési érték nulla, mivel nincs több kiválasztható intervallum.

#### Idő- és térkomplexitás

A rekurzív megközelítés időkomplexitása általában exponenciális $O(2^n)$, mivel minden egyes intervallum esetén két ágat vizsgálunk (kiválasztás és elutasítás), és ezek mindegyike újabb rekurzív hívásokat eredményez.

A térkomplexitás $O(n)$, mivel a rekurzív hívások mélysége legfeljebb $n$, azaz az intervallumok száma.

#### Előnyök és hátrányok

**Előnyök:**
- **Egyszerű és intuitív:** A rekurzív megközelítés könnyen megérthető és implementálható, különösen kisebb problémaméretek esetén.
- **Természetes problémamegoldás:** A rekurzió természetes módon illeszkedik a probléma felosztásához kisebb részproblémákra.

**Hátrányok:**
- **Exponenciális időkomplexitás:** A rekurzív megközelítés időkomplexitása gyorsan növekszik a problémaméret növekedésével, ami nagyobb intervallumkészletek esetén hatékonysági problémákhoz vezethet.
- **Memóriahasználat:** A rekurzív hívások mélysége nagy memóriahasználatot eredményezhet, különösen nagyobb problémaméretek esetén.

#### Optimalizációs lehetőségek

A rekurzív megközelítés optimalizálására több módszer is létezik:

- **Memoizáció:** A memoizáció segítségével eltárolhatjuk a már kiszámított részproblémák eredményeit, így elkerülve az ismételt számításokat. Ez jelentősen csökkentheti az időkomplexitást.
- **Dinamikus programozás:** A dinamikus programozás iteratív megközelítést alkalmaz, amely szintén eltárolja és újrafelhasználja a részeredményeket, így elkerülve a redundáns számításokat. Ez a módszer jelentősen javíthatja a megoldás hatékonyságát.

#### Következtetés

A rekurzív megközelítés az intervallum ütemezési problémára egyszerű és intuitív módszer, amely lehetőséget nyújt a probléma mélyebb megértésére és a különböző megoldási lehetőségek felfedezésére. Bár a rekurzív megközelítés idő- és térkomplexitása korlátozhatja annak alkalmazhatóságát nagyobb problémaméretek esetén, a memoizáció és a dinamikus programozás segítségével jelentősen javítható a megoldás hatékonysága. A következő alfejezetben részletesebben megvizsgáljuk a dinamikus programozási megközelítést, amely további betekintést nyújt a probléma komplexitásába és a hatékony megoldási módszerekbe.

### 3.8.3. Megoldás dinamikus programozással

Az intervallum ütemezési probléma dinamikus programozási megközelítése hatékony módszer a maximális számú nem átfedő intervallum kiválasztására. A dinamikus programozás különösen hasznos, amikor a probléma kisebb, egymást átfedő részproblémákra bontható, és ezek a részproblémák optimalizálhatók. Ebben a fejezetben részletesen bemutatjuk a dinamikus programozási megközelítést az intervallum ütemezési probléma megoldására, beleértve a probléma felbontását, az optimális alstruktúra tulajdonságait, a dinamikus programozás elveit és a megoldási algoritmus lépéseit.

#### Probléma meghatározása és felbontása

Az intervallum ütemezési probléma lényege, hogy adott egy sor intervallum, és ezek közül ki kell választani a maximális számú nem átfedő intervallumot. A probléma formálisan az alábbiak szerint definiálható:
- Legyen adott egy sor $\{(s_1, f_1), (s_2, f_2), \ldots, (s_n, f_n)\}$ intervallum, ahol $s_i$ az $i$-edik intervallum kezdőpontja és $f_i$ az $i$-edik intervallum végpontja.
- A cél az, hogy kiválasszuk a lehető legtöbb olyan intervallumot, hogy azok ne fedjék át egymást, azaz ha $(s_i, f_i)$ és $(s_j, f_j)$ két kiválasztott intervallum, akkor $s_j \geq f_i$ vagy $s_i \geq f_j$.

#### Dinamikus programozási megközelítés

A dinamikus programozás alapja, hogy a problémát kisebb részproblémákra bontjuk, és az ezekre vonatkozó optimális megoldásokat eltároljuk, majd ezek segítségével építjük fel a teljes probléma megoldását. Az intervallum ütemezési probléma esetében az alábbi lépések követhetők:

1. **Intervallumok rendezése:** Először rendezzük az intervallumokat a befejezési időpontjuk szerint. Ez biztosítja, hogy az intervallumok feldolgozása során mindig a legkorábban befejeződő intervallumokat vegyük figyelembe, ami egyszerűsíti az optimális megoldás megtalálását.

2. **Részproblémák definiálása:** Határozzuk meg a részproblémákat. Legyen $dp[i]$ az az érték, amely a maximális számú nem átfedő intervallumot jelöli az első $i$ intervallumból kiválasztva. A célunk $dp[n]$ meghatározása, ahol $n$ az intervallumok száma.

3. **Optimális alstruktúra:** Az optimális alstruktúra tulajdonsága alapján, ha $i$-edik intervallumot kiválasztjuk, akkor az összes korábbi intervallum közül csak azok lehetnek a megoldás részei, amelyek nem fedik át az $i$-edik intervallumot. Definiáljuk $p(i)$-t úgy, mint az utoljára befejeződő intervallum indexe, amely nem átfedő az $i$-edik intervallummal.

4. **Állapotátmenet:** Az állapotátmenet az alábbiak szerint definiálható:
   $$
   dp[i] = \max(dp[i-1], 1 + dp[p(i)])
   $$
   Ez azt jelenti, hogy az $i$-edik intervallum kiválasztása vagy elutasítása alapján döntünk. Ha az $i$-edik intervallumot elutasítjuk, akkor $dp[i] = dp[i-1]$. Ha az $i$-edik intervallumot kiválasztjuk, akkor $dp[i] = 1 + dp[p(i)]$.

5. **Inicializálás:** Kezdetben $dp[0] = 0$, mivel nincs intervallum, amit ki kellene választani.

6. **Iteráció:** Iteráljunk végig az összes intervallumon 1-től $n$-ig, és alkalmazzuk az állapotátmenetet minden egyes intervallumra.

Az alábbi C++ kód szemlélteti a dinamikus programozás alkalmazását az intervallum ütemezési problémára:

```cpp
#include <vector>
#include <algorithm>
#include <iostream>

// Definiáljuk az intervallumot
struct Interval {
    int start;
    int end;
};

// Összehasonlító függvény az intervallumok rendezéséhez a végpontjuk szerint
bool compareInterval(Interval i1, Interval i2) {
    return (i1.end < i2.end);
}

// Bináris keresés a megfelelő p(i) megtalálásához
int binarySearch(const std::vector<Interval>& intervals, int index) {
    int low = 0, high = index - 1;
    while (low <= high) {
        int mid = (low + high) / 2;
        if (intervals[mid].end <= intervals[index].start) {
            if (intervals[mid + 1].end <= intervals[index].start) {
                low = mid + 1;
            } else {
                return mid;
            }
        } else {
            high = mid - 1;
        }
    }
    return -1;
}

int intervalSchedulingDP(std::vector<Interval>& intervals) {
    std::sort(intervals.begin(), intervals.end(), compareInterval);
    int n = intervals.size();
    std::vector<int> dp(n, 0);
    
    dp[0] = 1; // Az első intervallum kiválasztása

    for (int i = 1; i < n; ++i) {
        int incl = 1;
        int l = binarySearch(intervals, i);
        if (l != -1) {
            incl += dp[l];
        }
        dp[i] = std::max(dp[i - 1], incl);
    }

    return dp[n - 1];
}

int main() {
    std::vector<Interval> intervals = {{1, 4}, {2, 6}, {5, 7}, {3, 8}, {8, 9}};
    int result = intervalSchedulingDP(intervals);
    std::cout << "Maximum number of non-overlapping intervals: " << result << std::endl;
    return 0;
}
```

#### Algoritmus részletei és elemzése

1. **Intervallumok rendezése:** Az intervallumokat a végpontjuk szerint rendezzük $O(n \log n)$ idő alatt.

2. **Részproblémák definiálása:** A $dp$ tömb minden eleme az adott részprobléma optimális megoldását tartalmazza.

3. **Optimális alstruktúra:** Az optimális alstruktúra tulajdonsága biztosítja, hogy az $i$-edik intervallum kiválasztásával az előző nem átfedő intervallumok optimális megoldását felhasználhatjuk.

4. **Állapotátmenet:** Az állapotátmenetek alkalmazása biztosítja, hogy minden egyes intervallum kiválasztása vagy elutasítása alapján a maximális számú nem átfedő intervallumot találjuk meg.

5. **Inicializálás:** Kezdetben $dp[0] = 1$, mivel az első intervallum kiválasztása mindig egy intervallumot eredményez.

6. **Iteráció:** Az összes intervallumon végig iterálunk és alkalmazzuk az állapotátmeneteket.

#### Idő- és térkomplexitás

Az algoritmus időkomplexitása $O(n \log n)$, amely az intervallumok rendezéséből adódik, és további $O(n \log n)$ a bináris keresés miatt. Így a teljes időkomplexitás $O(n \log n)$.

A térkomplexitás $O(n)$, mivel egy $n$ elemű tömböt használunk a részproblémák eredményeinek tárolására.

#### Előnyök és hátrányok

**Előnyök:**

- **Hatékonyság:** A dinamikus programozási megközelítés jelentősen hatékonyabb, mint a rekurzív megközelítés, különösen nagyobb problémaméretek esetén.
- **Optimális megoldás:** Az algoritmus garantáltan megtalálja a maximális számú nem átfedő intervallumot.

**Hátrányok:**

- **Komplexitás:** A dinamikus programozási algoritmus tervezése és implementálása bonyolultabb lehet, mint a rekurzív megközelítés.

#### Gyakorlati alkalmazások

A dinamikus programozási megközelítés számos gyakorlati alkalmazással rendelkezik:

- **Feladatütemezés:** Feladatok ütemezése korlátozott erőforrásokkal rendelkező rendszerekben, például CPU időosztás.
- **Időbeosztás:** Konferencia termek, előadótermek vagy egyéb erőforrások időbeosztásának optimalizálása.
- **Hálózati kapcsolatok:** Adatátvitel optimalizálása hálózatokban, ahol a sávszélesség hatékony kihasználása a cél.

#### Következtetés

A dinamikus programozás erőteljes és hatékony megközelítést kínál az intervallum ütemezési probléma megoldására. Azáltal, hogy a problémát kisebb részproblémákra bontjuk és ezek eredményeit eltároljuk, jelentősen csökkenthetjük a számítási időt és a memóriahasználatot. Az ilyen optimalizációs technikák alapvető fontosságúak a modern számítástechnikai alkalmazásokban, és lehetővé teszik a komplex problémák hatékony megoldását. A dinamikus programozási megközelítés segítségével a maximális számú nem átfedő intervallum kiválasztása gyorsan és megbízhatóan elvégezhető, ami számos gyakorlati alkalmazásban hasznos lehet.