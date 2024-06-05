\newpage

# 3. Dinamikus programozás

A dinamikus programozás az algoritmusok tervezésének egy hatékony módszere, amely különösen hasznos olyan problémák megoldására, amelyek részproblémákra bonthatók, és ezek a részproblémák többször is ismétlődnek. A módszer lényege, hogy az ismétlődő részproblémák eredményeit tároljuk, így elkerülve az ugyanazon probléma többszöri megoldását. Ez jelentősen csökkenti az időbeli és számítási komplexitást, különösen nagy méretű problémák esetén. A dinamikus programozás alkalmazása számos területen elterjedt, beleértve a számítógépes tudományokat, operációkutatást, és a gazdasági modellezést. Ebben a fejezetben megvizsgáljuk a dinamikus programozás alapelveit, valamint néhány klasszikus példát és algoritmust, amelyek segítségével jobban megérthetjük ennek a hatékony módszernek az alkalmazását és erejét.

\newpage

## 3.1. Alapelvek
A dinamikus programozás alapelvei fejezet célja, hogy átfogó képet nyújtson ennek a hatékony algoritmikus módszernek a kulcsfogalmairól és technikáiról. Először megvizsgáljuk az optimalitás és az alstruktúra fogalmát, amelyek a dinamikus programozás alapvető építőkövei. Ezt követően részletezzük a top-down és bottom-up megközelítéseket, amelyek különböző stratégiákat kínálnak a problémák megoldására. A memoizáció és a táblázat kitöltés módszereinek bemutatásával rávilágítunk arra, hogyan lehet hatékonyan tárolni az alproblémák eredményeit a számítások gyorsítása érdekében. Végül, összefoglaljuk az általános lépéseket, amelyek irányt adnak a dinamikus programozás alkalmazásában, biztosítva, hogy az olvasó magabiztosan alkalmazhassa ezeket a technikákat különböző problémák megoldására.

### 3.1.1. Optimalitás és alstruktúra

A dinamikus programozás alapvető építőkövei közé tartozik az optimalitás és az alstruktúra fogalma. Ezek a koncepciók alapvető fontosságúak a dinamikus programozási problémák felismerésében és hatékony megoldásában. Ebben az alfejezetben részletesen megvizsgáljuk ezeket a fogalmakat, bemutatva, hogyan járulnak hozzá a dinamikus programozás hatékonyságához és alkalmazhatóságához.

#### Optimalitás (Optimal Substructure)

A dinamikus programozásban az optimalitás azt jelenti, hogy egy adott probléma optimális megoldása tartalmazza az alproblémák optimális megoldásait. Más szavakkal, ha egy probléma optimális megoldásának részproblémái is optimális megoldásokkal rendelkeznek, akkor az eredeti probléma dinamikus programozással oldható meg. Ez a tulajdonság az úgynevezett "optimal substructure" (optimális alstruktúra).

Az optimalitás feltételezése lehetővé teszi, hogy egy problémát kisebb részproblémákra bontsunk, majd ezek megoldásait kombinálva érjük el az eredeti probléma megoldását. Az optimális alstruktúra fogalma kulcsfontosságú a dinamikus programozásban, mert biztosítja, hogy az egyszer kiszámított részmegoldások újrafelhasználhatóak legyenek, így jelentős számítási erőforrásokat takarítva meg.

Példaként tekintsünk egy klasszikus problémát, a leghosszabb közös részszekvencia (Longest Common Subsequence, LCS) problémát. Két karakterlánc, $X$ és $Y$ adott, és meg kell találni a leghosszabb karakterláncot, amely mindkettőben szekvenciaként előfordul. Az LCS-probléma optimális alstruktúrával rendelkezik, mivel a két karakterlánc közötti LCS tartalmazza a részproblémák LCS-eit is.

Vegyünk egy másik konkrét példát: a legrövidebb út problémáját egy gráfban. Legyen $G(V, E)$ egy irányított gráf, ahol $V$ a csúcsok halmaza és $E$ az élek halmaza. Az $u$ és $v$ közötti legrövidebb út $G$-ben az a legrövidebb út, amely az $u$-ból indul és $v$-be érkezik. Tegyük fel, hogy ez az út áthalad egy $w$ csúcson. Az $u$-ból $v$-be vezető legrövidebb út optimális alstruktúrával rendelkezik, mivel tartalmazza az $u$-ból $w$-be és a $w$-ből $v$-be vezető legrövidebb utakat. Ez a tulajdonság lehetővé teszi a dinamikus programozás alkalmazását, mivel az $u$-ból $w$-be és $w$-ből $v$-be vezető legrövidebb utak önmagukban is optimálisak.

#### Alstruktúra (Substructure)

Az alstruktúra fogalma szorosan kapcsolódik az optimalitáshoz. A dinamikus programozásban a probléma alstruktúrára bontása azt jelenti, hogy az eredeti problémát kisebb, kezelhetőbb részproblémákra bontjuk. Ezek a részproblémák függetlenek lehetnek egymástól, de összességükben az eredeti probléma megoldásához vezetnek.

Az alstruktúra két típusa létezik: átfedő részproblémák (overlapping subproblems) és független részproblémák (independent subproblems). Az átfedő részproblémák esetében a részproblémák többször is előfordulhatnak a számítás során, míg a független részproblémák egyszerűen kombinálhatók az eredeti probléma megoldásához.

Az átfedő részproblémák esetében a memoizáció (memorizálás) és a táblázat kitöltés (tabulation) technikák alkalmazhatók a redundáns számítások elkerülése érdekében. A memoizáció során a részproblémák megoldásait egy adatstruktúrában (például egy asszociatív tömbben) tároljuk, hogy a későbbiekben újra felhasználhassuk őket. A táblázat kitöltés pedig egy iteratív megközelítés, ahol egy táblázatban tároljuk a részproblémák megoldásait, és ezt fokozatosan töltjük ki, amíg el nem érjük az eredeti probléma megoldását.

#### Példa: Leghosszabb Közös Részszekvencia (LCS)

Tekintsük a leghosszabb közös részszekvencia problémát részletesebben. Adott két karakterlánc $X$ és $Y$, amelyek hossza $m$ és $n$. Az LCS-probléma célja, hogy megtaláljuk a leghosszabb közös szekvenciát ezen két karakterlánc között. Az LCS-probléma optimális alstruktúrával rendelkezik, mivel a következő rekurzív relációval írható le:

$$
LCS(X[1..m], Y[1..n]) = \begin{cases}
0 & \text{ha } m = 0 \text{ vagy } n = 0 \\
LCS(X[1..m-1], Y[1..n-1]) + 1 & \text{ha } X[m] = Y[n] \\
\max(LCS(X[1..m-1], Y[1..n]), LCS(X[1..m], Y[1..n-1])) & \text{ha } X[m] \neq Y[n]
\end{cases}
$$

Ez a reláció azt mutatja, hogy ha az utolsó karakterek megegyeznek ($X[m] = Y[n]$), akkor az LCS hossza megnövekszik eggyel, és a probléma mérete csökken mindkét karakterlánc egy-egy karakterének elhagyásával. Ha az utolsó karakterek nem egyeznek meg, akkor az LCS-t az utolsó karakterek elhagyásával kapott részproblémák maximális értéke adja.

A következő C++ kód bemutatja, hogyan valósítható meg az LCS probléma dinamikus programozással:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

// Function to find the length of the Longest Common Subsequence (LCS)
int LCS(const std::string &X, const std::string &Y) {
    int m = X.length();
    int n = Y.length();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));

    // Build the dp table in bottom-up fashion
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (X[i - 1] == Y[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }

    // The length of LCS is in the dp[m][n]
    return dp[m][n];
}

int main() {
    std::string X = "AGGTAB";
    std::string Y = "GXTXAYB";
    std::cout << "Length of LCS is " << LCS(X, Y) << std::endl;
    return 0;
}
```

A fenti kód egy táblázatot (két dimenziós vektort) használ a részproblémák megoldásainak tárolására. Az LCS hosszát iteratív módon számítjuk ki, kitöltve a táblázatot az alap esetektől kiindulva (üres karakterláncok esete) a teljes probléma megoldásáig. Az algoritmus időbeli komplexitása $O(m \cdot n)$, ahol $m$ és $n$ az input karakterláncok hossza, ami lényegesen hatékonyabb, mint az egyszerű rekurzív megoldás.

#### Példa: Mátrixlánc Szorzás (Matrix Chain Multiplication)

A Matrix Chain Multiplication probléma célja, hogy megtaláljuk a mátrixok szorzási sorrendjét úgy, hogy a szorzások számának összköltsége minimális legyen. Tegyük fel, hogy adott egy $p = \{p_0, p_1, p_2, \ldots, p_n\}$ dimenzióvektor, ahol az $i$-edik mátrix mérete $p_{i-1} \times p_i$.

Az optimalitás és az alstruktúra fogalmát alkalmazva a problémát a következőképpen oldjuk meg:

1. **Alstruktúra meghatározása**: A probléma kisebb részproblémákra bontható, ahol az $M[i, j]$ a $i$-től $j$-ig terjedő mátrixok optimális szorzási költségét jelenti. Ezeket a részproblémákat kombinálva kapjuk meg az eredeti probléma megoldását.

2. **Optimalitás kihasználása**: A mátrixok szorzásának minimális költsége a következő rekurzív relációval írható le:
   $$
   M[i, j] = \min_{i \leq k < j} \{M[i, k] + M[k+1, j] + p_{i-1} \cdot p_k \cdot p_j\}
   $$
   ahol $M[i, k]$ és $M[k+1, j]$ a részproblémák optimális megoldásai.

A következő C++ kód bemutatja a Matrix Chain Multiplication probléma dinamikus programozásos megoldását:

```cpp
#include <iostream>
#include <vector>
#include <limits.h>

// Function to find the minimum multiplication cost for a chain of matrices
int MatrixChainMultiplication(const std::vector<int> &p) {
    int n = p.size() - 1;
    std::vector<std::vector<int>> dp(n, std::vector<int>(n, 0));

    for (int length = 2; length <= n; ++length) {
        for (int i = 0; i < n - length + 1; ++i) {
            int j = i + length - 1;
            dp[i][j] = INT_MAX;
            for (int k = i; k < j; ++k) {
                int cost = dp[i][k] + dp[k+1][j] + p[i]*p[k+1]*p[j+1];
                dp[i][j] = std::min(dp[i][j], cost);
            }
        }
    }

    return dp[0][n-1];
}

int main() {
    std::vector<int> p = {1, 2, 3, 4};
    std::cout << "Minimum number of multiplications is " << MatrixChainMultiplication(p) << std::endl;
    return 0;
}
```

#### Az optimalitás és alstruktúra fontossága

Az optimalitás és alstruktúra fogalmainak megértése kulcsfontosságú a dinamikus programozási problémák felismerésében és megoldásában. Ezek a fogalmak biztosítják, hogy a problémákat hatékonyan tudjuk kisebb részproblémákra bontani, és az optimális megoldásokat újra felhasználhatjuk. Ezen alapelvek alkalmazásával jelentős mértékben csökkenthető a számítási igény, különösen nagy méretű problémák esetén.

Az optimalitás és alstruktúra olyan erős eszközök, amelyek lehetővé teszik a bonyolult algoritmikus problémák hatékony megoldását. A dinamikus programozás számos területen alkalmazható, beleértve a számítógépes tudományokat, az operációkutatást, és a gazdasági modellezést, ahol az optimalizáció és a hatékonyság kiemelt fontosságú.

#### Összegzés

Az optimalitás és alstruktúra a dinamikus programozás alapvető fogalmai, amelyek nélkülözhetetlenek a hatékony algoritmusok tervezésében. Az optimalitás lehetővé teszi a problémák optimális megoldásainak részproblémákra bontását, míg az alstruktúra segít a részproblémák szerkezetének és kapcsolatának megértésében. Ezek a fogalmak alapvető fontosságúak a dinamikus programozás sikeres alkalmazásához, és számos különböző területen, például a számítógépes tudományokban, operációkutatásban és a gazdasági modellezésben használhatók. Az alábbiakban a dinamikus programozás alkalmazásának további példáit és technikáit fogjuk megvizsgálni, beleértve a top-down és bottom-up megközelítéseket, valamint a memoizáció és táblázat kitöltés módszereit.


### 3.1.2. Top-down és bottom-up megközelítések

A dinamikus programozás két alapvető megközelítési módja a top-down (felülről lefelé) és a bottom-up (alulról felfelé) megközelítés. Mindkét technika hatékonyan alkalmazható különféle kombinatorikus problémák megoldására, és ezek alapvető különbségei, előnyei és hátrányai vannak. Ebben a fejezetben részletesen megvizsgáljuk mindkét módszert, bemutatva azok működési mechanizmusait és alkalmazási területeit.

#### Top-down megközelítés (Memoization)

A top-down megközelítés rekurzív természetű, és a probléma felbontását magas szintről kezdi. Az alapötlet az, hogy a problémát kisebb részproblémákra bontjuk, majd ezeket rekurzívan oldjuk meg. Az optimalizáció érdekében a már kiszámított részproblémák eredményeit tároljuk (memoizáljuk), hogy elkerüljük a redundáns számításokat.

##### Működési mechanizmus

1. **Rekurzív felbontás**: A probléma megoldása rekurzív módon történik. A rekurzív függvény minden hívás során további részproblémákra bontja a feladatot.
2. **Memoizáció**: A részproblémák megoldásait egy adatstruktúrában, gyakran egy hash-táblában vagy tömbben tároljuk. Ha egy részprobléma eredménye már korábban ki lett számítva, a memoizált értéket használjuk újraszámítás helyett.
3. **Optimalizáció**: Az optimális megoldás elérése érdekében az összes lehetséges részprobléma megoldását figyelembe vesszük, és a memoizáció segítségével elkerüljük a redundáns számításokat.

##### Példa

A Fibonacci-számok kiszámítása top-down megközelítéssel:

```cpp
#include <iostream>
#include <vector>

std::vector<int> memo;

int fibonacci(int n) {
    if (n <= 1) return n;
    if (memo[n] != -1) return memo[n];
    memo[n] = fibonacci(n - 1) + fibonacci(n - 2);
    return memo[n];
}

int main() {
    int n = 10;
    memo.assign(n + 1, -1);
    std::cout << "Fibonacci(" << n << ") = " << fibonacci(n) << std::endl;
    return 0;
}
```

##### Előnyök

1. **Egyszerű implementáció**: A rekurzív természet miatt az algoritmus gyakran egyszerűbb és intuitívabb.
2. **Kód tisztaság**: A memoizációval kombinált rekurzió kódja gyakran tisztább és olvashatóbb.

##### Hátrányok

1. **Rekurzív hívások költsége**: A rekurzív hívások jelentős memória- és időbeli költséget jelenthetnek, különösen nagy probléma méretek esetén.
2. **Stack Overflow veszélye**: Mély rekurzív hívások esetén a program stack overflow-t tapasztalhat.

#### Bottom-up megközelítés (Tabulation)

A bottom-up megközelítés iteratív módon közelíti meg a problémát, az alsó szintű részproblémáktól indulva fokozatosan építi fel a megoldást. Ez a módszer elkerüli a rekurziót, és gyakran hatékonyabb, ha a rekurzió mélysége túl nagy lenne.

##### Működési mechanizmus

1. **Táblázat inicializálása**: Egy táblázatot inicializálunk, amely az összes részprobléma megoldásait tárolja.
2. **Iteratív kitöltés**: A táblázatot iteratív módon töltjük ki, a legkisebb részproblémáktól indulva egészen az eredeti probléma megoldásáig.
3. **Tárolt értékek felhasználása**: A korábban kiszámított és táblázatban tárolt értékeket felhasználjuk az új részproblémák megoldására.

##### Példa

A Fibonacci-számok kiszámítása bottom-up megközelítéssel:

```cpp
#include <iostream>
#include <vector>

int fibonacci(int n) {
    if (n <= 1) return n;
    std::vector<int> fib(n + 1, 0);
    fib[1] = 1;
    for (int i = 2; i <= n; ++i) {
        fib[i] = fib[i - 1] + fib[i - 2];
    }
    return fib[n];
}

int main() {
    int n = 10;
    std::cout << "Fibonacci(" << n << ") = " << fibonacci(n) << std::endl;
    return 0;
}
```

##### Előnyök

1. **Nincs rekurzió**: Az iteratív megközelítés elkerüli a rekurziót és a stack overflow veszélyét.
2. **Hatékony memóriahasználat**: A memóriahasználat előre meghatározható és optimalizálható.

##### Hátrányok

1. **Kód bonyolultsága**: Az iteratív megközelítés kódja gyakran bonyolultabb és kevésbé intuitív, mint a rekurzív változat.
2. **Kezdő értékek inicializálása**: A táblázat inicializálása és a kezdeti értékek meghatározása néha bonyolult lehet.

#### Összehasonlítás és alkalmazási területek

A top-down és bottom-up megközelítések különböző előnyökkel és hátrányokkal rendelkeznek, és a választás gyakran a konkrét probléma jellegzetességeitől és az alkalmazási környezettől függ.

1. **Top-down megközelítés**:

    - **Előnyök**: Könnyebb megérteni és implementálni, különösen, ha a probléma természetesen rekurzív. Gyakran intuitívabb és tisztább kódot eredményez.
    - **Hátrányok**: Rekurzív hívások költségesek lehetnek nagy probléma méretek esetén. Stack overflow kockázata, ha a rekurzió mélysége túl nagy.

2. **Bottom-up megközelítés**:

    - **Előnyök**: Nincs rekurzió, így nincs stack overflow veszély. Hatékonyabb memóriahasználat és időbeli teljesítmény, különösen nagy probléma méretek esetén.
    - **Hátrányok**: Az iteratív kód bonyolultabb és kevésbé intuitív lehet. A táblázat inicializálása és kezelése néha összetett lehet.

#### Konkrét alkalmazási példák

1. **Leghosszabb közös részszekvencia (Longest Common Subsequence, LCS)**:

    - **Top-down**: A memoizáció segítségével tároljuk az előzőleg kiszámított részproblémákat.
    - **Bottom-up**: Egy két dimenziós táblázatot használunk az összes részprobléma megoldásának tárolására, és iteratív módon töltjük ki a táblázatot.

2. **Rugós hátizsák probléma (0/1 Knapsack Problem)**:

    - **Top-down**: Rekurzív hívásokkal bontjuk a problémát kisebb részproblémákra, memoizációval tárolva az eredményeket.
    - **Bottom-up**: Egy táblázatot használunk, amelyet iteratív módon töltünk ki, figyelembe véve a tárgyak súlyát és értékét.

#### Összegzés

A top-down és bottom-up megközelítések mindkettő hatékony eszközök a dinamikus programozásban, és a választás gyakran a konkrét probléma és az alkalmazási környezet függvénye. A top-down megközelítés egyszerűbb és intuitívabb lehet, míg a bottom-up megközelítés gyakran hatékonyabb idő- és memóriahasználatot eredményez. Mindkét módszer alapos megértése és alkalmazása kulcsfontosságú a dinamikus programozási technikák sikeres alkalmazásához. A következő alfejezetben a memoizáció és a táblázat kitöltés részletes technikáit vizsgáljuk meg, amelyek ezeknek a megközelítéseknek az alapját képezik.

### 3.1.3. Memoizáció és táblázat kitöltés

A dinamikus programozás két alapvető technikája a memoizáció és a táblázat kitöltés (tabulation). Ezek a technikák kulcsfontosságúak a hatékony algoritmusok tervezésében és megvalósításában, különösen olyan problémák esetében, amelyek átfedő részproblémákat tartalmaznak. Ebben az alfejezetben részletesen megvizsgáljuk mindkét technikát, bemutatva azok működését, előnyeit, hátrányait és alkalmazási területeit.

#### Memoizáció (Memoization)

A memoizáció egy top-down megközelítés, amely rekurzív hívásokkal oldja meg a problémát, miközben az előzőleg kiszámított részproblémák eredményeit tárolja egy adatstruktúrában, általában egy hash-táblában vagy tömbben. Ez a technika lehetővé teszi az ismétlődő számítások elkerülését, ami jelentősen csökkenti a futási időt.

##### Működési mechanizmus

1. **Rekurzív megoldás**: A probléma rekurzív módon történő megoldása.
2. **Tárolás**: A részproblémák eredményeit egy adatstruktúrában tároljuk.
3. **Újrafelhasználás**: Ha egy részprobléma megoldására újra szükség van, a tárolt értéket használjuk, nem számítjuk újra.

##### Példa

Vizsgáljuk meg a Fibonacci-számok kiszámítását memoizációval:

```cpp
#include <iostream>
#include <vector>

std::vector<int> memo;

int fibonacci(int n) {
    if (n <= 1) return n;
    if (memo[n] != -1) return memo[n];
    memo[n] = fibonacci(n - 1) + fibonacci(n - 2);
    return memo[n];
}

int main() {
    int n = 10;
    memo.assign(n + 1, -1);
    std::cout << "Fibonacci(" << n << ") = " << fibonacci(n) << std::endl;
    return 0;
}
```

A memoizáció előnyei és hátrányai:

##### Előnyök

1. **Hatékonyság**: Jelentős futásidő-csökkenést eredményez az ismétlődő számítások elkerülése révén.
2. **Egyszerű implementáció**: A rekurzív természet miatt az algoritmus gyakran egyszerűbb és intuitívabb.
3. **Tisztább kód**: A memoizációval kombinált rekurzió kódja gyakran tisztább és olvashatóbb.

##### Hátrányok

1. **Memóriahasználat**: A memoizáció nagy memóriaigényt jelenthet, ha sok részproblémát kell tárolni.
2. **Stack overflow veszélye**: Mély rekurzív hívások esetén a program stack overflow-t tapasztalhat.
3. **Rekurzív hívások költsége**: A rekurzív hívások jelentős memória- és időbeli költséget jelenthetnek, különösen nagy probléma méretek esetén.

#### Táblázat kitöltés (Tabulation)

A táblázat kitöltés egy bottom-up megközelítés, amely iteratív módon oldja meg a problémát. A technika lényege, hogy a legkisebb részproblémáktól indulva fokozatosan építjük fel a megoldást, egy táblázatban tárolva az eredményeket. Ez a módszer elkerüli a rekurziót, és gyakran hatékonyabb idő- és memóriahasználatot eredményez.

##### Működési mechanizmus

1. **Táblázat inicializálása**: Kezdésként egy táblázatot inicializálunk, amely az összes részprobléma megoldását tárolja.
2. **Iteratív kitöltés**: A táblázatot iteratív módon töltjük ki, a legkisebb részproblémáktól indulva egészen az eredeti probléma megoldásáig.
3. **Tárolt értékek felhasználása**: A korábban kiszámított és táblázatban tárolt értékeket felhasználjuk az új részproblémák megoldására.

##### Példa

A Fibonacci-számok kiszámítása táblázat kitöltéssel:

```cpp
#include <iostream>
#include <vector>

int fibonacci(int n) {
    if (n <= 1) return n;
    std::vector<int> fib(n + 1, 0);
    fib[1] = 1;
    for (int i = 2; i <= n; ++i) {
        fib[i] = fib[i - 1] + fib[i - 2];
    }
    return fib[n];
}

int main() {
    int n = 10;
    std::cout << "Fibonacci(" << n << ") = " << fibonacci(n) << std::endl;
    return 0;
}
```

A táblázat kitöltés előnyei és hátrányai:

##### Előnyök

1. **Nincs rekurzió**: Az iteratív megközelítés elkerüli a rekurziót és a stack overflow veszélyét.
2. **Hatékony memóriahasználat**: A memóriahasználat előre meghatározható és optimalizálható.
3. **Jobb teljesítmény**: Gyakran hatékonyabb idő- és memóriahasználat, különösen nagy probléma méretek esetén.

##### Hátrányok

1. **Kód bonyolultsága**: Az iteratív megközelítés kódja gyakran bonyolultabb és kevésbé intuitív, mint a rekurzív változat.
2. **Kezdő értékek inicializálása**: A táblázat inicializálása és a kezdeti értékek meghatározása néha bonyolult lehet.

#### Memoizáció és táblázat kitöltés összehasonlítása

A memoizáció és a táblázat kitöltés közötti választás gyakran a konkrét probléma jellegétől és az alkalmazási környezettől függ. Mindkét módszer hatékony eszköz a dinamikus programozásban, de különböző előnyökkel és hátrányokkal rendelkeznek.

1. **Hatékonyság**:

    - **Memoizáció**: Gyorsabb lehet, ha a probléma természetesen rekurzív, mivel csak a szükséges részproblémákat számítja ki.
    - **Táblázat kitöltés**: Gyakran hatékonyabb idő- és memóriahasználatot eredményez, különösen nagy probléma méretek esetén.

2. **Kód bonyolultsága**:

    - **Memoizáció**: A rekurzív természet miatt az algoritmus gyakran egyszerűbb és intuitívabb.
    - **Táblázat kitöltés**: Az iteratív megközelítés kódja gyakran bonyolultabb és kevésbé intuitív.

3. **Memóriahasználat**:

    - **Memoizáció**: Nagy memóriaigényt jelenthet, ha sok részproblémát kell tárolni.
    - **Táblázat kitöltés**: A memóriahasználat előre meghatározható és optimalizálható.

4. **Stack overflow veszélye**:

    - **Memoizáció**: Mély rekurzív hívások esetén a program stack overflow-t tapasztalhat.
    - **Táblázat kitöltés**: Az iteratív megközelítés elkerüli a stack overflow veszélyét.

#### Alkalmazási területek

1. **Leghosszabb közös részszekvencia (Longest Common Subsequence, LCS)**:

    - **Memoizáció**: A részproblémák eredményeit tároljuk egy hash-táblában, hogy elkerüljük az ismétlődő számításokat.
    - **Táblázat kitöltés**: Egy két dimenziós táblázatot használunk az összes részprobléma megoldásának tárolására, és iteratív módon töltjük ki a táblázatot.

2. **Rugós hátizsák probléma (0/1 Knapsack Problem)**:

    - **Memoizáció**: Rekurzív hívásokkal bontjuk a problémát kisebb részproblémákra, memoizációval tárolva az eredményeket.
    - **Táblázat kitöltés**: Egy táblázatot használunk, amelyet iteratív módon töltünk ki, figyelembe véve a tárgyak súlyát és értékét.

3. **Mátrixlánc Szorzás (Matrix Chain Multiplication)**:

    - **Memoizáció**: Rekurzív hívásokkal bontjuk a problémát kisebb részproblémákra, és az eredményeket egy hash-táblában tároljuk.
    - **Táblázat kitöltés**: Egy táblázatot használunk, amelyet iteratív módon töltünk ki a mátrixok optimális szorzási sorrendjének meghatározásához.

#### Összegzés

A memoizáció és a táblázat kitöltés alapvető technikák a dinamikus programozásban, és mindkettő hatékony eszköz a kombinatorikus problémák megoldásában. A memoizáció top-down megközelítése egyszerűbb és intuitívabb lehet, míg a táblázat kitöltés bottom-up megközelítése gyakran hatékonyabb idő- és memóriahasználatot eredményez. A választás gyakran a konkrét probléma jellegétől és az alkalmazási környezettől függ. A következő alfejezetben az általános lépéseket és technikákat fogjuk megvizsgálni, amelyek a dinamikus programozás sikeres alkalmazásához szükségesek.

### 3.1.4. Általános lépések dinamikus programozásnál

A dinamikus programozás egy hatékony algoritmustervezési módszer, amely különösen hasznos olyan problémák megoldására, amelyek részproblémákra bonthatók, és ezek a részproblémák átfedéseket tartalmaznak. A dinamikus programozási algoritmusok megtervezésekor számos általános lépést követhetünk, amelyek segítenek a probléma megoldásában és az algoritmus hatékonyságának optimalizálásában. Ebben az alfejezetben részletesen megvizsgáljuk ezeket az általános lépéseket, bemutatva azok alkalmazását és a gyakorlati példákban való felhasználását.

#### 1. A probléma azonosítása és a megfelelő adatstruktúra kiválasztása

Az első lépés a probléma pontos meghatározása és annak felismerése, hogy a probléma alkalmas-e dinamikus programozás alkalmazására. A következő kérdéseket kell feltenni:

- Van-e a problémának optimális alstruktúrája?
- Vannak-e átfedő részproblémák?

Ha a válasz mindkét kérdésre igen, akkor a probléma valószínűleg jól megoldható dinamikus programozással. Ezen kívül fontos kiválasztani a megfelelő adatstruktúrát a részproblémák megoldásainak tárolására. Ez lehet egy tömb, mátrix vagy más megfelelő adatstruktúra.

#### 2. A részproblémák felosztása

A következő lépés a probléma kisebb részproblémákra való bontása. Ennek során meg kell határozni a részproblémák közötti kapcsolatot, és azt, hogy hogyan lehet ezek megoldásait kombinálni az eredeti probléma megoldásához. Például a Fibonacci-számok esetében a probléma felosztása a következőképpen történik:

$$
F(n) = F(n-1) + F(n-2)
$$

#### 3. Az alap esetek meghatározása

A dinamikus programozás során fontos meghatározni azokat az alap eseteket, amelyeket nem lehet tovább bontani. Ezek az alap esetek képezik az algoritmus kiindulópontját, és az összes többi részprobléma megoldása ezekre épül. A Fibonacci-számok példájában az alap esetek a következők:

$$
F(0) = 0
$$
$$
F(1) = 1
$$

#### 4. A megoldások tárolására szolgáló adatstruktúra inicializálása

Miután meghatároztuk a részproblémákat és az alap eseteket, létre kell hozni egy adatstruktúrát a részproblémák megoldásainak tárolására. Ez az adatstruktúra lehet egy tömb, mátrix vagy más típusú adatstruktúra, amely lehetővé teszi a részproblémák megoldásainak hatékony tárolását és elérését. A Fibonacci-számok esetében ez egy tömb lehet, amelynek minden eleme egy részprobléma megoldását tárolja.

#### 5. Az optimális megoldás rekurzív meghatározása (top-down megközelítés) vagy iteratív kitöltése (bottom-up megközelítés)

A top-down megközelítés esetében a probléma megoldása rekurzívan történik, miközben az előzőleg kiszámított részproblémák eredményeit memoizáljuk. A bottom-up megközelítés esetében iteratív módon töltjük ki a táblázatot, a legkisebb részproblémáktól indulva egészen az eredeti probléma megoldásáig.

##### Top-down példa: Fibonacci-számok

```cpp
#include <iostream>
#include <vector>

std::vector<int> memo;

int fibonacci(int n) {
    if (n <= 1) return n;
    if (memo[n] != -1) return memo[n];
    memo[n] = fibonacci(n - 1) + fibonacci(n - 2);
    return memo[n];
}

int main() {
    int n = 10;
    memo.assign(n + 1, -1);
    std::cout << "Fibonacci(" << n << ") = " << fibonacci(n) << std::endl;
    return 0;
}
```

##### Bottom-up példa: Fibonacci-számok

```cpp
#include <iostream>
#include <vector>

int fibonacci(int n) {
    if (n <= 1) return n;
    std::vector<int> fib(n + 1, 0);
    fib[1] = 1;
    for (int i = 2; i <= n; ++i) {
        fib[i] = fib[i - 1] + fib[i - 2];
    }
    return fib[n];
}

int main() {
    int n = 10;
    std::cout << "Fibonacci(" << n << ") = " << fibonacci(n) << std::endl;
    return 0;
}
```

#### 6. Az eredmény visszakeresése

Miután az összes részproblémát megoldottuk és tároltuk, az eredeti probléma megoldása könnyen visszakereshető az adatstruktúrából. Az optimális megoldás az adatstruktúra megfelelő elemében található.

#### 7. Az idő- és tárigény elemzése

A dinamikus programozási algoritmusok megtervezésekor fontos az idő- és tárigény elemzése. Az időkomplexitás meghatározásához meg kell vizsgálni, hány részproblémát oldunk meg és mennyi idő szükséges egy-egy részprobléma megoldásához. A tárkomplexitás meghatározásához meg kell vizsgálni, mekkora adatstruktúrát használunk a részproblémák tárolására.

##### Példa: Fibonacci-számok idő- és tárkomplexitása

- **Top-down megközelítés**: Az időkomplexitás $O(n)$, mivel minden részproblémát egyszer oldunk meg. A tárkomplexitás $O(n)$, mivel egy tömbben tároljuk az $n$ részprobléma megoldásait.
- **Bottom-up megközelítés**: Az időkomplexitás $O(n)$, mivel iteratív módon töltjük ki a táblázatot. A tárkomplexitás $O(n)$, mivel egy tömbben tároljuk az $n$ részprobléma megoldásait.

#### 8. A megoldás helyességének és hatékonyságának ellenőrzése

A dinamikus programozási algoritmusok megvalósítása után fontos ellenőrizni a megoldás helyességét és hatékonyságát. Ez magában foglalja a megoldás tesztelését különböző bemenetekkel, az eredmények összehasonlítását a várt értékekkel és a futási idő elemzését.

##### Tesztelés és validálás

A Fibonacci-számok esetében például tesztelhetjük az algoritmust különböző bemenetekkel (például $n = 5$, $n = 10$, $n = 20$), és ellenőrizhetjük, hogy az eredmények megfelelnek-e a várt értékeknek.

```cpp
int main() {
    for (int n : {5, 10, 20}) {
        memo.assign(n + 1, -1);
        std::cout << "Fibonacci(" << n << ") = " << fibonacci(n) << std::endl;
    }
    return 0;
}
```

#### Összegzés

A dinamikus programozás egy hatékony eszköz a kombinatorikus problémák megoldásában, és az általános lépések követése segít az algoritmusok helyes és hatékony megvalósításában. A probléma azonosítása, a részproblémák felosztása, az alap esetek meghatározása, az adatstruktúra inicializálása, az optimális megoldás meghatározása, az eredmény visszakeresése, valamint az idő- és tárigény elemzése mind kulcsfontosságú lépések a dinamikus programozási algoritmusok tervezése és megvalósítása során. Ezen lépések alapos megértése és alkalmazása lehetővé teszi a dinamikus programozás széles körű alkalmazását különféle problémák megoldására.

