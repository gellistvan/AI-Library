\newpage

## 3.5. Mátrix lánc szorzás (Matrix Chain Multiplication)

A mátrix lánc szorzás egy klasszikus optimalizálási probléma a számítástudományban és a matematikában, amelynek célja a mátrixok sorozatának olyan sorrendben történő szorzása, hogy a műveletek teljes számát minimalizáljuk. A probléma alapja, hogy több mátrix szorzásánál a szorzás sorrendje befolyásolja a szükséges műveletek számát, így a megfelelő sorrend megtalálása jelentős teljesítménybeli előnyöket biztosíthat. Ez a fejezet részletesen bemutatja a mátrix lánc szorzás problémájának definícióját és alkalmazásait, ismerteti a rekurzív megoldási megközelítést, majd bemutatja, hogyan optimalizálható a probléma dinamikus programozási technikákkal.


### 3.5.1. Definíció és alkalmazások

#### Definíció

A mátrix lánc szorzás (Matrix Chain Multiplication) problémája egy olyan optimalizálási feladat, amely arra irányul, hogy több mátrix szorzásának sorrendjét úgy válasszuk meg, hogy a szükséges műveletek számát minimalizáljuk. Amikor több mátrixot szorzunk össze, a szorzás sorrendje jelentősen befolyásolja a számítási költséget. A mátrixok szorzása asszociatív, azaz a szorzás sorrendje nem befolyásolja az eredményt, de a műveletek száma változhat.

Formálisan, legyenek adott mátrixok $A_1, A_2, \ldots, A_n$, ahol az $A_i$ mátrix dimenziója $p_{i-1} \times p_i$. A cél olyan sorrendet találni a mátrixok szorzására, amely minimalizálja a szükséges skalár szorzások számát. A probléma tehát az optimális zárójelezés megtalálása, amely minimalizálja a műveletek számát.

#### Példa

Tegyük fel, hogy négy mátrixunk van:
- $A_1$ mérete $10 \times 30$
- $A_2$ mérete $30 \times 5$
- $A_3$ mérete $5 \times 60$
- $A_4$ mérete $60 \times 20$

Ha ezeket a mátrixokat különböző sorrendben szorozzuk össze, a szükséges műveletek száma drasztikusan eltérhet. Például:
1. Ha a szorzási sorrend $(A_1(A_2(A_3A_4)))$, a műveletek száma $10 \cdot 30 \cdot 5 + 10 \cdot 5 \cdot 60 + 10 \cdot 60 \cdot 20 = 1500 + 3000 + 12000 = 16500$.
2. Ha a szorzási sorrend $((A_1A_2)(A_3A_4))$, a műveletek száma $10 \cdot 30 \cdot 5 + 10 \cdot 5 \cdot 60 + 5 \cdot 60 \cdot 20 = 1500 + 3000 + 6000 = 10500$.

Az optimális zárójelezés tehát jelentős teljesítménybeli előnyt jelenthet.

#### Matematikai háttér

A mátrixok szorzásának asszociativitása azt jelenti, hogy bárhogyan is helyezzük el a zárójeleket, a végeredmény ugyanaz lesz. Azonban a szorzási műveletek száma a zárójelezéstől függ. Minden lehetséges zárójelezési módot megvizsgálni exponenciális számú lehetőséget jelent, ami nagy $n$ esetén nem hatékony.

A mátrix szorzás műveleteinek száma az alábbiak szerint alakul:
- Két mátrix $A$ (mérete $p \times q$) és $B$ (mérete $q \times r$) szorzásához $p \cdot q \cdot r$ skalár szorzás szükséges.

Az optimális zárójelezés megtalálásához a problémát kisebb részproblémákra bontjuk. Definiáljuk a következőket:

* $m[i, j]$: Az $A_i$ és $A_j$ közötti mátrixok szorzásának minimális műveletszáma.
* A cél $m[1, n]$ meghatározása.

A részproblémák rekurzív összefüggése:
$$
m[i, j] = \min_{i \leq k < j} (m[i, k] + m[k+1, j] + p_{i-1} \cdot p_k \cdot p_j)
$$

#### Alkalmazások

A mátrix lánc szorzás problémája számos alkalmazási területtel rendelkezik a matematikában és a számítástudományban, különösen azokban az esetekben, ahol nagyméretű mátrixok szorzására van szükség. Az alábbiakban néhány fontos alkalmazási területet tárgyalunk:

1. **Számítógépes grafika**:
    - A számítógépes grafikában gyakran szükség van transzformációs mátrixok alkalmazására, például forgatás, nyújtás vagy eltolás. Ezek a transzformációk gyakran több mátrix szorzatát igénylik, és az optimális zárójelezés jelentős sebességnövekedést eredményezhet a grafikai műveletek végrehajtásában.

2. **Tudományos számítások**:
    - A tudományos számításokban, különösen a lineáris algebra területén, gyakran dolgozunk nagyméretű mátrixokkal. Az optimális zárójelezés révén a mátrixok szorzása gyorsabbá válik, ami elengedhetetlen a nagy mennyiségű adat feldolgozásához és a szimulációk futtatásához.

3. **Adatbázis-kezelés**:
    - Az adatbázis-kezelő rendszerek (DBMS) gyakran végeznek lekérdezésoptimalizálást, ahol a különböző táblák közötti összekapcsolási műveletek hatékonysága kulcsfontosságú. A mátrix lánc szorzáshoz hasonló optimalizálási technikák alkalmazásával a lekérdezések végrehajtási ideje jelentősen csökkenthető.

4. **Neurális hálózatok**:
    - A neurális hálózatokban a súlyok és aktivációs mátrixok szorzása alapvető művelet. Az optimális mátrix szorzási sorrend meghatározása révén a hálózatok tanítása és futtatása gyorsabbá válhat, ami különösen fontos a nagy méretű hálózatok esetében.

5. **Gépi tanulás és adatbányászat**:
    - A gépi tanulás és adatbányászat során gyakran használnak mátrixalapú számításokat, például a főkomponens-analízis (PCA) vagy a mátrix faktorizáció során. Az optimális mátrix szorzás segíthet ezeknek az algoritmusoknak a hatékonyságát növelni.

6. **Operációkutatás**:
    - Az operációkutatásban különböző optimalizálási feladatok megoldásához mátrixműveleteket használnak. A mátrix lánc szorzás optimalizálása hozzájárulhat a lineáris programozási, hálózatelemzési és más optimalizálási problémák hatékonyabb megoldásához.

#### Következtetés

A mátrix lánc szorzás problémája egy alapvető optimalizálási feladat, amely számos gyakorlati alkalmazással rendelkezik a tudományos és mérnöki területeken. Az optimális zárójelezés megtalálása jelentős számítási költséget takaríthat meg, különösen nagyméretű mátrixok szorzása esetén. Ebben az alfejezetben áttekintettük a probléma definícióját és alapjait, valamint bemutattuk néhány gyakorlati alkalmazását. A következő alfejezetekben részletesen tárgyaljuk a rekurzív megoldási megközelítést, majd bemutatjuk a probléma dinamikus programozással történő optimalizálását.

### 3.5.2. Megoldás hagyományos úton és rekurzióval

#### Bevezetés

A mátrix lánc szorzás (Matrix Chain Multiplication) problémájának célja, hogy megtaláljuk a mátrixok sorozatának optimális szorzási sorrendjét, amely minimalizálja a szorzások számát. A szorzás sorrendje drámai módon befolyásolhatja a számítási költséget, így a feladat megoldása kulcsfontosságú a nagy teljesítményű számítógépes alkalmazásokban. Ebben az alfejezetben részletesen bemutatjuk a rekurzív megoldást, beleértve a rekurzív formulát és annak implementációját C++ nyelven.

#### Hagyományos megközelítés

A mátrixok szorzásának alapvető szabálya szerint két mátrix $A$ és $B$ szorzata akkor létezik, ha $A$ oszlopainak száma megegyezik $B$ sorainak számával. Ha $A$ mérete $p \times q$, és $B$ mérete $q \times r$, akkor a $C = AB$ mátrix mérete $p \times r$, és a szorzáshoz szükséges skalár szorzatok száma $p \cdot q \cdot r$. Több mátrix szorzásakor a szorzási sorrend jelentős hatással van a szükséges műveletek számára.

#### Rekurzív megoldás

A mátrix lánc szorzás problémájának rekurzív megközelítése a probléma kisebb részproblémákra bontásán alapul. A rekurzió segítségével minden lehetséges szorzási sorrendet figyelembe vehetünk, és kiválaszthatjuk azt, amelyik a legkevesebb műveletet igényli.

##### Rekurzív formula

Legyen adott $n$ darab mátrix $A_1, A_2, \ldots, A_n$, ahol az $A_i$ mátrix dimenziója $p_{i-1} \times p_i$. A mátrixok szorzásának optimális költsége a következő rekurzív formulával számítható ki:

$$
m[i, j] = \min_{i \leq k < j} (m[i, k] + m[k+1, j] + p_{i-1} \cdot p_k \cdot p_j)
$$

ahol:

* $m[i, j]$ az $A_i$-től $A_j$-ig terjedő mátrixok szorzásának minimális költsége,
* $p$ egy sorozat, amely a mátrixok dimenzióit tartalmazza.

##### Bázis eset

A bázis esetben, ha csak egy mátrixot kell megszoroznunk, akkor nincs szükség szorzásra, így a költség nulla:
$$
m[i, i] = 0 \quad \text{minden} \, i
$$

##### Rekurzív algoritmus lépései

1. **Bázis eset kezelése**:
    - Ha csak egy mátrix van, a szorzás költsége nulla.

2. **Rekurzív eset kezelése**:
    - Bontsuk fel a mátrixláncot minden lehetséges módon két részre, és számítsuk ki az egyes részproblémák költségét rekurzívan.
    - A teljes költség a két részprobléma költségének és a két részprobléma összeillesztésének költségeként adódik.

3. **Minimális költség kiválasztása**:
    - Minden felbontás esetén számítsuk ki a költséget, és válasszuk ki a minimális költséget adó felbontást.

#### Rekurzív algoritmus implementáció C++ nyelven

Az alábbiakban bemutatjuk a mátrix lánc szorzás rekurzív algoritmusának implementációját C++ nyelven:

```cpp
#include <iostream>
#include <vector>
#include <limits.h>

using namespace std;

// Rekurzív függvény a mátrix lánc szorzás minimális költségének kiszámítására
int MatrixChainOrder(vector<int>& p, int i, int j) {
    // Ha i egyenlő j-vel, akkor csak egy mátrix van, így nincs szükség szorzásra
    if (i == j)
        return 0;

    int minCost = INT_MAX;

    // Próbáljunk meg minden lehetséges helyen zárójeleket tenni
    for (int k = i; k < j; k++) {
        // Számoljuk ki az aktuális költséget a szorzásokra
        int cost = MatrixChainOrder(p, i, k) + MatrixChainOrder(p, k + 1, j) + p[i - 1] * p[k] * p[j];

        // Válasszuk ki a minimális költséget
        if (cost < minCost)
            minCost = cost;
    }

    return minCost;
}

int main() {
    vector<int> p = {10, 30, 5, 60, 20};
    int n = p.size();

    cout << "Minimum number of multiplications is " << MatrixChainOrder(p, 1, n - 1) << endl;

    return 0;
}
```

#### Példa a rekurzív megközelítés működésére

Tekintsük az előző példát, ahol négy mátrix van:
- $A_1$: $10 \times 30$
- $A_2$: $30 \times 5$
- $A_3$: $5 \times 60$
- $A_4$: $60 \times 20$

A dimenziókat tartalmazó sorozat: $p = [10, 30, 5, 60, 20]$.

1. **Kezdjük az első és az utolsó mátrix összeszorításával**:
    - Számoljuk ki az $A_1$ és $A_4$ közötti összes részprobléma költségét.

2. **Rekurzív részproblémák megoldása**:
    - $m[1, 4]$ költségét a következőképpen számíthatjuk ki:
     $$
     m[1, 4] = \min_{1 \leq k < 4} (m[1, k] + m[k+1, 4] + p_0 \cdot p_k \cdot p_4)
     $$
    - Vizsgáljuk meg az összes lehetséges felbontást:
        - $k = 1$: $(A_1) \cdot (A_2A_3A_4)$
        - $k = 2$: $(A_1A_2) \cdot (A_3A_4)$
        - $k = 3$: $(A_1A_2A_3) \cdot (A_4)$

3. **Minden esetben számoljuk ki a költséget**:
    - Például, ha $k = 1$:
     $$
     m[1, 1] + m[2, 4] + 10 \cdot 30 \cdot 20
     $$
    - Számoljuk ki a többi esetet is hasonlóan.

4. **Minimális költség kiválasztása**:
    - Az összes eset közül válasszuk ki a legkisebb költséget.

#### Előnyök és hátrányok

##### Előnyök

1. **Egyszerűség**:
    - A rekurzív megközelítés egyszerű és jól érthető, különösen kis méretű problémák esetén.

2. **Átláthatóság**:
    - A rekurzív algoritmus jól szemlélteti a probléma felbontását kisebb részproblémákra.

##### Hátrányok

1. **Exponenciális időbonyolultság**:
    - A rekurzív megoldás időbonyolultsága exponenciális, $O(2^n)$, ami nagy $n$ esetén nem hatékony.

2. **Redundáns számítások**:
    - Sok részproblémát többször is kiszámít, ami felesleges számítási költséget eredményez.

#### Következtetés

A mátrix lánc szorzás rekurzív megoldása egy alapvető megközelítés, amely jól szemlélteti a probléma természetét és a részproblémák rekurzív felbontását. Habár a rekurzív algoritmus egyszerű és átlátható, időbonyolultsága és redundáns számításai miatt nem hatékony nagyobb problémaméretek esetén. A következő alfejezetben bemutatjuk, hogyan optimalizálhatjuk a mátrix lánc szorzás problémáját dinamikus programozási technikák alkalmazásával, hogy jelentős teljesítményjavulást érjünk el.

### 3.5.3. Optimalizálás dinamikus programozással

#### Bevezetés

A mátrix lánc szorzás (Matrix Chain Multiplication) problémájának rekurzív megoldása jól szemlélteti a probléma felbontását kisebb részproblémákra. Azonban a rekurzív megoldás időbonyolultsága exponenciális, ami nagyobb mátrixok esetén nem hatékony. Ezen a ponton lép be a dinamikus programozás (DP) technikája, amely lehetővé teszi a probléma hatékony megoldását azáltal, hogy elkerüli a redundáns számításokat és tárolja a részproblémák eredményeit. Ebben az alfejezetben részletesen tárgyaljuk a dinamikus programozás alapú optimalizálást a mátrix lánc szorzás problémájában, valamint bemutatjuk az algoritmus implementációját C++ nyelven.

#### Dinamikus programozás alapjai

A dinamikus programozás egy olyan technika, amelyet gyakran alkalmaznak optimalizálási problémák megoldására, ahol a problémát kisebb, átfedő részproblémákra bontják. Az alapötlet az, hogy a részproblémákat egyszer oldjuk meg, és az eredményeket tároljuk, hogy később újra felhasználhassuk, így elkerülve a redundáns számításokat. A mátrix lánc szorzás problémájában a dinamikus programozás segítségével hatékonyan megtalálhatjuk az optimális szorzási sorrendet.

#### Rekurzív formulák és dinamikus programozás

A mátrix lánc szorzás optimális költségének meghatározása során a rekurzív formulát használjuk:

$$
m[i, j] = \min_{i \leq k < j} (m[i, k] + m[k+1, j] + p_{i-1} \cdot p_k \cdot p_j)
$$

A dinamikus programozás megközelítésében ezt a rekurzív formulát egy táblázat segítségével számoljuk ki, amely tárolja a részproblémák eredményeit.

#### Lépések a dinamikus programozási megoldáshoz

1. **Táblázat inicializálása**:
    - Hozzunk létre egy$n \times n$ méretű táblázatot$m$, ahol$n$ a mátrixok száma. A$m[i][j]$ cella azt a minimális költséget tárolja, amely szükséges az$A_i$-től$A_j$-ig terjedő mátrixok szorzásához.

2. **Alap esetek kezelése**:
    - Ha csak egy mátrixot kell megszoroznunk, a költség nulla:

     $$
     m[i][i] = 0 \quad \text{minden} \, i
     $$

3. **Táblázat feltöltése**:
    - A táblázatot fokozatosan töltjük fel az alsó háromszög feletti részen, növekvő$l$ (a szorzandó mátrixok számának hossza) szerint.
    - Minden$i$ és$j = i + l - 1$ esetén kiszámítjuk$m[i][j]$-t az összes lehetséges$k$ alapján:
     $$
     m[i][j] = \min_{i \leq k < j} (m[i, k] + m[k+1, j] + p_{i-1} \cdot p_k \cdot p_j)
     $$

4. **Optimalizálás**:
    - Minden részprobléma eredményét tároljuk a táblázatban, így elkerülve a redundáns számításokat.
    - Az optimális költség a táblázat$m[1][n-1]$ cellájában található.

#### Példa

Vegyünk egy konkrét példát, ahol négy mátrix van:
-$A_1$:$10 \times 30$
-$A_2$:$30 \times 5$
-$A_3$:$5 \times 60$
-$A_4$:$60 \times 20$

A dimenziókat tartalmazó sorozat:$p = [10, 30, 5, 60, 20]$.

##### Részletes lépések:

1. **Két mátrix szorzása**:

    -$m[1][2]$ költsége:

     $$
     m[1][2] = 10 \times 30 \times 5 = 1500
     $$

    -$m[2][3]$ költsége:

     $$
     m[2][3] = 30 \times 5 \times 60 = 9000
     $$

    -$m[3][4]$ költsége:

     $$
     m[3][4] = 5 \times 60 \times 20 = 6000
     $$

2. **Három mátrix szorzása**:

    -$m[1][3]$ költsége:

     $$
     m[1][3] = \min((m[1][1] + m[2][3] + 10 \times 30 \times 60), (m[1][2] + m[3][3] + 10 \times 5 \times 60))
     $$
* Első eset:

$$
m[1][1] + m[2][3] + 10 \times 30 \times 60 = 0 + 9000 + 18000 = 27000
$$

* Második eset:

$$
m[1][2] + m[3][3] + 10 \times 5 \times 60 = 1500 + 0 + 3000 = 4500
$$

* Tehát,$m[1][3] = 4500$. 
* $m[2][4]$ költsége:

$$
m[2][4] = \min((m[2][2] + m[3][4] + 30 \times 5 \times 20), (m[2][3] + m[4][4] + 30 \times 60 \times 20))
$$

* Első eset:

$$
m[2][2] + m[3][4] + 30 \times 5 \times 20 = 0 + 6000 + 3000 = 9000
$$

* Második eset:

$$
m[2][3] + m[4][4] + 30 \times 60 \times 20 = 9000 + 0 + 36000 = 45000
$$

* Tehát,$m[2][4] = 9000$.

3. **Négy mátrix szorzása**:

* $m[1][4]$ költsége:

$$
m[1][4] = \min((m[1][1] + m[2][4] + 10 \times 30 \times 20), (m[1][2] + m[3][4] + 10 \times 5 \times 20), (m[1][3] + m[4][4] + 10 \times 60 \times 20))
$$

* Első eset:

$$
m[1][1] + m[2][4] + 10 \times 30 \times 20 = 0 + 9000 + 6000 = 15000
$$

* Második eset:

$$
m[1][2] + m[3][4] + 10 \times 5 \times 20 = 1500 + 6000 + 1000 = 8500
$$

* Harmadik eset:

$$
m[1][3] + m[4][4] + 10 \times 60 \times 20 = 4500 + 0 + 12000 = 16500
$$

* Tehát,$m[1][4] = 8500$.

#### Táblázat feltöltése

A táblázat feltöltése során a következő értékeket kapjuk:

| i/j | 1   | 2    | 3    | 4    |
|-----|-----|------|------|------|
| 1   | 0   | 1500 | 4500 | 8500 |
| 2   |     | 0    | 9000 | 9000 |
| 3   |     |      | 0    | 6000 |
| 4   |     |      |      | 0    |

Az optimális szorzási költség$m[1][4] = 8500$.

#### Dinamikus programozási algoritmus implementációja C++ nyelven

Az alábbiakban bemutatjuk a mátrix lánc szorzás dinamikus programozási algoritmusának implementációját C++ nyelven:

```cpp
#include <iostream>
#include <vector>
#include <limits.h>

using namespace std;

int MatrixChainOrder(vector<int>& p) {
    int n = p.size() - 1;
    vector<vector<int>> m(n+1, vector<int>(n+1, 0));

    for (int l = 2; l <= n; l++) { // l is chain length
        for (int i = 1; i <= n - l + 1; i++) {
            int j = i + l - 1;
            m[i][j] = INT_MAX;
            for (int k = i; k <= j - 1; k++) {
                int q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j];
                if (q < m[i][j]) {
                    m[i][j] = q;
                }
            }
        }
    }

    return m[1][n];
}

int main() {
    vector<int> p = {10, 30, 5, 60, 20};
    cout << "Minimum number of multiplications is " << MatrixChainOrder(p) << endl;
    return 0;
}
```

#### Előnyök és hátrányok

##### Előnyök

1. **Hatékonyság**:
    - A dinamikus programozás drámai módon csökkenti a számítási időt a rekurzív megoldáshoz képest, mivel elkerüli a redundáns számításokat. Az időbonyolultság$O(n^3)$, ahol$n$ a mátrixok száma.

2. **Memóriafelhasználás**:
    - A térbonyolultság$O(n^2)$ a táblázat tárolásához szükséges memória miatt.

##### Hátrányok

1. **Kiegészítő memória**:
    - A dinamikus programozásnak szüksége van egy kétdimenziós táblázatra a részproblémák eredményeinek tárolásához, ami jelentős memóriafelhasználást igényel nagyobb problémák esetén.

#### Következtetés

A dinamikus programozás technikája jelentős teljesítményjavulást eredményez a mátrix lánc szorzás problémájának megoldásában a rekurzív megközelítéshez képest. A táblázatos megoldás segítségével hatékonyan elkerülhetjük a redundáns számításokat és gyorsan megtalálhatjuk az optimális szorzási sorrendet. Ez a megközelítés széles körben alkalmazható a számítástudomány és a matematikai optimalizálás különböző területein, ahol nagy méretű mátrixok szorzására van szükség. A következő részben további optimalizálási technikákat és fejlesztéseket tárgyalunk a dinamikus programozás alapú megoldásokhoz.
