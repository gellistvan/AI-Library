\newpage

## 3.3. Leghosszabb közös részsorozat (Longest Common Subsequence)

A leghosszabb közös részsorozat (LCS) problémája egy alapvető feladat a szövegfeldolgozás és a számítástudomány területén. Az LCS két vagy több szekvencia közös részsorozatainak keresésére irányul, oly módon, hogy a megtalált részsorozat legyen a leghosszabb lehetséges közös szakasz. E probléma megoldása számos gyakorlati alkalmazással bír, többek között a szöveg- és DNS-szekvenciák összehasonlításában, a verziókezelő rendszerek különböző fájlverzióinak összevetésében, valamint az adatkompressziós algoritmusok fejlesztésében. Ebben a fejezetben részletesen tárgyaljuk az LCS problémájának definícióját és alkalmazásait, bemutatjuk a rekurzív megoldási megközelítést, és végül ismertetjük az LCS algoritmust és annak implementációját.

### 3.3.1. Definíció és alkalmazások

#### Definíció

A leghosszabb közös részsorozat (Longest Common Subsequence, LCS) problémája két vagy több szekvencia közötti azon részsorozat megtalálásáról szól, amely a leghosszabb lehetséges közös szakasz. Egy részsorozat olyan szekvencia, amely egy másik szekvenciából úgy nyerhető ki, hogy elhagyunk bizonyos elemeket anélkül, hogy megváltoztatnánk a megmaradt elemek sorrendjét.

Formálisan, legyen adott két szekvencia $X = \{x_1, x_2, \ldots, x_m\}$ és $Y = \{y_1, y_2, \ldots, y_n\}$. Az $X$ és $Y$ szekvenciák egy közös részsorozata $Z = \{z_1, z_2, \ldots, z_k\}$, ahol $Z$ elemei mind $X$-ből, mind $Y$-ból származnak, és az $X$ és $Y$ sorrendjének megfelelően jelennek meg. Az LCS célja a maximális $k$ hosszúságú $Z$ megtalálása.

#### Példa

Tegyük fel, hogy két szekvenciánk van: $X = \text{"ABCBDAB"}$ és $Y = \text{"BDCABA"}$. Az $X$ és $Y$ közötti leghosszabb közös részsorozat a "BDAB", amely négy karakter hosszú. A LCS problémájának megoldása során több közös részsorozat is lehet, de a leghosszabbak közül bármelyik érvényes.

#### Alkalmazások

A leghosszabb közös részsorozat probléma számos gyakorlati területen és alkalmazásban játszik fontos szerepet, különösen az alábbiakban:

1. **Szöveg- és DNS-szekvenciák összehasonlítása**:
    - Az LCS algoritmusokat széles körben alkalmazzák biológiai szekvenciák (például DNS, RNS és fehérjék) összehasonlítására. A DNS-szekvenciák közötti leghosszabb közös részsorozat segíthet megérteni az evolúciós kapcsolatokat, azonosítani a konzervált géneket és szakaszokat, valamint segíthet a fajok közötti genetikai hasonlóságok felderítésében.

2. **Verziókezelés**:
    - A verziókezelő rendszerek (VCS), mint például Git, az LCS algoritmusokat használják a fájlok különböző verzióinak összehasonlítására és a változások nyomon követésére. Az LCS segít meghatározni, hogy mely részek változtak, melyek maradtak ugyanazok, és így hatékonyan lehet összevonni különböző változtatásokat, kezelni a konfliktusokat és biztosítani a kód integritását.

3. **Adatkompresszió**:
    - Az LCS algoritmusok adatkompressziós technikákban is alkalmazhatók. Például a szekvenciák közötti ismétlődő minták felismerése révén az adatkompressziós algoritmusok csökkenthetik az adatok tárolási méretét, miközben megőrzik az eredeti információ tartalmát. Ez különösen fontos a nagy mennyiségű adatok, például szövegek, képek és videók hatékony tárolásában és továbbításában.

4. **Szövegbányászat és természetes nyelv feldolgozás (NLP)**:
    - Az LCS technikákat használják szövegbányászati feladatokban, például dokumentumok közötti hasonlóságok keresésére, plágiumfelismerésre és szövegösszehasonlításra. Az NLP területén az LCS segíthet azonosítani a közös nyelvi mintákat, mondatszerkezeteket és a nyelvi hasonlóságokat különböző szövegek között.

5. **Genomika és proteomika**:
    - A genomika és proteomika területén az LCS algoritmusokat használják a különböző gének és fehérjeszekvenciák összehasonlítására. Az LCS lehetővé teszi a kutatók számára, hogy azonosítsák a konzervált régiókat, melyek funkcionálisan vagy evolúciósan jelentősek, valamint segítséget nyújt az új gének vagy fehérjék funkcióinak megjóslásában.

6. **E-learning és intelligens oktatási rendszerek**:
    - Az intelligens oktatási rendszerekben az LCS algoritmusokat használják a diákok által készített válaszok összehasonlítására a referencia válaszokkal. Ez segít az automatikus értékelésben, a plágium felismerésben, és a személyre szabott oktatási tartalmak generálásában, amelyek jobban illeszkednek a diákok egyéni tanulási szükségleteihez.

#### Matematikai háttér

Az LCS probléma megértéséhez szükséges néhány alapvető matematikai fogalom tisztázása. Egy szekvencia egy rendezett elemlánc, míg egy részsorozat egy olyan szekvencia, amelyet az eredeti szekvencia elemeinek elhagyásával nyerünk úgy, hogy az elemek sorrendje megmarad. Az LCS probléma célja a két szekvencia közötti legnagyobb hosszúságú részsorozat megtalálása.

Például legyen az $X$ szekvencia "ACCGGTCGAGTGCGCGGAAGCCGGCCGAA" és a $Y$ szekvencia "GTCGTTCGGAATGCCGTTGCTCTGTAAA". Az LCS ebben az esetben "GTCGTCGGAAGCCGGCCGAA", ami 19 karakter hosszú.

Az LCS problémája szorosan kapcsolódik más kombinatorikai optimalizálási problémákhoz, mint például a szerkesztési távolság (edit distance) és a legnagyobb közös előtag (longest common prefix). Ezek a problémák gyakran együtt jelennek meg a bioinformatikai és szövegfeldolgozási alkalmazásokban.

#### Összefoglalás

A leghosszabb közös részsorozat probléma egy alapvető és széles körben alkalmazott probléma a számítástudományban és a bioinformatikában. A probléma célja a két szekvencia közötti leghosszabb közös részsorozat megtalálása, amely számos gyakorlati alkalmazással bír. Az LCS algoritmusok hatékony megoldásokat kínálnak a szöveg- és DNS-szekvenciák összehasonlítására, a verziókezelésre, az adatkompresszióra, a szövegbányászatra, valamint a genomikai és proteomikai kutatásokra. A következő alfejezetekben részletesen tárgyaljuk az LCS megoldási módszereit, beleértve a rekurzív megközelítést és a dinamikus programozáson alapuló algoritmusokat.

### 3.3.2. Megoldás rekurzióval

A leghosszabb közös részsorozat (LCS) probléma rekurzív megoldása az egyik alapvető megközelítés, amely a probléma természetéből adódóan jól szemlélteti a probléma felbontását kisebb részproblémákra. Ebben az alfejezetben részletesen tárgyaljuk a rekurzív megoldást, beleértve a rekurzió működését, a rekurzív formulát és annak implementációját.

#### A rekurzió alapelvei

A rekurzió egy olyan technika, amelyben egy függvény önmagát hívja meg, hogy egy nagyobb problémát kisebb részproblémákra bontson. Az LCS probléma esetében a rekurzív megoldás azon az elven alapul, hogy a két szekvencia közötti leghosszabb közös részsorozat megtalálása az első karakterek összehasonlításával és a szekvenciák további részének vizsgálatával történik.

#### Rekurzív formula

Legyen $X = \{x_1, x_2, \ldots, x_m\}$ és $Y = \{y_1, y_2, \ldots, y_n\}$ a két szekvencia. A leghosszabb közös részsorozat hossza $L(X, Y)$ az alábbi rekurzív formulával számítható ki:

1. **Bázis esetek**:
    - Ha az egyik szekvencia üres ($m = 0$ vagy $n = 0$), akkor a leghosszabb közös részsorozat hossza 0:
      $$
      L(X, Y) = 0 \quad \text{ha} \, m = 0 \, \text{vagy} \, n = 0
      $$

2. **Általános esetek**:
    - Ha az utolsó karakterek megegyeznek ($x_m = y_n$), akkor az LCS az $x_m$-et és az előző karakterek LCS-ét tartalmazza:
      $$
      L(X, Y) = 1 + L(\{x_1, x_2, \ldots, x_{m-1}\}, \{y_1, y_2, \ldots, y_{n-1}\})
      $$
    - Ha az utolsó karakterek nem egyeznek meg ($x_m \neq y_n$), akkor az LCS hossza a két lehetőség maximuma:
      $$
      L(X, Y) = \max(L(\{x_1, x_2, \ldots, x_{m-1}\}, Y), L(X, \{y_1, y_2, \ldots, y_{n-1}\}))
      $$

#### Példa a rekurzió működésére

Vegyük példának az alábbi szekvenciákat:
- $X = \text{"ABCBDAB"}$
- $Y = \text{"BDCABA"}$

A rekurzív megoldás során az alábbi lépéseket követjük:

1. Összehasonlítjuk az utolsó karaktereket: $B$ és $A$. Mivel nem egyeznek, a következő két esetet vizsgáljuk:
    - Az $X$ utolsó karakter nélküli szekvenciája ($\text{"ABCBDAB"} \rightarrow \text{"ABCBDAB"}$) és az $Y$ teljes szekvenciája.
    - Az $Y$ utolsó karakter nélküli szekvenciája ($\text{"BDCABA"} \rightarrow \text{"BDCABA"}$) és az $X$ teljes szekvenciája.

2. Az egyes esetekben folytatjuk az utolsó karakterek összehasonlítását és a szekvenciák részproblémákra bontását, amíg el nem érjük a bázis eseteket.

#### Rekurzív algoritmus pseudocode

Az alábbiakban bemutatjuk az LCS rekurzív algoritmusának pseudocode formáját:

```
function LCS(X, Y, m, n):
    if m == 0 or n == 0:
        return 0
    if X[m-1] == Y[n-1]:
        return 1 + LCS(X, Y, m-1, n-1)
    else:
        return max(LCS(X, Y, m-1, n), LCS(X, Y, m, n-1))
```

#### C++ implementáció

Az alábbiakban bemutatjuk a rekurzív algoritmus C++ nyelvű megvalósítását:

```cpp
#include <iostream>
#include <string>
using namespace std;

int LCS(string X, string Y, int m, int n) {
    if (m == 0 || n == 0)
        return 0;
    if (X[m-1] == Y[n-1])
        return 1 + LCS(X, Y, m-1, n-1);
    else
        return max(LCS(X, Y, m-1, n), LCS(X, Y, m, n-1));
}

int main() {
    string X = "ABCBDAB";
    string Y = "BDCABA";
    int m = X.length();
    int n = Y.length();
    cout << "Length of LCS is " << LCS(X, Y, m, n) << endl;
    return 0;
}
```

#### Elemzés

A rekurzív megoldás egyszerű és jól érthető, de nem hatékony nagyobb szekvenciák esetén, mivel a számítási bonyolultsága exponenciális, $O(2^{\max(m, n)})$. Ez azért van, mert sok részproblémát többször is kiszámít, ami jelentős redundanciát eredményez.

#### Optimális megoldás: Memoizáció

A rekurzív megoldás hatékonyságának javítására memoizációt alkalmazhatunk, amely egy olyan technika, amely eltárolja a már kiszámított részproblémák eredményeit. Ezzel elkerülhető a redundáns számítás és jelentősen csökkenthető az időbonyolultság.

A memoizált rekurzív algoritmus egy táblázatot (általában egy kétdimenziós tömböt) használ az LCS hosszak tárolására a különböző részproblémák esetében. Ezzel a technikával az időbonyolultság $O(m \cdot n)$-re csökken, ahol $m$ és $n$ a két szekvencia hossza.

#### Következtetés

A rekurzív megoldás fontos szerepet játszik az LCS probléma megértésében és a részproblémák felismerésében. Bár a tiszta rekurzív megközelítés nem hatékony nagy szekvenciák esetén, az alapelvek megértése lehetővé teszi hatékonyabb algoritmusok, például a memoizáció és a dinamikus programozás alkalmazását. A következő alfejezetben részletesen tárgyaljuk az LCS algoritmus és implementációját dinamikus programozással, amely jelentősen javítja a teljesítményt és skálázhatóságot.

### 3.3.3. LCS algoritmus és implementáció

A leghosszabb közös részsorozat (Longest Common Subsequence, LCS) probléma hatékony megoldása dinamikus programozási technikák alkalmazásával történik. A dinamikus programozás (DP) segítségével a rekurzív megoldás összes részproblémáját tároljuk és újrafelhasználjuk, így jelentősen csökkentve az idő- és térbonyolultságot. Ebben az alfejezetben részletesen tárgyaljuk az LCS algoritmus működését, annak dinamikus programozási megközelítését, valamint bemutatjuk a megoldás implementációját.

#### Dinamikus programozás alapjai az LCS problémában

A dinamikus programozás alapötlete az, hogy a problémát kisebb részproblémákra bontjuk, majd ezeket a részproblémákat egyszer megoldjuk és eltároljuk. Az LCS problémában a két szekvencia minden egyes prefixére kiszámítjuk a leghosszabb közös részsorozatot, és ezeket az eredményeket egy táblázatban tároljuk.

#### Lépések az LCS algoritmushoz

1. **Táblázat inicializálása**:
    - Hozzunk létre egy $(m+1) \times (n+1)$ méretű kétdimenziós táblázatot $L$, ahol $m$ az $X$ szekvencia hossza, $n$ pedig a $Y$ szekvencia hossza.
    - A $L[i][j]$ cella azt a leghosszabb közös részsorozatot tárolja, amely az $X$ első $i$ karaktere és a $Y$ első $j$ karaktere között található.

2. **Alap esetek feltöltése**:
    - Ha az egyik szekvencia hossza 0, akkor az LCS hossza is 0.
      $$
      L[i][0] = 0 \quad \text{minden} \, i = 0, 1, \ldots, m
      $$
      $$
      L[0][j] = 0 \quad \text{minden} \, j = 0, 1, \ldots, n
      $$

3. **Táblázat feltöltése**:
    - Töltsük fel a táblázatot a következő szabályok alapján:
        - Ha az aktuális karakterek megegyeznek $( X[i-1] == Y[j-1] )$, akkor a cella értéke az előző karakterekhez tartozó LCS értéke plusz egy.
          $$
          L[i][j] = L[i-1][j-1] + 1
          $$
        - Ha az aktuális karakterek nem egyeznek, akkor a cella értéke az előző karakterek közül a maximális LCS érték.
          $$
          L[i][j] = \max(L[i-1][j], L[i][j-1])
          $$

#### Példa

Vegyük az alábbi két szekvenciát:
- $X = \text{"ABCBDAB"}$
- $Y = \text{"BDCABA"}$

A táblázat inicializálása és feltöltése a következő lépésekben történik:

1. **Inicializálás**:
    - Készítsünk egy $8 \times 7$ méretű táblázatot (a szekvenciák hosszához egyet hozzáadva).

2. **Táblázat feltöltése**:
    - Az első sor és első oszlop 0-val van feltöltve, mivel bármely szekvencia és egy üres szekvencia közös részsorozata 0.
    - A táblázat feltöltése az alábbi módon történik (példa):

|   |   | B | D | C | A | B | A |
|---|---|---|---|---|---|---|---|
|   | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| A | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| B | 0 | 1 | 1 | 1 | 1 | 2 | 2 |
| C | 0 | 1 | 1 | 2 | 2 | 2 | 2 |
| B | 0 | 1 | 1 | 2 | 2 | 3 | 3 |
| D | 0 | 1 | 2 | 2 | 2 | 3 | 3 |
| A | 0 | 1 | 2 | 2 | 3 | 3 | 4 |
| B | 0 | 1 | 2 | 2 | 3 | 4 | 4 |

A táblázat utolsó cellájában lévő érték, $L[7][6]$, adja meg a leghosszabb közös részsorozat hosszát, amely ebben az esetben 4.

#### LCS visszafejtése

A táblázat feltöltése után az LCS értéke kinyerhető az utolsó cellából. Azonban magát a leghosszabb közös részsorozatot is vissza kell fejtenünk. Ehhez a következő lépéseket kell követni:

1. **Indulás az utolsó cellából**:
    - Kezdjük $L[m][n]$ cellából.

2. **Karakterek összehasonlítása**:
    - Ha $X[i-1] == Y[j-1]$, akkor ez a karakter része az LCS-nek, és lépjünk vissza mindkét szekvenciában egy-egy karakterrel (balra és fel).

3. **Maximális értékek keresése**:
    - Ha $X[i-1] \neq Y[j-1]$, akkor lépjünk arra a cellára, amelyik a nagyobb értéket tartalmazza (balra vagy fel).

4. **LCS felépítése**:
    - Addig folytassuk, amíg el nem érünk a táblázat elejére.

#### Algoritmus pseudocode

Az alábbiakban bemutatjuk az LCS algoritmus pseudocode formáját:

```
function LCS(X, Y):
    m = length(X)
    n = length(Y)
    L = array(0..m, 0..n)

    for i from 0 to m:
        for j from 0 to n:
            if i == 0 or j == 0:
                L[i][j] = 0
            else if X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])

    return L[m][n]

function printLCS(X, Y, L):
    i = length(X)
    j = length(Y)
    LCS = []

    while i > 0 and j > 0:
        if X[i-1] == Y[j-1]:
            LCS.append(X[i-1])
            i -= 1
            j -= 1
        elif L[i-1][j] > L[i][j-1]:
            i -= 1
        else:
            j -= 1

    LCS.reverse()
    return ''.join(LCS)
```

#### C++ implementáció

Az alábbiakban bemutatjuk az LCS algoritmus és a visszafejtés C++ nyelvű megvalósítását:

```cpp
#include <iostream>
#include <vector>
#include <string>
using namespace std;

int LCS(string X, string Y, vector<vector<int>>& L) {
    int m = X.length();
    int n = Y.length();
    
    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) {
            if (i == 0 || j == 0)
                L[i][j] = 0;
            else if (X[i-1] == Y[j-1])
                L[i][j] = L[i-1][j-1] + 1;
            else
                L[i][j] = max(L[i-1][j], L[i][j-1]);
        }
    }
    return L[m][n];
}

string printLCS(string X, string Y, vector<vector<int>>& L) {
    int i = X.length();
    int j = Y.length();
    string lcs = "";
    
    while (i > 0 && j > 0) {
        if (X[i-1] == Y[j-1]) {
            lcs = X[i-1] + lcs;
            i--;
            j--;
        } else if (L[i-1][j] > L[i][j-1])
            i--;
        else
            j--;
    }
    return lcs;
}

int main() {
    string X = "ABCBDAB";
    string Y = "BDCABA";
    int m = X.length();
    int n = Y.length();
    vector<vector<int>> L(m+1, vector<int>(n+1));
    
    cout << "Length of LCS is " << LCS(X, Y, L) << endl;
    cout << "LCS is " << printLCS(X, Y, L) << endl;
    
    return 0;
}
```

#### Idő- és térbonyolultság

Az LCS algoritmus dinamikus programozással történő megoldása jelentős javulást eredményez a rekurzív megoldáshoz képest. Az időbonyolultság $O(m \cdot n)$, ahol $m$ és $n$ a szekvenciák hossza. A térbonyolultság szintén $O(m \cdot n)$ a táblázat miatt.

#### Következtetés

A leghosszabb közös részsorozat problémájának dinamikus programozási megközelítése hatékony és skálázható megoldást kínál, amely lehetővé teszi a nagy szekvenciák gyors összehasonlítását. A részletes táblázatos megközelítés és a visszafejtési technika révén nemcsak a leghosszabb közös részsorozat hosszát, hanem magát a részsorozatot is megtalálhatjuk. Ez a megközelítés széles körben alkalmazható a bioinformatika, a szövegfeldolgozás, a verziókezelés és sok más területen.