\newpage

## 3.2.   Hátizsák probléma (Knapsack Problem)

A hátizsák probléma az algoritmusok területén egy klasszikus optimalizálási probléma, amely gyakran kerül elő különféle valós életbeli és elméleti alkalmazásokban. A probléma lényege, hogy adott egy hátizsák, amelynek van egy maximális teherbíró kapacitása, valamint adott egy halmaz különböző értékű és súlyú tárgyakból. A cél az, hogy úgy válasszuk ki a tárgyakat, hogy azok összértéke a lehető legnagyobb legyen, miközben a hátizsák kapacitása nem léphető túl. E fejezetben különböző típusú hátizsák problémákat fogunk megvizsgálni, beleértve a 0/1 hátizsák problémát, a tört hátizsák problémát, valamint a megoldási módszereket rekurzióval, visszavezetéssel és dinamikus programozással. A dinamikus programozás különösen fontos eszköz ezen problémák hatékony megoldásában, ezért részletesen tárgyaljuk alkalmazását és előnyeit.

### 3.2.1. 0/1 hátizsák probléma

A 0/1 hátizsák probléma egy alapvető optimalizálási probléma, amelynek számos alkalmazási területe van az operációkutatás, a számítógépes tudományok és a gazdaságtudományok terén. A probléma leírása és megértése fontos lépés az algoritmusok és a dinamikus programozás technikáinak mélyebb megismerése felé. Ebben a fejezetben részletesen áttekintjük a 0/1 hátizsák problémát, annak matematikai modelljét, valamint megoldási módszereit.

#### Probléma leírása

A 0/1 hátizsák probléma során adott egy $W$ maximális kapacitású hátizsák, valamint $n$ darab tárgy, ahol minden tárgyhoz tartozik egy $w_i$ súly és egy $v_i$ érték, ahol $i \in \{1, 2, \ldots, n\}$. A cél az, hogy kiválasszuk a tárgyak egy olyan részhalmazát, hogy azok összsúlya ne haladja meg a hátizsák kapacitását, és az összértékük maximális legyen. Fontos megjegyezni, hogy minden tárgyból legfeljebb egy darabot választhatunk, innen ered a "0/1" elnevezés.

Matematikailag a problémát a következőképpen írhatjuk le:

$$
\text{Maximalizálni:} \sum_{i=1}^{n} v_i x_i
$$
$$
\text{Feltételek:} \sum_{i=1}^{n} w_i x_i \leq W
$$
$$
x_i \in \{0, 1\}, \quad i \in \{1, 2, \ldots, n\}
$$

ahol $x_i$ egy bináris változó, amely 1, ha az $i$-edik tárgyat kiválasztjuk, és 0, ha nem.

#### Matematikai Modell

A 0/1 hátizsák probléma egy klasszikus példa a kombinatorikus optimalizálási problémák közül, amelyekben a cél egy adott halmaz részhalmazának kiválasztása, hogy maximalizáljuk (vagy minimalizáljuk) egy adott objektív függvényt. A probléma NP-nehéz, ami azt jelenti, hogy jelenlegi ismereteink szerint nincs rá polinomiális idejű megoldás, de heurisztikus és megközelítő algoritmusokkal hatékonyan kezelhető gyakorlati méretű esetekben.

#### Rekurzív megoldás

A probléma megoldásának egyik legegyszerűbb módszere a rekurzív megközelítés, amely azonban exponenciális időbeli bonyolultsággal jár. A rekurzív algoritmus lényege, hogy minden egyes tárgyról eldöntjük, hogy belevesszük-e a hátizsákba vagy sem, majd rekurzívan megoldjuk a kisebb alproblémákat.

A rekurzív megoldás pseudo-kódja a következőképpen néz ki:

```cpp
int knapsack_recursive(int W, int wt[], int val[], int n) {
    // Base Case
    if (n == 0 || W == 0)
        return 0;

    // If weight of the nth item is more than Knapsack capacity W, then
    // this item cannot be included in the optimal solution
    if (wt[n-1] > W)
        return knapsack_recursive(W, wt, val, n-1);

    // Return the maximum of two cases:
    // (1) nth item included
    // (2) not included
    else
        return max(val[n-1] + knapsack_recursive(W-wt[n-1], wt, val, n-1),
                   knapsack_recursive(W, wt, val, n-1));
}
```

#### Dinamikus Programozás

A dinamikus programozás (DP) technikája hatékonyabb megoldást kínál a 0/1 hátizsák problémára azáltal, hogy elkerüli az ismétlődő számításokat, amelyek a rekurzív megközelítés során jelentkeznek. A DP megoldás lényege, hogy egy táblázatban tároljuk az alproblémák megoldásait, és ezek alapján építjük fel a teljes probléma megoldását.

A DP megközelítés esetén létrehozunk egy $K$ kétdimenziós tömböt, ahol $K[i][w]$ az az érték, amely elérhető az első $i$ tárgy és $w$ súlykorlát mellett. Az algoritmus feltöltési szabálya a következő:

$$
K[i][w] = \begin{cases}
0 & \text{ha } i = 0 \text{ vagy } w = 0 \\
K[i-1][w] & \text{ha } w_i > w \\
\max(K[i-1][w], v_i + K[i-1][w-w_i]) & \text{egyébként}
\end{cases} 
$$

Az algoritmus pseudo-kódja a következőképpen néz ki:

```cpp
int knapsack_dp(int W, int wt[], int val[], int n) {
    int K[n+1][W+1];

    // Build table K[][] in bottom up manner
    for (int i = 0; i <= n; i++) {
        for (int w = 0; w <= W; w++) {
            if (i == 0 || w == 0)
                K[i][w] = 0;
            else if (wt[i-1] <= w)
                K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]], K[i-1][w]);
            else
                K[i][w] = K[i-1][w];
        }
    }

    return K[n][W];
}
```

#### Elemzés

A dinamikus programozási megközelítés időbeli komplexitása $O(nW)$, ahol $n$ a tárgyak száma és $W$ a hátizsák kapacitása. Ez lényegesen hatékonyabb a rekurzív megoldás exponenciális idejű bonyolultságához képest. Az algoritmus térbeli komplexitása szintén $O(nW)$, mivel egy kétdimenziós tömböt használunk az alproblémák megoldásainak tárolására.

#### Gyakorlati alkalmazások

A 0/1 hátizsák probléma számos gyakorlati alkalmazási területtel rendelkezik, többek között:
- Készletezési és raktározási problémák
- Erőforrás allokáció és projektmenedzsment
- Pénzügyi tervezés és portfólió optimalizálás
- Kiválasztási problémák, például útvonaltervezés és készletezés

Ezek az alkalmazások jól mutatják a probléma fontosságát és az optimális megoldások keresésének jelentőségét a különféle területeken.

#### Következtetések

A 0/1 hátizsák probléma egy alapvető optimalizálási probléma, amelynek megoldása jelentős kihívásokat rejt magában, különösen nagy méretű problémák esetén. A dinamikus programozás hatékony eszközt kínál e kihívások leküzdésére, lehetővé téve az optimális megoldások hatékony és gyors megtalálását. E fejezetben bemutatott módszerek és algoritmusok alapvetőek a kombinatorikus optimalizálás és az algoritmuselmélet terén, és fontos szerepet játszanak számos gyakorlati alkalmazásban is.



### 3.2.2. Megoldás rekurzióval

A rekurzív megoldás lényege, hogy a probléma megoldását kisebb részproblémákra bontjuk. A rekurzív formulák alapja a következő:

1. Ha $n$ a tárgyak száma és $W$ a hátizsák maximális kapacitása, akkor a probléma megoldása függ:
   - az $n-1$ tárggyal megoldott probléma eredményétől, amennyiben az $n$ tárgy nem kerül be a hátizsákba,
   - az $n-1$ tárggyal és $W - w_n$ kapacitással megoldott probléma eredményétől, amennyiben az $n$ tárgy bekerül a hátizsákba.

A fenti megfontolások alapján a következő rekurzív relációt kapjuk:

$$
K(n, W) = \max \left( K(n-1, W), v_n + K(n-1, W - w_n) \right)
$$

ahol $K(n, W)$ a maximális érték, amit az első $n$ tárggyal és $W$ kapacitású hátizsákkal elérhetünk.

#### Bázis esetek

A rekurzív megoldás bázisesetei:
- Ha nincs tárgy vagy a hátizsák kapacitása 0, akkor az elérhető maximális érték 0:
  $$
  K(0, W) = 0
  $$
  $$
  K(n, 0) = 0
  $$

#### Pseudocode

A rekurzív algoritmus megvalósítása pseudocode formájában a következőképpen néz ki:

```
function KnapSack(n, W)
    if n == 0 or W == 0
        return 0
    if w[n-1] > W
        return KnapSack(n-1, W)
    else
        return max(KnapSack(n-1, W), v[n-1] + KnapSack(n-1, W - w[n-1]))
```

#### C++ kód

A fenti pseudocode C++ nyelvű megvalósítása:

```cpp
#include <iostream>

#include <vector>
using namespace std;

int KnapSack(int W, vector<int>& w, vector<int>& v, int n) {
    if (n == 0 || W == 0)
        return 0;
    if (w[n-1] > W)
        return KnapSack(W, w, v, n-1);
    else
        return max(KnapSack(W, w, v, n-1), v[n-1] + KnapSack(W - w[n-1], w, v, n-1));
}

int main() {
    vector<int> values = {60, 100, 120};
    vector<int> weights = {10, 20, 30};
    int W = 50;
    int n = values.size();
    cout << "Maximum value in Knapsack = " << KnapSack(W, weights, values, n) << endl;
    return 0;
}
```

#### Elemzés

A rekurzív megoldás egyszerű és jól követhető, de nem hatékony nagyobb adathalmazokra, mivel az időbonyolultsága exponenciális, $O(2^n)$. Ez azért van, mert sok részproblémát többször is kiszámít. Ez a megoldás azonban jól szemlélteti a dinamikus programozás alapjait, ahol a memoizációval jelentősen csökkenthető a számítási igény. Ezt a következő fejezetben részletesen tárgyaljuk.

### 3.2.3. Megoldás dinamikus programozással

A dinamikus programozás (DP) hatékony megoldást kínál számos optimalizálási problémára, beleértve a hátizsák problémát is. Míg a rekurzív megközelítés sok részproblémát többször is kiszámít, addig a dinamikus programozás ezeket a részproblémákat memoizációval vagy táblázatos formában tárolja, így elkerülve a redundáns számításokat és jelentősen csökkentve az időbonyolultságot.

#### A dinamikus programozás alapjai

A dinamikus programozás alapelve, hogy egy bonyolult problémát kisebb, átfedő részproblémákra bontunk, majd ezeket a részproblémákat egyszer megoldjuk és az eredményüket eltároljuk. Az így kapott részmegoldásokból építjük fel az eredeti probléma megoldását. A 0/1 hátizsák probléma dinamikus programozás segítségével történő megoldásánál egy kétdimenziós táblázatot használunk, ahol a sorok a tárgyak számát, az oszlopok pedig a hátizsák lehetséges kapacitásait reprezentálják.

#### A táblázatos megoldás részletei

A dinamikus programozás során egy $K[n+1][W+1]$ méretű táblázatot használunk, ahol $n$ a tárgyak száma, $W$ pedig a hátizsák kapacitása. A táblázat celláiban az $i$-edik tárgy és a $j$ kapacitású hátizsák esetén elérhető maximális értéket tároljuk. Az $i$-edik tárgy hozzáadása előtt és után az értékek közötti választás alapján töltjük fel a táblázatot.

##### Alap esetek

Az alap esetek kezelésére a következő szabályokat alkalmazzuk:
1. Ha nincs tárgy ($i = 0$), vagy a hátizsák kapacitása 0 ($j = 0$), akkor a maximális érték 0:
   $$
   K[0][j] = 0 \quad \text{minden} \, j
   $$
   $$
   K[i][0] = 0 \quad \text{minden} \, i
   $$

##### Általános eset

Az általános esetben két lehetőséget kell megvizsgálni:
1. Az $i$-edik tárgy nem kerül be a hátizsákba: ebben az esetben az érték ugyanaz, mint az előző tárgynál ugyanazon kapacitás mellett:
   $$
   K[i][j] = K[i-1][j]
   $$
2. Az $i$-edik tárgy bekerül a hátizsákba: ebben az esetben az $i$-edik tárgy értékét ($v_i$) hozzáadjuk a maradék kapacitással ($j - w_i$) elérhető maximális értékhez:
   $$
   K[i][j] = v_i + K[i-1][j - w_i]
   $$

A táblázat feltöltése során minden egyes $i$ és $j$ esetén a maximális értéket a következőképpen számítjuk ki:
$$
K[i][j] = \max \left( K[i-1][j], v_i + K[i-1][j - w_i] \right)
$$

#### Példa

Vegyük példának az alábbi tárgyakat és kapacitást:

| Tárgy | Súly (w) | Érték (v) |
|-------|----------|-----------|
| 1     | 10       | 60        |
| 2     | 20       | 100       |
| 3     | 30       | 120       |

A hátizsák kapacitása $W = 50$. A dinamikus programozás táblázat feltöltése a következőképpen történik:

1. Inicializálás:
   $$
   K[0][j] = 0 \quad \text{minden} \, j
   $$
   $$
   K[i][0] = 0 \quad \text{minden} \, i
   $$

2. Táblázat feltöltése:

|       | 0  | 10 | 20 | 30 | 40 | 50 |
|-------|----|----|----|----|----|----|
| 0     | 0  | 0  | 0  | 0  | 0  | 0  |
| 1     | 0  | 60 | 60 | 60 | 60 | 60 |
| 2     | 0  | 60 | 100| 160| 160| 160|
| 3     | 0  | 60 | 100| 160| 180| 220|

A maximális érték, amit a hátizsákban elérhetünk, a $K[n][W]$ értéke, ami ebben az esetben 220.

#### C++ kód

Az alábbiakban bemutatjuk a dinamikus programozás megoldás C++ nyelvű implementációját:

```cpp
#include <iostream>

#include <vector>
using namespace std;

int knapSack(int W, vector<int>& w, vector<int>& v, int n) {
    vector<vector<int>> K(n+1, vector<int>(W+1));

    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= W; j++) {
            if (i == 0 || j == 0)
                K[i][j] = 0;
            else if (w[i-1] <= j)
                K[i][j] = max(v[i-1] + K[i-1][j - w[i-1]], K[i-1][j]);
            else
                K[i][j] = K[i-1][j];
        }
    }
    return K[n][W];
}

int main() {
    vector<int> values = {60, 100, 120};
    vector<int> weights = {10, 20, 30};
    int W = 50;
    int n = values.size();
    cout << "Maximum value in Knapsack = " << knapSack(W, weights, values, n) << endl;
    return 0;
}
```

#### Elemzés és időbonyolultság

A dinamikus programozás megközelítése lényegesen hatékonyabb a rekurzív megoldásnál. Az időbonyolultsága $O(nW)$, ahol $n$ a tárgyak száma és $W$ a hátizsák kapacitása. A térbonyolultsága szintén $O(nW)$ a kétdimenziós táblázat miatt.

#### Következtetés

A dinamikus programozás egy hatékony eszköz a hátizsák probléma megoldására, különösen nagyobb adathalmazok esetén. A rekurzív megoldás alapelveinek megértése után a dinamikus programozás alkalmazása lehetőséget nyújt a redundáns számítások elkerülésére, ezáltal jelentős teljesítményjavulást eredményezve. A táblázatos megoldás áttekinthetősége és a részproblémák eredményeinek tárolása révén egyértelművé válik, hogyan érjük el az optimális megoldást lépésről lépésre.

### 3.2.4. Tört hátizsák probléma

A tört hátizsák probléma (fractional knapsack problem) az egyik legismertebb és leggyakrabban vizsgált optimalizálási probléma, amely több különböző megoldási technikát igényel a klasszikus 0/1 hátizsák problémához képest. Ebben a fejezetben részletesen tárgyaljuk a tört hátizsák probléma matematikai alapjait, megoldási módszereit és a kapcsolódó algoritmusokat.

#### A tört hátizsák probléma definíciója

A tört hátizsák probléma esetén egy $n$ darab tárgyból álló halmaz adott, ahol minden tárgyhoz tartozik egy súly ($w_i$) és egy érték ($v_i$). A cél, hogy egy $W$ maximális teherbírású hátizsákba úgy válogassuk be a tárgyakat vagy azok tört részét, hogy a hátizsákba kerülő tárgyak összsúlya ne haladja meg $W$-t, és az összértékük maximális legyen. Ellentétben a 0/1 hátizsák problémával, itt megengedett a tárgyak tört részének a kiválasztása is.

#### Matematikai formalizmus

A tört hátizsák probléma formalizmusát a következőképpen fogalmazhatjuk meg:

Legyen adott egy $n$ elemű tárgyhalmaz, ahol minden tárgyhoz tartozik egy súly $w_i$ és egy érték $v_i$. A cél az alábbi kifejezés maximalizálása:

$$
\text{Maximalizálandó: } \sum_{i=1}^{n} v_i \cdot x_i
$$

Olyan feltételek mellett, hogy:

$$
\sum_{i=1}^{n} w_i \cdot x_i \leq W
$$
$$
0 \leq x_i \leq 1 \quad \text{minden} \, i = 1, 2, \ldots, n
$$

Ahol $x_i$ a kiválasztott $i$-edik tárgy tört része.

#### Megoldási stratégia: Greedy algoritmus

A tört hátizsák probléma hatékonyan megoldható egy mohó (greedy) algoritmus segítségével, mivel a probléma optimális részproblémákból áll. A mohó algoritmus lényege, hogy mindig azt a tárgyat vagy annak tört részét választja ki, amelyik az aktuális legmagasabb érték/súly aránnyal rendelkezik, amíg a hátizsák kapacitása megengedi.

##### Lépések:

1. **Érték/súly arány kiszámítása**:
   Minden tárgyra kiszámítjuk az érték/súly arányt:
   $$
   r_i = \frac{v_i}{w_i}
   $$

2. **Tárgyak rendezése**:
   A tárgyakat csökkenő $r_i$ arány szerint rendezzük.

3. **Tárgyak kiválasztása**:
   Kezdjük a legnagyobb $r_i$-val rendelkező tárggyal, és amennyiben lehetséges, teljes egészében vegyük be a hátizsákba. Ha a tárgy súlya meghaladja a rendelkezésre álló kapacitást, csak annyit veszünk belőle, amennyit a hátizsák elbír.

##### Algoritmus pseudocode

Az alábbiakban bemutatjuk a greedy algoritmus pseudocode formáját:

```
function fractionalKnapsack(W, weights, values, n):
    items = list of (value, weight, value/weight) for each item
    sort items in decreasing order by value/weight
    totalValue = 0
    
    for i from 0 to n-1:
        if weights[i] <= W:
            W = W - weights[i]
            totalValue = totalValue + values[i]
        else:
            totalValue = totalValue + values[i] * (W / weights[i])
            break
    
    return totalValue
```

##### C++ implementáció

A fenti pseudocode C++ nyelvű megvalósítása:

```cpp
#include <iostream>

#include <vector>
#include <algorithm>

using namespace std;

struct Item {
    int value, weight;
    double valuePerWeight;
    Item(int v, int w) : value(v), weight(w), valuePerWeight(double(v) / w) {}
};

bool compare(Item a, Item b) {
    return a.valuePerWeight > b.valuePerWeight;
}

double fractionalKnapsack(int W, vector<int>& values, vector<int>& weights, int n) {
    vector<Item> items;
    for (int i = 0; i < n; ++i)
        items.push_back(Item(values[i], weights[i]));
    
    sort(items.begin(), items.end(), compare);
    
    double totalValue = 0.0;
    for (int i = 0; i < n; ++i) {
        if (items[i].weight <= W) {
            W -= items[i].weight;
            totalValue += items[i].value;
        } else {
            totalValue += items[i].value * ((double) W / items[i].weight);
            break;
        }
    }
    return totalValue;
}

int main() {
    vector<int> values = {60, 100, 120};
    vector<int> weights = {10, 20, 30};
    int W = 50;
    int n = values.size();
    cout << "Maximum value in Knapsack = " << fractionalKnapsack(W, values, weights, n) << endl;
    return 0;
}
```

#### Elemzés és időbonyolultság

A mohó algoritmus hatékonyságát és helyességét az garantálja, hogy a tört hátizsák probléma egyaránt kielégíti az optimalitási elvet és a részproblémákra való felosztás tulajdonságát. Az algoritmus időbonyolultsága $O(n \log n)$, amely a tárgyak rendezéséből adódik, míg az egyes tárgyak kiválasztása lineáris időben ($O(n)$) történik. Ez az időbonyolultság lényegesen hatékonyabb, mint a 0/1 hátizsák probléma dinamikus programozásának $O(nW)$ időbonyolultsága.

#### Következtetés

A tört hátizsák probléma egy klasszikus optimalizálási probléma, amely hatékonyan megoldható egy mohó algoritmus segítségével. Az algoritmus alapja az érték/súly arány szerinti rendezés és a tárgyak tört részének kiválasztása, ami garantálja az optimális megoldást. A probléma megoldása gyors és egyszerű, ami különösen hasznos nagyobb adathalmazok esetén. Ezzel a megközelítéssel számos valós életbeli probléma is modellezhető és megoldható hatékonyan.

