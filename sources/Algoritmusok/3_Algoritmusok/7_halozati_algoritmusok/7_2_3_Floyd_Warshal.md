\newpage

## 7.2.3. Floyd-Warshall algoritmus

A Floyd-Warshall algoritmus egy kiemelkedően hatékony és széles körben alkalmazott módszer, amely lehetőséget nyújt a legolcsóbb utak megtalálására egy irányított vagy irányítatlan gráfban, figyelembe véve az adott gráf minden lehetséges csúcsait és élét. Az algoritmus különlegessége abban rejlik, hogy képes kezelni a negatív élhosszúságokat is, amennyiben a gráf nem tartalmaz negatív súlyú köröket. Ez az algoritmus dinamikus programozási megközelítést alkalmazva építi fel fokozatosan a legrövidebb utak mátrixát, lehetővé téve ezzel a teljes páros legolcsóbb utak azonosítását. A következő alfejezetekben bemutatjuk az algoritmus alapelveit és részletes implementációját, majd részletesen tárgyaljuk a teljes páros legolcsóbb utak keresésének gyakorlati jelentőségét és alkalmazási területeit.

### 7.2.3.1. Alapelvek és implementáció

A Floyd-Warshall algoritmus az egyik legismertebb és leghatékonyabb algoritmus a teljes páros legolcsóbb utak keresésére súlyozott, irányított és irányítatlan gráfokban. Ez az algoritmus rendkívül népszerű, mert egyszerű, könnyen megérthető és egyaránt alkalmazható mind pozitív, mind negatív él súlyú gráfokra, amennyiben nincs negatív hosszúságú kör.

#### Alapelvek

A Floyd-Warshall algoritmus dinamikai programozási megközelítést alkalmaz a több pont-pont közötti út megkeresésére. Az alapelve az, hogy fokozatosan építi fel a legolcsóbb utak hosszát tartalmazó mátrixot, amely minden lehetséges pontpárosításra vonatkozik.

Az algoritmus használ egy háromdimenziós tömböt, $D$, ahol $D[k][i][j]$ az $i$-ből $j$-be vezető út legolcsóbb hosszát jelöli azáltal, hogy megengedett köztes csúcsok indexe legfeljebb $k$.

Az inicializálás során:
- $D[0][i][i] = 0$ az összes $i$-re, mivel bármely csúcs önmagába történő eljutása költség nélküli.
- $D[0][i][j] = w(i, j)$, ahol $w(i, j)$ az $i$-ből $j$-be vezető él súlya, ha van ilyen él; különben végtelen.

Ezután az algoritmus három beágyazott ciklus segítségével frissíti a távolságokat:
1. A legkülső ciklus indexeli a köztes csúcsokat (k).
2. A középső ciklus indexeli a kiindulási csúcsokat (i).
3. A belső ciklus indexeli a célcsúcsokat (j).

Minden iterációnál ellenőrzi azt, hogy az adott csúcs (k) közbeiktatásával lehet-e rövidebb utat találni:
$$
D[k][i][j] = \min(D[k-1][i][j], D[k-1][i][k] + D[k-1][k][j])
$$

#### Implementáció

Az algoritmus egy egyszerű és érthető kóddal is megvalósítható C++ nyelven. Az alábbiakban bemutatok egy lehetséges implementációt.

```cpp
#include <iostream>

#include <vector>
#include <limits>

const int INF = std::numeric_limits<int>::max();

// Floyd-Warshall algorithm in C++
void floydWarshall(std::vector<std::vector<int>>& dist) {
    int V = dist.size();

    // Adding vertices individually
    for (int k = 0; k < V; ++k) {
        for (int i = 0; i < V; ++i) {
            for (int j = 0; j < V; ++j) {
                if (dist[i][k] != INF && dist[k][j] != INF) {
                    dist[i][j] = std::min(dist[i][j], dist[i][k] + dist[k][j]);
                }
            }
        }
    }
}

int main() {
    int V = 4; // Number of vertices in the graph
    std::vector<std::vector<int>> dist(V, std::vector<int>(V, INF));

    // Example initialization of the graph's edge weights
    dist[0][0] = 0; dist[0][1] = 5; dist[0][2] = INF; dist[0][3] = 10;
    dist[1][0] = INF; dist[1][1] = 0; dist[1][2] = 3; dist[1][3] = INF;
    dist[2][0] = INF; dist[2][1] = INF; dist[2][2] = 0; dist[2][3] = 1;
    dist[3][0] = INF; dist[3][1] = INF; dist[3][2] = INF; dist[3][3] = 0;

    // Run Floyd-Warshall algorithm
    floydWarshall(dist);

    // Print the shortest distances
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (dist[i][j] == INF)
                std::cout << "INF ";
            else
                std::cout << dist[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
```

#### Elemzés

Az időbonyolultság a Floyd-Warshall algoritmus esetében $O(V^3)$, ahol $V$ a gráf csúcsainak száma. Ennek oka három, egymásba ágyazott ciklus, amelyek mindegyike $V$ iterációt fut. Az algoritmus területbonyolultsága szintén $O(V^2)$, mert egy $V \times V$ méretű mátrixot tarja fenn a távolságok rögzítéséhez.

Előnyei közé tartozik az egyszerűség, és hogy képes kezelni negatív élsúlyokat is, feltéve, hogy nincs negatív hosszúságú kör. Hátránya viszont, hogy nagy gráfok esetében az $O(V^3)$ időkomplexitás jelentős redundanciákat okozhat, különösen olyan alkalmazásokban, amelyek viszonylag ritkán használnak minden pont-pont kapcsolatot.

#### Specializált felhasználások

A Floyd-Warshall algoritmus különösen hasznos olyan területeken, mint a hálózati rendszerek, közlekedési útvonalak optimalizálása és dinamikus programozási problémák, ahol minden csúcs közötti összeköttetés és annak költségei ismertek kell legyenek. Az olyan problémás esetek, mint az adatok hálózaton belüli továbbítása vagy a hibamentes útvonalak azonosítása, hatékonyan kezelhetők ezzel az algoritmussal.

### 7.2.3.2. Teljes páros legolcsóbb utak keresése

A Floyd-Warshall algoritmus egy robusztus módszer a teljes páros legolcsóbb útvonalak kiszámítására egy adott irányított vagy irányítatlan, súlyozott gráfban. Ebben az alfejezetben mélyebben megvizsgáljuk az algoritmus elveit, az implementációját, valamint annak matematikai alapjait és jelentőségét. A teljes páros legolcsóbb utak keresése egy alapvető probléma számos alkalmazási területen, beleértve a hálózattervezést, az utazó ügynök problémájának megoldását, és az útvonalkeresést a térinformatikai rendszerekben.

#### 7.2.3.2.1. Matematikai Alapok

A Floyd-Warshall algoritmus fő célja egy gráf minden csúcspárjára meghatározni a legolcsóbb út költségét. Formálisan, legyen G = (V, E) egy súlyozott gráf, ahol V a csúcsok halmaza és E az élek halmaza, valamint w(u, v) az u-ból v-be vezető él súlya. Az algoritmus a d(i, j) távolsági mátrixot használva dolgozik, ahol d[i][j] az i és j csúcs közötti legolcsóbb út költségét jelenti.

Az alapelve a dinamikus programozásnak egy speciális formájára épül, ahol az iterációk során fokozatosan finomítják a távolságértékeket minden csúcspárra vonatkozóan.

#### 7.2.3.2.2. Dinamikus Programozási Megközelítés

Vegyük át részletesen az algoritmus alapötletét. Az algoritmus három egymásba ágyazott ciklussal dolgozik, ahol a lépések száma a csúcspontok számának köbével arányos, azaz $O(|V|^3)$ időkomplexitással rendelkezik.

A dinamikus programozáshoz tartozó képlet:

$$
d[i][j] = \min(d[i][j], d[i][k] + d[k][j])
$$

Ez a reláció intuitívan azt jelenti, hogy a legolcsóbb út i-ből j-be lehet az aktuálisan ismert közvetlen útvonal (d[i][j]), vagy egy köztes csúcson keresztül (k), azaz i-ből k-be és k-ból j-be (d[i][k] + d[k][j]). Az algoritmus kezdeti lépései beállítják az önhivatkozó távolságokat nullára (d[i][i] = 0 minden i-re) és az élek súlyait a távolsági mátrix megfelelő elemeivé (d[u][v] = w(u, v)).

#### 7.2.3.2.3. Pszeudo-Belső Ciklus Struktúra

Az algoritmusnak három fő ciklusa van, az alábbiak szerint:

```cpp
for (int k = 0; k < V; ++k) {
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (distance[i][k] + distance[k][j] < distance[i][j]) {
                distance[i][j] = distance[i][k] + distance[k][j];
            }
        }
    }
}
```

Itt V a gráf csúcsainak száma. Minden iteráció során a algoritmus finomítja a d[i][j] értékeit az összes csúcs k-ra vonatkozóan, ami lehetővé teszi új, rövidebb útvonalak felfedezését.

#### 7.2.3.2.4. Algoritmus Az Optimális Út Kinyerésére

A csak a legolcsóbb költségek meghatározása mellett sok alkalmazásban az is fontos, hogy magát az útvonalat is visszanyerjük. Ehhez egy iránytáblát (predecessor matrix) használunk, amely megőrzi azt az intermediális csúcsot, amelyen keresztül az optimális útvonal halad.

```cpp
for (int k = 0; k < V; ++k) {
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (distance[i][k] + distance[k][j] < distance[i][j]) {
                distance[i][j] = distance[i][k] + distance[k][j];
                predecessor[i][j] = predecessor[k][j];
            }
        }
    }
}
```

Ez az implementáció azt biztosítja, hogy a predecessor[i][j] tartalmazza a j elődjét azon útvonalon, amely a k közbülső csúcspontot használja az i-től j-ig.

#### 7.2.3.2.5. Path Reconstruction

A legolcsóbb út rekonstruálásához rekurzívan kell használni az előd mátrixot. Kezdve a célcsúccsal, visszalépve az előd mátrixon keresztül, végül elérhetjük a forráscsúcsot:

```cpp
void printPath(int i, int j) {
    if (i != j) 
        printPath(i, predecessor[i][j]);
    cout << j << " ";
}
```

Ez a rekurzív funkció visszafejtheti az i-től j-ig tartó utat.

#### 7.2.3.2.6. Alkalmazások és Jelentőség

A Floyd-Warshall algoritmus számos gyakorlati alkalmazással bír:

- **Kommunikációs Hálózatok:** Optimalizálás az adatok eljuttatásában különböző hálózati csomópontok között.
- **Logisztikai Hálózatok:** Hatékony útvonaltervezés áruszállításra, például raktárak és kiskereskedelmi pontok között.
- **Térinformatikai Rendszerek:** Optimalizált útvonalak tervezése különböző földrajzi helyek között.

A Floyd-Warshall algoritmus hatékonyan működik kisebb és közepes méretű gráfokkal, ahol a $O(|V|^3)$ időkomplexitás elfogadható. Nagy méretű hálózatok esetén más technikák, mint például a Johnson-algoritmus, lehet előnyösebb.

#### 7.2.3.2.7. Részletes C++ Implementáció

Alább egy teljes C++ implementáció, beleértve az útrekonstrukciós részletet is:

```cpp
#include <iostream>

#include <vector>
#include <limits.h>

using namespace std;
const int INF = INT_MAX;  // Using INT_MAX to represent infinity

void floydWarshall(int V, vector<vector<int>>& dist, vector<vector<int>>& pred) {
    for (int k = 0; k < V; ++k) {
        for (int i = 0; i < V; ++i) {
            for (int j = 0; j < V; ++j) {
                if (dist[i][k] != INF && dist[k][j] != INF && dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                    pred[i][j] = pred[k][j];
                }
            }
        }
    }
}

void printPath(vector<vector<int>>& pred, int i, int j) {
    if (i != j) 
        printPath(pred, i, pred[i][j]);
    cout << j << " ";
}

int main() {
    int V = 4;
    vector<vector<int>> dist(V, vector<int>(V, INF));
    vector<vector<int>> pred(V, vector<int>(V));

    // Example initialization
    dist[0][0] = 0; dist[0][1] = 3; dist[0][2] = INF; dist[0][3] = 7;
    dist[1][0] = 8; dist[1][1] = 0; dist[1][2] = 2; dist[1][3] = INF;
    dist[2][0] = 5; dist[2][1] = INF; dist[2][2] = 0; dist[2][3] = 1;
    dist[3][0] = 2; dist[3][1] = INF; dist[3][2] = INF; dist[3][3] = 0;

    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (dist[i][j] != INF) 
                pred[i][j] = i;
            else 
                pred[i][j] = -1;
        }
    }

    floydWarshall(V, dist, pred);

    cout << "Shortest distances:\n";
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (dist[i][j] == INF) 
                cout << "INF ";
            else 
                cout << dist[i][j] << " ";
        }
        cout << endl;
    }

    cout << "\nPaths:\n";
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (i != j && dist[i][j] != INF) {
                cout << "Path from " << i << " to " << j << ": ";
                cout << i << " ";
                printPath(pred, i, j);
                cout << endl;
            }
        }
    }

    return 0;
}
```

Ez a C++ kód inícializálja a távolsági mátrixot és az elődmátrixot, majd futtatja a Floyd-Warshall algoritmust. Az eredmények megjelenítik a legolcsóbb távolságok mátrixát és az összes csúcspárhoz tartozó legolcsóbb útvonalakat.
Ezek alapján a Floyd-Warshall algoritmus egy rendkívül hasznos eszköz a teljes páros legolcsóbb utak meghatározására egy súlyozott gráfban, amely számos valós problémában alkalmazható.

