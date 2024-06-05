\newpage

## 7.2.2. Bellman-Ford algoritmus

A Bellman-Ford algoritmus egy alapvető és hatékony megoldás a legrövidebb utak megtalálására súlyozott gráfokban, különösen olyan esetekben, amikor a gráf negatív súlyú éleket is tartalmaz. Ez az algoritmus nemcsak a legolcsóbb út meghatározásában nyújt segítséget, hanem képes azonosítani a negatív súlyú köröket is, amelyek jelentős szerepet játszanak a hálózati optimalizálási problémákban. A következő alfejezetekben részletesen bemutatjuk a Bellman-Ford algoritmus alapelveit és implementációját, majd külön tárgyaljuk, hogyan képes az algoritmus hatékonyan kezelni a negatív súlyú éleket, ezáltal biztosítva a megbízható és pontos útvonalmeghatározást.

### 7.2.2.1. Alapelvek és implementáció

A Bellman-Ford algoritmus egy olyan algoritmus, amely megoldja az ún. egyszeres forrás-egyszeres cél (single-source shortest path) problémát egy adott grafikonon. Ez az algoritmus a legolcsóbb út megtalálására szolgál egy súlyozott, irányított gráfban, még akkor is, ha az élek negatív súlyozásúak. Ezen algoritmusnak fontos tulajdonsága, hogy felismeri a negatív súlyú kört is a grafikonban, ami lehetőséget ad különböző hibaellenőrzésekre.

#### Alapelvek

A Bellman-Ford algoritmus alapgondolata a relaxáció technikájára épül, amely azt jelenti, hogy a gráf minden élét többször is vizsgálja a folyamat során, és minden él vizsgálatakor frissíti az adott csúcs távolságát a forráscsúcstól, ha az újonnan talált út költsége alacsonyabb az korábban ismerteknél.

A Bellman-Ford algoritmus a következő lépésekből áll:

1. **Inicializáció**:
    - Kezdetben minden csúcshoz egy végtelen távolságot rendelünk, kivéve a forráscsúcsot, melynek távolságát 0-ra állítjuk.
    - Azaz, ha a forráscsúcs $S$, akkor $d[S] = 0$, és minden más csúcs $V$-re, $d[V] = \infty$.

2. **Relaxáció**:
    - A gráf minden egyes élére a következő relaxációs műveletet végezzük $V-1$ alkalommal, ahol $V$ a csúcsok száma. Ha egy él $u \rightarrow v$ súlya $w$, a következő feltételt vizsgáljuk:
      $$
      \text{ha } d[u] + w < d[v] \text{ akkor } d[v] = d[u] + w.
      $$
    - Ez az eljárás szakaszonként biztosítja, hogy minden csúcs számára meghatározásra kerüljön a legrövidebb út a forrástól.

3. **Negatív kör felismerése**:
    - Végül, megnézzük, hogy létezik-e negatív súlyú kör. Ezt úgy vizsgáljuk, hogy a gráf minden élére újra elvégezzük a relaxációs műveletet.
    - Ha találunk egy olyan élt, amely tovább csökkentené egy csúcs távolságát, akkor az azt jelenti, hogy a gráf negatív súlyú kört tartalmaz.

#### Algoritmus elemzése

*A Bellman-Ford algoritmus futási ideje*, az alkalmazott relaxációs technika miatt $O(V \times E)$, ahol $V$ a csúcsok száma, és $E$ az élek száma. Emiatt jelentősen lassabb lehet a Dijkstra-algoritmusnál. Azonban, a Dijkstra-algoritmussal ellentétben, a Bellman-Ford algoritmus képes kezelni a negatív súlyú éleket, így ez az algoritmus megbízhatóbb azokban az esetekben, amikor a súlyok nem feltétlenül pozitívak.

#### Bellman-Ford algoritmus C++ nyelvű implementációja

Az alábbiakban bemutatok egy egyszerű, bár részletes Bellman-Ford algoritmus implementációt C++ nyelven, amely illusztrálja a fent leírt eljárás lépéseit.

```cpp
#include <iostream>

#include <vector>
#include <limits>

using namespace std;

struct Edge {
    int source, destination, weight;
};

void BellmanFord(const vector<Edge>& edges, int V, int start) {
    // Step 1: Initialize distances from start to all vertices as INFINITE and distance to start as 0
    vector<int> distance(V, numeric_limits<int>::max());
    distance[start] = 0;

    // Step 2: Relax all edges |V| - 1 times.
    for (int i = 1; i <= V - 1; ++i) {
        for (const auto& edge : edges) {
            if (distance[edge.source] != numeric_limits<int>::max() &&
                distance[edge.source] + edge.weight < distance[edge.destination]) {
                distance[edge.destination] = distance[edge.source] + edge.weight;
            }
        }
    }

    // Step 3: Check for negative-weight cycles.
    for (const auto& edge : edges) {
        if (distance[edge.source] != numeric_limits<int>::max() &&
            distance[edge.source] + edge.weight < distance[edge.destination]) {
            cout << "Graph contains a negative-weight cycle." << endl;
            return;
        }
    }

    // Print the calculated shortest distances
    cout << "Vertex distances from source:" << endl;
    for (int i = 0; i < V; ++i) {
        cout << "Vertex " << i << ": " << distance[i] << endl;
    }
}

int main() {
    int V = 5;  // Number of vertices in the graph
    int E = 8;  // Number of edges in the graph

    // Initialize graph using edge list representation
    vector<Edge> edges = {
        {0, 1, -1}, {0, 2, 4}, {1, 2, 3}, {1, 3, 2},
        {1, 4, 2}, {3, 2, 5}, {3, 1, 1}, {4, 3, -3}
    };

    // Run Bellman-Ford algorithm from source vertex 0
    BellmanFord(edges, V, 0);

    return 0;
}
```

#### Magyarázat az implementációhoz:

1. **Adatszerkezetek**:
    - `Edge` struktúra reprezentál egy gráf élét a forrás csomópont, cél csomópont és súly hármasával.
    - `edges` vektor tárolja a gráf összes élének listáját.

2. **Inicializálás**:
    - A `distance` vektor minden csomópontra végtelen távolságot állít be kezdetben, kivéve a forrás csomópontot, amelyek távolsága nullára van állítva.

3. **Relaxáció**:
    - Az élek relaxációja az algoritmus alapelveit követi: az élek többszöri vizsgálata révén frissítjük a csomópontok távolságait.

4. **Negatív Kör Ellenőrzés**:
    - Egy további iterációt hajtunk végre minden élen, hogy ellenőrizzük, van-e még ZÉRÓ távolság frissítés.

5. **Eredmények kiírása**:
    - Az eredmény kiíratása a végső távolságokat jeleníti meg a forrástól minden egyes csomópontig.

Ez a C++ implementáció és a részletes tudományos magyarázat segít megérteni a Bellman-Ford algoritmus működését és alkalmazhatóságát. Az algoritmus rugalmassága és a negatív súlyú körök kezelésének képessége különösen hasznosá teszi a valós életbeni problémák kezelésében.

### 7.2.2.2. Negatív súlyú élek kezelése

A Bellman-Ford algoritmus egyik legfontosabb tulajdonsága, hogy képes kezelni negatív súlyú éleket is, melyek jelentős problémát okozhatnak a legtöbb másik útvonal kereső algoritmusnak, mint például a Dijkstra algoritmusnak. Ennek megértése kritikus fontosságú a hálózati algoritmusok alapos ismeretéhez, és jelentős gyakorlati alkalmazásokkal rendelkezik a való világ problémáinak megoldásában, ahol a gráfokban található élek súlyai nem mindig nem-negatívak.

#### Mi a negatív súlyú él és miért problémás?

Egy gráfban egy él (u, v) negatív súlyú, ha az élhez tartozó költség vagy súly értéke negatív. Ez azt jelenti, hogy ha egy útvonal tartalmaz ilyen éleket, akkor a teljes útvonal költsége csökkenhet, ellentétben a pozitív súlyú élekkel, ahol a költség növekszik.

Negatív súlyú élek problémákat okozhatnak:
- **Útvonal optimalizálásban:** Egyes útvonal kereső algoritmusok, mint amilyen a Dijkstra, a pozitív élek feltételezésén alapulnak. Ezen algoritmusok hibásan viselkednek vagy nem adnak megfelelő eredményeket, ha negatív éleket tartalmazó gráffal találkoznak.
- **Negatív ciklusok jelenlétében:** Egy negatív ciklus egy olyan zárt lánc a gráfban, ahol az összes él súlyának összege negatív. Egy ilyen ciklust tetszőleges számú alkalommal bejárva az útvonal költsége tetszőlegesen alacsonyra csökkenthető, ami lehetetlenné teszi a bármi értelmes útvonal költség meghatározását.

#### Bellman-Ford és negatív élek kezelése

A Bellman-Ford algoritmus egy réteges megközelítést alkalmaz a legrövidebb útvonalak kereséséhez, amely lehetővé teszi a negatív súlyú élek kezelését is. Lássuk, hogy ez hogyan történik lépésről lépésre:

1. **Inicializáció:** Kezdetben az összes csúcs végtelen távolságra van állítva, kivéve a kezdő csúcsot, amely nullra van állítva, hiszen az önmagához vezető út költsége nulla.

2. **Relaxálás:** Az algoritmus V-1 alkalommal (ahol V a csúcsok száma) relaxál minden élt. A relaxálás művelete egy élre (u, v) nézve annyit jelent, hogy ha az u csúcsból rövidebb útvonal vezet v-hez, mint amit korábban ismertünk, akkor frissítjük a v-hez vezető út távolságát.

   A relaxálás lépése:
   $$
   \text{if } d[u] + w(u, v) < d[v] \\
   \text{then } d[v] = d[u] + w(u, v)
   $$

   Itt $d[u]$ az u csúcsba jutás költsége, $w(u, v)$ pedig az (u, v) él súlya.

3. **Negatív ciklus detektálása:** A V-1 relaxációs lépés után az algoritmus még egyszer végigmegy az összes élen. Ha bármelyik élnél további csökkentést találunk a költségekben, akkor ez egy negatív ciklus jelenlétét jelenti a gráfban. Egy ilyen ciklus azt jelenti, hogy végtelenül csökkenthető az adott út költsége, ezért nincs értelme legrövidebb útvonalról beszélni.

```cpp
#include <iostream>

#include <vector>

struct Edge {
    int source, destination, weight;
};

void bellmanFord(const std::vector<Edge>& edges, int V, int start) {
    std::vector<int> distance(V, INT_MAX);
    distance[start] = 0;

    // Relaxing all edges (V-1) times
    for (int i = 1; i < V; ++i) {
        for (const auto& edge : edges) {
            if (distance[edge.source] != INT_MAX && distance[edge.source] + edge.weight < distance[edge.destination]) {
                distance[edge.destination] = distance[edge.source] + edge.weight;
            }
        }
    }

    // Detecting negative cycles
    for (const auto& edge : edges) {
        if (distance[edge.source] != INT_MAX && distance[edge.source] + edge.weight < distance[edge.destination]) {
            std::cout << "Graph contains a negative weight cycle" << std::endl;
            return;
        }
    }

    // Output the results
    for (int i = 0; i < V; ++i) {
        std::cout << "Distance to vertex " << i << " is " << distance[i] << std::endl;
    }
}

int main() {
    int V = 5; // Number of vertices
    std::vector<Edge> edges = {
        {0, 1, -1}, {0, 2, 4},
        {1, 2, 3}, {1, 3, 2}, {1, 4, 2},
        {3, 2, 5}, {3, 1, 1},
        {4, 3, -3}
    };

    int start = 0;
    bellmanFord(edges, V, start);

    return 0;
}
```

#### Részletes elemzés és megjegyzések

Az algoritmus időkomplexitása O(V * E), ami azt jelenti, hogy a futási ideje lineárisan függ a csúcsok és élek számának szorzatától. Ez általában lassabb, mint más algoritmusok, mint például Dijkstra (O(E + V log V)) a pozitív élekhez, de a Bellman-Ford algoritmus előnye éppen a teljeskörűségben és a negatív élek kezelésében rejlik.

A fenti C++ implementáció valóságos alkalmazásaiban megbízhatóan detektálja a negatív ciklusokat és meghatározza a legrövidebb utakat, ha a gráfban nincsenek negatív ciklusok.

#### Gyakorlati alkalmazás

Negatív élek gyakran előfordulnak pénzügyi hálózatokban, például arbitrázs lehetőségek modellezése során, ahol a negatív élek profitot jelentenek egy kereskedelmi úton. Ezen túlmunka különösen érdekes áramlási problémák, különösen a legkisebb költségű áramlási problémák terén is.

Összességében a Bellman-Ford algoritmus alkalmazása megfelelő választás minden olyan helyzetben, ahol a gráf tartalmazhat negatív súlyú éleket, illetve az algoritmus megbízhatóan felismeri és jelzi a negatív ciklusok jelenlétét, ezáltal fontos eszközt biztosítva a hálózati algoritmusok tárházában.

