\newpage

## 7.1.2. Kruskal algoritmusa

A minimális feszítőfa keresése során a Kruskal algoritmusa egy közismert és gyakran alkalmazott módszer, amely hatékonyan találja meg egy gráf összes csúcsát összekötő legkisebb súlyú feszítőfát. Ez az algoritmus a szegmensek szerinti rendezésen alapul, amely lépésről lépésre épít fel egy összefüggő fát úgy, hogy mindig a legkisebb súlyú élt választja, amely nem hoz létre kört. Ez a „szegmenskiválasztó” megközelítés garantálja a minimális súlyt, miközben fokozatosan bővíti az épülő fát. A fejezet során részletesen áttekintjük a Kruskal algoritmusának alapelveit, lépéseit és implementációját, valamint megvizsgáljuk annak gyakorlati alkalmazásait különböző területeken. Az elméleti összefüggések feltárásával és gyakorlati példákon keresztül bemutatva mélyebb betekintést nyerhetünk az algoritmus működésébe és hasznosságába.

### 7.1.2.1. Alapelvek és implementáció

#### Bevezetés

A minimális feszítőfa (MST) fogalma az algoritmusok tanulmányozásában jelentős szerepet játszik, különösen a gráfelmélet és a hálózati tervezés területén. A minimális feszítőfa egy adott súlyozott gráf olyan feszítőfa (spanning tree), amely minimális összsúlyú. Az MST előállítására több algoritmus létezik, ezek közül az egyik legismertebb és leghatékonyabb a Kruskal algoritmusa, melyet Joseph Kruskal fejlesztett ki 1956-ban.

Kruskal algoritmusa egy univerzális módszer, amely kiválóan alkalmazható különféle típusú gráfokra, legyenek azok sűrűn vagy ritkán kapcsoltak. Az algoritmus alapelve viszonylag egyszerű és közvetlen: mindig a legolcsóbb él hozzáadásával építi fel a feszítőfát, amíg el nem éri a teljes gráf összes csomópontját.

#### Alapelvek

Kruskal algoritmusa a következő alapelvekre épül:

1. **Súlyozott Él Gráf Rendezése:**
   Az összes él növekvő sorrendben való rendezése a súlyuk alapján.

2. **Ciklikus Kötések Elkerülése:**
   Olyan élek kiválasztása, amelyek hozzáadása nem hoz létre ciklust a már kiválasztott élek halmazában.

3. **Szaksztruktúra Használata:**
   Szaksztruktúrák (disjoint-set) alkalmazása, hogy hatékonyan kezelje a csomópontok összekapcsoltsági információit és elkerülje a ciklusok keletkezését.

4. **Minimális Összsúly:**
   Olyan feszítőfa kialakítása, amely a lehető legalacsonyabb összsúlyú a gráf csomópontjait összekötő összes lehetséges megoldás közül.

#### Lépések

A Kruskal algoritmus részletes lépései a következők:

1. **Iniciálás:**
    - Adjuk meg a gráf csúcsainak és éleinek halmazát.
    - Kezdetben minden csúcs különálló komponensként szerepel.

2. **Élrendezés:**
    - Az összes él növekvő sorrendben való rendezése az él súlya (cost) alapján. Ezáltal előre meghatározható, hogy mely éleket kell először megvizsgálni.

3. **Szak Halmazok Inicializálása:**
    - Minden csúcs különálló szülőcsomópontként szerepel. Használhatjuk a „find” és „union” műveleteket a komponensek menedzselésére.

4. **Élválasztás és Feszítőfa Építése:**
    - Az élek sorba rendezése után iteráljunk végig rajtuk.
    - Minden élnél vizsgáljuk meg, hogy az él két végpontja különböző komponensben van-e. Ha igen, vegyük hozzá az élt a feszítőfához és egyesítsük a két komponenst.
    - Ez a folyamat addig folytatódik, amíg az összes csúcs egyetlen összefüggő komponensbe (feszítőfa) kerül.

#### Szaksztruktúra és Hatékonyság

A Kruskal algoritmus hatékony működéséhez kulcsfontosságú a szaksztruktúra (disjoint-set, union-find) használata. Ezen adatstruktúra két alapvető művelete:

1. **Find:**
    - Meghatározza, hogy melyik komponenshez tartozik egy adott elem. Ezt rekurzív módon lehet megvalósítani path compression technikával, amely javítja a művelet hatékonyságát.

2. **Union:**
    - Egyesít két különböző komponenst. Az optimalizált változata a union by rank. Ebben a technikában a kisebb rangu fát illesztjük a nagyobb rangu fához, így akár a fa mélysége is optimalizálható.

Az algoritmus ezen kívül sorting műveletet is tartalmaz, amely alapvetően O(E log E) időben fut, ahol E az élek száma. A disjoint-set adatstruktúrák alkalmazása O(E log V) futási időt tesz lehetővé, így az algoritmus teljes időkomplexitása O(E log E), ami nagyságrendileg megegyezik O(E log V)-vel is, mivel E erős felső korlátja V^2.

#### Pseudokód és Implementáció

A Kruskal algoritmus pseudokódja a következőképpen fest:

```pseudo
function Kruskal(graph):
    create a forest F (a set of trees), where each vertex in the graph is a separate tree
    create a set S containing all the edges in the graph
    while S is nonempty and F is not yet a spanning tree:
        remove an edge with minimum weight from S
        if that edge connects two different trees:
            add that edge to the forest, combining two trees into a single tree
    return the forest F
```

A következő C++ kód egy példa a Kruskal algoritmus működésére:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class Edge {
public:
    int src, dest, weight;
};

class Graph {
public:
    int V, E;
    vector<Edge> edges;

    Graph(int V, int E) {
        this->V = V;
        this->E = E;
    }

    void addEdge(int u, int v, int w) {
        Edge edge{u, v, w};
        edges.push_back(edge);
    }

    // Find function for Union-Find
    int find(vector<int>& parent, int i) {
        if (parent[i] == i)
            return i;
        return parent[i] = find(parent, parent[i]); // path compression
    }

    // Union function for Union-Find
    void Union(vector<int>& parent, vector<int>& rank, int x, int y) {
        int xroot = find(parent, x);
        int yroot = find(parent, y);

        if (rank[xroot] < rank[yroot]) {
            parent[xroot] = yroot;
        } else if (rank[xroot] > rank[yroot]) {
            parent[yroot] = xroot;
        } else {
            parent[yroot] = xroot;
            rank[xroot]++;
        }
    }

    // Kruskal's algorithm to find MST
    void KruskalMST() {
        vector<Edge> result; // Store the resultant MST
        sort(edges.begin(), edges.end(), [](Edge a, Edge b) {
            return a.weight < b.weight;
        });

        vector<int> parent(V);
        vector<int> rank(V, 0);

        for (int i = 0; i < V; ++i) {
            parent[i] = i;
        }

        int e = 0;
        for (Edge& edge : edges) {
            if (e >= V - 1) break;
            int x = find(parent, edge.src);
            int y = find(parent, edge.dest);

            if (x != y) {
                result.push_back(edge);
                Union(parent, rank, x, y);
                e++;
            }
        }

        cout << "Edges in MST:" << endl;
        for (Edge& edge : result) {
            cout << edge.src << " - " << edge.dest << ": " << edge.weight << endl;
        }
    }
};

int main() {
    int V = 4, E = 5;
    Graph graph(V, E);

    graph.addEdge(0, 1, 10);
    graph.addEdge(0, 2, 6);
    graph.addEdge(0, 3, 5);
    graph.addEdge(1, 3, 15);
    graph.addEdge(2, 3, 4);

    graph.KruskalMST();

    return 0;
}
```

#### Elemzés és Jellemzők

##### Előnyök:
- A Kruskal algoritmus különösen jól működik ritkán kapcsolt gráfok esetén, mivel az O(E log E) időbeli komplexitás kedvező nagy számú élek esetén.
- Az algoritmus egyértelmű és egyszerűen implementálható, még nagy méretű gráfok esetében is.
- Az „Union-Find” struktúra alkalmazása optimalizálja a csúcsok összekapcsoltságának kezelését.

##### Hátrányok:
- A rendezés művelete az algoritmus egyik időigényes lépése, bár igen hatékony algoritmusok léteznek erre (például QuickSort, MergeSort).
- Sűrű gráfok esetén az élhalmaz rendezése és kezelése relatíve költséges lehet.

#### Összegzés

Kruskal algoritmusa egy alapvető és hatékony megoldás a minimális feszítőfa problémájára, amely széleskörűen alkalmazható a különböző típusú súlyozott gráfok esetében. Az algoritmus részletezése, az „Union-Find” struktúra optimalizálásának és a rendezési műveletek megértése kritikus szerepet játszanak abban, hogy hatékonyan alkalmazzuk ezt az algoritmust gyakorlati problémák megoldására.

### 7.1.2.2. Alkalmazások és példák

Kruskal algoritmusa az egyik legismertebb algoritmus a minimális feszítőfa (MST) problémájának megoldására. Az alábbiakban részletesen tárgyaljuk Kruskal algoritmusának különböző alkalmazásait és gyakorlati példákat.

#### Alkalmazások

Kruskal algoritmusának számos területen hasznát veszik, ahol szükség van a minimális összekötési költségű hálózatok kialakítására. Az alábbiakban néhány konkrét alkalmazást részletezünk.

1. **Távközlési hálózatok tervezése**:
   A távközlési hálózatok tervezése során gyakran előfordul olyan feladat, hogy különböző telephelyeket összekössünk, minimális költséggel. Ilyen esetekben a csomópontok lehetnek városok, az élek pedig az optikai kábelösszeköttetések vagy más kommunikációs infrastruktúrák. Kruskal algoritmusa segítségével meghatározható az optimális hálózat, amely minimalizálja az összekötési költségeket.

2. **Elektromos hálózatok**:
   Elektromos hálózatok tervezésekor a célt az optimális vezetékek elhelyezése jelenti a generátorok és a fogyasztók között, úgy hogy a költségek minimálisak legyenek. Az algoritmus itt is hatékonyan alkalmazható az optimális hálózati struktúra megtalálására.

3. **Közlekedési hálózatok tervezése**:
   Városok vagy városrészek közötti utak és hidak tervezésekor gyakran szükség van minimális költségű összeköttetések meghatározására. Az utak és hidak költségeit figyelembe véve Kruskal algoritmusa segíthet a leghatékonyabb úthálózat megtervezésében.

4. **Klasterezési algoritmusok**:
   Adatok klaszterezésénél, különösen a hierarchikus klaszterezési eljárások esetén, Kruskal algoritmusa felhasználható az adathalmaz optimális elágazásainak meghatározására. Az algoritmus segítségével olyan fa struktúrájú hierarchiát lehet létrehozni, amely a legjobb kapcsolódási költségeket tükrözi.

5. **Nyomtávellenőrzés**:
   Kruskal algoritmusa a nyomtávellenőrzésben (tracking) is felhasználható, ahol a cél egy objektum mozgását követni különböző időpillanatok során. A minimális feszítőfa meghatározásával könnyen azonosítható a legrövidebb összekötési útvonal.

#### Példák

Részletesen megvizsgálunk egy konkrét példát a Kruskal algoritmus alkalmazására, hogy jobban érzékeltessük a működését és az előnyeit.

##### Példa: Városok összekapcsolása

Tegyük fel, hogy van hat város, amelyeket sztrádákkal szeretnénk összekötni. Az egyes utak építésének költségei az alábbi táblázat szerint alakulnak:

| Városok | Sztráda (Él) | Költség |
|---------|--------------|---------|
| A - B   | 1            | 4       |
| A - F   | 2            | 2       |
| B - C   | 3            | 5       |
| B - D   | 4            | 10      |
| C - D   | 5            | 3       |
| C - E   | 6            | 7       |
| D - E   | 7            | 1       |
| F - E   | 8            | 6       |

Lépésenként követjük az algoritmus végrehajtását:

1. **Költség szerinti rendezés**:
   Az éleket a költségek szerint növekvő sorrendbe rendezzük:
   ```
   4 (A - F), 1 (D - E), 3 (C - D), 2 (A - F), 8 (F - E), 6 (C - E), 7 (D - E), 10 (B - D)
   ```

2. **Élek kiválasztása**:
   Lépésről lépésre hozzáadjuk a legalacsonyabb költségű éleket, amíg az MST-t meg nem kapjuk.
    - Első él: (D - E) költség 1 — Nincs ciklus.
    - Második él: (A - F) költség 2 — Nincs ciklus.
    - Harmadik él: (C - D) költség 3 — Nincs ciklus.
    - Negyedik él: (A - B) költség 4 — Nincs ciklus.
    - Ötödik él: (C - E) költség 6 — Nincs ciklus.

   Ezen a ponton elértük a minimális feszítőfát, mivel minden város (A, B, C, D, E, F) összekötésre került.

Ez azt jelenti, hogy a Kruskal algoritmus segítségével a következő minimális feszítőfát kapjuk:
   ```
   Élek: {(D - E), (A - F), (C - D), (A - B), (C - E)}
   Összköltség: 1 + 2 + 3 + 4 + 6 = 16
   ```

Megjegyzés: Az algoritmus által végrehajtott döntések mindig figyelembe veszik, hogy az újonnan kiválasztott él ne alkosson ciklust az aktuális fa struktúrában.

#### Példakód (C++)

Az alábbiakban bemutatjuk a Kruskal algoritmusának egy egyszerű C++ implementációját.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// Structure to represent an edge
struct Edge {
    int source, destination, weight;
};

// Structure to represent a graph
struct Graph {
    int V, E;
    vector<Edge> edges;
};

// Structure to represent a subset for union-find
struct Subset {
    int parent, rank;
};

// Utility function to find the subset of an element e
int find(vector<Subset>& subsets, int i) {
    if (subsets[i].parent != i)
        subsets[i].parent = find(subsets, subsets[i].parent);
    return subsets[i].parent;
}

// A utility function to do union of two subsets
void Union(vector<Subset>& subsets, int x, int y) {
    int xroot = find(subsets, x);
    int yroot = find(subsets, y);

    // Attach smaller rank tree under root of high rank tree
    if (subsets[xroot].rank < subsets[yroot].rank)
        subsets[xroot].parent = yroot;
    else if (subsets[xroot].rank > subsets[yroot].rank)
        subsets[yroot].parent = xroot;
    else {
        subsets[yroot].parent = xroot;
        subsets[xroot].rank++;
    }
}

// Compare function to sort edges based on their weights
bool compare(Edge a, Edge b) {
    return a.weight < b.weight;
}

// Function to construct MST using Kruskal's algorithm
void KruskalMST(Graph &graph) {
    int V = graph.V;
    vector<Edge> result;  // This will store the resultant MST
    result.reserve(V - 1);

    // Step 1: Sort all the edges in non-decreasing order of their weight
    sort(graph.edges.begin(), graph.edges.end(), compare);

    // Allocate memory for creating V subsets
    vector<Subset> subsets(V);
    for (int v = 0; v < V; ++v) {
        subsets[v].parent = v;
        subsets[v].rank = 0;
    }

    // Number of edges to be taken is equal to V-1
    int e = 0;  // An index used to pick next edge
    int i = 0;  // An index used to iterate through sorted edges
    while (e < V - 1 && i < graph.edges.size()) {
        // Step 2: Pick the smallest edge. Check if it forms a cycle with the spanning-tree 
        // formed so far. If cycle is not formed, include this edge. Else, discard it.
        Edge next_edge = graph.edges[i++];

        int x = find(subsets, next_edge.source);
        int y = find(subsets, next_edge.destination);

        // If including this edge does not cause cycle, include it in result
        // and move ahead in this process
        if (x != y) {
            result.push_back(next_edge);
            Union(subsets, x, y);
            e++;
        }
        // Else discard the next_edge
    }

    // print the contents of result[] to display the built MST
    cout << "Following are the edges in the constructed MST\n";
    for (i = 0; i < result.size(); ++i)
        cout << result[i].source << " -- " << result[i].destination << " == " << result[i].weight << endl;
}

int main() {
    int V = 6;  // Number of vertices in graph
    int E = 8;  // Number of edges in graph
    Graph graph{V, E};

    graph.edges = {
        {0, 1, 4}, {0, 5, 2}, {1, 2, 5}, {1, 3, 10}, {2, 3, 3}, {2, 4, 7}, {3, 4, 1}, {5, 4, 6}
    };

    KruskalMST(graph);

    return 0;
}
```

Ez a kód bemutatja Kruskal algoritmusának lépéseit egy konkrét példa alapján. Az élek és költségeik megfelelően definiáltak, és az algoritmus segítségével a minimális feszítőfa létrehozása történik.
