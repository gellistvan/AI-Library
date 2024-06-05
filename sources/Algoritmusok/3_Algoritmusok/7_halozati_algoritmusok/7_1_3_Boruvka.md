\newpage

## 7.1.3. Boruvka algoritmusa

Boruvka algoritmusa egy klasszikus, ám még mindig széles körben alkalmazott módszer a minimális feszítőfa (MST) keresésére egy összefüggő, súlyozott gráfban. Az algoritmus egyszerűsége és párhuzamosíthatósága különösen vonzóvá teszi nagy méretű hálózatok és gráfok esetében. Boruvka algoritmusa iteratívan építi fel a minimális feszítőfát, mindeközben az éleket és csúcsokat csoportosítva, így fokozatosan csökkentve a gráf méretét, amíg egyetlen összefüggő komponens nem marad. Ez a fejezet részletesen bemutatja az algoritmus alapelveit és annak implementációját, majd gyakorlati alkalmazásokon és példákon keresztül szemlélteti a módszer használatát.

### 7.1.3.1. Alapelvek és implementáció

Boruvka algoritmusa, amelyet Otakar Borůvka cseh matematikus fejlesztett ki 1926-ban, az első algoritmus volt, amelyet specifikusan a minimális feszítőfa megtalálására terveztek. Az algoritmus alapötlete viszonylag egyszerű, de hatékony: minden csúcs a hálózatban kiválasztja a hozzá kapcsolódó legkisebb súlyú élt, majd ezekből az élekből fokozatosan hálózatot építve összefüggővé tesszük a gráfot úgy, hogy közben folyamatosan az élek súlyának összegét minimalizáljuk. Az algoritmus iteratív módon működik, minden lépésben csökkentve a hálózatban lévő fák számát, amíg végül egy összefüggő fát nem kapunk.

#### Az algoritmus alaplépései:

1. **Kezdeti állapot**: Kezdjük azzal, hogy minden csúcs saját komponensként (alkomponensként) áll.

2. **Kis súly választás**:
    - Minden csúcs kiválasztja a hozzá tartozó legkisebb súlyú élt, amely egy másik komponenshez csatlakozik.

3. **Élek hozzáadása**:
    - Az összes ilyen kijelölt él hozzáadása a fahoz, ha az nem okoz kört.

4. **Komponensek egyesítése**:
    - Kombináljuk azokat a komponenseket, amelyek az élek hozzáadásával összekapcsolódnak.

5. **Ismétlés**:
    - Ha egynél több komponens marad, ismételjük meg a 2-4 lépéseket, amíg a komponensek száma egyre csökken és végül egyetlen egyösszefüggő gráfot kapunk.

#### Az algoritmus részletes leírása

Boruvka algoritmusa többszörös iterációban működik. Minden iterációban a gráf egyre kevesebb komponensből áll, és minden iteráció közel felére csökkenti a komponensek számát. Az iterációk száma így legfeljebb O(log V) (V egyenlő a csúcsok száma), és az algoritmus minden egyes iterációban minden egyes csúcsot vizsgál, így a teljes időkomplexitása O(E log V) (E egyenlő az élek száma).

#### Formális leírás:

1. Jelöljük a gráfot G(V, E) - ahol V a csúcsok halmaza és E az élek halmaza.
2. Kezdésként minden csúcs saját komponensként van kezelve: K = {{v1}, {v2}, ..., {vn}}.
3. Minden csúcs választja a minimális súlyú szomszédját, amely különbözik a saját komponensétől.
4. Az összes kiválasztott él egy gráfba egyesítése.
5. Az újonnan összekapcsolt komponensek új egyesített komponenseket alkotnak.
6. Ha maradt egynél több komponens, térjünk vissza a 3. lépéshez.

#### Implementáció C++ nyelven

Az alábbiakban egy lehetséges C++ implementáció következik. Az implementáció során a csúcsokat és az éleket reprezentáló adatstruktúrákat használjuk, és a komponenst felépítő fák unióját egy egyesítési-find struktúrán (Union-Find) keresztül valósítjuk meg.

```cpp
#include <iostream>

#include <vector>
#include <algorithm>

#include <utility>
#include <climits>

using namespace std;

class DisjointSet {
public:
    DisjointSet(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; ++i) {
            parent[i] = i;
        }
    }
    
    int find(int u) {
        if (u != parent[u]) {
            parent[u] = find(parent[u]);
        }
        return parent[u];
    }
    
    void unite(int u, int v) {
        int rootU = find(u);
        int rootV = find(v);
        
        if (rootU != rootV) {
            if (rank[rootU] > rank[rootV]) {
                parent[rootV] = rootU;
            } else {
                parent[rootU] = rootV;
                if (rank[rootU] == rank[rootV]) {
                    ++rank[rootV];
                }
            }
        }
    }

private:
    vector<int> parent;
    vector<int> rank;
};

struct Edge {
    int u, v, weight;
    bool operator<(const Edge &other) const {
        return weight < other.weight;
    }
};

void boruvkaMST(vector<Edge> &edges, int V) {
    DisjointSet ds(V);
    vector<Edge> mst;
    int numComponents = V;

    while (numComponents > 1) {
        vector<Edge> minEdge(V, {-1, -1, INT_MAX});
        
        // Find the minimum weight edge for each component
        for (auto &edge : edges) {
            int uSet = ds.find(edge.u);
            int vSet = ds.find(edge.v);
            
            if (uSet != vSet) {
                if (edge.weight < minEdge[uSet].weight) {
                    minEdge[uSet] = edge;
                }
                if (edge.weight < minEdge[vSet].weight) {
                    minEdge[vSet] = edge;
                }
            }
        }

        // Add the selected edges to the MST and unite the components
        for (auto &edge : minEdge) {
            if (edge.u != -1 && ds.find(edge.u) != ds.find(edge.v)) {
                mst.push_back(edge);
                ds.unite(edge.u, edge.v);
                --numComponents;
            }
        }
    }
    
    // Output the MST
    cout << "Edges in Minimum Spanning Tree:\n";
    for (auto &edge : mst) {
        cout << edge.u << " - " << edge.v << " : " << edge.weight << "\n";
    }
}

int main() {
    int V = 4; // Number of vertices
    vector<Edge> edges = {
        {0, 1, 10}, {0, 2, 6}, {0, 3, 5}, {1, 3, 15}, {2, 3, 4}
    };

    boruvkaMST(edges, V);
    return 0;
}
```

#### Magyarázat:

- **DisjointSet osztály**: Ez az osztály kezeli a halmazstruktúrát (Union-Find), ahol a `parent` vektor a csúcsok szülőit jelzi, és a `rank` vektor a fa mélyítését követi a kompresszió hatékonyságának növelése érdekében.
- **Edge Struktúra**: Az Edge (Él) struktúra tárolja az éleket, valamint azok végpontjait és súlyát.
- **boruvkaMST Function**: Ez a függvény valósítja meg Boruvka algoritmusát. Először inicializálja a disjoint set-et, majd egy hurokban az összes csúcs legalacsonyabb éleit választja ki. Ezek után összevonja a komponenseket, és hozzáadja az éleket az MST-hez.
- **Main Function**: Példát hoz a fenti függvény használatára.

A fenti implementáció lépésről lépésre követi Boruvka algoritmusának alapelveit: az aktuális komponensek összekapcsolása minimális éleken keresztül iteratív módon, míg az algoritmus végül egyetlen összefüggő komponensként áll össze. A disjoint set struktúra használata biztosítja, hogy az egyes komponensek egyesítése hatékonyan történjen, és az élek hozzáadása ne okozzon ciklust.

Boruvka algoritmusának előnye, hogy minden lépésben párhuzamosan dolgozik a gráf különböző részein, így nagyon hatékonnyá válik nagy, szparzáló gráfok esetében is. Az implementáció könnyen adaptálható különböző programnyelvekre és alkalmazási területekre. Ennek köszönhetően széles körben használják, például telekommunikációs hálózatok optimalizálására és operációs rendszerek fájlrendszereinek kezelésekor.

### 7.1.3.2. Alkalmazások és példák

A Borůvka algoritmusa, amelyet Otakar Borůvka cseh matematikusról neveztek el, az egyik legkorábbi és leghatékonyabb módszer a minimális feszítőfa (MST) megtalálására egy súlyozott, élsúlyokkal jelölt gráfban. Ez az algoritmus különösen fontos szerepet játszik a hálózatok kezelésében, legyen szó telekommunikációs hálózatokról, közlekedési rendszerekről, vagy akár biológiai hálózatokról. Ebben a fejezetben részletesen tárgyaljuk a Borůvka algoritmus alkalmazásait és konkrét példákat mutatunk be, amelyek bemutatják az algoritmus gyakorlati használatát.

#### Alkalmazások

##### 1. Telekommunikációs hálózatok tervezése

A telekommunikációs hálózatok tervezése során gyakori feladat a hálózati infrastruktúra költségének minimálisra csökkentése, miközben minden csomópont elérhetősége biztosított. Ebben az esetben a hálózat gráfként modellezhető, ahol a csomópontok városokat, a szélek pedig a különböző városok közötti összeköttetéseket reprezentálják. Az élsúlyok pedig az egyes összeköttetések költségét jelzik. A minimális feszítőfa megtalálása ezen a gráfon lehetővé teszi a legköltséghatékonyabb hálózatfelépítés meghatározását. Borůvka algoritmusa gyors és hatékony módon képes ezt a feladatot elvégezni, különösen nagy méretű hálózatok esetén.

##### 2. Közlekedési hálózatok optimalizálása

Hasonlóan a telekommunikációs hálózatokhoz, a közlekedési rendszerek optimalizálása is tipikusan minimális feszítőfát igényel. Egy város közlekedési hálózata gráffal modellezhető, ahol a csomópontok az útkereszteződéseket, a szélek pedig az utakat jelölik. A súlyok az útvonalakon való közlekedés költségeit vagy távolságait jelentik. A Borůvka algoritmus alkalmazásával a közlekedési hálózat tervezői minimális érintkezési költséggel és távolsággal optimalizálhatják az útvonalakat.

##### 3. Elektromos hálózatok tervezése

Az elektromos hálózatok tervezése során is rendkívül fontos a költséghatékonyság. Egy város elektromos hálózata esetén a cél, hogy minden fogyasztó (háztartás, ipari létesítmény stb.) a legkisebb költséggel legyen ellátva elektromos árammal. Ez gráfkonstrukcióval könnyen modellezhető, ahol a cél a minimális feszítőfa megtalálása. Borůvka algoritmusa gyors eredményt adhat, különösen akkor, ha a hálózat nagy méretű.

##### 4. Szociális hálózatok elemzése

A szociális hálózatok elemzése során is gyakran alkalmazzák a minimális feszítőfa algoritmusokat. Például, ha egy adott közösség kapcsolati hálóját próbálják megismerni és optimalizálni, a Borůvka algoritmus segíthet feltérképezni a minimális kapcsolatokat, amelyek még mindig biztosítják az összekapcsoltságot. Ezzel megérthetők az alapvető kapcsolati struktúrák és a közösség dinamikája.

#### Példák

##### Példa 1: Egyszerű gráf

Vegyünk egy egyszerű, súlyozott gráfot, amely az alábbi csomópontokból és élekből áll:

- Csomópontok: A, B, C, D
- Élek és súlyok:
   - A-B (4)
   - B-C (3)
   - C-D (2)
   - D-A (1)
   - B-D (5)

A Borůvka algoritmus lépésről lépésre így működik ezen a gráfon:

1. **Kezdeti állapot:** Mindegyik csomópont külön komponenst alkot.
2. **Első kör:** Minden komponens kiválasztja a legkisebb súlyú élét.
   - A választja: D-A (1)
   - B választja: B-C (3)
   - C szintén: B-C (3)
   - D választja: C-D (2)

3. **Komponensek egyesítése:** Az első kör után, az új élek hozzáadásával a komponensek egyesítése megtörténik.
   - A és D egyesül D-A éllel (1)
   - B és C egyesül B-C éllel (3)
   - Új élek hozzáadása: C-D (2) => A-D-C-B (ösvény)

A Borůvka algoritmus izlóján a minimális feszítőfa: D-A (1), B-C (3) és C-D (2) élekkel.

##### Példa 2: Implementáció

Az alábbiakban bemutatjuk a Borůvka algoritmusának egy egyszerű implementációját C++ nyelven. Ez az implementáció mutatja, hogyan lehet megkeresni a minimális feszítőfát egy súlyozott, nem irányított gráfban.

```cpp
#include <iostream>

#include <vector>
#include <algorithm>

#include <climits>
using namespace std;

class Edge {
public:
    int src, dest, weight;
    bool operator<(const Edge& edge) const {
        return this->weight < edge.weight;
    }
};

class Graph {
public:
    int V, E; // V: number of vertices, E: number of edges
    vector<Edge> edges;
    Graph(int V, int E) {
        this->V = V;
        this->E = E;
    }
    void addEdge(int u, int v, int w) {
        edges.push_back({u, v, w});
    }
    int find(vector<int>& parent, int i) {
        if (parent[i] == -1)
            return i;
        return parent[i] = find(parent, parent[i]);
    }
    void unionSets(vector<int>& parent, vector<int>& rank, int x, int y) {
        int xroot = find(parent, x);
        int yroot = find(parent, y);

        if (rank[xroot] < rank[yroot])
            parent[xroot] = yroot;
        else if (rank[xroot] > rank[yroot])
            parent[yroot] = xroot;
        else {
            parent[yroot] = xroot;
            rank[xroot]++;
        }
    }
    void boruvkaMST() {
        vector<int> parent(V, -1);
        vector<int> rank(V, 0);
        vector<int> cheapest(V, -1); 

        int numTrees = V;
        int MSTweight = 0;

        while (numTrees > 1) {
            fill(cheapest.begin(), cheapest.end(), -1);

            for (int i = 0; i < E; i++) {
                int u = edges[i].src;
                int v = edges[i].dest;
                int w = edges[i].weight;

                int set1 = find(parent, u);
                int set2 = find(parent, v);

                if (set1 != set2) {
                    if (cheapest[set1] == -1 || edges[cheapest[set1]].weight > w)
                        cheapest[set1] = i;
                    if (cheapest[set2] == -1 || edges[cheapest[set2]].weight > w)
                        cheapest[set2] = i;
                }
            }
            for (int i = 0; i < V; i++) {
                if (cheapest[i] != -1) {
                    int u = edges[cheapest[i]].src;
                    int v = edges[cheapest[i]].dest;
                    int w = edges[cheapest[i]].weight;

                    int set1 = find(parent, u);
                    int set2 = find(parent, v);

                    if (set1 != set2) {
                        MSTweight += w;
                        unionSets(parent, rank, set1, set2);
                        cout << "Edge " << u << "-" << v << " included in MST with weight " << w << endl;
                        numTrees--;
                    }
                }
            }
        }
        cout << "Weight of MST is " << MSTweight << endl;
    }
};

int main() {
    int V = 4, E = 5;
    Graph g(V, E);
    g.addEdge(0, 1, 10);
    g.addEdge(0, 2, 6);
    g.addEdge(0, 3, 5);
    g.addEdge(1, 3, 15);
    g.addEdge(2, 3, 4);

    g.boruvkaMST();
    return 0;
}
```

Ez a kód egy Graph osztályt definiál, amely tartalmazza a gráf csúcspontjainak és élének számát. Az addEdge() metódus hozzáad egy új élt a gráfhoz, míg a boruvkaMST() metódus végrehajtja a Borůvka algoritmusát, hogy megtalálja a minimális feszítőfát.
