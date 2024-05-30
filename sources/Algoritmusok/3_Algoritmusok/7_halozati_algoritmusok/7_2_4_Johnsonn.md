\newpage

## 7.2.4. Johnson algoritmus

A Johnson algoritmus egy hatékony kombinált módszer, amely a Bellman-Ford és a Dijkstra algoritmusok előnyeit használja ki annak érdekében, hogy a legolcsóbb utak problémáját megoldja súlyozott, irányított grafokban, amelyek negatív él súlyokat is tartalmazhatnak. Az algoritmus kifejezetten hasznos nagy méretű grafok esetén, mivel képes mind a pozitív, mind a negatív súlyú élek kezelésére, miközben időbeli hatékonyságot biztosít azáltal, hogy Dijkstra algoritmusát alkalmazza egy átsúlyozott grafon. Az alábbi fejezetek részletesen bemutatják a Johnson algoritmus alapelveit, annak lépésről lépésre történő implementációját, valamint azokat a technikákat, amelyek segítségével hatékonyan alkalmazható akkor is, ha a graf mérete jelentős.

### 7.2.4.1. Alapelvek és implementáció

A Johnson algoritmus egy hatékony technika, amely lehetővé teszi a legolcsóbb út (shortest-path) meghatározását irányított, súlyozott grafokban, még akkor is, ha a graf tartalmaz negatív súlyú éleket, feltéve, hogy nincs negatív súlyú kör (negative weight cycle). Az algoritmus egyik legfontosabb előnye, hogy kombinálja a Bellman-Ford algoritmus képességét a negatív súlyú élek kezelésére és a Dijkstra algoritmus hatékonyságát.

#### Alapelvek

A Johnson algoritmus három fő lépésben működik:

1. **Átalakító fázis**: Ebben a lépésben egy új csúcsot (s) adunk a grafhoz, amelyet összekötünk az összes többi csúccsal nulla súlyú élekkel. Ezután a Bellman-Ford algoritmust alkalmazzuk az új csúcsra, hogy meghatározzuk az összes többi csúcshoz vezető legolcsóbb utat. Ezeket az utakat később a súlyok átalakításához használjuk.

2. **Súlyozás átszámítása**: Az átalakító fázis után módosítjuk a graf él súlyait egy speciális módszerrel, amely garantálja, hogy minden él súlya nem negatív legyen. Az új él súlyokat az $h(u)$ és $h(v)$ funkciók alapján számítjuk ki, ahol $h$ az új csúcs és az összes többi csúcs közötti legolcsóbb utat képviseli a Bellman-Ford algoritmussal meghatározottak szerint.

3. **Dijkstra algoritmus alkalmazása**: Miután az élsúlyokat átszámítottuk, Dijkstra algoritmusát használjuk minden csúcsra a legolcsóbb utak meghatározásához. A végső eredményeket visszafordítjuk az eredeti súlyokhoz a súly átalakítást alkalmazva.

#### Részletes magyarázat

1. **Átalakító fázis**:
    - Hozzunk létre egy új csúcsot $s$, amelyből nulla súlyú élek indulnak az összes többi csúcsba.
    - Alkalmazzuk a Bellman-Ford algoritmust $s$ csúcsra. Ha a Bellman-Ford negatív ciklust talál, akkor a grafban nincs megoldható legolcsóbb út probléma.

    ```c++
    // Create a new vertex 's' and add edges from 's' to all other vertices with weight 0
    // Use Bellman-Ford algorithm to find shortest path estimate 'h' from 's' to all vertices
    std::vector<int> bellmanFord(Graph &graph, int s) {
        int V = graph.vertices;
        std::vector<int> distance(V + 1, INT_MAX);
        distance[s] = 0;

        for (int i = 1; i <= V; i++) {
            for (const Edge &edge : graph.edges) {
                if (distance[edge.u] != INT_MAX &&
                    distance[edge.u] + edge.weight < distance[edge.v]) {
                    distance[edge.v] = distance[edge.u] + edge.weight;
                }
            }
        }

        for (const Edge &edge : graph.edges) {
            if (distance[edge.u] != INT_MAX &&
                distance[edge.u] + edge.weight < distance[edge.v]) {
                throw std::runtime_error("Graph contains a negative-weight cycle");
            }
        }

        return distance;
    }
    ```

2. **Súlyozás átszámítása**:
    - Definiáljuk az új súlyokat a következőképpen:
      $$
      w'(u, v) = w(u, v) + h(u) - h(v)
      $$
      Ez biztosítja, hogy minden átszámított él súlya nem negatív legyen. Ebben az esetben $h$ az az érték, amelyet a Bellman-Ford algoritmus adott.

    ```c++
    void reweightEdges(Graph &graph, const std::vector<int> &h) {
        for (Edge &edge : graph.edges) {
            edge.weight = edge.weight + h[edge.u] - h[edge.v];
            // Note: Assuming valid indices and that h is properly computed
        }
    }
    ```

3. **Dijkstra algoritmus alkalmazása**:
    - Miután átszámítottuk az él súlyokat, Dijkstra algoritmust alkalmazunk minden csúcsra, hogy meghatározzuk a legolcsóbb utakat.
    - Az eredményeket az eredeti súlyokra vissza alakítjuk:
      $$
      d(u, v) = d'(u, v) + h(v) - h(u)
      $$

    ```c++
    std::vector<int> dijkstra(const Graph &graph, int start) {
        int V = graph.vertices;
        std::vector<int> dist(V, INT_MAX);
        dist[start] = 0;
        
        using PII = std::pair<int, int>; 
        std::priority_queue<PII, std::vector<PII>, std::greater<PII>> pq;
        pq.push({0, start});

        while (!pq.empty()) {
            int u = pq.top().second;
            int d = pq.top().first;
            pq.pop();

            if (d != dist[u]) {
                continue;
            }

            for (const auto &edge : graph.adj[u]) {
                int v = edge.first;
                int weight = edge.second;

                if (dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    pq.push({dist[v], v});
                }
            }
        }

        return dist;
    }

    std::vector<std::vector<int>> johnson(Graph &graph) {
        int V = graph.vertices;

        // Step 1: Augment the graph with an additional vertex `s`
        Graph augmentedGraph = graph;
        int s = V;
        for (int i = 0; i < V; ++i) {
            augmentedGraph.edges.push_back(Edge{s, i, 0});
        }

        // Step 2: Run Bellman-Ford to obtain `h`
        std::vector<int> h;
        try {
            h = bellmanFord(augmentedGraph, s);
        } catch (const std::runtime_error &e) {
            std::cerr << e.what() << std::endl;
            // If there's a negative-weight cycle
            throw;
        }

        // Step 3: Reweight the edges
        reweightEdges(graph, h);

        // Step 4: Use Dijkstra's algorithm to find shortest paths
        std::vector<std::vector<int>> allPairsShortestPaths(V, std::vector<int>(V, INT_MAX));
        for (int u = 0; u < V; ++u) {
            allPairsShortestPaths[u] = dijkstra(graph, u);
        }

        // Reverse the reweighting
        for (int u = 0; u < V; ++u) {
            for (int v = 0; v < V; ++v) {
                if (allPairsShortestPaths[u][v] < INT_MAX) {
                    allPairsShortestPaths[u][v] = allPairsShortestPaths[u][v] + h[v] - h[u];
                }
            }
        }

        return allPairsShortestPaths;
    }
    ```

#### Algoritmikus komplexitás

A Johnson algoritmus időkomplexitása $O(V^2 \log V + VE)$:
- Bellman-Ford időkomplexitása $O(VE)$.
- Átszámítás $O(E)$.
- Minden Dijkstra futtatás $O(V \log V + E)$, és mivel minden csúcson fut, összesen $O(V (V \log V + E))$.

Ennek eredménye $O(VE + V^2 \log V + VE)$, amit általában $O(V^2 (E + \log V))$-ként elég endő.

### 7.2.4.2. Nagy méretű grafok kezelése

A Johnson algoritmus egy olyan hatékony módszer, amely lehetővé teszi, hogy megtaláljuk a legrövidebb utakat minden csúcsból minden másik csúcsba egy súlyozott, irányított gráfban. Ez az algoritmus különösen előnyös akkor, amikor a gráfban negatív él-hosszok is előfordulhatnak. Az algoritmus kombinálja a Bellman-Ford algoritmus előnyeit a Dijkstra algoritmuséval, így képes egyaránt kezelni negatív és nem-negatív él-hosszokat.

#### Johnson algoritmus alapelvei

A Johnson algoritmus lényege, hogy először egy új csúcsot ad a gráfhoz, majd ezen új csúcsból minden eredeti csúcsba hozzáad egy null hosszúságú élt. Ezt követően a Bellman-Ford algoritmust futtatja az új csúcsból, hogy meghatározza az ún. potenciál (h) értékeket minden csúcsra vonatkozóan. Ezen potenciál értékek segítségével újrasúlyozza az eredeti gráf éleit úgy, hogy az új él-súlyok nem lesznek negatívak. Végül a Dijkstra algoritmust használja kiinduló csúcsonként a legrövidebb utak megtalálására az új, átsúlyozott gráfban.

#### Nagy méretű grafok kezelése

Amikor a Johnson algoritmust nagy méretű grafok esetén alkalmazzuk, számos kihívással kell szembenéznünk. Ezek a kihívások magukba foglalják az idő- és memóriaigény növekedését, az algoritmus hatékonyságát és a párhuzamos feldolgozás lehetőségét.

**1. Idő- és memóriaigény**

Nagy méretű grafok esetén a Johnson algoritmus időigénye megnő. Az időkomplexitás O(V^2 log V + VE), ahol V a csúcsok száma és E az élek száma. A memóriaigény is fontos szempont, mivel minden csúcsra és élre vonatkozóan tárolni kell az adatokat.

Ennek mérséklése érdekében elengedhetetlen optimalizációkat és hatékony adatszerkezeteket bevezetni. Például:
- **Használjunk min-kupacot**: A Dijkstra algoritmus hatékonyságának javítása érdekében a prioritási sorban végrehajtott műveletek optimalizálásához.
- **Gráfreprezentáció optimalizálása**: Az élhalmazok tárolására használhatunk szomszédsági listákat vagy mátrixokat a szükséges memória csökkentésére.

**2. Párhuzamos feldolgozás**

A modern számítógépek lehetőséget kínálnak párhuzamos feldolgozásra, amely drasztikusan növelheti az algoritmus hatékonyságát nagy méretű gráfok esetén. Az algoritmus párhuzamosítható részeket is tartalmaz, például:
- **Bellman-Ford algoritmus párhuzamosítása**: Minden csúcsra párhuzamosan számolhatjuk ki a Bellman-Ford algoritmust.
- **Dijkstra algoritmus párhuzamosítása**: Az algoritmus különböző kiinduló csúcsai esetén párhuzamosan futhat.

**3. Külső memória és streaming algoritmusok**

Ha az egész gráfot nem tudjuk a memóriába betölteni, alkalmazhatunk külső memóriás algoritmusokat vagy streaming algoritmusokat. Ezek az algoritmusok lehetővé teszik, hogy a gráfot fokozatosan dolgozzuk fel, miközben csak a szükséges részeket tartjuk memóriában.

**4. Skálázhatóság**

A Johnson algoritmus hatékonyságának megőrzése érdekében megfelelően kell skálázni az adatstruktúrákat és a lépések implementációját. A large-scale gráfoknál célszerű növekvő szimbolikus hivatkozásokat és optimalizációkat alkalmazni, mint például a szikratérkép (sparsity map) használatát a ritkás gráfoknál.

#### Példa: Johnson algoritmus implementációja C++ nyelven

Az alábbi példa bemutatja a Johnson algoritmus fő lépéseit és annak részleteit C++ nyelven.

```cpp
#include <vector>
#include <limits>
#include <queue>
#include <algorithm>

using namespace std;

struct Edge {
    int u, v;
    int weight;
};

const int INF = numeric_limits<int>::max();

class JohnsonAlgorithm {
public:
    JohnsonAlgorithm(int vertices, const vector<Edge>& edges)
        : V(vertices), E(edges), adj(V), h(V, INF), dist(V, vector<int>(V, INF)) {}

    bool bellmanFord() {
        vector<int> dist(V + 1, INF);
        dist[V] = 0;

        for (int i = 0; i < V; ++i) {
            for (const auto& edge : E) {
                if (dist[edge.u] != INF && dist[edge.u] + edge.weight < dist[edge.v]) {
                    dist[edge.v] = dist[edge.u] + edge.weight;
                }
            }
        }

        for (const auto& edge : E) {
            if (dist[edge.u] != INF && dist[edge.u] + edge.weight < dist[edge.v]) {
                return false;
            }
        }

        for (int i = 0; i < V; ++i) {
            h[i] = dist[i];
        }

        return true;
    }

    void dijkstra(int src) {
        vector<int> dist(V, INF);
        dist[src] = 0;
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
        pq.push({0, src});

        while (!pq.empty()) {
            int d = pq.top().first;
            int u = pq.top().second;
            pq.pop();

            if (d != dist[u]) continue;

            for (const auto& edge : adj[u]) {
                int v = edge.first, weight = edge.second;
                if (dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    pq.push({dist[v], v});
                }
            }
        }

        for (int i = 0; i < V; ++i) {
            if (dist[i] != INF) {
                this->dist[src][i] = dist[i] + h[i] - h[src];
            } else {
                this->dist[src][i] = INF;
            }
        }
    }

    vector<vector<int>> johnson() {
        for (const auto& edge : E) {
            adj[edge.u].push_back({edge.v, edge.weight});
        }

        if (!bellmanFord()) {
            throw runtime_error("Graph contains a negative weight cycle");
        }

        for (auto& edge : E) {
            edge.weight += h[edge.u] - h[edge.v];
        }

        for (int u = 0; u < V; ++u) {
            adj[u].clear();
        }
        for (const auto& edge : E) {
            adj[edge.u].push_back({edge.v, edge.weight});
        }

        for (int u = 0; u < V; ++u) {
            dijkstra(u);
        }

        return dist;
    }

private:
    int V;
    vector<Edge> E;
    vector<vector<pair<int, int>>> adj;
    vector<int> h;
    vector<vector<int>> dist;
};
```



