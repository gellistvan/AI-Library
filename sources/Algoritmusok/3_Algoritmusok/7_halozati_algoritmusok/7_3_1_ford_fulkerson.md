\newpage

# 7.3. Áramlási hálózatok algoritmusai

Az áramlási hálózatok algoritmusai a hálózati gráfok egyik legfontosabb és legérdekesebb területét képviselik. Ezek az algoritmusok a hálózati rendszerek optimalizálására és a maximális áramlás megtalálására összpontosítanak, különösen olyan környezetekben, ahol az erőforrások áramlása kulcsfontosságú. Legyen szó közlekedési hálózatokról, kommunikációs rendszerekről vagy logisztikai láncokról, az áramlási hálózatok megoldásai nélkülözhetetlenek a hatékony működés biztosításához. Ebben a szekcióban elmélyülünk az áramlási hálózatok elméleti alapjaiban, valamint bemutatjuk a legismertebb algoritmusokat, köztük a Ford-Fulkerson algoritmust, az Edmonds-Karp algoritmust és a Dinic algoritmust. Ezek az eszközök nemcsak a számítástudomány iránt érdeklődők számára jelenthetnek komoly kihívást és tanulási lehetőséget, hanem gyakorlati alkalmazásaikkal sok iparág működési hatékonyságát is növelhetik.

### 7.3.1. Ford-Fulkerson algoritmus

Az áramlási hálózatok elemzése és optimalizálása során az egyik legfontosabb és legeffektívebb módszer a Ford-Fulkerson algoritmus. Ez az algoritmus alapvető szerepet játszik az áramlások maximalizálásában irányított és kapacitáskorlátos hálózatokban. Alapvető működési elve az, hogy folyamatosan keres olyan útvonalakat a forrástól a nyelőig, amelyeken keresztül áramlást képes növelni, amíg további növelésre már nem adódik lehetőség. A következő alfejezetekben mélyrehatóan foglalkozunk a Ford-Fulkerson algoritmus működésének és implementációjának alapelveivel, valamint bemutatjuk, hogyan kezelhetők a kapacitáskorlátos hálózatok ezen módszer alkalmazásával. Felfedezzük a részleteket és kihívásokat, hogy az olvasó teljes képet kapjon az algoritmus hatékonyságáról és gyakorlati alkalmazásáról.

### 7.3.1.1. Alapelvek és implementáció

A Ford-Fulkerson algoritmus az áramlási hálózatok maximális áramlásának meghatározására szolgáló algoritmusok egyike. Az algoritmus a hálózat áramlási és kapacitási tulajdonságait kihasználva iteratív módon találja meg a forrástól a nyelőig terjedő legnagyobb áramlási útvonalakat, amelyek növelik a hálózaton keresztül átvihető áramlás mértékét. Az alábbiakban részletesen bemutatjuk az algoritmus alapelveit és egy C++ nyelven írt implementációját.

#### Alapelvek

Az algoritmus alapját az áramlási hálózat (flow network) fogalma képezi. Egy áramlási hálózat egy irányított gráf, amelynek csúcsai (vertices) vannak, és élei (edges) maximum kapacitással (capacity) rendelkeznek, amelyek meghatározzák, hogy mennyi áramlás folyhat át rajtuk. Az áramlási hálózat egy kiemelt forrást (source) és nyelőt (sink) is tartalmaz. Az algoritmus célja a forrástól a nyelőig terjedő maximális áramlás meghatározása.

##### 1. **Kezdeti állapot**:

Az algoritmus egy kezdeti állapottal indul, ahol minden élre vonatkozó kezdeti áramlás nulla.

##### 2. **Maradék hálózat** (Residual Network):

Az algoritmus a maradék hálózatot használja az aktuális áramlás és a még használható kapacitás meghatározására. A maradék hálózat az eredeti hálózat egy olyan változata, ahol minden él kapacitása csökkent a már átfolyt áramlás mértékével. Ha egy (u, v) él eredeti kapacitása `c(u, v)` és áramlása `f(u, v)` akkor a maradék hálózatban (u, v) él kapacitása `c'(u, v) = c(u, v) - f(u, v)`. Ezen kívül, a fordított él (v, u) kapacitása `c'(v, u) = f(u, v)` (ez lehetővé teszi az áramlás visszafordítását, ha szükséges).

##### 3. **Útkeresés**:

Az algoritmus egy útkeresési módszert alkalmaz a maradék hálózatban, hogy megtaláljon egy forrástól a nyelőig tartó utat, amelyen még további áramlást lehet vinni (ezt hívjuk növelőútnak, augmenting path). Tipikusan szélességi keresést (BFS) vagy mélységi keresést (DFS) használnak erre a feladatra.

##### 4. **Áramlás növelése**:

Ha egy növelőutat találunk, az út mentén lévő összes él maximális kapacitását figyelembe véve meghatározza a növelési értéket (bottleneck capacity), amely a legkisebb kapacitás az út mentén. Ezután ezen a növelőúton ennek megfelelően megnöveli az áramlást.

##### 5. **Iteráció**:

Az algoritmus addig ismétli az útkeresést és áramlás növelését, amíg nem talál több növelőutat. Ekkor az aktuális áramlás lesz a maximális áramlás a hálózatban.

#### Pseudocode és C++ kód

Az alábbiakban egy egyszerű útmutató található az algoritmus lépéseinek implementálásához, valamint egy minta C++ nyelven írt kóddal.

Pseudocode:
```text
1. Initialize flow f to 0 for all edges
2. While there exists an augmenting path p from source to sink in the residual network
   1. Find the residual capacity along the path (bottleneck capacity)
   2. For each edge(u, v) in p:
        - Increase flow f(u, v) by bottleneck capacity
        - Decrease flow f(v, u) by bottleneck capacity (to handle reverse edges)
3. Return the total flow from source to sink
```

C++ Implementáció:
```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <cstring>

#define V 6  // Number of vertices in the graph

using namespace std;

// A utility function to check if there is a path from source to sink in the residual graph.
// This implementation uses Breadth-First Search (BFS).
bool bfs(int rGraph[V][V], int s, int t, int parent[]) {
    bool visited[V];
    memset(visited, 0, sizeof(visited));

    queue<int> q;
    q.push(s);
    visited[s] = true;
    parent[s] = -1;

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v = 0; v < V; v++) {
            if (visited[v] == false && rGraph[u][v] > 0) {
                if (v == t) {
                    parent[v] = u;
                    return true;
                }
                q.push(v);
                parent[v] = u;
                visited[v] = true;
            }
        }
    }

    return false;
}

// Returns the maximum flow from s to t in the given graph.
int fordFulkerson(int graph[V][V], int s, int t) {
    int u, v;

    // Create a residual graph and fill it with initial capacities from the input graph.
    int rGraph[V][V];
    for (u = 0; u < V; u++) {
        for (v = 0; v < V; v++) {
            rGraph[u][v] = graph[u][v];
        }
    }

    int parent[V];  // This array is used to store the augmenting path.
    int max_flow = 0;  // Initialize the maximum flow to 0.

    // Augment the flow while there is a path from source to sink.
    while (bfs(rGraph, s, t, parent)) {
        // Find the maximum flow through the found path.
        int path_flow = INT_MAX;
        for (v = t; v != s; v = parent[v]) {
            u = parent[v];
            path_flow = min(path_flow, rGraph[u][v]);
        }

        // Update the residual capacities of the edges and reverse edges in the path.
        for (v = t; v != s; v = parent[v]) {
            u = parent[v];
            rGraph[u][v] -= path_flow;
            rGraph[v][u] += path_flow;
        }

        // Add path flow to the overall flow.
        max_flow += path_flow;
    }

    return max_flow;
}

int main() {
    // Create a graph represented as an adjacency matrix.
    int graph[V][V] = {
                        {0, 16, 13, 0, 0, 0},
                        {0, 0, 10, 12, 0, 0},
                        {0, 4, 0, 0, 14, 0},
                        {0, 0, 9, 0, 0, 20},
                        {0, 0, 0, 7, 0, 4},
                        {0, 0, 0, 0, 0, 0}
                      };

    cout << "The maximum possible flow is " << fordFulkerson(graph, 0, 5) << endl;

    return 0;
}
```

#### Magyarázat

A fenti C++ implementációban a `bfs` függvény szélességi keresést használ a maradék hálózatban, hogy megtalálja a növelőutat a forrástól a nyelőig. Ha ilyen utat talál, akkor az áramlást növeli a legkeskenyebb kapacitásnak megfelelően az út mentén. Az `fordFulkerson` függvény a maximum áramlást keresi iterálva addig, amíg nem talál több növelőutat. Végül, visszaadja a maximum áramlást.

Az algoritmus hatékonysága nagyban függ az alkalmazott útkeresési módszertől. A legrosszabb esetben az algoritmus időbonyolultsága O(E * |f*|), ahol E a gráf éleinek száma, és |f*| a maximális áramlás értéke. Azonban a gyakorlatban gyakran elegendő megfelelő növelőutakat találni ahhoz, hogy hatékonyan működjön.

Megjegyzés: A fenti implementáció csak egy alapvető példa és nem tartalmaz minden lehetséges optimalizálást vagy edge case-t. Az önálló implementációk során érdemes figyelembe venni a specifikus alkalmazási környezet és követelmények sajátosságait.


### 7.3.1.2. Kapacitáskorlátos hálózatok kezelése

Az áramlási hálózatok algoritmusainak vizsgálatánál gyakran találkozunk kapacitáskorlátos hálózatokkal. Ezek lényege, hogy az egyes éleken korlátozott kapacitás áll rendelkezésre, amely meghatározza, hogy az adott él mentén mekkora maximális áramlás bonyolítható le. Ebben a részben részletesen megvizsgáljuk, hogyan kezelhetjük ezen korlátokat a Ford-Fulkerson algoritmus és annak variánsai segítségével.

#### 1. Kapacitáskorlátos hálózatok fogalma

Egy kapacitáskorlátos hálózatban a gráf egyes éleihez rendelünk egy kapacitásértéket, amely jelzi az adott él maximális átbocsátóképességét. Formálisan, legyen $G = (V, E)$ egy irányított gráf, ahol $V$ a csúcsok halmaza és $E$ az élek halmaza. Továbbá, rendelünk minden $e \in E$ élhez egy $c(e)$ kapacitásértéket, ami reprezentálja az él által megengedett maximális áramlást:

$$
c: E \rightarrow \mathbb{R}^+
$$

Ezen kívül definiáljuk a forrás ($s \in V$) és nyelő ($t \in V$) csúcsokat, ahol a forrás az az csúcs, ahonnan az áramlás indul, a nyelő pedig az a csúcs, ahová az áramlás megérkezik.

#### 2. Ford-Fulkerson algoritmus áttekintése

A Ford-Fulkerson algoritmus egy iteratív eljárás, amely keresési módszereken (pl. mélységi vagy szélességi keresés) alapulva talál augmentáló utakat a kapacitáskorlátos hálózatban, majd ezen utakon való áramlás növelésével törekszik a maximális áramlás megtalálására. Az algoritmus az alábbi lépésekből áll:

1. **Inicializálás:** Kezdjük az áramlást nullával.
2. **Augmentáló út keresése:** Keressünk egy úgynevezett augmentáló utat a forrás és a nyelő között. Egy augmentáló út egy olyan ösvény, amely mentén növelhető az áramlás.
3. **Áramlás növelése:** Az augmentáló úton növeljük az áramlást olyan mértékben, amennyire a legszűkebb kapacitás engedi.
4. **Primitív lépés ismétlése:** Ismételjük meg a folyamatot, amíg nem találunk több augmentáló utat.

#### 3. Kapacitáskorlátok kezelése

A kapacitáskorlátok figyelembevételét a keresés során biztosítani kell. Ennek érdekében használhatjuk a következő módszereket:

##### 3.1. Maradékhálózat

A maradékhálózat fogalma alapvető a Ford-Fulkerson algoritmus megértéséhez. A maradékhálózatban figyelembe vesszük az aktuális áramlást, és ennek megfelelően határozzuk meg az egyes élek kapacitását:

- Ha egy él mentén még van hely az áramlás növelésére, akkor az él maradék kapacitása $c(u, v) - f(u, v)$, ahol $c(u, v)$ az él kapacitása és $f(u, v)$ az aktuális áramlás.
- Ha egy él mentén már van áramlás, akkor létrehozunk egy visszacsatoló élt, amelyen az aktuális áramlás visszavezethető, ennek kapacitása $f(u, v)$.

A maradékhálózatban végzett kereséssel és az augmentáló utak megtalálásával biztosítani tudjuk, hogy az új áramlás ne haladja meg a kapacitásokat.

##### 3.2. Augmentáló út keresése és áramlás frissítése

Az augmentáló út keresése történhet mélységi kereséssel (DFS) vagy szélességi kereséssel (BFS). Amint megtaláltuk az augmentáló utat, az új áramlást az alábbiak szerint frissítjük minden élen az úton:

- Ha az élen előrehaladunk, növeljük az áramlást: $f(u, v) = f(u, v) + \Delta$
- Ha az élen visszafelé haladunk, csökkentjük az áramlást: $f(v, u) = f(v, u) - \Delta$

Ahol $\Delta$ a legkisebb maradék kapacitás az augmentáló úton. Ezen lépések biztosítják, hogy a kapacitáskorlátokat betartjuk az áramlást minden lépésben.

#### 4. Pseudo-kód Implementáció (C++)

Az alábbiakban egy átfogó példa következik a maradékhálózat kezelésére és az augmentáló utak keresésére a Ford-Fulkerson algoritmusban:

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <cstring>

using namespace std;

const int MAX_N = 1000;
int capacity[MAX_N][MAX_N]; // capacity[u][v] gives the capacity of the edge from u to v
int flow[MAX_N][MAX_N];     // flow[u][v] gives the flow from u to v
vector<int> adj[MAX_N];     // adjacency list of the graph

// Breadth-First Search to find an augmenting path
bool bfs(int s, int t, vector<int>& parent) {
    fill(parent.begin(), parent.end(), -1);
    parent[s] = -2;
    queue<pair<int, int>> q;
    q.push({s, INT_MAX});

    while (!q.empty()) {
        int u = q.front().first;
        int curr_flow = q.front().second;
        q.pop();

        for (int v : adj[u]) {
            if (parent[v] == -1 && capacity[u][v] - flow[u][v] > 0) {
                parent[v] = u;
                int new_flow = min(curr_flow, capacity[u][v] - flow[u][v]);
                if (v == t)
                    return new_flow;

                q.push({v, new_flow});
            }
        }
    }

    return 0;
}

int ford_fulkerson(int s, int t) {
    int max_flow = 0;
    vector<int> parent(adj.size());

    int new_flow;

    while (new_flow = bfs(s, t, parent)) {
        max_flow += new_flow;
        int u = t;
        while (u != s) {
            int v = parent[u];
            flow[v][u] += new_flow;
            flow[u][v] -= new_flow;
            u = v;
        }
    }

    return max_flow;
}

int main() {
    int n, m; // Number of nodes and edges
    cin >> n >> m;

    for (int i = 0; i < m; ++i) {
        int u, v, c;
        cin >> u >> v >> c;
        capacity[u][v] = c;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    int s, t; // source and sink
    cin >> s >> t;

    cout << "Maximum Flow: " << ford_fulkerson(s, t) << endl;

    return 0;
}
```

#### 5. Az algoritmus komplexitása

A Ford-Fulkerson algoritmus futási idejét a keresési módszer határozza meg:

- DFS használata esetén az algoritmus nem garantáltan véges lépésszámú, különösen ha az élkapacitások valós számok. Az algoritmus végtelen ciklusba kerülhet.
- BFS használata esetén az algoritmus Edmonds-Karp verziója polinomiális időben fut, futási ideje $O(E^2V)$, ahol $E$ az élek száma és $V$ a csúcsok száma.

#### 6. Esettanulmány és alkalmazások

A Ford-Fulkerson algoritmus és annak variánsai széles körben alkalmazhatóak különféle problémák megoldására, például:

- **Hálózati tervezés:** Adatok hatékony szállítása a hálózaton.
- **Szerszámgépek ütemezése:** Optimális munkatervezés korlátos erőforrásokkal.
- **Közlekedési rendszerek:** Forgalomirányítás és kapacitáskezelés.
- **Projekt csoportok:** Feladatok szétosztása erőforráskorlátokkal rendelkező csapatok között.

A kapacitáskorlátos hálózatok kezelése egy alapvető és kritikus elem a Ford-Fulkerson algoritmusban, amely biztosítja a kapacitások korlátozását minden lépésben, ezáltal optimalizálva az áramlást és biztosítva, hogy a maximális áramlás problémája helyes és hatékony módon legyen megoldva.

