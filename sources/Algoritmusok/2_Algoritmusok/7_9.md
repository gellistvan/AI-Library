\newpage

## 7.3.2. Edmonds-Karp algoritmus

Az Edmonds-Karp algoritmus a hálózatok elméletében egy gyakran alkalmazott módszer az áramlási problémák megoldására. Különösen a maximális áramlás meghatározására használják egy forrás és egy nyelő között egy irányított hálózatban, ahol az élek kapacitásokkal vannak ellátva. Az algoritmus az alapkoncepciójában a jól ismert Ford-Fulkerson módszerre épül, de egy lényeges kiegészítést tartalmaz: a szélességi keresés (BFS) alkalmazásával mindig a legrövidebb augmentáló utat keresi, azaz azt az utat, amelyen a legkevesebb él halad át. Ez a megközelítés garantálja az algoritmus hatékonyságát és polynomialitását, ami több gyakorlati alkalmazásban előnyös. Ebben a fejezetben részletesen bemutatjuk az Edmonds-Karp algoritmus alapelveit és implementációját, valamint megvizsgáljuk teljesítményét és optimalizálási lehetőségeit.

### 7.3.2.1. Alapelvek és implementáció

Az Edmonds-Karp algoritmus egy fontos és közkedvelt megoldás a maximális áramlási problémák megoldására áramlási hálózatokban. Az algoritmus lényegében a Ford-Fulkerson módszer egy specifikus implementációja, amely a szélességi keresés (BFS) algoritmust használja az augmentáló utak megtalálására a hálózatban. Az alábbiakban részletesen tárgyaljuk az Edmonds-Karp algoritmus alapelveit, és bemutatjuk a lépésről lépésre történő implementációját.

#### Az Edmonds-Karp algoritmus alapelvei

Az algoritmus fő célja, hogy meghatározza a forrás (s) és a nyelő (t) között maximálisan áramoltatható kapacitást egy irányított gráfban, ahol az élek kapacitásokkal vannak ellátva. Az algoritmus a következő módon működik:

1. **Inicializálás**: Kezdjük a hálózatot úgy, hogy az összes él kezdeti áramlása 0.
2. **Augmentáló út keresése**: Használjuk a szélességi keresést (BFS) a forrás (s) csomópontból indulva, hogy megtaláljuk a nyelő (t) csomópontig vezető utat, amely mentén még növelhetjük az áramlást. Az augmentáló út olyan út, amelyen minden él mentén van rendelkezésre álló kapacitás.
3. **Áramlás növelése**: Az augmentáló út mentén növeljük az áramlást a legkisebb rendelkezésre álló kapacitás mértékével.
4. **Visszatérő élek kezelése**: Frissítjük a hálózatot úgy, hogy az augmentáló út mentén a megadott áramlás növelésével csökkentjük az él kapacitását és növeljük a visszatérő él (reverz él) kapacitását.
5. **Iterálás**: Ismételjük a 2-4. lépéseket mindaddig, amíg nem találunk több augmentáló utat a forrástól a nyelőig.

#### Az Edmonds-Karp algoritmus implementációja

Az algoritmus részletes implementációját C++ nyelven az alábbiakban mutatjuk be, ahol a hálózatot egy irányított gráfos reprezentációban ábrázoljuk.

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <cstring>

const int INF = std::numeric_limits<int>::max(); // Defináljuk a végtelen kapacitást

class Graph {
    int V; // Csomópontok száma
    std::vector<std::vector<int>> capacity; // Kapacitás mátrix
    std::vector<std::vector<int>> adj; // Szomszédsági lista

public:
    Graph(int V) : V(V), capacity(V, std::vector<int>(V, 0)), adj(V) {}

    void addEdge(int u, int v, int cap) {
        capacity[u][v] = cap;
        adj[u].push_back(v);
        adj[v].push_back(u); // Reverz élek
    }

    int bfs(int s, int t, std::vector<int>& parent) {
        std::fill(parent.begin(), parent.end(), -1);
        parent[s] = -2;
        std::queue<std::pair<int, int>> q; // (csomópont, áramlás)
        q.push({s, INF});

        while (!q.empty()) {
            int curr = q.front().first;
            int flow = q.front().second;
            q.pop();

            for (int next : adj[curr]) {
                if (parent[next] == -1 && capacity[curr][next]) { // Ha még nem látogattuk meg, és van kapacitás
                    parent[next] = curr;
                    int new_flow = std::min(flow, capacity[curr][next]);
                    if (next == t)
                        return new_flow;
                    q.push({next, new_flow});
                }
            }
        }

        return 0;
    }

    int edmondsKarp(int s, int t) {
        int flow = 0;
        std::vector<int> parent(V);
        int new_flow;

        while (new_flow = bfs(s, t, parent)) {
            flow += new_flow;
            int curr = t;
            while (curr != s) {
                int prev = parent[curr];
                capacity[prev][curr] -= new_flow;
                capacity[curr][prev] += new_flow;
                curr = prev;
            }
        }

        return flow;
    }
};

int main() {
    Graph g(6);
    g.addEdge(0, 1, 16);
    g.addEdge(0, 2, 13);
    g.addEdge(1, 2, 10);
    g.addEdge(1, 3, 12);
    g.addEdge(2, 1, 4);
    g.addEdge(2, 4, 14);
    g.addEdge(3, 2, 9);
    g.addEdge(3, 5, 20);
    g.addEdge(4, 3, 7);
    g.addEdge(4, 5, 4);

    std::cout << "The maximum possible flow is " << g.edmondsKarp(0, 5) << std::endl;

    return 0;
}
```

#### Az implementáció fontos lépései

1. **Gráf reprezentálása**: A gráfot egy mátrixban (capacity) tároljuk, ahol a kapacitások tárolása történik, valamint egy szomszédsági listában (adj), hogy nyomon kövessük a csomópontok közötti kapcsolatokat és visszatérő éleket kezeljük.
2. **Szélességi keresés (BFS) implementálása**: A BFS a legfontosabb része az Edmonds-Karp algoritmusnak, mivel ezen keresztül határozzuk meg az augmentáló utakat. A BFS során egy queue (sor) segítségével járjuk be a gráfot az s csomópontból indulva, amíg el nem érjük a t csomópontot vagy kiürül a sor.
3. **Augmentáló út frissítése**: Miután egy augmentáló utat találtunk, frissítjük az élek kapacitását és a visszatérő élek kapacitását, hogy reflectáljuk az új áramlást. Ezt úgy érjük el, hogy a visszafelé mozogva csökkentjük az eredeti él kapacitását és növeljük a reverz élek kapacitását.
4. **Iteráció**: Az augmentáló út keresése és frissítése mindaddig ismétlődik, amíg lehet augmentáló utat találni.

Ezen implementáció során különösen fontos a BFS megfelelő implementációja és a kapacitás-mátrix (capacity) frissítése az augmentáló utak mentén történő áramlás változtatásakor. Az algoritmus időbeli komplexitása O(VE^2), ahol V a csomópontok száma és E az élek száma, mivel a BFS maximálisan E élt vizsgál egy iterációban, és legfeljebb O(E) augmentáló utat találhatunk.

### 7.3.2.2. Teljesítmény elemzés és optimalizálás


#### Futási Idő

Az Edmonds-Karp algoritmus futási ideje $O(VE^2)$, ahol $V$ a csúcsok száma és $E$ az élek száma a hálózatban. Az alábbiak szerint elemezzük a futási időt:

1. **BFS Időkomplexitása**:
    - A BFS végrehajtása $O(E)$ időt igényel, mivel minden élt legfeljebb egyszer látogatunk meg mindkét irányban.
    - Mivel minden BFS keresési szinten minden élt egyszer járunk be, az egy augmentáló ösvény keresésének ideje $O(E)$.

2. **Maximális Augmentáló Ösvények Száma**:
    - Az augmentáló ösvények maximális száma $O(VE)$-re becsülhető. Ez azért van, mert minden legszélesebb keresés során legalább egy él teljes kihasználtsággal fog működni, csökkentve a lehetséges augmentáló ösvények számát.

Ebből adódik a teljes időkomplexitás:
$$
O(VE) \text{ augmentáló ösvény } * O(E) \text{ keresési idő } = O(VE^2)
$$

#### Tárkomplexitás

A tárkomplexitás szintén kritikus tényező az Edmonds-Karp algoritmus elemzésénél. Ez tartalmazza a következő összetevőket:

1. **Adatszerkezet tárolása**:
    - A hálózatot egy szomszédsági lista vagy egy irányított gráf táblázat (adjacency matrix) képviseli. A szomszédsági lista $O(E)$ területet, míg az adjacency matrix $O(V^2)$ területet igényel.
    - Az augmentáló ösvény kereséséhez szükséges további memóriát, beleértve a BFS implementálásához használt queue-t és a csúcsok látogatott státuszát.

Ezért a szükséges memória mennyisége összességében $O(V + E)$, ami általánosan hatékony kezelést, garantált stabilitást és méretezhetőséget biztosít a különböző méretű hálózatok esetén.

#### Optimalizálási Stratégia

Az algoritmus teljesítményének optimalizálása több szempontból lehetséges, beleértve az idő- és tárkomplexitás minimalizálását. Az alábbiakban részletezünk néhány fontos optimalizálási technikát:

1. **Érlehető Hálózatok**:
    - Az élek kapacitásainak normalizálása és előzetes skálázása, hogy csökkentsük a keresési szintek számát.

2. **Hatékony Adatszerkezetek**:
    - Szomszédsági lista használata a memóriaterület csökkentése érdekében irányított gráfok esetén.
    - A BFS implementáció finomítása gyorsabb hozzáférési idők eléréséhez.

3. **Paralelizáció és Elosztott Feldolgozás**:
    - Az augmentáló ösvény keresésének és frissítésének paralelizálása.
    - Párhuzamos BFS és visszaható élek kapacitásainak korrigálása több processzor felhasználásával.

4. **Dinamikus Lemma Függvény**:
    - Dinamikus adatszerkezetek használata, mint például a Dinamikus Garázsgraf (Dynamic Graph) technológia a gyakori változtatások hatékony kezelésére.
    - Heurisztikák alkalmazása az augmentáló ösvény keresésében, hogy gyorsan elérjük a legfontosabb csomópontokat.

5. **Él Szűrés**:
    - Csökkentett kapacitású élek kizárása az augmentáló ösvény kereséséből, hogy elkerüljük azokat az éleket, amelyek minimális hatást gyakorolnak az áramlásra.
    - Kapacitáskorlátok finomítása az augmentációs lépésekhez, hogy jobb eredményeket érjünk el minimális keresési idő mellett.

#### C++ Implementáció

Az Edmonds-Karp algoritmus C++ nyelvű implementációja a következőképpen nézhet ki:

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <limits.h>
#include <algorithm>
#include <cstring>

using namespace std;

#define V 6 // Number of vertices in the graph

bool bfs(int rGraph[V][V], int s, int t, int parent[]) {
    bool visited[V];
    memset(visited, 0, sizeof(visited));
    queue<int> queue;
    queue.push(s);
    visited[s] = true;
    parent[s] = -1;

    while (!queue.empty()) {
        int u = queue.front();
        queue.pop();

        for (int v = 0; v < V; v++) {
            if (visited[v] == false && rGraph[u][v] > 0) {
                if (v == t) {
                    parent[v] = u;
                    return true;
                }
                queue.push(v);
                parent[v] = u;
                visited[v] = true;
            }
        }
    }
    return false;
}

int edmondsKarp(int graph[V][V], int s, int t) {
    int u, v;

    int rGraph[V][V];
    for (u = 0; u < V; u++)
        for (v = 0; v < V; v++)
            rGraph[u][v] = graph[u][v];

    int parent[V];
    int max_flow = 0;

    while (bfs(rGraph, s, t, parent)) {
        int path_flow = INT_MAX;

        for (v = t; v != s; v = parent[v]) {
            u = parent[v];
            path_flow = min(path_flow, rGraph[u][v]);
        }

        for (v = t; v != s; v = parent[v]) {
            u = parent[v];
            rGraph[u][v] -= path_flow;
            rGraph[v][u] += path_flow;
        }

        max_flow += path_flow;
    }

    return max_flow;
}

int main() {
    int graph[V][V] = { {0, 16, 13, 0, 0, 0},
                        {0, 0, 10, 12, 0, 0},
                        {0, 4, 0, 0, 14, 0},
                        {0, 0, 9, 0, 0, 20},
                        {0, 0, 0, 7, 0, 4},
                        {0, 0, 0, 0, 0, 0} };

    cout << "The maximum possible flow is " << edmondsKarp(graph, 0, 5);
    return 0;
}
```

Az Edmonds-Karp algoritmus egy hatékony módszer a maximális áramlás problémájának megoldására hálózatokban, azonban mint minden algoritmus, ez is finomítást igényel a különböző típusú hálózatok és erőforrások tekintetében. A futási és tárkomplexitásának alapos elemzése és optimalizálása elengedhetetlen a skálázhatóság és a gyorsaság szempontjából. Az implementáció és optimalizációk megfelelő mérlegelésével az Edmonds-Karp algoritmus a gyakorlatban is kiváló teljesítményt nyújthat, különösen nagy és komplex hálózatok kezelésekor.

