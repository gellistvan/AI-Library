\newpage

## 7.3.3. Dinic algoritmus

Az áramlási hálózatok elméletének egyik leghatékonyabb és legtöbbet kutatott algoritmusa a Dinic algoritmus. Ezt az algoritmust az izraeli matematikus, Yefim Dinitz fejlesztette ki 1970-ben, és azóta is az egyik legfontosabb eszköz a maximális áramlás problémájának megoldására. A Dinic algoritmus különlegessége az, hogy többé-kevésbé kombinálja a szélességi keresést és az iterációs mélységi keresést a maradék hálózatban, hogy rétegzett hálózatokat hozzon létre, amelyek lehetővé teszik az áramlás blokkoló jellegű frissítését. A következő alfejezetekben bemutatjuk az alapelveket és az algoritmus lépéseit, majd részletesen tárgyaljuk a blokkoló áramlások kezelésének technikáit, amelyek az algoritmus hatékonyságának kulcsfontosságú elemei.

### 7.3.3.1. Alapelvek és implementáció

#### Az algoritmus alapelvei

Dinic algoritmusának alapelvei az alábbiakban foglalhatók össze:

1. **Szintezett hálózat**: Az algoritmus minden iterációban létrehoz egy szintezett hálózatot a forrásból kiindulva egy keresési eljárással, amely gyakran Breadth-First Search (BFS). Ez a szintezett hálózat rétegeket hoz létre, amelyeket felhasználva az áramlás növelése történik.

2. **Blokkoló áramlás**: A szintezett hálózaton belül a Dinic algoritmus megkeresi a blokkói áramlást. Ez az a maximális áramlás, amelyet az adott szintezett hálózaton belül lehet találni, és amely minden utat blokkol, azaz minden úton visszatérünk a forráshoz vagy egy olyan ponthoz, ahol már nincs további kapacitás a növelésre.

3. **Áramlás növelése és frissítése**: Az algoritmus frissíti a hálózat kapacitásait az áramlás visszaigazításával, és ha még lehet áramlást növelni, új szintezett hálózatot hoz létre, amíg nem talál több blokkói áramlást.

#### Az algoritmus lépései

Az alábbi lépések összefoglalják a Dinic algoritmus működését:

1. **Szintezett hálózat építése**:
    - Végezze el a BFS-t a forrásból, hogy létrehozzon egy szintezett hálózatot. A szintezett hálózat minden csúcsát az alapján osztályozzuk, hogy hány él szükséges az eléréséhez a forrásból.

2. **Blokkoló áramlás keresése**:
    - Használjon Depth-First Search (DFS) algoritmust a blokkói áramlás megtalálására a szintezett hálózatban. Minden utat követve, ameddig nem talál további lehetséges áramlási utat, majd visszaigazítja az áramlást és az élkapacitásokat a hálózatban.

3. **Áramlás növelése és frissítése**:
    - Frissítse a kapacitásokat és az áramlásokat az összes feltárt út mentén. Ha újabb blokkói áramlásokat talál, ismételje meg az előző lépéseket.

4. **Az algoritmus befejezése**:
    - Az algoritmus akkor fejeződik be, amikor nem talál több blokkói áramlást, azaz a BFS nem talál új szintezett hálózatot.

#### Implementáció

A következő példa bemutat egy C++ nyelvű implementációt a Dinic algoritmusra.

```cpp
#include <iostream>

#include <vector>
#include <queue>

#include <algorithm>
#include <climits>

#include <cstring>

using namespace std;

class Dinic {
public:
    struct Edge {
        int v, flow, C, rev;
    };

    int level[1000], start[1000];
    vector<Edge> adj[1000];

    void addEdge(int u, int v, int C) {
        Edge a = {v, 0, C, adj[v].size()};
        Edge b = {u, 0, 0, adj[u].size()};
        adj[u].push_back(a);
        adj[v].push_back(b);
    }

    bool BFS(int s, int t) {
        memset(level, -1, sizeof(level));
        level[s] = 0;
        queue<int> q;
        q.push(s);

        while (!q.empty()) {
            int u = q.front();
            q.pop();

            for (Edge &e : adj[u]) {
                if (level[e.v] < 0 && e.flow < e.C) {
                    level[e.v] = level[u] + 1;
                    q.push(e.v);
                }
            }
        }
        return level[t] >= 0;
    }

    int sendFlow(int u, int flow, int t, int start[]) {
        if (u == t) return flow;

        for (; start[u] < adj[u].size(); start[u]++) {
            Edge &e = adj[u][start[u]];

            if (level[e.v] == level[u] + 1 && e.flow < e.C) {
                int curr_flow = min(flow, e.C - e.flow);
                int temp_flow = sendFlow(e.v, curr_flow, t, start);

                if (temp_flow > 0) {
                    e.flow += temp_flow;
                    adj[e.v][e.rev].flow -= temp_flow;
                    return temp_flow;
                }
            }
        }
        return 0;
    }

    int maxFlow(int s, int t) {
        if (s == t) return -1;

        int total = 0;
        while (BFS(s, t)) {
            memset(start, 0, sizeof(start));

            while (int flow = sendFlow(s, INT_MAX, t, start))
                total += flow;
        }
        return total;
    }
};

int main() {
    Dinic dinic;
    int n, m;
    cin >> n >> m;
    for (int i = 0; i < m; i++) {
        int u, v, c;
        cin >> u >> v >> c;
        dinic.addEdge(u, v, c);
    }

    cout << dinic.maxFlow(0, n - 1) << endl;
    return 0;
}
```

#### A Dinic algoritmus hatékonysága

A Dinic algoritmus időbeli hatékonysága O(V^2 * E), ahol V a csúcsok száma és E az élek száma a hálózatban. Fontos megjegyezni, hogy ez az idő bővítési és összehúzási stratégiák nélküli legrosszabb esetbeli futási idő. Az algoritmus azonban gyakran sokkal gyorsabb bizonyos típusú hálózatokban, különösen ha kevés blokkói áramlás van.

#### Alkalmazások és fontosság

Dinic algoritmusának egyik legfontosabb alkalmazása a távközlési és közlekedési hálózatok kapacitásának optimalizálása, de használható bármilyen olyan esetben, ahol maximális áramlás szükséges egy hálózatban, például logisztikai rendszerek vagy adatfeldolgozási hálózatok tervezése során.

Az algoritmus részleteinek mélyebb megértéséhez érdemes megismerni az áramlási hálózatok alapjait, valamint más, hasonló algoritmusokat, mint például az Edmonds-Karp algoritmust vagy a Push-Relabel algoritmust. A Dinic algoritmus különlegessége az általa használt szintezett hálózati struktúrában és a blokkói áramlások keresésének hatékonyságában rejlik.

### 7.3.3.2. Blokkoló áramlások kezelése

A hálózati áramlás problémái közül különösen fontos a blokkoló áramlások kezelése, mivel ezek gyakran kulcsszerepet játszanak az optimalizációs folyamatban és a hálózat maximális áramlásának meghatározásában. Dinic algoritmus, amelyet Evgeny Dinitz fejlesztett ki, kimondottan a maximális áramlás probléma hatékony megoldására szolgál, köszönhetően többek között a rendezett bejárásnak (levelezõ bejárás).

#### Blokkoló áramlás fogalma és jellemzői

Blokkoló áramlás egy olyan részáramlást jelent Dinic algoritmusában, amikor az augmentációs folyamat során elérjük azt az állapotot, hogy nem létezik több augmentációs út, amely még növelni tudná az áramlást. Ez azt jelenti, hogy minden él vagy teljesen telített vagy visszafelé él. Az ilyen struktúra kialakulása rendkívül fontos, mivel ez az a pont, ahol a további növekedés már nem lehetséges az adott rétegrácsban.

Matematikailag, egy blokkoló áramlásban található minden út a forrástól a nyelőig tartalmaz legalább egy telített élt (vagy kapacitású élt). Ennek következtében az élő együttállás következő szintjére történő bejutás blokkolva van, ami azt jelenti, hogy az augmentáció szempontjából nincs további hasznosítható út.

#### Blokkoló áramlások az algoritmus szintjén

A Dinic algoritmus során az eljárás úgy működik, hogy először BFS-t (szélességi keresést) alkalmazunk egy rétegzett háló készítéséhez, majd DFS-sel (mélységi keresés) létrehozunk egy blokkoló áramlást. Ennek során minden rétegzett háló feldolgozása során egy új blokkoló áramlás jön létre, amíg a globális áramlás el nem éri a maximumot. Az alábbiakban részletesebben leírjuk ezen lépések belső mechanizmusait.

1. **Rétegzett háló generálása (Level Graph)**: Ez történik egy BFS alkalmazása során az eredeti hálóra. Ez a folyamat minden csúcsnak adott szintet rendel hozzá, amely annak távolságát jelzi a forrástól (source-tól). Ezzel szinteket határozunk meg és biztosítjuk, hogy az augmentáció a rétegeken belül található élekkel történik.

2. **Blokkoló áramlás megtalálása**: Miután létrehoztunk egy rétegzett hálót, megkeressük a blokkoló áramlást DFS alkalmazásával. Ezen eljárással követjük a rétegzett háló éleit annak érdekében, hogy megtaláljuk az összes telítési pontot. Amikor egy él telítetté válik, azt eltávolítjuk az éllistából, és így alakítjuk ki a blokkoló áramlást.

#### Példa a blokkoló áramlásra C++ nyelven

Az alábbiakban egy mintakódot mutatok C++ nyelven, amely szemlélteti a fentieket. Ez a kód része egy teljes Dinic algoritmusnak:

```cpp
#include <iostream>

#include <vector>
#include <queue>

#include <climits>
#include <cstring>

using namespace std;

struct Edge {
    int to, rev;
    int flow, cap;
};

class Dinic {
public:
    vector<vector<Edge>> adj;
    vector<int> level, ptr;
    int n, src, dest;

    Dinic(int n, int src, int dest) : n(n), src(src), dest(dest) {
        adj.resize(n);
        level.resize(n);
        ptr.resize(n);
    }

    void add_edge(int u, int v, int cap) {
        Edge a = {v, (int)adj[v].size(), 0, cap};
        Edge b = {u, (int)adj[u].size(), 0, 0}; // reverse edge
        adj[u].push_back(a);
        adj[v].push_back(b);
    }

    bool bfs() {
        fill(level.begin(), level.end(), -1);
        level[src] = 0;
        queue<int> q;
        q.push(src);
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (Edge &e : adj[u]) {
                if (level[e.to] == -1 && e.flow < e.cap) {
                    level[e.to] = level[u] + 1;
                    q.push(e.to);
                }
            }
        }
        return level[dest] != -1;
    }

    int dfs(int u, int pushed) {
        if (pushed == 0) return 0;
        if (u == dest) return pushed;
        for (int &index = ptr[u]; index < adj[u].size(); index++) {
            Edge &e = adj[u][index];
            if (level[e.to] == level[u] + 1 && e.flow < e.cap) {
                int flow = min(pushed, e.cap - e.flow);
                int temp_flow = dfs(e.to, flow);
                if (temp_flow > 0) {
                    e.flow += temp_flow;
                    adj[e.to][e.rev].flow -= temp_flow;
                    return temp_flow;
                }
            }
        }
        return 0;
    }

    int max_flow() {
        int flow = 0;
        while (bfs()) {
            fill(ptr.begin(), ptr.end(), 0);
            while (int pushed = dfs(src, INT_MAX)) {
                flow += pushed;
            }
        }
        return flow;
    }
};

int main() {
    int n = 6;
    int src = 0, dest = 5;
    Dinic dinic(n, src, dest);

    dinic.add_edge(0, 1, 16);
    dinic.add_edge(0, 2, 13);
    dinic.add_edge(1, 2, 10);
    dinic.add_edge(1, 3, 12);
    dinic.add_edge(2, 1, 4);
    dinic.add_edge(2, 4, 14);
    dinic.add_edge(3, 2, 9);
    dinic.add_edge(3, 5, 20);
    dinic.add_edge(4, 3, 7);
    dinic.add_edge(4, 5, 4);

    cout << "Maximum flow: " << dinic.max_flow() << endl;

    return 0;
}
```

#### Magyarázat

Ebben a példában a `Dinic` osztálysablon implementálja a Dinic algoritmus fő lépéseit. Különösen figyelmet kell szentelni a következő módszereknek:

- `bfs()': A rétegzett háló elkészítésére szolgál, szélességi bejárás (BFS) segítségével. Ez az eljárás meghatározza minden csúcs szintjét, és ezt a távolságot tárolja a `level` vektorban.
- `dfs()`: A blokkoló áramlás meghatározására szolgál, mélységi bejárás (DFS) segítségével, és növeli az áramlást, amennyit csak lehet a rétegzett hálóban.
- `max_flow()`: Ez a fő metódus, amely ismételten használja a BFS-t és a DFS-t, hogy addig találjon blokkoló áramlásokat, amíg a teljes áramlást nem tudja tovább növelni.

#### Blokkoltság kezelése és élettel telített élek felismerése

A Dinic algoritmus egyik legnagyobb erénye, hogy képes hatékonyan kezelni a blokkoló áramlásokat. A folyamat során fontos megállapításokat tehetünk blokkoló áramlásokkal kapcsolatban, például:

- Az élő rétegzett hálóban egy él akkor tekinthető telítettnek, ha annak kapacitása egyenlővé válik az átlagított áramlással.
- A blokkoló áramlások dinamikusan változhatnak az áramlás növekedése során, ami azt jelenti, hogy a háló új szintek és telítettségi pontok szerint rendeződhet és alakítható.

Összességében a blokkoló áramlások Dinic algoritmusában történő kezelése kulcsfontosságú szempont a maximális áramlás probléma megoldásában. Azáltal, hogy a jelenlegi állapotot rétegzett hálóval modellezik, és addig keresnek augmentációs utakat, amíg telítődnek, az algoritmus hatékonyan és elegánsan képes nagy méretű hálózatok áramlásának optimalizálására.

