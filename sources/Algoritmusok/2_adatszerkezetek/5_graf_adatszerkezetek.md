\newpage

## 5. Irányított és irányítatlan grafok

A grafok az adatszerkezetek egy különösen fontos csoportját alkotják, amelyek számos alkalmazási területen megjelennek, a hálózatok elemzésétől kezdve a mesterséges intelligencia algoritmusokig. Ebben a fejezetben megvizsgáljuk az irányított és irányítatlan grafok alapvető fogalmait, reprezentációit és a rajtuk végezhető műveleteket. A grafok struktúrája csúcsokból (vagy pontokból) és élekből áll, ahol az élek összekapcsolják a csúcsokat, és irányított grafok esetében ezek az élek irányítottak is lehetnek. Az alapfogalmak áttekintését követően különböző reprezentációs módszereket, mint a szomszédsági mátrix és szomszédsági lista, tárgyaljuk. Végül bemutatjuk a grafokon végezhető műveleteket, valamint a különbségeket és alkalmazási területeket az irányított és irányítatlan grafok között. E fejezet célja, hogy átfogó képet adjon a grafokkal kapcsolatos alapvető ismeretekről és eszközökről, amelyek elengedhetetlenek a komplex adatszerkezetek megértéséhez és kezeléséhez.

## 5.1. Alapfogalmak (csúcsok, élek, fokszámok)

A grafelmélet az egyik legfontosabb és legszélesebb körben alkalmazott területe a diszkrét matematikának és az informatikának. Grafok segítségével modellezhetünk hálózatokat, útvonalakat, kapcsolatrendszereket és számos egyéb strukturált adatot. Ebben az alfejezetben részletesen bemutatjuk a grafok alapfogalmait, mint a csúcsok, élek és fokszámok, valamint ezek matematikai és algoritmikus jelentőségét.

### Csúcsok (Vertices vagy Nodes)

A grafok legfontosabb elemei a csúcsok, amelyeket gyakran pontoknak is nevezünk. Egy graf $G = (V, E)$ definíciójában $V$ a csúcsok halmazát jelöli. A csúcsok lehetnek egyének egy szociális hálózatban, városok egy útvonalhálózatban vagy állapotok egy állapotgépben. Az egyes csúcsokat általában egyedi azonosítóval látják el, például számokkal vagy betűkkel.

#### Formális definíció

Legyen $G = (V, E)$ egy graf, ahol $V$ a csúcsok halmaza. Ha a graf $n$ csúcsot tartalmaz, akkor $|V| = n$.

#### Példa

Vegyünk egy egyszerű grafot $G$, amelyben a csúcsok halmaza $V = \{A, B, C, D\}$. Ebben az esetben négy csúcsunk van, és $|V| = 4$.

### Élek (Edges)

Az élek a csúcsok közötti kapcsolatokat reprezentálják. Az élek halmazát $E$ jelöli. Az él lehet irányítatlan (amikor a kapcsolat kétirányú) vagy irányított (amikor a kapcsolat egyirányú).

#### Irányítatlan élek

Irányítatlan graf esetén az éleket rendezetlen csúcspárok alkotják, azaz az él bármelyik irányban járható. Ha $u$ és $v$ csúcsok között irányítatlan él van, akkor ezt $\{u, v\}$-vel jelöljük.

#### Irányított élek

Irányított grafban az élek rendezett párokként szerepelnek, azaz az él csak egy irányban járható. Ha $u$-ból $v$-be van él, akkor ezt $(u, v)$-vel jelöljük.

#### Formális definíció

Legyen $G = (V, E)$ egy graf. Az élek halmaza $E$ tartalmazhat rendezetlen (irányítatlan graf) vagy rendezett (irányított graf) csúcspárokat. Ha a graf $m$ élt tartalmaz, akkor $|E| = m$.

#### Példa

Az előző példában a csúcsok halmaza $V = \{A, B, C, D\}$. Tegyük fel, hogy az élek halmaza $E = \{\{A, B\}, \{B, C\}, \{C, D\}, \{D, A\}\}$ egy irányítatlan graf esetén. Ebben az esetben négy élünk van, és $|E| = 4$.

### Fokszámok (Degrees)

A fokszám egy csúcsra érkező vagy abból kiinduló élek számát jelenti. Az irányítatlan grafokban egy csúcs fokszáma az összes rá kapcsolódó él száma. Az irányított grafokban megkülönböztetjük a bejövő fokszámot (in-degree) és a kimenő fokszámot (out-degree).

#### Irányítatlan graf fokszáma

Egy irányítatlan grafban egy $v$ csúcs fokszámát $deg(v)$-vel jelöljük, és az összes $v$-re érkező él számát jelenti.

$deg(v) = \text{csúcsra érkező élek száma}$

#### Irányított graf fokszámai

Irányított grafban egy $v$ csúcs bejövő fokszámát $deg^-(v)$-vel, kimenő fokszámát pedig $deg^+(v)$-vel jelöljük.

$deg^-(v) = \text{csúcsra bejövő élek száma}$
$deg^+(v) = \text{csúcsból kiinduló élek száma}$

#### Példa

Tegyük fel, hogy van egy irányítatlan grafunk az alábbi élekkel:

$E = \{\{A, B\}, \{A, C\}, \{B, C\}, \{C, D\}\}$

Ebben az esetben a fokszámok a következőképpen alakulnak:

$deg(A) = 2, deg(B) = 2, deg(C) = 3, deg(D) = 1$

### Matematikai és algoritmikus jelentőség

A csúcsok, élek és fokszámok alapvető fontosságúak a grafelméleti problémák és algoritmusok szempontjából. A gráfok elemzésének és feldolgozásának számos módszere ezen alapfogalmakra épül. Például a szélességi és mélységi keresés (BFS, DFS) algoritmusok az élek és csúcsok bejárásával működnek, míg a fokszámok elemzése segíthet az erősen összefüggő komponensek vagy a hálózat legfontosabb csomópontjainak azonosításában.

#### Szélességi keresés (BFS) C++ példakód

A szélességi keresés egy algoritmus, amely a graf csúcsait rétegezett módon járja be.

```cpp
#include <iostream>

#include <vector>
#include <queue>

using namespace std;

void BFS(int start, vector<vector<int>>& adjList) {
    vector<bool> visited(adjList.size(), false);
    queue<int> q;
    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        cout << node << " ";

        for (int neighbor : adjList[node]) {
            if (!visited[neighbor]) {
                q.push(neighbor);
                visited[neighbor] = true;
            }
        }
    }
}

int main() {
    int n = 5; // number of vertices
    vector<vector<int>> adjList(n);

    adjList[0] = {1, 2};
    adjList[1] = {0, 3, 4};
    adjList[2] = {0};
    adjList[3] = {1};
    adjList[4] = {1};

    cout << "BFS starting from vertex 0: ";
    BFS(0, adjList);

    return 0;
}
```

Ez a kód egy egyszerű példája a szélességi keresés algoritmusának egy irányítatlan grafban, ahol a szomszédsági listával ábrázolt gráfot járjuk be.

### Alkalmazási területek

A grafok és azok alapfogalmai számos alkalmazási területen jelennek meg. A közlekedési hálózatokban a csúcsok városokat vagy útkereszteződéseket, míg az élek az utakat vagy útvonalakat reprezentálják. A szociális hálózatokban a csúcsok egyéneket, az élek pedig a kapcsolataikat jelképezik. Az interneten a weboldalakat és a közöttük lévő hiperlinkeket is grafként lehet modellezni. Ezen kívül a biológiai hálózatokban, például a génhálózatokban és a fehérje-fehérje interakciós hálózatokban is hasonló graf modelleket alkalmaznak.

Összefoglalva, a csúcsok, élek és fokszámok megértése elengedhetetlen a grafok hatékony modellezéséhez és elemzéséhez, valamint számos gyakorlati alkalmazás alapját képezik.

## 5.2. Reprezentációk: szomszédsági mátrix, szomszédsági lista

A grafok különböző reprezentációs módszerei lehetővé teszik számunkra, hogy hatékonyan tároljuk és manipuláljuk a gráf adatait. Két alapvető módszer a gráfok reprezentálására a szomszédsági mátrix és a szomszédsági lista. Ezek a reprezentációk különböző előnyökkel és hátrányokkal rendelkeznek, és az adott probléma természetétől függően egyik vagy másik lehet előnyösebb. Ebben az alfejezetben részletesen bemutatjuk mindkét módszert, megvizsgálva azok szerkezetét, előnyeit, hátrányait és felhasználási eseteit.

### 5.2.1. Szomszédsági mátrix

A szomszédsági mátrix egy kétdimenziós mátrix, amelyben a gráf minden csúcspárjához tartozik egy érték, amely jelzi, hogy van-e él a két csúcs között, és ha igen, akkor annak súlyát.

#### 5.2.1.1. Szomszédsági mátrix szerkezete

A szomszédsági mátrix $A$ egy $n \times n$ méretű mátrix, ahol $n$ a gráf csúcsainak száma. Az $A[i][j]$ elem a következőt jelenti:
- $A[i][j] = 0$: nincs él az $i$ és $j$ csúcs között.
- $A[i][j] = w$: van egy él az $i$ és $j$ csúcs között, súlya $w$.

Az irányítatlan gráfok esetében a mátrix szimmetrikus, azaz $A[i][j] = A[j][i]$. Irányított gráfok esetében ez nem feltétlenül igaz.

#### 5.2.1.2. Szomszédsági mátrix előnyei és hátrányai

**Előnyök:**
- **Egyszerűség**: A szomszédsági mátrix könnyen érthető és egyszerűen megvalósítható.
- **Gyors hozzáférés**: Az él jelenlétének ellenőrzése $O(1)$ időben történik.
- **Egyszerű műveletek**: Az olyan műveletek, mint a mátrix szorzás (pl. legrövidebb út keresése Floyd-Warshall algoritmussal), egyszerűen végezhetők el.

**Hátrányok:**
- **Memóriaigény**: A mátrix $O(n^2)$ memóriát igényel, ami nagy csúcsszám esetén jelentős lehet, különösen ritka (sparse) gráfok esetén.
- **Élek száma**: Mivel minden lehetséges csúcspárhoz tartozik egy érték, ritka gráfok esetén sok fölösleges helyet foglal el.

#### 5.2.1.3. Szomszédsági mátrix felhasználási esetei

- **Teljes gráfok**: Olyan gráfok esetén, ahol sok él van, a szomszédsági mátrix hatékonyan használható.
- **Gyors hozzáférés igénye**: Olyan algoritmusoknál, ahol gyakran kell ellenőrizni az élek jelenlétét (pl. Floyd-Warshall algoritmus).

**C++ implementáció:**

```cpp
#include <iostream>

#include <vector>

class Graph {
public:
    Graph(int n) : adjMatrix(n, std::vector<int>(n, 0)) {}

    void addEdge(int u, int v, int w) {
        adjMatrix[u][v] = w;
        adjMatrix[v][u] = w; // Irányítatlan gráf esetén
    }

    void print() {
        for (const auto& row : adjMatrix) {
            for (int val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

private:
    std::vector<std::vector<int>> adjMatrix;
};

int main() {
    Graph g(5);
    g.addEdge(0, 1, 1);
    g.addEdge(0, 2, 1);
    g.addEdge(1, 2, 1);
    g.addEdge(1, 3, 1);
    g.addEdge(2, 4, 1);

    g.print();
    return 0;
}
```

### 5.2.2. Szomszédsági lista

A szomszédsági lista egy olyan adatstruktúra, ahol minden csúcshoz egy lista tartozik, amely tartalmazza az adott csúcshoz közvetlenül kapcsolódó csúcsokat és az élek súlyát.

#### 5.2.2.1. Szomszédsági lista szerkezete

A szomszédsági lista egy vektor vagy lista, ahol minden elem egy lista vagy vektor, amely az adott csúcshoz kapcsolódó éleket tárolja. Az élek súlyát is tárolhatjuk, ha súlyozott gráffal van dolgunk.

#### 5.2.2.2. Szomszédsági lista előnyei és hátrányai

**Előnyök:**
- **Memóriahatékonyság**: A szomszédsági lista csak a valóban létező éleket tárolja, így ritka gráfok esetén memóriahatékonyabb.
- **Hatékony iteráció**: Könnyen iterálhatunk egy adott csúcs szomszédjain, ami sok algoritmusnál hasznos.

**Hátrányok:**
- **Él jelenlétének ellenőrzése**: Az él jelenlétének ellenőrzése nem $O(1)$ időben történik, hanem a csúcs szomszédainak számától függően $O(d)$ időben, ahol $d$ a csúcs fokszáma.
- **Összetettebb szerkezet**: A szomszédsági lista implementációja bonyolultabb lehet, mint a szomszédsági mátrixé.

#### 5.2.2.3. Szomszédsági lista felhasználási esetei

- **Ritka gráfok**: Olyan gráfok esetén, ahol kevés él van a csúcsokhoz képest, a szomszédsági lista hatékonyabb.
- **Algoritmusok, ahol fontos a szomszédok gyors elérése**: Olyan algoritmusoknál, ahol gyakran kell iterálni egy csúcs szomszédjain, mint például a DFS vagy a BFS.

**C++ implementáció:**

```cpp
#include <iostream>

#include <vector>
#include <list>

class Graph {
public:
    Graph(int n) : adjList(n) {}

    void addEdge(int u, int v, int w) {
        adjList[u].emplace_back(v, w);
        adjList[v].emplace_back(u, w); // Irányítatlan gráf esetén
    }

    void print() {
        for (int i = 0; i < adjList.size(); ++i) {
            std::cout << i << ": ";
            for (const auto& pair : adjList[i]) {
                std::cout << "(" << pair.first << ", " << pair.second << ") ";
            }
            std::cout << std::endl;
        }
    }

private:
    std::vector<std::list<std::pair<int, int>>> adjList;
};

int main() {
    Graph g(5);
    g.addEdge(0, 1, 1);
    g.addEdge(0, 2, 1);
    g.addEdge(1, 2, 1);
    g.addEdge(1, 3, 1);
    g.addEdge(2, 4, 1);

    g.print();
    return 0;
}
```

### 5.2.3. Szomszédsági mátrix vs. szomszédsági lista: Összehasonlítás és alkalmazások

Mindkét reprezentációs módszernek megvannak a maga előnyei és hátrányai, és az adott probléma jellegétől függ, hogy melyik a jobb választás. Az alábbiakban összehasonlítjuk a két módszert különböző szempontok alapján.

#### 5.2.3.1. Memóriafelhasználás

- **Szomszédsági mátrix**: $O(n^2)$ memória, ahol $n$ a csúcsok száma. Teljes gráfok esetén hatékony, de ritka gráfoknál sok felesleges helyet foglal.
- **Szomszédsági lista**: $O(n + m)$ memória, ahol $m$ az élek száma. Ritka gráfok esetén sokkal hatékonyabb.

#### 5.2.3.2. Műveletek időbeli összetettsége

- **Él jelenlétének ellenőrzése**:
    - Szomszédsági mátrix: $O(1)$
    - Szomszédsági lista: $O(d)$, ahol $d$ a csúcs fokszáma.
- **Szomszédok bejárása**:
    - Szomszédsági mátrix: $O(n)$
    - Szomszédsági lista: $O(d)$, ahol $d$ a csúcs fokszáma.
- **Élek hozzáadása**:
    - Szomszédsági mátrix: $O(1)$
    - Szomszédsági lista: $O(1)$

#### 5.2.3.3. Alkalmazási területek

- **Szomszédsági mátrix**:
    - Olyan algoritmusok, amelyek gyakran ellenőrzik az élek jelenlétét.
    - Teljes vagy közel teljes gráfok.
    - Kis méretű gráfok, ahol a memóriahasználat nem kritikus.
- **Szomszédsági lista**:
    - Ritka gráfok.
    - Olyan algoritmusok, amelyek gyakran iterálnak a szomszédokon.
    - Nagy méretű gráfok, ahol a memóriahatékonyság fontos.

### 5.2.4. További megfontolások

A gráf reprezentációjának megválasztása előtt érdemes megfontolni a konkrét feladat és az alkalmazott algoritmusok sajátosságait. Néhány további szempont:

- **Algoritmus kompatibilitás**: Egyes algoritmusok természetüknél fogva jobban működnek egyik vagy másik reprezentációval. Például a Floyd-Warshall algoritmus a szomszédsági mátrixot használja, míg a DFS és BFS hatékonyabb lehet szomszédsági listával.
- **Dinamikus gráfok**: Olyan helyzetekben, ahol a gráf gyakran változik (élek hozzáadása vagy eltávolítása), a szomszédsági lista általában jobb választás, mert rugalmasabb a szerkezete.
- **Adatstruktúra kiterjesztése**: Komplexebb gráfok esetén, ahol további adatokat (pl. csúcsok tulajdonságai, élek attribútumai) kell tárolni, a szomszédsági lista jobban kiterjeszthető egyedi igények szerint.

### 5.2.5. Összegzés

A szomszédsági mátrix és a szomszédsági lista mindkettő hasznos eszközök a gráfok reprezentálására, de különböző körülmények között különböző előnyökkel és hátrányokkal rendelkeznek. A megfelelő reprezentáció megválasztása kritikus a gráf-alapú algoritmusok hatékony működéséhez, és a döntést az adott feladat sajátosságainak és a graf struktúrájának figyelembevételével kell meghozni. Az alfejezet célja, hogy alapos áttekintést nyújtson ezen reprezentációkról, segítve az olvasót a megfelelő döntés meghozatalában az adott problémára vonatkozóan.

## 5.3. Műveletek grafokon

A grafokon végezhető műveletek széles skálája lehetőséget ad különböző problémák megoldására a számítástechnikában és más tudományterületeken. Ebben az alfejezetben részletesen áttekintjük a legfontosabb műveleteket, amelyek magukban foglalják a gráf bejárását, keresési algoritmusokat, legrövidebb út meghatározását, összefüggő komponensek megtalálását, és minimális feszítőfák képzését.

### 5.3.1. Gráf bejárása

A gráf bejárása során minden csúcsot és élt pontosan egyszer látogatunk meg. Két alapvető bejárási módszer létezik: a szélességi keresés (Breadth-First Search, BFS) és a mélységi keresés (Depth-First Search, DFS).

#### 5.3.1.1. Szélességi keresés (BFS)

A BFS egy réteges megközelítést alkalmaz, ahol először az aktuális csúcshoz közvetlenül kapcsolódó csúcsokat látogatjuk meg, majd ezek szomszédait, és így tovább. Ez az algoritmus kiválóan alkalmas a legrövidebb út keresésére nem súlyozott gráfokban.

**Algoritmus lépései:**
1. Helyezzük a kezdőcsúcsot egy sorba.
2. Amíg a sor nem üres:
    - Vegyük ki a sor elejét, és látogassuk meg a csúcsot.
    - Az aktuális csúcs összes szomszédját, amelyet még nem látogattunk meg, helyezzük a sor végére.

**C++ implementáció:**

```cpp
#include <iostream>

#include <vector>
#include <queue>

void BFS(int start, const std::vector<std::vector<int>>& adjList) {
    std::vector<bool> visited(adjList.size(), false);
    std::queue<int> q;

    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        std::cout << node << " ";

        for (int neighbor : adjList[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}

int main() {
    std::vector<std::vector<int>> adjList = {
        {1, 2},
        {0, 3, 4},
        {0, 4},
        {1, 5},
        {1, 2, 5},
        {3, 4}
    };

    BFS(0, adjList);

    return 0;
}
```

#### 5.3.1.2. Mélységi keresés (DFS)

A DFS egy rekurzív vagy verem alapú megközelítést alkalmaz, ahol mindig a legmélyebb, még nem látogatott csúcsot látogatjuk meg először. Az algoritmus alkalmas az összefüggő komponensek és a topológiai sorrend meghatározására.

**Algoritmus lépései:**
1. Indítsuk a keresést a kezdőcsúcsról.
2. Látogassuk meg az aktuális csúcsot, és jelöljük meg látogatottnak.
3. Rekurzívan látogassuk meg az összes szomszédot, amelyet még nem látogattunk meg.

**C++ implementáció:**

```cpp
#include <iostream>

#include <vector>

void DFSUtil(int node, const std::vector<std::vector<int>>& adjList, std::vector<bool>& visited) {
    visited[node] = true;
    std::cout << node << " ";

    for (int neighbor : adjList[node]) {
        if (!visited[neighbor]) {
            DFSUtil(neighbor, adjList, visited);
        }
    }
}

void DFS(int start, const std::vector<std::vector<int>>& adjList) {
    std::vector<bool> visited(adjList.size(), false);
    DFSUtil(start, adjList, visited);
}

int main() {
    std::vector<std::vector<int>> adjList = {
        {1, 2},
        {0, 3, 4},
        {0, 4},
        {1, 5},
        {1, 2, 5},
        {3, 4}
    };

    DFS(0, adjList);

    return 0;
}
```

### 5.3.2. Legrövidebb út keresése

A legrövidebb út keresése a grafokban gyakran előforduló probléma, különösen a hálózatok és a térképek esetén. Az alábbi algoritmusok a leghíresebbek és leggyakrabban használtak.

#### 5.3.2.1. Dijkstra algoritmus

A Dijkstra algoritmus egy hatékony módszer a legrövidebb út megtalálására súlyozott, irányított vagy irányítatlan gráfokban, ahol az élek súlya nem lehet negatív.

**Algoritmus lépései:**
1. Inicializáljuk a kezdőcsúcs távolságát 0-ra, a többi csúcs távolságát végtelenre.
2. Helyezzük a kezdőcsúcsot egy prioritási sorba.
3. Amíg a prioritási sor nem üres:
    - Vegyük ki a legkisebb távolságú csúcsot a sorból.
    - Frissítsük az aktuális csúcs szomszédainak távolságát.
    - Helyezzük a szomszédokat a prioritási sorba, ha azok távolsága frissült.

**C++ implementáció:**

```cpp
#include <iostream>

#include <vector>
#include <queue>

#include <utility>

#define INF 1e9

void Dijkstra(int start, const std::vector<std::vector<std::pair<int, int>>>& adjList) {
    std::vector<int> distances(adjList.size(), INF);
    distances[start] = 0;

    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>> pq;
    pq.push({0, start});

    while (!pq.empty()) {
        int distance = pq.top().first;
        int node = pq.top().second;
        pq.pop();

        if (distance > distances[node])
            continue;

        for (const auto& neighbor : adjList[node]) {
            int nextNode = neighbor.first;
            int weight = neighbor.second;

            if (distances[node] + weight < distances[nextNode]) {
                distances[nextNode] = distances[node] + weight;
                pq.push({distances[nextNode], nextNode});
            }
        }
    }

    for (int i = 0; i < distances.size(); ++i) {
        std::cout << "Distance to node " << i << " is " << distances[i] << std::endl;
    }
}

int main() {
    std::vector<std::vector<std::pair<int, int>>> adjList = {
        {{1, 2}, {2, 4}},
        {{2, 1}, {3, 7}},
        {{4, 3}},
        {{5, 1}},
        {{3, 2}, {5, 5}},
        {}
    };

    Dijkstra(0, adjList);

    return 0;
}
```

### 5.3.3. Összefüggő komponensek keresése

Egy gráf összefüggő komponenseinek keresése segít meghatározni a gráf azon részeit, amelyek belsőleg összefüggőek, de egymástól elválasztottak. Ez különösen hasznos a hálózatok elemzésénél.

#### 5.3.3.1. Mélységi keresés alapú módszer

Az összefüggő komponensek meghatározására a DFS-t használhatjuk, minden csúcsról indítva egy új keresést, ha az még nem volt látogatott.

**Algoritmus lépései:**
1. Inicializáljunk egy látogatott vektor.
2. Iteráljunk végig a csúcsokon.
3. Ha egy csúcs még nem volt látogatott, indítsunk egy DFS-t, és jegyezzük fel a komponenseket.

**C++ implementáció:**

```cpp
#include <iostream>

#include <vector>

void DFSComponent(int node, const std::vector<std::vector<int>>& adjList, std::vector<bool>& visited) {
    visited[node] = true;
    for (int neighbor : adjList[node]) {
        if (!visited[neighbor]) {
            DFSComponent(neighbor, adjList, visited);
        }
    }
}

int findConnectedComponents(const std::vector<std::vector<int>>& adjList) {
    std::vector<bool> visited(adjList.size(), false);
    int components = 0;

    for (int i = 0; i < adjList.size(); ++i) {
        if (!visited[i]) {
            DFSComponent(i, adjList, visited);
            components++;
        }
    }

    return components;
}

int main() {
    std::vector<std::vector<int>> adjList = {
        {1, 2},
        {0, 3},
        {0},
        {1},
        {5},
        {4}
    };

    int components = findConnectedComponents(adjList);
    std::cout << "Number of connected components: " << components << std::endl;

    return 0;
}
```

### 5.3.4. Minimális feszítőfa keresése

A minimális feszítőfa (Minimum Spanning Tree, MST) egy összefüggő, irányítatlan gráf egy részgráfja, amely összeköti az összes csúcsot a lehető legkisebb súlyú élekkel.

#### 5.3.4.1. Kruskal algoritmus

A Kruskal algoritmus egy olyan módszer, amely először rendezi az összes élt növekvő sorrendben, majd kiválasztja a legkisebb éleket, amelyek nem alkotnak kört, amíg az összes csúcs összekapcsolódik.

**Algoritmus lépései:**
1. Rendezzük az éleket súly szerint növekvő sorrendbe.
2. Inicializáljunk egy unió-find adatstruktúrát.
3. Iteráljunk végig az éleken, és adjuk hozzá az élt a feszítőfához, ha nem alkot kört.

**C++ implementáció:**

```cpp
#include <iostream>

#include <vector>
#include <algorithm>

struct Edge {
    int src, dest, weight;
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

class UnionFind {
public:
    UnionFind(int n) : parent(n), rank(n, 0) {
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
            } else if (rank[rootU] < rank[rootV]) {
                parent[rootU] = rootV;
            } else {
                parent[rootV] = rootU;
                rank[rootU]++;
            }
        }
    }

private:
    std::vector<int> parent, rank;
};

void KruskalMST(const std::vector<Edge>& edges, int numVertices) {
    std::vector<Edge> result;
    UnionFind uf(numVertices);

    std::vector<Edge> sortedEdges = edges;
    std::sort(sortedEdges.begin(), sortedEdges.end());

    for (const auto& edge : sortedEdges) {
        if (uf.find(edge.src) != uf.find(edge.dest)) {
            result.push_back(edge);
            uf.unite(edge.src, edge.dest);
        }
    }

    for (const auto& edge : result) {
        std::cout << edge.src << " -- " << edge.dest << " == " << edge.weight << std::endl;
    }
}

int main() {
    std::vector<Edge> edges = {
        {0, 1, 4}, {0, 2, 4}, {1, 2, 2}, {1, 3, 5},
        {2, 3, 5}, {2, 4, 1}, {3, 4, 7}
    };
    int numVertices = 5;

    KruskalMST(edges, numVertices);

    return 0;
}
```

### 5.3.5. Összegzés

Ebben az alfejezetben részletesen megvizsgáltuk a grafokon végezhető legfontosabb műveleteket. A bejárási algoritmusok (BFS és DFS) alapvető építőkövei a grafokkal kapcsolatos problémák megoldásának. A legrövidebb út keresésére szolgáló algoritmusok, mint a Dijkstra, kritikusak az útvonaltervezésben és a hálózati optimalizálásban. Az összefüggő komponensek keresése és a minimális feszítőfák meghatározása alapvető fontosságúak a gráfelmélet alkalmazásában. A bemutatott C++ kódpéldák segítenek a gyakorlati megvalósítás megértésében és a grafok hatékony kezelése.

## 5.4. Különbségek és alkalmazások

A grafok elmélete és gyakorlati felhasználása számos különböző területet ölel fel a számítástechnikában, matematikában, biológiában, közlekedésben és még sok más területen. Ebben az alfejezetben részletesen megvizsgáljuk a különbségeket a grafok különböző típusai és azok reprezentációi között, valamint áttekintjük, hogy ezek a különbségek hogyan befolyásolják az alkalmazási lehetőségeket. Külön figyelmet fordítunk a gyakorlati példákra és alkalmazási esetekre, hogy megértsük, mikor és miért érdemes egy adott graf reprezentációt vagy algoritmust választani.

### 5.4.1. Különbségek a grafok típusai között

A grafok különböző típusainak megértése alapvető fontosságú ahhoz, hogy megfelelően alkalmazzuk őket különböző problémák megoldására. Az alábbiakban áttekintjük a legfontosabb graf típusokat és azok jellemzőit.

#### 5.4.1.1. Irányított és irányítatlan grafok

- **Irányított grafok (Directed Graphs, Digraphs)**: Az élek irányítottak, azaz egy él egy kezdőpontból (forrás) egy végpontba (cél) vezet. Például egy weboldalak közötti hivatkozásokat leíró graf irányított, mivel a linkek egy irányba mutatnak.

- **Irányítatlan grafok (Undirected Graphs)**: Az élek irányítatlanok, azaz az él mindkét végpontjába lehet navigálni. Például egy közúthálózat gráfja irányítatlan, mivel az utak mindkét irányba járhatók.

#### 5.4.1.2. Súlyozott és súlyozatlan grafok

- **Súlyozott grafok (Weighted Graphs)**: Az élekhez súlyok tartoznak, amelyek általában az élek költségét, hosszát vagy kapacitását reprezentálják. Például egy közlekedési hálózatban az élek súlyai lehetnek az utak hossza vagy az utazási idő.

- **Súlyozatlan grafok (Unweighted Graphs)**: Az éleknek nincs súlyuk. Például egy szociális hálózat gráfjában az élek egyszerűen azt jelzik, hogy két személy ismerős.

#### 5.4.1.3. Ritka és sűrű grafok

- **Ritka grafok (Sparse Graphs)**: Olyan grafok, ahol az élek száma jóval kevesebb, mint a maximálisan lehetséges élek száma, azaz $m \ll n^2$, ahol $n$ a csúcsok száma és $m$ az élek száma. Például egy város metróhálózata gyakran ritka graf.

- **Sűrű grafok (Dense Graphs)**: Olyan grafok, ahol az élek száma közel van a maximálisan lehetséges élek számához, azaz $m \approx n^2$. Például egy teljes gráf, ahol minden csúcs össze van kötve minden más csúccsal, sűrű graf.

### 5.4.2. Különbségek a graf reprezentációk között

Mint korábban említettük, a két leggyakrabban használt graf reprezentáció a szomszédsági mátrix és a szomszédsági lista. Az alábbiakban részletesen összehasonlítjuk őket különböző szempontok alapján.

#### 5.4.2.1. Memóriahasználat

- **Szomszédsági mátrix**: $O(n^2)$ memóriát igényel, ahol $n$ a csúcsok száma. Ez akkor is igaz, ha a graf ritka, ami sok fölösleges helyet eredményezhet.

- **Szomszédsági lista**: $O(n + m)$ memóriát igényel, ahol $m$ az élek száma. Ez memóriahatékonyabb ritka grafok esetén.

#### 5.4.2.2. Műveletek hatékonysága

- **Élek jelenlétének ellenőrzése**:
    - Szomszédsági mátrix: $O(1)$ időben történik, mivel közvetlen hozzáférést biztosít.
    - Szomszédsági lista: $O(d)$ időben, ahol $d$ a csúcs fokszáma, mivel végig kell iterálni a szomszédokon.

- **Szomszédok bejárása**:
    - Szomszédsági mátrix: $O(n)$ időben, mivel végig kell iterálni az összes csúcson.
    - Szomszédsági lista: $O(d)$ időben, ahol $d$ a csúcs fokszáma, mivel csak a szomszédokon kell iterálni.

#### 5.4.2.3. Műveletek bonyolultsága

- **Szomszédsági mátrix**: Egyszerűbb adatstruktúra, könnyen érthető és implementálható.
- **Szomszédsági lista**: Bonyolultabb adatstruktúra, de rugalmasabb és memóriahatékonyabb ritka grafok esetén.

### 5.4.3. Gyakorlati alkalmazások és példák

A különböző graf típusok és reprezentációk számos gyakorlati alkalmazási területtel rendelkeznek, amelyek különböző problémák megoldására használhatók. Az alábbiakban néhány gyakorlati példát és alkalmazási területet mutatunk be.

#### 5.4.3.1. Közlekedési hálózatok

A közlekedési hálózatok, mint például az úthálózatok, vasúthálózatok és repülőjáratok hálózata, gyakran súlyozott gráfokként modellezhetők, ahol az élek súlya az utazási időt, távolságot vagy költséget jelenti.

**Alkalmazás**:
- **Útvonaltervezés**: Dijkstra algoritmus vagy A* keresés használható a legrövidebb út megtalálására két pont között.
- **Minimális feszítőfa**: Kruskal vagy Prim algoritmusok használhatók a hálózat kiépítésének minimalizálására.

#### 5.4.3.2. Szociális hálózatok

A szociális hálózatok, mint például a Facebook vagy Twitter, gyakran irányítatlan, súlyozatlan grafokként modellezhetők, ahol a csúcsok a felhasználókat, az élek pedig a baráti kapcsolatokat jelképezik.

**Alkalmazás**:
- **Közösségfelismerés**: Algoritmusok, mint a Girvan-Newman vagy Louvain módszer, segítenek a közösségek azonosításában a hálózatban.
- **Hatásvizsgálat**: A PageRank algoritmus használható a befolyásos felhasználók azonosítására.

#### 5.4.3.3. Biológiai hálózatok

A biológiai hálózatok, mint például a fehérje-fehérje interakciós hálózatok vagy a génregulációs hálózatok, gyakran irányított, súlyozott grafokként modellezhetők.

**Alkalmazás**:
- **Pathway analízis**: Algoritmusok, mint a DFS vagy BFS, használhatók a biokémiai útvonalak azonosítására.
- **Hálózati modulok felismerése**: Közösségfelismerő algoritmusok segíthetnek azonosítani a hálózat funkcionális moduljait.

#### 5.4.3.4. Telekommunikációs hálózatok

A telekommunikációs hálózatok, mint például az internethálózatok, gyakran irányított, súlyozott grafokként modellezhetők, ahol az élek súlya az átviteli kapacitást vagy késleltetést jelzi.

**Alkalmazás**:
- **Forrás-irányítás**: Algoritmusok, mint a Bellman-Ford vagy Dijkstra, használhatók az optimális adatforrás-irányítás meghatározására.
- **Hibakeresés**: A hálózat összefüggő komponenseinek azonosítása segíthet a hibák lokalizálásában.

### 5.4.4. Különbségek az algoritmusok között

A különböző algoritmusok különböző típusú grafok és problémák megoldására specializálódtak. Az alábbiakban néhány fontos algoritmust és azok alkalmazási területeit ismertetjük.

#### 5.4.4.1. Keresési algoritmusok

- **Breadth-First Search (BFS)**: Hatékonyan használható legrövidebb út keresésére nem súlyozott gráfokban, például labirintus megoldására.
- **Depth-First Search (DFS)**: Kiválóan alkalmas összefüggő komponensek és körök felismerésére.

#### 5.4.4.2. Legrövidebb út keresése

- **Dijkstra algoritmus**: Súlyozott gráfokban használható legrövidebb út keresésére, ahol az élek súlya nem negatív.
- **Bellman-Ford algoritmus**: Alkalmazható negatív súlyú élekkel rendelkező gráfokban is, bár lassabb, mint a Dijkstra algoritmus.

#### 5.4.4.3. Minimális feszítőfa

- **Kruskal algoritmus**: Olyan esetekben hasznos, ahol az élek súlyai előre ismertek és könnyen rendezhetők.
- **Prim algoritmus**: Olyan esetekben előnyös, ahol egy csúcsból kiindulva szeretnénk felépíteni a feszítőfát.

### 5.4.5. Különbségek a teljesítményben és komplexitásban

A különböző algoritmusok és reprezentációk teljesítménye és bonyolultsága nagyban függ a gráf típusától és a konkrét problémától. Az alábbiakban néhány általános szempontot ismertetünk.

#### 5.4.5.1. Időbeli komplexitás

- **Szomszédsági mátrix**: Általában gyorsabb lehet az él jelenlétének ellenőrzésében ($O(1)$), de lassabb a szomszédok bejárásában ($O(n)$).
- **Szomszédsági lista**: Gyorsabb lehet a szomszédok bejárásában ($O(d)$), de lassabb az él jelenlétének ellenőrzésében ($O(d)$).

#### 5.4.5.2. Térbeli komplexitás

- **Szomszédsági mátrix**: Memóriaigényesebb lehet ritka gráfok esetén ($O(n^2)$).
- **Szomszédsági lista**: Memóriahatékonyabb ritka gráfok esetén ($O(n + m)$).

#### 5.4.5.3. Skálázhatóság

- **Szomszédsági mátrix**: Nagy csúcsszám esetén skálázódási problémákkal szembesülhet.
- **Szomszédsági lista**: Jobban skálázható nagy csúcsszámú és ritka gráfok esetén.

### 5.4.6. Összegzés

A grafok reprezentációjának és algoritmusainak megválasztása kritikus fontosságú a különböző problémák hatékony megoldásához. A szomszédsági mátrix és a szomszédsági lista mindegyike saját előnyökkel és hátrányokkal rendelkezik, amelyek különböző alkalmazási területeken és graf típusok esetén más-más előnyöket kínálnak. A keresési algoritmusok, legrövidebb út keresési algoritmusok és minimális feszítőfa algoritmusok mindegyike specifikus problémákra optimalizált, és a megfelelő algoritmus kiválasztása kulcsfontosságú a hatékony megoldáshoz.

Az alfejezet célja az volt, hogy átfogó és részletes képet nyújtson a grafok különböző típusairól, reprezentációiról, algoritmusairól és azok alkalmazási területeiről, segítve az olvasót abban, hogy a megfelelő eszközöket válassza a konkrét feladat megoldására.
