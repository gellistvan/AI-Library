\newpage

## 7.3.4. Push-Relabel algoritmus

A hálózati áramlások optimalizálásának egyik hatékony és gyakran használt módszere a Push-Relabel algoritmus. Ez az algoritmus a hagyományos augmentáló útvonal algoritmusoktól eltérő megközelítést alkalmaz a maximális áramlás kiszámítására a hálózatban. Míg az augmentáló utak keresésére alapozó algoritmusok lineáris utakat követve növelik az áramlást, addig a Push-Relabel algoritmus lokálisan próbálja optimalizálni az áramlási értékeket, egyfajta "nyomás" (push) és "szintbeállítás" (relabel) műveleteken keresztül. E megközelítés rugalmasságának és hatékonyságának köszönhetően, a Push-Relabel algoritmus különösen hasznos nagy méretű és összetett hálózatokban. A következő alfejezetekben részletesen bemutatjuk az algoritmus alapelveit és implementációját, valamint megvizsgáljuk annak teljesítményét és gyakorlati alkalmazásait is.

### 7.3.4.1. Alapelvek és implementáció

A Push-Relabel algoritmus a hálózati áramlás optimalizációs algoritmusok közé tartozik, amelyet a maximális áramlás problémák megoldására használunk. Ez az algoritmus más néven Preflow-Push algoritmusként is ismert, és egy alternatív megközelítést kínál a Ford-Fulkerson és Edmonds-Karp algoritmusokkal szemben. A Push-Relabel algoritmus alapelve a lokális optimalizáció, melyben a csúcsok magasságának (height) és az elő-túlcsordulás (preflow/excess flow) fogalmakat használja annak érdekében, hogy hatékonyan találja meg a maximális áramlást egy hálózaton keresztül.

#### Alapelvek

A Push-Relabel algoritmus több kulcsfogalomra épül:

1. **Magasság (Height):** Minden csúcsnak van egy magassága, amely irányítja, hogy az áramlás mely irányba mozoghat. A forrás (source) csúcs kezdeti magassága a csúcsok számával egyenlő, míg minden más csúcs magassága kezdetben 0.

2. **Felesleges áramlás (Excess Flow):** Ez az érték határozza meg, hogy mennyi áramlás van jelenleg egy csúcsban, amely még nincs továbbítva a következő csúcs(ok)ba. Az algoritmus az áramlást az élek mentén push műveletekkel mozgatja, hogy csökkentse ezt a felesleget.

3. **Push művelet:** Ha egy csúcsban túlcsordulás van, és van szomszédos csúcs, amely alacsonyabb magasságon van, akkor az áramlás bizonyos része átkerülhet ehhez a szomszédos csúcshoz, amíg el nem éri az él kapacitási korlátját, vagy a túlcsordulás el nem fogy.

4. **Relabel művelet:** Ha egy csúcsban túlcsordulás van, de az összes szomszédos csúcs magasabban van, akkor megnövelhetjük a csúcs magasságát a szomszédos csúcsok magasságának maximuma plusz egy értékre. Ez lehetővé teszi, hogy a jövőben a push műveletek során áramlást továbbítsunk.

#### Implementáció

Az algoritmus az alábbi fő lépésekből áll:

1. **Inicializáció:** Kezdetben a forrás csúcsból minden lehetséges áramlást kipusholunk a szomszédos csúcsokhoz, azaz beállítjuk az elő-túlcsordulási áramlást. A magasságok megfelelően inicializálódnak: a forrás csúcs magassága a csúcsok száma, az összes többi csúcsé pedig 0.

2. **Iteráció:** Míg van olyan csúcs, ami túlcsordulással rendelkezik (azaz a felesleges áramlása nagyobb mint 0), alkalmazzuk a push vagy relabel műveleteket ezekre a csúcsokra.

3. **Push művelet:** Ha találunk egy élt egy **u** csúcsból egy **v** csúcsba, ahol **height[u] > height[v]** és **c(u, v) > f(u, v)** (azaz a fennmaradó kapacitás pozitív), akkor áramlást pusholunk **u**-ból **v**-be.

4. **Relabel művelet:** Ha nem tudunk több push műveletet végrehajtani **u**-nál, akkor növeljük **u** magasságát.

Ezek a lépések mindaddig ismétlődnek, amíg nincs több túltöltött csúcs (kivéve a forrás és a nyelő csúcsokat). Az algoritmus garantáltan befejeződik, és a nyelő csúcsba beérkezett teljes áramlás megadja a maximális áramlást.

#### Pseudokód és implementáció C++ nyelven

Az alábbiakban bemutatjuk a Push-Relabel algoritmus pseudokódját és egy esetleges C++ nyelvű implementációját.

**Pseudokód:**

```pseudo
procedure Push-Relabel(Graph, s, t)
    InitializePreflow(Graph, s)
    while there exists an overflowing vertex v in V do
        if there is an admissible edge (v, u) then
            Push(v, u)
        else
            Relabel(v)
```

Procedúra részei:

```pseudo
procedure InitializePreflow(Graph, s)
    for each vertex v in Graph do
        height[v] = 0
        excess[v] = 0
    height[s] = |V|
    for each vertex v adjacent to s do
        flow(s, v) = capacity(s, v)
        flow(v, s) = -flow(s, v)
        excess[v] = flow(s, v)
        excess[s] -= flow(s, v)

procedure Push(u, v)
    delta = min(excess[u], capacity(u, v) - flow(u, v))
    flow(u, v) += delta
    flow(v, u) -= delta
    excess[u] -= delta
    excess[v] += delta

procedure Relabel(v)
    minHeight = inf
    for each vertex u adjacent to v do
        if capacity(v, u) - flow(v, u) > 0 then
            minHeight = min(minHeight, height[u])
    height[v] = minHeight + 1
```

**C++ Implementáció:**

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <climits>

class PushRelabel {
    struct Edge {
        int from, to, capacity, flow;
        Edge(int from, int to, int capacity) : from(from), to(to), capacity(capacity), flow(0) {}
    };

    int n;
    std::vector<std::vector<int>> adj;
    std::vector<Edge> edges;
    std::vector<int> height, excess;
    std::vector<int> current;

public:
    PushRelabel(int n): n(n), adj(n), height(n), excess(n), current(n) {}

    void addEdge(int u, int v, int capacity) {
        edges.emplace_back(u, v, capacity);
        adj[u].push_back(edges.size() - 1);
        edges.emplace_back(v, u, 0);
        adj[v].push_back(edges.size() - 1);
    }

    void push(int u, int idx) {
        Edge& e = edges[idx];
        int flow = std::min(excess[u], e.capacity - e.flow);
        e.flow += flow;
        edges[idx ^ 1].flow -= flow;
        excess[u] -= flow;
        excess[e.to] += flow;
    }

    void relabel(int u) {
        int min_height = INT_MAX;
        for (int i : adj[u]) {
            Edge& e = edges[i];
            if (e.capacity > e.flow)
                min_height = std::min(min_height, height[e.to]);
        }
        if (min_height < INT_MAX)
            height[u] = min_height + 1;
    }

    void discharge(int u) {
        while (excess[u] > 0) {
            if (current[u] == adj[u].size()) {
                relabel(u);
                current[u] = 0;
            }
            else {
                int idx = adj[u][current[u]];
                Edge& e = edges[idx];
                if (e.capacity > e.flow && height[u] == height[e.to] + 1) {
                    push(u, idx);
                }
                else {
                    ++current[u];
                }
            }
        }
    }

    int getMaxFlow(int s, int t) {
        height[s] = n;
        excess[s] = INT_MAX;
        for (int i : adj[s]) {
            Edge& e = edges[i];
            push(s, i);
        }

        std::queue<int> q;
        for (int i = 0; i < n; ++i) {
            if (i != s && i != t && excess[i] > 0)
                q.push(i);
        }

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            discharge(u);
            for (int i : adj[u]) {
                Edge& e = edges[i];
                if (e.to != s && e.to != t && excess[e.to] > 0)
                    q.push(e.to);
            }
        }

        int max_flow = 0;
        for (int i : adj[s]) {
            Edge& e = edges[i];
            max_flow += e.flow;
        }

        return max_flow;
    }
};

int main() {
    int n = 6;
    PushRelabel pr(n);

    pr.addEdge(0, 1, 10);
    pr.addEdge(0, 2, 10);
    pr.addEdge(1, 2, 2);
    pr.addEdge(1, 3, 4);
    pr.addEdge(1, 4, 8);
    pr.addEdge(2, 4, 9);
    pr.addEdge(3, 5, 10);
    pr.addEdge(4, 3, 6);
    pr.addEdge(4, 5, 10);

    std::cout << "Maximum Flow: " << pr.getMaxFlow(0, 5) << std::endl;

    return 0;
}
```

#### Magyarázat:

1. **Inicializáció és Adatstruktúrák:** Az implementációban az éleket (Edge) egy szerkezettel reprezentáljuk, amely tartalmazza a forrást (`from`), a célt (`to`), a kapacitást (`capacity`), és a jelenlegi áramlást (`flow`). Az `adj` vektor minden csúcshoz tárol egy listát a hozzá kapcsolódó élek indexeivel. Az `excess` és `height` vektorok az egyes csúcsok felesleges áramlását és magasságát tárolják.

2. **Push művelet:** A push művelet során a `flow` értéket növeljük az élben, csökkentjük az ellenkező irányba folyó áramlást, és frissítjük az `excess` értékeket.

3. **Relabel művelet:** A relabel művelet növeli a magasságot annak érdekében, hogy lehetővé tegye a további push műveleteket. A magasság növelése a szomszédos csúcsok height értéke alapján történik.

4. **Discharge művelet:** A discharge művelet sorozatosan push és/vagy relabel műveleteket hajt végre, amíg a csúcsban van felesleges áramlás.

5. **Maximum Flow Számítás:** A `getMaxFlow` függvény végzi az algoritmust, amelyben először a forrásból kipusholjuk a lehetséges áramlást, majd minden túltöltött csúcsot `discharge` művelettel kezelünk, végül visszatérünk a nyelő csúcsba beérkezett áramlással, amely a maximális áramlást reprezentálja.

Egy hálózati áramlás maximumának meghatározása kritikus fontosságú sok gyakorlati probléma megoldása során, és a Push-Relabel algoritmus hatékony eszközt kínál ezen problémák kezelésére. A részletes implementáció bemutatása érdekében a C++ nyelvi környezetben biztosítjuk, hogy a megoldás nemcsak elméleti, hanem gyakorlati szempontból is megérthető és alkalmazható legyen.

### 7.3.4.2. Teljesítmény elemzés és alkalmazások

A Push-Relabel algoritmus egy hatékony megoldási módszer a maximális áramlás problémájának kezelésére az áramlási hálózatokban. Az algoritmus alapelvei és implementációja részletesen tárgyalva voltak az előző szakaszban, most az algoritmus teljesítményét elemezzük és bemutatjuk annak különféle alkalmazási területeit.

#### Teljesítmény Elemzés

A Push-Relabel algoritmus teljesítményének értékelése több szempontból is kritikus fontosságú. Az algoritmus számos módosított változatát vizsgálták és fejlesztették, hogy különböző hálózati környezetekben optimális teljesítményt nyújtson. Az alábbiakban részletesen megvizsgáljuk az algoritmus legfontosabb teljesítményjellemzőit.

1. **Időkomplexitás**:

   Az időkomplexitás az algoritmus futtatása során végrehajtott műveletek számát jelzi a bemenet méretének függvényében. A Push-Relabel algoritmus legrosszabb esetben elérheti az $O(V^2 E)$ komplexitást, ahol $V$ a csúcspontok száma és $E$ az élek száma a hálózatban.

   Egyes finomított változatok, mint a FIFO Push-Relabel algoritmus, hatékonyabb végrehajtást nyújtanak, mivel csökkentik a felesleges műveletek számát. Az Admissible Push-Relabel algoritmus pedig az $O(V^3)$ komplexitást éri el, mely a csúcspontok számának kockás függvénye.

2. **Térkomplexitás**:

   Az algoritmus térbeli komplexitása az algoritmus futtatása során szükséges memória mennyiségét jelöli. A Push-Relabel algoritmus térkomplexitása $O(V + E)$, mivel minden csúcs és él esetében tároljuk a kapacitásokat, áramlásokat, és a magasságokat.

3. **Preflow**:

   A preflow alapvetően különbözik a hagyományos áramlástól, mivel megengedi a műveletek közti ideiglenes többletáramlást. Ennek a többletnek a kezelése az algoritmus alapvető feladatai közé tartozik, és ennek hatékony feldolgozása jelentősen befolyásolja az algoritmus teljesítményét.

4. **Skálázhatóság**:

   A Push-Relabel algoritmus skálázhatósága azt jelzi, hogy mennyire képes hatékonyan kezelni a különböző méretű hálózatokat. Az algoritmus jól skálázható közepes méretű hálózatok esetében, mivel képes gyorsan beállítani az áramlást minden egyes szinten.

5. **Heurisztikák és optimalizációk**:

   Számos heurisztika és optimalizáció létezik, amelyek javítják az algoritmus teljesítményét, például a Global Relabeling heurisztika, amely időnként frissíti az összes csúcs magasságát az aktuális állapot szerint. Ez gyakran drasztikusan csökkenti a szükséges műveletek számát.

#### Alkalmazások

Az áramlási hálózatok és a Push-Relabel algoritmus számos gyakorlati alkalmazási területtel rendelkezik, az alábbiakban az egyik legfontosabbakat részletezzük.

1. **Távközlés és Hálózati Optimalizáció**:

   Az áramlási hálózatok optimalizálásának egyik fő területe a távközlési hálózatok esetében van. Az adatcsatornák és kapacitások hatékony kezelése létfontosságú a hálózat teljesítményének és megbízhatóságának biztosítása érdekében. A Push-Relabel algoritmust számos routing és bandwidth allocation probléma megoldására használják.

2. **Szállítási és Logisztikai Hálózatok**:

   A maximális áramlás problémák szállítási és logisztikai alkalmazásokban is megjelennek. Például a Push-Relabel algoritmus használható a legnagyobb árukapacitású útvonalak kiszámítására, vagy a közlekedési rendszerek hatékonyabb koordinálására.

3. **Képfeldolgozás**:

   Képfeldolgozási feladatokban a Push-Relabel algoritmus alkalmazható például az optimális kettéosztás keresésére a képpontok halmazában, ami hatékony képsegmentációval segíthet.

4. **GPU és Parallel Computation**:

   Az algoritmus természetéből adódóan jól párhuzamosítható, amely előnyt jelent a GPU-alapú feldolgozás során. Az előretolás és címkézés műveletei egyszerre több csúcs esetében is feldolgozhatók, amely jelentősen felgyorsítja a feldolgozási időt nagyléptékű hálózatok esetében.

5. **Ellenőrzés és Változáskövetés**:

   A dinamikus hálózatok, melyek állandóan változó kapacitásokkal és csúcsokkal rendelkeznek, szintén profitálhatnak a Push-Relabel algoritmus alkalmazásából. Az algoritmus rugalmassága lehetővé teszi, hogy gyorsan adaptálódjon a változó környezeti feltételekhez, biztosítva ezzel az optimális áramlást.

A fentiek alapján jól látható, hogy a Push-Relabel algoritmus egy rendkívül sokoldalú és hatékony módszer, amely számos különféle alkalmazási területet képes lefedni. A maximális teljesítmény elérése érdekében azonban nagyon fontos a megfelelő heurisztikák és frissítési stratégiák alkalmazása, valamint az algoritmus finomhangolása az adott probléma specifikus igényeihez igazodva.


