\newpage

# 7.1. Legolcsóbb út algoritmusai

A hálózati algoritmusok a számítógéptudomány és az operációkutatás kulcsfontosságú eszközei, amelyek különböző hálózati problémák hatékony megoldását teszik lehetővé. Ezek közé tartozik a legolcsóbb út megtalálása, amely kritikus szerepet játszik számos alkalmazási területen, beleértve az internetes adatforgalom optimalizálását, a logisztikai útvonalak tervezését vagy éppen a közlekedési hálózatok áteresztőképességének javítását. Ebben a részben megismerkedünk a legolcsóbb út problémájával, amelynek célja megtalálni két csomópont között azt az útvonalat, amely minimalizálja az utazási költséget vagy időt. Kezdve a klasszikus algoritmusokkal, mint a Dijkstra-algoritmus, és tovább haladva az A* keresési módszerig, bemutatjuk ezek működését, hatékonyságát, valamint azt, hogyan adaptálhatók különféle gyakorlati problémák megoldására.

## 7.2.1. Dijkstra algoritmusa

A hálózati algoritmusok egyik legfontosabb területe a legolcsóbb út megtalálása egy adott gráfban. Ezen problémák megoldására számos algoritmust dolgoztak ki, amelyből az egyik legelterjedtebb és legismertebb a Dijkstra algoritmus. Ezt az algoritmust Edsger W. Dijkstra holland számítástechnikus fejlesztette ki 1956-ban, és azóta is alapvető eszköz a legkisebb költségű úttal kapcsolatos problémák megoldására, legyen szó útvonaltervezésről, hálózati optimalizációról vagy más alkalmazási területekről. A következő alfejezetekben részletesen bemutatjuk a Dijkstra algoritmus működési elveit, lépéseit és implementációját, majd összehasonlítjuk más gyakran használt algoritmusokkal, hogy megérthessük, mikor és miért előnyös ennek az algoritmusnak a használata.

### 7.2.1.1. Alapelvek és implementáció

A Dijkstra algoritmus egy meghatározó módszer a legolcsóbb vagy legrövidebb utak keresésére irányított vagy irányítatlan, nem negatív súlyú gráfokban. Edsger W. Dijkstra holland informatikus találta fel 1956-ban, és azóta az egyik leggyakrabban alkalmazott algoritmus a számítástechnika és az operációkutatás területén. Az algoritmus célja, hogy megtalálja a legolcsóbb utat egy forráspontból (source) a gráf többi csúcsába, minimális súlyösszeggel.

#### Alapelvek

A Dijkstra algoritmus alapvetően egy szélességi keresésre hasonlít, de speciálisan úgy van kialakítva, hogy figyelembe vegye a gráf éleihez rendelt súlyokat. A módszer iteratív, és minden lépésben kiválasztja azt a csúcsot, amelynek a legkisebb az "ideiglenes" távolsága a forrásponttól. Az algoritmus két halmazzal dolgozik:

1. **S**: Ez a halmaz tartalmazza azokat a csúcsokat, amelyek végleges távolságát már meghatároztuk a forrásponttól.
2. **Q**: Ez a halmaz tartalmazza az összes többi csúcsot, amelyek távolságát még nem rögzítettük.

Az algoritmus lépései a következőképpen épülnek fel:

1. **Inicializálás**:
    - Minden csúcshoz hozzárendelünk egy kezdő távolságértéket. A forráscsúcs távolsága 0, a többi csúcsé végtelen ($\infty$).
    - A forráscsúcs és képviselője kerül az S halmazba, és a Q halmazba bejegyeztük az összes csúcsot a megfelelő távolságértékkel.

2. **Iteratív lépések**:
    - Kiválasztjuk a Q halmazból azt a csúcsot, amelyiknek a legkisebb a távolsága a forrástól (minimum keresés).
    - Az aktuális csúcsot hozzáadjuk az S halmazhoz, és eltávolítjuk a Q halmazból.
    - Ellenőrizzük az aktuális csúcs szomszédjait (közvetlen élekkel kapcsolódó csúcsokat). Ha elérjük valamelyik szomszédos csúcsot gyorsabban egy új úttal (összekötő él súlyának figyelembevételével), akkor frissítjük annak ideiglenes távolságát.

3. **Be fejezés**:
    - Az iterációk addig folytatódnak, amíg a Q halmaz ki nem üresedik. Az S halmaz tartalmazza az összes csúcs végleges minimum távolságát a forrástól.

#### Formális algoritmus

Vegyük formálisan a Dijkstra algoritmust az alábbiak szerint:

```markdown
Dijkstra(G, w, s):
1. init_single_source(G, s)
2. S = $\emptyset$
3. Q = G.V
4. while Q $\noteq \emptyset$:
5.    u = Extract_Min(Q) # A legkisebb távolságú csúcs kiválasztása
6.    S = S $\cup$ {u}
7.    for each vertex v $\in$ Adj[u]
8.        Relax(u, v, w)
```

- `init_single_source(G, s)`: Inicializálja a távolságokat minden csúcshoz. A forrás csúcshoz (s) 0 és a többiekhez végtelen értéket ad.
- `Extract_Min(Q)`: Visszaadja és eltávolítja azt a csúcsot a Q halmazból, amelynek minimális a távolsága.
- `Relax(u, v, w)`: Frissíti a csúcs távolságát, ha új rövidebb utat találnak.

#### Pseudokód
Az algoritmus egy részletesebb pseudokód formájában is leírható (nem kódszintű implementáció):

```markdown
function Dijkstra(Graph, source):
    // Initializations
    create vertex set Q
    for each vertex v in Graph: 
        dist[v] = INFINITY // Initial distance from source to vertex
        prev[v] = UNDEFINED // Previous node in optimal path from source
        add v to Q // All nodes are added to the queue
    dist[source] = 0 // Distance from source to source
    
    while Q is not empty:
        u = vertex in Q with min dist[u]
        remove u from Q
        
        for each neighbor v of u: // Only v still in Q
            alt = dist[u] + length(u, v)
            if alt < dist[v]: // A shorter path to v has been found
                dist[v] = alt
                prev[v] = u
    
    return dist[], prev[]
```

#### Példakód C++-ban

Az alábbiakban bemutatunk egy részletes implementációt C++ nyelven:

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <utility>
#include <limits>
#define INF std::numeric_limits<int>::max()

using namespace std;

typedef pair<int, int> PII; // Pair of (graph_vertex, weight)

// Dijkstra function
vector<int> dijkstra(int V, vector<vector<PII>> &adj, int source) {
    priority_queue<PII, vector<PII>, greater<PII>> pq;
    vector<int> dist(V, INF);
    
    pq.push(make_pair(0, source));
    dist[source] = 0;

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();

        for (auto &neighbor : adj[u]) {
            int v = neighbor.first;
            int weight = neighbor.second;

            if (dist[v] > dist[u] + weight) {
                dist[v] = dist[u] + weight;
                pq.push(make_pair(dist[v], v));
            }
        }
    }

    return dist;
}

int main() {
    int V = 5;
    vector<vector<PII>> adj(V);
    
    // Add edges (u, v) and weights
    adj[0].emplace_back(1, 10);
    adj[0].emplace_back(4, 5);
    adj[1].emplace_back(2, 1);
    adj[2].emplace_back(3, 4);
    adj[3].emplace_back(0, 7);
    adj[4].emplace_back(1, 3);
    adj[4].emplace_back(2, 9);
    adj[4].emplace_back(3, 2);

    vector<int> distances = dijkstra(V, adj, 0);

    cout << "Vertex distances from source:\n";
    for (int i = 0; i < V; ++i)
        cout << i << " \t " << distances[i] << "\n";

    return 0;
}
```

#### Algoritmus hatékonysága

Az algoritmus hatékonysága nagyban függ a minimum kiválasztás módszerétől. Az alapvető bináris heap (halom) használata esetén az algoritmus időbeli komplexitása $O((E + V)\log V)$, ahol $E$ az élek és $V$ a csúcsok száma. A Fibonacci heap használata esetén az időbeli komplexitás tovább csökkenthető, egészen $O(E + V\log V)$-ig. Általánosságban a memóriakövetelmények $O(V)$ méretűek, amely magába foglalja a csúcsok és a távolságértékek tárolását.

A Dijkstra algoritmus széles körben alkalmazható problémák megoldására, a valós idejű térképnavigációtól kezdve a hálózati útvonal-optimalizációig.

Mindent összevetve, a Dijkstra algoritmus egy kulcsfontosságú eszköz a legolcsóbb út megtalálása érdekében különböző gráfokban és hálózatokban, és a hatékony megvalósítása, valamint az előnyök és korlátok figyelembevétele elengedhetetlen az alkalmazások széles spektrumában.

### 7.2.1.2. Összehasonlítás más algoritmusokkal

A Dijkstra algoritmusa kiemelkedő szerepet játszik a legolcsóbb út megtalálásában a hálózatok elméletében, különösen súlyozott gráfok esetén pozitív élhosszúság mellett. Azonban, annak érdekében, hogy valóban megértsük az algoritmus erősségeit és korlátait, elengedhetetlen, hogy összevessük más jól ismert legolcsóbb út algoritmusokkal, mint például az A* algoritmus, Bellman-Ford algoritmus, Floyd-Warshall algoritmus és Johnson algoritmus.

#### A* Algoritmus

Az A* algoritmus egy heurisztikus keresési algoritmus, amelyet széles körben használnak az útkeresés és a gráfkeresési problémák megoldására. Az A* algoritmus hasonló a Dijkstra-hoz, de további heurisztikus információkat használ a célállapot eléréséhez, amellyel potenciálisan felgyorsíthatja a keresést.

**Összehasonlítás:**
- **Heurisztika**: Az A* alkalmazása során a keresési tér jelentősen csökkenthető egy megfelelő heurisztikus függvény választásával, ami gyorsabb megoldást eredményezhet nagy gráfok esetén. Ezzel szemben a Dijkstra algoritmusa nem használ heurisztikát.
- **Alkalmazhatóság**: Az A*-t elsősorban konkrét célállapotok keresésére használják (pl. GPS navigáció), míg a Dijkstra leginkább arra alkalmas, hogy egy forráspontból minden más pontba megtalálja a legolcsóbb útvonalakat.
- **Optimalitás**: Megfelelő heurisztikus függvény választása esetén az A* algoritmus optimális megoldást ad, míg a Dijkstra mindig optimális eredményt ad megfelelő feltételek mellett.

#### Bellman-Ford Algoritmus

A Bellman-Ford algoritmus szintén egy széles körben használt módszer a legrövidebb utak meghatározására. Ez különösen akkor jön jól, amikor negatív súlyú élek is szerepelhetnek a gráfban.

**Összehasonlítás:**
- **Súlyok**: A Dijkstra algoritmusa csak pozitív súlyú élekkel működik helyesen, addig a Bellman-Ford algoritmus képes kezelni a negatív súlyú éleket is.
- **Komplexitás**: A Bellman-Ford algoritmus időösszetettsége $O(V \cdot E)$, ahol $V$ a csúcsok, $E$ pedig az élek száma. A Dijkstra algoritmusa optimális esetben $O(V \log V)$ vagy $O(E + V \log V)$ (pl. prioritásos sor használatával) időkomplexitással rendelkezik.
- **Rugalmasabb alkalmazás**: A Bellman-Ford algoritmus rugalmasabb a gráfok nehézségeit illetően (pl. élek irányultsága, ciklusok), míg a Dijkstra algoritmus ezen a területen korlátozottabb.

#### Floyd-Warshall Algoritmus

A Floyd-Warshall algoritmus egy másik népszerű megközelítés a legrövidebb utak megtalálására, de különleges a maga nemében, mivel az összes páros csúcs közötti legolcsóbb utak megtalálására használják.

**Összehasonlítás:**
- **Minden páros megoldás**: Míg a Dijkstra algoritmusa egyetlen forráspontból indulva határozza meg a legolcsóbb útvonalakat, addig a Floyd-Warshall az összes csúcs között dolgozik, és mindegyik csúcsból mindegyikbe meghatározza a legrövidebb utat.
- **Komplexitás**: A Floyd-Warshall algoritmus időkomplexitása $O(V^3)$, ami nagyobb gráfoknál nem igazán hatékony. Ezzel szemben a Dijkstra algoritmus több szempontból is skálázhatóbb.
- **Terhelhetőség**: A Floyd-Warshall elsősorban kisebb gráfok esetén praktikus, míg a Dijkstra nagy és sűrű hálózatoknál is jól alkalmazható.

#### Johnson Algoritmus

A Johnson algoritmus egy komplex módszer, amely a Bellman-Ford és a Dijkstra algoritmus kombinációját használja. Ez az algoritmus alkalmas a legrövidebb utak meghatározására olyan helyzetekben, amikor az összes forrás-cél pár közötti utak szükségesek, és az élhosszúk keverten pozitívak és negatívak.

**Összehasonlítás:**

- **Kombinált megközelítés**: Azáltal, hogy ötvözi a Bellman-Ford és Dijkstra algoritmusokat, a Johnson algoritmus képes mindkét világ előnyeit felhasználni: a negatív súlyú élkezelést és a Dijkstra által nyújtott gyors keresést.
- **Komplexitás és skálázhatóság**: A Johnson algoritmus időkomplexitása $O(V^2 \log V + VE)$, így gyakran előnyösebb, ha az összes-pontos legolcsóbb út feladatot kell megoldani nagy gráfok esetén is.
- **Negatív élek kezelése**: Bár hasonlóan a Bellman-Ford-hoz pozitív és negatív súlyokat is kezel, potenciálisan gyorsabb és hatékonyabb lehet, ha gyorsan próbáljuk meghatározni az összes legrövidebb utat.

#### Az algoritmusok gyakorlati alkalmazása és választása

Az, hogy melyik algoritmust használjuk, több tényezőtől is függ, mint például a gráf mérete, sajátosságai (pl. élek súlyai), valamint a problémamegoldási kontextustól. Míg a Dijkstra algoritmusa az egyik legelterjedtebb és leggyakrabban használt algoritmus a legtöbb pozitív súlyú útkeresési problémára, más algoritmusokat érdemes figyelembe venni bonyolultabb esetekben:

1. **Kis gráfok és minden-páros probléma**: Ilyen esetekben a Floyd-Warshall algoritmus lehet a legésszerűbb választás az egyszerűsége és átfedése miatt.
2. **Nagy, vegyes súlyú gráfok és sok forrás-cél pár**: Johnson algoritmusa hatékonyabb lehet az összes-páros legolcsóbb út feladatok megoldásában.
3. **Negatív élek jelenléte**: Bellman-Ford algoritmus a használatos, amikor negatív súlyok is előfordulnak a gráfban.

