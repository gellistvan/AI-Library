\newpage

# 4. Mohó algoritmusok

A mohó algoritmusok olyan megközelítések, amelyek az optimalizálási problémák megoldására szolgálnak azáltal, hogy minden lépésben a lokálisan legjobb választást teszik meg annak reményében, hogy ez elvezet a globálisan optimális megoldáshoz. Ezek az algoritmusok egyszerűen implementálhatók és gyakran hatékonyak, azonban nem minden esetben garantálják a legjobb megoldást. Ebben a részben áttekintjük a mohó algoritmusok alapelveit, vizsgálunk néhány klasszikus példát, és megismerkedünk azokkal a körülményekkel, amikor ez a stratégia sikeresen alkalmazható. Megértjük, hogyan működnek a mohó algoritmusok, és milyen feltételek mellett biztosítanak optimális megoldást, valamint megvizsgáljuk néhány tipikus alkalmazási területüket is.

## 4.1. Alapelvek

Ebben a fejezetben a mohó algoritmusok alapelveit és működésének lényegét tárgyaljuk. A mohó stratégia egy olyan megközelítés, amely minden lépésben a jelenlegi legjobb döntést hozza meg, anélkül hogy visszalépne korábbi választásokhoz. Elsőként a mohó stratégia lényegét és a mohó választás tulajdonságát ismertetjük, majd megvizsgáljuk az optimális alstruktúra fogalmát, amely kulcsfontosságú a mohó algoritmusok működésében. Végezetül, a mohó algoritmusok elemzésére és bizonyítására összpontosítunk, bemutatva, hogyan igazolható ezeknek az algoritmusoknak a helyessége és hatékonysága különböző problémák esetén. Ezek az alapelvek szilárd alapot nyújtanak a mohó algoritmusok megértéséhez és alkalmazásához a gyakorlatban.


### 4.1.1. Mohó stratégia lényege

A mohó stratégia egy olyan algoritmikus megközelítés, amely optimalizálási problémák megoldására szolgál azáltal, hogy minden lépésben a jelenleg legjobb, vagyis a lokálisan optimális választást teszi meg. A mohó algoritmusok egyszerűségük és hatékonyságuk miatt széles körben alkalmazhatók, azonban nem minden esetben garantálják a globálisan optimális megoldást. Az alábbiakban részletesen ismertetjük a mohó stratégia lényegét, valamint annak főbb komponenseit és tulajdonságait.

#### A mohó algoritmusok alapvető jellemzői

1. **Lokálisan optimális választás**: A mohó algoritmusok minden lépésben a legjobb azonnali választást teszik meg. Ez a választás nem feltétlenül veszi figyelembe a későbbi lépéseket vagy a globális optimális megoldást, csak a jelenlegi állapotban a legjobb döntést hozza.

2. **Nem visszalépő természet**: A mohó algoritmusok egyirányúak, ami azt jelenti, hogy miután egy lépést megtettek, nem térnek vissza a korábbi állapotokhoz, hogy módosítsák a választásukat. Ez különbözik a dinamikus programozástól vagy a visszalépéses kereséstől, amelyek gyakran visszatérnek és újraértékelik a döntéseket.

3. **Hatékonyság**: A mohó algoritmusok általában gyorsak és hatékonyak, mivel minden lépésben egy egyszerű döntést hoznak, ami gyakran lineáris vagy polinomiális időben végrehajtható. Ez különösen előnyös nagy méretű problémák esetén, ahol más megközelítések túl lassúak vagy erőforrás-igényesek lennének.

#### Példák a mohó stratégiára

A mohó stratégia számos klasszikus problémában alkalmazható, mint például a minimális feszítőfa probléma, a Dijkstra algoritmus a legrövidebb út keresésére, a hátizsák probléma bizonyos változatai, és az intervallum kiválasztási probléma. Az alábbiakban részletesen bemutatunk néhányat ezek közül.

##### Minimális feszítőfa

A minimális feszítőfa problémában egy súlyozott gráf összes csúcsát összekötő legkisebb összsúlyú feszítőfát kell megtalálni. Két jól ismert mohó algoritmus létezik erre a problémára: a Kruskal és a Prim algoritmus.

- **Kruskal algoritmus**: A Kruskal algoritmus a gráf éleit súly szerint növekvő sorrendben rendezi, majd addig adja hozzá az éleket a feszítőfához, amíg az összes csúcs össze nem kapcsolódik, ügyelve arra, hogy ne keletkezzen ciklus.

  ```cpp
  #include <vector>
  #include <algorithm>

  struct Edge {
      int u, v, weight;
      bool operator<(Edge const& other) {
          return weight < other.weight;
      }
  };

  int find_set(int v, std::vector<int>& parent) {
      if (v == parent[v])
          return v;
      return parent[v] = find_set(parent[v], parent);
  }

  void union_sets(int a, int b, std::vector<int>& parent, std::vector<int>& rank) {
      a = find_set(a, parent);
      b = find_set(b, parent);
      if (a != b) {
          if (rank[a] < rank[b])
              std::swap(a, b);
          parent[b] = a;
          if (rank[a] == rank[b])
              rank[a]++;
      }
  }

  int main() {
      int n;
      std::vector<Edge> edges;
      
      // Initialize edges and number of vertices

      std::sort(edges.begin(), edges.end());

      std::vector<int> parent(n);
      std::vector<int> rank(n, 0);
      for (int i = 0; i < n; i++)
          parent[i] = i;

      int cost = 0;
      std::vector<Edge> result;
      for (Edge e : edges) {
          if (find_set(e.u, parent) != find_set(e.v, parent)) {
              cost += e.weight;
              result.push_back(e);
              union_sets(e.u, e.v, parent, rank);
          }
      }

      // Output the result
  }
  ```

- **Prim algoritmus**: A Prim algoritmus egy kezdő csúcsból indul ki, és fokozatosan bővíti a feszítőfát úgy, hogy mindig a legkisebb súlyú, még nem kapcsolódó élt választja.

  ```cpp
  #include <vector>
  #include <queue>

  const int INF = 1e9;

  struct Edge {
      int to, weight;
  };

  int main() {
      int n;
      std::vector<std::vector<Edge>> adj;

      // Initialize adjacency list and number of vertices

      int total_weight = 0;
      std::vector<int> min_edge(n, INF);
      std::vector<bool> selected(n, false);
      min_edge[0] = 0;

      std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> pq;
      pq.push({0, 0});

      for (int i = 0; i < n; i++) {
          if (pq.empty()) {
              // The graph is not connected
              return 1;
          }
          int v = pq.top().second;
          pq.pop();
          selected[v] = true;
          total_weight += min_edge[v];

          for (Edge e : adj[v]) {
              if (!selected[e.to] && e.weight < min_edge[e.to]) {
                  min_edge[e.to] = e.weight;
                  pq.push({min_edge[e.to], e.to});
              }
          }
      }

      // Output the total weight of the MST
  }
  ```

##### Dijkstra algoritmus

A Dijkstra algoritmus a legrövidebb utat keresi egy csúcsból a gráf összes többi csúcsába. A mohó stratégia itt abban nyilvánul meg, hogy mindig a legközelebbi, még nem feldolgozott csúcsot választja ki.

```cpp
#include <vector>

#include <queue>

const int INF = 1e9;

struct Edge {
    int to, weight;
};

int main() {
    int n, start;
    std::vector<std::vector<Edge>> adj;

    // Initialize adjacency list, number of vertices, and starting vertex

    std::vector<int> dist(n, INF);
    dist[start] = 0;
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> pq;
    pq.push({0, start});

    while (!pq.empty()) {
        int v = pq.top().second;
        int d_v = pq.top().first;
        pq.pop();

        if (d_v != dist[v])
            continue;

        for (Edge edge : adj[v]) {
            int to = edge.to;
            int len = edge.weight;

            if (dist[v] + len < dist[to]) {
                dist[to] = dist[v] + len;
                pq.push({dist[to], to});
            }
        }
    }

    // Output the shortest distances from start to all vertices
}
```

#### Mohó választás tulajdonsága

A mohó algoritmusok sikerességének egyik kulcsfontosságú tényezője a mohó választás tulajdonsága. Ez a tulajdonság azt jelenti, hogy mindig lehetséges olyan döntést hozni, amely a lokálisan legjobb választást jelenti, és ez a döntés egy optimális megoldás része. Más szóval, a probléma optimális megoldása érdekében nem szükséges a globális szempontokat figyelembe venni minden egyes lépésben.

#### Optimális alstruktúra

Az optimális alstruktúra tulajdonság azt jelenti, hogy egy probléma optimális megoldása tartalmazza az alproblémák optimális megoldásait. A mohó algoritmusok esetében ez a tulajdonság biztosítja, hogy a lokálisan optimális választásokból összeálló megoldás globálisan is optimális lesz, amennyiben a mohó választás tulajdonsága teljesül.

#### Mohó algoritmusok elemzése és bizonyítása

A mohó algoritmusok elemzése és bizonyítása általában a következő lépéseket foglalja magában:

1. **A mohó választás tulajdonságának bizonyítása**: Bemutatjuk, hogy mindig létezik olyan lokálisan optimális választás, amely része egy globálisan optimális megoldásnak.

2. **Az optimális alstruktúra tulajdonságának bizonyítása**: Megmutatjuk, hogy a probléma optimális megoldása tartalmazza az alproblémák optimális megoldásait.

3. **Algoritmus leírása és elemzése**: Részletezzük az algoritmus lépéseit és elemzést végzünk az idő- és térbeli bonyolultság szempontjából.

4. **Helyesség bizonyítása**: Bizonyítjuk, hogy az algoritmus mindig helyes megoldást ad, amennyiben a mohó választás és az optimális alstruktúra tulajdonsága teljesül.

Összegzésként a mohó algoritmusok hatékony és egyszerű módszert kínálnak számos optimalizálási probléma megoldására. Bár nem minden esetben garantálják a globálisan optimális megoldást, számos gyakorlati alkalmazásban sikerrel használhatók, különösen akkor, ha a mohó választás és az optimális alstruktúra tulajdonsága teljesül.

### 4.1.2. Mohó választás tulajdonsága

A mohó választás tulajdonsága egy kritikus eleme a mohó algoritmusok működésének és sikerességének. Ez a tulajdonság azt írja le, hogy egy probléma optimális megoldása mindig tartalmaz egy olyan részmegoldást, amely a lokálisan optimális választások sorozatából áll. Ebben az alfejezetben részletesen megvizsgáljuk ezt a tulajdonságot, bemutatva annak elméleti alapjait, gyakorlati példáit, és hogyan bizonyítható egy adott probléma esetében.

#### A mohó választás tulajdonságának elméleti alapjai

A mohó választás tulajdonsága azt jelenti, hogy az adott probléma optimális megoldása elérhető úgy, hogy minden lépésben a lokálisan legjobb választást tesszük meg. Ez a tulajdonság lehetővé teszi, hogy a probléma megoldásához szükséges döntéseket egyszerűsítsük, és a probléma méretét lépésről lépésre csökkentsük.

A mohó algoritmusoknak két kulcsfontosságú tulajdonságuk van, amelyek a helyességüket és hatékonyságukat biztosítják:

1. **Lokálisan optimális választás**: Minden lépésben olyan döntést hozunk, amely a jelenlegi állapotban a legjobb, anélkül, hogy figyelembe vennénk a jövőbeli következményeket. Ez a döntés biztosítja, hogy az aktuális lépés optimális legyen.
2. **Optimális alstruktúra**: Az optimális megoldás tartalmazza az alproblémák optimális megoldásait. Ez azt jelenti, hogy a probléma nagyobb megoldásait úgy építjük fel, hogy kisebb, optimális részmegoldásokat kombinálunk.

#### Példák a mohó választás tulajdonságára

A mohó választás tulajdonságának megértéséhez nézzünk meg néhány konkrét példát, ahol ez a tulajdonság jól alkalmazható.

##### 1. Minimális feszítőfa probléma (Kruskal és Prim algoritmus)

A minimális feszítőfa (Minimum Spanning Tree, MST) problémában egy súlyozott gráf összes csúcsát úgy kell összekapcsolni, hogy a felhasznált élek összsúlya minimális legyen. A Kruskal és Prim algoritmusok egyaránt alkalmazzák a mohó választás tulajdonságát.

- **Kruskal algoritmus**: Az algoritmus először sorba rendezi az éleket súly szerint, majd minden lépésben hozzáadja a legkisebb súlyú élt a feszítőfához, ha az nem hoz létre ciklust. A mohó választás itt a legkisebb súlyú, még nem felhasznált él kiválasztása.
- **Prim algoritmus**: Az algoritmus egy kezdő csúcsból indul ki, és minden lépésben hozzáadja a feszítőfához a legkisebb súlyú élt, amely összekapcsol egy már kiválasztott csúcsot egy még nem kiválasztott csúccsal. A mohó választás itt a legkisebb súlyú él kiválasztása a feszítőfa bővítéséhez.

##### 2. Dijkstra algoritmus

A Dijkstra algoritmus a legrövidebb út keresésére szolgál egy csúcsból a gráf összes többi csúcsába. Az algoritmus minden lépésben a legközelebbi, még nem feldolgozott csúcsot választja ki, és frissíti a hozzá vezető legrövidebb utakat. A mohó választás itt a legkisebb távolságú csúcs kiválasztása a feldolgozáshoz.

##### 3. Hátizsák probléma (Knapsack Problem)

A hátizsák probléma egy optimalizálási probléma, ahol egy adott kapacitású hátizsákba különböző tárgyakat kell úgy bepakolni, hogy a tárgyak összértéke maximális legyen. A probléma egy változatában, a törtrészes hátizsák problémában (Fractional Knapsack Problem), a mohó algoritmus úgy működik, hogy mindig a legnagyobb érték-súly arányú tárgyat választja ki, és amennyire lehet, bepakolja a hátizsákba.

#### A mohó választás tulajdonságának bizonyítása

Annak bizonyítása, hogy egy adott probléma rendelkezik a mohó választás tulajdonságával, általában két lépést igényel:

1. **Lokális optimalitás bizonyítása**: Bizonyítanunk kell, hogy minden lépésben a lokálisan optimális választás hozzájárul az optimális megoldáshoz. Ez azt jelenti, hogy a választás nem csak egy adott lépésre jó, hanem része lehet egy globálisan optimális megoldásnak is.

2. **Globális optimalitás bizonyítása**: Azt is be kell bizonyítani, hogy a lokálisan optimális választások sorozata végül egy globálisan optimális megoldáshoz vezet. Ezt gyakran matematikai indukcióval vagy kontradikcióval lehet elérni.

#### Példa a mohó választás tulajdonságának bizonyítására: A Kruskal algoritmus

A Kruskal algoritmus esetében bizonyítható, hogy minden lépésben a legkisebb súlyú élt kiválasztva, amely nem hoz létre ciklust, végül egy minimális feszítőfához vezet. A bizonyítás lépései a következők:

1. **Lokális optimalitás**: Tegyük fel, hogy az algoritmus egy lépésében a legkisebb súlyú élt választjuk ki. Ha ez az él nem hoz létre ciklust, akkor az hozzáadása egy részleges feszítőfa bővítését jelenti. Mivel ez az él a legkisebb súlyú, ez a választás lokálisan optimális.

2. **Globális optimalitás**: Tegyük fel, hogy létezik egy optimális feszítőfa, amely nem tartalmazza az algoritmus által kiválasztott legkisebb súlyú élt. Ebben az esetben cseréljük ki az optimális feszítőfában egy nagyobb súlyú élt az algoritmus által kiválasztott éllel. Az így kapott új feszítőfa kisebb súlyú lenne, ami ellentmondás. Ez bizonyítja, hogy az algoritmus által kiválasztott élek részei egy minimális feszítőfának.

#### Összegzés

A mohó választás tulajdonsága az egyik legfontosabb kritérium a mohó algoritmusok helyességének és hatékonyságának biztosításához. Ennek a tulajdonságnak a megléte lehetővé teszi, hogy a probléma megoldásához vezető lépéseket egyszerűsítsük és optimalizáljuk. Bár nem minden probléma rendelkezik a mohó választás tulajdonságával, számos gyakorlati probléma esetében ez a megközelítés hatékony és elegáns megoldásokat eredményez.

### 4.1.3. Optimális alstruktúra

Az optimális alstruktúra tulajdonság az algoritmusok tervezésének és elemzésének egyik alapvető fogalma, különösen a dinamikus programozás és a mohó algoritmusok esetében. Ez a tulajdonság azt jelenti, hogy egy probléma optimális megoldása tartalmazza az alproblémák optimális megoldásait. Az optimális alstruktúra megléte elengedhetetlen ahhoz, hogy egy probléma hatékonyan megoldható legyen mohó algoritmusokkal vagy dinamikus programozással. Ebben a fejezetben részletesen megvizsgáljuk az optimális alstruktúra tulajdonságát, példákkal illusztrálva annak fontosságát és alkalmazását.

#### Az optimális alstruktúra definíciója

Az optimális alstruktúra tulajdonság formálisan azt jelenti, hogy egy probléma optimális megoldásának részei az alproblémák optimális megoldásai. Másképpen megfogalmazva, ha egy probléma kisebb alproblémákra bontható, akkor ezen alproblémák optimális megoldásaiból állítható össze az eredeti probléma optimális megoldása. Ez a tulajdonság különösen fontos a rekurzív algoritmusok tervezésénél, ahol az eredeti problémát kisebb részekre bontjuk, és ezekre alkalmazzuk ugyanazt az algoritmikus megközelítést.

#### Példák az optimális alstruktúrára

Az optimális alstruktúra tulajdonsága számos klasszikus probléma megoldásánál jelen van. Az alábbiakban néhány ilyen példát vizsgálunk meg részletesen.

##### Legkisebb súlyú út (Shortest Path Problem)

A legrövidebb út problémája egy súlyozott gráfban azt a feladatot jelenti, hogy megtaláljuk a legkisebb összsúlyú utat két csúcs között. A Dijkstra algoritmus és a Bellman-Ford algoritmus is kihasználja az optimális alstruktúra tulajdonságát.

- **Dijkstra algoritmus**: Az algoritmus minden lépésben a legkisebb súlyú, még nem feldolgozott csúcsot választja, és frissíti az ahhoz csatlakozó csúcsok távolságát. Az optimális alstruktúra itt abban nyilvánul meg, hogy a csúcsok közötti legrövidebb utak részei más csúcsok közötti legrövidebb utaknak.

- **Bellman-Ford algoritmus**: Ez az algoritmus iteratívan frissíti a csúcsok közötti távolságokat, figyelembe véve az összes élt. Az optimális alstruktúra tulajdonság itt is jelen van, mivel a részútvonalak legrövidebb távolságai befolyásolják az egész útvonal legrövidebb távolságát.

##### Dinamikus programozás és a hátizsák probléma (Knapsack Problem)

A dinamikus programozás egy másik olyan terület, ahol az optimális alstruktúra tulajdonsága kritikus szerepet játszik. A hátizsák probléma egyik legismertebb példája ennek az alkalmazásnak.

- **0/1 hátizsák probléma**: Ebben a problémában adott egy hátizsák, amelynek véges kapacitása van, és egy sor tárgy, amelyek különböző súlyúak és értékűek. A cél az, hogy a hátizsákba úgy válogassuk be a tárgyakat, hogy azok összsúlya ne haladja meg a kapacitást, és az összérték maximális legyen. Az optimális alstruktúra tulajdonság itt azt jelenti, hogy a hátizsák problémára adott optimális megoldás tartalmazza az alproblémák (azaz kisebb kapacitású hátizsákok és kevesebb tárgy) optimális megoldásait.

  ```cpp
  #include <vector>
  #include <algorithm>

  int knapsack(int W, const std::vector<int>& weights, const std::vector<int>& values) {
      int n = weights.size();
      std::vector<std::vector<int>> dp(n + 1, std::vector<int>(W + 1, 0));

      for (int i = 1; i <= n; ++i) {
          for (int w = 0; w <= W; ++w) {
              if (weights[i-1] <= w) {
                  dp[i][w] = std::max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1]);
              } else {
                  dp[i][w] = dp[i-1][w];
              }
          }
      }
      return dp[n][W];
  }
  ```

Az optimális alstruktúra itt világosan látható, mivel a probléma minden állapota (adott kapacitás és adott számú tárgy) csak az előző állapotok optimális megoldásaira épül.

##### Feszítőfa és a Prim algoritmus

A minimális feszítőfa problémában, ahogy korábban már említettük, az optimális alstruktúra tulajdonság szintén jelen van. A Prim algoritmusban például minden lépésben a legkisebb súlyú élt adjuk hozzá a már kialakult feszítőfához, és ez az új él mindig része egy optimális feszítőfának.

#### Az optimális alstruktúra tulajdonságának bizonyítása

Az optimális alstruktúra tulajdonságának bizonyítása általában az alábbi lépéseket tartalmazza:

1. **Az alproblémák meghatározása**: Először is azonosítjuk, hogy az eredeti probléma milyen kisebb alproblémákra bontható.

2. **Rekurzív felbontás**: Megmutatjuk, hogy az eredeti probléma megoldása kifejezhető ezen alproblémák megoldásainak kombinációjaként.

3. **Optimalitás igazolása**: Bizonyítjuk, hogy az alproblémák optimális megoldásainak kombinálása optimális megoldást eredményez az eredeti problémára.

##### Példa: Fibonaccis számok

A Fibonacci-sorozat előállítása egy egyszerű példa az optimális alstruktúra tulajdonságára. A Fibonacci-számokat úgy definiáljuk, hogy minden szám az előző két szám összege:

$$
F(n) = F(n-1) + F(n-2)
$$

Itt az optimális alstruktúra tulajdonsága nyilvánvaló, mivel minden Fibonacci-szám az előző két Fibonacci-szám optimális kombinációja.

```cpp
#include <vector>

int fibonacci(int n) {
    if (n <= 1) return n;
    std::vector<int> fib(n + 1);
    fib[0] = 0;
    fib[1] = 1;
    for (int i = 2; i <= n; ++i) {
        fib[i] = fib[i-1] + fib[i-2];
    }
    return fib[n];
}
```

#### Az optimális alstruktúra jelentősége a gyakorlatban

Az optimális alstruktúra tulajdonságának felismerése és kihasználása lehetővé teszi az algoritmusok hatékony tervezését és megoldását számos területen, mint például a hálózati tervezés, az ütemezési problémák, a gráf algoritmusok, és még sok más. A dinamikus programozás és a mohó algoritmusok sikerének kulcsa az, hogy a problémát olyan módon strukturáljuk, hogy az alproblémák megoldásaiból építkezve érjük el az optimális megoldást.

#### Az optimális alstruktúra és a mohó algoritmusok kapcsolata

A mohó algoritmusok alkalmazása során az optimális alstruktúra tulajdonsága kritikus szerepet játszik. A mohó stratégia azonnali, lokális döntéseken alapul, amelyek csak akkor vezetnek globálisan optimális megoldáshoz, ha az alproblémák optimális megoldásai összhangban vannak az egész probléma optimális megoldásával. Ha egy probléma nem rendelkezik optimális alstruktúrával, a mohó megközelítés valószínűleg nem fog működni, és más módszerekre lesz szükség.

Összefoglalva, az optimális alstruktúra tulajdonsága az algoritmusok tervezésének egyik legfontosabb koncepciója, amely lehetővé teszi a komplex problémák hatékony és optimális megoldását. Az ilyen tulajdonsággal rendelkező problémák esetén a dinamikus programozás és a mohó algoritmusok alkalmazása különösen eredményes lehet, és jelentős mértékben hozzájárulhat a számítási hatékonyság növeléséhez.

### 4.1.4. Mohó algoritmusok elemzése és bizonyítása

A mohó algoritmusok elemzése és bizonyítása kritikus fontosságú lépés ahhoz, hogy megértsük és biztosítsuk ezeknek az algoritmusoknak a helyességét és hatékonyságát. A mohó algoritmusok általában egyszerűek és gyorsak, de nem minden esetben garantálják a globálisan optimális megoldást. Ezért fontos alaposan megvizsgálni, hogy milyen feltételek mellett működnek helyesen, és hogyan bizonyíthatjuk az optimalitásukat. Ebben a fejezetben részletesen áttekintjük a mohó algoritmusok elemzésének és bizonyításának módszereit, valamint gyakorlati példákon keresztül mutatjuk be ezeket a technikákat.

#### Mohó algoritmusok elemzésének alapjai

Az elemzés során két fő szempontot kell figyelembe venni: a helyességet és a hatékonyságot. A helyesség azt jelenti, hogy az algoritmus valóban optimális megoldást talál a problémára, míg a hatékonyság azt, hogy az algoritmus milyen gyorsan és milyen erőforrás-igénnyel oldja meg a feladatot.

1. **Helyesség**: A helyesség elemzésekor meg kell mutatnunk, hogy a mohó algoritmus által generált megoldás valóban optimális. Ehhez általában két tulajdonságot kell igazolni:
  - **Mohó választás tulajdonsága**: Minden lépésben a lokálisan optimális választás egy globálisan optimális megoldás része.
  - **Optimális alstruktúra**: A probléma optimális megoldása tartalmazza az alproblémák optimális megoldásait.

2. **Hatékonyság**: A hatékonyság elemzésekor az algoritmus futási idejét és memóriahasználatát kell vizsgálni. Ez általában aszimptotikus idő- és tárbonyolultsági analízissel történik.

#### Példák a mohó algoritmusok elemzésére és bizonyítására

##### Kruskal algoritmus

A Kruskal algoritmus a minimális feszítőfa (MST) problémájának megoldására szolgál. Az algoritmus minden lépésben a legkisebb súlyú élt választja, amely nem hoz létre ciklust, és hozzáadja azt a feszítőfához.

**Helyesség bizonyítása:**

1. **Mohó választás tulajdonsága**: A Kruskal algoritmus minden lépésben a legkisebb súlyú élt választja, amely nem hoz létre ciklust. Ez a választás mindig része egy optimális feszítőfának, mert ha nem így lenne, akkor lenne egy másik, kisebb súlyú él, amely része az optimális megoldásnak, de az algoritmus már kiválasztotta volna ezt az élt.

2. **Optimális alstruktúra**: Ha egy gráf minimális feszítőfáját két részre bontjuk, akkor mindkét részre külön-külön is igaz, hogy a részproblémák optimális megoldásai együtt alkotják az eredeti probléma optimális megoldását. Ez a tulajdonság biztosítja, hogy a Kruskal algoritmus minden lépése során a kiválasztott élek optimális megoldást alkotnak.

**Hatékonyság elemzése:**

A Kruskal algoritmus futási ideje az élek rendezése miatt $O(E \log E)$, ahol $E$ az élek száma. Az élek egyesítése és az összefüggő komponensek kezelése diszkrét egyesítési (union-find) struktúrákkal történik, amely szintén hatékony, különösen ha rang és útösszegzést (path compression) használunk, ami szinte konstans időben működik.

```cpp
#include <vector>

#include <algorithm>

struct Edge {
    int u, v, weight;
    bool operator<(Edge const& other) {
        return weight < other.weight;
    }
};

int find_set(int v, std::vector<int>& parent) {
    if (v == parent[v])
        return v;
    return parent[v] = find_set(parent[v], parent);
}

void union_sets(int a, int b, std::vector<int>& parent, std::vector<int>& rank) {
    a = find_set(a, parent);
    b = find_set(b, parent);
    if (a != b) {
        if (rank[a] < rank[b])
            std::swap(a, b);
        parent[b] = a;
        if (rank[a] == rank[b])
            rank[a]++;
    }
}

int main() {
    int n;
    std::vector<Edge> edges;
    
    // Initialize edges and number of vertices

    std::sort(edges.begin(), edges.end());

    std::vector<int> parent(n);
    std::vector<int> rank(n, 0);
    for (int i = 0; i < n; i++)
        parent[i] = i;

    int cost = 0;
    std::vector<Edge> result;
    for (Edge e : edges) {
        if (find_set(e.u, parent) != find_set(e.v, parent)) {
            cost += e.weight;
            result.push_back(e);
            union_sets(e.u, e.v, parent, rank);
        }
    }

    // Output the result
}
```

##### Prim algoritmus

A Prim algoritmus szintén a minimális feszítőfa problémáját oldja meg, de más megközelítést alkalmaz. Az algoritmus egy kezdő csúcsból indul, és fokozatosan bővíti a feszítőfát úgy, hogy mindig a legkisebb súlyú, még nem kapcsolódó élt választja.

**Helyesség bizonyítása:**

1. **Mohó választás tulajdonsága**: A Prim algoritmus minden lépésben a legkisebb súlyú élt választja, amely összeköti a már kiválasztott csúcsokat egy még nem kiválasztott csúccsal. Ez a lokálisan optimális választás mindig része egy globálisan optimális feszítőfának, mert ha nem így lenne, akkor egy másik, kisebb súlyú él lenne kiválasztható, de az algoritmus már kiválasztotta volna ezt az élt.

2. **Optimális alstruktúra**: Ha a gráfot két részre bontjuk úgy, hogy az egyik rész már a feszítőfa része, a másik rész pedig még nem, akkor a két rész összekötésére kiválasztott legkisebb súlyú él mindig része lesz az optimális megoldásnak.

**Hatékonyság elemzése:**

A Prim algoritmus futási ideje függ a használt adatstruktúrától. Egy prioritási sor használatával a futási idő $O(E \log V)$, ahol $E$ az élek száma és $V$ a csúcsok száma. Fibonacci kupacok használatával ez tovább csökkenthető $O(E + V \log V)$-re.

```cpp
#include <vector>

#include <queue>

const int INF = 1e9;

struct Edge {
    int to, weight;
};

int main() {
    int n, start;
    std::vector<std::vector<Edge>> adj;

    // Initialize adjacency list, number of vertices, and starting vertex

    std::vector<int> dist(n, INF);
    dist[start] = 0;
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> pq;
    pq.push({0, start});

    while (!pq.empty()) {
        int v = pq.top().second;
        int d_v = pq.top().first;
        pq.pop();

        if (d_v != dist[v])
            continue;

        for (Edge edge : adj[v]) {
            int to = edge.to;
            int len = edge.weight;

            if (dist[v] + len < dist[to]) {
                dist[to] = dist[v] + len;
                pq.push({dist[to], to});
            }
        }
    }

    // Output the shortest distances from start to all vertices
}
```

##### Hátizsák probléma

A hátizsák probléma (knapsack problem) egyik változata, a töredékes hátizsák probléma (fractional knapsack problem) kiváló példa a mohó algoritmusok helyességének és hatékonyságának elemzésére. Ebben a problémában a tárgyak töredékeit is be lehet tenni a hátizsákba.

**Helyesség bizonyítása:**

1. **Mohó választás tulajdonsága**: A töredékes hátizsák probléma esetén a mohó választás tulajdonsága azt jelenti, hogy mindig azt a tárgyat vagy annak töredékét választjuk, amely a legnagyobb érték/súly aránnyal rendelkezik. Ez a választás biztosítja, hogy a hátizsákban lévő tárgyak összértéke maximális lesz.

2. **Optimális alstruktúra**: A probléma optimális megoldása a tárgyak optimális kiválasztására épül, ahol minden lépésben a legjobb érték/súly arányú tárgyat választjuk ki. Az optimális alstruktúra itt azt jelenti, hogy az előző lépésben kiválasztott tárgyak optimális kombinációja része az egész probléma optimális megoldásának.

**Hatékonyság elemzése:**

A töredékes hátizsák probléma mohó algoritmussal történő megoldása gyors és hatékony. Az algoritmus futási ideje $O(n \log n)$, ahol $n$ a tárgyak száma, mivel az érték/súly arányok szerinti rendezés dominálja a számítási időt.

```cpp
#include <vector>

#include <algorithm>

struct Item {
    int value, weight;
    bool operator<(Item const& other) {
        return (double)value / weight > (double)other.value / other.weight;
    }
};

double fractional_knapsack(int W, std::vector<Item>& items) {
    std::sort(items.begin(), items.end());

    double total_value = 0.0;
    for (const auto& item : items) {
        if (W == 0) break;
        if (item.weight <= W) {
            W -= item.weight;
            total_value += item.value;
        } else {
            total_value += item.value * ((double)W / item.weight);
            W = 0;
        }
    }

    return total_value;
}
```

#### Mohó algoritmusok bizonyítása

A mohó algoritmusok bizonyítása általában a következő lépéseket tartalmazza:

1. **Mohó választás tulajdonságának bizonyítása**: Bemutatjuk, hogy minden lépésben a lokálisan optimális választás része egy globálisan optimális megoldásnak.

2. **Induktív bizonyítás**: Használhatunk matematikai indukciót a mohó algoritmus helyességének bizonyítására. Az indukció alapja az, hogy az algoritmus első lépése helyes, az indukciós lépés pedig azt mutatja, hogy ha az algoritmus helyesen működik az első $k$ lépésben, akkor a $k+1$-edik lépésben is helyes megoldást ad.

3. **Ellentmondásos bizonyítás**: Gyakran alkalmazhatunk ellentmondásos bizonyítást is. Feltesszük, hogy létezik egy jobb megoldás, mint amit a mohó algoritmus talál, és bebizonyítjuk, hogy ez ellentmond a feltételezéseinknek.

##### Példa: Intervallum kiválasztási probléma

Az intervallum kiválasztási probléma célja a lehető legtöbb, nem átfedő intervallum kiválasztása egy adott intervallumkészletből.

**Mohó algoritmus:**

1. Rendezzük az intervallumokat a végpontjuk szerint növekvő sorrendben.
2. Válasszuk ki az első intervallumot.
3. Válasszuk ki a következő intervallumot, amely nem átfedő az utoljára kiválasztottal.

**Helyesség bizonyítása:**

1. **Mohó választás tulajdonsága**: Az algoritmus mindig a legkorábban végződő intervallumot választja, amely nem átfedő az eddig kiválasztottakkal. Ha lenne egy jobb megoldás, amely több intervallumot tartalmaz, akkor az ellentmondana annak a ténynek, hogy az első kiválasztott intervallum a legkorábban végződik.

2. **Optimális alstruktúra**: Az optimális megoldás mindig tartalmazza a legkorábban végződő intervallumot, majd az ezt követő optimális megoldást a fennmaradó intervallumokra.

**Hatékonyság elemzése:**

Az algoritmus futási ideje az intervallumok rendezése miatt $O(n \log n)$, ahol $n$ az intervallumok száma. A kiválasztási folyamat lineáris időben történik.

#### Összegzés

A mohó algoritmusok elemzése és bizonyítása alapvető fontosságú a helyességük és hatékonyságuk biztosításához. Az ilyen algoritmusok alkalmazása számos területen előnyös, mivel egyszerűek és gyorsak, feltéve hogy a probléma rendelkezik a szükséges tulajdonságokkal, mint például a mohó választás tulajdonsága és az optimális alstruktúra. Az elemzési és bizonyítási technikák, mint a mohó választás tulajdonságának igazolása, az induktív bizonyítás és az ellentmondásos bizonyítás, segítenek biztosítani, hogy ezek az algoritmusok megbízhatóan működjenek a gyakorlatban.
