\newpage

# 7. Hálózati algoritmusok

# 7.1. Minimális feszítőfa 

A hálózatok tervezése és elemzése területén az egyik legfontosabb és leggyakrabban használt eszköz a minimális feszítőfa algoritmus. Ezek az algoritmusok ahhoz nyújtanak alapvető segítséget, hogy egy hálózat minden pontját úgy kössük össze, hogy a kapcsolatok összköltsége a lehető legkisebb legyen. Legyen szó távközlési hálózatokról, útvonaltervezésről vagy éppen elektromos hálózatok kialakításáról, a minimális feszítőfa algoritmusok biztosítják, hogy a kívánt csomópontok minden esetben hatékony és költségtakarékos módon legyenek összekapcsolva. Ebben a fejezetben megismerhetjük a legfontosabb minimális feszítőfa algoritmusokat, beleértve Prim és Kruskal algoritmusait, valamint betekintést nyerhetünk ezek matematikai hátterébe és gyakorlati alkalmazásukba is.

## 7.1.1. Prim algoritmusa

A minimális feszítőfa (MST) egy olyan részgráf, amely a gráf összes csúcsát összekapcsolja a lehető legkisebb összsúlyú élekkel. Számos módszer létezik ennek a célkitűzésnek az elérésére, és az egyik leggyakrabban használt algoritmus a Prim algoritmusa. A Prim algoritmus lépésről lépésre építi fel a minimális feszítőfát, mindig hozzáadva a legkisebb súlyú élt, amely összeköti a fát egy új csúccsal. Ez az elegáns és hatékony módszer különösen hasznos a súlyozott és összefüggő gráfok esetében. Ebben a fejezetben először a Prim algoritmus alapelveit és implementációját mutatjuk be, majd az algoritmus különböző alkalmazási területeit és konkrét példákat tárgyalunk.

### 7.1.1.1. Alapelvek és implementáció

A Prim algoritmus egy széleskörűen használt módszer, amely minimális feszítőfát (Minimum Spanning Tree, MST) generál súlyozott, összefüggő, irányítatlan gráfokban. A minimális feszítőfa egy olyan feszítőfa, amelynek az összes élének súlyösszege minimális az összes lehetséges feszítőfa közül az adott gráfban. Az algoritmus nevét felfedezőjéről, Robert C. Primről kapta, aki 1957-ben dolgozta ki azt.

#### Alapelvek

A Prim algoritmus alapelve az ún. "greedy" (kapzsi) módszeren alapul. Az algoritmus lépésről lépésre építi fel a minimális feszítőfát azáltal, hogy mindig a legkisebb súlyú élt választja, amely hozzáad egy új csúcsot a már épülő feszítőfához. Az algoritmus kezdetekor egy tetszőleges kezdőcsúcsot választunk, majd iteratívan növeljük a fát a következő lépések mentén:

1. **Initializálás:** Válasszunk egy tetszőleges csúcsot $u$ a gráfban G=(V, E). Jelöljünk meg $u$-t a fa első elemeként.
2. **Élek kiválasztása:** Azokat az éleket keressük, amelyek a már kiválasztott csúcsok (kezdetben csak $u$) és a még nem kiválasztott csúcsok között húzódnak.
3. **Minimális él kiválasztása:** Válasszuk ki a minimális súlyú élt, amely az aktuális fának új csúcsot ad hozzá.
4. **Ismétlés:** Ismételjük meg az előző két lépést, amíg minden csúcs bekerül a feszítőfába.

#### Részletes folyamat

Az algoritmus részletes leírása az alábbi:

1. **Adatstruktúra választása:**
    - Tároljuk a gráfot szomszédsági listával vagy szomszédsági mátrixszal.
    - Használjunk egy prioritási sort (általában minimális halmot) a csúcspárok tárolásához és a legkisebb súlyú él kiválasztásához.
    - Két segédtömböt (vektort) használhatunk: egyik a kiderült minimális súlyú éleket tárolja (keys), a másik pedig a fát építő éleket (mstSet).

2. **Kezdőcsúcs és inicializáció:**
   ```cpp
   int V = graph.size(); // number of vertices
   vector<int> key(V, INT_MAX); // minimum weight edge for each vertex initially set to infinity
   vector<bool> inMST(V, false); // to track vertices included in MST initially set to false
   vector<int> parent(V, -1); // to store MST
    
   key[start_vertex] = 0; // Starting with vertex 0 (or any vertex)
   priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
   pq.push({0, start_vertex});
   ```

3. **Iterációk és kiválasztás:**
    - Mindig a legkisebb súlyú élt választjuk.
   ```cpp
   while (!pq.empty()) {
       int u = pq.top().second;
       pq.pop();
       inMST[u] = true; // Include u in MST
    
       // Update keys and parent index of adjacent vertices
       for (auto [v, weight] : graph[u]) {
           if (!inMST[v] && key[v] > weight) {
               key[v] = weight;
               pq.push({key[v], v});
               parent[v] = u;
           }
       }
   }
   ```

4. **Eredmény kiértékelése:**
    - Az `mstSet` tömb segítségével nyomon követhetjük az MST-t alkotó éleket.
    - A fő program blokkból kinyert él lista és azok súlyai jelentik a minimális feszítőfát.

#### Implementáció

Az alábbiakban bemutatjuk a Prim algoritmus C++ nyelvű implementációját. A kód szomszédsági listával dolgozik, mint a gráf reprezentációjának egyik módja.

```cpp
#include <iostream>

#include <vector>
#include <queue>

#include <climits>

using namespace std;

typedef pair<int, int> Pair;

void PrimMST(vector<vector<Pair>>& graph, int vertices) {
    // Priority queue to store (weight, vertex) pairs
    priority_queue<Pair, vector<Pair>, greater<Pair>> pq;
 
    int start_vertex = 0;
    vector<int> key(vertices, INT_MAX);  // Initialize all key values as infinite
    vector<int> parent(vertices, -1);    // Stores constructed MST
    vector<bool> inMST(vertices, false); // MST set
 
    key[start_vertex] = 0;
    pq.push(make_pair(0, start_vertex));
 
    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();
        inMST[u] = true;
 
        for (auto [v, weight] : graph[u]) {
            if (!inMST[v] && key[v] > weight) {
                key[v] = weight;
                pq.push(make_pair(key[v], v));
                parent[v] = u;
            }
        }
    }
 
    for (int i = 1; i < vertices; ++i)
        cout << parent[i] << " - " << i << "\n";
}
 
int main() {
    int vertices = 9;
    vector<vector<Pair>> graph(vertices);
 
    // Add edges to the graph
    graph[0].push_back(make_pair(1, 4));
    graph[0].push_back(make_pair(7, 8));
    graph[1].push_back(make_pair(0, 4));
    graph[1].push_back(make_pair(2, 8));
    graph[1].push_back(make_pair(7, 11));
    graph[2].push_back(make_pair(1, 8));
    graph[2].push_back(make_pair(3, 7));
    graph[2].push_back(make_pair(5, 4));
    graph[2].push_back(make_pair(8, 2));
    graph[3].push_back(make_pair(2, 7));
    graph[3].push_back(make_pair(4, 9));
    graph[3].push_back(make_pair(5, 14));
    graph[4].push_back(make_pair(3, 9));
    graph[4].push_back(make_pair(5, 10));
    graph[5].push_back(make_pair(2, 4));
    graph[5].push_back(make_pair(3, 14));
    graph[5].push_back(make_pair(4, 10));
    graph[6].push_back(make_pair(7, 1));
    graph[6].push_back(make_pair(8, 6));
    graph[7].push_back(make_pair(0, 8));
    graph[7].push_back(make_pair(1, 11));
    graph[7].push_back(make_pair(6, 1));
    graph[7].push_back(make_pair(8, 7));
    graph[8].push_back(make_pair(2, 2));
    graph[8].push_back(make_pair(6, 6));
    graph[8].push_back(make_pair(7, 7));
 
    PrimMST(graph, vertices);
 
    return 0;
}
```

#### Értékelés és optimalizáció

Az algoritmus időbeli komlexitása függ az alkalmazott adatstruktúráktól. A bemutatott implementáció prioritási soron keresztül működik:
- Ha bináris halmot használunk a prioritási sorhoz, akkor az időbeli komlexitás O((V+E) log V), ahol V a csúcsok száma és E az élek száma.
- Ez elképesztően hatékony, különösen akkor, ha a gráf ritka, azaz az élek száma lineárisan aránylik a csúcsok számához.

A Prim algoritmus viszonylag egyszerű, és nagymértékben optimalizálható speciális adatstruktúrák (pl. Fibonacci-hip) és implementációs trükkök segítségével. Az így készült optimalizált változat még nagyobb gráfok esetén is jól teljesít.

Összegzésképpen, a Prim algoritmus a minimális feszítőfa keresésének egyik leghatékonyabb és legismertebb módszere, amely rengeteg gyakorlati alkalmazási területtel rendelkezik, beleértve a hálózati tervezést, képkompressziót és más optimalizációs problémákat, amelyek hálózati konfigurációk hatékonyságát célozzák meg.

### 7.1.1.2. Alkalmazások és példák

Prim algoritmusa az egyik leghatékonyabb algoritmus a minimális feszítőfák (MST) megtalálásában a gráfelmélet területén. Ennek az algoritmusnak a széles körű alkalmazása számos területen elengedhetetlen a számítástudományban és mérnöki alkalmazásokban. Ebben a fejezetben részletesen tárgyaljuk az algoritmus különböző alkalmazásait és példákat mutatunk be annak működésére.

#### Alkalmazások

**1. Hálózatok tervezése:**

Az egyik leggyakoribb alkalmazási terület a hálózatok tervezése. Ez magába foglalja a telekommunikációs hálózatok, számítógépes hálózatok, útvonaltervezés és az elektromos áramhálózatok optimális kialakítását.

* **Telekommunikációs hálózatok:** Az MST-k segítenek minimalizálni a kábelhosszok összegét egy telekommunikációs hálózat kiépítése során, biztosítva ezzel az összes csomópont kapcsolatát minimális költséggel.
* **Elektromos áramhálózatok:** Az energiaátviteli hálózat tervezésekor az MST-t használhatják arra, hogy minimalizálják a vezetők hosszát, ezzel csökkentve a létesítési költségeket és az energiaveszteséget.

**2. Klaszteranalízis:**

Klaszteranalízis során az MST-k segítségével az adatpontok csoportosítása történik, hogy a csoportokban lévő pontok közötti távolság minimalizálódjon.

* **Képfeldolgozás:** Az MST algoritmust képfeldolgozásban is használják, például szegmentációs feladatokban, ahol az algoritmus segít az objektumok azonosításában és különálló részekre bontásában.
* **Adatbányászat:** Az adathalmazok klaszterezésében az MST alapú algoritmusok lehetővé teszik az adatok legjellemzőbb struktúráinak megtalálását.

**3. Infrastruktúra tervezése:**

Prim algoritmusa széles körben alkalmazható különböző infrastruktúrák, például közúti, vasúti és csővezeték rendszerek tervezésekor, ahol a minimális költségű összeköttetés kialakítása a cél.

* **Városi tervezés:** Út- és közlekedési hálózatok optimális kialakítása, hogy minimalizálják az építési és karbantartási költségeket, miközben maximális kapcsolatot biztosítanak.
* **Vasúti hálózat:** A vasúti hálózatok tervezésénél a Prim algoritmusa minimális hosszúságú vágányrendszert biztosít a különböző állomások között.

#### Példák

Az alábbiakban bemutatunk egy részletes példát a Prim algoritmus konkrét alkalmazására és annak implementációjára.

**Példa 1: Telekommunikációs hálózat tervezése**

Tegyük fel, hogy van egy telekommunikációs cég, amelyik városok között szeretne kiépíteni egy hálózatot úgy, hogy a kábelek hossza minimális legyen. Az alábbi gráfban a csúcsok városokat, az élek pedig a kábelek lehetséges útjait jelölik, az élek súlya pedig a kábeltávot jelenti.

```
  (A)---4---(B)
   |       /  |
   9     7    2
   |   /      |
  (C)---8---(D)
   |
   3
   |
  (E)
```

A következő C++ kód valósítja meg a Prim algoritmust egy ilyen hálózat tervezésében:

```cpp
#include <iostream>

#include <vector>
#include <queue>

#include <climits>

using namespace std;

typedef pair<int, int> pii;

void primMST(vector<vector<pii>> &graph) {
    int n = graph.size();
    vector<int> key(n, INT_MAX);
    vector<int> parent(n, -1);
    vector<bool> inMST(n, false);

    priority_queue<pii, vector<pii>, greater<pii>> pq;
    key[0] = 0;
    pq.push({0, 0}); // {weight, vertex}

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();
        inMST[u] = true;

        for (auto &[weight, v] : graph[u]) {
            if (!inMST[v] && key[v] > weight) {
                key[v] = weight;
                pq.push({key[v], v});
                parent[v] = u;
            }
        }
    }

    cout << "Edge \tWeight\n";
    for (int i = 1; i < n; ++i)
        cout << parent[i] << " - " << i << " \t" << key[i] << "\n";
}

int main() {
    int V = 5;
    vector<vector<pii>> graph(V);
    
    graph[0].emplace_back(4, 1);
    graph[0].emplace_back(9, 2);
    graph[2].emplace_back(3, 4);
    graph[1].emplace_back(7, 2);
    graph[2].emplace_back(8, 3);
    graph[1].emplace_back(2, 3);

    graph[1].emplace_back(4, 0);
    graph[2].emplace_back(9, 0);
    graph[4].emplace_back(3, 2);
    graph[2].emplace_back(7, 1);
    graph[3].emplace_back(8, 2);
    graph[3].emplace_back(2, 1);

    primMST(graph);

    return 0;
}
```

A fenti implementációban egy gráfot inicializálunk, és Prim algoritmusát alkalmazzuk az MST megkeresésére. Az eredményül kapott élek és azok súlyai biztosítják a minimális költségű összeköttetést a városok között.

##### Magyarázat:

1. **Gráf inicializálása:**
    * A `graph` változó tárolja a gráfot vektorok vektoraként, ahol minden belső vektor egy csúcs szomszédos csúcsait és súlyait tartalmazza.
    * Az élek oda-vissza hozzáadásával a gráf szimmetrikus lesz, mivel az élek kétirányúak.

2. **Prim's MST Algoritmus:**
    * `key` vektor felhasználásával tároljuk egy csúcs legkisebb súlyú élének súlyát.
    * `parent` vektor tartalmazza az MST-ben lévő csúcs szülőjét.
    * `inMST` vektor annak nyomon követésére szolgál, hogy egy csúcs már része-e az MST-nek.
    * Az algoritmus a prioritási sor (`priority_queue`) segítségével választja ki a legkisebb súlyú élt.

3. **Eredmény kiírása:**
    * Az élek és azok súlyai a `key` és `parent` vektorok segítségével kerülnek kiíratásra.

**Példa 2: Klaszteranalízis**

Egy másik példa a klaszteranalízis. Tekintsünk egy adatbázist, ahol az adatpontok közötti hasonlóságokat egy gráf segítségével ábrázoljuk. Az MST segítségével az adatpontokat klaszterekbe csoportosíthatjuk úgy, hogy a csoporton belüli hasonlóság maximális, a csoportok közötti távolság pedig minimalizált legyen.

