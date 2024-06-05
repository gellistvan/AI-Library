\newpage

## 1.13. Helyfüggő keresési algoritmusok (Spatial Search)

A helyfüggő keresési algoritmusok elengedhetetlenek a térinformatikai adatok kezelésében és elemzésében. Ezek az algoritmusok lehetővé teszik a nagy mennyiségű térbeli adat hatékony tárolását, keresését és visszakeresését, amelyek nélkülözhetetlenek a modern alkalmazásokban, mint például a térképszolgáltatások, a robotika, vagy az autonóm járművek navigációja. A következő alfejezetekben részletesen bemutatjuk a három legismertebb helyfüggő keresési struktúrát: a K-d fákat, az R-fákat és a quadtrees-eket. Ezek az adatszerkezetek különböző módszerekkel közelítik meg a térbeli adatok feldolgozását, mindegyikük különböző előnyökkel és felhasználási területekkel rendelkezik.

### 1.13.1. K-d fák (K-D Trees)

A K-d fa (k-dimenziós fa) egy bináris keresőfa, amelyet k-dimenziós térbeli adatok hatékony tárolására és keresésére használnak. Az algoritmus először Jon Bentley által lett bevezetve 1975-ben, és azóta széles körben elterjedt a számítástechnika különböző területein, például a gépi tanulásban, a számítógépes grafikában és a térinformatikai rendszerekben.

#### Alapfogalmak és motiváció

A K-d fa alapvetően egy bináris keresőfa általánosítása, amely lehetővé teszi a többdimenziós keresést. Míg egy hagyományos bináris keresőfa csak egy dimenzióban rendezi az adatokat, a K-d fa több dimenzióban is képes hatékonyan kezelni az adatokat. A fa minden csomópontja egy k-dimenziós pontot tartalmaz, és a fa minden szintjén az egyik dimenzió mentén történik a szétválasztás. Ezáltal a keresési idő jelentősen csökken a hagyományos lineáris keresési módszerekhez képest.

#### K-d fa felépítése

A K-d fa felépítése rekurzív folyamat, amely a következő lépésekből áll:
1. **Adatok rendezése:** Az adatpontokat a választott dimenzió mentén rendezzük.
2. **Medián kiválasztása:** A rendezetlen adatok mediánját választjuk ki a szétválasztáshoz. Ez lesz a jelenlegi csomópont.
3. **Rekurzív felosztás:** A mediántól balra és jobbra lévő adatpontokkal ugyanezt a folyamatot ismételjük meg, mindaddig, amíg minden adatpont csomóponttá nem válik.

A fenti folyamat során a szétválasztási dimenziót minden szinten növekvő sorrendben váltogatjuk, azaz az első szinten az első dimenzió mentén, a második szinten a második dimenzió mentén, és így tovább. Amikor elérjük az utolsó dimenziót, újra az első dimenzióval kezdjük a következő szinten.

#### K-d fa bejárása

A K-d fa keresési műveletei – például a legközelebbi szomszéd keresés (nearest neighbor search) vagy a tartománykeresés (range search) – a fa rekurzív bejárásával történnek.

**Legközelebbi szomszéd keresés:**
A legközelebbi szomszéd keresési algoritmus célja megtalálni a fa legközelebbi pontját egy adott keresési ponthoz. Az algoritmus lépései a következők:
1. **Rekurzív leszállás:** Kezdjük a gyökércsomópontnál, és lépjünk lefelé a fa mentén a keresési pont és a csomópontok összehasonlítása alapján.
2. **Legközelebbi pont frissítése:** Minden meglátogatott csomópontnál ellenőrizzük, hogy az adott csomópont közelebb van-e a keresési ponthoz, mint a jelenlegi legközelebbi pont. Ha igen, frissítjük a legközelebbi pontot.
3. **Alfák ellenőrzése:** Ha a keresési pont és a csomópont szétválasztási síkja közötti távolság kisebb, mint a jelenlegi legközelebbi pont és a keresési pont közötti távolság, akkor ellenőrizzük a másik ágat is.

**Tartománykeresés:**
A tartománykeresési algoritmus célja megtalálni az összes olyan pontot a fában, amelyek egy adott k-dimenziós téglalapon belül vannak. Az algoritmus lépései a következők:
1. **Rekurzív leszállás:** Kezdjük a gyökércsomópontnál, és lépjünk lefelé a fa mentén.
2. **Pontok összehasonlítása:** Minden csomópontnál ellenőrizzük, hogy a pont a keresési tartományon belül van-e.
3. **Alfák ellenőrzése:** Ha a csomópont szétválasztási síkja áthalad a keresési tartományon, ellenőrizzük mindkét ágat.

#### Példakód C++ nyelven

Az alábbiakban bemutatunk egy egyszerű K-d fa implementációt C++ nyelven:

```cpp
#include <iostream>

#include <vector>
#include <algorithm>

struct Point {
    std::vector<float> coordinates;
    Point(std::vector<float> coords) : coordinates(coords) {}
};

struct Node {
    Point point;
    Node* left;
    Node* right;
    Node(Point pt) : point(pt), left(nullptr), right(nullptr) {}
};

class KDTree {
public:
    KDTree(const std::vector<Point>& points) {
        root = buildTree(points, 0);
    }

    Node* buildTree(const std::vector<Point>& points, int depth) {
        if (points.empty()) return nullptr;

        int k = points[0].coordinates.size();
        int axis = depth % k;

        std::vector<Point> sortedPoints = points;
        std::sort(sortedPoints.begin(), sortedPoints.end(), [axis](const Point& a, const Point& b) {
            return a.coordinates[axis] < b.coordinates[axis];
        });

        int medianIndex = sortedPoints.size() / 2;
        Node* node = new Node(sortedPoints[medianIndex]);

        std::vector<Point> leftPoints(sortedPoints.begin(), sortedPoints.begin() + medianIndex);
        std::vector<Point> rightPoints(sortedPoints.begin() + medianIndex + 1, sortedPoints.end());

        node->left = buildTree(leftPoints, depth + 1);
        node->right = buildTree(rightPoints, depth + 1);

        return node;
    }

private:
    Node* root;
};

int main() {
    std::vector<Point> points = { Point({2.0, 3.0}), Point({5.0, 4.0}), Point({9.0, 6.0}),
                                  Point({4.0, 7.0}), Point({8.0, 1.0}), Point({7.0, 2.0}) };

    KDTree tree(points);

    return 0;
}
```

#### K-d fák előnyei és hátrányai

A K-d fáknak számos előnye és hátránya van, amelyeket érdemes figyelembe venni a használatuk során:

**Előnyök:**
1. **Hatékony keresés:** A K-d fák jelentősen csökkentik a keresési időt nagy mennyiségű térbeli adat esetén.
2. **Egyszerű implementáció:** A K-d fák viszonylag egyszerűen implementálhatók és használhatók.
3. **Dimenziófüggetlenség:** A K-d fák bármilyen dimenziójú térbeli adatok kezelésére alkalmasak.

**Hátrányok:**
1. **Egyenlőtlen eloszlás kezelése:** A K-d fák nem mindig működnek jól egyenlőtlenül eloszlott adatok esetén, mivel a medián alapú szétválasztás nem biztosít egyenletes eloszlást a csomópontok között.
2. **Magas dimenziójú adatok:** A K-d fák hatékonysága csökken, ahogy a dimenziók száma növekszik (ez a jelenség az úgynevezett "dimenziós átok").
3. **Dinamikus adatok kezelése:** A K-d fák nehezebben kezelik a dinamikusan változó adatokat, mint például az adatok beszúrását és törlését, mivel ezek az operációk újraépítést igényelhetnek a fa egyensúlyának fenntartása érdekében.

Összességében a K-d fák hatékony és gyakran használt adatszerkezetek a térbeli keresési problémák megoldására, különösen alacsony dimenziós adatok esetén. Azonban fontos mérlegelni a hátrányokat és a konkrét alkalmazási környezet igényeit, mielőtt döntést hozunk a használatuk mellett.

### 1.13.2. R-fák (R-Trees)

Az R-fák egy hatékony adatszerkezet a többdimenziós térbeli adatok tárolására és lekérdezésére, amelyet Antonin Guttman fejlesztett ki 1984-ben. Az R-fák különösen hasznosak a térbeli adatbázisokban, térinformatikai rendszerekben, és más olyan alkalmazásokban, amelyek nagy mennyiségű térbeli adatot kezelnek.

#### Alapfogalmak és motiváció

Az R-fák célja a többdimenziós objektumok, például pontok, vonalak és téglalapok hatékony tárolása és keresése. A K-d fáktól eltérően, amelyek csak pontokat tárolnak, az R-fák képesek összetett objektumok kezelésére is. Az R-fák lehetővé teszik az objektumok hatékony keresését és a tartománykeresést, miközben minimalizálják a szükséges lemezelérések számát.

Az R-fák alapvető ötlete a hierarchikus térbeli felosztás, ahol a fa minden csomópontja egy téglalap alakú burkoló (bounding rectangle) régiót reprezentál, amely tartalmazza az összes gyermekcsomópont által lefedett régiót. Az ilyen téglalapok neve minimális burkoló téglalap (Minimum Bounding Rectangle, MBR).

#### R-fák szerkezete

Az R-fák egy fastruktúrát követnek, ahol minden csomópont egy MBR-t és a hozzá tartozó elemeket tartalmazza. Az R-fákban két típusú csomópont található: belső csomópontok és levélcsomópontok. A belső csomópontok más csomópontokra mutatnak, míg a levélcsomópontok a tényleges adatobjektumokat tárolják.

##### Csomópontok felépítése

- **MBR:** Minden csomópont egy minimális burkoló téglalapot tartalmaz, amely lefedi az összes benne lévő adatobjektumot vagy gyermekcsomópontot.
- **Gyermekek:** A csomópont gyermekcsomópontokra mutat. A belső csomópontok gyermekcsomópontokat tartalmaznak, míg a levélcsomópontok adatobjektumokat.

##### Csomópont feltöltése

Az R-fák csomópontjai általában korlátozott számú elemet tartalmaznak. Ha egy csomópont megtelik, fel kell osztani két részre, amelyeket külön-külön kezelünk. Ezt a folyamatot osztásnak nevezzük.

#### R-fák működése

Az R-fák három alapvető műveletet támogatnak: beszúrás, törlés és keresés. Mindezek a műveletek az R-fa szerkezetét és a csomópontok MBR-jait használják a hatékony végrehajtáshoz.

##### Beszúrás

Az adatobjektum beszúrása az R-fába a következő lépésekben történik:
1. **Megfelelő levélcsomópont kiválasztása:** A fa bejárásával megtaláljuk azt a levélcsomópontot, amelynek MBR-je a legkisebb bővítést igényli az új objektum befogadásához.
2. **Levélcsomópont bővítése:** Az új objektumot a kiválasztott levélcsomópontba illesztjük, és szükség esetén frissítjük az MBR-t.
3. **Csomópont osztása:** Ha a levélcsomópont megtelik, akkor két új csomópontra osztjuk, és az MBR-eket frissítjük.
4. **MBR-k frissítése:** Visszafelé haladva frissítjük az összes szülőcsomópont MBR-jét.

##### Törlés

Az adatobjektum törlése az R-fából a következő lépésekben történik:
1. **Levélcsomópont megtalálása:** A fa bejárásával megtaláljuk azt a levélcsomópontot, amely az eltávolítandó objektumot tartalmazza.
2. **Objektum eltávolítása:** Eltávolítjuk az objektumot a levélcsomópontból.
3. **MBR-k frissítése:** Visszafelé haladva frissítjük az MBR-eket, és szükség esetén egyesítjük a csomópontokat, ha azokban túl kevés objektum maradt.

##### Keresés

Az R-fák keresési műveletei, mint például a tartománykeresés vagy a legközelebbi szomszéd keresés, a következő lépésekben történnek:
1. **Rekurzív bejárás:** A fa rekurzív bejárása során minden olyan csomópontot ellenőrzünk, amelynek MBR-je átfedi a keresési régiót.
2. **Levélcsomópontok ellenőrzése:** Ha levélcsomópontot találunk, ellenőrizzük az összes benne lévő adatobjektumot, hogy megfelelnek-e a keresési feltételeknek.
3. **Gyermekcsomópontok bejárása:** Ha belső csomópontot találunk, rekurzívan bejárjuk annak gyermekcsomópontjait is.

#### Példakód C++ nyelven

Az alábbiakban egy egyszerű R-fa implementációt mutatunk be C++ nyelven:

```cpp
#include <iostream>

#include <vector>
#include <algorithm>

#include <limits>

struct MBR {
    std::vector<float> min_coords;
    std::vector<float> max_coords;

    MBR(int dimensions) : min_coords(dimensions, std::numeric_limits<float>::max()), max_coords(dimensions, std::numeric_limits<float>::lowest()) {}

    void expand(const MBR& other) {
        for (size_t i = 0; i < min_coords.size(); ++i) {
            min_coords[i] = std::min(min_coords[i], other.min_coords[i]);
            max_coords[i] = std::max(max_coords[i], other.max_coords[i]);
        }
    }

    bool overlaps(const MBR& other) const {
        for (size_t i = 0; i < min_coords.size(); ++i) {
            if (min_coords[i] > other.max_coords[i] || max_coords[i] < other.min_coords[i]) {
                return false;
            }
        }
        return true;
    }
};

struct Node {
    MBR mbr;
    std::vector<Node*> children;
    std::vector<MBR> entries;
    bool is_leaf;

    Node(int dimensions, bool leaf) : mbr(dimensions), is_leaf(leaf) {}

    void add_entry(const MBR& entry) {
        entries.push_back(entry);
        mbr.expand(entry);
    }

    void add_child(Node* child) {
        children.push_back(child);
        mbr.expand(child->mbr);
    }
};

class RTree {
public:
    RTree(int dimensions, int max_entries) : dims(dimensions), max_entries(max_entries) {
        root = new Node(dimensions, true);
    }

    void insert(const MBR& entry) {
        Node* leaf = choose_leaf(root, entry);
        leaf->add_entry(entry);

        if (leaf->entries.size() > max_entries) {
            Node* new_node = split_node(leaf);
            adjust_tree(leaf, new_node);
        }
    }

    bool search(const MBR& query, std::vector<MBR>& results) const {
        return search_recursive(root, query, results);
    }

private:
    Node* root;
    int dims;
    int max_entries;

    Node* choose_leaf(Node* node, const MBR& entry) {
        if (node->is_leaf) {
            return node;
        }

        Node* best_child = nullptr;
        float min_increase = std::numeric_limits<float>::max();
        for (Node* child : node->children) {
            MBR temp_mbr = child->mbr;
            temp_mbr.expand(entry);
            float increase = volume(temp_mbr) - volume(child->mbr);
            if (increase < min_increase) {
                min_increase = increase;
                best_child = child;
            }
        }
        return choose_leaf(best_child, entry);
    }

    Node* split_node(Node* node) {
        // Implement a simple linear split strategy
        int axis = 0;
        float min_coord = std::numeric_limits<float>::max();
        float max_coord = std::numeric_limits<float>::lowest();
        for (const MBR& entry : node->entries) {
            min_coord = std::min(min_coord, entry.min_coords[axis]);
            max_coord = std::max(max_coord, entry.max_coords[axis]);
        }
        float split_coord = (min_coord + max_coord) / 2;

        Node* new_node = new Node(dims, node->is_leaf);
        auto it = std::partition(node->entries.begin(), node->entries.end(), [&](const MBR& entry) {
            return entry.min_coords[axis] < split_coord;
        });

        new_node->entries.assign(it, node->entries.end());
        node->entries.erase(it, node->entries.end());

        return new_node;
    }

    void adjust_tree(Node* node, Node* new_node) {
        if (node == root) {
            root = new Node(dims, false);
            root->add_child(node);
            root->add_child(new_node);
        } else {
            Node* parent = find_parent(root, node);
            parent->add_child(new_node);
            if (parent->children.size() > max_entries) {
                Node* new_parent = split_node(parent);
                adjust_tree(parent, new_parent);
            }
        }
    }

    Node* find_parent(Node* root, Node* node) {
        if (root->is_leaf) return nullptr;
        for (Node* child : root->children) {
            if (child == node || find_parent(child, node)) {
                return root;
            }
        }
        return nullptr;
    }

    bool search_recursive(Node* node, const MBR& query, std::vector<MBR>& results) const {
        if (!node->mbr.overlaps(query)) {
            return false;
        }

        if (node->is_leaf) {
            for (const MBR& entry : node->entries) {
                if (entry.overlaps(query)) {
                    results.push_back(entry);
                }
            }
        } else {
            for (Node* child : node->children) {
                search_recursive(child, query, results);
            }
        }

        return true;
    }

    float volume(const MBR& mbr) const {
        float vol = 1.0f;
        for (size_t i = 0; i < mbr.min_coords.size(); ++i) {
            vol *= mbr.max_coords[i] - mbr.min_coords[i];
        }
        return vol;
    }
};

int main() {
    int dimensions = 2;
    int max_entries = 4;
    RTree rtree(dimensions, max_entries);

    rtree.insert(MBR({0.0, 0.0}, {1.0, 1.0}));
    rtree.insert(MBR({2.0, 2.0}, {3.0, 3.0}));
    rtree.insert(MBR({1.5, 1.5}, {2.5, 2.5}));

    std::vector<MBR> results;
    rtree.search(MBR({1.0, 1.0}, {3.0, 3.0}), results);

    for (const MBR& result : results) {
        std::cout << "Found MBR: (" << result.min_coords[0] << ", " << result.min_coords[1] << ") - (" << result.max_coords[0] << ", " << result.max_coords[1] << ")" << std::endl;
    }

    return 0;
}
```

#### R-fák előnyei és hátrányai

Az R-fák számos előnnyel és hátránnyal rendelkeznek, amelyek befolyásolják a használhatóságukat különböző alkalmazásokban.

**Előnyök:**
1. **Hatékony keresés és tárolás:** Az R-fák hatékonyan tárolják és keresik a térbeli adatokat, különösen nagy adatbázisok esetén.
2. **Alacsony lemezelérési költségek:** Az R-fák minimalizálják a szükséges lemezelérések számát, amely fontos szempont a nagy adatbázisok kezelésénél.
3. **Rugalmasság:** Az R-fák különböző típusú térbeli objektumokat képesek kezelni, beleértve a pontokat, vonalakat és téglalapokat.

**Hátrányok:**
1. **Karbantartási költségek:** Az R-fák karbantartása, különösen a beszúrások és törlések során, bonyolult lehet, és újraszervezést igényelhet.
2. **Összetett struktúra:** Az R-fák szerkezete és működése bonyolultabb, mint más adatstruktúráké, például a K-d fákké.
3. **Teljesítmény függ a térbeli eloszlástól:** Az R-fák teljesítménye nagymértékben függ az adatok térbeli eloszlásától. Egyenlőtlen eloszlású adatok esetén az R-fák hatékonysága csökkenhet.

Összefoglalva, az R-fák hatékony és rugalmas adatstruktúrák a térbeli adatok tárolására és keresésére, különösen nagy adatbázisok esetén. Azonban fontos figyelembe venni a hátrányokat és az adott alkalmazás igényeit, hogy optimálisan használjuk őket.

### 1.13.3. Quadtrees

A quadtree (kvadtfa) egy fa alapú adatszerkezet, amelyet elsősorban kétdimenziós térbeli adatok hatékony tárolására és keresésére használnak. A quadtreek alkalmazásai közé tartozik a térinformatika, a számítógépes grafika, a képkompresszió és a játékmenet-követés. Az adatszerkezetet Raphael Finkel és J.L. Bentley vezette be 1974-ben.

#### Alapfogalmak és motiváció

A quadtree egy rekurzív adatszerkezet, amely a tér egy adott régióját négy részre, vagy kvadránsra osztja. Minden csomópont egy régiót reprezentál, és legfeljebb négy gyermekcsomópontja lehet, amelyek a régió négy kvadránsát fedik le. Ez a felosztás addig folytatódik, amíg minden régió elég kicsi lesz, vagy egy adott kritériumot kielégít.

A quadtree előnye, hogy képes hatékonyan kezelni a nagy mennyiségű kétdimenziós adatot, különösen akkor, ha az adatok nem egyenletesen oszlanak el. Az adatszerkezet dinamikus, azaz az adatok beszúrása és törlése gyorsan és hatékonyan végrehajtható.

#### Quadtree szerkezete

A quadtree szerkezete hierarchikus, ahol minden csomópont egy téglalap alakú régiót reprezentál. A csomópontok két fő típusra oszlanak: belső csomópontok és levélcsomópontok. A belső csomópontok négy gyermekcsomópontot tartalmaznak, míg a levélcsomópontok az adatokat tárolják.

##### Csomópontok felépítése

- **Régió:** Minden csomópont egy téglalap alakú régiót reprezentál, amely a tér egy részét fedi le.
- **Gyermekek:** A belső csomópontok négy gyermekcsomóponttal rendelkeznek, amelyek a régió négy kvadránsát fedik le.
- **Adatok:** A levélcsomópontok adatokat tárolnak, például pontokat vagy más térbeli objektumokat.

##### Csomópont felosztása

Amikor egy csomópont megtelik, az adott régiót négy egyenlő részre osztjuk, és a benne lévő adatokat a megfelelő gyermekcsomópontokba helyezzük át. Ez a folyamat addig folytatódik, amíg minden régió elég kicsi lesz, vagy a csomópontok egy adott kritériumot kielégítenek.

#### Quadtree működése

A quadtreek három fő műveletet támogatnak: beszúrás, törlés és keresés. Ezek a műveletek a quadtree szerkezetét és a csomópontok régióit használják a hatékony végrehajtáshoz.

##### Beszúrás

Az adat beszúrása a quadtree-be a következő lépésekben történik:
1. **Megfelelő levélcsomópont kiválasztása:** Kezdjük a gyökércsomópontnál, és haladjunk lefelé a fa mentén, amíg el nem érjük azt a levélcsomópontot, amely a beszúrandó adat régióját tartalmazza.
2. **Levélcsomópont bővítése:** Az adatot a kiválasztott levélcsomópontba illesztjük.
3. **Csomópont felosztása:** Ha a levélcsomópont megtelik, a régiót négy részre osztjuk, és az adatokat a megfelelő gyermekcsomópontokba helyezzük át.

##### Törlés

Az adat törlése a quadtree-ből a következő lépésekben történik:
1. **Levélcsomópont megtalálása:** Kezdjük a gyökércsomópontnál, és haladjunk lefelé a fa mentén, amíg el nem érjük azt a levélcsomópontot, amely az eltávolítandó adatot tartalmazza.
2. **Adat eltávolítása:** Eltávolítjuk az adatot a levélcsomópontból.
3. **Csomópont egyesítése:** Ha a törlés után a levélcsomópontokban lévő adatok száma túl kicsi lesz, akkor a csomópontokat összevonjuk.

##### Keresés

A quadtree keresési műveletei, mint például a tartománykeresés vagy a legközelebbi szomszéd keresés, a következő lépésekben történnek:
1. **Rekurzív bejárás:** A fa rekurzív bejárása során minden olyan csomópontot ellenőrzünk, amelynek régiója átfedi a keresési régiót.
2. **Levélcsomópontok ellenőrzése:** Ha levélcsomópontot találunk, ellenőrizzük az összes benne lévő adatot, hogy megfelelnek-e a keresési feltételeknek.
3. **Gyermekcsomópontok bejárása:** Ha belső csomópontot találunk, rekurzívan bejárjuk annak gyermekcsomópontjait is.

#### Példakód C++ nyelven

Az alábbiakban bemutatunk egy egyszerű quadtree implementációt C++ nyelven:

```cpp
#include <iostream>

#include <vector>
#include <memory>

struct Point {
    float x, y;
    Point(float _x, float _y) : x(_x), y(_y) {}
};

struct Rect {
    float x, y, width, height;
    Rect(float _x, float _y, float _width, float _height)
        : x(_x), y(_y), width(_width), height(_height) {}

    bool contains(const Point& point) const {
        return (point.x >= x - width &&
                point.x <= x + width &&
                point.y >= y - height &&
                point.y <= y + height);
    }

    bool intersects(const Rect& range) const {
        return !(range.x - range.width > x + width ||
                 range.x + range.width < x - width ||
                 range.y - range.height > y + height ||
                 range.y + range.height < y - height);
    }
};

class Quadtree {
public:
    Quadtree(const Rect& boundary, int capacity)
        : boundary(boundary), capacity(capacity), divided(false) {}

    bool insert(const Point& point) {
        if (!boundary.contains(point)) {
            return false;
        }

        if (points.size() < capacity) {
            points.push_back(point);
            return true;
        } else {
            if (!divided) {
                subdivide();
            }
            if (northeast->insert(point)) return true;
            if (northwest->insert(point)) return true;
            if (southeast->insert(point)) return true;
            if (southwest->insert(point)) return true;
        }
        return false;
    }

    void query(const Rect& range, std::vector<Point>& found) const {
        if (!boundary.intersects(range)) {
            return;
        }

        for (const Point& point : points) {
            if (range.contains(point)) {
                found.push_back(point);
            }
        }

        if (divided) {
            northeast->query(range, found);
            northwest->query(range, found);
            southeast->query(range, found);
            southwest->query(range, found);
        }
    }

private:
    void subdivide() {
        float x = boundary.x;
        float y = boundary.y;
        float w = boundary.width / 2;
        float h = boundary.height / 2;

        Rect ne(x + w, y - h, w, h);
        Rect nw(x - w, y - h, w, h);
        Rect se(x + w, y + h, w, h);
        Rect sw(x - w, y + h, w, h);

        northeast = std::make_unique<Quadtree>(ne, capacity);
        northwest = std::make_unique<Quadtree>(nw, capacity);
        southeast = std::make_unique<Quadtree>(se, capacity);
        southwest = std::make_unique<Quadtree>(sw, capacity);

        divided = true;
    }

    Rect boundary;
    int capacity;
    bool divided;
    std::vector<Point> points;
    std::unique_ptr<Quadtree> northeast;
    std::unique_ptr<Quadtree> northwest;
    std::unique_ptr<Quadtree> southeast;
    std::unique_ptr<Quadtree> southwest;
};

int main() {
    Rect boundary(0, 0, 200, 200);
    Quadtree qt

(boundary, 4);

    qt.insert(Point(-50, -50));
    qt.insert(Point(50, 50));
    qt.insert(Point(-75, 75));
    qt.insert(Point(75, -75));

    Rect range(0, 0, 100, 100);
    std::vector<Point> found;
    qt.query(range, found);

    for (const Point& p : found) {
        std::cout << "Point found: (" << p.x << ", " << p.y << ")\n";
    }

    return 0;
}
```

#### Quadtrees előnyei és hátrányai

A quadtreeknek számos előnye és hátránya van, amelyeket érdemes figyelembe venni a használatuk során:

**Előnyök:**
1. **Hatékony keresés:** A quadtreek hatékonyan csökkentik a keresési időt nagy mennyiségű kétdimenziós adat esetén.
2. **Rugalmas struktúra:** A quadtreek dinamikusak, azaz az adatok beszúrása és törlése gyorsan és hatékonyan végrehajtható.
3. **Egyszerű implementáció:** A quadtreek viszonylag egyszerűen implementálhatók és használhatók.

**Hátrányok:**
1. **Egyenlőtlen eloszlás kezelése:** A quadtreek nem mindig működnek jól egyenlőtlenül eloszlott adatok esetén, mivel a felosztás nem biztosít egyenletes eloszlást a csomópontok között.
2. **Magasabb dimenziókban kevésbé hatékony:** Bár a quadtreek hatékonyak kétdimenziós adatok esetén, magasabb dimenziókban (például három vagy több dimenzió) a hatékonyságuk csökken.
3. **Túlfinomított felosztás:** Ha az adatok sűrűsége magas, a quadtreek túlfinomított felosztásokhoz vezethetnek, amelyek növelhetik a fa méretét és a műveletek bonyolultságát.

Összefoglalva, a quadtreek hatékony és rugalmas adatszerkezetek kétdimenziós térbeli adatok tárolására és keresésére. Azonban fontos mérlegelni a hátrányokat és az alkalmazás specifikus igényeit, hogy optimálisan használjuk őket.

