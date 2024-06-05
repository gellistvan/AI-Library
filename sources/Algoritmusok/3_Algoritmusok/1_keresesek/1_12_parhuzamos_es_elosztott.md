\newpage

## 1.12.   Párhuzamos és elosztott keresési algoritmusok

A modern számítástechnika világában a hatalmas adatmennyiségek gyors és hatékony feldolgozása kulcsfontosságú. Ahogy az adatbázisok mérete növekszik és a valós idejű feldolgozási igények nőnek, egyre nagyobb szükség van olyan keresési algoritmusokra, amelyek képesek kihasználni a párhuzamos és elosztott rendszerek nyújtotta lehetőségeket. Ebben a fejezetben két fő témakört vizsgálunk: először a párhuzamos rendszerekben történő keresést, ahol több feldolgozó egység egyidejűleg dolgozik egy probléma megoldásán, majd az elosztott keresési algoritmusokat, amelyek különböző hálózati csomópontok között osztják el a feldolgozási feladatokat. Ezek az algoritmusok nemcsak növelik a keresés hatékonyságát, hanem javítják a rendszer megbízhatóságát és skálázhatóságát is.

### 1.12.1. Keresés párhuzamos rendszerekben

A párhuzamos keresési algoritmusok a modern számítástechnika egyik alapvető kutatási területét képezik, mivel lehetővé teszik nagy adatmennyiségek gyors és hatékony feldolgozását. A párhuzamos rendszerekben történő keresés során a keresési feladat több feldolgozó egység között kerül felosztásra, ami lehetővé teszi a munkaterhelés egyenletes elosztását és a keresési folyamat gyorsítását. Ebben a fejezetben részletesen tárgyaljuk a párhuzamos keresési algoritmusok különböző típusait, azok implementációs kihívásait és előnyeit, valamint gyakorlati alkalmazásaikat.

#### Párhuzamos keresési algoritmusok típusai

A párhuzamos keresési algoritmusok két fő típusba sorolhatók: adatparalelizmus és feladatparalelizmus.

1. **Adatparalelizmus**: Az adatparalelizmus esetén a teljes adatkészlet kisebb részekre kerül felosztásra, és minden részhalmaz párhuzamosan kerül feldolgozásra. Ez a megközelítés különösen hasznos, amikor nagy adatkészletekkel dolgozunk, mint például adatbázisok vagy big data rendszerek esetében.

2. **Feladatparalelizmus**: A feladatparalelizmus során a keresési feladat különböző lépései vagy fázisai párhuzamosan kerülnek végrehajtásra. Ez a megközelítés akkor hasznos, ha a keresési folyamat különböző lépései függetlenek egymástól, és így egyszerre végrehajthatók.

#### Párhuzamos keresési algoritmusok implementációs kihívásai

A párhuzamos keresési algoritmusok implementálása számos kihívással jár, amelyek közül néhány a következő:

1. **Szinkronizáció és versenyhelyzetek**: A párhuzamos feldolgozás során gyakran szükség van a különböző feldolgozó egységek közötti szinkronizációra. A versenyhelyzetek elkerülése érdekében gondoskodni kell arról, hogy az egyes egységek ne módosítsák egyszerre ugyanazt az adatot.

2. **Terheléselosztás**: A hatékony párhuzamos keresés érdekében fontos a feldolgozási feladatok egyenletes elosztása a rendelkezésre álló feldolgozó egységek között. A nem megfelelő terheléselosztás egyik feldolgozó egységet túlterhelheti, míg a többi egység kihasználatlan marad.

3. **Kommunikációs költségek**: A párhuzamos keresési algoritmusoknál figyelembe kell venni a feldolgozó egységek közötti kommunikáció költségeit is. A túl gyakori kommunikáció jelentősen megnövelheti az algoritmus végrehajtási idejét.

#### Párhuzamos keresési algoritmusok előnyei

A párhuzamos keresési algoritmusok számos előnnyel járnak, amelyek közül néhány:

1. **Nagyobb teljesítmény**: A párhuzamos keresés lehetővé teszi a nagy adatkészletek gyorsabb feldolgozását, mivel több feldolgozó egység egyidejűleg dolgozik a feladaton.

2. **Skálázhatóság**: A párhuzamos keresési algoritmusok jól skálázhatók, mivel a feldolgozó egységek számának növelésével arányosan növelhető a teljesítmény is.

3. **Robusztusság**: A párhuzamos rendszerek általában robusztusabbak, mivel egy-egy feldolgozó egység meghibásodása esetén a többi egység továbbra is folytatni tudja a munkát.

#### Párhuzamos keresési algoritmusok gyakorlati alkalmazásai

A párhuzamos keresési algoritmusokat számos területen alkalmazzák, beleértve az adatbázisok kezelését, a webes keresőmotorokat, a genomikai kutatást és a mesterséges intelligencia rendszereket. Az alábbiakban bemutatunk néhány konkrét példát.

1. **Adatbázisok**: A nagy méretű adatbázisokban történő keresés során gyakran alkalmaznak párhuzamos algoritmusokat a lekérdezések gyorsítása érdekében. Az adatbázisok különböző részeire vonatkozó lekérdezések párhuzamosan futtathatók, ami jelentősen csökkenti a válaszidőt.

2. **Webes keresőmotorok**: A webes keresőmotorok hatalmas mennyiségű adatot dolgoznak fel, és a párhuzamos keresési algoritmusok alkalmazása lehetővé teszi a keresési eredmények gyorsabb előállítását. A keresési indexek párhuzamos feldolgozása révén a keresőmotorok nagy sebességgel képesek releváns találatokat szolgáltatni.

3. **Genomikai kutatás**: A genomikai adatok elemzése során párhuzamos algoritmusokkal gyorsítják fel a szekvenciák összehasonlítását és azonosítását. Ez különösen fontos a nagy genomikai adatbázisok kezelése és a genetikai kutatások előmozdítása szempontjából.

4. **Mesterséges intelligencia**: A gépi tanulási algoritmusok és a neurális hálózatok tanításához gyakran használnak párhuzamos keresési technikákat. Ezek a technikák lehetővé teszik a modellek gyorsabb és hatékonyabb tanítását nagy adatkészletek felhasználásával.

#### Példakód párhuzamos kereséshez C++ nyelven

Az alábbiakban egy egyszerű példakódot mutatunk be C++ nyelven, amely egy párhuzamos keresési algoritmust valósít meg egy tömbben. A példában a párhuzamos keresést a C++ Standard Library `thread` osztályával valósítjuk meg.

```cpp
#include <iostream>

#include <vector>
#include <thread>

#include <mutex>

std::mutex mtx; // mutex for critical section
bool found = false; // flag to indicate if the element is found

void parallelSearch(const std::vector<int>& data, int target, int start, int end) {
    for (int i = start; i < end && !found; ++i) {
        if (data[i] == target) {
            std::lock_guard<std::mutex> lock(mtx);
            found = true;
            std::cout << "Element found at index " << i << std::endl;
            return;
        }
    }
}

int main() {
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int target = 7;
    int numThreads = 4;
    int dataSize = data.size();
    int blockSize = dataSize / numThreads;

    std::vector<std::thread> threads;

    for (int i = 0; i < numThreads; ++i) {
        int start = i * blockSize;
        int end = (i == numThreads - 1) ? dataSize : start + blockSize;
        threads.emplace_back(parallelSearch, std::ref(data), target, start, end);
    }

    for (auto& th : threads) {
        th.join();
    }

    if (!found) {
        std::cout << "Element not found" << std::endl;
    }

    return 0;
}
```

Ez a kód egy egyszerű példát mutat arra, hogyan lehet párhuzamos keresést végrehajtani egy tömbben több szál segítségével. A `parallelSearch` függvény különböző szálakon párhuzamosan fut, és az adott részhalmazban keresi a célelem előfordulását. A szálak szinkronizációjához mutexet használunk, hogy elkerüljük a versenyhelyzeteket.

#### Összefoglalás

A párhuzamos keresési algoritmusok alapvető fontosságúak a modern számítástechnika számos területén. Az adatparalelizmus és a feladatparalelizmus különböző megközelítései lehetővé teszik a nagy adatmennyiségek hatékony és gyors feldolgozását.

Az implementáció során figyelembe kell venni a szinkronizáció és a terheléselosztás kihívásait, valamint a kommunikációs költségeket. A párhuzamos keresési algoritmusok előnyei közé tartozik a nagyobb teljesítmény, a skálázhatóság és a rendszer robusztussága. Számos gyakorlati alkalmazásban használják őket, beleértve az adatbázisok kezelését, a webes keresőmotorokat, a genomikai kutatást és a mesterséges intelligenciát.

### 1.12.2. Elosztott keresési algoritmusok

Az elosztott keresési algoritmusok kulcsfontosságú szerepet játszanak a nagy méretű és széleskörűen elosztott rendszerek hatékony adatfeldolgozásában. Az ilyen rendszerekben az adatokat nem egy központi helyen tárolják, hanem több, gyakran földrajzilag is elkülönített csomópont között osztják el. Az elosztott keresési algoritmusok célja, hogy biztosítsák az adatok gyors és megbízható elérését ebben a heterogén és dinamikus környezetben. Ebben a fejezetben részletesen bemutatjuk az elosztott keresési algoritmusok különböző típusait, az implementálási kihívásokat és előnyöket, valamint a gyakorlati alkalmazásaikat.

#### Elosztott keresési algoritmusok típusai

Az elosztott keresési algoritmusok több kategóriába sorolhatók, a legfontosabbak a következők:

1. **Flooding algoritmusok**: Az egyik legegyszerűbb elosztott keresési módszer, ahol a keresési kérést minden szomszédos csomópontnak továbbítják. Bár egyszerű, a flooding algoritmusok jelentős hálózati forgalmat generálhatnak és nem skálázhatók jól.

2. **Random Walk algoritmusok**: A keresési kérés véletlenszerűen választott útvonalon halad a hálózatban. Ez a megközelítés csökkenti a hálózati forgalmat, de a keresés hatékonysága alacsonyabb lehet.

3. **Hierarchikus keresési algoritmusok**: Ezek az algoritmusok hierarchikus struktúrát használnak, ahol a csomópontok különböző szintekre vannak osztva. A keresési kérés először a magasabb szintű csomópontokra irányul, majd fokozatosan halad a cél felé.

4. **DHT (Distributed Hash Table) alapú algoritmusok**: A DHT algoritmusok, mint például a Chord, Pastry, Tapestry, és Kademlia, elosztott hash táblákat használnak az adatok indexelésére és keresésére. Ezek az algoritmusok hatékony és skálázható módot kínálnak az adatok elérésére az elosztott rendszerekben.

#### Elosztott keresési algoritmusok implementációs kihívásai

Az elosztott keresési algoritmusok implementálása számos kihívással jár, amelyek közül néhány a következő:

1. **Hálózati megbízhatóság**: Az elosztott rendszerekben a hálózati kapcsolatok gyakran megbízhatatlanok, és a csomópontok kieshetnek vagy új csomópontok csatlakozhatnak a hálózathoz. Az algoritmusoknak képesnek kell lenniük kezelni ezeket a dinamikus változásokat.

2. **Skálázhatóság**: Az elosztott rendszerek mérete jelentősen változhat, így az algoritmusoknak képesnek kell lenniük hatékonyan működni mind kis, mind nagy méretű hálózatokban.

3. **Terheléselosztás**: Az egyenletes terheléselosztás kritikus fontosságú az elosztott rendszerekben, hogy elkerüljék a hálózat egyes részeinek túlterhelését és biztosítsák a rendszer hatékonyságát.

4. **Hálózati forgalom optimalizálása**: Az elosztott keresési algoritmusoknak minimalizálniuk kell a hálózati forgalmat, hogy csökkentsék a hálózati erőforrások igénybevételét és javítsák a válaszidőt.

#### Elosztott keresési algoritmusok előnyei

Az elosztott keresési algoritmusok számos előnnyel járnak, amelyek közül néhány:

1. **Hibatűrés**: Az elosztott rendszerek hibatűrőbbek, mivel az adatokat több csomóponton tárolják. Egy csomópont kiesése esetén az adatok továbbra is elérhetők más csomópontokról.

2. **Skálázhatóság**: Az elosztott keresési algoritmusok jól skálázhatók, mivel a csomópontok számának növekedésével arányosan növelhető a rendszer teljesítménye.

3. **Adatelérés gyorsítása**: Az adatok több csomóponton történő tárolása lehetővé teszi az adatokhoz való gyorsabb hozzáférést, mivel a keresési kérés a legközelebbi csomópontra irányulhat.

4. **Terheléselosztás**: Az elosztott rendszerekben az adatfeldolgozás terhe több csomópont között oszlik meg, ami lehetővé teszi a nagyobb terhelés hatékonyabb kezelését.

#### Elosztott keresési algoritmusok gyakorlati alkalmazásai

Az elosztott keresési algoritmusokat számos területen alkalmazzák, beleértve a peer-to-peer hálózatokat, a felhőalapú szolgáltatásokat, a nagy adatbázisokat és a tartalomszolgáltatási hálózatokat. Az alábbiakban bemutatunk néhány konkrét példát.

1. **Peer-to-peer hálózatok**: Az elosztott keresési algoritmusokat széles körben használják a peer-to-peer hálózatokban, mint például a fájlmegosztó rendszerekben (pl. BitTorrent). Ezekben a hálózatokban az adatokat a hálózatban lévő csomópontok között osztják el, és a keresési algoritmusok biztosítják az adatok gyors és hatékony elérését.

2. **Felhőalapú szolgáltatások**: A felhőalapú szolgáltatásokban az adatokat gyakran elosztott adatközpontokban tárolják. Az elosztott keresési algoritmusok lehetővé teszik az adatok gyors elérését és a szolgáltatások megbízhatóságának növelését.

3. **Nagy adatbázisok**: A nagy méretű adatbázisok kezelésére elosztott keresési algoritmusokat használnak, amelyek lehetővé teszik az adatok hatékony indexelését és keresését. Az ilyen adatbázisok gyakran több szerveren vagy adatközpontban helyezkednek el.

4. **Tartalomszolgáltatási hálózatok (CDN-ek)**: A CDN-ek elosztott szerverhálózatokat használnak a tartalmak gyorsabb elérése érdekében. Az elosztott keresési algoritmusok segítenek optimalizálni a tartalmak elosztását és hozzáférhetőségét a felhasználók számára.

#### Példakód elosztott kereséshez C++ nyelven

Az alábbiakban egy egyszerű példakódot mutatunk be C++ nyelven, amely egy elosztott hash tábla (DHT) alapú keresési algoritmust valósít meg. Ebben a példában a Chord algoritmust használjuk.

```cpp
#include <iostream>

#include <map>
#include <vector>

#include <cmath>
#include <mutex>

#include <thread>

class Node {
public:
    int id;
    std::map<int, int> data; // key-value pairs
    Node* successor;

    Node(int id) : id(id), successor(this) {}

    void join(Node* existingNode) {
        if (existingNode) {
            successor = existingNode->findSuccessor(id);
        }
    }

    Node* findSuccessor(int key) {
        if (key > id && key <= successor->id) {
            return successor;
        } else {
            return successor->findSuccessor(key);
        }
    }

    void store(int key, int value) {
        Node* node = findSuccessor(key);
        node->data[key] = value;
    }

    int retrieve(int key) {
        Node* node = findSuccessor(key);
        return node->data[key];
    }
};

class Chord {
public:
    std::vector<Node*> nodes;

    Chord(int numNodes) {
        for (int i = 0; i < numNodes; ++i) {
            nodes.push_back(new Node(i));
        }
        for (int i = 1; i < numNodes; ++i) {
            nodes[i]->join(nodes[0]);
        }
    }

    void store(int key, int value) {
        nodes[0]->store(key, value);
    }

    int retrieve(int key) {
        return nodes[0]->retrieve(key);
    }

    ~Chord() {
        for (Node* node : nodes) {
            delete node;
        }
    }
};

int main() {
    Chord chord(10); // Create a Chord ring with 10 nodes

    chord.store(5, 100);
    chord.store(15, 200);

    std::cout << "Value at key 5: " << chord.retrieve(5) << std::endl;
    std::cout << "Value at key 15: " << chord.retrieve(15) << std::endl;

    return 0;
}
```

Ez a kód egy egyszerű Chord algoritmus implementációt mutat be, amely elosztott hash táblát valósít meg. A `Node` osztály reprezentál egy csomópontot a Chord gyűrűben, amely adatokat tárol és keresési műveleteket végez. A `Chord` osztály létrehozza a Chord gyűrűt és biztosítja az adatok tárolását és lekérdezését.

#### Összefoglalás

Az elosztott keresési algoritmusok alapvető fontosságúak a modern elosztott rendszerek hatékony működéséhez. Különböző típusú algoritmusok léteznek, mint például a flooding, random walk, hierarchikus és DHT alapú algoritmusok, amelyek mindegyike különböző előnyöket és kihívásokat kínál. Az implementáció során figyelembe kell venni a hálózati megbízhatóságot, a skálázhatóságot, a terheléselosztást és a hálózati forgalom optimalizálását. Az elosztott keresési algoritmusok előnyei közé tartozik a hibatűrés, a skálázhatóság, az adatelérés gyorsítása és a terheléselosztás. Számos gyakorlati alkalmazásban használják őket, beleértve a peer-to-peer hálózatokat, a felhőalapú szolgáltatásokat, a nagy adatbázisokat és a tartalomszolgáltatási hálózatokat.

