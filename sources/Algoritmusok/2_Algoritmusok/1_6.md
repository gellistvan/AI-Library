\newpage

## 1.6. Kettős ugrásos keresés (Jump and Skip Search)

A kettős ugrásos keresés (Jump and Skip Search) egy hatékony adatkeresési technika, amely az alapvető ugrásos keresési módszerek továbbfejlesztésén alapul. Ebben a fejezetben bemutatjuk, hogyan kombinálhatók különböző ugrásos keresési technikák annak érdekében, hogy gyorsabb és hatékonyabb adatkeresést érjünk el különböző adatszerkezetekben. Ezen túlmenően részletesen elemezzük ezen módszerek teljesítményét és komplexitását, hogy átfogó képet nyújtsunk alkalmazhatóságukról és hatékonyságukról. A fejezet célja, hogy az olvasók megértsék a kettős ugrásos keresési algoritmusok alapelveit, valamint azok előnyeit és korlátait különböző szituációkban.

### 1.6.1. Kombinált ugrásos keresési technikák

A kombinált ugrásos keresési technikák olyan algoritmusok, amelyek az ugrásos keresési módszerek különböző változatait integrálják annak érdekében, hogy javítsák a keresés hatékonyságát és csökkentsék a keresési időt. Az alapvető ugrásos keresési technika, közismert nevén Jump Search, úgy működik, hogy az elemeket egy adott lépésközzel, azaz "ugrással" vizsgálja. Amennyiben az aktuális ugrási pontban található elem nagyobb, mint a keresett érték, az algoritmus visszatér és lineárisan keres a korábbi ugrási pontok között. Ez a módszer hatékonyabb a lineáris keresésnél, különösen nagy adathalmazok esetén.

Azonban a Jump Search hatékonysága tovább növelhető kombinálásával más keresési technikákkal, például a Skip Search-sel. A Skip Search lényege, hogy az adathalmazon belüli ugrások közötti intervallumokban speciális "skip list" struktúrákat alkalmazunk, amelyek lehetővé teszik az elemek gyorsabb elérését és az ugrások optimalizálását. Ezek a kombinált technikák a különböző keresési módszerek előnyeit egyesítik, és adaptív módon alkalmazkodnak az adathalmaz struktúrájához és méretéhez.

#### Elméleti alapok

A kombinált ugrásos keresési technikák elméleti hátterének megértése érdekében először vizsgáljuk meg az alapvető ugrásos keresési algoritmusokat, majd térjünk át a kombinációs módszerek részleteire.

##### Jump Search

A Jump Search algoritmus az elemeket egy előre meghatározott lépésközönként vizsgálja, amelyet általában a következőképpen választanak meg:

$m = \sqrt{n}$

ahol $n$ az elemek száma az adathalmazban, és $m$ az ugrási lépésköz. Az algoritmus lépései a következők:

1. Az elemeket $m$ lépésközönként vizsgáljuk, amíg meg nem találjuk az első olyan elemet, amely nagyobb vagy egyenlő a keresett értéknél.
2. Visszatérünk az előző ugrási pontra, és lineárisan keresünk az elemek között, amíg meg nem találjuk a keresett értéket vagy el nem érjük a következő ugrási pontot.

A Jump Search előnye, hogy jelentősen csökkenti a keresési lépések számát, különösen nagy adathalmazok esetén, mivel az ugrások gyorsan szűkítik a keresési intervallumot.

##### Skip Search

A Skip Search alapja egy speciális adatszerkezet, a skip list, amely lényegében egy hierarchikus láncolt lista. Minden elemhez több szintű pointerek tartoznak, amelyek lehetővé teszik az elemek gyors átlépését. Ez a szerkezet lehetővé teszi az elemek gyors keresését az alábbi módon:

1. Az elemeket egy alap láncolt listába rendezzük.
2. Különböző szinteken további pointereket adunk az elemekhez, amelyek lehetővé teszik az elemek közötti gyors ugrást.
3. A keresés során először a legmagasabb szintű pointereket használjuk az elemek közötti ugráshoz, majd fokozatosan lejjebb lépünk az alacsonyabb szintekre, amíg meg nem találjuk a keresett elemet.

A Skip Search előnye, hogy még nagyobb adathalmazok esetén is gyors keresést biztosít, mivel a skip list szerkezete lehetővé teszi az elemek közötti gyors navigációt.

#### Kombinált Technika

A kombinált ugrásos keresési technikák célja, hogy az alapvető Jump Search és Skip Search módszereket egyesítve optimalizálják a keresési folyamatot. A kombináció előnyei abban rejlenek, hogy az egyes technikák gyengeségeit ellensúlyozzák a másik módszer erősségei.

##### Algoritmus

A kombinált ugrásos keresési algoritmus a következő lépéseket tartalmazza:

1. **Előkészítés:**
    - Az adathalmazt rendezzük és egy skip list struktúrába szervezzük.
    - Meghatározzuk az ugrási lépésközt $m$ a Jump Search számára.

2. **Ugrási szakasz:**
    - A skip list legmagasabb szintjét használva ugrunk $m$ lépésközökkel az adathalmazban.
    - Ha az aktuális elem nagyobb vagy egyenlő a keresett értéknél, áttérünk a következő szakaszra.

3. **Finomított keresés:**
    - Az előző ugrási ponttól visszatérve, a skip list alacsonyabb szintjeit használva lineárisan keresünk az elemek között, amíg meg nem találjuk a keresett értéket vagy el nem érjük a következő ugrási pontot.

4. **Adaptív lépésszám:**
    - Az ugrási lépésközt dinamikusan módosítjuk az adathalmaz méretének és struktúrájának megfelelően, hogy optimalizáljuk a keresési időt.

##### Pseudo-kód

Az alábbiakban egy egyszerű pseudo-kód bemutatja a kombinált ugrásos keresési algoritmus működését:

```cpp
#include <iostream>
#include <vector>
#include <cmath>

struct Node {
    int value;
    Node* next;
    Node* skip;
};

Node* createSkipList(const std::vector<int>& data) {
    Node* head = new Node{data[0], nullptr, nullptr};
    Node* current = head;
    Node* skipNode = head;
    int skipDistance = sqrt(data.size());

    for (size_t i = 1; i < data.size(); ++i) {
        current->next = new Node{data[i], nullptr, nullptr};
        current = current->next;
        if (i % skipDistance == 0) {
            skipNode->skip = current;
            skipNode = current;
        }
    }
    return head;
}

Node* combinedJumpSkipSearch(Node* head, int target) {
    Node* current = head;
    Node* prev = nullptr;

    while (current && current->value < target) {
        prev = current;
        if (current->skip && current->skip->value <= target) {
            current = current->skip;
        } else {
            current = current->next;
        }
    }

    return (current && current->value == target) ? current : nullptr;
}

int main() {
    std::vector<int> data = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    Node* head = createSkipList(data);
    int target = 7;
    Node* result = combinedJumpSkipSearch(head, target);
    
    if (result) {
        std::cout << "Element " << target << " found." << std::endl;
    } else {
        std::cout << "Element " << target << " not found." << std::endl;
    }

    // Cleanup memory (omitted for brevity)
    return 0;
}
```

##### Teljesítmény elemzése

A kombinált ugrásos keresési technikák teljesítményét több tényező befolyásolja, mint például az adathalmaz mérete, az adatszerkezet elrendezése és az ugrási lépésköz megválasztása. Az alábbiakban részletezzük ezen tényezők hatását:

1. **Adathalmaz mérete:** Ahogy az adathalmaz mérete növekszik, az ugrási lépésköz optimális megválasztása egyre fontosabbá válik. A négyzetgyök alapú ugrási lépésköz általában jól működik, de bizonyos esetekben finomhangolásra lehet szükség.
2. **Adatszerkezet elrendezése:** A skip list szerkezete lehetővé teszi az elemek gyors átlépését, ami különösen hasznos, ha az adatok egyenletesen vannak elosztva. Ha az adatok nem egyenletesen oszlanak el, az ugrási lépések hatékonysága csökkenhet.
3. **Ugrási lépésköz:** A dinamikusan változó ugrási lépésköz adaptálható az adathalmaz jellemzőihez, ami javíthatja a keresés hatékonyságát. Az adaptív algoritmusok gyakran jobb teljesítményt nyújtanak, mivel alkalmazkodnak az adathalmaz struktúrájához és méretéhez.

A kombinált ugrásos keresési technikák hatékonyak és rugalmasak, különösen nagy és összetett adathalmazok esetén. Az ilyen algoritmusok alkalmazása jelentősen csökkentheti a keresési időt és növelheti a keresés pontosságát, ezáltal fontos eszközt kínálva a modern adatelemzés és információkezelés területén.

### 1.6.2. Teljesítmény és komplexitás elemzése

A kettős ugrásos keresési technikák teljesítményének és komplexitásának elemzése elengedhetetlen ahhoz, hogy megértsük ezen algoritmusok hatékonyságát különböző adatszerkezetekben és alkalmazási környezetekben. Ebben az alfejezetben részletesen tárgyaljuk az idő- és tárhely-komplexitást, a legjobb, legrosszabb és átlagos eseteket, valamint a gyakorlati teljesítmény értékelésének módszereit.

#### Időkomplexitás

A keresési algoritmusok időkomplexitása a legfontosabb szempont, amikor a hatékonyságukat vizsgáljuk. A kombinált ugrásos keresési technikák esetében az időkomplexitás különböző komponensekből tevődik össze:

##### Jump Search Időkomplexitása

A Jump Search algoritmus időkomplexitása az adathalmaz méretének ($n$) függvényében a következőképpen alakul:

1. **Ugrási lépések száma:** Az elemeket $\sqrt{n}$ lépésközönként vizsgáljuk, így az ugrási lépések száma $\sqrt{n}$.
2. **Lineáris keresés:** Miután megtaláltuk az első olyan elemet, amely nagyobb vagy egyenlő a keresett értéknél, visszatérünk és lineárisan keresünk az elemek között. Ebben az esetben a legrosszabb esetben $\sqrt{n}$ lépést kell megtenni.

Ennek eredményeként a Jump Search algoritmus teljes időkomplexitása a legrosszabb esetben:

$O(\sqrt{n} + \sqrt{n}) = O(\sqrt{n})$

##### Skip Search Időkomplexitása

A Skip Search algoritmus időkomplexitása a skip list struktúrájának köszönhetően másképp alakul:

1. **Skip lista szintjei:** A skip lista $\log n$ szinttel rendelkezik, mivel minden szint a korábbi szint elemeinek felét tartalmazza.
2. **Szintek közötti ugrások:** A keresési művelet során a legmagasabb szinttől indulunk, és fokozatosan lejjebb lépünk a szinteken, ami $O(\log n)$ lépést jelent.
3. **Lineáris keresés:** Az egyes szinteken belüli keresés lineáris időben történik, de mivel a szintek száma korlátozott, ez az összkomplexitást nem növeli jelentősen.

Ennek eredményeként a Skip Search algoritmus teljes időkomplexitása:

$O(\log n)$

##### Kombinált Ugrásos Keresés Időkomplexitása

A kombinált ugrásos keresési technika az előző két módszer elemeit integrálja. Az időkomplexitás a következőképpen alakul:

1. **Ugrási szakasz:** Az algoritmus a skip lista legmagasabb szintjét használva ugrik $m$ lépésközökkel. Az optimális $m$ értéke $\sqrt{n}$.
2. **Finomított keresés:** Az ugrási szakasz után az alacsonyabb szinteken lineárisan keresünk az elemek között.

Ez alapján a kombinált ugrásos keresési technika teljes időkomplexitása a legrosszabb esetben:

$O(\sqrt{n} + \log n)$

Az átlagos esetben az időkomplexitás általában alacsonyabb, mivel az algoritmus gyorsan szűkíti a keresési intervallumot az ugrási és skip műveletek kombinációjával.

#### Tárhely-komplexitás

A tárhely-komplexitás szintén fontos szempont a keresési algoritmusok értékelésénél. A kombinált ugrásos keresési technikák tárhelyigénye a skip list struktúra miatt változik.

##### Skip Lista Tárhely-komplexitása

A skip lista tárhely-komplexitása a következő tényezőkből áll:

1. **Alap lista:** Az alap lista $n$ elemet tartalmaz, ahol $n$ az adathalmaz mérete.
2. **Pointerek:** Minden elem több szintű pointerekkel rendelkezik. Átlagosan minden elem $\log n$ pointerrel rendelkezik.

Ez alapján a skip lista tárhely-komplexitása:

$O(n \log n)$

##### Kombinált Ugrásos Keresés Tárhely-komplexitása

A kombinált ugrásos keresési technika tárhely-komplexitása a skip lista tárhelyigényével megegyezik, mivel az alapvető Jump Search módszer nem igényel további tárhelyet.

#### Legjobb, legrosszabb és átlagos esetek

A kombinált ugrásos keresési technikák teljesítménye a különböző esetekben eltérő lehet.

##### Legjobb eset

A legjobb eset akkor fordul elő, ha a keresett elem az első ugrási pontban található. Ebben az esetben az algoritmus csak néhány ugrási lépést hajt végre, és az időkomplexitás:

$O(1)$

##### Legrosszabb eset

A legrosszabb eset akkor fordul elő, ha a keresett elem az adathalmaz végén található, vagy egyáltalán nem található meg. Ebben az esetben az algoritmus végigmegy az összes ugrási és lineáris keresési lépésen. Az időkomplexitás:

$O(\sqrt{n} + \log n)$

##### Átlagos eset

Az átlagos esetben az időkomplexitás az adathalmaz méretének és az elemek eloszlásának függvényében változik. Az átlagos esetben az időkomplexitás általában a legrosszabb és a legjobb esetek közötti intervallumban van, jellemzően:

$O(\log n)$

#### Gyakorlati teljesítmény értékelése

A kombinált ugrásos keresési technikák gyakorlati teljesítményét többféle módon értékelhetjük:

1. **Empirikus tesztelés:** Az algoritmust különböző méretű és elrendezésű adathalmazokon tesztelhetjük, és mérhetjük a futási időt és a memóriakihasználtságot.
2. **Szintetikus adatok:** Szintetikus adatok használata lehetővé teszi az algoritmus viselkedésének vizsgálatát különböző extrém esetekben, például nagyon egyenletes vagy nagyon szórt adatok esetén.
3. **Valós adatok:** Valós adathalmazok használata segít megérteni az algoritmus teljesítményét valós környezetekben és alkalmazásokban.

##### Példa C++ implementációval

Az alábbi példa egy kombinált ugrásos keresési algoritmust mutat be C++ nyelven:

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// Node structure for the skip list
struct Node {
    int value;
    Node* next;
    Node* skip;
};

// Function to create a skip list from a sorted vector
Node* createSkipList(const std::vector<int>& data) {
    Node* head = new Node{data[0], nullptr, nullptr};
    Node* current = head;
    Node* skipNode = head;
    int skipDistance = sqrt(data.size());

    for (size_t i = 1; i < data.size(); ++i) {
        current->next = new Node{data[i], nullptr, nullptr};
        current = current->next;
        if (i % skipDistance == 0) {
            skipNode->skip = current;
            skipNode = current;
        }
    }
    return head;
}

// Combined jump and skip search algorithm
Node* combinedJumpSkipSearch(Node* head, int target) {
    Node* current = head;
    Node* prev = nullptr;

    while (current && current->value < target) {
        prev = current;
        if (current->skip && current->skip->value <= target) {
            current = current->skip;
        } else {
            current = current->next;
        }
    }

    return (current && current->value == target) ? current : nullptr;
}

int main() {
    std::vector<int> data = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    Node* head = createSkipList(data);
    int target = 7;

    auto start = std::chrono::high_resolution_clock::now();
    Node* result = combinedJumpSkipSearch(head, target);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> duration = end - start;
    
    if (result) {
        std::cout << "Element " << target << " found." << std::endl;
    } else {
        std::cout << "Element " << target << " not found." << std::endl;
    }

    std::cout << "Search completed in " << duration.count() << " seconds." << std::endl;

    // Cleanup memory (omitted for brevity)
    return 0;
}
```

Ez a példa bemutatja, hogyan használhatjuk a kombinált ugrásos keresési technikát egy egyszerű C++ programban. Az időmérési részlet lehetővé teszi a keresési művelet futási idejének mérését, ami segít a gyakorlati teljesítmény értékelésében.

#### Összefoglalás

A kombinált ugrásos keresési technikák hatékony és rugalmas megoldást kínálnak a nagy és összetett adathalmazok keresési feladataira. Az idő- és tárhely-komplexitás részletes elemzése, valamint az empirikus teljesítményértékelés alapján megállapítható, hogy ezek a technikák jelentős előnyöket nyújtanak a hagyományos keresési módszerekhez képest. Az adaptív algoritmusok, amelyek kombinálják a Jump Search és Skip Search módszereit, lehetővé teszik a keresési folyamat optimalizálását az adott adathalmaz jellemzőihez igazodva, ezáltal növelve a keresés hatékonyságát és pontosságát.

