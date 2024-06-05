\newpage

## 6. Trie

Az adatszerkezetek világában a trie, vagy más néven prefixfa, egy különösen hasznos és hatékony eszköz a sztringek kezelésében és keresésében. A trie lehetővé teszi a gyors keresést, beszúrást és törlést, ami különösen fontos, amikor nagy mennyiségű szöveges adatot kell kezelni. Ebben a fejezetben bemutatjuk a trie alapfogalmait és tulajdonságait, megismerkedünk a legfontosabb műveletekkel, mint a beszúrás, keresés és törlés, valamint megvizsgáljuk a tömörített trie-t és annak különböző implementációit. Végül bemutatjuk a trie gyakorlati alkalmazásait, például a szöveges keresést és az auto-complete funkciót, amelyek mindennapi életünkben is fontos szerepet játszanak.

### 6.1. Alapfogalmak és tulajdonságok

A trie adatszerkezet, más néven prefixfa, a sztringek hatékony kezelésére szolgál. Az adatszerkezet a sztringek közös prefixeit használja fel a redundancia csökkentésére és a műveletek gyorsítására. Ebben az alfejezetben részletesen tárgyaljuk a trie alapvető fogalmait, tulajdonságait, valamint a mögöttes elméleti hátteret.

#### Trie Szerkezete

A trie egy faalapú adatszerkezet, amelyben minden csomópont egy adott karaktert reprezentál. A sztringek karakterei a gyökértől a levelekig vezető útvonalon helyezkednek el. Minden útvonal egy-egy sztringet képvisel, és a sztringek közös prefixei közös útvonalat osztanak meg.

#### Alapfogalmak

- **Gyökér (Root):** A trie gyökércsomópontja nem tartalmaz karaktert, de minden sztring ebből a pontból indul ki.
- **Csomópont (Node):** A trie csomópontjai egy-egy karaktert reprezentálnak. Minden csomópontnak lehet több gyermeke, amelyek az adott karakter után következő karaktereket reprezentálják.
- **Levél (Leaf):** A trie levelei azok a csomópontok, amelyek egy teljes sztring végét jelölik.
- **Élek (Edges):** Az élek a csomópontok közötti kapcsolatokat reprezentálják. Minden él egy karakterrel van címkézve.
- **Útvonal (Path):** Az útvonal a gyökértől egy csomóponton keresztül a levélig vezető sorozat.

#### Trie Tulajdonságai

1. **Hierarchikus Szerkezet:** A trie hierarchikus szerkezetű, amely lehetővé teszi a sztringek közös prefixeinek hatékony kezelését. Az azonos prefixekkel rendelkező sztringek közös útvonalon osztoznak a fában.
2. **Gyors Keresés:** A trie lehetővé teszi a sztringek gyors keresését, mivel a keresési idő a sztring hosszától függ, és független a trie-ben tárolt sztringek számától.
3. **Prefix Keresés:** A trie különösen hatékony a prefixek keresésében, mivel a keresési művelet során a közös prefixeket egyetlen útvonalon követjük.
4. **Memóriaigény:** A trie memóriaigénye nagy lehet, különösen akkor, ha sok rövid sztringet tárolunk, mivel minden karakter külön csomópontban van tárolva. Azonban a közös prefixek megosztása révén a memóriaigény optimalizálható.

#### Trie Működése

##### Beszúrás (Insertion)

A beszúrási művelet során egy új sztringet adunk hozzá a trie-hez. Az algoritmus a következő lépésekből áll:

1. Kezdjük a gyökércsomópontnál.
2. Az új sztring minden egyes karakterére:
    - Ha a karakterrel címkézett él létezik a jelenlegi csomópontban, kövessük az élt a következő csomópontra.
    - Ha az él nem létezik, hozzunk létre egy új csomópontot a karakterrel, és kapcsoljuk össze a jelenlegi csomóponttal egy új éllel.
3. A sztring végén jelöljük meg a jelenlegi csomópontot levélként, jelezve, hogy itt végződik egy teljes sztring.

##### Keresés (Search)

A keresési művelet során ellenőrizzük, hogy egy adott sztring megtalálható-e a trie-ben. Az algoritmus a következő lépésekből áll:

1. Kezdjük a gyökércsomópontnál.
2. A keresett sztring minden egyes karakterére:
    - Ha a karakterrel címkézett él létezik a jelenlegi csomópontban, kövessük az élt a következő csomópontra.
    - Ha az él nem létezik, a sztring nem található a trie-ben.
3. Ha elértük a sztring végét, ellenőrizzük, hogy a jelenlegi csomópont levél-e. Ha igen, a sztring megtalálható, különben nem.

##### Törlés (Deletion)

A törlési művelet során egy meglévő sztringet távolítunk el a trie-ből. Az algoritmus a következő lépésekből áll:

1. Kezdjük a gyökércsomópontnál.
2. A törölni kívánt sztring minden egyes karakterére:
    - Kövessük az élt a megfelelő csomópontra.
    - Ha az él nem létezik, a sztring nem található a trie-ben, így a törlés sikertelen.
3. A sztring végén:
    - Jelöljük meg a jelenlegi csomópontot nem levélként, jelezve, hogy itt nem végződik többé egy teljes sztring.
4. Visszafelé haladva töröljük az összes olyan csomópontot, amely már nem része más sztringnek (nem elágazó és nem levél).

#### Tömörített Trie

A tömörített trie, más néven Patricia trie vagy radix fa, optimalizált változata a hagyományos trie-nak, amely csökkenti a memóriaigényt és növeli a keresési sebességet. A tömörített trie-ben az egyes csomópontok több karaktert is reprezentálhatnak, ha nincs elágazás az útvonalon.

##### Tömörített Trie Szerkezete

A tömörített trie csomópontjai több karaktert tartalmaznak, és csak az elágazási pontokon hozunk létre új csomópontokat. Ezáltal a trie kompaktabbá válik, és a közös prefixeket még hatékonyabban használjuk ki.

##### Tömörített Trie Beszúrása

A tömörített trie-be történő beszúrás hasonló a hagyományos trie-hoz, de figyelembe veszi a hosszabb karakterláncokat:

1. Kezdjük a gyökércsomópontnál.
2. Az új sztring minden egyes karakterére:
    - Ha a jelenlegi csomópont több karaktert tartalmaz, vizsgáljuk meg a közös prefixet.
    - Ha nincs közös prefix, hozzunk létre új csomópontokat és éleket.
    - Ha van közös prefix, folytassuk a következő karakterekkel.
3. A sztring végén jelöljük meg a jelenlegi csomópontot levélként.

#### Alkalmazások

A trie és a tömörített trie számos gyakorlati alkalmazással bír, különösen a szöveges keresés és az auto-complete területén. Ezek az adatszerkezetek lehetővé teszik a gyors és hatékony keresést, amely nélkülözhetetlen a modern szöveges feldolgozó rendszerekben.

#### Összegzés

A trie adatszerkezet egy erőteljes eszköz a sztringek kezelésére, amely lehetővé teszi a gyors keresést, beszúrást és törlést. A tömörített trie tovább optimalizálja a hagyományos trie-t, csökkentve a memóriaigényt és növelve a hatékonyságot. Az alapfogalmak és tulajdonságok megértése kulcsfontosságú a trie alkalmazásában és implementálásában.

### 6.2. Műveletek: beszúrás, keresés, törlés

A trie adatszerkezet egyik legfontosabb jellemzője a hatékonysága a különböző műveletek, mint a beszúrás, keresés és törlés végrehajtásában. Ebben az alfejezetben részletesen bemutatjuk ezen műveletek algoritmusait, elméleti hátterét és gyakorlati megvalósítását. A példakódok C++ nyelven kerülnek bemutatásra, hogy szemléltessük a trie működését és implementációját.

#### Beszúrás (Insertion)

A beszúrási művelet célja egy új sztring hozzáadása a trie-hez. Az algoritmus lépései a következők:

1. **Kiindulás a gyökérből:** Kezdjük a gyökércsomópontnál.
2. **Karakterenkénti beszúrás:** Az új sztring minden egyes karakterére:
    - **Él követése:** Ha a karakterrel címkézett él létezik a jelenlegi csomópontban, kövessük az élt a következő csomópontra.
    - **Új csomópont létrehozása:** Ha az él nem létezik, hozzunk létre egy új csomópontot a karakterrel, és kapcsoljuk össze a jelenlegi csomóponttal egy új éllel.
3. **Levél jelölése:** A sztring végén jelöljük meg a jelenlegi csomópontot levélként, jelezve, hogy itt végződik egy teljes sztring.

##### C++ Példakód

```cpp
#include <iostream>
#include <unordered_map>

class TrieNode {
public:
    std::unordered_map<char, TrieNode*> children;
    bool isEndOfWord;

    TrieNode() : isEndOfWord(false) {}
};

class Trie {
private:
    TrieNode* root;

public:
    Trie() {
        root = new TrieNode();
    }

    void insert(const std::string& word) {
        TrieNode* node = root;
        for (char ch : word) {
            if (node->children.find(ch) == node->children.end()) {
                node->children[ch] = new TrieNode();
            }
            node = node->children[ch];
        }
        node->isEndOfWord = true;
    }
};
```

#### Keresés (Search)

A keresési művelet célja annak ellenőrzése, hogy egy adott sztring megtalálható-e a trie-ben. Az algoritmus lépései a következők:

1. **Kiindulás a gyökérből:** Kezdjük a gyökércsomópontnál.
2. **Karakterenkénti keresés:** A keresett sztring minden egyes karakterére:
    - **Él követése:** Ha a karakterrel címkézett él létezik a jelenlegi csomópontban, kövessük az élt a következő csomópontra.
    - **Hiányzó él:** Ha az él nem létezik, a sztring nem található a trie-ben.
3. **Levél ellenőrzése:** Ha elértük a sztring végét, ellenőrizzük, hogy a jelenlegi csomópont levél-e. Ha igen, a sztring megtalálható, különben nem.

##### C++ Példakód

```cpp
bool search(const std::string& word) {
    TrieNode* node = root;
    for (char ch : word) {
        if (node->children.find(ch) == node->children.end()) {
            return false;
        }
        node = node->children[ch];
    }
    return node->isEndOfWord;
}
```

#### Törlés (Deletion)

A törlési művelet célja egy meglévő sztring eltávolítása a trie-ből. Az algoritmus lépései a következők:

1. **Rekurzív törlés:** A törlési művelet rekurzív, és az alábbi lépésekből áll:
    - **Alap eset:** Ha a sztring üres, állítsuk a jelenlegi csomópont `isEndOfWord` attribútumát `false` értékre. Ha a jelenlegi csomópontnak nincsenek gyermekei, térjünk vissza `true` értékkel, jelezve, hogy a csomópont törölhető.
    - **Rekurzív eset:** Kövessük a sztring első karakterével címkézett élt a megfelelő gyermekcsomópontra. Ha a rekurzív törlés sikeres volt, és a gyermekcsomópontnak nincsenek további gyermekei, töröljük a gyermeket, és térjünk vissza `true` értékkel. Ha a gyermeknek további gyermekei vannak, vagy az `isEndOfWord` attribútuma `true`, térjünk vissza `false` értékkel.

##### C++ Példakód

```cpp
bool remove(TrieNode* node, const std::string& word, int depth) {
    if (depth == word.size()) {
        if (!node->isEndOfWord) {
            return false;
        }
        node->isEndOfWord = false;
        return node->children.empty();
    }

    char ch = word[depth];
    TrieNode* childNode = node->children[ch];
    if (childNode == nullptr) {
        return false;
    }

    bool shouldDeleteChild = remove(childNode, word, depth + 1);

    if (shouldDeleteChild) {
        node->children.erase(ch);
        return node->children.empty() && !node->isEndOfWord;
    }
    return false;
}

void remove(const std::string& word) {
    remove(root, word, 0);
}
```

#### Elméleti Háttér

A trie adatszerkezet és műveleteinek hatékonysága számos tényezőtől függ, többek között a tárolt sztringek hosszától és a közös prefixek számától.

##### Időkomplexitás

- **Beszúrás:** Az időkomplexitás O(L), ahol L a beszúrandó sztring hossza. Ez azért van, mert minden karakterhez egy csomópontot vagy élt hozunk létre.
- **Keresés:** Az időkomplexitás szintén O(L), mivel minden karakterhez egy csomópontot követünk.
- **Törlés:** Az időkomplexitás O(L), mivel a törlési művelet is karakterenként halad végig a sztringen. A rekurzív törlés során minden karakterhez egy csomópontot vizsgálunk és esetleg törlünk.

##### Memóriaigény

A trie memóriaigénye a sztringek számától és hosszától függ. A legrosszabb esetben minden karakter külön csomópontban van tárolva, ami O(N * L) memóriaigényt jelent, ahol N a sztringek száma, és L az átlagos hossz.

#### Gyakorlati Megvalósítás

A trie gyakorlati megvalósítása során figyelembe kell venni a hatékonyságot és a memóriahasználatot. A C++ példakódokban bemutatott megközelítések segítenek a trie adatszerkezet hatékony implementálásában és alkalmazásában.

##### Optimalizációk

- **Tömörített Trie:** A tömörített trie csökkenti a memóriaigényt azáltal, hogy több karaktert is tárol egy csomópontban, ha nincs elágazás az útvonalon.
- **HashMap használata:** A gyermekcsomópontok tárolására használt hash map (unordered_map) segít a gyors keresésben és beszúrásban.

#### Összegzés

A trie adatszerkezet műveleteinek részletes vizsgálata rámutat arra, hogy mennyire hatékonyan képes kezelni a sztringeket. A beszúrás, keresés és törlés műveletek mindegyike lineáris időkomplexitással rendelkezik a sztring hosszára nézve, ami különösen előnyös nagy mennyiségű sztring esetén. A trie implementációjának és optimalizációjának megértése kulcsfontosságú a hatékony sztringkezeléshez és -kereséshez számos alkalmazási területen.

### 6.3. Tömörített Trie és implementációi

A hagyományos trie adatszerkezet jelentős memóriaigénnyel járhat, különösen akkor, ha sok rövid sztringet kell tárolnunk. A tömörített trie, más néven Patricia trie (Practical Algorithm to Retrieve Information Coded in Alphanumeric), hatékonyabb módot kínál a sztringek tárolására és kezelésére azáltal, hogy csökkenti a csomópontok számát és az utak hosszát. Ebben az alfejezetben részletesen bemutatjuk a tömörített trie elméleti hátterét, működését, valamint gyakorlati implementációit.

#### Tömörített Trie Alapjai

A tömörített trie a hagyományos trie egy optimalizált változata, amelyben az egyes csomópontok több karaktert is tartalmazhatnak, ha nincs elágazás az útvonalon. Ez a megközelítés jelentősen csökkenti a csomópontok számát és a trie méretét, különösen akkor, ha sok sztring közös prefixekkel rendelkezik.

##### Tömörített Trie Szerkezete

A tömörített trie szerkezete hasonló a hagyományos trie-hoz, de az egyes csomópontok több karaktert tartalmazhatnak. A csomópontok az alábbi tulajdonságokkal rendelkeznek:

- **Karakterlánc (Edge Label):** Minden csomópont egy vagy több karaktert tartalmazó éllel van címkézve.
- **Gyermekek (Children):** A csomópontoknak lehetnek gyermekcsomópontjaik, amelyek az adott karakterlánc után következő karaktereket reprezentálják.
- **Levél (Leaf):** A levelek teljes sztringek végét jelölik, mint a hagyományos trie-ban.

#### Tömörített Trie Működése

A tömörített trie műveletei, mint a beszúrás, keresés és törlés, hasonlóak a hagyományos trie-hoz, de figyelembe kell venni a hosszabb karakterláncokat.

##### Beszúrás (Insertion)

A beszúrási művelet során egy új sztringet adunk hozzá a trie-hez. Az algoritmus lépései a következők:

1. **Kiindulás a gyökérből:** Kezdjük a gyökércsomópontnál.
2. **Karakterlánc összehasonlítása:** Az új sztring első karakterétől kezdve hasonlítsuk össze a jelenlegi csomópont élének karakterláncával.
    - **Közös prefix:** Ha a karakterlánc egy része megegyezik, haladjunk tovább a következő csomópontra, és folytassuk az összehasonlítást.
    - **Részleges egyezés:** Ha a karakterlánc egy része megegyezik, de nem teljesen, bontsuk ketté az élt, és hozzunk létre új csomópontokat a fennmaradó karakterekhez.
    - **Nincs egyezés:** Ha nincs egyezés, hozzunk létre új csomópontokat és éleket az új karakterlánchoz.
3. **Levél jelölése:** A sztring végén jelöljük meg a jelenlegi csomópontot levélként, jelezve, hogy itt végződik egy teljes sztring.

##### Keresés (Search)

A keresési művelet során ellenőrizzük, hogy egy adott sztring megtalálható-e a trie-ben. Az algoritmus lépései a következők:

1. **Kiindulás a gyökérből:** Kezdjük a gyökércsomópontnál.
2. **Karakterlánc összehasonlítása:** A keresett sztring minden egyes karakterére:
    - **Közös prefix:** Ha a karakterlánc egy része megegyezik, haladjunk tovább a következő csomópontra, és folytassuk az összehasonlítást.
    - **Nincs egyezés:** Ha nincs egyezés, a sztring nem található a trie-ben.
3. **Levél ellenőrzése:** Ha elértük a sztring végét, ellenőrizzük, hogy a jelenlegi csomópont levél-e. Ha igen, a sztring megtalálható, különben nem.

##### Törlés (Deletion)

A törlési művelet során egy meglévő sztringet távolítunk el a trie-ből. Az algoritmus lépései a következők:

1. **Rekurzív törlés:** A törlési művelet rekurzív, és az alábbi lépésekből áll:
    - **Alap eset:** Ha a sztring üres, állítsuk a jelenlegi csomópont `isEndOfWord` attribútumát `false` értékre. Ha a jelenlegi csomópontnak nincsenek gyermekei, térjünk vissza `true` értékkel, jelezve, hogy a csomópont törölhető.
    - **Rekurzív eset:** Kövessük a sztring első karakterével címkézett élt a megfelelő gyermekcsomópontra. Ha a rekurzív törlés sikeres volt, és a gyermekcsomópontnak nincsenek további gyermekei, töröljük a gyermeket, és térjünk vissza `true` értékkel. Ha a gyermeknek további gyermekei vannak, vagy az `isEndOfWord` attribútuma `true`, térjünk vissza `false` értékkel.

#### Tömörített Trie Implementációja

A tömörített trie implementálása C++ nyelven hasonló a hagyományos trie-hoz, de figyelembe kell venni a hosszabb karakterláncokat és az elágazási pontok kezelését.

##### C++ Példakód

```cpp
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>

class CompressedTrieNode {
public:
    std::unordered_map<std::string, CompressedTrieNode*> children;
    bool isEndOfWord;

    CompressedTrieNode() : isEndOfWord(false) {}
};

class CompressedTrie {
private:
    CompressedTrieNode* root;

    void insert(CompressedTrieNode* node, const std::string& word, size_t index) {
        if (index == word.size()) {
            node->isEndOfWord = true;
            return;
        }

        std::string currentLabel = word.substr(index);
        for (auto& child : node->children) {
            const std::string& childLabel = child.first;
            size_t commonLength = commonPrefixLength(currentLabel, childLabel);

            if (commonLength > 0) {
                if (commonLength < childLabel.size()) {
                    splitEdge(node, child.first, commonLength);
                }
                insert(child.second, word, index + commonLength);
                return;
            }
        }

        node->children[currentLabel] = new CompressedTrieNode();
        node->children[currentLabel]->isEndOfWord = true;
    }

    size_t commonPrefixLength(const std::string& str1, const std::string& str2) {
        size_t length = 0;
        while (length < str1.size() && length < str2.size() && str1[length] == str2[length]) {
            length++;
        }
        return length;
    }

    void splitEdge(CompressedTrieNode* node, const std::string& label, size_t length) {
        CompressedTrieNode* child = node->children[label];
        std::string newLabel = label.substr(0, length);
        std::string remainderLabel = label.substr(length);

        node->children.erase(label);
        node->children[newLabel] = new CompressedTrieNode();
        node->children[newLabel]->children[remainderLabel] = child;
    }

public:
    CompressedTrie() {
        root = new CompressedTrieNode();
    }

    void insert(const std::string& word) {
        insert(root, word, 0);
    }

    bool search(const std::string& word) {
        CompressedTrieNode* node = root;
        size_t index = 0;

        while (index < word.size()) {
            std::string currentLabel = word.substr(index);
            bool found = false;

            for (const auto& child : node->children) {
                const std::string& childLabel = child.first;
                size_t commonLength = commonPrefixLength(currentLabel, childLabel);

                if (commonLength == childLabel.size()) {
                    node = child.second;
                    index += commonLength;
                    found = true;
                    break;
                }
            }

            if (!found) {
                return false;
            }
        }

        return node->isEndOfWord;
    }
};
```

#### Elméleti Háttér

A tömörített trie hatékonysága és memóriaigénye számos tényezőtől függ, beleértve a tárolt sztringek hosszát és a közös prefixek számát.

##### Időkomplexitás

- **Beszúrás:** Az időkomplexitás O(L), ahol L a beszúrandó sztring hossza. Az algoritmus minden karakterláncot összehasonlít, és szükség esetén szétválasztja az éleket.
- **Keresés:** Az időkomplexitás szintén O(L), mivel minden karakterláncot végig kell követni a sztring keresése során.
- **Törlés:** Az időkomplexitás O(L), mivel a törlési művelet is karakterláncokkal dolgozik, és rekurzívan végigjárja a trie-t.

##### Memóriaigény

A tömörített trie memóriaigénye jelentősen kisebb lehet, mint a hagyományos trie-é, különösen akkor, ha sok sztring közös prefixekkel rendelkezik. A tömörített trie-ben az egyes csomópontok hosszabb karakterláncokat tartalmazhatnak, így csökkentve a csomópontok számát és az utak hosszát.

#### Gyakorlati Megvalósítás

A tömörített trie gyakorlati megvalósítása során fontos figyelembe venni az optimalizációkat és a hatékonyságot. A C++ példakódokban bemutatott megközelítések segítenek a tömörített trie adatszerkezet hatékony implementálásában és alkalmazásában.

##### Optimalizációk

- **Karakterláncok szétválasztása:** Az élek szétválasztása és új csomópontok létrehozása szükséges, ha a karakterláncok részlegesen egyeznek.
- **Közös prefixek kezelése:** A közös prefixek hatékony kezelése csökkenti a csomópontok számát és növeli a keresési sebességet.

#### Összegzés

A tömörített trie adatszerkezet egy hatékony megoldás a sztringek tárolására és kezelésére, amely jelentősen csökkenti a memóriaigényt és növeli a műveletek sebességét a hagyományos trie-hoz képest. A tömörített trie implementálása és optimalizálása kulcsfontosságú a sztringkezelő rendszerek hatékonyságának növeléséhez, és számos gyakorlati alkalmazásban előnyös lehet.

### 6.4. Alkalmazások: szöveges keresés, auto-complete

A trie és a tömörített trie adatszerkezetek különösen hasznosak számos gyakorlati alkalmazásban, ahol a hatékony sztringkezelés és gyors keresés elengedhetetlen. Ebben az alfejezetben részletesen bemutatjuk a trie alkalmazását a szöveges keresés és az auto-complete funkciók területén. Megvizsgáljuk az elméleti alapokat, a gyakorlati megvalósítást, és a teljesítmény szempontjait.

#### Szöveges Keresés

A szöveges keresés az egyik leggyakoribb alkalmazása a trie adatszerkezetnek. A trie lehetővé teszi a sztringek gyors és hatékony keresését, különösen akkor, ha nagyszámú sztringet kell kezelni.

##### Prefix Keresés

A trie egyik legnagyobb előnye a prefix keresés hatékonysága. A prefix keresés során egy adott prefixszel kezdődő összes sztringet keressük meg a trie-ben. Az algoritmus lépései a következők:

1. **Kiindulás a gyökérből:** Kezdjük a gyökércsomópontnál.
2. **Prefix követése:** Kövessük a prefix minden karakterét, amíg el nem érjük a prefix végét vagy egy olyan csomópontot, amely nem tartalmazza a következő karaktert.
3. **Sztringek összegyűjtése:** Ha elértük a prefix végét, gyűjtsük össze az összes sztringet, amely a jelenlegi csomópont alatti részfában található.

##### C++ Példakód

```cpp
#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>

class TrieNode {
public:
    std::unordered_map<char, TrieNode*> children;
    bool isEndOfWord;

    TrieNode() : isEndOfWord(false) {}
};

class Trie {
private:
    TrieNode* root;

    void collectWords(TrieNode* node, std::string prefix, std::vector<std::string>& result) {
        if (node->isEndOfWord) {
            result.push_back(prefix);
        }
        for (auto& child : node->children) {
            collectWords(child.second, prefix + child.first, result);
        }
    }

public:
    Trie() {
        root = new TrieNode();
    }

    void insert(const std::string& word) {
        TrieNode* node = root;
        for (char ch : word) {
            if (node->children.find(ch) == node->children.end()) {
                node->children[ch] = new TrieNode();
            }
            node = node->children[ch];
        }
        node->isEndOfWord = true;
    }

    std::vector<std::string> searchByPrefix(const std::string& prefix) {
        TrieNode* node = root;
        for (char ch : prefix) {
            if (node->children.find(ch) == node->children.end()) {
                return {};
            }
            node = node->children[ch];
        }
        std::vector<std::string> result;
        collectWords(node, prefix, result);
        return result;
    }
};
```

#### Auto-Complete

Az auto-complete funkció célja, hogy a felhasználó által bevitt részleges sztring alapján javaslatokat adjon a lehetséges teljes sztringekre. Ez a funkció széles körben elterjedt a modern szövegszerkesztőkben, keresőmotorokban és egyéb felhasználói felületeken.

##### Algoritmus

Az auto-complete algoritmus alapja a prefix keresés, amelyet a trie adatszerkezet segítségével valósítunk meg:

1. **Prefix követése:** Kövessük a felhasználó által bevitt prefix minden karakterét a trie-ben, amíg el nem érjük a prefix végét vagy egy olyan csomópontot, amely nem tartalmazza a következő karaktert.
2. **Javaslatok gyűjtése:** Ha elértük a prefix végét, gyűjtsük össze az összes lehetséges teljes sztringet, amelyek a prefixszel kezdődnek.
3. **Javaslatok megjelenítése:** A gyűjtött sztringeket javaslatként jelenítsük meg a felhasználónak.

##### Teljesítmény

Az auto-complete funkció hatékonysága nagymértékben függ a trie adatszerkezet szervezésétől és optimalizálásától. A tömörített trie használata jelentősen csökkentheti a keresési időt és a memóriaigényt, különösen nagy adatbázisok esetén.

##### C++ Példakód

A fenti példában bemutatott `searchByPrefix` függvény tökéletesen illeszkedik az auto-complete funkcióhoz. A gyűjtött sztringeket javaslatként használhatjuk fel:

```cpp
std::vector<std::string> getAutoCompleteSuggestions(const std::string& prefix) {
    return searchByPrefix(prefix);
}
```

#### Teljesítmény Optimalizálás

A trie és a tömörített trie teljesítménye optimalizálható többféle módon, amelyek közül néhányat az alábbiakban ismertetünk.

##### Közös Prefixek Kihasználása

A trie egyik legfontosabb előnye a közös prefixek kihasználása, ami jelentősen csökkenti a csomópontok számát és a memóriaigényt. A tömörített trie tovább optimalizálja ezt azáltal, hogy egy csomópontban több karaktert is tárol, ha nincs elágazás az útvonalon.

##### HashMap Alapú Gyermekkezelés

A gyermekcsomópontok tárolására használt hash map (unordered_map) segít a gyors keresésben és beszúrásban. Ez különösen fontos nagy adatbázisok esetén, ahol a keresési idő minimalizálása kritikus.

##### Párhuzamos Feldolgozás

A párhuzamos feldolgozás alkalmazása a trie műveletek során tovább növelheti a teljesítményt. Például a sztringek beszúrása és keresése párhuzamos szálakon végezhető, ami jelentősen csökkenti a válaszidőt nagy mennyiségű adat esetén.

#### Gyakorlati Alkalmazások

A trie és a tömörített trie számos gyakorlati alkalmazásban megtalálható, amelyek közül néhányat az alábbiakban mutatunk be.

##### Keresőmotorok

A keresőmotorokban a trie-t gyakran használják az auto-complete funkció megvalósítására, amely segíti a felhasználókat a keresési kifejezések gyors és hatékony bevitelében. A keresési javaslatok valós időben történő generálása kritikus a felhasználói élmény szempontjából.

##### Szövegszerkesztők

A modern szövegszerkesztők gyakran használnak trie-t az auto-complete funkció megvalósításához, amely segíti a programozókat a kód gyorsabb és hibamentesebb írásában. A trie segítségével a szövegszerkesztők gyorsan és hatékonyan kínálnak javaslatokat a beírt kódrészletek alapján.

##### Nyelvi Modellalkalmazások

A nyelvi modellekben, például a helyesírás-ellenőrzőkben és szövegkiegészítőkben a trie és a tömörített trie gyakran alkalmazott adatszerkezetek a szavak tárolására és gyors keresésére. A nyelvi modellek hatékonysága nagymértékben függ a trie teljesítményétől.

#### Összegzés

A trie és a tömörített trie adatszerkezetek kiválóan alkalmasak a szöveges keresés és az auto-complete funkciók megvalósítására. A közös prefixek kihasználása, a hash map alapú gyermekkezelés, és a párhuzamos feldolgozás mind hozzájárulnak a műveletek hatékonyságához és teljesítményéhez. A trie gyakorlati alkalmazásai széleskörűek, és számos területen kritikus szerepet játszanak a gyors és hatékony sztringkezelésben.

