\newpage 

# 2. Hash táblák

A hash táblák az egyik legfontosabb és leggyakrabban használt adatszerkezetek közé tartoznak a számítástechnikában. Különleges hatékonyságuk miatt széles körben alkalmazzák őket a programozásban, különösen akkor, amikor gyors adatkeresésre és -tárolásra van szükség. Ebben a fejezetben részletesen bemutatjuk a hash táblák működését, az alapvető koncepcióktól kezdve a különböző hash függvényeken és ütközés kezelési módszereken át, egészen a teljesítmény elemzéséig és gyakorlati alkalmazásokig. Végül, konkrét példákon keresztül mutatjuk be, hogyan valósítható meg egy hatékony hash tábla C++ nyelven.

## 2.1 Hash függvények és tulajdonságaik

A hash függvények a hash táblák alapvető komponensei, amelyek az adatokat egyedi kulcsokhoz rendelik. Ezen függvények célja, hogy egy tetszőleges méretű adatból (mint például egy karakterlánc vagy egy szám) egy fix méretű értéket, azaz hash értéket generáljanak. Ez az érték aztán egy pozíciót vagy indexet jelöl a hash táblában, ahol az adott adat tárolva lesz.

### Hash függvények célja és felhasználása

A hash függvények fő célja a gyors keresés, beszúrás és törlés műveletek támogatása. Ezek a műveletek ideális esetben O(1) időbonyolultságúak, azaz a művelet végrehajtásának ideje független az adatszerkezet méretétől. A hash függvények különösen hasznosak nagy mennyiségű adat kezelésénél, ahol a teljesítmény kritikus fontosságú.

### Hash függvények tulajdonságai

A jó hash függvényeknek az alábbi tulajdonságokkal kell rendelkezniük:

1. **Determináltság**: Ugyanazon bemenet esetén mindig ugyanazt az eredményt kell visszaadniuk.
2. **Egyenletes eloszlás**: Az eredményeknek egyenletesen kell eloszlaniuk a hash térben, minimalizálva az ütközések számát.
3. **Gyors számítás**: A hash érték kiszámításának gyorsnak kell lennie.
4. **Minimalizált ütközések**: Azonos hash értéket adó különböző bemenetek (ütközések) számának minimálisnak kell lennie.

### Hash függvények típusai

Többféle hash függvény létezik, amelyeket különböző célokra használnak:

- **Egyszerű modulo alapú hash függvények**: Ezek a függvények egyszerű matematikai műveletekkel számítják ki a hash értéket. Például, egy egész szám hash értékét úgy kaphatjuk meg, hogy az adott számot elosztjuk a tábla méretével, és vesszük a maradékot (modulo operáció). Ez azonban nem mindig eredményez egyenletes eloszlást.
- **Kriptográfiai hash függvények**: Ezek biztonságosabbak és egyenletesebb eloszlást biztosítanak, de számításuk általában lassabb. Ilyen függvények például az SHA-256 vagy MD5.
- **Univerzális hash függvények**: Ezek célja, hogy minimalizálják az ütközéseket különböző bemeneti adatok esetén, akár véletlenül, akár szándékosan generáltak azok.

### Példa: Egyszerű hash függvény C++ nyelven

Az alábbiakban bemutatunk egy egyszerű, de hatékony hash függvényt C++ nyelven, amely egy karakterláncot alakít át hash értékké.

```cpp
#include <iostream>
#include <string>

unsigned int simpleHash(const std::string &str, unsigned int tableSize) {
    unsigned int hash = 0;
    for (char ch : str) {
        hash = 31 * hash + ch;
    }
    return hash % tableSize;
}

int main() {
    std::string key = "example";
    unsigned int tableSize = 101; // például egy 101 méretű hash tábla
    unsigned int hashValue = simpleHash(key, tableSize);
    std::cout << "Hash érték: " << hashValue << std::endl;
    return 0;
}
```

Ebben a példában a `simpleHash` függvény a bemeneti karakterlánc minden egyes karakteréhez hozzáad egy súlyozott értéket, majd az eredményt modulo művelettel a tábla méretéhez igazítja.

### Ütközéskezelés

Még a legjobb hash függvények esetében is előfordulhatnak ütközések, amikor két különböző bemenet azonos hash értéket eredményez. Az ütközések kezelésére több módszer létezik:

1. **Nyílt címzés**: Ütközés esetén a hash tábla következő szabad helyére tesszük az adatot. Különböző technikák léteznek, mint a lineáris próbálkozás, kvadratikus próbálkozás, és kettős hash-elés.

2. **Láncolás**: Minden hash tábla pozícióhoz egy láncolt lista tartozik, ahol az adott indexhez tartozó összes elem tárolható. Az új elemeket egyszerűen hozzáfűzzük a lista végéhez.

### Hash függvények hatékonysága és bonyolultsága

A hash függvények hatékonysága nagymértékben függ a hash tábla méretétől és a hash függvény minőségétől. A jó hash függvények segítenek minimalizálni az ütközések számát, ezáltal növelve a műveletek átlagos teljesítményét. A bonyolultság szempontjából a hash függvények általában O(1) időbonyolultságúak, ami rendkívül hatékonnyá teszi őket.

### Gyakorlati alkalmazások

A hash függvények és hash táblák számos területen alkalmazhatók, például:

- **Gyors adatkeresés**: Például szótárak, telefonkönyvek, és adatbázis indexek esetében.
- **Adatok összefűzése**: Például fájlok integritásának ellenőrzése hash értékek segítségével.
- **Kriptográfia**: Adatok biztonságos tárolása és ellenőrzése.

Összefoglalva, a hash függvények alapvető szerepet játszanak a hatékony adatkezelésben, és megértésük elengedhetetlen a modern programozási technikák alkalmazásához. A megfelelő hash függvény kiválasztása és az ütközés kezelési módszerek helyes alkalmazása jelentős mértékben javíthatja az alkalmazások teljesítményét és megbízhatóságát.


## 2.2 Ütközés kezelési módszerek

A hash táblák egyik legnagyobb kihívása az ütközések kezelése. Ütközés akkor lép fel, amikor két különböző bemenet ugyanazt a hash értéket eredményezi, azaz ugyanarra a pozícióra kerülne a hash táblában. Mivel a hash függvények nem tudják garantálni az ütközések elkerülését, különféle ütközés kezelési módszereket kell alkalmazni. Az alábbiakban részletesen bemutatjuk a leggyakoribb módszereket, azok használati módját, előnyeit, bonyolultságát és hatékonyságát.

### 1. Nyílt címzés (Open Addressing)

Nyílt címzés esetén, ha egy adott pozíció már foglalt, az algoritmus megkeresi a következő szabad helyet a hash táblában. Többféle nyílt címzési módszer létezik:

**1.1. Lineáris próbálkozás (Linear Probing)**

A lineáris próbálkozás során az ütközés esetén az algoritmus lineárisan halad előre a táblában, amíg egy üres helyet nem talál.

**Példa kód C++ nyelven:**

```cpp
#include <iostream>
#include <vector>
#include <string>

const int TABLE_SIZE = 101;

struct HashTableEntry {
    std::string key;
    std::string value;
    bool isOccupied = false;
};

class HashTable {
private:
    std::vector<HashTableEntry> table;

    int hashFunction(const std::string &key) {
        int hash = 0;
        for (char ch : key) {
            hash = 31 * hash + ch;
        }
        return hash % TABLE_SIZE;
    }

public:
    HashTable() : table(TABLE_SIZE) {}

    void insert(const std::string &key, const std::string &value) {
        int hash = hashFunction(key);
        int originalHash = hash;

        while (table[hash].isOccupied) {
            hash = (hash + 1) % TABLE_SIZE;
            if (hash == originalHash) {
                std::cerr << "Hash table is full!" << std::endl;
                return;
            }
        }

        table[hash] = {key, value, true};
    }

    std::string search(const std::string &key) {
        int hash = hashFunction(key);
        int originalHash = hash;

        while (table[hash].isOccupied) {
            if (table[hash].key == key) {
                return table[hash].value;
            }
            hash = (hash + 1) % TABLE_SIZE;
            if (hash == originalHash) {
                break;
            }
        }

        return "Not found";
    }
};

int main() {
    HashTable ht;
    ht.insert("example", "This is an example");
    std::cout << "Search result: " << ht.search("example") << std::endl;
    return 0;
}
```

**Előnyök:**
- Egyszerű implementáció.
- Hatékony memóriahasználat.

**Hátrányok:**
- Elsődleges klaszterezés: a hosszú, egymást követő foglalt helyek láncolatát eredményezi, ami lassítja a keresést és beszúrást.

**1.2. Kvadratikus próbálkozás (Quadratic Probing)**

Kvadratikus próbálkozás során a léptetés nem lineáris, hanem kvadratikusan nő (pl. 1, 4, 9, 16, stb.), ami csökkenti az elsődleges klaszterezés hatását.

**Előnyök:**
- Csökkenti az elsődleges klaszterezést.

**Hátrányok:**
- Másodlagos klaszterezést eredményezhet: bizonyos minták még mindig klasztereket eredményezhetnek.
- Komplexebb implementáció, mint a lineáris próbálkozás.

**1.3. Kettős hash-elés (Double Hashing)**

Kettős hash-elés során két különböző hash függvényt használunk, az ütközések kezelésére.

**Példa kód C++ nyelven:**

```cpp
#include <iostream>
#include <vector>
#include <string>

const int TABLE_SIZE = 101;

struct HashTableEntry {
    std::string key;
    std::string value;
    bool isOccupied = false;
};

class HashTable {
private:
    std::vector<HashTableEntry> table;

    int hashFunction1(const std::string &key) {
        int hash = 0;
        for (char ch : key) {
            hash = 31 * hash + ch;
        }
        return hash % TABLE_SIZE;
    }

    int hashFunction2(const std::string &key) {
        int hash = 0;
        for (char ch : key) {
            hash = 17 * hash + ch;
        }
        return 1 + (hash % (TABLE_SIZE - 1));
    }

public:
    HashTable() : table(TABLE_SIZE) {}

    void insert(const std::string &key, const std::string &value) {
        int hash1 = hashFunction1(key);
        int hash2 = hashFunction2(key);
        int hash = hash1;

        while (table[hash].isOccupied) {
            hash = (hash + hash2) % TABLE_SIZE;
        }

        table[hash] = {key, value, true};
    }

    std::string search(const std::string &key) {
        int hash1 = hashFunction1(key);
        int hash2 = hashFunction2(key);
        int hash = hash1;

        while (table[hash].isOccupied) {
            if (table[hash].key == key) {
                return table[hash].value;
            }
            hash = (hash + hash2) % TABLE_SIZE;
        }

        return "Not found";
    }
};

int main() {
    HashTable ht;
    ht.insert("example", "This is an example");
    std::cout << "Search result: " << ht.search("example") << std::endl;
    return 0;
}
```

**Előnyök:**
- Csökkenti az ütközések számát és az elsődleges klaszterezést.

**Hátrányok:**
- Komplexebb hash függvények szükségesek.
- Nehezebb implementáció.

### 2. Láncolás (Chaining)

Láncolás esetén minden hash tábla pozíció egy láncolt listát tartalmaz, amely az adott pozícióhoz tartozó összes elemet tárolja. Ütközés esetén az új elemet egyszerűen hozzáfűzzük a lista végéhez.

**Példa kód C++ nyelven:**

```cpp
#include <iostream>
#include <list>
#include <vector>
#include <string>

const int TABLE_SIZE = 101;

struct HashTableEntry {
    std::string key;
    std::string value;
};

class HashTable {
private:
    std::vector<std::list<HashTableEntry>> table;

    int hashFunction(const std::string &key) {
        int hash = 0;
        for (char ch : key) {
            hash = 31 * hash + ch;
        }
        return hash % TABLE_SIZE;
    }

public:
    HashTable() : table(TABLE_SIZE) {}

    void insert(const std::string &key, const std::string &value) {
        int hash = hashFunction(key);
        table[hash].emplace_back(HashTableEntry{key, value});
    }

    std::string search(const std::string &key) {
        int hash = hashFunction(key);
        for (const auto &entry : table[hash]) {
            if (entry.key == key) {
                return entry.value;
            }
        }
        return "Not found";
    }
};

int main() {
    HashTable ht;
    ht.insert("example", "This is an example");
    std::cout << "Search result: " << ht.search("example") << std::endl;
    return 0;
}
```

**Előnyök:**
- Egyszerű implementáció.
- Hatékonyan kezeli az ütközéseket.
- Nem igényel újrapróbálkozást a hash táblán belül.

**Hátrányok:**
- Több memória szükséges a láncolt listák miatt.
- A láncok hossza megnőhet, ami lassítja a keresést és beszúrást.

### 3. Egyéb módszerek

**3.1. Többszörös hash táblázat (Multiple Hash Tables)**

Ez a módszer több hash táblát használ, és az elemeket ezek között osztja szét. Ez csökkenti az ütközések számát és növeli a teljesítményt.

**3.2. Cuckoo Hashing**

Ez egy speciális módszer, amely két hash függvényt használ, és ha ütközés lép fel, az elemet áthelyezi egy másik helyre, potenciálisan újabb elemek áthelyezését eredményezve.

**Előnyök:**
- Nagyon alacsony ütközési arány.
- Hatékony memóriahasználat.

**Hátrányok:**
- Komplex implementáció.
- Néha szükség lehet az egész tábla újraszervezésére.

### Összegzés

Az ütközés kezelési módszerek kiválasztása nagyban befolyásolja a hash táblák teljesítményét és hatékonyságát. Minden módszernek vannak előnyei és hátrányai, és a választás gyakran az adott alkalmazási területtől és a hash tábla várható terhelésétől függ. A megfelelő ütközés kezelési stratégia alkalmazása kritikus a hash táblák hatékony működéséhez és a gyors adatkezelés biztosításához.



## 2.3 Az adatszerkezet ismertetése és kapcsolata a hash függvényekkel

A hash táblák olyan adatszerkezetek, amelyek lehetővé teszik az adatok gyors tárolását és visszakeresését. Az alapötlet az, hogy az adatokhoz egyedi kulcsokat rendelünk, majd ezeket a kulcsokat egy hash függvény segítségével egy fix méretű táblában helyezzük el. A hash függvénynek köszönhetően a keresési, beszúrási és törlési műveletek átlagos esetben O(1) időbonyolultságúak, ami rendkívül hatékony.

### Hash táblák célja és felhasználása

A hash táblákat számos területen használják, ahol gyors adatelérésre van szükség. Tipikus felhasználási módok közé tartoznak a következők:

- **Szótárak és térképek**: Az adatok kulcs-érték párok formájában történő tárolása és gyors keresése.
- **Gyorsítótárak (cache)**: Adatok ideiglenes tárolása a gyors elérés érdekében.
- **Adatbázis indexelés**: Adatbázisokban az adatok gyors elérésének biztosítása.
- **Gyors keresési műveletek**: Bármilyen olyan alkalmazás, ahol nagy mennyiségű adat között gyorsan kell keresni.

### Hash táblák előnyei

1. **Gyorsaság**: Az átlagos keresési, beszúrási és törlési műveletek időbonyolultsága O(1).
2. **Egyszerű implementáció**: A hash táblák viszonylag egyszerűen implementálhatók és használhatók.
3. **Rugalmasság**: A hash táblák különböző típusú adatok tárolására alkalmasak, beleértve az egyszerű számokat, karakterláncokat és összetett adatstruktúrákat is.

### Hash függvények és adatszerkezet kapcsolatának részletezése

A hash táblák hatékonysága nagymértékben függ a hash függvény minőségétől és a használt ütközés kezelési módszertől. A hash függvény az adatkulcsokat egy fix méretű hash táblához rendel, amely tárolja az adatokat. Az alábbiakban bemutatjuk a hash függvények és a hash táblák közötti kapcsolatot, valamint a hash tábla felépítését.

### Hash tábla felépítése

A hash tábla egy fix méretű tömb, amelyben minden egyes hely (vödör) tárolhat egy adatot vagy egy láncolt lista kezdetét, amely több adatot is tárolhat (láncolásos ütközéskezelés esetén). Az adatok elhelyezéséhez és visszakereséséhez a hash függvény által generált indexet használjuk.

**Példa kód C++ nyelven:**

Az alábbi példa bemutatja egy egyszerű hash tábla implementációját C++ nyelven, láncolásos ütközéskezeléssel.

```cpp
#include <iostream>
#include <list>
#include <vector>
#include <string>

const int TABLE_SIZE = 101;

struct HashTableEntry {
    std::string key;
    std::string value;
};

class HashTable {
private:
    std::vector<std::list<HashTableEntry>> table;

    int hashFunction(const std::string &key) {
        int hash = 0;
        for (char ch : key) {
            hash = 31 * hash + ch;
        }
        return hash % TABLE_SIZE;
    }

public:
    HashTable() : table(TABLE_SIZE) {}

    void insert(const std::string &key, const std::string &value) {
        int hash = hashFunction(key);
        table[hash].emplace_back(HashTableEntry{key, value});
    }

    std::string search(const std::string &key) {
        int hash = hashFunction(key);
        for (const auto &entry : table[hash]) {
            if (entry.key == key) {
                return entry.value;
            }
        }
        return "Not found";
    }

    void remove(const std::string &key) {
        int hash = hashFunction(key);
        auto &entries = table[hash];
        for (auto it = entries.begin(); it != entries.end(); ++it) {
            if (it->key == key) {
                entries.erase(it);
                return;
            }
        }
    }
};

int main() {
    HashTable ht;
    ht.insert("example", "This is an example");
    ht.insert("test", "This is a test");
    
    std::cout << "Search result for 'example': " << ht.search("example") << std::endl;
    std::cout << "Search result for 'test': " << ht.search("test") << std::endl;
    
    ht.remove("example");
    std::cout << "Search result for 'example' after removal: " << ht.search("example") << std::endl;

    return 0;
}
```

### Bonyolultság és hatékonyság

A hash táblák átlagos esetben nagyon hatékonyak, az átlagos időbonyolultság O(1) a keresésre, beszúrásra és törlésre. Azonban a legrosszabb esetben, ha sok ütközés van és minden elem egyetlen vödörbe kerül, a bonyolultság O(n) is lehet, ahol n az elemek száma. Ezért fontos, hogy jól megtervezett hash függvényt használjunk, és megfelelő ütközéskezelési módszert válasszunk.

### Hatékonysági megfontolások

1. **Hash függvény minősége**: A jó hash függvény egyenletesen osztja el az adatokat a táblában, minimalizálva az ütközéseket.
2. **Tábla mérete**: A tábla méretét úgy kell megválasztani, hogy elegendő vödör legyen az adatok számára, de ne legyen túl nagy sem, mivel ez pazarláshoz vezet.
3. **Terhelési tényező**: A terhelési tényező (load factor) a táblában lévő elemek számának és a vödörök számának aránya. A magas terhelési tényező több ütközést eredményez, míg az alacsony terhelési tényező jobb teljesítményt biztosít.

### Összefoglalás

A hash táblák és hash függvények szoros kapcsolatban állnak egymással. A hash függvények határozzák meg, hogy az adatok hol helyezkednek el a hash táblában, és hogy milyen hatékonyan tudunk hozzáférni ezekhez az adatokhoz. A megfelelő hash függvény kiválasztása és az ütközések hatékony kezelése kulcsfontosságú a hash táblák teljesítményének optimalizálása érdekében. A bemutatott C++ példakód jól illusztrálja a hash táblák működését és a különböző műveletek implementációját, bemutatva azok hatékonyságát és egyszerűségét.

## 2.4 Teljesítmény és komplexitás

A hash táblák egyik legnagyobb előnye a hatékonyságuk, különösen akkor, ha az adatok gyors tárolása és visszakeresése szükséges. Ebben az alfejezetben részletesen megvizsgáljuk a hash táblák teljesítményét és komplexitását, bemutatva azok előnyeit, bonyolultságát, valamint az alkalmazási területeiket.

### Hash táblák teljesítménye

A hash táblák teljesítménye nagymértékben függ a következő tényezőktől:

1. **Hash függvény minősége**: Egy jó hash függvény egyenletesen osztja el az elemeket a hash táblában, minimalizálva az ütközéseket. Az egyenletes eloszlás biztosítja, hogy a keresési, beszúrási és törlési műveletek időbonyolultsága O(1) maradjon.

2. **Terhelési tényező (Load Factor)**: A terhelési tényező a hash táblában tárolt elemek számának és a hash tábla méretének aránya. A magas terhelési tényező növeli az ütközések valószínűségét, ami lassítja a műveleteket. Általában az optimális terhelési tényező 0,7 és 0,8 között van.

3. **Ütközéskezelési módszerek**: Az ütközéskezelési módszerek, mint például a láncolás vagy a nyílt címzés, szintén befolyásolják a hash táblák teljesítményét. Az alábbiakban részletesen megvizsgáljuk a különböző módszerek bonyolultságát és hatékonyságát.

### Teljesítmény és bonyolultság elemzése

**1. Láncolás (Chaining)**

A láncolásos ütközéskezelés esetén minden vödör egy láncolt listát tartalmaz, amely az adott hash értékhez tartozó összes elemet tárolja.

- **Keresés**: Átlagos esetben O(1), de legrosszabb esetben O(n), ha minden elem ugyanabba a vödörbe kerül.
- **Beszúrás**: Átlagos esetben O(1), de legrosszabb esetben O(n).
- **Törlés**: Átlagos esetben O(1), de legrosszabb esetben O(n).

Példa kód C++ nyelven:

```cpp
#include <iostream>
#include <list>
#include <vector>
#include <string>

const int TABLE_SIZE = 101;

struct HashTableEntry {
    std::string key;
    std::string value;
};

class HashTable {
private:
    std::vector<std::list<HashTableEntry>> table;

    int hashFunction(const std::string &key) {
        int hash = 0;
        for (char ch : key) {
            hash = 31 * hash + ch;
        }
        return hash % TABLE_SIZE;
    }

public:
    HashTable() : table(TABLE_SIZE) {}

    void insert(const std::string &key, const std::string &value) {
        int hash = hashFunction(key);
        table[hash].emplace_back(HashTableEntry{key, value});
    }

    std::string search(const std::string &key) {
        int hash = hashFunction(key);
        for (const auto &entry : table[hash]) {
            if (entry.key == key) {
                return entry.value;
            }
        }
        return "Not found";
    }

    void remove(const std::string &key) {
        int hash = hashFunction(key);
        auto &entries = table[hash];
        for (auto it = entries.begin(); it != entries.end(); ++it) {
            if (it->key == key) {
                entries.erase(it);
                return;
            }
        }
    }
};

int main() {
    HashTable ht;
    ht.insert("example", "This is an example");
    ht.insert("test", "This is a test");

    std::cout << "Search result for 'example': " << ht.search("example") << std::endl;
    std::cout << "Search result for 'test': " << ht.search("test") << std::endl;

    ht.remove("example");
    std::cout << "Search result for 'example' after removal: " << ht.search("example") << std::endl;

    return 0;
}
```

**2. Nyílt címzés (Open Addressing)**

A nyílt címzés esetén az elemeket ütközés esetén egy másik vödörbe helyezzük a hash táblában. Többféle nyílt címzési módszer létezik:

- **Lineáris próbálkozás (Linear Probing)**: Ütközés esetén az algoritmus lineárisan halad előre a következő szabad helyig.
    - **Keresés**: Átlagos esetben O(1), de legrosszabb esetben O(n).
    - **Beszúrás**: Átlagos esetben O(1), de legrosszabb esetben O(n).
    - **Törlés**: Átlagos esetben O(1), de legrosszabb esetben O(n).

- **Kvadratikus próbálkozás (Quadratic Probing)**: Ütközés esetén a léptetés kvadratikusan növekszik (1, 4, 9, ...).
    - **Keresés**: Átlagos esetben O(1), de legrosszabb esetben O(n).
    - **Beszúrás**: Átlagos esetben O(1), de legrosszabb esetben O(n).
    - **Törlés**: Átlagos esetben O(1), de legrosszabb esetben O(n).

- **Kettős hash-elés (Double Hashing)**: Két különböző hash függvényt használunk, és ütközés esetén a második hash függvény értékével lépünk tovább.
    - **Keresés**: Átlagos esetben O(1), de legrosszabb esetben O(n).
    - **Beszúrás**: Átlagos esetben O(1), de legrosszabb esetben O(n).
    - **Törlés**: Átlagos esetben O(1), de legrosszabb esetben O(n).

Példa kód C++ nyelven kettős hash-eléssel:

```cpp
#include <iostream>
#include <vector>
#include <string>

const int TABLE_SIZE = 101;

struct HashTableEntry {
    std::string key;
    std::string value;
    bool isOccupied = false;
};

class HashTable {
private:
    std::vector<HashTableEntry> table;

    int hashFunction1(const std::string &key) {
        int hash = 0;
        for (char ch : key) {
            hash = 31 * hash + ch;
        }
        return hash % TABLE_SIZE;
    }

    int hashFunction2(const std::string &key) {
        int hash = 0;
        for (char ch : key) {
            hash = 17 * hash + ch;
        }
        return 1 + (hash % (TABLE_SIZE - 1));
    }

public:
    HashTable() : table(TABLE_SIZE) {}

    void insert(const std::string &key, const std::string &value) {
        int hash1 = hashFunction1(key);
        int hash2 = hashFunction2(key);
        int hash = hash1;

        while (table[hash].isOccupied) {
            hash = (hash + hash2) % TABLE_SIZE;
        }

        table[hash] = {key, value, true};
    }

    std::string search(const std::string &key) {
        int hash1 = hashFunction1(key);
        int hash2 = hashFunction2(key);
        int hash = hash1;

        while (table[hash].isOccupied) {
            if (table[hash].key == key) {
                return table[hash].value;
            }
            hash = (hash + hash2) % TABLE_SIZE;
        }

        return "Not found";
    }

    void remove(const std::string &key) {
        int hash1 = hashFunction1(key);
        int hash2 = hashFunction2(key);
        int hash = hash1;

        while (table[hash].isOccupied) {
            if (table[hash].key == key) {
                table[hash].isOccupied = false;
                return;
            }
            hash = (hash + hash2) % TABLE_SIZE;
        }
    }
};

int main() {
    HashTable ht;
    ht.insert("example", "This is an example");
    ht.insert("test", "This is a test");

    std::cout << "Search result for 'example': " << ht.search("example") << std::endl;
    std::cout << "Search result for 'test': " << ht.search("test") << std::endl;

    ht.remove("example");
    std::cout << "Search result for 'example' after removal: " << ht.search("example") << std::endl;

    return 0;
}
```

### Hash táblák alkalmazási területei

1. **Gyors adatkeresés**: Hash táblákat gyakran használnak olyan alkalmazásokban, ahol gyors adatkeresés szükséges, például szótárakban és térképekben.
2. **Gyorsítótárak (cache)**: Az ideiglenes adat tárolása és gyors elérése érdekében.
3. **Adatbázis indexelés**: Az adatok gyors keresése és rendezése.
4. **Számítógépes hálózatok**: IP címek tárolása és gyors keresése.
5. **Kriptográfia**: Hash függvények használata adatbiztonság és integritás biztosítására.

### Előnyök és hátrányok

**Előnyök:**
- Gyors keresési, beszúrási és törlési műveletek átlagos esetben.
- Egyszerű és hatékony memóriahasználat.
- Rugalmasság különböző típusú adatok tárolására.

**Hátrányok:**
- Ütközések kezelése bonyolultabbá teheti az implementációt.
- A legrosszabb esetben a teljesítmény O(n) lehet, ha a hash függvény nem megfelelő.
- Nagyobb terhelési tényező esetén csökken a hatékonyság.

### Összefoglalás

A hash táblák rendkívül hatékony adatszerkezetek, amelyek gyors adatkeresést, beszúrást és törlést tesznek lehetővé. A hash függvények és az ütközéskezelési módszerek minősége jelentősen befolyásolja a hash táblák teljesítményét. Az optimális hash függvény és megfelelő ütközéskezelési módszer kiválasztása kulcsfontosságú a hash táblák hatékony működéséhez. A bemutatott C++ példakódok jól illusztrálják a hash táblák működését és a különböző ütközéskezelési módszerek alkalmazását, bemutatva azok előnyeit és bonyolultságát.


## 2.5 Alkalmazások és gyakorlati példák

A hash táblák számos területen alkalmazhatók a számítástechnikában, köszönhetően a gyors adatkeresési, beszúrási és törlési képességeiknek. Ebben az alfejezetben részletesen bemutatjuk a hash táblák különböző gyakorlati alkalmazásait, azok előnyeit és bonyolultságát. A bemutatott példák C++ nyelven íródtak, hogy illusztrálják a hash táblák használatát valós alkalmazásokban.

### Alkalmazási területek

1. **Szótárak és térképek**
2. **Gyorsítótárak (Cache)**
3. **Adatbázis indexelés**
4. **Számítógépes hálózatok**
5. **Kriptográfia**
6. **Játékfejlesztés**

### 1. Szótárak és térképek

A szótárak (dictionary) és térképek (map) kulcs-érték párok tárolására szolgálnak. A hash táblák lehetővé teszik, hogy ezeket a párokat gyorsan tároljuk és visszakeressük.

**Példa kód C++ nyelven:**

```cpp
#include <iostream>
#include <unordered_map>
#include <string>

int main() {
    std::unordered_map<std::string, std::string> dictionary;

    // Beszúrás
    dictionary["apple"] = "A fruit that is sweet and crisp.";
    dictionary["banana"] = "A long, yellow fruit that is soft and sweet.";

    // Keresés
    std::string key = "apple";
    if (dictionary.find(key) != dictionary.end()) {
        std::cout << "Definition of " << key << ": " << dictionary[key] << std::endl;
    } else {
        std::cout << key << " not found in dictionary." << std::endl;
    }

    // Törlés
    dictionary.erase("banana");

    // Ellenőrzés törlés után
    key = "banana";
    if (dictionary.find(key) != dictionary.end()) {
        std::cout << "Definition of " << key << ": " << dictionary[key] << std::endl;
    } else {
        std::cout << key << " not found in dictionary." << std::endl;
    }

    return 0;
}
```

**Előnyök:**
- Gyors keresés, beszúrás és törlés.
- Egyszerű használat és implementáció.

**Hátrányok:**
- Hash függvény és ütközéskezelés minősége befolyásolja a teljesítményt.

### 2. Gyorsítótárak (Cache)

A gyorsítótárak olyan adatszerkezetek, amelyek ideiglenesen tárolnak adatokat a gyors elérés érdekében. A hash táblák gyakran használatosak gyorsítótárak megvalósítására, mivel gyors adatelérést biztosítanak.

**Példa kód C++ nyelven:**

```cpp
#include <iostream>
#include <unordered_map>
#include <list>
#include <string>

class LRUCache {
private:
    int capacity;
    std::unordered_map<std::string, std::pair<int, std::list<std::string>::iterator>> cache;
    std::list<std::string> lru;

public:
    LRUCache(int cap) : capacity(cap) {}

    int get(const std::string &key) {
        if (cache.find(key) == cache.end()) {
            return -1; // A kulcs nem található
        }
        lru.erase(cache[key].second);
        lru.push_front(key);
        cache[key].second = lru.begin();
        return cache[key].first;
    }

    void put(const std::string &key, int value) {
        if (cache.find(key) != cache.end()) {
            lru.erase(cache[key].second);
        } else if (cache.size() >= capacity) {
            cache.erase(lru.back());
            lru.pop_back();
        }
        lru.push_front(key);
        cache[key] = {value, lru.begin()};
    }
};

int main() {
    LRUCache cache(2);
    cache.put("apple", 1);
    cache.put("banana", 2);

    std::cout << "Get apple: " << cache.get("apple") << std::endl; // Output: 1

    cache.put("cherry", 3); // Túlcsordulás, a "banana" eltávolításra kerül

    std::cout << "Get banana: " << cache.get("banana") << std::endl; // Output: -1
    std::cout << "Get cherry: " << cache.get("cherry") << std::endl; // Output: 3

    return 0;
}
```

**Előnyök:**
- Gyors adatelérés.
- Hatékony memóriahasználat.

**Hátrányok:**
- Korlátozott kapacitás, amely bonyolultabbá teheti az adatkezelést.

### 3. Adatbázis indexelés

Az adatbázisokban a hash táblák használhatók az adatok gyors keresésének és rendezésének megkönnyítésére. Az indexek segítségével a lekérdezések végrehajtási ideje jelentősen csökkenhet.

**Példa kód C++ nyelven:**

```cpp
#include <iostream>
#include <unordered_map>
#include <string>

class Database {
private:
    std::unordered_map<int, std::string> index;

public:
    void insert(int id, const std::string &data) {
        index[id] = data;
    }

    std::string search(int id) {
        if (index.find(id) != index.end()) {
            return index[id];
        } else {
            return "Data not found";
        }
    }

    void remove(int id) {
        index.erase(id);
    }
};

int main() {
    Database db;
    db.insert(1, "Data for ID 1");
    db.insert(2, "Data for ID 2");

    std::cout << "Search result for ID 1: " << db.search(1) << std::endl; // Output: Data for ID 1
    std::cout << "Search result for ID 2: " << db.search(2) << std::endl; // Output: Data for ID 2

    db.remove(1);
    std::cout << "Search result for ID 1 after removal: " << db.search(1) << std::endl; // Output: Data not found

    return 0;
}
```

**Előnyök:**
- Gyors adatkeresés és rendezés.
- Hatékony adatkezelés nagy mennyiségű adathoz.

**Hátrányok:**
- Hash függvény és ütközéskezelés minősége befolyásolja a teljesítményt.

### 4. Számítógépes hálózatok

A hash táblák a számítógépes hálózatokban is fontos szerepet játszanak, például IP címek tárolásában és gyors keresésében.

**Példa kód C++ nyelven:**

```cpp
#include <iostream>
#include <unordered_map>
#include <string>

class IPAddressTable {
private:
    std::unordered_map<std::string, std::string> ipTable;

public:
    void addIPAddress(const std::string &hostname, const std::string &ip) {
        ipTable[hostname] = ip;
    }

    std::string getIPAddress(const std::string &hostname) {
        if (ipTable.find(hostname) != ipTable.end()) {
            return ipTable[hostname];
        } else {
            return "IP address not found";
        }
    }

    void removeIPAddress(const std::string &hostname) {
        ipTable.erase(hostname);
    }
};

int main() {
    IPAddressTable ipTable;
    ipTable.addIPAddress("example.com", "192.168.1.1");
    ipTable.addIPAddress("test.com", "192.168.1.2");

    std::cout << "IP address of example.com: " << ipTable.getIPAddress("example.com") << std::endl; // Output: 192.168.1.1
    std::cout << "IP address of test.com: " << ipTable.getIPAddress("test.com") << std::endl; // Output: 192.168.1.2

    ipTable.removeIPAddress("example.com");
    std::cout << "IP address of example.com after removal: " << ipTable.getIPAddress("example.com") << std::endl; // Output: IP address not found

    return 0;
}
```

**Előnyök:**
- Gyors IP cím keresés és tárolás.
- Hatékony hálózati adatkezelés.

**Hátrányok:**
- Hash függvény minősége és ütközéskezelés befolyásolja a teljesítményt.

### 5. Kriptográfia

A hash függvények a kriptográfiában is elengedhetetlenek, például digitális aláírások és adatintegritás ellenőrzésére.

**Példa kód C++ nyelven (SHA-256 hash függvény használata):**

```cpp
#include <iostream>
#include <openssl/sha.h>
#include <iomanip>
#include <sstream>

std::string sha256(const std::string str) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, str.c_str(), str.size());
    SHA256_Final(hash, &sha256);
    std::stringstream ss;
    for(int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    return ss.str();
}

int main() {
    std::string data = "example data";
    std::string hash = sha256(data);
    std::cout << "SHA-256 hash: " << hash << std::endl;

    return 0;
}
```

**Előnyök:**
- Adatbiztonság és integritás biztosítása.
- Széles körben használható különböző kriptográfiai alkalmazásokban.

**Hátrányok:**
- Magasabb számítási bonyolultság a hash függvényeknél.

### 6. Játékfejlesztés

A hash táblák a játékfejlesztésben is hasznosak, például objektumok gyors kereséséhez vagy események kezeléséhez.

**Példa kód C++ nyelven:**

```cpp
#include <iostream>
#include <unordered_map>
#include <string>

class GameObject {
public:
    std::string name;
    int x, y;

    GameObject(const std::string &name, int x, int y) : name(name), x(x), y(y) {}
};

class GameWorld {
private:
    std::unordered_map<std::string, GameObject> objects;

public:
    void addObject(const std::string &id, const GameObject &object) {
        objects[id] = object;
    }

    GameObject* getObject(const std::string &id) {
        if (objects.find(id) != objects.end()) {
            return &objects[id];
        } else {
            return nullptr;
        }
    }

    void removeObject(const std::string &id) {
        objects.erase(id);
    }
};

int main() {
    GameWorld world;
    world.addObject("player1", GameObject("Player1", 10, 20));
    world.addObject("enemy1", GameObject("Enemy1", 30, 40));

    GameObject* player = world.getObject("player1");
    if (player) {
        std::cout << "Found player: " << player->name << " at (" << player->x << ", " << player->y << ")" << std::endl;
    }

    world.removeObject("player1");

    player = world.getObject("player1");
    if (!player) {
        std::cout << "Player1 not found." << std::endl;
    }

    return 0;
}
```

**Előnyök:**
- Gyors objektumkezelés.
- Hatékony játékfejlesztési eszköz.

**Hátrányok:**
- Hash függvény minősége és ütközéskezelés befolyásolja a teljesítményt.

### Összefoglalás

A hash táblák rendkívül sokoldalú adatszerkezetek, amelyek számos területen alkalmazhatók a számítástechnikában. A bemutatott példák és alkalmazási területek jól illusztrálják a hash táblák hatékonyságát és egyszerűségét. A megfelelő hash függvény és ütközéskezelési módszer kiválasztása kritikus a hash táblák teljesítményének optimalizálása érdekében. A C++ példakódok segítségével könnyen megérthető és implementálható a hash táblák használata valós alkalmazásokban.