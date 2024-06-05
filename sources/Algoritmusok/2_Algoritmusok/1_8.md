\newpage

## 1.8. Hash alapú keresés

A modern számítástechnika egyik leggyakrabban használt és leghatékonyabb módszere a gyors adatkeresésre a hash alapú keresés. Ez a technika kulcsszerepet játszik az adatbázisok, az algoritmusok és a különböző adatstruktúrák világában, ahol az adatok gyors és hatékony elérése kritikus fontosságú. A hash alapú keresés lényege, hogy az adatokhoz tartozó kulcsokat egy hash függvény segítségével egyedi értékekké alakítjuk át, amelyek alapján az adatok könnyen és gyorsan megtalálhatók. Ebben a fejezetben megismerkedünk a hashing alapjaival, a különböző hash függvényekkel és ütközés kezelési módszerekkel, valamint a hash táblák gyakorlati alkalmazásával a keresési műveletek során. Az alábbiakban részletesen bemutatjuk, hogyan működik a hashing, milyen problémák merülhetnek fel és hogyan lehet ezeket hatékonyan kezelni, hogy biztosítsuk az optimális keresési teljesítményt.

### 1.8.1. Hashing alapjai

A hash alapú keresés alapvető építőköve a hashing technika, amely az adatok hatékony elérésének és kezelésének egyik leghatékonyabb módszere. A hashing egy olyan eljárás, amely során egy kulcsot, amely általában valamilyen adat, egy determinisztikus algoritmus segítségével egy hash értékké alakítunk át. Ez az átalakítás egy fix hosszúságú bit-sorozatot eredményez, amelyet hash értéknek vagy egyszerűen hash-nek nevezünk.

#### A hashing koncepciója

A hashing alapja egy hash függvény használata, amely egy bemeneti kulcsot egy kimeneti hash értékké alakít. A hash függvények célja, hogy az adatok egyenletes eloszlását biztosítsák a hash értékek között, minimalizálva az ütközések számát, amikor két különböző kulcs azonos hash értéket kap. Az ideális hash függvény gyorsan végrehajtható, determinisztikus, és azonos bemenetre mindig azonos kimenetet ad, ugyanakkor különböző bemenetekre különböző kimeneteket generál.

#### Hash függvények

Számos hash függvény létezik, mindegyiknek megvan a maga előnye és hátránya. A hash függvények kiválasztása függ az alkalmazás specifikus követelményeitől, például a sebességtől, a biztonságtól és a hash értékek eloszlásától. A leggyakoribb hash függvények közé tartoznak:

1. **Egyszerű modulo hash függvény**: Ez az egyik legegyszerűbb hash függvény, amely a kulcsot egy adott `m` szám szerinti maradékával osztja:

   $h(k) = k \mod m$

   ahol `k` a kulcs és `m` a hash tábla mérete. Ez a módszer gyors, de gyakran nem biztosít egyenletes eloszlást, különösen akkor, ha `m` nem egy prím szám.

2. **Multiplikatív hash függvény**: Ez a módszer a kulcsot egy konstanssal szorozza meg, majd a törtbeli részét veszi figyelembe:

   $h(k) = \lfloor m \cdot (k \cdot A \mod 1) \rfloor$

   ahol `A` egy irracionális szám, leggyakrabban az aranymetszés $(\sqrt{5} - 1) / 2$.

3. **Kriptográfiai hash függvények**: Ezeket a függvényeket gyakran használják olyan alkalmazásokban, ahol a biztonság kiemelt fontosságú, például az adatok hitelesítésénél és digitális aláírásoknál. Ilyen hash függvények például az MD5, SHA-1 és a SHA-256.

#### Ütközés kezelési módszerek

Mivel a hash tábla mérete véges, elkerülhetetlen, hogy időnként két különböző kulcs azonos hash értéket kapjon. Ezt az eseményt ütközésnek nevezzük. Számos módszer létezik az ütközések kezelésére:

1. **Nyílt címzés (Open Addressing)**: Az ütközések kezelésének egyik módja, hogy a hash tábla másik pozíciójába helyezzük a kulcsot. Ennek több változata is van:
    - **Lineáris próba (Linear Probing)**: Ha egy hely foglalt, akkor a következő szabad helyet keressük meg.

      $h_i(k) = (h(k) + i) \mod m$

    - **Kvadratikus próba (Quadratic Probing)**: A következő szabad helyet kvadratikusan növekvő távolságra keressük meg.

      $h_i(k) = (h(k) + c_1 \cdot i + c_2 \cdot i^2) \mod m$

    - **Kettős hash (Double Hashing)**: Két különböző hash függvényt használunk, és a második hash függvény által adott értékkel lépünk tovább.

      $h_i(k) = (h_1(k) + i \cdot h_2(k)) \mod m$

2. **Láncolás (Chaining)**: Minden hash értékhez egy lista vagy lánc tartozik, amely az azonos hash értékű kulcsokat tárolja. Ha ütközés történik, az új elemet a lista végére fűzzük.

   ```cpp
   class HashTable {
   private:
       static const int TABLE_SIZE = 10;
       std::list<int> table[TABLE_SIZE];
       
       int hashFunction(int key) {
           return key % TABLE_SIZE;
       }

   public:
       void insertItem(int key) {
           int index = hashFunction(key);
           table[index].push_back(key);
       }

       void deleteItem(int key) {
           int index = hashFunction(key);
           table[index].remove(key);
       }

       bool searchItem(int key) {
           int index = hashFunction(key);
           for (int item : table[index]) {
               if (item == key)
                   return true;
           }
           return false;
       }
   };
   ```

#### Hash tábla struktúrák

A hash tábla egy olyan adatstruktúra, amely egyedi kulcsok és hozzájuk tartozó értékek tárolására szolgál. A hash tábla fő elemei a következők:

- **Hash függvény**: A kulcsokat hash értékekké alakítja.
- **Tábla**: Az a memória terület, ahol a hash értékekhez rendelt adatok tárolódnak.
- **Ütközés kezelési mechanizmus**: Az ütközések kezelésére szolgál, például láncolás vagy nyílt címzés.

#### Hash tábla működése

1. **Beszúrás (Insertion)**: A beszúrás során a kulcsot a hash függvény segítségével egy hash értékké alakítjuk, majd az értéket a megfelelő helyre illesztjük a hash táblába.
2. **Keresés (Search)**: A keresés során a kulcsot hash értékké alakítjuk, majd az értéket a hash tábla megfelelő helyén keressük.
3. **Törlés (Deletion)**: A törlés során a kulcsot hash értékké alakítjuk, majd az értéket a hash tábla megfelelő helyéről eltávolítjuk.

#### Hash tábla teljesítménye

A hash tábla teljesítménye nagymértékben függ a hash függvény minőségétől és az ütközés kezelési módszertől. Az ideális hash tábla:

- **Gyors beszúrást, keresést és törlést** biztosít, általában $O(1)$ időbonyolultsággal.
- **Egyenletesen osztja el az elemeket** a táblában, minimalizálva az ütközések számát.
- **Hatékonyan kezeli az ütközéseket**, biztosítva, hogy az ütközések minimális hatással legyenek a teljesítményre.

A hash tábla tehát egy rendkívül hatékony adatstruktúra, amely számos alkalmazásban használható, beleértve az adatbázisok kezelését, a gyors adatkeresést és a különböző algoritmusok optimalizálását. Azonban fontos a megfelelő hash függvény és ütközés kezelési módszer kiválasztása a hatékony működés érdekében.

### 1.8.2. Hash függvények és ütközés kezelési módszerek

A hash alapú keresés hatékonysága nagymértékben függ a hash függvények minőségétől és az ütközés kezelésének módszereitől. Ebben a fejezetben részletesen megvizsgáljuk a különböző hash függvények tulajdonságait, előnyeit és hátrányait, valamint az ütközések kezelésének különböző technikáit.

#### Hash függvények

A hash függvények célja, hogy a bemeneti kulcsokat fix hosszúságú, általában bináris értékekké alakítsák át. Egy jó hash függvény jellemzői közé tartozik a gyors végrehajtás, a determinisztikusság, és az, hogy az egyenletes eloszlást biztosít a hash értékek között. Íme néhány ismert hash függvény:

1. **Egyszerű modulo hash függvény**:
   Az egyik legegyszerűbb hash függvény a kulcsot egy adott szám (`m`) szerinti maradékával osztja.

   $h(k) = k \mod m$

   Ez a módszer gyors, de gyakran nem biztosít egyenletes eloszlást, különösen akkor, ha `m` nem prím szám.

2. **Multiplikatív hash függvény**:
   Ez a módszer a kulcsot egy konstanssal (`A`) szorozza meg, majd a törtbeli részét veszi figyelembe:

   $h(k) = \lfloor m \cdot (k \cdot A \mod 1) \rfloor$

   Itt `A` egy irracionális szám, gyakran az aranymetszés $(\sqrt{5} - 1) / 2$. Ez a módszer jobban elosztja az értékeket a táblában.

3. **Kriptográfiai hash függvények**:
   Ezek a függvények magas szintű biztonságot biztosítanak, és gyakran használják adatbiztonsági alkalmazásokban. Ilyen például az MD5, SHA-1, és SHA-256. Ezek a függvények bonyolultak, és biztosítják, hogy még apró változások a bemenetben is jelentős változásokat eredményezzenek a kimenetben.

#### Ütközés kezelési módszerek

Az ütközések elkerülhetetlenek bármilyen hash alapú rendszerben, ezért különféle módszereket fejlesztettek ki az ütközések kezelésére. Az alábbiakban a leggyakrabban használt módszereket tárgyaljuk.

1. **Nyílt címzés (Open Addressing)**:
   A nyílt címzés során, ha egy hely foglalt, egy új helyet keresünk a hash táblában az alábbi módszerek egyikével:

    - **Lineáris próba (Linear Probing)**:
      Ha egy hely foglalt, akkor a következő szabad helyet keressük meg sorban.

      $h_i(k) = (h(k) + i) \mod m$

      Azonban ez a módszer hajlamos klaszterizálódást okozni, ahol az ütközések egymás után következnek be, ami lelassíthatja a keresést és a beszúrást.

    - **Kvadratikus próba (Quadratic Probing)**:
      A következő szabad helyet kvadratikusan növekvő távolságra keressük meg.

      $h_i(k) = (h(k) + c_1 \cdot i + c_2 \cdot i^2) \mod m$

      Ez a módszer csökkenti a klaszterizációt, de nem garantálja, hogy mindig megtalálunk egy szabad helyet, különösen akkor, ha a tábla majdnem tele van.

    - **Kettős hash (Double Hashing)**:
      Két különböző hash függvényt használunk, és a második hash függvény által adott értékkel lépünk tovább.

      $h_i(k) = (h_1(k) + i \cdot h_2(k)) \mod m$

      A kettős hash módszer hatékonyabb ütközéskezelést biztosít, mivel a második hash függvény egy második szempontot biztosít az új hely kiválasztásához.

2. **Láncolás (Chaining)**:
   A láncolás során minden hash értékhez egy lista vagy lánc tartozik, amely az azonos hash értékű kulcsokat tárolja. Ez lehetővé teszi, hogy minden lista különböző hosszúságú legyen, így hatékonyan kezelve az ütközéseket.

   ```cpp
   class HashTable {
   private:
       static const int TABLE_SIZE = 10;
       std::list<int> table[TABLE_SIZE];
       
       int hashFunction(int key) {
           return key % TABLE_SIZE;
       }

   public:
       void insertItem(int key) {
           int index = hashFunction(key);
           table[index].push_back(key);
       }

       void deleteItem(int key) {
           int index = hashFunction(key);
           table[index].remove(key);
       }

       bool searchItem(int key) {
           int index = hashFunction(key);
           for (int item : table[index]) {
               if (item == key)
                   return true;
           }
           return false;
       }
   };
   ```

   A láncolás előnye, hogy a hash tábla mérete rugalmasabb, és könnyen bővíthető. Azonban a láncok hossza befolyásolhatja a keresési időt, különösen akkor, ha sok ütközés van.

3. **Nyílt láncolás (Open Chaining)**:
   Ez a módszer a láncolás egy változata, ahol a láncok nyíltan kapcsolódnak a hash táblán kívül is. Ez megkönnyíti a bővítést és a karbantartást.

#### Gyakorlati példák és implementáció

Az alábbiakban bemutatunk néhány gyakorlati példát C++ nyelven, hogy illusztráljuk a hash függvények és ütközés kezelési módszerek alkalmazását.

**Egyszerű hash tábla láncolással**:

```cpp
#include <iostream>
#include <list>
#include <iterator>

class HashTable {
private:
    static const int TABLE_SIZE = 10;
    std::list<int> table[TABLE_SIZE];
    
    int hashFunction(int key) {
        return key % TABLE_SIZE;
    }

public:
    void insertItem(int key) {
        int index = hashFunction(key);
        table[index].push_back(key);
    }

    void deleteItem(int key) {
        int index = hashFunction(key);
        table[index].remove(key);
    }

    void displayHash() {
        for (int i = 0; i < TABLE_SIZE; i++) {
            std::cout << i;
            for (auto x : table[i])
                std::cout << " --> " << x;
            std::cout << std::endl;
        }
    }
};

int main() {
    int keys[] = {15, 11, 27, 8, 12};
    int n = sizeof(keys)/sizeof(keys[0]);

    HashTable h;
    for (int i = 0; i < n; i++)
        h.insertItem(keys[i]);

    h.deleteItem(12);

    h.displayHash();

    return 0;
}
```

Ez a példa egy egyszerű hash tábla implementációját mutatja be, amely láncolást használ az ütközések kezelésére. A program beszúr néhány kulcsot a hash táblába, töröl egy kulcsot, majd megjeleníti a hash tábla tartalmát.

**Kettős hash tábla implementáció**:

```cpp
#include <iostream>
#include <vector>

class DoubleHashTable {
private:
    std::vector<int> table;
    int TABLE_SIZE;
    int PRIME;
    
    int hash1(int key) {
        return key % TABLE_SIZE;
    }
    
    int hash2(int key) {
        return PRIME - (key % PRIME);
    }

public:
    DoubleHashTable(int size, int prime) : TABLE_SIZE(size), PRIME(prime) {
        table.resize(TABLE_SIZE, -1);
    }
    
    void insertItem(int key) {
        int index = hash1(key);
        if (table[index] != -1) {
            int i = 1;
            while (true) {
                int newIndex = (index + i * hash2(key)) % TABLE_SIZE;
                if (table[newIndex] == -1) {
                    table[newIndex] = key;
                    break;
                }
                i++;
            }
        } else {
            table[index] = key;
        }
    }

    void displayHash() {
        for (int i = 0; i < TABLE_SIZE; i++) {
            if (table[i] != -1)
                std::cout << i << " --> " << table[i] << std::endl;
            else
                std::cout << i << std::endl;
        }
    }
};

int main() {
    int keys[] = {19, 27, 36, 10, 64};
    int n = sizeof(keys)/sizeof(keys[0]);
    int TABLE_SIZE = 7;
    int PRIME = 5;

    DoubleHashTable h(TABLE_SIZE, PRIME);
    for (int i = 0; i < n; i++)
        h.insertItem(keys[i]);

    h.displayHash();

    return 0;
}
```

Ez a példa egy kettős hash tábla implementációját mutatja be, ahol két különböző hash függvényt használunk az ütközések kezelésére.

#### Összegzés

A hash függvények és ütközés kezelési módszerek kritikus szerepet játszanak a hash alapú keresés hatékonyságában. A megfelelő hash függvény kiválasztása és az ütközések hatékony kezelése biztosítja a gyors és megbízható adatkezelést. Az egyszerű modulo hash függvényektől kezdve a komplex kriptográfiai hash függvényekig, valamint a különböző ütközés kezelési módszerek alkalmazása mind hozzájárul ahhoz, hogy a hash táblák széles körben alkalmazhatók legyenek a modern számítástechnikában.

### 1.8.3. Hash táblák alkalmazása keresésre

A hash táblák kiemelkedően fontos adatstruktúrák a számítástechnikában, különösen akkor, amikor gyors keresésre van szükség. Ez az alfejezet a hash táblák különböző alkalmazásait tárgyalja a keresési műveletek során, bemutatva a gyakorlati szempontokat, előnyöket és a teljesítmény optimalizálásának módszereit.

#### Hash táblák alapelvei

A hash táblák alapelve az, hogy az adatokhoz tartozó kulcsokat egy hash függvény segítségével egyedi hash értékekké alakítják, majd ezeket az értékeket használják az adatok tárolására és visszakeresésére. A hash függvény által generált hash érték meghatározza az adat tárolási helyét a hash táblában.

#### Hash táblák szerkezete

Egy hash tábla két fő komponensből áll:
1. **Hash függvény**: Ez az algoritmus határozza meg, hogy egy adott kulcs milyen pozícióba kerül a hash táblában.
2. **Tábla (tároló struktúra)**: Ez az adatstruktúra, amely a hash értékek alapján tárolja az adatokat. Leggyakrabban egy fix méretű tömb vagy dinamikusan növekvő lista.

#### Hash táblák előnyei és hátrányai

**Előnyök**:
- **Gyors adatkeresés**: Általában a keresési, beszúrási és törlési műveletek időbonyolultsága $O(1)$.
- **Egyszerű implementáció**: A hash táblák viszonylag könnyen implementálhatók és kezelhetők.
- **Rugalmasság**: Különböző típusú adatok tárolására alkalmasak, például sztringek, egész számok stb.

**Hátrányok**:
- **Ütközések kezelése**: A hash függvények nem garantálják az egyedi hash értékeket, ezért ütközések kezelése szükséges.
- **Tábla méretének megválasztása**: A tábla méretének megválasztása kritikus, mivel túl kicsi tábla esetén sok ütközés történik, túl nagy tábla esetén pedig pazarló a memóriahasználat.
- **Nem rendezett adatok**: A hash táblák nem tartják fenn az elemek rendezett sorrendjét.

#### Hash táblák gyakorlati alkalmazásai

1. **Adatbázisok indexelése**:
   Az adatbázisokban a hash táblákat gyakran használják indexelésre, hogy gyors hozzáférést biztosítsanak a rekordokhoz. Az indexek hash táblával történő megvalósítása lehetővé teszi a gyors keresést az adatbázisban, különösen nagy mennyiségű adat esetén.

2. **Gyors keresési algoritmusok**:
   Olyan alkalmazásokban, ahol gyors keresés szükséges, mint például keresőmotorok, valós idejű rendszerek, vagy játékok, a hash táblák biztosítják a gyors hozzáférést és frissítést. A keresési műveletek időbonyolultsága általában $O(1)$, ami jelentős előnyt biztosít más adatstruktúrákhoz képest.

3. **Adatok deduplikálása**:
   A hash táblákat gyakran használják adatok deduplikálására, azaz az ismétlődő elemek kiszűrésére. Az egyes elemek hash értékét tárolják, és ha egy új elem érkezik, amelynek hash értéke már létezik, az elem ismétlődőnek tekinthető.

4. **Memóriakezelés és cache**:
   A memóriakezelés és cache rendszerekben a hash táblák hatékonyan használhatók az adatok gyors elérésére. Például, a CPU cache rendszerek gyakran használnak hash alapú struktúrákat az adatok gyors eléréséhez és tárolásához.

5. **Szótárak és nyelvfeldolgozás**:
   A hash táblák széles körben használatosak szótárak és egyéb nyelvfeldolgozó alkalmazásokban. Például, egy szövegben található szavak gyakoriságának meghatározására, vagy fordítási adatbázisok kezelése során.

#### Teljesítmény optimalizálás

A hash táblák teljesítményének optimalizálása számos tényezőtől függ, többek között a hash függvény minőségétől, a hash tábla méretétől, és az ütközés kezelési módszerektől.

**1. Hatékony hash függvény választása**:
Egy jó hash függvény biztosítja az adatok egyenletes eloszlását a hash táblában, minimalizálva az ütközések számát. A rossz hash függvények hajlamosak az ütközésekre és a klaszterizációra, ami jelentősen ronthatja a teljesítményt.

**2. A tábla méretének megfelelő megválasztása**:
A hash tábla méretének megválasztása kulcsfontosságú a teljesítmény optimalizálása szempontjából. A tábla mérete lehetőleg prím szám legyen, ami csökkenti az ütközések valószínűségét és segíti az egyenletes eloszlást.

**3. Dinamikus átméretezés**:
Dinamikus átméretezési technikákat alkalmazva a hash tábla mérete automatikusan nő vagy csökken a bejegyzések számának megfelelően. Ez biztosítja, hogy a tábla soha ne legyen túl terhelt vagy túl üres, fenntartva a jó teljesítményt.

**4. Optimalizált ütközés kezelési módszerek**:
A különböző ütközés kezelési módszerek alkalmazása, mint a láncolás vagy a nyílt címzés megfelelő kombinációja, segít minimalizálni az ütközések okozta teljesítményproblémákat. Például, a kettős hash módszer hatékonyabban kezeli az ütközéseket, mint az egyszerű lineáris próba.

**Példa C++ implementációra**:

Az alábbiakban bemutatunk egy hash tábla implementációt, amely dinamikus átméretezést és kettős hash függvényt alkalmaz.

```cpp
#include <iostream>
#include <vector>

class DoubleHashTable {
private:
    std::vector<int> table;
    int TABLE_SIZE;
    int PRIME;
    int current_size;

    int hash1(int key) {
        return key % TABLE_SIZE;
    }
    
    int hash2(int key) {
        return PRIME - (key % PRIME);
    }

    void resize() {
        std::vector<int> old_table = table;
        TABLE_SIZE = TABLE_SIZE * 2;
        table.clear();
        table.resize(TABLE_SIZE, -1);
        current_size = 0;
        
        for (int key : old_table) {
            if (key != -1) {
                insertItem(key);
            }
        }
    }

public:
    DoubleHashTable(int size, int prime) : TABLE_SIZE(size), PRIME(prime), current_size(0) {
        table.resize(TABLE_SIZE, -1);
    }
    
    void insertItem(int key) {
        if (current_size >= TABLE_SIZE / 2) {
            resize();
        }

        int index = hash1(key);
        if (table[index] != -1) {
            int i = 1;
            while (true) {
                int newIndex = (index + i * hash2(key)) % TABLE_SIZE;
                if (table[newIndex] == -1) {
                    table[newIndex] = key;
                    break;
                }
                i++;
            }
        } else {
            table[index] = key;
        }
        current_size++;
    }

    void displayHash() {
        for (int i = 0; i < TABLE_SIZE; i++) {
            if (table[i] != -1)
                std::cout << i << " --> " << table[i] << std::endl;
            else
                std::cout << i << std::endl;
        }
    }
};

int main() {
    int keys[] = {19, 27, 36, 10, 64};
    int n = sizeof(keys)/sizeof(keys[0]);
    int TABLE_SIZE = 7;
    int PRIME = 5;

    DoubleHashTable h(TABLE_SIZE, PRIME);
    for (int i = 0; i < n; i++)
        h.insertItem(keys[i]);

    h.displayHash();

    return 0;
}
```

Ez a példa egy dinamikusan átméretezhető hash tábla implementációját mutatja be, amely kettős hash függvényt használ az ütközések kezelésére. A tábla mérete megduplázódik, amikor a bejegyzések száma eléri a tábla méretének felét, biztosítva a folyamatosan jó teljesítményt.

#### Összegzés

A hash táblák rendkívül hatékony adatstruktúrák, amelyek számos alkalmazási területen használhatók a gyors keresési műveletekhez. A megfelelő hash függvény kiválasztása, a tábla méretének optimalizálása, és az ütközés kezelési módszerek kombinációja kulcsfontosságú a hash táblák teljesítményének maximalizálása érdekében. A hash táblák alkalmazása lehetővé teszi az adatok gyors és hatékony elérését, legyen szó adatbázisok indexeléséről, keresési algoritmusokról, memóriakezelésről vagy nyelvfeldolgozásról. Az optimalizált hash táblák alkalmazása jelentősen javítja az adatkezelési rendszerek teljesítményét és megbízhatóságát.