\newpage

## 6.2. Veszteségmentes adatkompressziós algoritmusok

A veszteségmentes adatkompressziós algoritmusok célja az, hogy az adatokat úgy csökkentsék méretükben, hogy azok eredeti formájukban bármikor visszaállíthatók legyenek. Ezek az algoritmusok kulcsfontosságúak olyan területeken, ahol a pontosság megőrzése elengedhetetlen, például szöveges dokumentumok, programkódok vagy bizonyos multimédiás fájlok esetén. A következő alfejezetekben részletesen bemutatjuk a legismertebb veszteségmentes adatkompressziós módszereket: a Huffman kódolást, az LZ77 és LZ78 algoritmusokat, valamint az LZW (Lempel-Ziv-Welch) algoritmust, amelyek mind különböző megközelítésekkel érik el a hatékony adatcsökkentést.

### 6.2.1. Huffman kódolás

A Huffman kódolás egy széles körben alkalmazott veszteségmentes adattömörítési algoritmus, amelyet David A. Huffman fejlesztett ki 1952-ben. Az algoritmus alapelve a karakterek gyakoriságán alapuló változó hosszúságú kódok használata, amely minimalizálja a kódolt üzenet átlagos hosszát. A Huffman kódolás az egyik leghatékonyabb módszer a szövegek tömörítésére, és számos alkalmazási területen, például fájlkompresszióban és adatátvitelben használják.

#### Alapelvek és Működés

A Huffman kódolás fő célja, hogy a leggyakrabban előforduló karakterekhez rövidebb kódokat rendeljen, míg a ritkábban előforduló karakterek hosszabb kódokat kapnak. Ezt a célt egy bináris fa (Huffman-fa) segítségével éri el, amelyet a karakterek gyakorisági eloszlása alapján épít fel.

##### Lépések a Huffman kódolásban

1. **Karakterek gyakoriságának meghatározása:** Az első lépés az, hogy megszámoljuk az üzenetben előforduló egyes karakterek gyakoriságát.
2. **Prioritási sor létrehozása:** A karaktereket gyakoriságuk alapján egy prioritási sorban (minimális prioritási sor) helyezzük el, ahol a legkisebb gyakoriságú karakter kerül a sor elejére.
3. **Huffman-fa építése:** A prioritási sor segítségével egy bináris fát építünk úgy, hogy a legkisebb gyakoriságú karaktereket összekapcsoljuk egy közös csomóponttal, amelynek gyakorisága a két karakter összegével egyenlő. Ezt a folyamatot addig folytatjuk, amíg egyetlen fa nem marad.
4. **Kódok hozzárendelése:** A bináris fa alapján minden karakterhez egyedi bináris kódot rendelünk, amelyet a fa gyökércsomópontjától a karaktert tartalmazó levélig vezető úttal határozunk meg. A balra vezető út általában '0'-t, míg a jobbra vezető út '1'-et jelent.

##### Példa Huffman kódolásra

Tegyük fel, hogy az üzenetünk a következő karakterekből áll: "ABRACADABRA". Először meghatározzuk a karakterek gyakoriságát:

- A: 5
- B: 2
- R: 2
- C: 1
- D: 1

Ezután létrehozzuk a prioritási sort és megépítjük a Huffman-fát. A folyamat a következő lépésekből áll:

1. Hozzunk létre levélcsomópontokat minden karakterhez, és helyezzük őket a prioritási sorba a gyakoriságuk alapján.
2. Vegyük ki a két legkisebb gyakoriságú csomópontot (C és D), és hozzunk létre egy új csomópontot, amelynek gyakorisága ezek összegével egyenlő (C + D). Helyezzük vissza az új csomópontot a prioritási sorba.
3. Folytassuk ezt a folyamatot, amíg egyetlen csomópont nem marad.

A Huffman-fa felépítése után a következő kódokat kapjuk:

- A: 0
- B: 101
- R: 100
- C: 1110
- D: 1111

Az üzenet kódolása a következőképpen néz ki: "010110011101111001011".

##### Pseudo-kód Huffman kódoláshoz

Az alábbiakban bemutatjuk a Huffman kódolás pseudo-kódját:

```
def huffman_encode(data):
    # Számoljuk meg a karakterek gyakoriságát
    frequency = {}
    for char in data:
        if char not in frequency:
            frequency[char] = 0
        frequency[char] += 1
    
    # Hozzunk létre egy prioritási sort (minimális prioritási sor)
    priority_queue = [[weight, [char, ""]] for char, weight in frequency.items()]
    heapq.heapify(priority_queue)
    
    while len(priority_queue) > 1:
        # Vegyük ki a két legkisebb gyakoriságú csomópontot
        lo = heapq.heappop(priority_queue)
        hi = heapq.heappop(priority_queue)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        # Hozzunk létre egy új csomópontot és tegyük vissza a prioritási sorba
        heapq.heappush(priority_queue, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    # Generáljuk a Huffman-kódokat
    huffman_codes = sorted(heapq.heappop(priority_queue)[1:], key=lambda p: (len(p[-1]), p))
    return huffman_codes

data = "ABRACADABRA"
huffman_codes = huffman_encode(data)
print("Huffman Codes:", huffman_codes)
```

#### Huffman kódolás hatékonysága és alkalmazásai

A Huffman kódolás hatékonyságát számos tényező befolyásolja, például a karakterek gyakorisági eloszlása és az üzenet hossza. Az optimális esetben, amikor az üzenetben erősen eltérő gyakoriságú karakterek vannak, a Huffman kódolás jelentős tömörítést érhet el. Azonban ha az összes karakter gyakorisága közel azonos, a tömörítés mértéke csökken.

**Fájlkompresszió:** A Huffman kódolás egyik leggyakoribb alkalmazási területe a fájlkompresszió, ahol a cél a fájlméret csökkentése. A ZIP és a gzip formátumok például Huffman kódolást használnak a DEFLATE algoritmus részeként.

**Adatátvitel:** Az adatátvitel során a Huffman kódolás segíthet a sávszélesség hatékonyabb kihasználásában azáltal, hogy csökkenti a továbbítandó adatmennyiséget.

**Kép- és hangkódolás:** A Huffman kódolás alkalmazható képek és hangok veszteségmentes tömörítésére is. A JPEG képformátum például Huffman kódolást használ a kvantált értékek tömörítésére.

**Adatbázisok:** Nagy méretű adatbázisok esetén a Huffman kódolás segíthet a tárolási hely csökkentésében és a lekérdezési sebesség javításában.

#### Huffman kódolás variánsai és továbbfejlesztései

Az eredeti Huffman algoritmus mellett számos variáns és továbbfejlesztés létezik, amelyek célja a kódolás hatékonyságának növelése és az alkalmazási területek bővítése.

**Adaptív Huffman kódolás:** Az adaptív Huffman kódolás (más néven dinamikus Huffman kódolás) egy olyan módszer, amely a karakterek gyakorisági eloszlását az adatfolyam feldolgozása közben folyamatosan frissíti. Ez lehetővé teszi, hogy az algoritmus alkalmazkodjon az üzenet változó karakterisztikáihoz, és így jobb tömörítési arányt érjen el dinamikus adatok esetén.

**Huffman-Storer-Szymanski algoritmus:** Ez a variáns a Huffman kódolás és az LZ77 algoritmus kombinációja, amelyet a tömörítési arány és a teljesítmény javítására terveztek. Az algoritmus először az LZ77 algoritmust használja a redundáns minták eltávolítására, majd a maradékot Huffman kódolással tömöríti.

#### Huffman kódolás implementáció C++-ban

Az alábbiakban egy Huffman kódolás C++ nyelvű implementációját mutatjuk be.

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>

struct Node {
    char character;
    int frequency;
    Node* left;
    Node* right;

    Node(char c, int freq) : character(c), frequency(freq), left(nullptr), right(nullptr) {}
};

// Összehasonlító függvény a prioritási sorhoz
struct Compare {
    bool operator()(Node* left, Node* right) {
        return left->frequency > right->frequency;
    }
};

void buildHuffmanTree(const std::string& data, std::unordered_map<char, std::string>& huffmanCodes) {
    std::unordered_map<char, int> frequency;
    for (char c : data) {
        frequency[c]++;
    }

    std::priority_queue<Node*, std::vector<Node*>, Compare> minHeap;
    for (auto pair : frequency) {
        minHeap.push(new Node(pair.first, pair.second));
    }

    while (minHeap.size() != 1) {
        Node* left = minHeap.top();
        minHeap.pop();
        Node* right = minHeap.top();
        minHeap.pop();

        Node* sum = new Node('\0', left->frequency + right->frequency);
        sum->left = left;
        sum->right = right;
        minHeap.push(sum);
    }

    Node* root = minHeap.top();
    std::string currentCode;
    std::function<void(Node*, const std::string&)> generateCodes = [&](Node* node, const std::string& code) {
        if (!node) return;
        if (node->character != '\0') {
            huffmanCodes[node->character] = code;
        }
        generateCodes(node->left, code + "0");
        generateCodes(node->right, code + "1");
    };

    generateCodes(root, currentCode);
}

int main() {
    std::string data = "ABRACADABRA";
    std::unordered_map<char, std::string> huffmanCodes;

    buildHuffmanTree(data, huffmanCodes);

    std::cout << "Huffman Codes:\n";
    for (const auto& pair : huffmanCodes) {
        std::cout << pair.first << ": " << pair.second << "\n";
    }

    return 0;
}
```

#### Összefoglalás

A Huffman kódolás egy hatékony veszteségmentes adattömörítési módszer, amely a karakterek gyakorisági eloszlását kihasználva minimalizálja a kódolt üzenet átlagos hosszát. A bináris fa segítségével történő kódolás lehetővé teszi a gyakori karakterek rövidebb, míg a ritkább karakterek hosszabb kódokkal történő ábrázolását. Az algoritmus széles körben alkalmazható különböző területeken, beleértve a fájlkompressziót, adatátvitelt, kép- és hangkódolást, valamint adatbázisok kezelését. Az eredeti algoritmus variánsai és továbbfejlesztései pedig tovább növelik a Huffman kódolás hatékonyságát és alkalmazási lehetőségeit.


### 6.2.2. LZ77 és LZ78

A LZ77 és LZ78 algoritmusok a veszteségmentes adatkompresszió terén kiemelkedő jelentőséggel bírnak, mivel ők alapozzák meg a modern szövegtömörítési technikákat. Mindkét algoritmus a Lempel és Ziv által kifejlesztett módszerek, amelyek hatékonyan használják ki az ismétlődő mintákat az adatcsökkentés érdekében. Az LZ77 és LZ78 algoritmusok alapelve, hogy az adatfolyamban előforduló ismétlődéseket, redundanciákat találják meg és tárolják, így csökkentve a szükséges tárolási helyet.

#### LZ77 algoritmus

Az LZ77 algoritmus 1977-ben került bemutatásra Abraham Lempel és Jacob Ziv által. Az algoritmus az adatfolyamot egy csúszó ablak segítségével dolgozza fel, amelynek segítségével visszatekint az adatfolyam korábbi részeire, hogy ismétlődéseket találjon.

##### Működési elv

Az LZ77 algoritmus lényege, hogy az adatfolyamot egy adott méretű ablakon belül vizsgálja. Az ablak két részből áll: a keresési tartományból (search buffer) és a nézési tartományból (look-ahead buffer). Az algoritmus a keresési tartományban keresi a leghosszabb, a nézési tartományban található mintát.

Példa egy adatfolyamra és a csúszó ablakra:

```
Adatfolyam: aabcaabcaa
Ablak mérete: 6 karakter
```

Az algoritmus megpróbálja megtalálni a leghosszabb egyezést az aktuális pozíciótól kezdve a nézési tartományban.

##### Kimeneti formátum

Az LZ77 algoritmus kimenete három komponensből áll: (offset, length, next character). Ezek az elemek azt jelölik, hogy hány pozícióval kell visszalépni (offset), az egyező karakterlánc hosszát (length), és a következő karaktert, amely nem része az egyezésnek.

Például az "aabcaabcaa" adatfolyam esetén az első néhány lépés a következő lehet:

1. `a` – nincs egyezés, (0, 0, 'a')
2. `a` – nincs egyezés, (0, 0, 'a')
3. `b` – nincs egyezés, (0, 0, 'b')
4. `c` – nincs egyezés, (0, 0, 'c')
5. `a` – egyezés az első karakterrel, (4, 1, 'a')
6. `a` – egyezés a második karakterrel, (4, 2, 'b')

A fenti példában látható, hogy az algoritmus fokozatosan növeli az egyezések hosszát, így tömörítve az adatokat.

##### Pseudo-kód LZ77 algoritmushoz

Az alábbiakban egy LZ77 algoritmus pseudo-kódja látható, amely segíthet megérteni a működését:

```
def lz77_compress(data):
    search_buffer_size = N
    look_ahead_buffer_size = L
    compressed_data = []
    i = 0
    
    while i < len(data):
        match = (0, 0, data[i])
        
        for j in range(1, min(search_buffer_size, i) + 1):
            length = 0
            
            while (length < look_ahead_buffer_size and
                   data[i - j + length] == data[i + length]):
                length += 1
                
            if length > match[1]:
                match = (j, length, data[i + length] if i + length < len(data) else '')
                
        compressed_data.append(match)
        i += match[1] + 1
    
    return compressed_data
```

#### LZ78 algoritmus

Az LZ78 algoritmus 1978-ban jelent meg, szintén Abraham Lempel és Jacob Ziv által. Ez az algoritmus eltér az LZ77-től abban, hogy nem használ csúszó ablakot, hanem egy dinamikusan bővülő szótárat (dictionary) épít az adatfolyam feldolgozása során.

##### Működési elv

Az LZ78 algoritmus az adatfolyamot karakterekre bontja és egy szótárat épít, amelyben minden egyes bejegyzés egy korábban látott karakterláncot tartalmaz. Amikor egy új karakterláncot talál, amely már szerepel a szótárban, az algoritmus az adott karakterláncot hozzáadja a szótárhoz, majd a kimenetben egy indexet és egy karaktert ad meg, amely az új karakterláncot azonosítja.

##### Kimeneti formátum

Az LZ78 algoritmus kimenete két komponensből áll: (index, character). Az index a szótár azon bejegyzésére utal, amely az eddig feldolgozott legnagyobb egyezést tartalmazza, a character pedig az új karakter, amely az új karakterláncot kiegészíti.

Példa az "aabcaabcaa" adatfolyamra:

1. `a` – nincs egyezés, (0, 'a')
2. `a` – egyezés az első karakterrel, (1, 'a')
3. `b` – nincs egyezés, (0, 'b')
4. `c` – nincs egyezés, (0, 'c')
5. `a` – egyezés az első karakterrel, (1, 'a')
6. `a` – egyezés az ötödik karakterrel, (4, 'a')

##### Pseudo-kód LZ78 algoritmushoz

Az alábbiakban egy LZ78 algoritmus pseudo-kódja látható:

```
def lz78_compress(data):
    dictionary = {}
    compressed_data = []
    current_string = ""
    dict_size = 1
    
    for character in data:
        new_string = current_string + character
        if new_string in dictionary:
            current_string = new_string
        else:
            if current_string:
                compressed_data.append((dictionary[current_string], character))
            else:
                compressed_data.append((0, character))
                
            dictionary[new_string] = dict_size
            dict_size += 1
            current_string = ""
    
    if current_string:
        compressed_data.append((dictionary[current_string], ""))
    
    return compressed_data
```

#### Összehasonlítás és Alkalmazások

Az LZ77 és LZ78 algoritmusok eltérő megközelítéseket alkalmaznak a tömörítéshez, így különböző előnyökkel és hátrányokkal rendelkeznek. Az LZ77 csúszó ablaka lehetővé teszi az ismétlődő minták azonnali felismerését, míg az LZ78 szótár alapú megközelítése dinamikusabb és rugalmasabb adattárolást tesz lehetővé.

Az LZ77 algoritmus előnyei közé tartozik, hogy egyszerűen implementálható és hatékony a kis méretű adatablakok esetén. Azonban nagy adatfolyamok esetén a teljesítmény csökkenhet, mivel a keresési tartomány mérete korlátozott.

Az LZ78 algoritmus előnye, hogy nagyobb adatfolyamok esetén is hatékony marad, mivel a szótár folyamatosan bővül és újra felhasználható bejegyzéseket tartalmaz. Hátránya lehet, hogy a szótár mérete növekedhet, ami memóriakezelési problémákhoz vezethet.

Mindkét algoritmus széles körben alkalmazható, különösen azokban az esetekben, amikor az adatok redundánsak és ismétlődő mintákat tartalmaznak. Az alábbiakban néhány gyakori alkalmazási területet mutatunk be mindkét algoritmus számára.

#### Alkalmazások

**Szövegtömörítés:** Az LZ77 és LZ78 algoritmusok széles körben alkalmazhatók szöveges adatok tömörítésére. A gyakran ismétlődő karakterláncok és szavak tömörítése jelentős helymegtakarítást eredményezhet.

**Adatátvitel:** Az adatátvitel során a tömörített adatok gyorsabban és hatékonyabban továbbíthatók. Az LZ77 és LZ78 algoritmusokat használó tömörítési technikák, mint például a DEFLATE (amelyet a ZIP és a gzip formátumok használnak), nagy mértékben csökkenthetik az adatátviteli időt és a sávszélesség használatot.

**Multimédiás fájlok:** Az LZ77 és LZ78 algoritmusok alkalmazhatók multimédiás fájlok, például képek, hangok és videók tömörítésére is. Bár ezek az algoritmusok alapvetően veszteségmentesek, kombinálhatók veszteséges tömörítési technikákkal is a még hatékonyabb adattömörítés érdekében.

**Adatbázisok:** Nagy adatbázisok tömörítésére is használhatók, különösen azokban az esetekben, amikor az adatok sok redundanciát tartalmaznak. A tömörítés csökkentheti a tárolási költségeket és javíthatja a hozzáférési sebességet.

#### LZ77 és LZ78 variánsai

Az eredeti LZ77 és LZ78 algoritmusok számos variánsa létezik, amelyek továbbfejlesztették és optimalizálták ezeket a módszereket különböző alkalmazási területekre. Néhány ilyen variáns:

**LZSS (Lempel-Ziv-Storer-Szymanski):** Az LZ77 egyik variánsa, amely az egyezések kódolásának hatékonyságát javítja azáltal, hogy az egyezéseket és a literálokat különböző módon tárolja. Ez csökkenti a tömörített adat méretét és növeli a tömörítés hatékonyságát.

**LZMW (Lempel-Ziv-Miller-Wegman):** Az LZ78 egyik variánsa, amely a szótár kezelésének módját módosítja, hogy növelje a tömörítési hatékonyságot. Az LZMW algoritmus jobban kihasználja a korábbi szótárbejegyzéseket, így hatékonyabb tömörítést ér el.

**LZW (Lempel-Ziv-Welch):** Az LZ78 egyik legismertebb variánsa, amelyet Terry Welch fejlesztett ki. Az LZW algoritmus széles körben használt a GIF képfájlformátumban és számos tömörítési alkalmazásban.

#### Implementáció

Az alábbiakban bemutatunk egy példakódot az LZ77 és LZ78 algoritmusok C++ nyelvű implementálására.

##### LZ77 implementáció C++-ban

```cpp
#include <iostream>
#include <vector>
#include <string>

struct LZ77Tuple {
    int offset;
    int length;
    char nextChar;
};

std::vector<LZ77Tuple> LZ77Compress(const std::string& data, int searchBufferSize, int lookAheadBufferSize) {
    std::vector<LZ77Tuple> compressedData;
    int i = 0;

    while (i < data.size()) {
        int matchLength = 0;
        int matchOffset = 0;

        for (int j = 1; j <= searchBufferSize && i - j >= 0; ++j) {
            int length = 0;
            while (length < lookAheadBufferSize && i + length < data.size() && data[i - j + length] == data[i + length]) {
                ++length;
            }
            if (length > matchLength) {
                matchLength = length;
                matchOffset = j;
            }
        }

        char nextChar = (i + matchLength < data.size()) ? data[i + matchLength] : '\0';
        compressedData.push_back({matchOffset, matchLength, nextChar});
        i += matchLength + 1;
    }

    return compressedData;
}

int main() {
    std::string data = "aabcaabcaa";
    int searchBufferSize = 6;
    int lookAheadBufferSize = 4;

    std::vector<LZ77Tuple> compressedData = LZ77Compress(data, searchBufferSize, lookAheadBufferSize);

    for (const auto& tuple : compressedData) {
        std::cout << "(" << tuple.offset << ", " << tuple.length << ", " << tuple.nextChar << ")\n";
    }

    return 0;
}
```

##### LZ78 implementáció C++-ban

```cpp
#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>

struct LZ78Tuple {
    int index;
    char nextChar;
};

std::vector<LZ78Tuple> LZ78Compress(const std::string& data) {
    std::unordered_map<std::string, int> dictionary;
    std::vector<LZ78Tuple> compressedData;
    std::string currentString;
    int dictSize = 1;

    for (char character : data) {
        std::string newString = currentString + character;
        if (dictionary.find(newString) != dictionary.end()) {
            currentString = newString;
        } else {
            int index = (currentString.empty()) ? 0 : dictionary[currentString];
            compressedData.push_back({index, character});
            dictionary[newString] = dictSize++;
            currentString.clear();
        }
    }

    if (!currentString.empty()) {
        compressedData.push_back({dictionary[currentString], '\0'});
    }

    return compressedData;
}

int main() {
    std::string data = "aabcaabcaa";

    std::vector<LZ78Tuple> compressedData = LZ78Compress(data);

    for (const auto& tuple : compressedData) {
        std::cout << "(" << tuple.index << ", " << tuple.nextChar << ")\n";
    }

    return 0;
}
```

### 6.2.3. LZW (Lempel-Ziv-Welch)

Az LZW (Lempel-Ziv-Welch) algoritmus egy veszteségmentes adattömörítési módszer, amely az 1978-ban Abraham Lempel és Jacob Ziv által kidolgozott LZ78 algoritmus továbbfejlesztése. Terry Welch 1984-ben módosította és optimalizálta ezt az algoritmust, amely azóta széles körben használt technológia lett különböző adattömörítési alkalmazásokban.

#### Alapelvek és működési mechanizmus
Az LZW algoritmus a szimbólumok sorozatát ahelyett, hogy egyetlen szimbólumot tömörítene, egy index táblázatba (szótárba) kódolja, amely a már látott szimbólumokat vagy szimbólumsorozatokat tartalmazza. Az algoritmus két fő lépésből áll: a kódolási és dekódolási fázisból.

##### Kódolás
1. **Szótár inicializálása**
   Az algoritmus kezdetekor egy előre definiált szótárat használ, amely tartalmazza az összes lehetséges egyedi input karaktert. Ez azt jelenti, hogy egy 8 bites kódrendszer esetén a kezdeti szótár 256 bejegyzésből áll (0-255), amely az egyes karakterekhez tartozik.

2. **Bejövő karakterek feldolgozása**
    - **Előtag keresése**: Az algoritmus elején együres előtag (üres sorozat) van kiválasztva.
    - **Karakter hozzáadása az előtaghoz**: Minden bejövő karakter hozzáadásra kerül az előtaghoz. Ha az így létrejött mintázat a szótárban található, akkor ez az új előtag.
    - **Szótármódosítás**: Ha a kombinált előtag és az aktuális karakter előfordulása nem található meg a szótárban, akkor az előtag kódját kimenetként adjuk ki, és az új kombinációt felvesszük a szótárba.

3. **Iteráció és szótár növekedése**
   Az előző lépéseket ismételjük mindaddig, amíg minden bejövő karaktert feldolgozunk. A szótár mérete folyamatosan növekszik, új bejegyzésekkel gazdagítva a lenyomozott mintázatokat.

##### Dekódolás
1. **Szótár inicializálása**
   Az algoritmus dekódoló partnere is ugyanazt a kezdeti szótárat használja, mint a kódoló oldal.

2. **Kódolt értékek feldolgozása**
    - **Első kód kezelése**: A bemenet első kódjára a szótárban található karakterként tekintünk, és az ezt követő kódokat különböző dekódolási lépésekkel dolgozzuk fel.
    - **Előtag és Karakter hozzáadása**: A korábban dekódolt előtaghoz hozzáadjuk a jelenlegi kódnak megfelelő karaktert, frissítve az előtagot és a szótárat is.

3. **Szótár kibővítése**
   Az előző kód alapján frissítjük a szótárat, új bejegyzéseket hozzáadva.

#### Algoritmus C++ kódja
A következő példa bemutatja az LZW algoritmus egyszerű C++ implementációját kódoláshoz.

```cpp
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

class LZW {
public:
    // Compress a string to a list of output symbols.
    static std::vector<int> compress(const std::string& input) {
        int dictSize = 256;
        std::unordered_map<std::string, int> dictionary;

        // Initialize dictionary with single characters
        for (int i = 0; i < 256; i++) {
            dictionary[std::string(1, i)] = i;
        }

        std::string w;
        std::vector<int> result;
        for (char c : input) {
            std::string wc = w + c;
            if (dictionary.count(wc)) {
                w = wc;
            } else {
                result.push_back(dictionary[w]);
                // Add wc to the dictionary.
                dictionary[wc] = dictSize++;
                w = std::string(1, c);
            }
        }

        // Output the code for w.
        if (!w.empty()) {
            result.push_back(dictionary[w]);
        }
        return result;
    }

    // Decompress a list of output ks to a string.
    static std::string decompress(const std::vector<int>& compressed) {
        int dictSize = 256;
        std::unordered_map<int, std::string> dictionary;

        // Initialize dictionary with single characters
        for (int i = 0; i < 256; i++) {
            dictionary[i] = std::string(1, i);
        }

        std::string w(1, compressed[0]);
        std::string result = w;
        for (size_t i = 1; i < compressed.size(); i++) {
            std::string entry;
            if (dictionary.count(compressed[i])) {
                entry = dictionary[compressed[i]];
            }
            else if (compressed[i] == dictSize) {
                entry = w + w[0];
            }
            result += entry;

            // Add w+entry[0] to the dictionary.
            dictionary[dictSize++] = w + entry[0];

            w = entry;
        }
        return result;
    }
};

int main() {
    std::string input = "TOBEORNOTTOBEORTOBEORNOT";
    std::vector<int> compressed = LZW::compress(input);

    std::cout << "Compressed: ";
    for (int code : compressed) {
        std::cout << code << " ";
    }
    std::cout << std::endl;

    std::string decompressed = LZW::decompress(compressed);
    std::cout << "Decompressed: " << decompressed << std::endl;

    return 0;
}
```

#### Algoritmus analízis
Az LZW algoritmus hatékonyan képes tömöríteni a visszatérő mintázatokat, anélkül, hogy különleges a priori tudásra lenne szükség az adatok eloszlásáról. Ez az alapvető előnye számos más tömörítési technikával szemben.

1. **Teljesítmény**:
    - **Kódolás során**: A kódolás O(n) időkomplexitású, ahol n az input karakterek száma. A gyakorlatban, a hasító táblázatos keresési stratégiák segítségével egy szinte konstans idejű hozzáadás és keresés is elérhető.
    - **Dekódolás során**: A dekódolás szintén O(n) időkomplexitású, ahol n a kódolt karakterek száma.

2. **Tárolási követelmények**:
    - Az LZW keresztül a tömörítés során növekvő méretű szótárból kiolvasott szimbólumokat használ, ezáltal szükséges lehet elegendő memória fenntartása a szótár számára.

3. **Tömörítési hatékonyság**:
    - Az LZW algoritmus kimagaslóan jól teljesít akkor, ha az input adatokban többször előforduló alzsúfoltságok helyezkednek el, ilyen esetekben a tömörítési ráta jelentősen növelhet.

#### Alkalmazások
Az LZW-t széles körben használják különböző adattömörítési alkalmazásokban:
1. **GIF képfájl formátum**:
   Az LZW az egyik legismertebb algoritmus, amelyet a GIF képfájl formátum használ. Az 1980-as évek végén ez az egyik első színes képfájl formátum volt, amely veszteségmentes adattömörítési technikát alkalmazott.

2. **UNIX compress parancs**:
   Az LZW algoritmus másik ismert alkalmazási területe a UNIX operációs rendszer "compress" parancsában található. Ez a parancs az LZW-t használja az adatok hatékony tömörítésére.

3. **PDF fájlok**:
   A PDF (Portable Document Format) dokumentum formátuma szintén néhány esetben használ LZW-t a szöveg és a képek tömörítésére, hogy csökkentse a dokumentumok méretét anélkül, hogy veszteséget okozna az információ minőségében.

4. **TIFF képfájl formátum**:
   A TIFF (Tagged Image File Format) képfájl formátum az LZW-t egy a több választható tömörítési algoritmus egyikeként használja, különösen akkor, ha veszteségmentes tömörítésre van szükség.

#### Összefoglalás
Az LZW egy rendkívül hatékony és széles körben használt adattömörítési algoritmus, amelyet széles körben alkalmaznak különböző fájlformátumok és adattömörítési technológiák. A szimbólumok visszatérő mintázatait és előfordulását kihasználó szótár alapú megközelítése miatt könnyen implementálható és gyorsan végrehajtható, ez teszi lehetővé a széleskörű felhasználását a gyakorlatban.
