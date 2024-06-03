\newpage

## 4.2. Klasszikus hash algoritmusok

### 4.2.1. MD5

Az MD5 (Message-Digest Algorithm 5) egy széles körben ismert és használt hash algoritmus, amelyet Ronald Rivest fejlesztett ki 1991-ben. Az MD5 célja, hogy egy 128 bites (16 bájt) hash értéket generáljon egy tetszőleges hosszúságú bemeneti üzenetből. Bár az MD5-t eredetileg biztonságos hash algoritmusnak tervezték, az évek során több súlyos sebezhetőséget fedeztek fel benne, amelyek miatt ma már nem ajánlott kriptográfiai célokra. Mindazonáltal az MD5 továbbra is széles körben használatos nem kriptográfiai alkalmazásokban, például ellenőrzőösszegek készítésére és adatintegritás ellenőrzésére.

#### Történeti háttér

Az MD5 az MD4 hash algoritmus utódjaként jött létre, amelyet szintén Ronald Rivest fejlesztett ki. Az MD4 gyengeségei és a hash funkciók iránti megnövekedett biztonsági igények miatt az MD5-t úgy tervezték, hogy kiküszöbölje az MD4 hibáit és nagyobb ellenállást biztosítson a kriptográfiai támadásokkal szemben. Az MD5 azóta számos alkalmazásban és protokollban jelent meg, beleértve az SSL/TLS-t, a digitális aláírásokat és a fájlok integritásának ellenőrzését.

#### Az MD5 algoritmus működése

Az MD5 egy iteratív, blokkalapú hash algoritmus, amely az alábbi fő lépésekből áll: előfeldolgozás, üzenet feldarabolása, inicializáció, körök ismétlése és végső összeállítás.

##### Előfeldolgozás

Az előfeldolgozás célja, hogy az üzenetet olyan formára alakítsa, amely kompatibilis az MD5 algoritmus belső működésével. Az előfeldolgozás három lépésből áll:

1. **Padding (kitöltés)**: Az eredeti üzenet hosszát olyan módon egészítjük ki, hogy az üzenet hossza 448 legyen (mod 512). A kitöltés mindig egy '1' bittel kezdődik, amelyet nullák követnek, amíg az üzenet hossza el nem éri a kívánt értéket. Ez a padding biztosítja, hogy az üzenet hossza pontosan 512 bit (64 bájt) többszöröse legyen, kivéve az utolsó 64 bitet, amely a következő lépésben kerül hozzáadásra.

2. **Hossz hozzáadása**: A padding után az üzenethez hozzáadjuk az eredeti üzenet hosszát 64 bites kis-endian formátumban. Ez a lépés biztosítja, hogy az üzenet eredeti hosszára vonatkozó információ megőrződik az előfeldolgozás során.

##### Üzenet feldarabolása

A kitöltött üzenetet 512 bites blokkokra bontjuk. Minden blokkot külön-külön dolgozunk fel az MD5 körök során.

##### Inicializáció

Az MD5 algoritmus négy 32 bites állapotregisztert használ, amelyek kezdeti értékeit előre meghatározott konstansokkal inicializáljuk:

- A: 0x67452301
- B: 0xEFCDAB89
- C: 0x98BADCFE
- D: 0x10325476

Ezek az állapotregiszterek a hash érték részleges eredményeit tárolják minden egyes blokk feldolgozása során.

##### Körök ismétlése

Az MD5 algoritmus négy fő körből áll, mindegyik kör 16 lépésből áll. Ezek a körök különböző nemlineáris függvényeket használnak, hogy az üzenet blokkjait keverjék és keverjék a hash értéket. A négy kör a következő:

1. **Kör 1**: A nemlineáris függvény \( F \) használata.
   - \( F(B, C, D) = (B \land C) \lor (\neg B \land D) \)
2. **Kör 2**: A nemlineáris függvény \( G \) használata.
   - \( G(B, C, D) = (B \land D) \lor (C \land \neg D) \)
3. **Kör 3**: A nemlineáris függvény \( H \) használata.
   - \( H(B, C, D) = B \oplus C \oplus D \)
4. **Kör 4**: A nemlineáris függvény \( I \) használata.
   - \( I(B, C, D) = C \oplus (B \lor \neg D) \)

Minden egyes körben a bemenetek keverednek az állapotregiszterekkel, és előre meghatározott forgatási és hozzáadási műveletek történnek, amelyek növelik a hash érték összetettségét és biztonságát.

##### Végső összeállítás

Miután minden blokkot feldolgoztunk, a négy állapotregisztert egyesítjük, hogy megkapjuk a végső 128 bites hash értéket. Az állapotregiszterek végső értékei egyesítve adják az üzenet MD5 hash-ét.

#### Biztonsági elemzés

Az MD5 széles körben használt volt az 1990-es és 2000-es évek elején, de az idő előrehaladtával számos biztonsági hiányosságot fedeztek fel. Az alábbiakban néhány főbb gyengeséget és támadást említünk:

1. **Ütközések (Collisions)**: Az MD5 egyik legjelentősebb gyengesége az, hogy viszonylag könnyű ütközéseket találni, ahol két különböző bemenet ugyanazt a hash értéket generálja. A kínai kutatók által 2004-ben felfedezett ütközési támadás jelentős figyelmet keltett, és rámutatott az MD5 ezen alapvető gyengeségére.

2. **Preimage és Second Preimage támadások**: Bár az MD5 ellen preimage és second preimage támadások még nem praktikusak nagy léptékben, a hash algoritmus gyengeségei miatt ezeket a támadásokat sem lehet kizárni hosszú távon.

3. **Gyorsabb ütközés-keresési technikák**: Az MD5 ellen számos optimalizált ütközés-keresési technikát fejlesztettek ki, amelyek tovább csökkentették az algoritmus biztonsági szintjét. Ezek a technikák lehetővé tették, hogy még gyorsabban találjanak ütközéseket, ezzel veszélyeztetve az MD5 alapú rendszerek biztonságát.

#### Használat és ajánlások

Az MD5 használata ma már nem ajánlott biztonsági célokra, különösen olyan alkalmazásokban, ahol az adatintegritás és a hitelesítés kritikus fontosságú. Az olyan biztonsági protokollok, mint az SSL/TLS, és a digitális aláírások már áttértek erősebb hash algoritmusokra, mint például a SHA-256 és a SHA-3.

Az MD5 azonban továbbra is hasznos lehet nem kriptográfiai alkalmazásokban, például fájlok ellenőrzőösszegeinek készítésére és adat integritásának ellenőrzésére olyan esetekben, ahol a biztonsági szempontok kevésbé kritikusak.

### Példa C++ implementáció

Az alábbiakban bemutatunk egy egyszerű C++ implementációt az MD5 algoritmusra. Bár ez az implementáció nem optimalizált, jól illusztrálja az MD5 működésének alapjait.

```cpp
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

class MD5 {
public:
    MD5();
    void update(const uint8_t* input, size_t length);
    void finalize();
    std::vector<uint8_t> digest() const;

private:
    void transform(const uint8_t block[64]);
    void encode(uint8_t* output, const uint32_t* input, size_t length);
    void decode(uint32_t* output, const uint8_t* input, size_t length);

    uint32_t state[4];
    uint32_t count[2];
    uint8_t buffer[64];
    uint8_t digest_[16];

    static const uint8_t PADDING[64];
    static const char HEX_DIGITS[16];
};

const uint8_t MD5::PADDING[64] = { 0x80 };
const char MD5::HEX_DIGITS[16] = "0123456789abcdef";

MD5::MD5() {
    state[0] = 0x67452301;
    state[1] = 0xEFCDAB89;
    state[2] = 0x98BADCFE;
    state[3] = 0x10325476;
    count[0] = count[1] = 0;
}

void MD5::update(const uint8_t* input, size_t length) {
    size_t index = count[0] / 8 % 64;
    if ((count[0] += length << 3) < (length << 3))
        count[1]++;
    count[1] += length >> 29;
    size_t firstpart = 64 - index;
    size_t i = 0;
    if (length >= firstpart) {
        memcpy(&buffer[index], input, firstpart);
        transform(buffer);
        for (i = firstpart; i + 63 < length; i += 64)
            transform(&input[i]);
        index = 0;
    }
    memcpy(&buffer[index], &input[i], length - i);
}

void MD5::finalize() {
    uint8_t bits[8];
    encode(bits, count, 8);
    size_t index = count[0] / 8 % 64;
    size_t padLen = (index < 56) ? (56 - index) : (120 - index);
    update(PADDING, padLen);
    update(bits, 8);
    encode(digest_, state, 16);
}

std::vector<uint8_t> MD5::digest() const {
    return std::vector<uint8_t>(digest_, digest_ + 16);
}

void MD5::transform(const uint8_t block[64]) {
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3], x[16];
    decode(x, block, 64);

    // MD5 transformation rounds (not fully shown here for brevity)

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
}

void MD5::encode(uint8_t* output, const uint32_t* input, size_t length) {
    for (size_t i = 0, j = 0; j < length; i++, j += 4) {
        output[j] = (input[i] & 0xff);
        output[j + 1] = ((input[i] >> 8) & 0xff);
        output[j + 2] = ((input[i] >> 16) & 0xff);
        output[j + 3] = ((input[i] >> 24) & 0xff);
    }
}

void MD5::decode(uint32_t* output, const uint8_t* input, size_t length) {
    for (size_t i = 0, j = 0; j < length; i++, j += 4) {
        output[i] = (input[j]) | (input[j + 1] << 8) | (input[j + 2] << 16) | (input[j + 3] << 24);
    }
}

int main() {
    MD5 md5;
    const std::string message = "The quick brown fox jumps over the lazy dog";
    md5.update(reinterpret_cast<const uint8_t*>(message.c_str()), message.size());
    md5.finalize();
    std::vector<uint8_t> hash = md5.digest();
    
    std::cout << "MD5 hash: ";
    for (uint8_t byte : hash) {
        std::cout << MD5::HEX_DIGITS[(byte >> 4) & 0xf];
        std::cout << MD5::HEX_DIGITS[byte & 0xf];
    }
    std::cout << std::endl;

    return 0;
}
```

### Összefoglalás

Az MD5 egy történelmileg jelentős hash algoritmus, amely széles körben elterjedt és alkalmazott volt, de ma már nem ajánlott kriptográfiai célokra a felfedezett gyengeségei miatt. Az MD5 példája jól mutatja, hogyan fejlődtek a hash algoritmusok és a biztonsági követelmények az idők során, valamint hangsúlyozza a folyamatos kriptográfiai kutatások és fejlesztések fontosságát a biztonságosabb hash algoritmusok megalkotásához.

### 4.2.2. SHA-1

A Secure Hash Algorithm 1, röviden SHA-1, egy kriptográfiai hash függvény, amelyet először 1993-ban publikált az Amerikai Nemzetbiztonsági Ügynökség (NSA) és az Nemzeti Szabványügyi és Technológiai Intézet (NIST). Az SHA-1 célja, hogy egy bemenethez egy egyedi, fix hosszúságú, 160 bites hash értéket rendeljen, amely elvileg nehezen visszafejthető és kimondottan alkalmas integritás ellenőrzésére. Az évek során azonban a kriptoanalitikai technikák fejlődésével kiderült, hogy SHA-1 számos gyengeséggel rendelkezik, amelyek komolyan megkérdőjelezik biztonságát. Ebben a fejezetben megvizsgáljuk az SHA-1 hash függvény működési elveit és feltárjuk azokat a gyengeségeket, amelyek végül az algoritmus elavulásához vezettek.

#### Működés és gyengeségek

A Secure Hash Algorithm 1 (SHA-1) egy kriptográfiai hash függvény, amelyet az Egyesült Államok Nemzeti Biztonsági Ügynöksége (NSA) fejlesztett ki, és amelyet a National Institute of Standards and Technology (NIST) hivatalosan közzétett 1993-ban. A SHA-1 egy 160 bites hash értéket állít elő, és sok éven keresztül széles körben használták különböző biztonsági protokollokban, mint például a Transport Layer Security (TLS) és a Secure Sockets Layer (SSL).

#### SHA-1 Működése

A SHA-1 algoritmus bemeneti adatoknak tetszőleges hosszúságú üzeneteit fogadja el, és fix hosszúságú, 160 bites (20 bájt) kivonatot generál belőlük. A következő lépésekben mutatjuk be a SHA-1 működését:

##### 1. Padding

Az üzenetet először ki kell egészíteni (padding) annak érdekében, hogy a hossza 512-gyel teljes mértékben osztható legyen. Az üzenet kiegészítése az alábbi módon történik:
1. Legyen az eredeti üzenet hossza bitben kifejezve L.
2. Az üzenethez először hozzáadunk egyetlen '1' bitet.
3. Az üzenethez hozzáadunk megfelelő számú '0' bitet, hogy a végső hossz mod 512 = 448 legyen.
4. Az utolsó szakaszban az eredeti üzenet L hosszát egy 64 bit hosszú bináris értékként hozzáadjuk a végeredményhez.

Ezt követően a padding-elés során az üzenet hossza 512 bájt többszöröse lesz.

##### 2. Inicializációs Vektor

A SHA-1 öt állapotregisztert használ, amelyeket A, B, C, D és E néven neveznek meg. Ezeket az alábbi értékekkel inicializáljuk:

\[
\begin{align*}
A_0 &= 0x67452301 \\
B_0 &= 0xEFCDAB89 \\
C_0 &= 0x98BADCFE \\
D_0 &= 0x10325476 \\
E_0 &= 0xC3D2E1F0
\end{align*}
\]

##### 3. Feldolgozási Blokkok és Kerekek

Az üzenetet 512 bit hosszúságú blokkokra osztjuk fel, és mindegyik blokkot 80 kerekben dolgozzuk fel. A hash érték előállítása során az algoritmus több logikai függvényt és kettős forgatási műveleteket használ. Minden blokk feldolgozása az alábbi lépésekben történik:

1. A blokkot 16 darab 32 bites szóra osztjuk fel (az első 16 word \( W_t \)).
2. A fennmaradó 64 szót az alábbi képlettel generáljuk:
   \[ W_t = (W_{t-3} \oplus W_{t-8} \oplus W_{t-14} \oplus W_{t-16}) \ll 1 \]

3. Az egyes körökben a következő logikai függvényeket használjuk:

\[
\begin{align*}
& \text{t=0} \text{ to } \text{t=19}: \quad F(B,C,D) = (B \land C) \lor (\neg B \land D) \\
& \text{t=20} \text{ to } \text{t=39}: \quad F(B,C,D) = B \oplus C \oplus D \\
& \text{t=40} \text{ to } \text{t=59}: \quad F(B,C,D) = (B \land C) \lor (B \land D) \lor (C \land D) \\
& \text{t=60} \text{ to } \text{t=79}: \quad F(B,C,D) = B \oplus C \oplus D \\
\end{align*}
\]

4. A kör állandói:
   \[
   \begin{align*}
   & \text{t=0} \text{ to } \text{t=19}: \quad K = 0x5A827999 \\
   & \text{t=20} \text{ to } \text{t=39}: \quad K = 0x6ED9EBA1 \\
   & \text{t=40} \text{ to } \text{t=59}: \quad K = 0x8F1BBCDC \\
   & \text{t=60} \text{ to } \text{t=79}: \quad K = 0xCA62C1D6 \\
   \end{align*}
   \]

5. Körönkénti frissítési képlet:
   \[
   \begin{align*}
   TEMP &= (A \ll 5) + F(B, C, D) + E + W_t + K \\
   E &= D \\
   D &= C \\
   C &= B \ll 30 \\
   B &= A \\
   A &= TEMP \\
   \end{align*}
   \]

6. Az 512 bites minden blokk feldolgozása után az álallapot regiszterek végső frissítése:
   \[
   \begin{align*}
   A_0 &= A_0 + A \\
   B_0 &= B_0 + B \\
   C_0 &= C_0 + C \\
   D_0 &= D_0 + D \\
   E_0 &= E_0 + E \\
   \end{align*}
   \]

##### 4. Végső Hash Érték

A végső hash érték az \( A_0, B_0, C_0, D_0, E_0 \) láncok együttes értéke. Ez képezi a végső 160 bit hosszúságú SHA-1 kivonatot.

#### Gyengeségek

Noha rendkívül népszerű volt évtizedeken keresztül, a SHA-1 mára elavult és nem biztonságosnak tekinthető számos jelentős kriptográfiai támadási módszer miatt. Az SHA-1 elleni jellegzetes támadások a következők:

##### 1. Kollíziós támadások

A SHA-1 egyik legjelentősebb gyengesége a kollíziós támadásokkal szembeni sebezhetősége. Egy kollíziós támadás esetén két eltérő bemenetet próbálnak találni, amelyek ugyanazt a hash értéket adják. Az első ilyen támadást 2005-ben fedezték fel, amelynek során Stefan Lucks, Xiaoyun Wang, Yiqun Lisa Yin bejelentették, hogy egy kollízió 2^69 művelettel található, amely lényegesen kevesebb, mint a brute-force támadás esetén várt 2^80.

##### 2. Shattered Támadás

2017 februárjában a Google és a CWI Institute bejelentettek egy gyakorlati támadást a SHA-1 ellen, amelyet SHAttered-nek neveztek el. Ez volt az első nyilvánosan elérhető példája egy gyakorlatban végrehajtott kollíziós támadásnak, amely két különböző PDF fájlt generált, amelyek ugyanazt a SHA-1 hash értéket adták vissza. A támadás végrehajtása körülbelül 9'223'372'036'854'775'808 SHA-1 számítást igényelt (2^63.1).

##### 3. Preimage támadások

Egy preimage támadással egy adott hash értékhez egy bemeneti szöveget próbálnak találni. Bár az SHA-1 elleni preimage támadások jelenleg kevésbé hatékonyak, a bevett biztonsági gyakorlatok szerint hosszú távon sem ajánlott az SHA-1 használata fontos kriptográfiai alkalmazásokban.

##### 4. Hossz kiterjesztési támadások

Mivel a SHA-1 függvény struktúrája a Merkle-Damgård konstrukción alapszik, hossz kiterjesztési támadások is alkalmazhatók ellene. Ezek a támadások lehetővé teszik a támadóknak, hogy ismert hash értékre alapozva új érvényes hash értéket generáljanak az üzenet végének manipulálásával.

#### Összefoglalás

Az SHA-1 hash függvény sokáig széleskörben használt algoritmus volt a kriptográfiai alkalmazásokban, azonban a kollíziós és más típusú támadások felfedezése miatt ma már rendkívül elavultnak és nem biztonságosnak tekinthető. Az informatikai biztonsági közösség egyértelműen azt javasolja, hogy az SHA-1 helyett erősebb hash algoritmusokra, például az SHA-256 vagy SHA-3-ra térjünk át.

