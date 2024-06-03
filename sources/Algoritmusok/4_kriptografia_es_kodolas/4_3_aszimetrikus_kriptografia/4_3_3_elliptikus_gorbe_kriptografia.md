\newpage

## 3.2. Elliptikus görbe kriptográfia (ECC)

Az elliptikus görbe kriptográfia (ECC) az aszimmetrikus kulcsú kriptográfia egy modern és hatékony megközelítése, amely az elliptikus görbék matematikai tulajdonságait használja fel a titkosítási műveletek végrehajtására. Az ECC egyik legnagyobb előnye a hagyományos algoritmusokkal szemben, mint például az RSA, az, hogy ugyanazon biztonsági szint eléréséhez jelentősen kisebb kulcsméret szükséges. Ez a kulcsméretbeli előny különösen fontos a mai digitális világban, ahol az erőforrások optimalizálása és az adatbiztonság egyaránt elsődleges szempontok. Ebben a fejezetben megvizsgáljuk az elliptikus görbék matematikai alapjait, bemutatjuk az ECC alapú titkosítási algoritmusokat, és összehasonlítjuk az ECC-t az RSA-val, hogy megértsük, miért válik egyre népszerűbbé ezen kriptográfiai megoldás alkalmazása különböző területeken.

### Elliptikus görbék alapjai és matematikai alapok

#### Elliptikus Görbék Áttekintése

Elliptikus görbék a 19. században kezdtek matematikai érdeklődést kapni, és azóta számos területen alkalmazták őket, ideértve a számelméletet és az algebrai geometriát. Az elliptikus görbék egyenletei általában a következő formában írhatók le:

\[ y^2 = x^3 + ax + b \]

Itt \(a\) és \(b\) valós számok vagy az adott mező elemei, és meg kell felelniük a következő feltételnek, hogy a görbe ne legyen szinguláris (ne legyen csomópontja vagy hegyes pontja):

\[ 4a^3 + 27b^2 \neq 0 \]

Ez a feltétel biztosítja, hogy az elliptikus görbe sima, vagyis nincsenek szinguláris pontjai. Az elliptikus görbéket általában a valós számok vagy egy véges mező felett definiálják, de a kriptográfiában gyakran előnyben részesítik a véges mezőket.

#### Véges Mezők és Elliptikus Görbék

Egy véges mező \( F_q \) egy véges elemszámú halmaz, amelyen a szokásos aritmetikai műveletek (összeadás, kivonás, szorzás, osztás) értelmezhetők és eleget tesznek a mezők axiómáinak. Itt \( q \) egy prímszám hatványa, azaz \( q = p^m \), ahol \( p \) egy prímszám és \( m \) egy pozitív egész szám.

Az elliptikus görbéket a véges mezők felett úgy definiáljuk, hogy az egyenletük \( F_q \) elemire vonatkozik. Például, ha \( q = p \) (azaz, \( F_q \) egy prímmodulusú mező), akkor az elliptikus görbe egyenlete:

\[ y^2 \equiv x^3 + ax + b \pmod{p} \]

Az ilyen elliptikus görbék pontjai azokat a \((x, y)\) párokat tartalmazzák, amelyek kielégítik ezt az egyenletet a \( p \) modulusában. Az elliptikus görbék egyik fontos tulajdonsága a véges mezők felett, hogy ezek a pontok egy Abel-csoportot alkotnak a pontok összeadási műveletével.

#### Elliptikus Görbék Pontösszeadási Művelete

Az elliptikus görbék egyik legfontosabb jellemzője, hogy a pontok összeadására algebrai művelet definiálható. Ha \( P \) és \( Q \) két pont a görbén, akkor az összeadási művelet eredménye egy másik pont \( R = P + Q \) a görbén, úgy definiálható, mint:

1. **Identitás P és Q esetén**: Van egy speciális identitáspont, O, amelyhez ha bármely pontot hozzáadjuk, az eredeti pontot kapjuk vissza:
   \[ P + O = P \]

2. **Inverz**: Minden pont \( P = (x, y) \) esetén létezik egy \( -P = (x, -y) \) pont, hogy \( P + (-P) = O \).

3. **Különböző pontok összeadása**: Ha \( P = (x_1, y_1) \) és \( Q = (x_2, y_2) \) ismertek, ahol \( P \neq Q \):
   \[
   \lambda = \frac{y_2 - y_1}{x_2 - x_1} \\
   x_3 = \lambda^2 - x_1 - x_2 \\
   y_3 = \lambda(x_1 - x_3) - y_1
   \]
   Az eredmény \( R = (x_3, y_3) \).

4. **Azonos pontok összeadása**: Ha \( P = Q \):
   \[
   \lambda = \frac{3x_1^2 + a}{2y_1} \\
   x_3 = \lambda^2 - 2x_1 \\
   y_3 = \lambda(x_1 - x_3) - y_1
   \]
   Az eredmény \( R = (x_3, y_3) \).

#### További Matematikai Tulajdonságok

Az elliptikus görbék esetén a diszkrét logaritmus problémája (ECDLP) nagy nehézségének köszönhetően sajátos matematikai biztonságra építenek. Az ECDLP megfogalmazható a következőképpen: Adott két pont, \( P \) és \( Q \) az elliptikus görbén, találj olyan \( k \) egész számot, amelyre \( Q = kP \). Az ilyen \( k \) megtalálása számítógéppel hatékonyan nagyon nehéz, ez biztosítja az ECC biztonságát.

#### Gyakorlati Alkalmazások és Algoritmusok

Az ECC-t széles körben alkalmazzák a modern kriptográfiai rendszerekben, ideértve a kulcsváltási, üzenet aláírás és titkosítási protokollokat. Az ECC alapú algoritmusok hatékonyabbak lehetnek, és rövidebb kulcsokat használnak, mint más kriptográfiai módszerek (pl. RSA), miközben ugyanolyan szintű biztonságot nyújtanak.

#### Elliptikus Görbékkel Kapcsolatos Példakód (C++)

Az alábbi C++ kód egy egyszerű példa az elliptikus görbék pontösszeadására a \( y^2 = x^3 + ax + b \pmod{p} \) egyenlet alapján.

```cpp
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <cmath>

struct Point {
    long long x;
    long long y;
    bool is_infinity;

    Point(long long x = 0, long long y = 0, bool is_infinity = false)
        : x(x), y(y), is_infinity(is_infinity) {}
};

class EllipticCurve {
public:
    EllipticCurve(long long a, long long b, long long p)
        : a(a), b(b), p(p) {
        if ((4 * a * a * a + 27 * b * b) % p == 0) {
            throw std::invalid_argument("Invalid curve parameters");
        }
    }

    Point add_points(const Point &P, const Point &Q) {
        if (P.is_infinity) return Q;
        if (Q.is_infinity) return P;

        long long lambda;
        if (P.x == Q.x && P.y == Q.y) {
            // Point doubling
            lambda = (3 * P.x * P.x + a) * mod_inverse(2 * P.y, p) % p;
        } else {
            // Point addition
            lambda = (Q.y - P.y) * mod_inverse(Q.x - P.x, p) % p;
        }

        long long x3 = (lambda * lambda - P.x - Q.x) % p;
        long long y3 = (lambda * (P.x - x3) - P.y) % p;

        return Point(x3, y3);
    }

private:
    long long a, b, p;

    long long mod_inverse(long long num, long long modulus) {
        long long t = 0, newT = 1;
        long long r = modulus, newR = num;

        while (newR != 0) {
            long long quotient = r / newR;
            std::tie(t, newT) = std::make_tuple(newT, t - quotient * newT);
            std::tie(r, newR) = std::make_tuple(newR, r - quotient * newR);
        }

        if (r > 1) throw std::invalid_argument("Number is not invertible");
        if (t < 0) t += modulus;

        return t;
    }
};

int main() {
    // Example of elliptic curve: y^2 = x^3 + 2x + 3 over F_97
    EllipticCurve curve(2, 3, 97);

    Point P(3, 6);
    Point Q(3, 91); // Inverse point of P

    try {
        Point R = curve.add_points(P, Q);
        std::cout << "Result of P + Q: (" << R.x << ", " << R.y << ")\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
```

Ez a példa egy egyszerű elliptikus görbe implementációt mutat be, amely képes két pont összeadására egy adott görbén. Az elliptikus görbék kriptográfiai alkalmazásaihoz gyakran sokkal átfogóbb és optimáltabb kódra van szükség, azonban ez a példa jó alapot nyújt a kezdeti megértéshez.

### ECC alapú titkosítási algoritmusok

Az elliptikus görbe kriptográfia (ECC) hatékony és erőteljes módszert kínál a biztonságos kommunikációhoz kisebb kulcsméretekkel, mint a hagyományosan használt módszerek, mint az RSA. Ebben a fejezetben részletesen bemutatjuk az ECC alapú titkosítási algoritmusokat, figyelembe véve a matematikai alapokat, a gyakorlati alkalmazásokat és a konkrét algoritmusokat.

#### Elliptikus görbék matematikai alapjai

Az ECC alapja az elliptikus görbék matematikája. Egy elliptikus görbe egy sík algebrai görbe, amelynek általános egyenlete a következő alakú:

\[ y^2 = x^3 + ax + b \]

ahol \( a \) és \( b \) valós számok, és az egyenlet teljesíti azt a feltételt, hogy a jobboldalon lévő kifejezésnek nincs ismétlődő gyökere:

\[ 4a^3 + 27b^2 \neq 0. \]

Az elliptikus görbéken definiált műveletek, különösen a pontösszeadás és a skáláris szorzás, alapvető szerepet játszanak az ECC működésében. Ezek a műveletek biztosítják azt a matematikai struktúrát, amelyre az ECC alapul.

#### ECC kulcspárok generálása

Az ECC titkosítási módszerben minden felhasználónak van egy publikus és egy privát kulcsa. A publikus kulcs egy elliptikus görbe egy pontja, míg a privát kulcs egy skaláris érték. A privát kulcs egy véletlenszerűen választott egész szám \( d \), és a hozzá tartozó publikus kulcs a következőképpen generálható:

1. Válasszunk egy alappontot \( G \) az elliptikus görbén.
2. Szorozzuk meg az alappontot a privát kulccsal \( d \):

\[ P = d \cdot G. \]

Itt \( P \) lesz a publikus kulcs.

#### ECC alapú titkosítási séma: ElGamal ECC

Az elliptikus görbék használhatók jól ismert titkosítási sémák, például az ElGamal titkosítás esetén. Az ElGamal ECC titkosítási séma a következőképpen működik:

1. **Kulcsgenerálás:**
    - Válasszunk egy privát kulcsot \( d \).
    - Generáljuk a publikus kulcsot \( P = d \cdot G \).

2. **Titkosítás:**
    - Az adatok titkosításához válasszunk egy véletlen számot \( k \).
    - Számítsunk ki két pontot:
        - \( C_1 = k \cdot G \),
        - \( C_2 = M + k \cdot P \),
          ahol \( M \) a titkosítandó üzenet.

3. **Dekódolás:**
    - A titkosított üzenet \( (C_1, C_2) \) dekódolásához számítsuk ki az alábbi pontot:
        - \( M = C_2 - d \cdot C_1 \).

Ez a folyamat biztosítja, hogy az üzenetet csak az a személy tudja dekódolni, aki rendelkezik a privát kulccsal \( d \).

Példakód a fenti ElGamal ECC sémához C++ nyelven:

```cpp
#include <iostream>
#include <cassert>
#include "ecc_library.h" // Choose a proper ECC library

int main() {
    // Define elliptic curve parameters and base point
    EllipticCurve curve("secp256k1");

    // Generate private and public keys
    BigNumber d = curve.randomScalar();
    Point G = curve.getBasePoint();
    Point P = d * G;

    // The message as a point on the curve (for simplicity)
    Point M = curve.encodeMessage("Hello, ECC!");

    // Encrypt the message
    BigNumber k = curve.randomScalar();
    Point C1 = k * G;
    Point C2 = M + k * P;

    // Display encrypted message
    std::cout << "Encrypted Message (C1, C2):" << std::endl;
    std::cout << "C1: " << C1 << std::endl;
    std::cout << "C2: " << C2 << std::endl;

    // Decrypt the message
    Point decryptedM = C2 - d * C1;

    // Decode the point to retrieve the original message
    std::string decryptedMessage = curve.decodeMessage(decryptedM);
    std::cout << "Decrypted Message: " << decryptedMessage << std::endl;

    assert(decryptedMessage == "Hello, ECC!");

    return 0;
}
```

#### ECC alapú titkosítás alkalmazásai

Az ECC titkosítás számos gyakorlati alkalmazással bír, többek között:

1. **Digitális aláírások:** Az ECC alapú digitális aláírás algoritmusok, mint az ECDSA (Elliptic Curve Digital Signature Algorithm), széles körben alkalmazottak az elektronikus aláírások területén. Az ECDSA hatékonyabb és biztonságosabb, mint az RSA-alapú DSA (Digital Signature Algorithm) azonos bitmérettel.

2. **Kulcscsere protokollok:** Az elliptikus görbék alapú Diffie-Hellman kulcscsere protokoll (ECDH) biztonságos módszert kínál két fél közötti kulcscserére, minimalizálva a támadások esélyét.

3. **Titkosított adatátvitel:** Az ECC segítségével hatékonyan titkosíthatók az adatátviteli csatornák, például a HTTPS kapcsolat, ami biztosítja a webes kommunikáció biztonságát.

4. **IoT (Internet of Things):** Az ECC kisebb kulcsméretei miatt különösen alkalmas alacsony erőforrásigényű eszközökhöz, mint az IoT eszközök, amelyek gyakran korlátozott számítási és tárolási kapacitással rendelkeznek.

#### ECC és RSA összehasonlítása

Az ECC és az RSA közötti összehasonlítás rávilágít az ECC előnyeire:

1. **Kulcsméret és biztonság:** Az ECC kisebb kulcsméretekkel éri el ugyanazt a biztonságot, mint az RSA. Például egy 256 bites ECC kulcs hasonló biztonságot nyújt, mint egy 3072 bites RSA kulcs.

2. **Számítási igény:** Az ECC műveletek kevesebb számítási kapacitást igényelnek, ezáltal gyorsabbak és kevesebb energiát fogyasztanak, ami ideális alacsony erőforrás-igényű alkalmazásokhoz.

3. **Tárhely igény:** Az ECC kisebb kulcsméretei kevesebb tárolóhelyet igényelnek, ami jelentős előnyt jelent a korlátozott memóriakapacitású rendszerekben.

#### Matematikai alapok: ECC és RSA

**RSA:**

Az RSA egy 1977-ben kifejlesztett, aszimmetrikus kulcsú kriptográfiai algoritmus, amely az egész számok faktorizálásán alapul. Az RSA kulcs generáláshoz két nagy prímszámot választanak ki, ezeket szorozzák, majd a kapott eredményt használják a publikus és a privát kulcs generálásához.

A kulcspár generálás menete:
1. Válassz két nagy prímszámot: \( p \) és \( q \).
2. Számítsd ki \( n \)-t: \( n = p \times q \).
3. Számítsd ki az Euler-függvényt: \( \varphi(n) = (p - 1) \times (q - 1) \).
4. Válassz egy \( e \) értéket, mely relatív prím \( \varphi(n) \)-hez (általában \( e = 65537 \)).
5. Számítsd ki \( d \)-t, amely az \( e \) multiplikatív inverze modulo \( \varphi(n) \) (tehát \( e \times d \equiv 1 \pmod{\varphi(n)} \)).

Ekkor a publikus kulcs a \((n, e)\) pár, a privát kulcs pedig a \((n, d)\) pár.

**ECC:**

Az elliptikus görbe kriptográfia a következő formában definiált elliptikus görbéken alapul: \( y^2 = x^3 + ax + b \), ahol \( 4a^3 + 27b^2 \neq 0 \) biztosítja, hogy a görbe nem degenerált. A kriptográfiai műveletek az elliptikus görbe pontjain definiált algebrai műveleteken alapulnak.

Az ECC kulcs generálás menete:
1. Válassz egy elliptikus görbét és egy alappontot \( G \) a görbén.
2. Válassz egy véletlenszerű privát kulcsot \( d \) (ez egy nagy szám).
3. Számítsd ki a publikus kulcsot \( P \) mint \( P = dG \) (ahol \( G \)egy pont és \( dG \) a pontoszorozás).

Másképpen mondva, az ECC az elliptikus görbék csoport-elméleti tulajdonságait használja a titkosítás és dekriptálás megvalósításához.

#### Biztonság és kulcshossz

**RSA Biztonság:**

Az RSA biztonsága az egész számok faktorizálásának nehézségén alapul. Ahhoz, hogy egy RSA kulcsot feltörjenek, a támadónak meg kell találnia a privát kulcsot \( d \), amely csak akkor lehetséges, ha a rendkívül nagy \( n \) értéket a prímtényezőire tudja bontani.

Az RSA esetében a jelenleg ajánlott kulcshossz legalább 2048-bit. A kvantumszámítógépek fejlődésével azonban az RSA sebezhetősége egyre inkább előtérbe kerül, mivel a Shor-algoritmus lehetővé teszi a kvantumszámítógépek számára, hogy hatékonyan faktorizálják az egész számokat.

**ECC Biztonság:**

Az ECC biztonsága az elliptikus görbéken alapuló diszkrét logaritmus probléma nehézségén alapul. Egy equivalens szintű biztonság eléréséhez az ECC szignifikánsan kisebb kulcshosszokat használhat, mint az RSA.

Például:
- Egy 256-bites ECC kulcs biztonságilag equivalens egy 3072-bites RSA kulccsal.
- Egy 384-bites ECC kulcs biztonságilag equivalens egy 7680-bites RSA kulccsal.

Az ECC kisebb kulcsméretei és az alacsonyabb kriptográfiai számítási igényei különösen előnyösek az erőforrás-korlátolt környezetekben, mint például az IoT eszközök és mobilok eszközök.

#### Teljesítmény

**RSA Teljesítmény:**

Az RSA titkosítási és aláírási sebessége gyakran lassúbb az ECC-hoz képest, különösen nagyobb kulcshosszak esetén. Az RSA dekriptálási folyamat és aláírás-ellenőrzés számítási igénye jelentős erőforrásokat igényel nagy kulcsméreteknél. Mindemellett az RSA műveletek párhuzamosíthatók, ami hasznos lehet a nagy teljesítményű számítási környezetekben.

**ECC Teljesítmény:**

Az ECC jelentősen kisebb kulcshosszokat igényel, ami gyorsabb és kevésbé erőforrás-igényes műveleteket eredményez. Az elliptikus görbéken végzett matematikai műveletek (pl. skalármultiplikáció) hatékonyabbak és kevesebb számítási időt igényelnek, ami gyorsabb titkosítást, aláírást, dekriptálást és ellenőrzést eredményez.

#### Alkalmazási területek

Mindkét algoritmus széleskörűen alkalmazható, de különböző környezetekben és különböző célokra használják őket.

**RSA:**
- Digitális aláírások
- SSL/TLS protokollok
- Email titkosítás (pl. PGP)
- Szoftver licenszelés és hitelesítés

**ECC:**
- SSL/TLS protokollok (különösen PFS-t igénylő környezetekben)
- Mobil és IoT eszközök (erőforrás-kímélő megoldások)
- Blokk-lánc technológiák és kriptovaluták (pl. Bitcoin)
- Kormányzati és katonai rendszerek

### Konklúzió

Az ECC és az RSA összehasonlításakor figyelembe kell venni a különféle kritériumokat, mint például a kulcsméretek, a biztonsági szint, a teljesítmény és az alkalmazási területek. Bár mindkét algoritmus robusztus és jól bevált, az ECC növekvő népszerűsége az alacsonyabb kulcsméreteinek és nagyobb hatékonyságának köszönhetően egyre inkább előtérbe kerül. Ahogy a számítástechnika fejlődik, és a kvantumszámítógépek valósággá válnak, az ECC alkalmazása várhatóan még inkább elterjedtté válik, köszönhetően a kisebb kulcsméretek és az erőforrás-takarékos műveletek által nyújtott előnyöknek. Mindazonáltal az RSA továbbra is fontos szerepet tölt be, különösen olyan területeken, ahol a külső tényezők hatása miatt az ECC bevezetése nem lehetséges vagy nem szükséges.


