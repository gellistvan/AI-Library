\newpage

## 5.2. Digitális aláírás algoritmusok

A digitális aláírások az informatikai biztonság egyik kulcsfontosságú elemei, melyek biztosítják az adatok integritását, hitelességét és a nem tagadhatóságot. Ebben a fejezetben a digitális aláírás algoritmusok különböző típusait vizsgáljuk meg, különös tekintettel az RSA, a DSA és az ECDSA módszerekre. Ezek az algoritmusok különböző matematikai alapokon nyugszanak, és más-más előnyöket kínálnak a felhasználás során, legyen szó számítási hatékonyságról, biztonsági szintről vagy kulcsméretről. Az alábbiakban részletesen bemutatjuk mindhárom algoritmus működési elvét, előnyeit és alkalmazási területeit, hogy átfogó képet kapjunk a digitális aláírások világáról.

### 5.2.1. RSA alapú digitális aláírás

Az RSA (Rivest-Shamir-Adleman) algoritmus az egyik legismertebb és legszélesebb körben használt nyilvános kulcsú kriptográfiai algoritmus, amelyet digitális aláírások létrehozására és ellenőrzésére is alkalmaznak. Az RSA algoritmus az aszimmetrikus kriptográfia elvén alapul, ahol két kulcsot használnak: egy nyilvános kulcsot a titkosításhoz és egy privát kulcsot a dekódoláshoz, illetve aláírás létrehozásához.

#### RSA alapú digitális aláírás működési elve

Az RSA alapú digitális aláírás létrehozása és ellenőrzése a következő lépésekben valósul meg:

1. **Kulcspárok generálása**:
    - Két nagy, véletlenszerűen választott prímszámot generálnak, jelölve \( p \) és \( q \).
    - A prímszámok szorzata \( n \) adja a modulus értékét: \( n = p \times q \).
    - Számítsuk ki \( \phi(n) \) értékét, ahol \( \phi \) az Euler-féle totient függvény: \( \phi(n) = (p-1)(q-1) \).
    - Válasszunk egy nyilvános kitevőt \( e \), amely általában egy kis prímszám, például 65537, és amely megfelel az \( 1 < e < \phi(n) \) feltételnek, valamint közös osztója 1 \( \phi(n) \)-nel (relatív prím).
    - Számítsuk ki a privát kitevőt \( d \), amely az \( e \) multiplikatív inverze modulo \( \phi(n) \), vagyis \( d \equiv e^{-1} \mod \phi(n) \).

2. **Aláírás létrehozása**:
    - A digitálisan aláírandó üzenetet először hash-függvénnyel látjuk el, hogy egy fix hosszúságú, az üzenethez egyedi hash-értéket kapjunk: \( H(m) \), ahol \( H \) a hash-függvény, és \( m \) az üzenet.
    - Az aláírás létrehozása a hash-érték privát kulccsal történő titkosításával történik: \( S = H(m)^d \mod n \).
    - Az aláírás \( S \) és az eredeti üzenet \( m \) együtt alkotják a digitális aláírást.

3. **Aláírás ellenőrzése**:
    - Az üzenet fogadója először kiszámítja az üzenet hash-értékét ugyanazzal a hash-függvénnyel: \( H(m) \).
    - Az aláírás \( S \) ellenőrzése a nyilvános kulccsal történő dekódolással történik: \( H(m) \stackrel{?}{=} S^e \mod n \).
    - Ha a dekódolt érték megegyezik a számított hash-értékkel, akkor az aláírás érvényes, és az üzenet nem sérült, valamint az aláíró hitelessége is megerősítést nyer.

#### RSA algoritmus matematikai háttere

Az RSA algoritmus biztonságát a nagy prímszámok faktorizációjának nehézsége adja. Az \( n \) modulusból kiindulva \( p \) és \( q \) prímszámokat nem lehet hatékonyan meghatározni, ha \( n \) elég nagy (például 2048 vagy 4096 bit hosszú). Az alábbiakban részletezzük az RSA alapú digitális aláírás néhány kulcsfontosságú matematikai vonatkozását:

1. **Euler-féle totient függvény**:
    - A totient függvény \( \phi(n) \) meghatározza azon pozitív egész számok számát, amelyek kisebbek \( n \)-nél és relatív prímek \( n \)-hez.
    - \( n \) két prímszám szorzata esetén \( \phi(n) = (p-1)(q-1) \).

2. **Multiplikatív inverz**:
    - A privát kitevő \( d \) az \( e \) multiplikatív inverze modulo \( \phi(n) \), azaz \( e \cdot d \equiv 1 \mod \phi(n) \).
    - Az inverz megtalálása az Extended Euclidean Algorithm segítségével történik.

3. **Moduláris aritmetika**:
    - Az RSA algoritmus alapműveletei a moduláris aritmetikán alapulnak, amely biztosítja, hogy a számítások a modulus \( n \)-en belül maradjanak, így nagy számok kezelésére is hatékony.

#### Példakód C++ nyelven

Az alábbiakban egy egyszerű C++ példakódot mutatunk be az RSA alapú digitális aláírás létrehozására és ellenőrzésére:

```cpp
#include <iostream>
#include <string>
#include <openssl/bn.h>
#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <openssl/err.h>
#include <openssl/sha.h>

using namespace std;

string sha256(const string str) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, str.c_str(), str.size());
    SHA256_Final(hash, &sha256);
    stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        ss << hex << setw(2) << setfill('0') << (int)hash[i];
    }
    return ss.str();
}

RSA* generateRSAKeyPair() {
    int bits = 2048;
    unsigned long e = RSA_F4;
    RSA* rsa = RSA_new();
    BIGNUM* bne = BN_new();
    BN_set_word(bne, e);
    RSA_generate_key_ex(rsa, bits, bne, NULL);
    BN_free(bne);
    return rsa;
}

string signMessage(RSA* rsa, const string& message) {
    string hash = sha256(message);
    unsigned char* signature = new unsigned char[RSA_size(rsa)];
    unsigned int sigLen;
    RSA_sign(NID_sha256, (unsigned char*)hash.c_str(), hash.length(), signature, &sigLen, rsa);
    string result((char*)signature, sigLen);
    delete[] signature;
    return result;
}

bool verifySignature(RSA* rsa, const string& message, const string& signature) {
    string hash = sha256(message);
    bool result = RSA_verify(NID_sha256, (unsigned char*)hash.c_str(), hash.length(), 
                             (unsigned char*)signature.c_str(), signature.length(), rsa);
    return result;
}

int main() {
    RSA* rsa = generateRSAKeyPair();

    string message = "This is a test message.";
    string signature = signMessage(rsa, message);

    bool valid = verifySignature(rsa, message, signature);
    if (valid) {
        cout << "Signature is valid." << endl;
    } else {
        cout << "Signature is invalid." << endl;
    }

    RSA_free(rsa);
    return 0;
}
```

#### Biztonsági megfontolások és gyakorlati alkalmazások

Az RSA algoritmus használata során fontos figyelembe venni néhány biztonsági szempontot:

1. **Kulcsméret**: A biztonságos kulcsméret választása kritikus. A modern biztonsági követelményeknek megfelelően általában legalább 2048 bites kulcsokat használnak.
2. **Hash-függvény**: A digitális aláírás integritásának biztosítása érdekében erős hash-függvényt kell alkalmazni, mint például a SHA-256.
3. **Randomizálás**: Az RSA aláírás létrehozása során szükséges randomizálást alkalmazni, hogy megakadályozzuk a támadások, például a replay attack, lehetőségét.
4. **Kulcskezelés**: A privát kulcs biztonságos tárolása és kezelése elengedhetetlen. Bármilyen kompromittáció az egész rendszer biztonságát veszélyezteti.

Az RSA alapú digitális aláírásokat széles körben alkalmazzák különböző területeken, például SSL/TLS protokollokban, biztonságos e-mailekben (S/MIME), szoftver aláírásokban és elektronikus dokumentumok hitelesítésében. Az RSA megbízhatósága és széles körű elfogadottsága miatt továbbra is kiemelkedő szerepet játszik a digitális aláírás technológiák között.

### 5.2.2. DSA (Digital Signature Algorithm)

A Digital Signature Algorithm (DSA) egy szabványosított digitális aláírási eljárás, amelyet az amerikai National Institute of Standards and Technology (NIST) fejlesztett ki és definiált a Digital Signature Standard (DSS) részeként az 1990-es évek elején. A DSA egy aszimmetrikus kriptográfiai algoritmus, amelyet kifejezetten digitális aláírások létrehozására terveztek. Az RSA-hoz hasonlóan a DSA is két kulcsot használ: egy privát kulcsot az aláírás létrehozásához és egy nyilvános kulcsot az aláírás ellenőrzéséhez. Azonban a DSA és az RSA működési elve és matematikai alapjai jelentősen különböznek.

#### DSA működési elve

A DSA alapú digitális aláírás létrehozása és ellenőrzése több lépésből áll, amelyek magukban foglalják a kulcsgenerálást, az aláírási folyamatot és az aláírás ellenőrzését.

1. **Kulcspárok generálása**:
    - **Paraméterek generálása**:
        - Válasszunk egy 1024, 2048 vagy 3072 bites prímszámot \( p \).
        - Válasszunk egy 160, 224 vagy 256 bites prímszámot \( q \), ahol \( q \) osztója \( p-1 \)-nek.
        - Válasszunk egy \( g \) generátort, amely kielégíti az \( g = h^{(p-1)/q} \mod p \) feltételt, ahol \( h \) egy \( 1 < h < p-1 \) egész szám.
    - **Privát és nyilvános kulcsok generálása**:
        - Válasszunk egy véletlenszerű titkos számot \( x \), ahol \( 0 < x < q \). Ez lesz a privát kulcs.
        - Számítsuk ki a nyilvános kulcsot \( y \) az alábbi módon: \( y = g^x \mod p \).

2. **Aláírás létrehozása**:
    - Az aláírandó üzenetet először hash-függvénnyel látjuk el, hogy egy fix hosszúságú hash-értéket kapjunk: \( H(m) \), ahol \( H \) a hash-függvény, és \( m \) az üzenet.
    - Válasszunk egy véletlenszerű \( k \) értéket, ahol \( 0 < k < q \).
    - Számítsuk ki az aláírás első részét \( r \)-t az alábbi módon: \( r = (g^k \mod p) \mod q \).
    - Számítsuk ki az aláírás második részét \( s \)-t az alábbi módon: \( s = (k^{-1}(H(m) + xr)) \mod q \).
    - Az aláírás az \( (r, s) \) páros.

3. **Aláírás ellenőrzése**:
    - Az üzenet fogadója először kiszámítja az üzenet hash-értékét ugyanazzal a hash-függvénnyel: \( H(m) \).
    - Ellenőrizze, hogy \( 0 < r < q \) és \( 0 < s < q \).
    - Számítsa ki \( w \)-t, ahol \( w = s^{-1} \mod q \).
    - Számítsa ki \( u_1 \)-et és \( u_2 \)-t: \( u_1 = (H(m) \cdot w) \mod q \) és \( u_2 = (r \cdot w) \mod q \).
    - Számítsa ki \( v \)-t: \( v = ((g^{u_1} \cdot y^{u_2}) \mod p) \mod q \).
    - Az aláírás érvényes, ha és csak ha \( v = r \).

#### DSA matematikai háttere

A DSA algoritmus matematikai háttere a diszkrét logaritmus problémán alapul, amelynek nehézsége biztosítja az algoritmus biztonságát. Az alábbiakban részletesebben bemutatjuk a DSA néhány kulcsfontosságú matematikai vonatkozását:

1. **Diszkrét logaritmus probléma**:
    - A diszkrét logaritmus probléma azt jelenti, hogy adott egy \( g \) alap és egy \( y \) érték, nehéz megtalálni az \( x \) kitevőt, amely kielégíti az \( y = g^x \mod p \) egyenletet. Ez a probléma biztosítja a DSA algoritmus biztonságát.

2. **Modulo aritmetika**:
    - A DSA algoritmus számos műveletet hajt végre a modulo aritmetikában, különösen a modulo \( p \) és \( q \) értékekkel. Ezek a műveletek biztosítják, hogy a számítások a megfelelő tartományban maradjanak.

3. **Multiplikatív inverz**:
    - A DSA aláírás létrehozása és ellenőrzése során többször kell kiszámítani a számok inverzét modulo \( q \), amely a titkos kulcsok és az aláírások kiszámításához szükséges.

#### Példakód C++ nyelven

Az alábbiakban egy egyszerű C++ példakódot mutatunk be a DSA alapú digitális aláírás létrehozására és ellenőrzésére:

```cpp
#include <iostream>
#include <openssl/bn.h>
#include <openssl/dsa.h>
#include <openssl/pem.h>
#include <openssl/sha.h>
#include <sstream>
#include <iomanip>

using namespace std;

string sha256(const string str) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, str.c_str(), str.size());
    SHA256_Final(hash, &sha256);
    stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        ss << hex << setw(2) << setfill('0') << (int)hash[i];
    }
    return ss.str();
}

DSA* generateDSAKeyPair() {
    DSA* dsa = DSA_new();
    DSA_generate_parameters_ex(dsa, 2048, NULL, 0, NULL, NULL, NULL);
    DSA_generate_key(dsa);
    return dsa;
}

pair<string, string> signMessage(DSA* dsa, const string& message) {
    string hash = sha256(message);
    unsigned char* sigret = new unsigned char[DSA_size(dsa)];
    unsigned int siglen;
    DSA_sign(0, (unsigned char*)hash.c_str(), hash.length(), sigret, &siglen, dsa);
    string r((char*)sigret, siglen / 2);
    string s((char*)sigret + siglen / 2, siglen / 2);
    delete[] sigret;
    return make_pair(r, s);
}

bool verifySignature(DSA* dsa, const string& message, const pair<string, string>& signature) {
    string hash = sha256(message);
    unsigned char* sigret = new unsigned char[signature.first.length() + signature.second.length()];
    memcpy(sigret, signature.first.c_str(), signature.first.length());
    memcpy(sigret + signature.first.length(), signature.second.c_str(), signature.second.length());
    bool result = DSA_verify(0, (unsigned char*)hash.c_str(), hash.length(), sigret, signature.first.length() + signature.second.length(), dsa);
    delete[] sigret;
    return result;
}

int main() {
    DSA* dsa = generateDSAKeyPair();

    string message = "This is a test message.";
    pair<string, string> signature = signMessage(dsa, message);

    bool valid = verifySignature(dsa, message, signature);
    if (valid) {
        cout << "Signature is valid." << endl;
    } else {
        cout << "Signature is invalid." << endl;
    }

    DSA_free(dsa);
    return 0;
}
```

#### Biztonsági megfontolások és gyakorlati alkalmazások

A DSA használata során szintén fontos figyelembe venni néhány biztonsági szempontot:

1. **Kulcsméret**: A kulcsméret kritikus a biztonság szempontjából. A modern biztonsági szabványok szerint ajánlott legalább 2048 bites kulcsokat használni.
2. **Hash-függvény**: A DSA megköveteli, hogy az aláírás létrehozásakor használt hash-függvény (például SHA-256) erős és biztonságos legyen.
3. **Véletlenszerű \( k \) érték**: A \( k \) érték véletlenszerűsége és titkossága kritikus fontosságú. Ha \( k \) értékét meg lehetne jósolni, az súlyosan veszélyeztetné az aláírási folyamat biztonságát. Ismert esetek vannak, amikor gyenge véletlenszám-generátorok miatt sikerült kompromittálni a DSA aláírásokat.
4. **Kulcskezelés**: A privát kulcs biztonságos tárolása és kezelése elengedhetetlen. Bármilyen kompromittáció az egész rendszer biztonságát veszélyezteti.

A DSA alapú digitális aláírásokat széles körben alkalmazzák különböző területeken, például az elektronikus dokumentumok hitelesítésében, szoftveraláírásokban és biztonságos kommunikációs protokollokban. Bár a DSA nem olyan széles körben elterjedt, mint az RSA, a specifikus alkalmazási területeken továbbra is népszerű választás a magas szintű biztonságot igénylő feladatokhoz.

## 5.2.3. ECDSA (Elliptic Curve Digital Signature Algorithm)

Az Elliptic Curve Digital Signature Algorithm (ECDSA) a digitális aláírások egy modern, hatékony módszere, amely az elliptikus görbe kriptográfia (ECC) matematikai alapjain nyugszik. Az ECDSA az elliptikus görbék tulajdonságait használja a biztonságos és kompakt digitális aláírások létrehozásához, jelentős előnyöket kínálva a hagyományos algoritmusokkal, például az RSA-val és a DSA-val szemben, különösen a kisebb kulcsméretek és a gyorsabb számítási műveletek terén.

#### ECDSA működési elve

Az ECDSA alapú digitális aláírás létrehozása és ellenőrzése a következő lépésekben valósul meg:

1. **Elliptikus görbe paraméterek generálása**:
   - Válasszunk egy megfelelő elliptikus görbét \( E \) egy véges test felett \( \mathbb{F}_q \), ahol \( q \) egy nagy prímszám.
   - Határozzuk meg az elliptikus görbe paramétereit, amelyek tartalmazzák a görbe egyenletét \( y^2 = x^3 + ax + b \mod q \), valamint egy generátor pontot \( G \), amely az aláírás létrehozásának alapjául szolgál.

2. **Kulcspárok generálása**:
   - Válasszunk egy véletlenszerű privát kulcsot \( d \), ahol \( 1 \leq d < n \), és \( n \) a generátor pont \( G \) rendje.
   - Számítsuk ki a nyilvános kulcsot \( Q \), amely az elliptikus görbén található pont: \( Q = d \cdot G \).

3. **Aláírás létrehozása**:
   - Az aláírandó üzenetet először hash-függvénnyel látjuk el, hogy egy fix hosszúságú hash-értéket kapjunk: \( H(m) \), ahol \( H \) a hash-függvény, és \( m \) az üzenet.
   - Válasszunk egy véletlenszerű számot \( k \), ahol \( 1 \leq k < n \).
   - Számítsuk ki az aláírás első részét \( r \), ahol \( r = (k \cdot G)_x \mod n \), és \( (k \cdot G)_x \) a pont \( k \cdot G \) x-koordinátája.
   - Számítsuk ki az aláírás második részét \( s \), ahol \( s = k^{-1}(H(m) + dr) \mod n \).
   - Az aláírás az \( (r, s) \) páros.

4. **Aláírás ellenőrzése**:
   - Az üzenet fogadója először kiszámítja az üzenet hash-értékét ugyanazzal a hash-függvénnyel: \( H(m) \).
   - Ellenőrizze, hogy \( 0 < r < n \) és \( 0 < s < n \).
   - Számítsa ki \( w \)-t, ahol \( w = s^{-1} \mod n \).
   - Számítsa ki \( u_1 \)-et és \( u_2 \)-t: \( u_1 = H(m) \cdot w \mod n \) és \( u_2 = r \cdot w \mod n \).
   - Számítsa ki a pontot \( P \), ahol \( P = u_1 \cdot G + u_2 \cdot Q \).
   - Az aláírás érvényes, ha és csak ha \( r \equiv P_x \mod n \), ahol \( P_x \) a pont \( P \) x-koordinátája.

#### ECDSA matematikai háttere

Az ECDSA algoritmus matematikai alapja az elliptikus görbék és a diszkrét logaritmus problémája az elliptikus görbéken. Az elliptikus görbe kriptográfia különböző nehézségi problémákra épül, amelyek biztosítják a rendszer biztonságát.

1. **Elliptikus görbék**:
   - Az elliptikus görbék az alábbi egyenlet által meghatározott görbék: \( y^2 = x^3 + ax + b \mod q \).
   - A pontok \( P \) és \( Q \) összeadását és egy pontnak egy skalárral való szorzását az elliptikus görbén határozzuk meg.

2. **Diszkrét logaritmus probléma**:
   - Az elliptikus görbén a diszkrét logaritmus probléma azt jelenti, hogy adott két pont, \( P \) és \( Q \), nehéz megtalálni az \( k \) egész számot, amely kielégíti az \( Q = k \cdot P \) egyenletet.

3. **Modulo aritmetika**:
   - Az ECDSA műveletei a modulo \( n \) értékekkel történnek, ahol \( n \) a generátor pont \( G \) rendje.

#### Példakód C++ nyelven

Az alábbiakban egy egyszerű C++ példakódot mutatunk be az ECDSA alapú digitális aláírás létrehozására és ellenőrzésére:

```cpp
#include <iostream>
#include <openssl/ecdsa.h>
#include <openssl/obj_mac.h>
#include <openssl/sha.h>
#include <sstream>
#include <iomanip>

using namespace std;

string sha256(const string str) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, str.c_str(), str.size());
    SHA256_Final(hash, &sha256);
    stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        ss << hex << setw(2) << setfill('0') << (int)hash[i];
    }
    return ss.str();
}

EC_KEY* generateECKeyPair() {
    EC_KEY* eckey = EC_KEY_new_by_curve_name(NID_X9_62_prime256v1);
    EC_KEY_generate_key(eckey);
    return eckey;
}

pair<string, string> signMessage(EC_KEY* eckey, const string& message) {
    string hash = sha256(message);
    unsigned char* sig = NULL;
    unsigned int sig_len;
    ECDSA_sign(0, (const unsigned char*)hash.c_str(), hash.length(), sig, &sig_len, eckey);
    string r((char*)sig, sig_len / 2);
    string s((char*)sig + sig_len / 2, sig_len / 2);
    OPENSSL_free(sig);
    return make_pair(r, s);
}

bool verifySignature(EC_KEY* eckey, const string& message, const pair<string, string>& signature) {
    string hash = sha256(message);
    unsigned char* sig = new unsigned char[signature.first.length() + signature.second.length()];
    memcpy(sig, signature.first.c_str(), signature.first.length());
    memcpy(sig + signature.first.length(), signature.second.c_str(), signature.second.length());
    bool result = ECDSA_verify(0, (const unsigned char*)hash.c_str(), hash.length(), sig, signature.first.length() + signature.second.length(), eckey);
    delete[] sig;
    return result;
}

int main() {
    EC_KEY* eckey = generateECKeyPair();

    string message = "This is a test message.";
    pair<string, string> signature = signMessage(eckey, message);

    bool valid = verifySignature(eckey, message, signature);
    if (valid) {
        cout << "Signature is valid." << endl;
    } else {
        cout << "Signature is invalid." << endl;
    }

    EC_KEY_free(eckey);
    return 0;
}
```

#### Biztonsági megfontolások és gyakorlati alkalmazások

Az ECDSA használata során fontos figyelembe venni néhány biztonsági szempontot:

1. **Kulcsméret**: Az ECC egyik előnye a kisebb kulcsméretek használata a magas biztonsági szint mellett. Például a 256 bites ECC kulcs biztonsági szintje megfelel egy 3072 bites RSA kulcs biztonsági szintjének.
2. **Hash-függvény**: Az ECDSA megköveteli, hogy az aláírás létrehozásakor használt hash-függvény (például SHA-256) erős és biztonságos legyen.
3. **Véletlenszerű \( k \) érték**: A \( k \) érték véletlenszerűsége és titkossága kritikus fontosságú. Ha \( k \) értékét meg lehetne jósolni, az súlyosan veszélyeztetné az aláírási folyamat biztonságát.
4. **Kulcskezelés**: A privát kulcs biztonságos tárolása és kezelése elengedhetetlen. Bármilyen kompromittáció az egész rendszer biztonságát veszélyezteti.

Az ECDSA alapú digitális aláírásokat széles körben alkalmazzák különböző területeken, például az SSL/TLS protokollokban, az elektronikus dokumentumok hitelesítésében, az okostelefonok és más beágyazott rendszerek biztonságos kommunikációjában, valamint a blokklánc-technológiákban, például a Bitcoinban és más kriptovalutákban. Az ECDSA hatékonysága és kompakt kulcsméretei miatt különösen népszerű a korlátozott erőforrásokkal rendelkező környezetekben.
