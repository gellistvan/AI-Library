## 2. Alapfogalmak és definíciók
A modern információs társadalomban a kriptográfia elengedhetetlen szerepet játszik az adatok védelmében és a kommunikáció biztonságának fenntartásában. Mindennapjaink digitális világában elengedhetetlen, hogy biztonságos módon tároljuk és továbbítsuk érzékeny információinkat, akár személyes adatok, üzleti titkok vagy pénzügyi tranzakciók formájában. E fejezet célja, hogy megismertessen bennünket a kriptográfia alapfogalmaival és definícióival, amelyek az adatvédelem alapkövei. Áttekintjük a titkosítás és visszafejtés folyamatát, megvizsgáljuk a kulcsok és kulcsmenedzsment fontosságát, valamint bemutatjuk a biztonsági alapelveket – a bizalmasságot, az integritást, a hitelességet és a nem-megtagadást –, amelyek segítenek megérteni, hogyan érhetjük el és tarthatjuk fenn az információbiztonságot.

## 1.2. Alapfogalmak és definíciók

### Titkosítás, visszafejtés

A titkosítás a kriptográfia egyik legfontosabb folyamata, amely egyértelműen olvasható (vagy 'plain-text') információt alakít át egy olyan formába (azaz 'ciphertext'), amely csak kifejezetten erre a célra felhatalmazott személyek vagy rendszerek számára olvasható. A titkosítás fő célja a bizalmasság fenntartása és az adatvédelem biztosítása, azaz hogy az információhoz illetéktelen személyek ne férjenek hozzá.

#### A titkosítás folyamata

A titkosítás folyamata több alapvető lépésből áll:
1. **Plaintext**: Az eredeti adatok, amelyeket titkosítani szeretnénk.
2. **Cipher**: A titkosító algoritmus, amely a plaintextet ciphertexté alakítja.
3. **Kulcs**: Egy sajátos információ, amelyet a titkosító algoritmus a titkosítás végrehajtásához használ.

Matematikai szempontból a titkosítási folyamat így reprezentálható:
\[ C = E_k(P) \]
ahol:
- \( C \) a ciphertext,
- \( E \) a titkosító függvény,
- \( k \) a titkosító kulcs,
- \( P \) a plaintext.

#### A visszafejtés folyamata

A visszafejtés folyamata az titkosított adatokat visszaalakítja az eredeti plaintext formába. Ez a folyamat a rejtjelzés (ciphertext) eredeti adatait rekonstruálja egy megfelelő dekódolási (visszafejtő) algoritmus segítségével, amelyhez szükség van a megfelelő kulcsra.

Matematikai értelemben a visszafejtési folyamat így néz ki:
\[ P = D_k(C) \]
ahol:
- \( P \) a plaintext,
- \( D \) a dekódoló függvény,
- \( k \) a visszafejtő kulcs,
- \( C \) a ciphertext.

#### Titkosítási algoritmusok típusai

Titkosítási módszerek többféle típusa létezik, amelyek közül a leghíresebbek a szimmetrikus (symmetrical) és az aszimmetrikus (asymmetrical) algoritmusok.

**Szimmetrikus titkosítás**: Ebben a módszerben ugyanazt a kulcsot használjuk mind a titkosításhoz, mind a visszafejtéshez. A szimmetrikus algoritmusok egyszerűbbek és gyorsabbak, de kihívást jelent a kulcsok biztonságos elosztása. Ismert szimmetrikus algoritmusok a DES, AES és a Blowfish.

**Aszimmetrikus titkosítás**: Ebben a módszerben két külön kulcsot használunk: egy publikus kulcsot a titkosításhoz és egy privát kulcsot a visszafejtéshez. Az aszimmetrikus algoritmusok nagyobb számítási teljesítményt igényelnek, de megkönnyítik a kulcscserét és a biztonságos kommunikáció létrehozását. Ismert aszimmetrikus algoritmusok az RSA és az ECC (Elliptic Curve Cryptography).

#### Matematikai alapok és szükséges eszközök

**Moduláris aritmetika**: A titkosítási algoritmusok gyakran alkalmaznak moduláris műveleteket. Például az RSA algoritmus erősen támaszkodik a moduláris exponenciálásra és a prímszámokra.

**Prímszámok**: Prímszámok a szimmetrikus és aszimmetrikus algoritmusok egyik alapvető építőkövei, különösen az RSA esetében, amely két nagy prímszám szorzatát használja a kulcsgeneráláshoz.

**Elliptikus görbék**: Az elliptikus görbék speciális típusú matematikai struktúrák, amelyek aszimmetrikus kriptográfiában játszanak nagy szerepet, az ECC alapját képezik.

#### Titkosítási algoritmusok és implementációk

**DES és 3DES (Triple DES)**

A DES (Data Encryption Standard) egy szimmetrikus kulcsú blokkrejtjel, amelyet az IBM fejlesztett ki az 1970-es években. A DES 56 bites kulcshosszúságot használ, ami ma már nem tekinthető biztonságosnak, legfőképpen a modern számítógépek számítási kapacitása miatt.

A 3DES, amely háromszoros DES-t jelent, három különálló DES titkosítást használ egymás után, és lényegesen nagyobb biztonságot nyújt.

**AES (Advanced Encryption Standard)**

Az AES a mai napig széles körben használt szimmetrikus titkosítási algoritmus. Ez a titkosítási módszer három különböző kulcshosszúságot támogat: 128-bit, 192-bit és 256-bit. Az AES-algoritmus egy iteratív protokoll, amely több forduló (round) alatt végzi el a titkosítást.

Implementáció C++ nyelven:

```cpp
#include <iostream>
#include <openssl/aes.h>

void encrypt(const unsigned char* plaintext, unsigned char* ciphertext, const unsigned char* key) {
    AES_KEY encryptKey;
    AES_set_encrypt_key(key, 128, &encryptKey);
    AES_encrypt(plaintext, ciphertext, &encryptKey);
}

void decrypt(const unsigned char* ciphertext, unsigned char* decryptedtext, const unsigned char* key) {
    AES_KEY decryptKey;
    AES_set_decrypt_key(key, 128, &decryptKey);
    AES_decrypt(ciphertext, decryptedtext, &decryptKey);
}

int main() {
    unsigned char plaintext[] = "Hello, this is a test!";
    unsigned char key[16] = "This is a key123";
    unsigned char ciphertext[16];
    unsigned char decryptedtext[16];

    // Plaintext encryption
    encrypt(plaintext, ciphertext, key);
    std::cout << "Encrypted text: ";
    for (int i = 0; i < 16; ++i) std::cout << std::hex << (int)ciphertext[i];
    std::cout << std::endl;

    // Ciphertext decryption
    decrypt(ciphertext, decryptedtext, key);
    std::cout << "Decrypted text: ";
    for (int i = 0; i < 16; ++i) std::cout << decryptedtext[i];
    std::cout << std::endl;

    return 0;
}
```

**RSA (Rivest–Shamir–Adleman)**

Az RSA az egyik legismertebb és leggyakrabban használt aszimmetrikus titkosítási algoritmus, amelyet 1977-ben fejlesztettek ki. Az RSA algoritmus a kulcsgenerálás során két nagy prímszámot választ, majd ezek szorzatát és speciális matematikai műveleteket (Euler-totient függvény, modulus) használ a publikus és privát kulcsok előállításához.

Az RSA alapú titkosítás és visszafejtés folyamata C++ nyelven az alábbiak szerint valósulhat meg az OpenSSL könyvtár segítségével:

```cpp
#include <iostream>
#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <openssl/err.h>

RSA* generate_key() {
    RSA* rsa = RSA_new();
    BIGNUM* bn = BN_new();
    BN_set_word(bn, RSA_F4);

    RSA_generate_key_ex(rsa, 2048, bn, nullptr);
    BN_free(bn);
    return rsa;
}

std::string encrypt(const std::string& plaintext, RSA* public_key) {
    std::string ciphertext(RSA_size(public_key), '\0');
    int len = RSA_public_encrypt(plaintext.size(), reinterpret_cast<const unsigned char*>(plaintext.c_str()), reinterpret_cast<unsigned char*>(&ciphertext[0]), public_key, RSA_PKCS1_OAEP_PADDING);
    if (len == -1) throw std::runtime_error(ERR_error_string(ERR_get_error(), nullptr));
    return ciphertext;
}

std::string decrypt(const std::string& ciphertext, RSA* private_key) {
    std::string plaintext(RSA_size(private_key), '\0');
    int len = RSA_private_decrypt(ciphertext.size(), reinterpret_cast<const unsigned char*>(ciphertext.c_str()), reinterpret_cast<unsigned char*>(&plaintext[0]), private_key, RSA_PKCS1_OAEP_PADDING);
    if (len == -1) throw std::runtime_error(ERR_error_string(ERR_get_error(), nullptr));
    plaintext.resize(len);
    return plaintext;
}

void free_key(RSA* key) {
    RSA_free(key);
}

int main() {
    RSA* key = generate_key();

    std::string plaintext = "Hello, RSA!";
    std::string ciphertext = encrypt(plaintext, key);
    std::cout << "Encrypted text: ";
    for (unsigned char c : ciphertext) std::cout << std::hex << (int)c;
    std::cout << std::endl;

    std::string decryptedtext = decrypt(ciphertext, key);
    std::cout << "Decrypted text: " << decryptedtext << std::endl;

    free_key(key);
    return 0;
}
```

### Kulcsok és kulcsmenedzsment

A kriptográfia egyik legkritikusabb aspektusa a kulcsok kezelése és menedzsmentje. A kulcsok biztosítják a titkosítás és a visszafejtés lehetőségét, és azok biztonsága alapvető a kriptográfiai rendszerek hatékonysága és bizalmassága szempontjából. Ebben a fejezetben mélyebben megvizsgáljuk a kulcsok típusait, generálásukat, tárolásukat, megosztásukat, valamint azokat a módszereket és protokollokat, amelyek segítségével ezek a feladatok biztonságosan elvégezhetők.

### Kulcsok típusai

#### Szimmetrikus kulcsok

A szimmetrikus kriptográfia, más néven titkos kulcsos kriptográfia, egy olyan módszer, amelyben ugyanazt a kulcsot használják mind a titkosítás, mind a visszafejtés során. Ez a megközelítés gyors és kevesebb számítási erőforrást igényel, de a kulcsok biztonságos megosztása nagy kihívást jelent.

Példák szimmetrikus algoritmusokra:
- AES (Advanced Encryption Standard)
- DES (Data Encryption Standard)
- Triple DES (3DES)

#### Aszimmetrikus kulcsok

Az aszimmetrikus kriptográfia, más néven nyilvános kulcsú kriptográfia, két különböző, de matematikailag összefüggő kulcsot használ: egy nyilvános kulcsot (amelyet bárki elérhet) és egy magánkulcsot (amelyet csak a címzett ismer). A nyilvános kulcs a titkosításhoz, míg a magánkulcs a visszafejtéshez használatos.

Példák aszimmetrikus algoritmusokra:
- RSA
- ECC (elliptikus görbe kriptográfia)
- ElGamal

#### Hibrid megközelítések

A hibrid kriptográfiai rendszerek ötvözik a szimmetrikus és az aszimmetrikus algoritmusok előnyeit. Általában a szimmetrikus kulcsok gyorsaságát és az aszimmetrikus kulcsok biztonságos kulcsmegosztását egyesítik. Egy tipikus példa erre a hibrid módszerre az, amikor egy szimmetrikus kulcsot titkosítanak egy aszimmetrikus kulccsal, így a szimmetrikus kulcs titkosítva kerül továbbításra.

### Kulcsgenerálás

A kulcsok generálása olyan folyamat, amely során kriptográfiailag erős véletlenszámok kerülnek előállításra, amelyekből a kulcsokat képzik. A generálási folyamatnak erős véletlenszerűséget és kiszámíthatatlanságot kell biztosítania ahhoz, hogy a titkos kulcsok biztonságosak legyenek.

#### Véletlenszám-generátorok (RNG) és Kriptográfiai Véletlenszám-generátorok (CSPRNG)

A véletlenszám-generátorok két fő típusa létezik:
- **Pszeudo-véletlenszám-generátorok (PRNG):** Determinista algoritmusok, amelyek látszólag véletlenszerű sorozatokat állítanak elő. Ezek általában nem elég biztonságosak kriptográfiai célokra.
- **Kriptográfiai szempontból biztonságos pszeudo-véletlenszám-generátorok (CSPRNG):** Ezek speciális PRNG-k, amelyeket úgy terveztek, hogy megfeleljenek a kriptográfiai biztonsági követelményeknek, mint a kiszámíthatatlanság és az előreláthatatlanság.

Példa CSPRNG implementációjára C++ nyelven:
```cpp
#include <iostream>
#include <random>
#include <iomanip>

int main() {
    std::random_device rd;  // A random device for generating initial seed.
    std::mt19937_64 gen(rd());  // A Mersenne Twister pseudo-random generator of 64-bit integers initialized with the seed.
    std::uniform_int_distribution<unsigned long long> dis;

    std::cout << "Generating cryptographic keys:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "Key " << i + 1 << ": " 
                  << std::hex << std::setw(16) << std::setfill('0') 
                  << dis(gen) << std::endl;  // Print random 64-bit number in hexadecimal format.
    }

    return 0;
}
```

### Kulcstárolás és kulcsmenedzsment rendszerek

A kulcsok biztonságos tárolása alapvető követelmény a kriptográfiai rendszerek számára. A következőkben részletezzük a kulcsmenedzsment rendszerek különböző aspektusait:

#### Hardveres biztonsági modulok (HSM)

A HSM-ek speciális készülékek, amelyeket úgy terveztek, hogy biztonságos környezetet nyújtsanak a kriptográfiai kulcsok generálására és tárolására. A HSM-ek célja, hogy megvédjék a kulcsokat fizikai és logikai támadásokkal szemben.

Előnyök:
- Fizikai védelmet biztosítanak a kulcsok számára
- Gyors és biztonságos kriptográfiai műveletek, pl. aláírások és titkosítások végrehajtása

#### Szoftver alapú kulcsmenedzsment

Szoftver alapú kulcsmenedzsment rendszerek lehetővé teszik a kulcsok központi tárolását és kezelését egy szerveren. Ezek a rendszerek gyakran különféle titkosítási technikákat használnak a kulcsok védelme érdekében, de a fizikai védelmük gyengébb, mint a HSM-ek esetében.

Példák:
- Key Management Service (KMS) az AWS-nél
- Azure Key Vault a Microsoft-nál

#### Kulcsok elosztása és megosztása

A kulcsok biztonságos elosztása és megosztása kritikus fontosságú, hogy megakadályozzuk azok illetéktelen kezekbe kerülését. Számos technika áll rendelkezésre a kulcsok biztonságos elosztására:

##### Közvetlen kulcsszállítás

Ebben a módszerben a kulcsokat közvetlenül juttatják el a címzettekhez, t.k. kurier vagy más fizikai szállítási módszer segítségével. Ez a módszer biztonságot nyújt, de kevésbé praktikus nagy távolságok esetében.

##### Nyilvános kulcs infrastruktúra (PKI)

A PKI-k a nyilvános kulcsú kriptográfia alapján működnek, és lehetővé teszik a biztonságos kulcscserét a digitális tanúsítványok segítségével. Egy PKI rendszerben a tanúsítványhatóság (CA) digitális tanúsítványokat ad ki, amelyek igazolják a kulcspárokhoz tartozó személyek vagy szervezetek azonosságát.

##### Diffie-Hellman kulcscsere

A Diffie-Hellman algoritmus lehetővé teszi két fél számára, hogy biztonságosan megosszanak egy közös titkot egy nyilvános csatornán keresztül.

```cpp
#include <iostream>
#include <cmath>

// Simple implementation of Diffie-Hellman key exchange

long long power(long long a, long long b, long long p) { 
    long long res = 1; 
    a = a % p; 
    while (b > 0) { 
        if (b & 1) 
            res = (res * a) % p; 
        b = b >> 1; 
        a = (a * a) % p; 
    } 
    return res; 
}

int main() {
    long long p = 23; // Prime number
    long long g = 5;  // Primitive root
    
    long long a, b; // Sender's and receiver's private keys
    a = 6; // Randomly chosen
    b = 15; // Randomly chosen
    
    long long A = power(g, a, p); // Calculate g^a mod p
    long long B = power(g, b, p); // Calculate g^b mod p
    
    long long shared_secret_A = power(B, a, p); 
    long long shared_secret_B = power(A, b, p); 
    
    std::cout << "Sender's shared secret: " << shared_secret_A << std::endl;
    std::cout << "Receiver's shared secret: " << shared_secret_B << std::endl;

    return 0;
}
```

### Kulcsok élettartama és megsemmisítése

A kulcsok biztonságos menedzselésének fontos része azok élettartamának meghatározása és a megfelelő időpontban történő megsemmisítése.

#### Kulcstokrotáció

A kulcstokrotáció során rendszeresen és előre meghatározott időközönként új kulcsokat generálnak és az elavultakat biztonságosan megsemmisítik. Ez csökkenti a lehetséges támadási felületet és növeli az általános rendszerek biztonságát.

#### Kulcsok megsemmisítése

Amikor egy kulcs élettartama véget ér, fontos annak biztonságos megsemmisítése, hogy ne lehessen újra felhasználni vagy kompromittálni. A kulcs megsemmisítése tipikusan annak minden létező példányának törlésével és felülírásával történik.

### Kulcsmenedzsment politikák és szabványok

A hatékony kulcsmenedzsmenthez világos és szabályozott politikákra van szükség, amelyek lefedik a kulcsok teljes életciklusát, azaz a generálástól a megsemmisítésig. Számos nemzetközi szabvány segíti a szervezeteket ezek kialakításában:

- **NIST SP 800-57: Key Management Guidelines**: A Nemzeti Szabványügyi és Technológiai Intézet (NIST) ajánlásai a kulcsmenedzsmentre.
- **ISO/IEC 11770: IT Security Techniques - Key Management**: Nemzetközi szabványok a kulcsmenedzsmentre és annak különböző aspektusaira.
- **FIPS 140-3**: A szövetségi információfeldolgozási szabvány, amely meghatározza a kriptográfiai modulok biztonsági követelményeit.


### Biztonsági alapelvek: bizalmasság, integritás, hitelesség, nem-megtagadás

A modern kriptográfia és kódolási algoritmusok célja információk védelme és az eredményesség garantálása az adatkommunikáció számos aspektusában. Az alapvető biztonsági alapelvek közé tartoznak a bizalmasság, az integritás, a hitelesség és a nem-megtagadás. Ezek az alapelvek mind különböző fenyegetésekre és támadási vektorokra reagálnak, és mindegyik elengedhetetlen a biztonságos rendszer kialakításához.

#### Bizalmasság (Confidentiality)
A bizalmasság az adatvédelmet jelenti, biztosítva azt, hogy az információkhoz csak az arra jogosult személyek férhetnek hozzá. Ennek biztosítása érdekében gyakran titkosítást alkalmaznak, amely az adatokat olvashatatlan formába átalakítja illetéktelenek számára.

**Szimmetrikus titkosítás**: Ez egy olyan titkosítási módszer, ahol a titkosításhoz és a visszafejtéshez ugyanazt a kulcsot használják. Ez gyors és hatékony, de a kulcsok megosztása és kezelése problémás lehet nagy rendszerek esetében.

**Aszimmetrikus titkosítás**: Ezzel ellentétben az aszimmetrikus titkosítás két különböző kulcsot használ: egy nyilvános kulcsot a titkosításhoz és egy privát kulcsot a visszafejtéshez. Ez kényelmesebb kulcskezelést biztosít, de általában lassabb és erőforrásigényesebb.

#### Integritás (Integrity)

Az integritás az adatok épségének és teljességének védelmére utal, biztosítva, hogy az adatok nem kerültek megváltoztatásra illetéktelenek által. A kriptográfiai hash függvényeket gyakran használják az integritás ellenőrzésére, mivel ezek biztosítják, hogy az adat bármilyen kis változása a hash érték szignifikáns változását eredményezi.

**Hash függvények**: Olyan matematikai függvények, amelyek bemenetként egy tetszőleges méretű adatot vesznek fel és fix méretű kimenetet állítanak elő. A jó hash függvények tulajdonságai közé tartoznak a determinisztikusság, az előrejelezhetetlenség, az ütközésállóság és a gyors számítás.

Például a Secure Hash Algorithm (SHA) családja széles körben használt hash függvények sorozata. Az SHA-256 példakódja C++ nyelven az OpenSSL könyvtáron keresztül lehet használható:

```cpp
#include <openssl/sha.h>
#include <iostream>
#include <iomanip>
#include <sstream>

std::string sha256(const std::string &data) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, data.c_str(), data.size());
    SHA256_Final(hash, &sha256);
    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    return ss.str();
}

int main() {
    std::string data = "Sample data to hash";
    std::cout << "SHA-256: " << sha256(data) << std::endl;
    return 0;
}
```

#### Hitelesség (Authenticity)
A hitelesség biztosítja, hogy az információk valóban attól a személytől származnak, aki állítja. Ezen elv kulcsfontosságú az adatok megbízhatóságának védelme érdekében. Digitális aláírások és tanúsítványokat alkalmaznak ennek a követelménynek a megvalósítására.

**Digitális aláírások**: Az aláírás az információkhoz adott titkosított adatrész, amelyet az adatküldő privát kulcsával hoztak létre, és amit a fogadó a nyilvános kulccsal dekódolhat az adatok érvényességének ellenőrzésére.

**Tanúsítványok**: Digitális tanúsítványok hitelesített entitások által kibocsátott digitális dokumentumok, amelyek összekötik a nyilvános kulcsot a hitelesített entitással. A legtöbb modern titkosítási rendszer, mint a TLS (Transport Layer Security), tanúsítványokra támaszkodik a hitelesség biztosításához.

#### Nem-megtagadás (Non-repudiation)
A nem-megtagadás biztosítja, hogy az adatok elküldője később nem tagadhatja meg az adatok elküldését. Ez különösen fontos jogi és gazdasági környezetekben, ahol a tranzakciók hitelessége és megcáfolhatatlansága kulcsfontosságú.

**Digitális aláírások a nem-megtagadás érdekében**: A digitális aláírások nemcsak az adatok hitelességét biztosítják, hanem azt is garantálják, hogy az információ küldője nem tagadhatja meg annak létrehozását és küldését. Egy kompromittálatlan privát kulccsal aláírt üzenet biztosítja, hogy csak az adott személy hozhatta létre az üzenetet.


