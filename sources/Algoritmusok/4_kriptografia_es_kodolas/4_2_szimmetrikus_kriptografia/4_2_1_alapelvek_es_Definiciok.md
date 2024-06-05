\newpage

# 2. Szimmetrikus kulcsú kriptográfia

## 2.1. Alapelvek és definíciók

A szimmetrikus kulcsú kriptográfia az információvédelmi technológiák egyik alapvető pillére, amely már több évtizedes múltra tekint vissza. Ebben a fejezetben bemutatjuk azokat az alapelveket és definíciókat, amelyek elengedhetetlenek a szimmetrikus kulcsú titkosítás és dekódolás megértéséhez. A szimmetrikus kulcsú titkosítás lényege, hogy ugyanazt a titkos kulcsot használjuk mind az adat titkosítására, mind annak visszafejtésére. Az ilyen rendszerekben tehát nagy jelentősége van a kulcs titkosságának és biztonságának, mivel a kulcs illetéktelen kezekbe kerülése esetén az összes rejtjelezett információ sérülékennyé válik. Továbbá, különbséget teszünk a blokk titkosítók és áram titkosítók között, megértve azok működési elveit és alkalmazási területeit. Mikro- és makroszintű megközelítések révén mélyedünk el ezekben a technikákban, hogy átfogó képet kapjunk a szimmetrikus kulcsú kriptográfia világáról.

### Szimmetrikus kulcsú titkosítás és dekódolás


Szimmetrikus kulcsú titkosítás az egyik legelterjedtebb és legősibb módszer az adatvédelem biztosítására a kriptográfiában. Az alapelv az, hogy mind a titkosításhoz, mind pedig a dekódoláshoz ugyanazt a kulcsot használjuk. Ebben a fejezetben részletesen ismertetjük a szimmetrikus kulcsú titkosítás alapelveit, történetét, működését, különféle típusait és a hozzá kapcsolódó matematikai hátteret.

#### Történeti áttekintés

A szimmetrikus kulcsú titkosítás egyik legkorábbi példája a Caesar-kód névre hallgató titkosítás, mely Julius Caesar római tábornokhoz és politikushoz kapcsolódik. Ebben az egyszerű titkosítási módszerben minden betű a titkosítandó üzenetben az ábécében egy meghatározott számú hellyel eltolódik. Hasonlóan egyszerű módszerek közé tartozik a Vigenère-kód, ami polialfabetikus titkosítást alkalmaz, vagyis több különböző helyettesítő táblázatot használ egyazon üzenet titkosítására.

Az információs szimmetrikus kulcsú titkosítás fejlődése során matematikailag kifinomultabb és hatékonyabb algoritmusok születtek, amelyek sokkal nehezebbé tették a feltörést. Az ilyen algoritmusok közé tartoznak például az adatvédelmi szabvány (DES) és a haladó titkosítási szabvány (AES).

#### Alapelvek

Szimmetrikus kulcsú titkosítás esetében mind a titkosító, mind a dekódoló fél ugyanazt a titkos kulcsot használja. Ez a kulcs egy olyan bit-sorozat, amelyet mindkét fél előzetesen megosztott és titkosan kezelt. A legfontosabb követelmény az, hogy bárki más ne férhessen hozzá ehhez a kulcshoz; különben az adatvédelem sérül, és az üzenetet bárki dekódolhatja.

Az algoritmusok két fő osztályba sorolhatók: blokk titkosítók és áram titkosítók. Blokk titkosítók rögzített méretű blokkban titkosítanak, míg áram titkosítók az adatfolyam egyes bitjeit egyenként.

#### Blokk titkosítók

Blokk titkosítók esetében a bemeneti adat egy-egy meghatározott hosszúságú blokkra bontva kerül titkosításra. Az ilyen algoritmusok általában a következőképpen működnek:

1. **Blokk osztás**: Az üzenetet rögzített hosszúságú blokkokra (pl. 128 bit) bontjuk.
2. **Titkosítás**: Minden blokkot külön-külön titkosítunk egy szimmetrikus kulcs segítségével. Az adott blokk egy ugyanilyen hosszúságú titkosított blokká alakul.
3. **Összefűzés**: A titkosított blokkokat összefűzzük, hogy a titkosított üzenetet megkapjuk.

Az ismert blokk titkosítási algoritmusok közé tartozik a DES, Triple DES (3DES), és AES.

##### Általános Blokk Titkosító Sémák

Blokk titkosítóknak többféle üzemmódja létezik, amelyek különböző védelmet nyújtanak különböző típusú támadások ellen:

- **Electronic Codebook (ECB)**: Minden blokkot külön titkosítunk. Ennek az üzemmódnak az egyik hátránya, hogy az azonos bemeneti blokkoknak azonos kimeneti blokkok felelnek meg, ami mintázatokat hozhat létre a titkosított üzenetben.
- **Cipher Block Chaining (CBC)**: Minden blokkot a megelőző blokk titkosított kimenetével kombinálunk egy XOR művelettel, mielőtt titkosítjuk. Ez biztosítja, hogy az azonos blokkok különböző kimenetre vezethetnek, amely növeli a biztonságot.
- **Counter (CTR)**: Minden blokk egy számlálón alapuló értéket titkosít, és ezt az értéket XOR-olja a blokk tartalmával. Ez az üzemmód nagyfokú párhuzamosságot tesz lehetővé, mert a számláló értékei előre kiszámíthatók.

#### Áram titkosítók

Az áram titkosító (stream cipher) az adatokat egy folyamatos bit- vagy byte-folyamként kezeli, és minden egyes bitet külön titkosít. Általában egy belső állapotot tartanak fenn, amelyet a titkos kulcs és egy kezdő érték (initialization vector, IV) kezdeményez.

Áram titkosítók jellemzően a XOR műveletet használják az eredeti üzenet és egy pszeudo-véletlen bit-folyam kombinációjára. Egyik legelterjedtebb áram titkosító az RC4, amely széles körben alkalmazott különböző hálózati protokollokban, mint például a WEP és egyes SSL/TLS verziók. Azonban az RC4 jelentős gyengeségekre derült fény az évek során, így sok modern alkalmazásban már kerülendő.

#### Matematikai Hátter

A szimmetrikus kulcsú titkosítási algoritmusok gyakran alkalmaznak többféle matematikai műveletet, mint például:

- **Helyettesítés (Substitution)**: Az eredeti üzenet egyes elemei (például karakterek vagy bitek) más elemekkel helyettesítők. Ez a technika biztosítja a konfidencialitást azáltal, hogy elhomályosítja az adat valódi értékét.
- **Permutáció (Permutation)**: Az eredeti üzenetbitjeinek sorrendjét megváltoztatják. Ez által az adat szerkezete teljesen átrendeződik, további nehézséget okozva egy támadónak.
- **Matematikai Moduláció (Modular Arithmetic)**: Sok algoritmusban kerül alkalmazásra, főleg azért, hogy nagy számokat egyszerű műveletekkel kezelhessünk (például AES esetében).

##### Példakód - DES Algoritmus C++-ban

Ha részletes példakódot szeretnénk megvizsgálni, nézzük meg a Data Encryption Standard (DES) egyszerű implementációját C++-ban:

```cpp
#include <iostream>

#include <bitset>
#include <string>

#include <array>

// Placeholder for S-box
const std::array<std::array<int, 64>, 8> S_BOX {{
    { ... /* S1 values */ },
    { ... /* S2 values */ },
    { ... /* S3 values */ },
    { ... /* S4 values */ },
    { ... /* S5 values */ },
    { ... /* S6 values */ },
    { ... /* S7 values */ },
    { ... /* S8 values */ }
}};

// Permutation function
std::bitset<64> permute(const std::bitset<64>& input, const int* permTable, int size) {
    std::bitset<64> output;
    for (int i = 0; i < size; ++i) {
        output[size - i - 1] = input[64 - permTable[i]];
    }
    return output;
}

// DES key schedule
std::array<std::bitset<48>, 16> generateSubkeys(const std::bitset<64>& key) {
    // Key generation implementation goes here
    std::array<std::bitset<48>, 16> subkeys;
    // ...
    return subkeys;
}

// DES Feistel (F) function
std::bitset<32> feistel(const std::bitset<32>& R, const std::bitset<48>& subkey) {
    std::bitset<48> expanded = permute(R, EXPANSION_TABLE, 48);
    std::bitset<48> xored = expanded ^ subkey;
    std::bitset<32> output;
    // S-Box substitution and P permutation
    // ...
    return output;
}

// DES encryption function
std::bitset<64> DES_encrypt(const std::bitset<64>& plainText, const std::bitset<64>& key) {
    std::array<std::bitset<48>, 16> subkeys = generateSubkeys(key);
    std::bitset<64> cipherText = permute(plainText, INIT_PERM_TABLE, 64);
    std::bitset<32> L = cipherText >> 32;
    std::bitset<32> R = cipherText & std::bitset<64>(0xFFFFFFFF);
    for (int i = 0; i < 16; ++i) {
        std::bitset<32> temp = R;
        R = L ^ feistel(R, subkeys[i]);
        L = temp;
    }
    cipherText = (R.to_ulong() << 32) | L.to_ulong();
    cipherText = permute(cipherText, FINAL_PERM_TABLE, 64);
    return cipherText;
}

int main() {
    std::bitset<64> key(0x133457799BBCDFF1);
    std::bitset<64> plainText(0x0123456789ABCDEF);
    std::bitset<64> cipherText = DES_encrypt(plainText, key);

    std::cout << "Cipher Text: " << cipherText.to_ulong() << std::endl;
    return 0;
}
```

#### Biztonsági Megfontolások

Bár a szimmetrikus kulcsú titkosítás gyors és hatékony, néhány veszélyt is rejt magában. Az egyik legnagyobb probléma a kulcskezelés. Mindkét félnek biztonságos csatornát kell találnia a közös kulcs megosztásához. Ha a kulcsot valamilyen módon megszereznék, az egész kommunikáció súlyosan veszélybe kerül.

Ráadásul a szimmetrikus kulcsú titkosítás nem alkalmas olyan környezetekben, ahol nagyszámú felhasználó létezik, mert minden egyes felhasználói párosnak külön kulccsal kell rendelkeznie.


### Blokk titkosítók és áram titkosítók

A szimmetrikus kulcsú kriptográfia a titkosítás egyik legelterjedtebb formája, amely a kulcs mindkét fél (a titkosítást végző és a dekódolást végző) számára közös. Ez a közös kulcs biztosítja az adatok bizalmasságát és integritását. A szimmetrikus kriptográfiai algoritmusok tovább oszthatók blokk titkosítókra és áram titkosítókra. Mindkettő különböző elveken és működési módokon alapul, és különböző előnyökkel és hátrányokkal rendelkezik.

### Blokk titkosítók

**Blokk titkosítók** (block ciphers) olyan algoritmusok, amelyek az adatokat előre meghatározott méretű blokkokban dolgozzák fel. Ezek a blokkok általában 64 vagy 128 bit méretűek, de lehetnek más méretek is. A blokk titkosítóknál az eredeti (vagy sima) szöveget először felosztják ezekre a blokkokra, majd minden egyes blokkot külön titkosítanak.

#### Működési elv

A titkosítási folyamat során minden blokk belső állapota attól függ, hogy milyen műveleteket végzünk el rajta a titkos kulcs használatával. Egy tipikus blokk titkosító az alábbi lépéseket követi:

1. **Felosztás blokkokra**: Az adatok felosztása előre meghatározott méretű blokkokra. Ha az adatok nem teljesen illeszkednek a blokk méretéhez, padding-et kell alkalmazni.
2. **Kulcs-generálás**: A szimmetrikus kulcs generálása és elosztása a felek között.
3. **Titkosítási körök**: A titkosítást többlépcsős („körök”) folyamatban végzik el, ahol minden körben különböző műveleteket végeznek a blokkon a kulcs részei felhasználásával.
4. **Kimeneti blokk**: A feldolgozott blokk kimeneteként kapott titkosított adat.

#### Példa algoritmusok

- **Data Encryption Standard (DES)**: 56 bites kulcsot használ, és 64 bites blokkokat dolgoz fel.
- **Advanced Encryption Standard (AES)**: Kulcsméret 128, 192 vagy 256 bit lehet, és 128 bites blokkokat dolgoz fel.

#### Példa egy AES titkosításra C++ nyelven (using OpenSSL):

```cpp
#include <openssl/aes.h>

#include <openssl/rand.h>
#include <iostream>

#include <cstring>

void handleErrors(void) {
    std::cerr << "An error occurred\n";
    abort();
}

int main() {
    // AES key and IV
    unsigned char key[16]; 
    unsigned char iv[16];
    
    // Generate random key and IV
    if (!RAND_bytes(key, sizeof(key)) || !RAND_bytes(iv, sizeof(iv))) {
        handleErrors();
    }

    // Plain text
    unsigned char text[] = "This is a secret!";

    // Buffer for encrypted text
    unsigned char encrypted[128];

    // Create and initialize the context
    AES_KEY enc_key;
    if (AES_set_encrypt_key(key, 128, &enc_key) < 0) {
        handleErrors();
    }

    // Perform encryption
    AES_cfb128_encrypt(text, encrypted, sizeof(text), &enc_key, iv, 0, AES_ENCRYPT);

    std::cout << "Encrypted text: " << encrypted << "\n";

    return 0;
}
```

### Áram titkosítók

**Áram titkosítók** (stream ciphers) eltérően működnek a blokk titkosítókhoz képest. Ezekben az algoritmusokban az adatokat nem blokkokra bontják, hanem folyamatos adatfolyamként kezelik, bitenként vagy byte-onként titkosítva. Az áram titkosítók úgy működnek, hogy egy pszeudo-véletlen bitszekvenciát generálnak a kulcs alapján, és ezt XOR művelettel egyesítik az eredeti adatokkal.

#### Működési elv

1. **Kulcs-generálás**: A szimmetrikus kulcs generálása és elosztása a felek között.
2. **Initializációs vektor (IV)**: Egy kezdeti véletlen érték (INITIALIZATION VECTOR) generálása a titkosítási folyamathoz.
3. **Pszeudo-véletlen bitszekvencia generálása**: A kulcs és az IV alapján a titkosító egy bitszekvenciát generál, amely hasonlóan véletlenszerű, mintha valódi véletlen sorozat lenne.
4. **XOR művelet**: Az eredeti adatokat bitenként vagy byte-onként XOR-oljuk a generált szekvenciával, létrehozva a titkosított adatokat.

#### Példa algoritmusok

- **RC4**: Egy egyszerű, de hatékony áram titkosító.
- **Salsa20**: Egy modern és biztonságos áram titkosító.

#### Példa RC4 titkosításra C++ nyelven:

```cpp
#include <openssl/rc4.h>

#include <openssl/rand.h>
#include <iostream>

#include <cstring>

void handleErrors(void) {
    std::cerr << "An error occurred\n";
    abort();
}

int main() {
    // Key and plaintext
    unsigned char key[] = "thekey";
    unsigned char text[] = "This is a secret!";

    // Context for RC4
    RC4_KEY rc4_key;

    // Set up the RC4 key structure
    RC4_set_key(&rc4_key, strlen((char *)key), key);

    // Buffer for output
    unsigned char output[128];

    // Perform encryption
    RC4(&rc4_key, sizeof(text), text, output);

    std::cout << "Encrypted text: " << output << "\n";

    return 0;
}
```
