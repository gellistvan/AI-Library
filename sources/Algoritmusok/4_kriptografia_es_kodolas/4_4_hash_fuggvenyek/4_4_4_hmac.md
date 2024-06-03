\newpage

## 4.4. HMAC (Hash-based Message Authentication Code)

A digitális világban az adatok integritásának és hitelességének megőrzése kiemelt fontosságú feladat. A Hash-based Message Authentication Code (HMAC) olyan kriptográfiai mechanizmus, amely biztosítja, hogy az adatokat nem módosították és a forrásuk hiteles. Az HMAC egy konkrét struktúrát használ, amely hash függvényeken alapul, hogy egyedi hitelesítési kódokat generáljon adatok és üzenetek számára. A következő alfejezetekben áttekintjük, hogyan működik az HMAC, milyen módszereket alkalmaz az üzenetek hitelesítésére, valamint hogy miképpen használható különböző hash függvényekkel, mint például a MD5, SHA-1 és SHA-256. Megértve és alkalmazva az HMAC mechanizmusát, képesek leszünk hatékonyan védekezni az adatok jogosulatlan módosítása és az ezzel járó biztonsági kockázatok ellen.

#### Bevezetés

A HMAC (Hash-based Message Authentication Code) egy speciális kriptográfiai konstrukció, amelyet arra használnak, hogy biztosítsák az üzenetek integritását és hitelesítését. A hitelesség és integritás biztosítása kulcsfontosságú a kommunikáció biztonsága szempontjából. A HMAC széles körben alkalmazott mechanizmus, amelyet különböző hash függvények, például SHA-256, SHA-512 és MD5 segítségével valósítanak meg.

#### Működés

A HMAC működésének alapját a kriptográfiai hash függvények adják. A standard HMAC algoritmus két fő összetevőt tartalmaz: a titkos kulcsot és a hash függvényt. Tekintsük meg a HMAC algoritmus lépéseit részletesen.

**1. Előkészítés:**

- A titkos kulcsot (`K`) először fix hosszúságra egészítik ki. Hagyományosan, ha a kulcs rövidebb mint a hash függvény blokk mérete, nullákkal egészítik ki, ha pedig hosszabb, akkor először egy hash függvényen futtatják át, hogy a kívánt hosszúságot megkapják.

**2. Kulcsok képzése:**

- Kulcs belső alapvektor: `K_ipad = K XOR ipad`, ahol `ipad` egy előre meghatározott belső kitöltés (0x36 értékkel feltöltött sorozat).
- Kulcs külső alapvektor: `K_opad = K XOR opad`, ahol `opad` egy előre meghatározott külső kitöltés (0x5c értékkel feltöltött sorozat).

**3. Hash-érték kiszámítása:**

- Belső hash: `H(K_ipad || message)`, ahol `||` a sorozatok összefűzését jelöli.
- Külső hash: `H(K_opad || belső_hash)`.

**4. Eredmény:**

- A végső HMAC érték a külső hash értéke: `HMAC(K, message) = H(K_opad || H(K_ipad || message))`.

A fenti lépések alapján látható, hogy a HMAC dupla hash függvényt használ a titkos kulcs és az üzenet közötti művelettel. Ez a technika biztosítja a magas fokú biztonságot és integritást.

#### Matematikai Modell

Formálisan a HMAC definíciója az alábbi matematikai képlettel írható le:
\[ \text{HMAC}_K(m) = H((K \oplus opad) \| H((K \oplus ipad) \| m)) \]
ahol \( H \) a hash függvény, \( K \) a kulcs, \( m \) az üzenet, \( \oplus \) az XOR művelet, és \( \| \) az összefűzés.

#### Biztonsági Tulajdonságok

A HMAC algoritmussal előállított kódok alapvetően két fontos biztonsági követelményt kielégítenek:

1. **Integritás**: Garantálja, hogy a kódolt üzenet nem módosítható anélkül, hogy a megfelelő HMAC ne változna.
2. **Hitelesség**: Biztosítja, hogy az üzenet eredetileg a megfelelő entitástól származik, feltételezve, hogy a titkos kulcs biztonságban van.

Ezen kívül a HMAC további előnyökkel is jár:

- **Hash függvény biztonsága**: A HMAC biztonsági szintje szorosan kapcsolódik az alkalmazott hash függvény biztonságához.
- **Kulcsfüggetlenség**: Még akkor is, ha a hash függvény bizonyos gyengeségei felmerülnek, a HMAC struktúra továbbra is erős marad, feltételezve, hogy a kulcsok megfelelő kezelése biztosított.

#### Gyakorlati Alkalmazások

A HMAC széles körben alkalmazható különböző biztonsági protokollokban és rendszerekben. Néhány gyakorlati példa, ahol a HMAC szerepet játszik:

1. **HTTPS és SSL/TLS**: Az SSL/TLS protokollok az internetes kommunikáció biztonságát szolgálják. A HMAC-ot használják az adat integritásának és hitelesítésének biztosítására.

2. **IPsec (Internet Protocol Security)**: A HMAC több IPsec protokollban, például az AH (Authentication Header) és ESP (Encapsulating Security Payload) protokollok esetében is alkalmazzák az adatcsomagok hitelesítésére.

3. **Digitális Aláírások**: A HMAC az autentikációs kód tulajdonságait kihasználva használható digitális aláírások esetében is, különösen a gyors hitelesítés és integritás biztosítása érdekében.

4. **Token alapú hitelesítési rendszerek**: Olyan rendszerekben, mint például az OAuth és JWT (JSON Web Token), a HMAC használatával hitelesítik és ellenőrzik a tokeneket a rendszer különböző komponensei között.

#### Implementációs Példák

A következő példában láthatjuk, hogyan lehet HMAC-ot implementálni C++ nyelven a SHA-256 hash függvény felhasználásával. Ehhez előfeltételezzük, hogy a SHA-256 függvény implementációja már rendelkezésre áll.

```cpp
#include <iostream>
#include <string>
#include <openssl/hmac.h>

// Function to compute HMAC using SHA-256
std::string computeHMAC(std::string key, std::string message) {
    unsigned char* digest;
    // EVP function for hashing
    digest = HMAC(EVP_sha256(), key.c_str(), key.length(), 
                  (unsigned char*)message.c_str(), message.length(), NULL, NULL);

    // Convert digest to hex string
    char mdString[65];
    for (int i = 0; i < 32; i++)
        sprintf(&mdString[i * 2], "%02x", (unsigned int)digest[i]);

    return std::string(mdString);
}

int main() {
    std::string key = "secret_key";
    std::string message = "message_to_be_authenticated";
    
    std::string hmac_result = computeHMAC(key, message);

    std::cout << "HMAC result: " << hmac_result << std::endl;

    return 0;
}
```

Ez a kód a HMAC számításának alapvető menetét demonstrálja C++ nyelven a SHA-256 hash függvény felhasználásával. Az `openssl/hmac.h` könyvtárat használjuk, amely az OpenSSL könyvtár része. Az OpenSSL egy jól ismert, nyílt forráskódú implementációja a SSL és TLS protokolloknak, amely számos kriptográfiai algoritmust tartalmaz.

### Működés és alkalmazás

A Hash-based Message Authentication Code (HMAC) egy kriptográfiai protokoll, amelyet az üzenet integritásának és hitelességének biztosítására fejlesztettek ki. Az HMAC egy hash függvényt kombinál egy titkos kulccsal, hogy egyedi azonosítót hozzon létre az üzenet számára. Ez a mechanizmus biztosítja, hogy az üzenetet ne lehessen módosítani anélkül, hogy a változást észlelnék, és hogy csak az illetéktelen személyek ne hozhassanak létre érvényes HMAC-értéket, mivel a titkos kulcsot nem ismerik.

Az HMAC működésének alapvető lépései a következők:
1. A titkos kulcs (K) mérete kiegyenlítésre kerül, hogy megegyezzen a hash függvény blokkméretével.
2. Az üzenetet kétszer hash-eljük két különböző kulccsal, amelyek a titkos kulcs különböző bit-szintű manipulációi.
    - Először egy belső kulccsal ("ipad") bővítjük a titkos kulcsot.
    - Majd egy külső kulccsal ("opad"), és az eredmények kombinálásával készítjük el az végleges HMAC értéket.

Az HMAC formális definiálása az alábbi módon történik:
- H: A hash függvény (pl. SHA-256, SHA-512, stb.)
- K: A titkos kulcs (bármely hosszúságú)
- B: Az H blokkmérete (általában 64 byte)
- L: Az H kimeneti mérete (pl. 256 bit a SHA-256 esetén)
- opad: B-byte hosszú érték, amelyben minden byte '0x5c'
- ipad: B-byte hosszú érték, amelyben minden byte '0x36'

Az HMAC algoritmus lépései:
1. Ha K > B: K = H(K) (ha a kulcs hosszabb mint a blokkméret, akkor hash-elezzük)
2. Ha K < B: K = K || 0x00..00 (a K-t nullákkal töltjük fel amíg el nem éri a B hosszúságot)
3. K' = K ⊕ opad (bitwise XOR a kitöltött kulcs és az opad között)
4. K'' = K ⊕ ipad (bitwise XOR a kitöltött kulcs és az ipad között)
5. HMAC_K(M) = H(K' || H(K'' || M)), ahol '||' a string konkatinációt jelzi, H(X) X hash-elt eredményét jelenti, M pedig a hitelesítendő üzenet.

A gyakorlatban az HMAC-t az alábbi célokra használják:
- Hitelésített szállítási protokollokban (pl. HTTPS, TLS).
- Az üzenet épségének biztosítása a kommunikációs rendszerekben.
- Az üzenetek hitelességének biztosítása a webes szolgáltatásokban, API-kban.

### HMAC használata különböző hash függvényekkel

Az HMAC implementálható különböző hash függvényekkel, mint például MD5, SHA-1, SHA-256 és SHA-512. A választott hash függvény meghatározza a biztonsági paramétereket — például a kimeneti méretet és a hash folyamat összetettségét. Az alábbiakban részletezzük, hogyan használható az HMAC különféle hash függvényekkel.

#### HMAC-MD5

Az MD5 (Message-Digest Algorithm 5) egy 128-bites hash funkció, amelyet széles körben használtak, de ma már biztonsági gyengeségei miatt kevésbé ajánlott. Az HMAC-MD5 használata még mindig elérhetőséget biztosít számos régi rendszer számára.

Például:
```cpp
#include <openssl/hmac.h>
#include <iostream>
#include <cstring>

void hmac_md5_example(const char* key, const char* data) {
    unsigned char* result;
    unsigned int len = 16;  // MD5 generates 16 bytes

    result = HMAC(EVP_md5(), key, strlen(key), (unsigned char*)data, strlen(data), NULL, &len);
    
    for (unsigned int i = 0; i < len; i++)
        printf("%02x", result[i]);
    printf("\n");
}

int main() {
    const char* key = "mysecretkey";
    const char* data = "Hello, World!";
    hmac_md5_example(key, data);
    return 0;
}
```

#### HMAC-SHA-1

A SHA-1 (Secure Hash Algorithm 1) 160-bites kimenetet generál, és bár sokkal biztonságosabb, mint az MD5, hajlamos néhány ismert gyengeségre, amelyek miatt nem ajánlják új fejlesztésekhez.

Például:
```cpp
#include <openssl/hmac.h>
#include <iostream>
#include <cstring>

void hmac_sha1_example(const char* key, const char* data) {
    unsigned char* result;
    unsigned int len = 20;  // SHA-1 generates 20 bytes

    result = HMAC(EVP_sha1(), key, strlen(key), (unsigned char*)data, strlen(data), NULL, &len);
    
    for (unsigned int i = 0; i < len; i++)
        printf("%02x", result[i]);
    printf("\n");
}

int main() {
    const char* key = "mysecretkey";
    const char* data = "Hello, World!";
    hmac_sha1_example(key, data);
    return 0;
}
```

#### HMAC-SHA-256

A SHA-256 a SHA-2 család tagja, amely 256-bites kimenettel rendelkezik. Sokkal biztonságosabb, mint a korábbi algoritmusok, és széles körben ajánlott.

Például:
```cpp
#include <openssl/hmac.h>
#include <iostream>
#include <cstring>

void hmac_sha256_example(const char* key, const char* data) {
    unsigned char* result;
    unsigned int len = 32;  // SHA-256 generates 32 bytes

    result = HMAC(EVP_sha256(), key, strlen(key), (unsigned char*)data, strlen(data), NULL, &len);
    
    for (unsigned int i = 0; i < len; i++)
        printf("%02x", result[i]);
    printf("\n");
}

int main() {
    const char* key = "mysecretkey";
    const char* data = "Hello, World!";
    hmac_sha256_example(key, data);
    return 0;
}
```

#### HMAC-SHA-512

A SHA-512 szintén a SHA-2 család tagja, de 512-bites kimenettel. Ez az egyik legbiztonságosabb hash algoritmus, és széles körben alkalmazzák a nagy biztonsági követelményű rendszerekben.

Például:
```cpp
#include <openssl/hmac.h>
#include <iostream>
#include <cstring>

void hmac_sha512_example(const char* key, const char* data) {
    unsigned char* result;
    unsigned int len = 64;  // SHA-512 generates 64 bytes

    result = HMAC(EVP_sha512(), key, strlen(key), (unsigned char*)data, strlen(data), NULL, &len);
    
    for (unsigned int i = 0; i < len; i++)
        printf("%02x", result[i]);
    printf("\n");
}

int main() {
    const char* key = "mysecretkey";
    const char* data = "Hello, World!";
    hmac_sha512_example(key, data);
    return 0;
}
```

### Választás az HMAC algoritmusok között

Az HMAC használata során a választott hash függvény jelentős hatással van a biztonsági modellekre és a teljesítményre. Az MD5 és SHA-1 már nem ajánlottak modern alkalmazásokhoz, mivel ismert sebezhetőségeik vannak. Ehelyett a SHA-256 vagy a SHA-512 ajánlott, amelyek jelenleg az egyik legbiztonságosabb hash funkciók.

#### Hash Függvények Összehasonlítása

| Hash Függvény | Kimeneti Méret (bit) | Biztonság | Alkalmazási Területek |
|:-------------:|:--------------------:|:----------:|:-----------------------|
| MD5           | 128                  | Alacsony   | Régi rendszerek, tesztelés |
| SHA-1         | 160                  | Közepes    | Néhány régi rendszer    |
| SHA-256       | 256                  | Magas      | Modern alkalmazások     |
| SHA-512       | 512                  | Nagyon magas | Magas biztonsági követelmények |

Az HMAC kiváló választás, ha integritást és hitelességet szeretnénk biztosítani egy üzenet számára. Az alkalmazások széles körében bevált módszer, például hitelesített adatkommunikációban, API hitelesítésben és fájl integritás ellenőrzésében. A választott hash függvény és a biztonsági követelmények meghatározzák az alkalmazás megfelelő HMAC mechanizmusát.



