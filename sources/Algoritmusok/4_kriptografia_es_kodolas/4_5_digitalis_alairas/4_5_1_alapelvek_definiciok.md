\newpage

# 5. Digitális aláírások

A digitális aláírások a modern kriptográfia egyik legfontosabb és legelterjedtebb alkalmazásai közé tartoznak. A digitális világban, ahol az információáramlás gyors és gyakran bizalmas adatok tranzakcióján alapul, alapvető fontosságú a kommunikáció hitelességének, integritásának és a részt vevők azonosításának biztosítása. A digitális aláírások választ adnak ezekre a kihívásokra, és lehetővé teszik, hogy az elektronikus dokumentumok éppúgy kapjanak megbízhatósági tanúsítványt, mint a hagyományos, papíralapú megfelelőik. Ebben a szekcióban áttekintjük a digitális aláírások alapelveit, működési mechanizmusát, valamint a leggyakrabban használt digitális aláírási algoritmusokat, amelyek napjainkban az online biztonság pilléreiként szolgálnak.

## 5.1. Alapelvek és definíciók

A digitális aláírások a modern kriptográfiai rendszerek alapvető eszközei közé tartoznak, melyek lehetővé teszik az elektronikus dokumentumok biztonságos hitelesítését és integritásának biztosítását. Ebben a fejezetben megvizsgáljuk a digitális aláírások működési elvét, bemutatjuk a legfontosabb tulajdonságaikat és részletezzük az aláírás létrehozásának és ellenőrzésének folyamatát. Megismerkedünk az aláírások mögött álló matematikai struktúrákkal és algoritmusokkal, valamint feltárjuk, hogyan használják ezeket a gyakorlatban a hitelesség, az integritás és a nem-visszautasíthatóság garantálása érdekében. Ez a rész átfogó képet nyújt a digitális aláírások technikai hátteréről, és rávilágít annak fontosságára a digitális kommunikáció és adatvédelem területén.

#### Digitális aláírások működése

A digitális aláírás egy kriptográfiai mechanizmus, amely lehetővé teszi az üzenetek vagy dokumentumok hitelességének és integritásának ellenőrzését, valamint az aláíró személyazonosságának igazolását. Az elektronikus üzenetek és dokumentumok megbízhatóságának biztosítása különösen fontos a digitális kommunikációban és tranzakciókban. A digitális aláírások gyakran alkalmaznak nyilvános kulcsú kriptográfiát, ahol egy privát kulcsot (titkos kulcs) és egy nyilvános kulcsot (publikus kulcs) használnak.

#### Folyamat alapjai

A digitális aláírás két kulcsfontosságú folyamatot foglal magában:
1. **Aláírás létrehozása (Signing)**: Az aláíró a saját privát kulcsával egyedi aláírást hoz létre az üzenet vagy dokumentum számára.
2. **Aláírás ellenőrzése (Verification)**: A fogadó fél az aláíró nyilvános kulcsának segítségével ellenőrzi, hogy az aláírás valóban az eredeti üzenethez vagy dokumentumhoz tartozik-e.

#### Algoritmusok és eljárások

1. **Hashing (Kivonatképzés)**:
    - Először az üzenetből vagy dokumentumból egy hash értéket képeznek egy kivonatképzési eljárással (például SHA-256). A hash függvények egy fix hosszúságú kivonatot állítanak elő az eredeti üzenet tetszőleges hosszúságáról, és bármilyen kis változás az üzenetben drámai változást eredményez a kivonatban.
    - A kivonat alkalmazása biztosítja, hogy az aláírás az üzenet egyedi lenyomatát tükrözi, és így bármilyen módosítást azonnal észlelhetővé tesz.

2. **Privát kulcsú aláírás**:
    - Az aláíró a saját privát kulcsát használja a hash érték kriptográfiai aláírására. Ez általában egy aszimmetrikus kriptográfiai algoritmussal történik, mint például az RSA vagy az ECDSA (Elliptic Curve Digital Signature Algorithm).
    - A titkos kulccsal végzett művelet eredménye maga a digitális aláírás, amely az eredeti üzenettel együtt küldhető.

3. **Nyilvános kulcsú ellenőrzés**:
    - A fogadó fél az aláíró nyilvános kulcsával visszafejtheti a digitálisan aláírt hash értéket.
    - Ezután a fogadó fél újra kiszámítja az üzenet hash értékét a használt kivonatképzési eljárással, és összehasonlítja azt a visszafejtett értékkel.
    - Ha a két hash érték egyezik, akkor az aláírás és az üzenet hitelesítése sikeres, és biztos, hogy az üzenet nem változott meg az aláírás óta.

Lássunk egy példát az RSA-algoritmus alkalmazására a digitális aláírásban:

```cpp
#include <openssl/rsa.h>

#include <openssl/pem.h>
#include <openssl/err.h>

#include <openssl/sha.h>
#include <iostream>

#include <fstream>

void generate_rsa_keypair() {
    int bits = 2048;
    unsigned long exp = RSA_F4;
    RSA *rsa = RSA_generate_key(bits, exp, NULL, NULL);
    BIO *pri = BIO_new_file("private_key.pem", "w+");
    PEM_write_bio_RSAPrivateKey(pri, rsa, NULL, NULL, 0, NULL, NULL);
    BIO *pub = BIO_new_file("public_key.pem", "w+");
    PEM_write_bio_RSAPublicKey(pub, rsa);
    RSA_free(rsa);
    BIO_free_all(pri);
    BIO_free_all(pub);
}

std::string sign_message(const std::string& message, const std::string& private_key_path) {
    RSA *rsa = RSA_new();
    BIO *pri = BIO_new_file(private_key_path.c_str(), "r");
    PEM_read_bio_RSAPrivateKey(pri, &rsa, NULL, NULL);

    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(message.c_str()), message.size(), hash);

    unsigned char *sig = new unsigned char[RSA_size(rsa)];
    unsigned int sig_len = 0;
    RSA_sign(NID_sha256, hash, SHA256_DIGEST_LENGTH, sig, &sig_len, rsa);

    std::string signature(reinterpret_cast<char*>(sig), sig_len);

    delete[] sig;
    BIO_free(pri);
    RSA_free(rsa);

    return signature;
}

bool verify_signature(const std::string& message, const std::string& signature, const std::string& public_key_path) {
    RSA *rsa = RSA_new();
    BIO *pub = BIO_new_file(public_key_path.c_str(), "r");
    PEM_read_bio_RSAPublicKey(pub, &rsa, NULL, NULL);

    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(message.c_str()), message.size(), hash);

    int result = RSA_verify(NID_sha256, hash, SHA256_DIGEST_LENGTH, reinterpret_cast<const unsigned char*>(signature.c_str()), signature.size(), rsa);

    BIO_free(pub);
    RSA_free(rsa);

    return result == 1;
}

int main() {
    generate_rsa_keypair();

    std::string message = "This is a confidential message.";
    std::string sig = sign_message(message, "private_key.pem");

    bool is_valid = verify_signature(message, sig, "public_key.pem");

    std::cout << "Signature valid: " << (is_valid ? "true" : "false") << std::endl;

    return 0;
}
```

Ez a kód az OpenSSL könyvtárat használja digitális aláírás létrehozására és ellenőrzésére RSA algoritmussal. A példában először egy RSA kulcspárt generálunk, majd létrehozzuk és ellenőrizzük az aláírást.

#### Digitális aláírások tulajdonságai

1. **Hitelesség (Authenticity)**:
    - A digitális aláírás biztosítja, hogy az aláíró valóban ő maga, és az üzenetet vagy dokumentumot ténylegesen ő küldte. Ez a tulajdonság az aláíró privát kulcsának és a fogadó fél privát kulcsának egyedi összhangján alapul.

2. **Integritás (Integrity)**:
    - Az üzenet vagy dokumentum tartalmának változatlansága ellenőrizhető az aláírás segítségével. Bármilyen módosítás az üzenetben vagy dokumentumban az aláírás érvénytelenségét eredményezi, mert a kivonat értéke megváltozik.

3. **Nem visszautasíthatóság (Non-repudiation)**:
    - Az aláíró nem tagadhatja le az aláírás létrehozását. Az aláírással és az aláíró nyilvános kulcsával bizonyítható, hogy az üzenet ténylegesen az adott aláírótól származik.

#### Kriptográfiai biztonság

A digitális aláírások biztonsága több rétegű, magában foglalja a kriptográfiai algoritmusok, kulcskezelési eljárások és protokollok összjátékát. A következő tényezőket kell figyelembe venni:

1. **Kulcsok hossza és erőssége**:
    - A használt kulcsok hossza jelentősen meghatározza az algoritmus biztonsági szintjét. Hosszabb kulcsok általában nagyobb biztonságot nyújtanak.
    - Az RSA esetén például legalább 2048 bites kulcs ajánlott a megfelelő biztonság érdekében.

2. **Hash függvények biztonsága**:
    - A használt hash függvények kiválasztására szintén nagy figyelmet kell fordítani. Biztonságos hash algoritmusok, mint például az SHA-256 vagy ennek nagyobb változatai, széles körben elfogadottak a kriptográfiai közösségben.

3. **Privát kulcs védelme**:
    - A privát kulcsok fizikai és digitális védelme kritikus fontosságú. Ha egy támadó hozzáfér a privát kulcshoz, képes lesz hamis aláírásokat létrehozni, és így teljesen kompromittálhatja az egész rendszer biztonságát.
    - Titkos kulcsokat általában titkosított tárolásban és mögöttes hardvereszközökön, például hardverbiztonsági modulokban (HSM-ek) tárolunk.

4. **Protokollok biztonsága**:
    - Az aláírás létrehozási és ellenőrzési protokolloknak biztonságosnak kell lenniük az időzítési támadásokkal, közvetítői támadásokkal és egyéb ismert kriptográfiai támadásokkal szemben.

Összefoglalva, a digitális aláírások összetett kriptográfiai mechanizmusokon alapulnak, amelyek biztosítják az üzenetek és dokumentumok hitelességét, integritását és nem visszautasíthatóságát. Ennek érdekében alaposan válogatott kriptográfiai algoritmusokra, kulcskezelési eljárásokra és biztonságos protokollokra van szükség.

### Digitális aláírások működése és tulajdonságai

A digitális aláírások a modern kriptográfiában kulcsfontosságú szerepet játszanak, mivel lehetővé teszik az információ hitelességének és integritásának ellenőrzését. Hasonlóan a kézzel írott aláíráshoz, a digitális aláírás bizonyítja a dokumentum létrehozójának személyazonosságát, és biztosítja annak érintetlenségét. Matematikailag alátámasztott protokollokra és algoritmusokra épül, amelyek biztosítják a nyilvános kulcsú kriptográfia rendszereinek következő fontos tulajdonságait:

1. **Hitelesség (Authenticity)**: Az aláírás egyértelműen azonosítja az aláírót.
2. **Integritás (Integrity)**: Garantálja, hogy a dokumentum nem változott az aláírás létrehozása óta.
3. **Visszautasíthatatlanság (Non-repudiation)**: Az aláíró nem tagadhatja meg egyértelműen az aláírását.

### Aláírás létrehozása és ellenőrzése

#### Alapvető lépések az aláírás létrehozásában és ellenőrzésében

A digitális aláírás létrehozásának és ellenőrzésének folyamata számos matematikai és kriptográfiai műveletből áll, amelyek nyilvános és privát kulcsok használatán alapulnak. Az eljárás két fő fázisból áll: az aláírás létrehozásából és az aláírás ellenőrzéséből.

##### Aláírás létrehozása

Az aláírás létrehozásakor az aláíró a következő lépéseket hajtja végre:

1. **Hash-képzés (Hashing)**: A dokumentum vagy az üzenet hash értékének létrehozása egy biztonságos hash-függvény segítségével, például SHA-256 alkalmazásával. Ez a lépés eredményezi a fix hosszúságú összegképet (message digest), amely a dokumentum egy egyedi ujjlenyomata.

   Példakód (angol, C++):
   ```cpp
   #include <openssl/sha.h>
   #include <iostream>
   #include <vector>

   std::vector<unsigned char> hashDocument(const std::string& document) {
       std::vector<unsigned char> hash(SHA256_DIGEST_LENGTH);
       SHA256_CTX sha256;
       SHA256_Init(&sha256);
       SHA256_Update(&sha256, document.c_str(), document.length());
       SHA256_Final(hash.data(), &sha256);
       return hash;
   }
   ```

2. **Privát kulcsú aláírás (Signing with Private Key)**: Az aláíró az összegképet a saját privát kulcsával titkosítja. Ez a művelet eredményezi a digitális aláírást. A titkosításhoz általában aszimmetrikus kriptográfiai algoritmusokat, például RSA-t vagy elliptikus görbéket használnak.

   Példakód (angol, C++):
   ```cpp
   #include <openssl/rsa.h>
   #include <openssl/pem.h>
   #include <openssl/err.h>
   #include <cstring>
   #include <cstdlib>

   std::vector<unsigned char> signDocument(const std::vector<unsigned char>& hash, const std::string& privateKeyStr) {
       RSA* rsa = nullptr;
       BIO* bio = BIO_new_mem_buf(privateKeyStr.c_str(), -1);
       PEM_read_bio_RSAPrivateKey(bio, &rsa, nullptr, nullptr);

       std::vector<unsigned char> signature(RSA_size(rsa));
       unsigned int signatureLen = 0;

       RSA_sign(NID_sha256, hash.data(), hash.size(), signature.data(), &signatureLen, rsa);

       BIO_free(bio);
       RSA_free(rsa);
       return signature;
   }
   ```

##### Aláírás ellenőrzése

Az ellenőrzési folyamat során a címzett a következő lépéseket végzi el:

1. **Hash érték újraszámítása (Re-compute the Hash)**: Az eredeti dokumentum (amelyet meg kell erősíteni) hash értékét újraszámolják ugyanazon hash függvény alkalmazásával, mint az aláírás létrehozásakor.

   Példakód (angol, C++):
   ```cpp
   std::vector<unsigned char> reHashDocument(const std::string& document) {
       return hashDocument(document);  // same as above
   }
   ```

2. **Aláírás visszafejtése (Decrypting the Signature)**: Az aláírást a közzétett nyilvános kulcs segítségével visszafejtik. Ez azt eredményezi, hogy az aláírt hash érték helyreáll.

   Példakód (angol, C++):
   ```cpp
   #include <openssl/rsa.h>
   #include <openssl/pem.h>
   #include <openssl/err.h>
   #include <vector>

   bool verifySignature(const std::vector<unsigned char>& hash, const std::vector<unsigned char>& signature, const std::string& publicKeyStr) {
       RSA* rsa = nullptr;
       BIO* bio = BIO_new_mem_buf(publicKeyStr.c_str(), -1);
       PEM_read_bio_RSA_PUBKEY(bio, &rsa, nullptr, nullptr);

       int result = RSA_verify(NID_sha256, hash.data(), hash.size(), signature.data(), signature.size(), rsa);

       BIO_free(bio);
       RSA_free(rsa);
       return (result == 1);
   }
   ```

3. **Hash értékek összehasonlítása (Compare Hashes)**: Az eredeti dokumentumból újraszámított hash értéket összehasonlítják az aláírás visszafejtéséből kapott hash értékkel. Ha a két hash egyezik, akkor a dokumentum hiteles és érintetlen, és a címzett biztos lehet abban, hogy az dokumentumot a feltételezett aláíró írta alá.

#### Kriptográfiai algoritmusok használata

A digitális aláírások létrehozásában és ellenőrzésében különböző kriptográfiai algoritmusokat használnak. Ezek közül a legismertebbek a következők:

1. **RSA** (Rivest-Shamir-Adleman): Az RSA egy széles körben használt aszimmetrikus kriptográfiai algoritmus, amely biztonságot nyújt az aláírás és a titkosítás számára is. Az RSA algorithmus a számelmélet egyik legfontosabb eredményén, a prímszám tényezők magánlengyületén alapul.

2. **DSA** (Digital Signature Algorithm): A DSA az Egyesült Államok kormányzata által kifejlesztett digitális aláírás algoritmus, amely a számelmélet egyik speciális tartományán, a diszkrét logaritmus problémán alapul.

3. **ECDSA** (Elliptic Curve Digital Signature Algorithm): Az ECDSA az elliptikus görbe kriptográfiára (ECC) épül, amely kisebb kulcsméretek mellett is hasonló vagy még nagyobb biztonságot nyújt, mint az RSA és a DSA.

### Hallmark Példák

A fent említett algoritmusok mindegyike a nyilvános kulcsú kriptográfia különböző elvein és matematikai konstrukcióin alapul, de a digitális aláírások mechanizmusa minden esetben ugyanazon alapvető lépésekből áll: hash-képzés, privát kulcsú aláírás, nyilvános kulcsú ellenőrzés és hash értékek összehasonlítása.

Ez a részletes vizsgálat kiemeli a digitális aláírások fontosságát és komplexitását a kriptográfiai rendszerekben, és biztosítja a megfelelő alapot az aláírások hitelességének, integritásának és visszautasíthatatlanságának biztosítására irányuló további kutatásokhoz és fejlesztésekhez.

