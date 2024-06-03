\newpage

## 4.3. Modern hash algoritmusok

## 4.3.1. SHA-2 család (SHA-224, SHA-256, SHA-384, SHA-512)

Az SHA-2 (Secure Hash Algorithms 2) család algoritmusait a National Security Agency (NSA) fejlesztette ki, majd 2001-ben a National Institute of Standards and Technology (NIST) által publikálták. Az SHA-2 család magában foglalja az alábbi hash függvényeket: SHA-224, SHA-256, SHA-384, és SHA-512. Ezek az algoritmusok különböző bitméretű kimeneteket (hash értékeket) eredményeznek, és különböző számú iterációt használnak a blokkfeldolgozó circuit-jükben.

#### Működés

##### Áttekintés

A SHA-2 algoritmusok blokk alapú hash függvények, ami azt jelenti, hogy az input adatot fix méretű blokkokra bontják, majd egy iteratív folyamaton hajtják keresztül, amely adatintegritás biztosítása céljából kriptográfiai műveleteket végez. Az alapvető működésük az alábbi lépéseket tartalmazza:

1. **Előkészítés (Padding):** Az input adatot úgy kell megváltoztatni, hogy annak hossza a blokk méretének (512 vagy 1024 bit) többszöröse legyen.
2. **Kezdeti hash értékek:** Minden SHA-2 algoritmus egy előre meghatározott kezdeti hash értékekkel (initial hash values) kezd dolgozni.
3. **Üzenet ütemezés (Message Schedule):** Az aktuális blokk feldarabolása és átrendezése történik, amely az iterációk során használt beviteli értékeket előállítja.
4. **Kompressziós függvény (Compression Function) iterációk:** Ez az iteratív folyamat minden blokkra alkalmazva frissíti a hash állapotot.
5. **Kimeneti hash érték:** Az összes blokk feldolgozása után az aktuális hash állapot lesz a végleges hash érték, amelyet a bemeneti adathoz rendelünk.

##### Padding

Hasonlóan az elődökhöz, például az SHA-1 alapú hash függvényekhez, a SHA-2 családban is előkészítési (padding) lépés szükséges ahhoz, hogy biztosítsuk a bemeneti adathossz fix többszörösét. Az előkészítés alapvetően az alábbiak szerint történik:

1. Hozzáadunk egy bitet érvényes "1" értékkel.
2. Padding bit nullákat adunk hozzá úgy, hogy a teljes hossz - 64 bittel kisebb, mint a blokk mérete - osztható legyen a blokk méretével.
3. Az utolsó 64 biten felvisszük az eredeti adat hosszát (bitben kifejezve).

##### Kezdeti Hash Értékek

A kezdeti hash értékek meghatározása nagyon szigorúan szabályozott, és minden egyes specifikus SHA-2 algoritmus esetében előre definiáltak. Ezek az értékek a következők:

- **SHA-224:** 8 db 32 bites érték.
- **SHA-256:** 8 db 32 bites érték.
- **SHA-384:** 8 db 64 bites érték.
- **SHA-512:** 8 db 64 bites érték.

##### Üzenet ütemezés (Message Schedule)

A bemeneti blokkokat fel kell darabolni kisebb darabokra, amit ütemezésnek (message schedule) hívunk. Mintavétel blokkhoz például egyetlen 512-bites blokkot 16 db 32-bites szóra darabolunk fel.

##### Kompressziós Függvény

A SHA-2 algoritmusok központi eleme a kompressziós függvény, amely iteratívan működik a hashelt blokkokon. Az algoritmusok előre meghatározott számú iterációt (gyorsítást) használnak, ami függ az algorítmus bitméretétől.

A kompressziós függvény:

\[ \text{CH}(x, y, z) = (x \, \& \, y) \, \oplus \, (\neg x \, \& \, z) \]

\[ \text{Maj}(x, y, z) = (x \, \& \, y) \, \oplus \, (x \, \& \, z) \, \oplus \, (y \, \& \, z) \]

\[ \text{Σ}_0(x) = \text{ROTR}^2(x) \, \oplus \, \text{ROTR}^{13}(x) \, \oplus \, \text{ROTR}^{22}(x) \]

\[ \text{Σ}_1(x) = \text{ROTR}^6(x) \, \oplus \, \text{ROTR}^{11}(x) \, \oplus \, \text{ROTR}^{25}(x) \]

\[ \sigma_0(x) = \text{ROTR}^7(x) \, \oplus \, \text{ROTR}^{18}(x) \, \oplus \, \text{SHFR}^3(x) \]

\[ \sigma_1(x) = \text{ROTR}^{17}(x) \, \oplus \, \text{ROTR}^{19}(x) \, \oplus \, \text{SHFR}^{10}(x) \]

A ROTR és SHFR a bit-forgatások és logikai műveletek különféle kombinációit jelentik.

#### Biztonsági Szintek

##### Collisions

Az SHA-2 algoritmusok ütközés-ellenállóak, azaz az algoritmus egyedi hash gyógát generál a különböző adatokhoz. Az SHA-2 család változatai különböző bit hosszúságú kimeneteket produkálnak, és az ütközési ellenállás az adott bit hosszúság kétszeres biztonsági szintjét adja:

- SHA-256: 2^128-szoros
- SHA-384: 2^192-szoros
- SHA-512: 2^256-szoros

##### Preimage Resistance

A SHA-2 algoritmusok preimage-ellenállók, azaz nehéz találni egy olyan bemenetet, amely adott hash értéket produkál.

##### Second Preimage Resistance

Második preimage-ellenállás azt jelenti, hogy az adott hash értékhez nincs egy második különböző bemenet, ami ugyanezzel a hash értékkel generálható. SHA-256 esetén ez a biztonsági szint 2^128, SHA-384 esetén 2^192, és SHA-512 esetén 2^256.

##### Attack Vectors

Amíg SHA-2 algoritmusok jelenlegi ismeretek alapján kriptográfiailag biztonságosnak tekinthetők, van néhány lehetséges támadási vektor, mint például:
- **Collision Attack**
- **Birthday Attack**
- **Length Extension Attack**

#### Példa

Ha példakódot írunk, C++ nyelven a SHA-256 algoritmushoz, azt az OpenSSL könyvtárral megtehetjük az alábbiak szerint:

```cpp
#include <openssl/sha.h>
#include <iostream>
#include <iomanip>
#include <cstring>

int main() {
    const char* str = "Hello, World!";
    unsigned char hash[SHA256_DIGEST_LENGTH];
    
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, str, strlen(str));
    SHA256_Final(hash, &sha256);
    
    std::cout << "SHA-256 of \"Hello, World!\" is: ";
    for(int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    std::cout << std::endl;
    
    return 0;
}
```

Ez a kódrészlet bemutatja az SHA-256 alkalmazását egy string "Hello, World!" hash értékének generálására. Az OpenSSL C++ könyvtár használatával egyszerűsítjük magát a hash folyamatot.

Az SHA-2 család tagjai napjaink legbiztonságosabb hash algoritmusai közé tartoznak. Tekintettel a hosszú távú biztonsági fenyegetésekre, az SHA-2 család robosztussága és integritása biztosítja, hogy továbbra is széles körben használják a modern számítástechnikában és digitális adatvédelemben.

### SHA-3 (Keccak)
Az SHA-3 (Secure Hash Algorithm 3), amelyet a Keccak algoritmusként is ismerünk, a Secure Hash Algorithm család legújabb tagja, amelyet az NIST (National Institute of Standards and Technology) 2015-ben fogadott el szabványként. Az SHA-3 új, innovatív megközelítést kínál a hash függvények területén, különösen a biztonság és rugalmasság tekintetében. Ellentétben elődeivel, mint például az SHA-1 és az SHA-2, az SHA-3 nem a Merkle-Damgård konstrukcióra épül, hanem a szivacsépítési módszert alkalmazza, amely lehetővé teszi változatos alkalmazások széles körű támogatását. Ebben a fejezetben megvizsgáljuk az SHA-3 működési mechanizmusát, a szivacsépítési módszer előnyeit, valamint azt, hogyan alkalmazható különböző iparágakban és technológiai megoldásokban az adatintegritás és a biztonság érdekében.

#### A Keccak működése

**Belső Szerkezet és Állapot:**

A Keccak egy úgynevezett szivacs funkciót alkalmaz, amely váltakozva abszorbeál és szivárog (squeeze) adatokat. A Keccak belső állapota egy háromdimenziós szerkezet, amely bitlogikával van feltöltve.

Az állapotméret az általános 1600 bit, amely 5x5 blokkokra osztható. Ezek a blokkok 64 bit szélességűek, így a teljes állapot 5x5x64 bittel rendelkezik.

A Keccak általános paraméterei:
* b: az állapot teljes mérete bitben
* r: az abszorbeálandó blokkok mérete bitben
* c: a kapacitás bitben, ami az állapot azon része, amely nem közvetlenül befolyásolt az inputból

A Keccak állapot mérete tehát: b = r + c.

**Folyamat:**

1. **Padding:**
   Az adatot először párnázással látják el (padding). A hash függvények esetében fontos, hogy a bemeneti blokk mérete mindig illeszkedjen az algoritmus által megkövetelt mérethez. Keccak esetében az MD-motivált x*1 padding szisztémát használják, amely biztosítja, hogy a bemeneti adat mindig megfelelő hosszúságú legyen.

2. **Abszorpciónés:**
   Az adattömb-abszorpció során a bemeneti adat blokkonként feldolgozásra kerül úgy, hogy XOR-ral (exclusive OR) hozzáadódik az aktuális állapothoz a puffer első r bitjénél. Utána a fázi transzformációt (F) alkalmazzák az állapot újraalkotására.

3. **Hash Kisugárzása (Output squeezing):**
   Miután az összes blokk feldolgozásra került, a szivacs addig szivárog adatot a keletkező hash kódhoz, amíg el nem éri a kívánt hash érték hosszat. Az F transzformáció ismételten alkalmazott a további blokkok feldolgozásához, ha szükséges.

**A Keccak f Permutáció:**

A Keccak f permutáció (f-function) a legkritikusabb része az algoritmusnak. Az F transzformáció több iterációt tartalmaz (általában 24 iteráció), és különböző bitműveleteken alapul (forgatás, XOR) a belső állapot bitjein. A Keccak F permutáció az alábbi lépésekből áll:

1. **Theta:** A bitállapotot minden egyes bitoszlop esetén XOR komplex művelet alkalmazásával módosítja.
2. **Rho (rotáció):** Bit elforgatásokat végez a szavakon (önálló blokkok).
3. **Pi (permute):** Az állapot elemeit átrendezi egy prediktív minta alapján.
4. **Chi (XOR és NOT):** Bitműveleteket alkalmaz, amelyek XOR és NON paraméterek keverékéből állnak.
5. **Iota:** Egyetlen szó bővítésével végzett XOR komplex művelet, amely relációs bővítményeket alkalmaz.

Ezek a tranzakciók közösen dolgoznak, hogy az állapot változás alapján végül egy igen összetett és keverő eljárást hozzanak létre.

#### SHA-3 Alkalmazások

SHA-3 számos alkalmazási területen használatos:

1. **Kriptográfiai protokollok:**
   Egy kulcsfontosságú komponens különböző kriptográfiai protokollokban és titkosítási eljárásokban, például a digitális aláírásoknál, kulcs deriválásnál, és tanúsítvány-rendszerekben.

2. **Adatok integritásának ellenőrzése:**
   Hash függvényként használata lehetővé teszi az adatok gyors ellenőrzését, többek között a fájlok hitelességének és integritásának állandó ellenőrzéséhez különböző alkalmazásokban, a szoftver-disztribúciótól kezdve az adatmentési folyamatokig.

3. **Konfliktus-rezisztencia:**
   SHA-3 viszonylag ellenáll a jelenlegi preimage és collision támadásokkal szemben, ami különösen fontossá teszi a biztonságos rendszerek kialakításában, és ipari szabványosításában, ahol az adat-egyértelműség alapvető jelentőségű.

4. **Digitális aláírás:**
   Az integrált hash függvények nélkülözhetetlen elemei a digitális aláírás rendszereknek, beleértve a dokumentumok digitális hitelesítésére szolgáló technológiákat, a tranzakció nyomkövetési biztonságrendszereket, és az e-aláírásokat.

SHA-3 fejlődése és standardizálása komoly lépéseket tett előre a kriptográfiai hash függvények bevezetésében, biztosítva az adatbiztonságot egyre nagyobb mértékben fejlődő digitális világunkban. Az elemzés során bemutatott elvek alapvetőek a tudományos igényességet igénylő informatikai rendszerek biztosításához és fenntartásához.
