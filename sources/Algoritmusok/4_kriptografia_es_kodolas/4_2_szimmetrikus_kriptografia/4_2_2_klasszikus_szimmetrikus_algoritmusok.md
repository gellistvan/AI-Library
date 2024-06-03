\newpage

## 2.2. Klasszikus szimmetrikus algoritmusok

### DES (Data Encryption Standard)

A szimmetrikus kulcsú kriptográfia területén az egyik legjelentősebb történelmi mérföldkő a DES (Data Encryption Standard), egy olyan titkosítási algoritmus, amely hosszú ideig az ipari szabványok gerincét képezte. Az 1970-es évek közepén bevezetett DES a titkosítás világában forradalmi újítással szolgált, egyaránt felkeltve a kormányzati szervezetek és a magánvállalatok figyelmét. A 2.2.1. fejezet célja, hogy részleteiben ismertesse a DES algoritmus felépítését és működését, miközben rámutat annak szerkezeti elemeire és belső logikájára. Ezen túlmenően, a fejezet külön kitér a DES már ismert gyengeségeire, valamint azokra a támadási módszerekre, amelyek az évek folyamán sikeresen feltörték vagy megkerülték az általa nyújtott biztonsági intézkedéseket. Ezek az elemzések nem pusztán történelmi érdekességgel bírnak, hanem fontos tanulságokat is rejtenek a modern kriptográfiai rendszerek tervezése és értékelése szempontjából.

#### DES felépítése és működése

Az 1970-es évek elején az IBM cég által kifejlesztett Data Encryption Standard (DES) a mai napig az egyik legismertebb szimmetrikus kulcsú kriptográfiai algoritmus, amelyet az Amerikai Szabványügyi Hivatal (NIST) fogadott el hivatalos titkosítási szabványként 1977-ben. Azóta DES-t széleskörűen alkalmazták banki rendszerekben, pénzügyi szolgáltatásokban és egyéb biztonságérzékeny alkalmazásokban. Bár ma már bizonyos gyengeségei miatt elavultnak tekinthető, történeti szempontból és az alapvető elvek megértése érdekében fontos megismerni felépítését és működését.

##### A DES alapfogalmai

DES egy 64-bites blokkos titkosítási algoritmus, amely 56-bites kulcsot használ. A titkosítás és dekódolás folyamata 16 körből (round) áll, amelyeken belül adattranszformációk történnek különböző részoperációk révén.

A főbb komponensek a következők:

- **Blokkméret**: 64 bit
- **Kulcsméret**: 56 bit (a teljes 64 bitből 8 bit ellenőrző bit, paritásellenőrzéshez)
- **Körök száma**: 16

##### DES alapvető lépései

A DES kódolási eljárás különböző matematikai és logikai műveletek révén byteok összekeverését, eltolását és más módosítását jelenti. Az egyes lépések a következők:

1. **Elsődleges Permutáció (Initial Permutation, IP)**: Az bemeneti 64-bites blokk egy rögzített permutáció alapján kerül átrendezésre.
2. **Feistel struktúra**: Ez az iteratív titkosítási séma DES operációs szívét képezi. A 64-bites blokk két 32-bites félblockra oszlik (bal és jobb rész), amelyek számára minden körben külön szerződéseket hajtanak végre.
3. **Funkció (F-függvény)**: Minden körben bonyolult transzformációt hajtanak végre az adatblokk jobb felén, melyek a következők:
    - E-bit bővítés (Expansion): A 32 bites jobb blokkot 48 bitesre bővítjük egy rögzített bővítési permutációval.
    - Körkulcs hozzáadása (Key Mixing): A 48 bites bővített blokkot XOR-ral hozzáadjuk a jelenlegi körkulcshoz.
    - S-doboz csere (Substitution): Az eredményül kapott 48-but hosszúságú blokkot 8 db 6-bites szegmensre bontják, és mindegyike végigmegy egy S-box substitúciós táblán, ami 32 bitre konvertálja azt.
    - P-permutáció (Permutation): Az S-boxból kialakuló eredményt egy P táblázattal elrendezi.
4. **Következő permutációs lépés (Switch)**:  A jobb és bal oldali blokkok szerepet cserélnek minden kör végén.
5. **Végső permutáció (Final Permutation, IP-1)**: Az utolsó kör után a két rész újraegyesül, és az adatblokk egy végső permutáción megy keresztül.

##### Keresztüljárási folyamat

A DES kódolási folyamatának mélyebb megértése érdekében vizsgáljuk meg az egyes lépéseket részletesebben:

1. **Elsődleges Permutáció (IP)**:
   ```
   Az IP egy fix permutációs táblázaton alapszik, amelyet a bemeneti 64-bites blokkra alkalmazva, újra rendezi a biteket.
   ```
2. **Újabb osztályozás és körkívánalmak**:
   ```
   Minden körben a jelenlegi blokkot két félre osztja, majd a Feistel struktúra elven alapuló következő transzformációkat végzi el.
   ```
3. **E-bit kiterjesztése**:
   ```
   A jobb 32-bites blokkot egy fix E-bit Expansion táblazattal bővíti 48 bitsre. Ezáltal a blokk olyan struktúrában fog létrejönni, amely bizonyos biteket megismétel.
   ```
4. **Körkulcs szorosítása**:
   ```
   A 48 bites bővített blokkal egy XOR operációt alkalmazunk az aktuális kör 48 bit hosszú kulcsával.
   ```
5. **S-bélyegzők alkalmazása**:
   ```
   A 48 bites eredményt 8 db 6-bites segmentre osztjuk és az S-bélyegzők segítségével 32 bitre konvertáljuk. Az S-bélyegzők szubstituciós táblázatok, amelyek nem-lineáris transzformációkat hajtanak végre.
   ```
6. **P-dobozos permutáció**:
   ```
   A 32-bites S-bélyegzők kimenete egy másik fix P-dobozos permutáción esik át, amely előállítja a szükséges keverési/titkosítási eredményeket.
   ```
7. **Switch és iteráció további lépései**:
   ```
   Minden körnek ezt az elvét alkalmazva, az egyes körök végére a bal és jobb blokkok szerepet cserélnek.
   ```

##### DES kulcsgenerálás

A DES 16 körének megvalósításához 16 körkulcsra van szükség, amelyek generálása egy speciális szubkulctábla (PC-1) és shift operációs technika révén történik:

- **Permutált választás (PC-1)**: Az eredeti 64-bites kulcsot (amely 56 bit használható) átrendezzük egy PC-1 permutáció alapján.
- **Körkulcs generálás**: Az átrendezett kulcsot két részre vágjuk (C és D), minden körben shift operációk révén eltoljuk őket, majd PC-2 permutációval kivonjuk a kívánt körkulcsot.

```cpp
#include <iostream>
#include <bitset>

// Példa C++ kódrészlet a permutációk demonstrálására
void permutate(std::bitset<64>& block, const int* table, int n) {
    std::bitset<64> result;
    for (int i = 0; i < n; i++) {
        result[n - 1 - i] = block[64 - table[i]];
    }
    block = result;
}

int main() {
    std::bitset<64> data(0x123456789ABCDEF0); // Példa adat
    const int IP[64] = { /*... itt helyezzük el az IP táblát ...*/};
    permutate(data, IP, 64);
    std::cout << "Permutated Data: " << data << std::endl;
    return 0;
}
```

Ez az egyszerű példa bemutatja a permutáció alapelveit a DES-ben, bár nem terjed ki az összes részletre. A konkrét DES megvalósítás ennél sokkal összetettebb, és sok további funkcionalitást igényel, mint például a Feistel-függvények és kulcs generálás.

#### Gyengeségek és támadási módszerek

A DES (Data Encryption Standard) egy, az NBS (National Bureau of Standards) által 1977-ben szövetségi szabvánnyá kinevezett szimmetrikus kulcsú titkosítási algoritmus, amely számos területen széles körben használt. Azonban a 21. században, különösen az internetes technológiák fejlődésével, a DES számos gyengesége fokozatosan nyilvánvalóvá vált, ami komoly biztonsági kockázatot jelentett. E fejezet célja, hogy részletesen bemutassa a DES gyengeségeit és a főbb támadási módszereket.

##### 1. Kulcshossz és Brute-Force támadások

Az egyik legjelentősebb gyengeség a DES esetében a kulcshossz. A DES 56 bites kulcsot használ, amely egykor elegendőnek és biztonságosnak számított, azonban a számítási teljesítmény növekedésével a brute-force (nyers erő használatával történő) támadások egyre hatékonyabbá váltak.

A brute-force támadás lényege, hogy a támadó az összes lehetséges kulcsot végigpróbálja, amíg megtalálja a helyeset. Mivel a DES kulcshossza csak 56 bit, összesen \(2^{56}\), azaz kb. 72 billió lehetséges kulcs van. Egy modern számítógép vagy speciálisan ehhez a célhoz tervezett eszköz (pl. FPGA) képes ezen kulcsok átkutatására ésszerű időn belül. Például 1998-ban az Electronic Frontier Foundation (EFF) épített egy olyan eszközt, amely egy 56 bites DES kulcsot kevesebb mint 24 óra alatt tört fel.

##### 2. Lineáris és Differenciális Kriptanalízis

A DES ellen hatékony módszerek közé tartozik a lineáris és a differenciális kriptanalízis is.

###### Lineáris Kriptanalízis

A lineáris kriptanalízis egy támadási módszer, amelyet Matsui 1993-ban fejlesztett ki. A módszer az iménti statisztikai elemzést használja arra, hogy a DES belső struktúrájában rejlő statisztikai anomáliákat kihasználja.

A lineáris kriptanalízis alapgondolata, hogy lineáris összefüggéseket keresünk a bemeneti és a kimeneti bitpárok között. E módszerrel a támadó képes kiszámítani bizonyos kulcsképlet valószínűségét, és ezzel fokozatosan szűkíti a lehetséges kulcsok körét.

###### Differenciális Kriptanalízis

A differenciális kriptanalízis egy másik hatékony támadási módszer, amelyet Biham és Shamir dolgozott ki az 1990-es évek elején. Ez a módszer az S-boxok (helyettesítési dobozok) outputjának különbségeire épít, melyeket különböző bemenetekkel táplálnak meg.

A módszerhez két bemeneti kódolás közötti különbségre van szükség, és az output különbségeket tanulmányozza. Ezzel a támadási módszerrel a DES 16-körös változata ellen akár 2^47 ismert szöveget is felhasználhatunk a kulcs megszerzéséhez.

##### 3. Meredek növekedés a Számítógépes Kapacitásban

A hardver fejlődése és a költségek csökkenése további problémákat jelent a DES számára. A számítógépek és speciális hardverek költsége jelentősen csökkent az 1990-es évek óta. Ennek eredményeképpen egyre több támadó fér hozzá olyan számítási kapacitáshoz, amely lehetővé teszi a brute-force támadásokat.

##### 4. Kulcsmenedzsment és Komplexitás

A szimmetrikus kulcsú kriptográfia, különösen a DES esetében, komoly kihívást jelent a kulcsmenedzsment terén. Mindkét kommunikáló félnek ugyanazt a titkos kulcsot kell birtokolnia és biztonságban kell tartania. Egy titkos kulcs kompromittálódása az egész kommunikációt veszélyeztetheti.

##### 5. Megszakítás és Ismétlések

A DES működési módjai, mint például az ECB (Electronic Codebook), szintén gyengeségekkel rendelkeznek. Az ECB mód egyszerűsége miatt ismétlődő blokkok azonos titkosított blokkokat eredményeznek, így a hosszú, ismétlődő minták a titkosított szövegben is jelen lesznek.

Egy másik probléma az aktív támadások, ahol a támadó magukat a titkosítást végző üzeneteket módosítja. Ismétlődő adatsorok a DES esetében lehetővé teszik az ilyen támadásokat.

##### 6. Preimage és Collision Támadások

A preimage és a collision támadások szintén fontos tényezők a DES esetében. Bár viszonylag nehéz sikeres collision támadásokat végrehajtani given egy szimmetrikus kulcsot, a támadó technikák fejlődésével ezek a támadások lehetségesek, különösen ha a hash-függvények gyengeségei is ki vannak használva.

##### Kód mint példákkal

Itt egy egyszerű példakód C++ nyelven, amely bemutatja a brute-force támadási módszer alapelveit DES-en:

```cpp
#include <iostream>
#include <cstring>
#include <openssl/des.h>

void brute_force_attack(const unsigned char* ciphertext, const unsigned char* plaintext) {
    DES_cblock key;
    DES_key_schedule schedule;
    unsigned char buffer[8];

    for (uint64_t i = 0; i < (1LL << 56); ++i) {
        std::memcpy(key, &i, 7);
        DES_set_odd_parity(&key);
        DES_set_key_checked(&key, &schedule);
        
        DES_ecb_encrypt((DES_cblock*) ciphertext, (DES_cblock*) buffer, &schedule, DES_DECRYPT);

        if (!std::memcmp(buffer, plaintext, 8)) {
            std::cout << "Key found: " << std::hex << i << std::endl;
            return;
        }
    }

    std::cout << "Key not found." << std::endl;
}

int main() {
    unsigned char plaintext[8] = "example";
    unsigned char ciphertext[8]; // should be set with the value of the ciphertext you want to decrypt

    brute_force_attack(ciphertext, plaintext);
    return 0;
}
```

Ezen a példán keresztül bemutatható, hogy egy brute-force támadás, amely az összes lehetséges kulcsot kipróbálja, reális lehetőséggé válik a modern számítástechnikai kapacitással szemben.

##### Gyenge Kulcsok

A DES-nek néhány kulcstípusa különösen gyenge. Ezek a kulcsok bizonyos kulcsok esetén az, hogy az S-boxokba történő belépési és kilépési pontok ismétlődni fognak, így ezek a kulcsok különösen sebezhetőek lesznek a differenciális kriptanalízis ellen.

### 2.2.2. 3DES (Triple DES)

A 3DES (Triple Data Encryption Standard) egy szimmetrikus kulcsú kriptográfiai algoritmus, amely a DES (Data Encryption Standard) továbbfejlesztéseként jött létre. A DES, amelyet az 1970-es évek végén szabványosítottak, az idő múlásával sebezhetővé vált a növekvő számítási teljesítmény és fejlettebb kriptanalitikai módszerek miatt. Ennek megfelelően dolgozták ki a 3DES algoritmust, amely háromszoros DES titkosítással igyekszik biztosítani a nagyobb biztonságot anélkül, hogy teljesen új alapokra helyezné a kiváltó algoritmust. A 3DES által alkalmazott többszörös rekurzió célja, hogy a titkosítási folyamatot ne csak észrevehetően erősebbé, hanem a visszamenőleges kompatibilitás szempontjából is megfelelővé tegye. Ez a részleteiben is különleges titkosítási módszer számos alkalmazásban nyújt megbízható védelmet, így kerül bemutatásra mind a működési elvei, mind a gyakorlati felhasználása szempontjából.

3DES három kulccsal dolgozik: \(K_1\), \(K_2\), és \(K_3\). Az algoritmus három fő lépésből áll:
1. A nyílt szöveg titkosítása \(K_1\) kulccsal.
2. Az eredményes kódolt szöveg vissza-titkosítása \(K_2\) kulccsal.
3. Az eredményes nyílt szöveg újra titkosítása \(K_3\) kulccsal.

Matematikai formában az elkódolás folyamat így írható le:
\[C = DES_{K_3}(DES^{-1}_{K_2}(DES_{K_1}(P)))\]
ahol \(P\) a nyílt szöveg, \(C\) a kódolt szöveg, és \(DES\) és \(DES^{-1}\) a DES titkosítási és visszafejtési funkciók.

##### Kulcs Management

3DES három lehetséges kulcs opciót támogat:
1. **K1 ≠ K2 ≠ K3**: Teljes kulcs hossz (168-bit) használata, maximális biztonság elérése.
2. **K1 = K3 ≠ K2**: Tényleges 112-bit kulcs hossz használata, amit a legtöbb alkalmazás támogat.
3. **K1 = K2 = K3**: Egyezik az egyszeri DES titkosítással (56-bit), visszafele kompatibilitással.

Az ajánlott gyakorlat az, hogy mindig a maximális kulcs hossz opciót (három különböző kulcsfájl) használjuk, mivel ez a legbiztonságosabb.

#### Titkosítási Módszer Lépcsők

1. **Előkészítés**:
    - A plain text blokkokat 64-bit hosszú darabokra osztják (tipikusan).
    - Kiválasztják a három 56-bit kulcskombinációt (K1, K2, és K3).

2. **Első DES alkalmazás (titkosítás)**:
    - K1 kulccsal titkosítják.

3. **Második DES alkalmazás (visszafejtés)**:
    - A keletkezett kódolt szöveget K2 kulccsal vissza-titkosítják.

4. **Harmadik DES alkalmazás (újra titkosítás)**:
    - A kapott nyílt szöveget K3 kulccsal újra titkosítják.

#### Algoritmus Implementációja C++ nyelven

Az alábbi példakód bemutatja, hogyan lehet a 3DES titkosítást megvalósítani C++ nyelven a jól ismert OpenSSL könyvtár használatával.

```cpp
#include <openssl/des.h>
#include <cstring>
#include <iostream>

// Helper function to convert a string to a 64-bit DES cblock
void stringToCBlock(const std::string &str, DES_cblock &cblock) {
    memset(cblock, 0, sizeof(cblock));
    std::memcpy(cblock, str.c_str(), str.size() > sizeof(cblock) ? sizeof(cblock) : str.size());
}

int main() {
    // Secret keys: should be 8 bytes each for DES
    std::string key1 = "12345678";
    std::string key2 = "abcdefgh";
    std::string key3 = "abcdefgh";

    // Data to be encrypted: should be multiple of 8 bytes for DES
    std::string data = "Hello 1234Hello 1234";

    DES_cblock k1, k2, k3;
    stringToCBlock(key1, k1);
    stringToCBlock(key2, k2);
    stringToCBlock(key3, k3);

    // Initialize DES keys
    DES_key_schedule ks1, ks2, ks3;
    DES_set_key_checked(&k1, &ks1);
    DES_set_key_checked(&k2, &ks2);
    DES_set_key_checked(&k3, &ks3);

    // Intermediate data
    unsigned char inter1[8], inter2[8], final[8];

    // Initialize the DES encryption/decryption process
    for (size_t i = 0; i < data.size(); i += 8) {
        DES_ecb_encrypt((DES_cblock *)(data.c_str() + i), (DES_cblock *)inter1, &ks1, DES_ENCRYPT);
        DES_ecb_encrypt((DES_cblock *)inter1, (DES_cblock *)inter2, &ks2, DES_DECRYPT);
        DES_ecb_encrypt((DES_cblock *)inter2, (DES_cblock *)final, &ks3, DES_ENCRYPT);
        std::cout.write((const char*)final, 8);
    }

    std::cout << std::endl;
    return 0;
}
```

#### Alkalmazások

3DES széles körben használt, különösen olyan rendszerekben, ahol a visszafele kompatibilitás kritikus és nem kívánják átdolgozni a meglévő infrastruktúrát. Néhány gyakran előforduló alkalmazásai a következők:

1. **Pénzügyi Szolgáltatások**:
    - ATMs (Automata Teller Machines) titkosított kommunikációja hagyományosan 3DES alapján történik a tranzakciós adatok biztonsága érdekében.

2. **Biztonságos e-mail**:
    - Biztonsági protokollok, mint az S/MIME, felhasználhatják a 3DES-t az e-mail és kapcsolódó tartalmak titkosítására.

3. **Adatbázisok és Érzékeny Adatok Titkosítása**:
    - Boszorkányos adathalmazok titkosítása a 3DES által a helyi tárolásban, biztosítva az adatok védelmét lopás vagy illetéktelen hozzáférés ellen.

#### Hátrányok és Limitációk

Annak ellenére, hogy a 3DES erősebb, mint a sima DES, még mindig rendelkezik néhány jelentős korlátozással:
- **Működési sebesség**: 3DES háromszoros DES alkalmazást jelent, így sokkal lassúbb, mint más modern szimmetrikus algoritmusok, mint például az AES.
- **Kulcsméret**: Habár nagyobb kulcselfogadási opciókat kínál, a kulcshossz (112-bit vagy 168-bit) még mindig nem hasonlítható a 256-bit AES kulcsokhoz, amelyek manapság gyakoriak.





