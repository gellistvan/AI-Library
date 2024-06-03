\newpage

# IV. Kriptográfiai és kódolási algoritmusok


# 1. Bevezetés a kriptográfiába

A modern információs társadalomban a biztonságos kommunikáció és az adatok védelme minden korábbinál fontosabb szerepet játszik. A kriptográfia, a rejtjelezés tudománya, a matematikai és informatikai módszereket ötvözve kínál megoldásokat arra, hogy megóvjuk az érzékeny információkat az illetéktelen hozzáféréstől. Ebben a fejezetben áttekintést nyújtunk a kriptográfia alapjairól, ismertetve annak történetét, alapelemeit és különböző módszereit, amelyek révén a titkosítás lehetővé vált a kommunikációs hálózatok világában. Bemutatjuk az alapvető fogalmakat, mint a szimmetrikus és aszimmetrikus kulcskódolás, hash függvények és digitális aláírások, melyek mindegyike kulcsfontosságú a digitális biztonság szempontjából. Célunk, hogy az olvasó átfogó képet kapjon a kriptográfiai technikákról és azok alkalmazásairól, elősegítve ezzel a biztonságos adatkezelési módszerek megértését és használatát.

## 1. Kriptográfia története
A kriptográfia, az információk titkosításának és védelmének művészete és tudománya, évezredekre nyúlik vissza a történelemben. Az emberek már az ókortól kezdve alkalmaztak különféle titkosítási módszereket, hogy megvédjék üzeneteiket a kíváncsi szemek elől, katonai, diplomáciai és kereskedelmi célokra egyaránt. E fejezet során megismerhetjük mind az ősi titkosítási módszereket, amelyek megalapozták a kriptográfia fejlődését, mind a modern kriptográfia kialakulását, amely a 20. századi technológiai innovációkkal és matematikai áttörésekkel forradalmasította az információbiztonságot. A kriptográfia, amely egykor csupán rejtjelző eljárásokat jelentett, mára komplex algoritmusok és protokollok széles körét öleli fel, amelyek nélkülözhetetlenek a digitális kor adatvédelmében.

### Ősi titkosítási módszerek

A kriptográfia története több évezredre nyúlik vissza, és az emberiség már az ókorban is szükségét érezte annak, hogy bizalmas kommunikációját megvédje az illetéktelenektől. Az ősi titkosítási módszerek meglehetősen kreatív és technológiailag fejlett megoldásokat tükröznek annak ellenére, hogy az akkori technikai infrastruktúra korlátozott volt. Ebben a fejezetben néhány jelentős ókori titkosítási módszert fogunk áttekinteni, köztük a szubstitúciós és transzpozíciós technikákat, valamint a cézáros és skítai titkosítást.

#### Szubstitúciós titkosítás

A szubstitúciós titkosítás lényege, hogy az üzenet minden egyes elemét (karakterét) egy másik, előre meghatározott karakterrel helyettesítik. Ez a módszer az egyik legrégebbi és legelterjedtebb titkosítási technika. A legismertebb szubstitúciós titkosítási módszerek közé tartozik a Cézár-kód és az Atbash titkosító.

##### Cézár-kód

Julius Caesar állítólag a nevét viselő titkosítást használta fontos katonai információk továbbítására. A Cézár-kód egy eltolásos szubstitúciós algoritmus, amelyben minden betűt egy előre meghatározott lépésértékkel tolunk el az ábécében. Például, egy 3-as lépésű eltolás esetén az 'A' betű 'D'-vé válik, a 'B' betű 'E'-vé, és így tovább.

A Cézár-kód C++ kóddal történő implementálása:

```cpp
#include <iostream>
#include <string>

std::string caesarCipher(const std::string &text, int shift) {
    std::string result = "";

    for (char ch: text) {
        if (isalpha(ch)) {
            char base = islower(ch) ? 'a' : 'A';
            ch = (ch - base + shift) % 26 + base;
        }
        result += ch;
    }
    return result;
}

int main() {
    std::string text = "HELLO WORLD";
    int shift = 3;
    std::string encrypted = caesarCipher(text, shift);
    std::cout << "Encrypted: " << encrypted << std::endl;
    return 0;
}
```

Ebben a programban az `isalpha` függvény ellenőrzi, hogy a karakter betű-e, és az `islower` függvény vizsgálja, hogy kisbetűs-e. A Caesar-féle eltolást a közvetlen matematikai műveletek végzik el.

##### Atbash titkosító

Az Atbash titkosító egy speciális szubstitúciós titkosítás, amelyben az ábécé minden betűjét annak "tükörképes" betűjével helyettesítjük (pl. 'A' -> 'Z', 'B' -> 'Y'). Ez a módszer a héber ábécé alapján alakult ki, de alkalmazható a latin ábécére is. Egyszerűsége ellenére az Atbash még mindig érdekes megközelítés a szubstitúció világában.

#### Transzpozíciós titkosítás

A transzpozíciós titkosítás során az üzenet karaktereit egy meghatározott rendszer szerint átrendezzük. Ebben a módszerben az eredeti karakterek megmaradnak, de a helyzetük megváltozik. Az egyik legismertebb transzpozíciós titkosítási forma a Szkíta titkosítás, más néven Scytale.

##### Szkíta titkosítás (Scytale)

Az ókori Spártaiak használták a Szkíta titkosítást katonai üzeneteik védelme érdekében. Ez a módszer egy henger alakú eszközt használ, amelyre a bőrcsíkot vagy papírtekercset spirális formában tekerik fel. Amikor a tekercset megfelelő hengerre tekerik, az üzenet olvashatóvá válik. Ha a tengelyre nem megfelelő hengereszközt használnak, az írás egy káoszos karakterlánccá válik.

A Szkíta titkosítás elvét C++ kóddal mutatjuk be:

```cpp
#include <iostream>
#include <string>
#include <cmath>

std::string scytaleEncrypt(const std::string &plainText, int diameter) {
    int length = plainText.length();
    int columns = ceil(static_cast<double>(length) / diameter);
    std::string cipherText(length, ' ');

    for (int i = 0, k = 0; i < columns; ++i) {
        for (int j = i; j < length; j += columns) {
            cipherText[k++] = plainText[j];
        }
    }
    return cipherText;
}

int main() {
    std::string plainText = "WEAREDISCOVEREDFLEEATONCE";
    int diameter = 5;
    std::string encrypted = scytaleEncrypt(plainText, diameter);
    std::cout << "Encrypted: " << encrypted << std::endl;
    return 0;
}
```

Ebben az implementációban a `ceil` függvényt használjuk arra, hogy meghatározzuk a sorok számát a hengerre való tekeréshez.

#### Polübius négyzete

A Polübius négyzet egy ősi görög titkosítási módszer, amelyet Polübiosz, egy görög történetíró fejlesztett ki. Az ábécé betűit egy 5x5-ös négyzetbe rendezzük, és a betűket koordinátáikkal helyettesítjük. Mivel az angol ábécé 26 betűből áll, az 'I' és 'J' betűket egyetlen négyzetbe helyezzük.

Polübius négyzet (angol ábécé):

```
1 2 3 4 5
1 A B C D E
2 F G H I/J K
3 L M N O P
4 Q R S T U
5 V W X Y Z
```

Például a "HELLO" szó kódolása a Polübius négyzettel:

```
H -> 23
E -> 15
L -> 31
L -> 31
O -> 34
```

Az eredmény: "2315313134"

#### Monoalfabetikus és Polialfabetikus titkosítás

A monoalfabetikus titkosítás olyan szubstitúciós technika, amelyben minden betűt egy más betűvel helyettesítenek az ábécében. Az ilyen módszerek közé tartozik a híres szitakötős titkosítás (Atbash kód). Azonban ezek könnyen megfejthetők a gyakoriságelemzéssel.

A polialfabetikus titkosításban, amelyet először Leon Battista Alberti ajánlott az 15. században, több különböző ábécét használunk az üzenet kódolásához. Egy példa erre a Vigenère titkosító, amely árnyalatosabbá és biztonságosabbá tette a titkosítást, mivel az ábécé rendszeresen váltakozik.

#### Összegzés

Az ősi titkosítási módszerek a kriptográfia lenyűgöző példái, amelyek jól tükrözik az emberi találékonyságot az információk védelme terén. A szubstitúciós és transzpozíciós technikák használata megalapozta a modern kriptográfia alapjait, amely tovább fejlődött a matematika, számítástechnika és technológia fejlődésével.


### Modern kriptográfia kialakulása

A modern kriptográfia kialakulása egy hosszú és bonyolult folyamat eredménye, amely szorosan kapcsolódik a matematikai fejlesztésekhez, a számítógépek terjedéséhez és az információbiztonsági igények növekedéséhez. Annak ellenére, hogy a kriptográfia már évezredek óta létezik, a modern kriptográfia kialakulása a 20. század közepétől datálható, amikor az elméleti és alkalmazott matematika erőteljes fejlődése, valamint az informatika robbanásszerű fejlődése lehetővé tette új és komplex titkosítási módszerek kifejlesztését.

#### Matematikai Alapok

A modern kriptográfia egyik alapvető eleme a matematikai szigor és formalizmus. A klasszikus kriptográfiai módszerek, mint a Caesar-kód vagy a Vigenère-kód, viszonylag egyszerű algoritmusokat alkalmaztak, amelyek könnyen megfejthetők voltak a megfelelő technikákkal. A modern kriptográfia jelentős újítása az, hogy olyan bonyolult matematikai problémákra alapozta a titkosítást, amelyek megoldása gyakorlati időn belül nem lehetséges.

1. **Számelmélet**: Az egyik legfontosabb matematikai terület, amely a modern kriptográfia alapjául szolgál, a számelmélet. Az olyan problémák, mint a prímszámok faktorizálása vagy a diszkrét logaritmus problémája, tökéletesen alkalmasak titkosítási célokra, mivel ezek megoldása exponenciálisan növekvő időt igényel a bemenet méretének növekedésével.

2. **Algebra**: A csoportelmélet, gyűrűk és testek elmélete is alapvető fontosságú a modern kriptográfiában. Például az elliptikus görbe kriptográfia (ECC) az elliptikus algebrai görbéket használja és ezek csoportszerkezetén alapul.

3. **Valószínűség és statisztika**: A véletlenszerűen generált kulcsok és algoritmusok hatékonyságának elemzésére, valamint a támadások valószínűségének értékelésére is kiterjedt valószínűségi és statisztikai módszerekre van szükség.

#### Számítógépes Algoritmusok és Komplexitáselmélet

A modern kriptográfia másik kulcsfontosságú tényezője a számítógépek elterjedése és a komplexitáselmélet fejlődése. A számítógépek lehetővé tették nagy mennyiségű adat gyors feldolgozását, ami új kihívásokat és lehetőségeket jelentett a titkosítási algoritmusok számára.

1. **Klasszikus titkosítási algoritmusok**: Az első széles körben használt modern titkosítási algoritmus a Data Encryption Standard (DES) volt, amelyet a Nemzeti Szabványügyi és Technológiai Intézet (NIST) az 1970-es években fejlesztett ki. A DES 56 bites kulcsot használt, amely az idő elteltével sebezhetővé vált a nyers erővel történő támadásokkal szemben. Ennek köszönhetően 2001-ben az Advanced Encryption Standard (AES) váltotta fel, amely sokkal erősebb, akár 256 bites kulcsokat is támogat.

2. **Nyilvános kulcsú kriptográfia**: A nyilvános kulcsú kriptográfia (public-key cryptography) koncepciójának bevezetése forradalmi áttörést jelentett. Az 1970-es években Whitfield Diffie és Martin Hellman bemutatták a Diffie-Hellman kulcscserét, amely lehetővé tette két fél számára, hogy biztonságosan megosszák titkosított információikat anélkül, hogy fizikailag találkozniuk kellene. 1978-ban Ronald Rivest, Adi Shamir és Leonard Adleman kifejlesztették az RSA algoritmust, amely elsőként valósította meg a nyilvános kulcsú titkosítást, és amely a nagy prímszámok faktorizálásának matematikai problémáján alapul.

3. **Elliptikus görbe kriptográfia (ECC)**: Az ECC az 1980-as évek végén került kifejlesztésre, és a csoportalgebrai struktúrákat kihasználva magasabb biztonságot nyújt kisebb kulcsméret mellett, mint az RSA. Az ECC hatékonyabb kulcsméretet és gyorsabb titkosítási-dekódolási műveleteket kínál, ami különösen fontos az erőforráshiányos környezetekben, mint például mobil eszközök vagy beágyazott rendszerek.

#### Kriptográfiai Protokollok

A modern kriptográfia nemcsak az egyes titkosítási algoritmusokat foglalja magában, hanem a protokollok és rendszerek szintjén is fontos szerepet játszik. Ezek a protokollok biztosítják az adatok biztonságos cseréjét, hitelesítését és integritását az információs rendszerekben.

1. **Transport Layer Security (TLS)**: Az internetbiztonság egyik legfontosabb protokollja a TLS, amely a Secure Sockets Layer (SSL) utódja. A TLS biztonságos kommunikációt biztosít a hálózatok között, egyaránt használva szimmetrikus és aszimmetrikus titkosítást.

2. **Kerberos**: Ez egy hálózati hitelesítési protokoll, amely szimmetrikus kulcsokkal működik és a világ különböző hálózataiban széles körben használják adatszolgáltatások hitelesítésére.

3. **IPsec**: Az IPsec (Internet Protocol Security) egy biztonsági protokollcsomag, amely az IP rétegben működik, biztosítva a titkosított kommunikációt a hálózatokon keresztül.

#### Gyakorlati Alkalmazások és Jövőbeli Kihívások

A modern kriptográfia alkalmazásai sokrétűek és magukban foglalják a mindennapi élet számos területét. A biztonságos online tranzakciók, adatbiztonság, hitelesítés, valamint a kiberbűnözés elleni védelem mind-mind modern kriptográfiai algoritmusok nélkülözhetetlen felhasználási területei.

1. **Digitális aláírások**: Ezek jogi érvényűek és biztonságosabbak, mint a hagyományos kézi aláírások. Nyilvános kulcsú algoritmusokra épülnek, biztosítva az aláírás hitelességét.

2. **Blokklánc és kriptovaluták**: A modern kriptográfia alapvető szerepet játszik a blokklánc technológiában és a kriptovaluták, mint például a Bitcoin alapú működésben.

3. **Post-quantum kriptográfia**: Az egyik legnagyobb kihívás a jelenlegi kriptográfiai rendszerek számára a kvantumszámítógépek megjelenése, amelyek potenciálisan képesek lesznek feltörni a legtöbb jelenlegi kriptográfiai algoritmust. Kutatások folynak új, kvantumbiztos algoritmusok fejlesztésére, amelyek ellenállnak a kvantumszámítógépek által jelentett fenyegetésnek.

### Kód Példa: RSA Kulcspár Generálása (C++)

Bár a modern kriptográfia egy rendkívül komplex és matematikaigényes tudományterület, itt van egy egyszerű példa, hogy hogyan generáljunk egy RSA kulcspárt C++ nyelven.

```cpp
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

long long gcd(long long a, long long b) {
    if (b == 0) return a;
    return gcd(b, a % b);
}

long long modexp(long long base, long long exp, long long mod) {
    long long res = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp % 2 == 1) {
            res = (res * base) % mod;
        }
        exp = exp >> 1;
        base = (base * base) % mod;
    }
    return res;
}

int main() {
    srand(time(0));
    long long p = 61;  // first prime number
    long long q = 53;  // second prime number

    long long n = p * q;               // modulus
    long long phi = (p - 1) * (q - 1); // totient

    long long e;  // public key
    do {
        e = rand() % phi;
    } while (gcd(e, phi) != 1);

    long long d; // private key
    for (d = 1; (d * e) % phi != 1; d++);

    std::cout << "Public Key: (" << e << ", " << n << ")" << std::endl;
    std::cout << "Private Key: (" << d << ", " << n << ")" << std::endl;
    
    return 0;
}
```

Ez az egyszerű program bemutatja, hogyan lehet generálni egy RSA kulcspárt két kis prímszám, \( p \) és \( q \) felhasználásával. Ugyan ez a példa nem használja a valós életben tapasztalt nagyméretű prímszámokat és nem kezeli az összes biztonsági kérdést, ez jó kiindulópontot nyújt a modern RSA algoritmus megértéséhez.

### Összefoglalás

A modern kriptográfia kialakulása elválaszthatatlan összefonódásban van a matematikai felfedezésekkel és a számítógépek fejlődésével, ami lehetővé tette a sokkal bonyolultabb és sokkal biztonságosabb titkosítási módszerek kidolgozását. A folyamatos fenyegetések és a kiberbiztonság iránti növekvő igény együttesen inspirálják a kriptográfiai kutatásokat, amelyek biztosítják, hogy adatok biztonsága mindig egy lépéssel a támadók előtt járjon.



