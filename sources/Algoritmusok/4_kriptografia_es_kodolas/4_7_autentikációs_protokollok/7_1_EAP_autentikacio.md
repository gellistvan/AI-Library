\newpage

# 7. Autentikációs protokollok

Az autentikációs protokollok a kriptográfia és az információbiztonság kulcsfontosságú elemei, amelyek lehetővé teszik a felhasználók és eszközök megbízható azonosítását és hitelesítését a hálózati kommunikáció során. Ezek a protokollok biztosítják, hogy csak a jogosult felhasználók és eszközök férhessenek hozzá a hálózati erőforrásokhoz, miközben megakadályozzák az illetéktelen hozzáférést és támadásokat. 

## 7.1. EAP-alapú protokollok
A különböző EAP (Extensible Authentication Protocol) alapú megoldások széles skáláját kínálják az autentikációs folyamatokhoz, amelyek különböző biztonsági követelményeknek és alkalmazási területeknek felelnek meg. A következő alfejezetekben részletesen bemutatjuk az EAP protokoll különböző változatait és alkalmazásukat, mint például az EAP-TLS, EAP-TTLS, PEAP, LEAP, EAP-FAST, EAP-SIM, EAP-AKA, EAP-MD5 és EAP-PSK. Mindezek a protokollok sajátos előnyöket és biztonsági mechanizmusokat kínálnak, amelyek hozzájárulnak az átfogó hálózati biztonság megvalósításához.

#### EAP Áttekintése

Az EAP definiálása az IETF (Internet Engineering Task Force) RFC 3748 dokumentumában található meg, amely a protokoll alapjait és működését részletesen bemutatja. Az EAP egy hálózati autentikációs keretrendszer, amely lehetővé teszi különböző autentikációs mechanizmusok alkalmazását, mint például a jelszavak, egyszeri jelszókódok (OTP-k), digitális tanúsítványok és az erőforrás-határoló protokollok.

Az EAP fő jellemzői közé tartozik a következők:
- **Rugalmasság**: Több autentikációs metodika támogatása.
- **Bővíthetőség**: Az EAP keretrendszere megengedi új autentikációs módszerek hozzáadását anélkül, hogy alapvető változtatásokat kellene alkalmazni a protokollon.
- **Hordozhatóság**: Számos hálózati környezetben használható, mint például PPP és IEEE 802 hálózatok.

#### EAP Alapú Kommunikáció

Az EAP alapú autentikáció három fő szereplőt foglal magába:
1. **EAP Peer (Authentikációs kliens)**: Az a kliens eszköz vagy felhasználó, aki az autentikációs szolgáltatáshoz csatlakozni kíván.
2. **EAP Server (Authentikációs szerver)**: Az a szerver vagy rendszer, amely az autentikációs döntéseket hozza meg.
3. **Authenticator (Hitelesítő)**: Az a köztes, amely a peer-t és a szervert köti össze. Példa lehet erre egy hozzáférési pont vagy hálózati eszköz.

##### EAP Működési Folyamat

A kommunikációs folyamat az alábbi fő lépésekből áll:
1. **Kezdeményezés és azonosítás**: Az EAP authentikátor egy EAP-kezdőcsomagot küld a peer részére, hogy az autentikációs folyamat elinduljon.

2. **EAP-kérelmek és válaszok**: Az authentikátor és a peer több EAP üzenetet cserélnek egymással (EAP-kérelem és EAP-válasz csomagok formájában), hogy az autentikációs módszert meghatározzák és végrehajtsák.

3. **Sikeres vagy sikertelen authentikáció**: Az autentikációs folyamat végén az EAP szerver küld egy értesítést az autentikátor számára az autentikáció sikerességéről vagy sikertelenségéről.

Az EAP szerkezete egyszerű, mivel mindössze négy különböző csomagtípussal dolgozik:
- **EAP-Request (kérelem)**: Az authentikátor küldi a peer részére.
- **EAP-Response (válasz)**: A peer küldi az authentikátor részére válaszként.
- **EAP-Success (siker)**: Az authentikátor küldi a peer részére, ha az authentikáció sikeres volt.
- **EAP-Failure (sikertelenség)**: Az authentikátor küldi a peer részére, ha az authentikáció sikertelen volt.

##### Példa EAP Üzenet Formátumra

A tipikus EAP üzenet formátum az alábbi részekből áll:
- **Code**: 1 bájt. Meghatározza az üzenet típusát (Request, Response, Success, Failure).
- **Identifier**: 1 bájt. Azonosítja az EAP üzenetet, hogy az eredeti kérelmet és választ összepárosítsa.
- **Length**: 2 bájt. Az EAP csomag teljes hosszát jelzi.
- **Type**: 1 bájt, csak a Request és Response csomagokban található meg. Meghatározza a specifikus authentikációs módszert.

Példa EAP-Request/Response csomag struktúrája:
```cpp
struct EAP_Packet {
    uint8_t code;
    uint8_t id;
    uint16_t length;
    uint8_t type;
    uint8_t data[];
};
```

#### EAP Módszerek és Típusok

Az EAP keretrendszer számos autentikációs módszert támogat. Ezek közé tartoznak, de nem kizárólagosan:
- **EAP-MD5**: Egy egyszerű challange-response autentikációs módszer, de II.szintű biztonságot nyújt. Hasznossága csökkent a hálózati támadások könnyűségénél fogva.
- **EAP-TLS (Transport Layer Security)**: Egyik legbiztonságosabb módszer, digitális tanúsítványokat használ az autentikációhoz.
- **EAP-TTLS (Tunneled Transport Layer Security)**: Használ titkosított csatornat a szerverbizonyítvány alapján, aztán egy másik authentikációs módszer belül fut.
- **PEAP (Protected EAP)**: Az EAP-Microsoft által fejlesztett, TLS-alapú, amely védett autentikációs csatornát biztosít.

Van még sok további típus, mint például az EAP-AKA (Authentication and Key Agreement), EAP-FAST (Flexible Authentication via Secure Tunneling), stb. Az EAP keretrendszer további új autentikációs módszerekkel is bővíthető, amelyek specifikus szükségletekre szabhatóak.

#### Biztonsági Szempontok

Az EAP sajátossága, hogy sok függ az adott autentikációs módszertől a biztonság tekintetében. Fontos azonban figyelni az alábbi biztonsági szempontokra:
- **Man-in-the-middle támadások**: Óvatossággal kell kezelni egyszerűbb EAP módszereket, mert azok sebezhetnek lehetnek.
- **Hitelesítés titkosítása**: Az olyan módszerek, mint az EAP-TLS vagy EAP-TTLS megfelelő titkosítást biztosítanak az adatok védelmére.
- **Tárolási védelem**: A hitelesítési adatok és a digitális tanúsítványok megfelelő tárolása kulcsfontosságú azok kompromittálódásának elkerülése érdekében.
- **Ülésszöveg kimerítése**: Az EAP-Peer és Authenticator közötti kommunikáció megfelelő korrelációval szinkronban kell lennie az eredeti hitelesítés érdekében.

#### Következtetés

Az EAP (Extensible Authentication Protocol) egy sokoldalú és hatékony hálózati autentikációs keretrendszer, amely számos különböző autentikációs módszert támogat. Az EAP rugalmassága és bővíthetősége lehetővé teszi, hogy különböző hálózati környezetekben alkalmazzuk, beleértve a vezeték nélküli és vezetékes hálózatokat is. A különböző EAP-típusú autentikációk egyszerűségüket vagy biztonságukat figyelembe véve könnyen alkalmazhatóak az adott hálózati igényeknek megfelelően. Az EAP egy dinamikus autentikációs rendszer, amely folyamatos fejlesztést és alkalmazást biztosít a jövőbeni hálózati igényekhez.### 7.1.2. EAP-TLS (EAP-Transport Layer Security)

### 7.1.2. Az EAP-TLS (Extensible Authentication Protocol-Transport Layer Security)

Az EAP-TLS (Extensible Authentication Protocol-Transport Layer Security) egyike a legelterjedtebb és legbiztonságosabb EAP alapú protokolloknak, amelyeket az autentikáció és a hitelesítés területén alkalmaznak. Az EAP-TLS az IEEE 802 szabványcsalád része, és főként az IEEE 802.1X szabványon keresztül használják, amely eredetileg a vezeték nélküli hálózatok hitelesítésére szolgált. Az EAP-TLS alapjai az SSL/TLS (Secure Sockets Layer/Transport Layer Security) protokollra épülnek, amelyet széleskörűen használnak a webes biztonságos kommunikáció során. Ebben a fejezetben részletesen bemutatjuk az EAP-TLS protokoll működését, előnyeit, kihívásait és gyakorlati alkalmazásait.

#### Bevezetés az EAP-TLS-be

Az EAP-TLS az EAP (Extensible Authentication Protocol) egyik módszere, amely az autentikációs keretrendszer különböző verziói és módszerei között az egyik legbiztonságosabb. Az EAP-TLS a hitelesítési folyamat során kölcsönösen hitelesíti mind a klienset (az eszközt), mind a szervert (a hitelesítési szervert), amely egyedülálló tulajdonsága a többi EAP módszerhez képest.

#### Az EAP-TLS Működése

Az EAP-TLS folyamat számos fázisra osztható:

1. **Előkészítés**:
    - A hitelesítési szerver (RADIUS/AAA szerver) és a kliens (például egy laptop vagy más vezeték nélküli eszköz) mindegyike rendelkezik egy X.509 tanúsítvánnyal. Ezek a tanúsítványok egy megbízható gyökér hitelesítési hatóságtól (CA) származnak.
    - Az eszközök és a szerver azonosítják és elrendezik a tanúsítványaikat.

2. **EAP Kezdeti Kézfogás**:
    - A kezdeti EAP üzenetcsere indul a kliens és a hitelesítési szerver között az autentikációs folyamat elindítására.

3. **TLS Kézfogás**:
    - Az EAP-TLS használja a TLS kézfogási protokollt az autentikáció részleteinek egy részének kezelésére.
    - A kliens és a szerver hitelesítő tanúsítványokat cserélnek, és a szerver hitelesíti a klienset a tanúsítvány alapján, míg fordítva is történik.

4. **Kulcs Deriválása és Biztosítás**:
    - A TLS kézfogás során egy titkosított közös kulcs kerül létrehozásra, amelyet mind a kliens, mind a szerver használ az adatcsomagok titkosítására.
    - A közös titkosítási kulcs biztonságát és integritását biztosító tanúsítványok felhasználásával jön létre.

5. **EAP Siker/EAP Hiba**:
    - Ha a hitelesítés sikeres, EAP siker üzenet kerül küldésre a kliensnek.
    - Ha a hitelesítés sikertelen, akkor EAP hiba üzenet kerül továbbításra.

#### EAP-TLS Előnyei

1. **Magas Biztonság**:
    - Az EAP-TLS az egyik legbiztonságosabb EAP módszer, köszönhetően a kölcsönös hitelesítésnek és a TLS protokoll használatának.
    - A TLS által biztosított erős titkosítási mechanizmusok megakadályozzák a lehallgatást, közbeékelődést, és más támadási formákat.

2. **Kölcsönös Hitelesítés**:
    - Mind a kliens, mind a szerver hitelesítik egymást tanúsítványok segítségével, így biztosítva, hogy mindkét fél megbízható.

3. **Széleskörű Sebesség és Méretezhetőség**:
    - EAP-TLS gyors és hatékony, alkalmas nagy mennyiségű kliens kezelésére is, ami különösen fontos nagyléptékű vállalati hálózatok esetén.

#### EAP-TLS Hátrányai és Kihívásai

1. **Tanúsítványkezelés**:
    - Az EAP-TLS megvalósításának legnagyobb kihívása a tanúsítványok kezelése és fenntartása.
    - Egy hiteles tanúsítvány infrastruktúra (PKI) szükséges, amely magában foglalja a tanúsítványok kiadását, visszavonását és megújítását.

2. **Hardver és Szoftver Követelmények**:
    - A TLS és az EAP-TLS megköveteli, hogy a kliens és a szerver megfelelő hardverrel és szoftverrel rendelkezzen a protokoll működésének biztosítására.

#### Gyakorlati Alkalmazások

Az EAP-TLS széles körben alkalmazott különböző vezeték nélküli hálózatokban, mint például a vállalati Wi-Fi hálózatok hitelesítésére. Ezen túlmenően, számos más környezetben is alkalmazható, mint például:

1. **Vezeték nélküli LAN-ok**:
    - Az IEEE 802.11i szabvány része, amely a WPA2 protokollban található.

2. **VPN Hálózatok**:
    - VPN kapcsolatok hitelesítésére, különösen, ahol magas szintű biztonság szükséges.

#### Összegzés

Az EAP-TLS az EAP protokollcsalád egyik kiemelkedő tagja, köszönhetően a kivételes biztonsági tulajdonságainak és a kölcsönös hitelesítési képességeinek. Bár a tanúsítványkezelés és a PKI infrastruktúra beállítása és karbantartása jelentős kihívásokkal jár, az általa kínált biztonság és megbízhatóság messze felülmúlja a ráfordítást. Az EAP-TLS továbbra is az egyik legjobb választás olyan környezetekben, ahol magas szintű védelem és integritás szükséges a hálózati autentikáció során.### 7.1.3 EAP-TTLS (EAP Tunneled Transport Layer Security)

### 7.1.3. EAP-TTLS (EAP Tunneled Transport Layer Security)
#### Bevezetés

Az Extensible Authentication Protocol (EAP) számos különböző hitelesítési módszert támogat, amelyek közül az egyik az EAP-TTLS (EAP Tunneled Transport Layer Security). Az EAP-TTLS kidolgozása a TTLS-re, vagyis a Tunneled Transport Layer Security-re épül, amelyet a kétirányú hitelesítés és az adatok titkosítása jellemez. Az EAP-TTLS célja a hitelesítési folyamat biztonságosságának növelése, különös tekintettel a hitelesítési adatok védelmére.

#### EAP-TTLS Áttekintés

Az EAP-TTLS egy kétfázisú hitelesítési protokoll. Az első fázisban egy TLS-tunnel (mártógató csatorna) jön létre a kliens és a szerver között. A második fázisban a valódi hitelesítési információk cseréje történik ezen a titkosított csatornán keresztül. A TLS-csatorna egyéni hitelesítésére használhatunk CA-alapú szerver tanúsítványokat, biztonságos kapcsot biztosítva a hitelesítő szerverrel.

#### EAP-TTLS Protokollfolyamat

##### 1. Fázis: TLS Tunnel Létrehozása

1. **Hello Üzenetek**: A kliens egy `ClientHello` üzenetet küld a szervernek, amely tartalmazza a kliens által támogatott kriptográfiai algoritmusokat és paramétereket. A szerver válaszol egy `ServerHello` üzenettel, amelyben a szerver kiválasztja a használni kívánt kriptográfiai paramétereket.

2. **Szerver Hitelesítése**: A szerver elküldi a saját tanúsítványát a kliensnek, amelyet egy hitelesítésszolgáltató (CA) írt alá. Ezt követően a kliens ellenőrzi a tanúsítvány érvényességét és hitelességét.

3. **Kulcscsere**: A kliens és a szerver egyeztetik a titkosító kulcsokat, amelyeket a továbbiakban a TLS-tunnel titkosítására használnak. Ezt általában a Diffie-Hellman algoritmus segítségével hajtják végre.

4. **TLS Tunnel Létrehozása**: Miután a kulcscsere sikeresen lezajlott, megfelelően titkosított csatorna (TLS-tunnel) jön létre, amelyen keresztül a hitelesítési folyamat második fázisa bonyolítható.

##### 2. Fázis: Hitelesítési Információk Átadása

1. **EAP Üzenetek Cseréje**: Az előzőleg felépített TLS-csatornán keresztül a kliens és a szerver különféle EAP üzeneteket cserélhet. Ezek az üzenetek hordozzák a hitelesítési információkat, például felhasználói nevet és jelszót, amelyek az eredeti EAP formátumban is lehetnek.

2. **Hitelesítési Módszer**: Az EAP-TTLS lehetőséget ad többféle hitelesítési módszer használatára is, mint például MSCHAPv2, PAP, vagy OTP (One-Time Password).

3. **Hitelesítés**: A szerver feldolgozza a kapott hitelesítési adatokat és dönt a hitelesítés sikerességéről vagy elutasításáról.

4. **EAP Siker/Hiba**: A szerver jelzi a hitelesítési folyamat kimenetelét egy EAP-Success vagy EAP-Failure üzenet küldésével.

#### EAP-TTLS Biztonsági Szempontok

Az EAP-TTLS számos biztonsági előnyt nyújt, beleértve:

1. **Költséghatékony Szerver Hitelesítés**: Mivel a kliens hitelesítési adatai titkosított csatornán keresztül kerülnek átadásra, a felhasználó jelszava és egyéb hitelesítési adatos nem lesznek a támadók számára láthatók.

2. **Rugalmasság**: Az EAP-TTLS támogat számos különféle belső hitelesítési protokollt, mint az MSCHAPv2, PAP, és OTP, amelyek szükség szerint választhatók.

3. **Adatok Bizalmassága és Integritása**: A TLS-tunnel biztosítja az adatok titkosítását és integritásának védelmét, megakadályozva az illetéktelen hozzáféréseket és adatmanipulációkat.

4. **Támadások Elleni Védelem**: Az EAP-TTLS védelmet nyújt középső ember (man-in-the-middle) támadások, a replay támadások és a brute-force jelszó támadások ellen a TLS mechanizmusok révén.

#### Implementáció C++ Nyelven

A következő példakódrészlet bemutatja, hogyan hozhatunk létre egy EAP-TTLS kliens C++ nyelven az OpenSSL könyvtár segítségével:

```cpp
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <iostream>

void initialize_ssl_library() {
    SSL_load_error_strings();
    OpenSSL_add_ssl_algorithms();
}

void cleanup_ssl_library() {
    EVP_cleanup();
}

SSL_CTX* create_context() {
    const SSL_METHOD* method = TLS_client_method();
    SSL_CTX* ctx = SSL_CTX_new(method);
    if (!ctx) {
        std::cerr << "Unable to create SSL context" << std::endl;
        ERR_print_errors_fp(stderr);
        exit(EXIT_FAILURE);
    }
    return ctx;
}

void configure_context(SSL_CTX* ctx) {
    SSL_CTX_set_ecdh_auto(ctx, 1);
  
    // Load client's certificate
    if (SSL_CTX_use_certificate_file(ctx, "client.crt", SSL_FILETYPE_PEM) <= 0) {
        ERR_print_errors_fp(stderr);
        exit(EXIT_FAILURE);
    }

    // Load client's private key
    if (SSL_CTX_use_PrivateKey_file(ctx, "client.key", SSL_FILETYPE_PEM) <= 0 ) {
        ERR_print_errors_fp(stderr);
        exit(EXIT_FAILURE);
    }

    // Load CA certificate
    if (!SSL_CTX_load_verify_locations(ctx, "ca.crt", NULL)) {
        ERR_print_errors_fp(stderr);
        exit(EXIT_FAILURE);
    }
}

int main() {
    initialize_ssl_library();

    SSL_CTX* ctx = create_context();
    configure_context(ctx);

    SSL* ssl;
    int server = 0; // Socket file descriptor.
  
    // Initialize socket connection to the server here.
    // server = ...

    ssl = SSL_new(ctx);
    SSL_set_fd(ssl, server);

    if (SSL_connect(ssl) <= 0) {
        ERR_print_errors_fp(stderr);
    } else {
        std::cout << "Connected with " << SSL_get_cipher(ssl) << " encryption" << std::endl;
        SSL_write(ssl, "Hello, World!", 13);
        char buffer[256];
        SSL_read(ssl, buffer, sizeof(buffer));
        std::cout << "Received: " << buffer << std::endl;
    }

    SSL_free(ssl);
    close(server);
    SSL_CTX_free(ctx);
    cleanup_ssl_library();

    return 0;
}
```

Ez a példa egy alapvető vázlat arra, hogyan lehet létrehozni egy SSL kapcsolatot C++-ban az OpenSSL könyvtár segítségével. A valós alkalmazásokban további részletek, például megfelelő hibakezelés, EAP üzenetek küldése és fogadása, valamint a hitelesítési protokoll specifikus lépései kivitelezéséhez további munkára és kódra van szükség.

#### Összefoglalás

Az EAP-TTLS egy hatékony és biztonságos hitelesítési módszer, amelyet széles körben használnak különféle alkalmazásokban, beleértve vállalati hálózatok védelmét, Wi-Fi hozzáférési pontokat és más hálózati infrastruktúrákat. A TLS-tunnel használata révén képes biztosítani a hitelesítési adatok bizalmasságát és integritását, miközben a különféle belső hitelesítési protokollok rugalmasságát is nyújtja. Az EAP-TTLS szignifikánsan növeli a hitelesítési folyamat biztonságát, és megbízható védelmet nyújt számos modern támadási forma ellen.### 7.1.4. PEAP (Protected Extensible Authentication Protocol)

### 7.1.4. PEAP (Protected Extensible Authentication Protocol)

A Protected Extensible Authentication Protocol (PEAP) egy autentikációs keretrendszer, amelyet az IEEE 802.1X szabvány támogat, és széles körben alkalmaznak vezeték nélküli hálózatokban és egyéb hálózati hozzáférési pontokon. A PEAP protokoll célja többek között a hálózati kommunikáció biztonságának megerősítése, különösen olyan környezetekben, ahol az átvitt adatok lehallgatása vagy módosítása valós veszély. A PEAP egy olyan autonóm, az Extensible Authentication Protocol (EAP) kiterjesztése, amely a felhasználói hitelesítés fokozott védelmét nyújtja, különösen TLS (Transport Layer Security) protokoll segítségével.

#### PEAP működési mechanizmus

A PEAP sajátossága, hogy az EAP protokollt egy TLS rétegbe ágyazza. A PEAP két fázisból áll: egy külső TLS-alapú hitelesítési rétegből és egy belső EAP-alapú hitelesítési szakaszból.

##### 1. fázis: TLS-alapú külső hitelesítés
A PEAP első fázisa egy TLS kapcsolat létesítése a kliens és az autentikációs szerver között. Ebben a fázisban mind a kliens, mind a szerver hitelesítése megtörténik, bár a kliens hitelesítése opcionális és a fázis későbbi szakaszában, az EAP-szekvencia során is végrehajtható.

- **TLS Handshake**:
  A TLS kézfogás során biztonságos csatorna jön létre a kliens és a szerver között. Ennek a folyamatnak részeként:

    - A szerver és a kliens kiválasztják a TLS verziót és a kívánt titkosítási algoritmust.
    - A szerver elküldi a hitelesítési tanúsítványát a kliensnek.
    - A kliens ellenőrzi a szerver tanúsítványát, megbizonyosodva annak érvényességéről.
    - A szerver és a kliens egy szimmetrikus kulcsot generálnak közös titok megállapítására a későbbi titkosításhoz.

- **KRYPTOGRAPHYALGORITHM_NEGOTIATION**:
  A legelterjedtebb algoritmusok a TLS kézfogás során a Diffie-Hellman kulcscsere, RSA és az ECC (Elliptic Curve Cryptography). Ezek az algoritmusok különböző biztonsági és teljesítményszintű követelményeknek is megfelelnek.

##### 2. fázis: Belső EAP hitelesítési szakasz

Miután a TLS kapcsolat sikeresen létrejött, a PEAP második fázisában történik a valódi felhasználói hitelesítés, amely további EAP-módszereken keresztül valósul meg. A PEAP második fázisa az alábbi módon működik:

- **EAP Inner Method**:
  Ebben a szakaszban az EAP-módszer által elvégzett hitelesítési folyamat a TLS titkosított csatornáján belül zajlik. Ez lehetővé teszi bármelyik hitelesítési mechanizmus használatát a PEAP védelmi ernyője alatt, mint például EAP-MSCHAPv2, EAP-GTC, vagy EAP-TLS.

- **Username and Password Transmission**:
  A felhasználói azonosítót és jelszót továbbítják a PEAP védelmi csatornáján keresztül, biztosítva ezzel a hitelesítő adatok titkosságát és integritását.

- **Success or Failure**:
  Sikeres hitelesítést követően a PEAP befejezi a hitelesítési folyamatot és hitelesítési adatokat biztosít a kliens és a szerver között. Ha a hitelesítés bármelyik szakaszában meghiúsul, a PEAP megszakítja a kapcsolatot és a hozzáférés nem engedélyezett.

#### PEAP Variációk

A PEAP legismertebb változatai közé tartozik a PEAPv0 (EAP-MSCHAPv2) és a PEAPv1 (EAP-GTC), és mindkettőt különféle hitelesítési kontextusokban alkalmazzák.

- **PEAPv0/EAP-MSCHAPv2**:
  Ez az egyik leggyakrabban használt verzió, ahol a belső EAP hitelesítési módszerhez az Microsoft Challenge Handshake Authentication Protocol verzió 2 (MS-CHAPv2) protokollt alkalmazzák.

- **PEAPv1/EAP-GTC**:
  Ez a változat az EAP Generic Token Card (EAP-GTC) rendszert használja a belső hitelesítési folyamat során. Az EAP-GTC főként szöveg alapú hitelesítést használ, amely képes egyszerűbb hitelesítési folyamatokat támogatni, mint például egyszeri jelszavas rendszereket (OTP).

#### Biztonsági aspektusok

A PEAP jelentős előnyei közé tartozik a TLS alapú külső hitelesítés és az ebből adódó biztonsági tulajdonságok:

- **Adatbiztonság és Integritás**: A TLS támogatásával minden hitelesítési adat titkosított csatornán keresztül kerül továbbításra, amelyek védelmet nyújtanak a lehallgatás és az adatmanipuláció ellen.

- **Kölcsönös hitelesítés**: A PEAP támogatja mind a szerver, mind a kliens hitelesítését, így biztosítva, hogy mindkét fél biztonságosan felismerje egymást a hitelesítési folyamat során.

- **Vízi Jejtek Védelme**: A TLS és az EAP kombinációja megakadályozza a man-in-the-middle (MITM) támadásokat, és erős védelmet nyújt az ismétléstámadások ellen.

#### Alkalmazási területek

A PEAP protokoll különösen fontos a vállalati és egyetemi hálózatokban, ahol szükség van a vezeték nélküli hozzáférési pontok és a hálózati erőforrások védelmére. A különböző olyan helyzetek, ahol a PEAP gyakran használatos:

- **Vállalati Hálózatok**: A PEAP protokoll széles körben használt a vállalati környezetben, ahol erős és megbízható hitelesítési mechanizmusra van szükség a belső hálózati hozzáférések biztonságára.

- **Oktatási Intézmények**: Az egyetemi hálózatokban a PEAP gyakran alkalmazott az oktatók és hallgatók hitelesítése során, biztosítva a személyes adatokat és egyéb érzékeny információkat.

- **Nyilvános Wi-Fi Hálózatok**: A nyilvános Wi-Fi hozzáférési pontokon a PEAP fokozott biztonságot nyújt, amely védi a felhasználók adatait a potenciális támadásokkal szemben.

#### Következtetés

A PEAP protokoll a mai digitális világban kulcsfontosságú eszközként szolgál a hálózati hozzáférés biztonságának növelésére. A TLS alapú külső hitelesítési mechanizmusának és EAP alapú belső hitelesítési megoldásainak kombinációjával a PEAP megbízható és erős védelmet nyújt a különféle támadások és kockázatok ellen, amelyek a hálózati hozzáférés során felmerülhetnek. Az alkalmazási területek széles skálája és a rugalmas hitelesítési mechanizmusa miatt a PEAP egy elterjedt és hatékony megoldás a mai biztonsági követelményeknek megfelelően.### 7.1.5. LEAP (Lightweight Extensible Authentication Protocol)

### 7.1.5. LEAP (Lightweight Extensible Authentication Protocol)

A Lightweight Extensible Authentication Protocol (LEAP), amelyet a Cisco Systems fejlesztett ki, az EAP (Extensible Authentication Protocol) alapú authentikációs protokollok családjába tartozik. A LEAP egy könnyű, skálázható autenetikációs eljárás, amelyet a vezeték nélküli hálózatok biztonságának növelésére terveztek. Annak ellenére, hogy az 1990-es évek végén fejlesztették ki, a LEAP szerteágazó implementációval rendelkezik a Cisco infrastruktúrában. Az EAP keretrendszer biztosítja az autentikációt különböző hálózati fizikai rétegek számára, de maga a LEAP konkrét authentikációs stratégia megtestesítése.

#### LEAP Folyamata

##### 1. Az Autentikációs Szerver és a Kliens Közt
A LEAP autenetikációs folyamat két fő szereplővel dolgozik: az authentikációs szerver és a kliens (vagy végpont). Az authentikációs szerver tipikusan egy Remote Authentication Dial-In User Service (RADIUS) szerver. A fő cél az, hogy biztosítsák, hogy a felhasználó, aki a kliens gépen van, valóban jogosult a hálózathoz való csatlakozásra.

##### 2. Felhasználói Név és Jelszó
A folyamat első szakasza a felhasználói név és jelszó begyűjtésére irányul. A kliens megadja a felhasználói nevet az authentikációs szervernek.

##### 3. MS-CHAPv2 Alapú Kihívás Válasz (Challenge-Response)
A LEAP egy kihívás-válasz mechanizmust használ az autentikációra, amely az MS-CHAPv2 (Microsoft Challenge Handshake Authentication Protocol Version 2) protokollra épít.

- **Kihívás:** Az authentikációs szerver véletlenszerűen generál egy bit-sorozatot (kihívás) és elküldi azt a kliensnek.
- **Válasz:** A kliens ezt a kihívást használja, hogy egy egyedi válasszal reagáljon, amely tartalmazza a felhasználói jelszót egy hash algoritmussal kombinálva.

##### 4. Mutuális Autentikáció

Az MS-CHAPv2 továbbá lehetővé teszi a mutuális autentikációt is, ahol a szerver is bemutat egy kihívást, amelyre a kliensnek válaszolnia kell, ezáltal mindkét fél megbizonyosodik egymás identitásáról.

##### 5. Tranzitív Titkosítási Kulcscsere
A LEAP továbbá végrehajt egy kulcscserét is, amely lehetővé teszi, hogy a két fél közösen létrehozzon egy titkosítási kulcsot. Ez a kulcs később használható a felhasználói adatok titkosítására a hálózatra való átvitel során.

#### LEAP Biztonsági Érvényesítése

A LEAP kezdeti növekedésével és elterjedésével kapcsolatos visszajelzések, valamint számos publikáció rámutatott a potenciális biztonsági kockázatokra. Az MS-CHAPv2 használatából adódóan a LEAP kevésbé biztonságos az olyan modern támadások ellen, mint például az offline jelszó megfejtés, ahol a kihívás-válasz párt visszafejthetik, ha a jelszó nem elég erős.

#### LEAP Modern Alternatívái

Az ilyen biztonsági kockázatok miatt a Cisco és a szélesebb hálózati biztonsági közösség számos alternatív protokollt javasolt és fejlesztett ki. Ilyen alternatívák közé tartozik például a Protected EAP (PEAP) és az EAP-TLS (Transport Layer Security), amelyek hatékonyabb biztonsági mechanizmusokat és erősebb TLS-alapú védelemre épülnek.

#### Példakód C++ Nyelven

Az alábbiakban bemutatunk egy leegyszerűsített C++ kódot, amely egy kihívás-válasz mechanizmust demonstrál:

```cpp
#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <openssl/sha.h>

std::string generateChallenge() {
    std::srand(std::time(nullptr));  // use current time as seed for random generator
    std::string challenge;
    for (int i = 0; i < 16; ++i) {
        challenge += static_cast<char>(std::rand() % 256);
    }
    return challenge;
}

std::string hashPassword(const std::string &password, const std::string &challenge) {
    std::string combined = password + challenge;
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(combined.c_str()), combined.size(), hash);
    
    std::string result;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        result += static_cast<char>(hash[i]);
    }
    return result;
}

bool authenticate(const std::string &hashedPassword, const std::string &userHash) {
    return hashedPassword == userHash;
}

int main() {
    // Server side
    std::string challenge = generateChallenge();
    std::cout << "Challenge: " << challenge << std::endl;

    // Client side
    std::string password = "UserPassword";
    std::string hashedPassword = hashPassword(password, challenge);
    std::cout << "Hashed Password: " << hashedPassword << std::endl;

    // Server side verification
    std::string userHash = hashPassword(password, challenge);  // This would normally come from the server's stored data
    if (authenticate(hashedPassword, userHash)) {
        std::cout << "Authentication successful!" << std::endl;
    } else {
        std::cout << "Authentication failed!" << std::endl;
    }

    return 0;
}
```

A fenti példakódban egy egyszerű kihívás-válasz mechanizmust mutatunk be. A valós LEAP implementáció persze ennél jóval összetettebb és magába foglal több biztonsági mechanizmust.

#### Összegzés

A LEAP egy fontos lépést jelentett a vezeték nélküli hálózatok autentikációs technológiájának fejlődésében, különösen azáltal, hogy bevezette a kihívás-válasz mechanizmust és a mutuális autentikációt. Azonban a modern biztonsági követelményeknek való megfelelés érdekében az újabb EAP-alapú protokollok, mint például a PEAP és az EAP-TLS, jobb alternatívákat kínálnak, amelyek hatékonyabb védelmet biztosítanak napjaink fejlett támadási vektorai ellen.

### 7.1.6. EAP-FAST (Flexible Authentication via Secure Tunneling)

EAP (Extensible Authentication Protocol) egy rugalmas keretrendszert biztosít különböző hálózati autentikációs módszerekhez. Az EAP több varianssal rendelkezik, melyek különböző hálózati környezetekben különböző biztonsági szinteket és funkcionalitásokat kínálnak. Az egyik ilyen variáns az EAP-FAST (Flexible Authentication via Secure Tunneling), amelyet a Cisco Systems fejlesztett ki, hogy megbízható és biztonságos módját nyújtsa a vezeték nélküli hálózati autentikációnak.

#### 7.1.6.1. Az EAP-FAST előzményei és koncepciója

Az EAP-FAST az autentikációs adatokat egy biztonságos csatornán keresztül továbbítja, hasonlóan más EAP-protokollokhoz, például az EAP-TLS-hez. Azonban az EAP-FAST nagy előnye, hogy nem igényel tanúsítványokat, amelyeket az EAP-TLS használ. Ehelyett egy más megközelítést alkalmaz ismert, mint Protected Access Credential (PAC, Védett Hozzáférési Hitelesítő).

A PAC egy kulcs alapú hitelesítő adat, amelyet a hitelesítő szerver és a kliens oszt meg. Miután a PAC megosztásra kerül, a PAC segítségével biztonságos alagút jön létre az adatok titkosítására és védelmére. Ez az alagút lehetővé teszi az érzékeny hitelesítési információk továbbítását egy biztonságos csatornán keresztül.

#### 7.1.6.2. Az EAP-FAST főbb részei

Az EAP-FAST három fázisra oszlik:

1. **PAC Provisioning (PAC kiosztás):**
    - Ez a lépés magában foglalja a PAC terjesztését a hitelesítési szerver és a kliens között. A PAC tartalmaz egy PAC-OTK (One-Time Key), PAC-Info és PAC-Key részt.
    - A PAC provisioningt meg lehet valósítani előre konfigurálással vagy az EAP-FAST protokoll segítségével, amelyet PAC provisioning protokollnak neveznek.

2. **Establishment of a Secure Tunnel (Biztonságos alagút létrehozása):**
    - Miután a PAC átadásra került, a PAC-OTK segítségével egy biztonságos alagút jön létre a kliens és a szerver között.
    - Ezen a ponton a PAC-Key használatos az adatok titkosítására és az új kulcsok cseréjére.

3. **Authentication over the Secure Tunnel (Hitelesítés a biztonságos alagúton keresztül):**
    - Az autentikációs folyamat a létrehozott alagút segítségével történik, amely titkosítja az átvitt autentikációs adatokat.
    - Ily módon például felhasználónév és jelszó cseréje biztonságosan történhet.

#### 7.1.6.3. EAP-FAST Keretrendszer és Adatstruktúrák

Az EAP-FAST protokoll különböző adatstruktúrákat használ, amelyek biztosítják a folyamat stabilitását és integritását.

- **PAC-OTK:** (One-Time Key) Egy egyszeri kulcs, amit a kezdeti PAC létrehozásához használnak.

- **PAC-Key:** Ez egy állandó kulcs, amelyet a biztonságos alagút létrehozásához használnak.

- **PAC-Info:**  Meta-adatokat tartalmaz, például PAC azonosítót, lejárati időt stb.

Itt van egy egyszerű adatstruktúra példa C++ nyelven, amely bemutatja a PAC szervezését:

```cpp
#include <string>
#include <vector>

class PAC {
public:
    std::string PAC_OTK;
    std::string PAC_Key;
    std::string PAC_Info;
    
    PAC(std::string otk, std::string key, std::string info) 
    : PAC_OTK(otk), PAC_Key(key), PAC_Info(info) {}
};

class EAP_FAST {
    std::vector<PAC> pac_store;

public:
    void addPAC(PAC pac) {
        pac_store.push_back(pac);
    }

    PAC getPAC(int index) {
        // Error handling should be more robust in real applications
        if (index < 0 || index >= pac_store.size()) {
            throw std::out_of_range("Index out of range");
        }
        return pac_store[index];
    }
};
```

#### 7.1.6.4. EAP-FAST Üzenetformátumok

Az EAP-FAST protokoll különböző típusú üzeneteket definiál az EAP keretrendszeren belül. Az üzenetformátum általánosan tartalmazza az EAP üzenet fejléceit, PAC adatokat, valamint a titkosított adatokat. Az EAP üzenet fejléce tartalmazza az azonosítókat, hosszúságokat és típusokat. Az EAP-FAST esetén különösen fontos a PAC adatokat megfelelően titkosítani a biztonság érdekében.

Példa EAP üzenet fejlécre és EAP-FAST specifikus üzenetformátumra:

```cpp
struct EAP_Header {
    uint8_t code;
    uint8_t identifier;
    uint16_t length;
    uint8_t type;
};

struct EAP_FAST_Message {
    EAP_Header header;
    std::string pac;
    std::string encrypted_data;
};
```
#### 7.1.6.5. Kihívások és Megoldások az EAP-FAST használatakor

A PAC nem tanúsítvány-alapú, tehát nincs szükség komplex tanúsítvány infrastruktúrára, amely megkönnyítheti a telepítést és működést. Azonban, a PAC-en alapuló rendszer biztonsága nagymértékben függ a PAC titoktartásától és megfelelő tárolásától.

**PAC Titoktartás és Védelem:**

1. **Biztonságos Tárolás:** A PAC-ot megfelelő körülmények között kell tárolni, és az adott módszerek megfelelőnek kell lenniük az érzékeny adatok tárolására, mint például hardverbiztonsági modulok (HSM).

2. **Regenerálás és Lejárat:** A PAC-okat rendszeresen frissíteni és regenerálni kell, hogy minimalizálják a visszaélés esélyeit. A lejárati időket be kell tartani.

3. **Köztes támadások elleni védelem:** Biztosítani kell, hogy a PAC átadása során a köztes támadások kivédhetőek legyenek. Segítségül jöhetnek a megbízható időpecsételés és szinkronizáció, valamint az aszimmetrikus kriptográfia a PAC átvitele során.

#### 7.1.6.6. Összefoglalás

Az EAP-FAST egy rugalmas és biztonságos autentikációs protokoll, amely a PAC alapú megközelítést alkalmazza, hogy megbízható és biztonságos csatornát biztosítson az adatok továbbításához és hitelesítéséhez. Az EAP-FAST protokoll lépései és adatstruktúrái biztosítják a hálózati hozzáférés biztonságát anélkül, hogy komplex PKI infrastruktúrákra lenne szükség. A megfelelő PAC kezelési politikák, mint az időszakos regenerálás és biztonságos tárolás, alapvetőek a rendszer biztonságos működtetéséhez.### 7.1.7. EAP-SIM (EAP for Subscriber Identity Module)

#### Bevezetés

Az Extensible Authentication Protocol (EAP) egy rugalmas keretrendszer, amely számos autentikációs módszert támogat azzal a céllal, hogy biztonságos kommunikációt biztosítson különböző hálózati rendszerek számára. Az EAP-SIM (EAP for Subscriber Identity Module) egy olyan EAP módszer, amelyet kifejezetten mobil hálózatokban használnak, ahol a felhasználói azonosítást és hitelesítést a SIM (Subscriber Identity Module) kártyák segítségével végzik. Az EAP-SIM különösen fontos a GSM (Global System for Mobile Communications) hálózatokban, ahol a SIM-kártyák elterjedt eszközök a biztonság és az ügyfélazonosítás biztosításában.

#### EAP és EAP-SIM áttekintése

Az EAP-SIM az egyik olyan EAP módszer, amely lehetővé teszi a GSM SIM-kártyák által tárolt információk használatát a hitelesítési folyamat során. Ez a módszer kihasználja a GSM hálózatokban használt meglévő hitelesítési és kulcskezelési infrastruktúrát. Az EAP-SIM támogatja a hitelesítést, a kulcscserét és az adattitkosítást, amelyek kulcsfontosságúak a biztonságos hálózati kommunikációhoz.

#### EAP-SIM Működési Mechanizmusa

Az EAP-SIM autentikációs folyamat három fő fázisból áll: a kihívás-válasz fázis, az autentikációs- és kulcscsere fázis, valamint a titkosított adatcsere fázisa.

1. **Kihívás-válasz fázis:**
   A hálózati hozzáférési szerver (NAS - Network Access Server) elküldi a hitelesítési adatokat (RAND - Random Number) a kliens, illetve a SIM-kártya felé. A SIM-kártya ezután az RAND-t és a saját titkos kulcsát (Ki) használja az autentikációs válasz (SRES - Signed Response) és a származtatott kulcs (Kc) előállításához. A válasz visszaküldésre kerül a NAS-hoz ellenőrzés céljából.

2. **Autentikációs- és kulcscsere fázis:**
   Ebben a fázisban az autentikációval végzett és a hozzáférési hálózat kölcsönös hitelesítése megtörténik. A kliensek és a hálózati komponensek között hitelesítési információ (például MSK - Master Session Key és EMSK - Extended Master Session Key) kerül cserére, amelyek további biztonsági mechanizmusokhoz szükségesek.

3. **Titkosított adatcsere fázis:**
   A létrehozott származtatott kulcsokat további titkosítás és hitelesítés céljára használják, biztosítva ezzel a kommunikáció titkosságát és integritását az ügyfél és a hálózat között.

#### Részletes Működés és Üzenetfolyam

Az alábbiakban részletesen áttekintjük az EAP-SIM protokoll üzenetfolyamatát, különös figyelemmel az üzenetek felépítésére és a kulcscserére.

1. **Identity Request/Response:**
    - **EAP-Request/Identity:** Az authentikációs szerver (AS - Authentication Server) küldi el, hogy azonosítsa a klienst.
    - **EAP-Response/Identity:** A kliens válaszol a SIM-kártya azonosítójával (IMSI - International Mobile Subscriber Identity).

2. **EAP-SIM Start:**
    - **EAP-Request/SIM-Start:** Az AS kezdeményezi az EAP-SIM autentikációt, és további információkat is küldhet.
    - **EAP-Response/SIM-Start:** A kliens válaszol, és opcionálisan további információkat küldhet a SIM-ről.

3. **Challenge:**
    - **EAP-Request/SIM-Challenge:** Az AS egy vagy több párt (RAND, AUTN) küld a kliensnek.
    - **EAP-Response/SIM-Challenge:** A kliens számítja ki az SRES és a Kc értékeket, majd visszaküldi az SRES-t az AS-nak.

4. **Result/Success:**
    - **EAP-Request/SIM-Result:** Az AS ellenőrzi az SRES-t. Ha sikeres, elküldi a siker üzenetet.
    - **EAP-Success:** A sikeres hitelesítést jelzi a kliens számára, és megkezdődhet a kulcscsere.

#### Biztonsági Szempontok

Az EAP-SIM protokoll számos biztonsági mechanizmust tartalmaz a támadások megelőzése érdekében, beleértve a következőket:

- **Replay Protection:** Az EAP-SIM kihívás-válasz mechanizmusa védi a rendszer a visszajátszási támadások ellen, mivel minden kihívás egyedülálló.
- **Man-in-the-Middle Protection:** Az autentikáció és kulcscsere folyamatai biztosítják, hogy a köztes támadó ne tudja elfogni és manipulálni az adatokat.
- **Mutual Authentication:** Az EAP-SIM protokoll kölcsönös hitelesítést biztosít, így mind a kliens, mind a hálózat megbízik egymás hitelességében.

#### Implementációs Példa

Az alábbiakban egy egyszerű EAP-SIM autentikációs folyamatot mutatunk be C++ nyelven, hogy példát adjunk a protokoll implementációjára.

```cpp
#include <iostream>
#include <string>
#include <vector>

class SIMCard {
public:
    SIMCard(const std::string& ki) : ki_(ki) {}

    std::pair<std::string, std::string> Authenticate(const std::string& rand) {
        // Calculation of SRES and Kc are simplified for illustration purposes.
        std::string sres = "calculated_sres";  // Simplified calculation.
        std::string kc = "calculated_kc";      // Simplified calculation.
        return { sres, kc };
    }

private:
    std::string ki_;
};

class Authenticator {
public:
    Authenticator() {
        // Simulate retrieval of Kc and SRES in a real scenario.
        expected_sres_ = "calculated_sres";
    }

    bool VerifyResponse(const std::string& sres) {
        return expected_sres_ == sres;
    }

private:
    std::string expected_sres_;
};

int main() {
    // Initialize SIM card with a subscriber key.
    SIMCard sim("subscriber_key");

    // Authenticator initiates a challenge with a random number.
    std::string rand = "random_number";

    // SIM card computes the response based on the random number.
    auto [sres, kc] = sim.Authenticate(rand);

    // Authenticator verifies the response.
    Authenticator authenticator;
    bool isAuthenticated = authenticator.VerifyResponse(sres);

    if (isAuthenticated) {
        std::cout << "Authentication succeeded!" << std::endl;
        std::cout << "Derived key: " << kc << std::endl;
    } else {
        std::cout << "Authentication failed." << std::endl;
    }

    return 0;
}
```

#### Összegzés
Az EAP-SIM egy robusztus és jól bevált autentikációs protokoll, amely kihasználja a GSM hálózatokban elterjedt SIM-kártyák adottságait a biztonságos hitelesítés biztosítására. Az EAP-SIM protokoll lehetővé teszi a megbízható és kölcsönösen hitelesített kommunikációt a kliensek és a hálózatok között, hozzájárulva ezzel a mobil hálózatok általános biztonságához és integritásához. Az EAP-SIM protokoll használata különösen fontos a mai napig, amikor az adatbiztonság és a hálózati integritás a technológiai fejlődés középpontjában áll.### 7.1.8. EAP-AKA (Extensible Authentication Protocol for Authentication and Key Agreement)

### 7.1.8. EAP-AKA (EAP for Authentication and Key Agreement)

Az EAP-AKA (Extensible Authentication Protocol for Authentication and Key Agreement) egy széles körben alkalmazott hitelesítési protokoll, amelyet a 3GPP (Third Generation Partnership Project) szabványozott elsősorban a mobil hálózatok, különösen az UMTS (Universal Mobile Telecommunication System) és LTE (Long Term Evolution) rendszerek számára. Ez a protokoll kiterjeszti az EAP (Extensible Authentication Protocol) keretrendszert egy adott hitelesítési és kulcs-megállapodási mechanizmussal, amely a SIM (Subscriber Identity Module) és az USIM (Universal Subscriber Identity Module) kártyák használatán alapul.

Az EAP-AKA a mobil hálózatok legfontosabb szempontjaihoz alkalmazkodik, mint például a biztonság, megbízhatóság, és a hálózati roaming támogatása. A protokoll célja, hogy biztonságos hitelesítést és kulcs-megállapodást biztosítson a felhasználói eszközök és a hálózati infrastruktúra között.

#### Feltételek

Az EAP-AKA működéséhez alapvető feltétel, hogy mind a kliens (mobil eszköz) és a szerver (mobil hálózat) rendelkezzen egy közös bizalmas kulccsal, amelyet a SIM vagy USIM kártya tárol. Ezek a kártyák tartalmazzák a következő elemeket:

- **K (Subscriber Key)**: Ez egy titkos kulcs, amelyet a SIM/USIM és a mobil hálózat megosztott módon tárol.
- **RAND (Random challenge)**: Ez egy véletlenszerűen generált 128-bit hosszú érték, amelyet az autentikáció során használnak.
- **AUTN (Authentication Token)**: Ez egy hitelesítési token, amelyet a hálózat generál és küld a kliensnek.
- **IK (Integrity Key) és CK (Ciphering Key)**: Ezek a kulcsok a hitelesítés és az adatok titkosításához szükségesek.

#### EAP-AKA Protokoll Lépcsői

Az EAP-AKA protokoll több fázisból áll, amelyek közösen biztosítják a hitelesítést és a kulcscserét. Az alábbiakban bemutatjuk a protokoll lépéseit:

##### 1. Kezdeményezés

Az EAP-AKA hitelesítési folyamat az EAP-Kezdés (EAP-Start) üzenettel indul, amelyet általában a kliens küld a szervernek (autentikációs szerver). A szerver válaszul egy EAP-Identity kérést küld, amelyben a kliens azonosítja magát (IMSI - International Mobile Subscriber Identity).

##### 2. EAP-Request/Identity

Amikor a szerver fogadja az EAP-Kezdés üzenetet, egy EAP-Request/Identity üzenetet küld a kliensnek. A kliens válaszol az EAP-Response/Identity üzenettel, amely tartalmazza az IMSI-t vagy egy ideiglenes azonosítót (Pseudonym), ha korábban már hitelesítette magát.

##### 3. Challenge Üzenet

Az autentikációs szerver generál egy véletlenszerű kihívást (RAND) és hitelesítési adatot (AUTN), majd elküldi azokat a kliensnek egy EAP-Request/AKA-Challenge üzenetben.

##### 4. Kliens Oldali Hitelesítés

A kliens az SIM/USIM kártyán található K kulcs segítségével kiszámolja az XRES (Expected Response), CK, IK, és AUTN elemeket. Ezek az értékek összehasonlításra kerülnek a szerver küldött értékeivel. Ha az értékek megegyeznek, a hitelesítés sikeres.

##### 5. Válasz Üzenet

A kliens válaszol egy EAP-Response/AKA-Challenge üzenetben, amely tartalmazza az XRES értéket a szerver számára. A szerver összehasonlítja ezt az értéket az általa számított és elvárt RES (Response) értékkel.

##### 6. Hitelesítés Befejezése

Ha a szerver és a kliens által számított értékek megegyeznek, a hitelesítés sikeresnek tekinthető. A szerver elküldi az EAP-Success üzenetet a kliensnek.

### Titkosítás és Integritás

Az EAP-AKA protokoll az autentikáció mellett biztonságos kommunikációt is biztosít, köszönhetően a CK és IK kulcsoknak. Az IK (Integrity Key) használatos az adatok integritásának ellenőrzésére, míg a CK (Ciphering Key) biztosítja az adatok titkosítását, így védve azokat a lehallgatástól és módosítástól.

### Biztonsági Szempontok

Az EAP-AKA számos biztonsági funkcióval rendelkezik, amelyek a következők:

1. **Védettség a visszajátszásos támadások ellen:** Az AUTN hitelesítési token és a RAND kihívások dinamikusan generálódnak minden hitelesítési esemény során, így védenek a visszajátszásos támadások ellen.

2. **Adatvédelmi biztosítékok:** A pseudonyme (álnév) és ideiglenes azonosítók használatával biztosítja a felhasználók személyes adatainak védelmét a hitelesítési folyamat során.

3. **Közös titkos kulcsok:** A kliens és a szerver közötti közös titkos kulcsok használata extra réteg biztonságot ad a kommunikáció során.

#### EAP-AKA Példa (C++ Nyelven)

Az alábbi példa bemutatja az EAP-AKA protokoll néhány alapvető műveletének C++ kódját:

```cpp
#include <iostream>
#include <string>
#include <openssl/hmac.h>
#include <openssl/sha.h>

// K (Subscriber Key) és RAND (Random Challenge)
const std::string K = "secrect_shared_key";
const std::string RAND = "random_challenge";

// Funkció a HMAC-SHA256 alapú XRES kiszámításához
std::string calculate_XRES(const std::string& K, const std::string& RAND) {
    unsigned char* result;
    result = HMAC(EVP_sha256(), K.c_str(), K.size(), 
                  (unsigned char*)RAND.c_str(), RAND.size(), NULL, NULL);
    
    std::string hmac_result(reinterpret_cast<char*>(result), SHA256_DIGEST_LENGTH);
    return hmac_result;
}

int main() {
    // Kliens oldalon kiszámított XRES
    std::string XRES = calculate_XRES(K, RAND);
    
    // Szerver oldalon kiszámított XRES
    std::string expected_XRES = calculate_XRES(K, RAND);

    // Ellenőrzés
    if (XRES == expected_XRES) {
        std::cout << "Authentication Successful" << std::endl;
    } else {
        std::cout << "Authentication Failed" << std::endl;
    }

    return 0;
}
```

Ez a példa egy egyszerű HMAC-SHA256 alapú XRES kiszámító funkciót mutat be, amelyet az autentikációs folyamatban használnak.

### Összefoglalás

Az EAP-AKA egy robusztus és megbízható hitelesítési protokoll, amelyet széles körben alkalmaznak a mobil hálózatokban. Ez a protokoll a SIM/USIM kártyák erejét használja, hogy biztonságos hitelesítést és kulcsszabályozást biztosítson. Az EAP-AKA támogatja a mobil hálózatok különleges követelményeit, beleértve a roamingot, adatvédelmet és a visszajátszásos támadások elleni védelmet is. Az ilyen protokollok kulcsfontosságúak a jövő telekommunikációs rendszerei számára, mivel egyre nagyobb hangsúlyt kap a biztonság és a megbízhatóság.


### 7.1.9. EAP-MD5 (EAP-Message Digest 5)


Az Extensible Authentication Protocol (EAP) egy széles körben használt keretrendszer, amely számos authentikációs mechanizmus támogatására szolgál különböző hálózati technológiákban, például vezeték nélküli LAN-ok, PPP kapcsolat és sok más hálózati konfiguráció. Az EAP lehetővé teszi az új authentikációs módszerek egyszerű integrációját és implementálását a meglévő rendszerekben. Az egyik legismertebb és leggyakrabban alkalmazott EAP protokoll az EAP-MD5.

#### Az EAP-MD5 Áttekintése

Az EAP-MD5 (EAP-Message Digest 5) egy egyszerű és könnyen implementálható authentikációs módszer, amely az MD5 hash algoritmust használja az authentikációs folyamat során. Az MD5 egy kriptográfiai hash függvény, amely egy bemenetet egy fix hosszúságú, 128 bites hash értékké alakít.

Az EAP-MD5 relevanciája elsősorban az egyszerűségében és a széles körű támogatásában rejlik. Az MD5 öregedése és a hozzátartozó biztonsági aggályok ellenére, az EAP-MD5 még mindig használt, különösen olyan környezetekben, ahol az alapszintű biztonság elegendő. Azonban fontos megjegyezni, hogy az MD5 sérülékenységei miatt az EAP-MD5 manapság nem ajánlott olyan helyeken, ahol magas szintű biztonság szükséges.

#### Az MD5 Hash Algoritmus

Az MD5 (Message Digest Algorithm 5) egy élő klasszikus kriptográfiai hash funkció, amelyet Ronald Rivest fejlesztett ki 1991-ben. Az MD5 célja az, hogy egy bemeneti adatbázist fix hosszúságú 128 bites (16 bájt) hash kóddá alakítson. Az MD5 alapvető működését az alábbi lépésekben lehet összefoglalni:

1. **Padding a bemenethez:** A bemenet hosszát 512 bitre növeljük úgy, hogy egy 1 bittel kezdődő bitcsoportot adunk hozzá, majd annyi 0 bittel egészítjük ki, hogy a teljes hossz mod 512 egyenlő legyen 448-cal. Ezután hozzáadjuk a bemenet eredeti hosszát 64 bites formátumban.

2. **A bemenet feldarabolása:** A kibővített bemenetet 512 bites blokkokra osztjuk.

3. **Az inicializációs vektor beállítása:** Az MD5 négy 32 bites állapotregistert (A, B, C, D) használ, amelyeknek kezdeti értékei specifikus konstans értékek.

4. **A feldolgozás ciklusai:** Az MD5 négy, 16 iterációs hurokból álló ciklust futtat. Minden iteráció egy keverési funkciót alkalmaz az állapotregiszterekre és a bemenet egy 32 bites alblokkra.

5. **A regiszterek kimenete:** A végső hash értéket az A, B, C és D regiszterek bitjeinek összefűzésével kapjuk meg.

Annak ellenére, hogy az MD5 széles körben elterjedt, a biztonsági hiányosságai miatt mai alkalmazása korlátozott maradt.

#### Az EAP-MD5 Protokoll Felépítése

Az EAP-MD5 működése három fő szakaszra bontható: az authentikáció és a kapcsolati időzítések kezelése.

1. **EAP-Igénylés/Identitás:** Az authentikáció kezdetekor az authentikátor (pl. hozzáférési pont) küld egy EAP-Igénylés/Identitás üzenetet a kliensnek (pl. vezeték nélküli eszköz). A kliens válaszol az EAP-Válasz/Identitás üzenettel, amely tartalmazza a kliens identitását (pl. felhasználónevet).

2. **EAP-Igénylés/MD5-Challenge:** Az authentikátor egy véletlenszerű kihívás értéket küld egy EAP-Igénylés/MD5-Challenge üzenetben. Az üzenet tartalmazza a kihívás értéket (pl. 16 bájt hosszúságú véletlenszerű adat).

3. **EAP-Válasz/MD5-Challenge:** A kliens megkapja a kihívás értéket, és létrehoz egy hash értéket a következő módon:

    - Először a kliens titkos jelszavát és a kapott kihívás értéket összefűzi.
    - Ezt az összefűzött értéket MD5 hash algoritmussal átalakítja.
    - A kapott hash értéket egy EAP-Válasz/MD5-Challenge üzenetben visszaküldi az authentikátornak.

4. **Az authentikáció érvényesítése:** Az authentikátor az autentikálási folyamat során azonos módon létrehozza a hash értéket a kliens által küldött identitás és a kihívás alapján. Ha az autentikátor által számított hash érték egyezik a kliens által küldött hash értékkel, az autentikátor megerősíti az authentikáció érvényességét és az authentikáció sikeres lesz.

Az alábbiakban egy példa C++ kód látható, amely az MD5 hash algoritmust jeleníti meg:

```cpp
#include <iostream>
#include <cstring>
#include <openssl/md5.h>

void computeMD5(const char* data, size_t length, unsigned char* digest) {
    MD5_CTX md5Context;
    MD5_Init(&md5Context);
    MD5_Update(&md5Context, data, length);
    MD5_Final(digest, &md5Context);
}

int main() {
    const char* data = "example_data";
    unsigned char digest[MD5_DIGEST_LENGTH];

    computeMD5(data, strlen(data), digest);

    std::cout << "MD5 Digest: ";
    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
        printf("%02x", digest[i]);
    }
    std::cout << std::endl;

    return 0;
}
```

#### Az EAP-MD5 Biztonsági Kihívásai és Korlátai

Az EAP-MD5 népszerűsége ellenére jelentős biztonsági kérdések merültek fel az MD5 hash algoritmus kapcsán. Az MD5 több különböző támadási formában sebezhető, beleértve az üzenet ütközések és a pre-image támadásokat. Ezek a támadási módszerek lehetővé teszik az MD5 hash érték manipulációját és a hash érték gyors kiszámítását, ami komolyan veszélyezteti a biztonságot.

Az ütközéstanulmányok (collision attacks) kimutatták, hogy léteznek hatékony algoritmusok az azonos MD5 hash értékeket produkáló különböző bemenetek generálására. Emellett a pre-image támadások (pre-image attacks) lehetővé teszik egy adott hash értékhez tartozó eredeti bemenet visszafejtését, amely különösen nagy veszélyt jelent, ha a hash egy népszerű hashtáblán (rainbow table) alapul.

#### Összegzés

Az EAP-MD5 egy egyszerű és könnyen megvalósítható EAP alapú authentikációs protokoll, amely az MD5 hash algoritmuson alapul. Bár széles körben támogatott és használható alacsony biztonsági követelményekkel rendelkező környezetekben, a különböző biztonsági sebezhetőségek miatt az EAP-MD5 nem ajánlott magas szintű biztonságot igénylő alkalmazásokhoz. Az EAP-MD5-t gyakran felváltják modernebb és biztonságosabb EAP alapú megoldások, mint például az EAP-TLS vagy az EAP-PEAP, amelyek robusztusabb és ellenállóbb autentikációs mechanizmusokat kínálnak.

### 7.1.10. EAP-PSK (EAP Pre-Shared Key)

Az EAP (Extensible Authentication Protocol) sokféle autentikációs mechanizmus számára biztosít rugalmas keretrendszert. Az EAP egyik ilyen változata az EAP-PSK (Pre-Shared Key), amelyet az RFC 4764 szabvány ír le. Az EAP-PSK célja, hogy egy egyszerű, ugyanakkor biztonságos módszert nyújtson a hitelesítéshez hitelesítő adatok (például jelszavak) megosztásával a kommunikáló felek között.

#### EAP-PSK Áttekintés

Az EAP-PSK egyike a leggyakrabban alkalmazott EAP-metódusoknak, főként mivel nem igényel tanúsítványokat vagy más összetettebb hitelesítő mechanizmusokat. Alapját egy előre megosztott kulcs (PSK) képezi, amely mindkét kommunikációs végponton rendelkezésre áll. Az EAP-PSK egyaránt használható vezeték nélküli hálózatokban (pl. Wi-Fi) és vezetékes hálózatokban is.

#### EAP-PSK Protokoll

Az EAP-PSK négy fő üzenetből áll, amelyeket "HALO" üzeneteknek is neveznek:

1. **Initiate (I)**
2. **Authenticate (A)**
3. **Confirm (C)**
4. **Finish (F)**

Ezek az üzenetek az alábbiak szerint épülnek fel és működnek együtt a hitelesítés során.

##### 1. Initiate (I)

A folyamat az Initiate üzenettel kezdődik, amelyet az EAP-PSK kliens (tehát az állomás) küld az EAP-PSK szervernek. Ez az állapotot és minden olyan információt tartalmaz, amely szükséges a további hitelesítési lépésekhez.

##### 2. Authenticate (A)

A második üzenet, az Authenticate, válaszként érkezik a szervertől. Tartalmazza a hitelesítéshez szükséges kihívást (challenge) és egy véletlenszerűen generált számot, amelyet további titkosítási lépések során használnak majd.

##### 3. Confirm (C)

A kliens az Authenticate üzenetre egy Confirm üzenettel válaszol, amely tartalmazza a szerver által küldött kihívásra adott megfelelő válaszokat, valamint a kliens által generált új véletlenszerű számot.

##### 4. Finish (F)

Végezetül a szerver küldi el a Finish üzenetet, amely tartalmazza mind a kliens, mind a szerver által generált véletlenszerű számokat, és igazolja, hogy a hitelesítés sikeres volt.

#### Üzenet Felépítés

Az EAP-PSK üzenetek felépítése jól meghatározott és rögzített az RFC 4764 szabványban. Minden üzenet különféle mezőket tartalmaz, beleértve a következőket:

- **Message type (Üzenet típus)**: Az üzenet típusa (I, A, C, F).
- **Random numbers (Véletlenszámok)**: A véletlenszerűen generált számok, amelyeket mindkét fél készít a hitelesítés során.
- **Key information (Kulcs információk)**: Tartalmazza az előre megosztott kulcsot (PSK) és egyéb kulcspár információkat.
- **MAC (Message Authentication Code)**: Az üzenet hitelesítését és integritását biztosító kód.

##### Példa EAP-PSK Üzenet C++ Nyelven

Az alábbi példa egy egyszerű EAP-PSK Initiate üzenet létrehozását mutatja be C++ nyelven.

```cpp
#include <iostream>
#include <cstring>

struct EapPskInitiate {
    uint8_t messageType;
    uint8_t version;
    uint8_t randomNumbers[16];

    // Constructor initializes with given parameters
    EapPskInitiate(uint8_t version, uint8_t type, const uint8_t* randomNumbers) 
        : version(version), messageType(type) {
        std::memcpy(this->randomNumbers, randomNumbers, 16);
    }

    // Function to display message content for debugging purposes
    void display() const {
        std::cout << "EAP-PSK Initiate Message:" << std::endl;
        std::cout << "Version: " << static_cast<int>(version) << std::endl;
        std::cout << "Message Type: " << static_cast<int>(messageType) << std::endl;
        std::cout << "Random Numbers: ";
        for (int i = 0; i < 16; ++i)
            std::cout << std::hex << static_cast<int>(randomNumbers[i]) << " ";
        std::cout << std::dec << std::endl;
    }
};

int main() {
    uint8_t randomNumbers[16] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};

    EapPskInitiate initiateMessage(1, 1, randomNumbers);
    initiateMessage.display();

    return 0;
}
```

#### Biztonsági Elemzés

Az EAP-PSK biztonsága nagyban függ az előre megosztott kulcs (PSK) biztonságos létrehozásától és terjesztésétől. Mivel az EAP-PSK egyszerű összetétele miatt kevésbé bonyolult az implementáció, ugyanakkor megfelelő konfiguráció mellett biztosít erős hitelesítést. Számos támadási vektor kiküszöbölhető erős kulcsok alkalmazásával és a kommunikációs csatorna megfelelő védelemével.

##### Támadások és Megelőzés

- **Közbeékelődéses támadások (MitM, Man-in-the-Middle)**: EAP-PSK védelmet nyújt a közbeékelődéses támadások ellen a MAC kódok használatával, amelyek biztosítják az üzenetek hitelességét és integritását.
- **Szótáras támadások (Dictionary Attacks)**: Az EAP-PSK technikailag érzékeny lehet a szótáras támadásokra, ha az előre megosztott kulcs gyenge. Nagyon fontos megfelelő hosszúságú és összetettségű kulcsokat használni.
- **Visszajátszásos támadások (Replay Attack)**: Az egyes üzenetekben használt véletlenszámok (nonces) megvédik a protokollt a visszajátszásos támadásoktól, mivel minden hitelesítési folyamat különböző véletlenszerű értékeket használ.

#### Összegzés

Az EAP-PSK egy robusztus és egyszerű hitelesítési metódus, amely minimális infrastruktúra igényével és erős biztonsági jellemzőivel egyaránt megfelelő vezetékes és vezeték nélküli hálózatok számára. A megfelelő konfigurálással és előre megosztott kulcsok biztonságos kezelésével az EAP-PSK képes biztosítani a kommunikáció hitelességét és integritását a hálózati eszközök között. Az RFC 4764 szabvány részletesen bemutatja a protokoll működését és ajánlásait a biztonságos implementációra, amely alapvető fontosságú a gyakorlatban való alkalmazás során.### 7.1.1. EAP (Extensible Authentication Protocol)

Az Extensible Authentication Protocol (EAP) egy rugalmas keretrendszer, amelyet különböző hálózati autentikációs technikák támogatására terveztek. Az EAP nem egyetlen autentikációs protokoll, hanem egy általános formátumot és keretrendszert biztosít az autentikációs módszerek számára, amelyek különböző hálózati technológiákban és környezetekben használhatóak. Ezt a keretrendszert széles körben alkalmazzák a vezeték nélküli hálózatokban, például a WiFi-ben (IEEE 802.11), továbbá az IEEE 802.1X, PPP (Point-to-Point Protocol) és más hálózati hozzáférési módszerekben.
