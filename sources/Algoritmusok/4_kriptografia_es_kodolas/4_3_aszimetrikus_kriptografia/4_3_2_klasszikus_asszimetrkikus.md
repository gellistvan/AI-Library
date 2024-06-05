\newpage

## 3.2. Klasszikus aszimmetrikus algoritmusok

Az aszimmetrikus kulcsú kriptográfia, más néven nyilvános kulcsú kriptográfia, a modern titkosítás rendkívül fontos és széles körben használt ágazata, amely forradalmasította a biztonságos adatkommunikációt. Míg a hagyományos, szimmetrikus titkosítási módszerek egy egyetlen közös kulcs használatán alapulnak a titkosításhoz és a visszafejtéshez, az aszimmetrikus kriptográfia két külön kulcsot alkalmaz: egy nyilvános kulcsot, amelyet bárki láthat és felhasználhat, valamint egy privát kulcsot, amelyet csak a címzett ismer. Ez a megközelítés számos biztonsági előnyt nyújt, többek között lehetővé teszi a digitális aláírások használatát, amelyek garantálják az üzenetek hitelességét és integritását, valamint támogatják a kulcsmegosztási mechanizmusokat anélkül, hogy titkos csatornákat kellene kialakítani a kulcsok továbbításához. Az alábbiakban részletesen ismertetjük az aszimmetrikus kriptográfia alapelveit, működési mechanizmusait és a gyakorlatban alkalmazott legnépszerűbb algoritmusokat, mint például az RSA, a Diffie-Hellman és az Elliptikus Görbe Kriptográfia (ECC).

### 3.2.1 RSA (Rivest-Shamir-Adleman)

Az RSA (Rivest-Shamir-Adleman) algoritmus egy olyan aszimmetrikus kulcsú kriptográfiai módszer, amelyet széles körben használnak a biztonságos adatátvitel érdekében. Az RSA különlegessége abban rejlik, hogy két különálló, mégis matematikailag összekapcsolt kulcspárt használ: egy nyilvános kulcsot a titkosításhoz és egy privát kulcsot a dekódoláshoz. Ez a megközelítés lehetővé teszi, hogy a titkos információkat biztonságosan küldjék, anélkül, hogy a küldőnek és a fogadónak előzetesen meg kellene osztania egy közös titkot. Az RSA algoritmus működése erőteljes matematikai alapokon nyugszik, különös tekintettel a prímszámok és az egész számok faktorizációjának nehézségére. A következő alfejezetekben részletesen bemutatjuk az RSA algoritmus működési elvét, kulcsgenerálási folyamatát, valamint a titkosítási és dekódolási lépéseket, amelyek együtt a modern kriptográfia egyik sarokkövét alkotják.

#### Algoritmus működése és matematikai alapjai

Az RSA (Rivest-Shamir-Adleman) algoritmus az egyik legismertebb és legszélesebb körben használt aszimmetrikus kulcsú kriptográfiai algoritmus. 1977-ben fejlesztették ki, és elsősorban a biztonságos adatátvitelre, digitális aláírásokra és kulcscserére használják. Az alapjául szolgáló matematikai problémák – nevezetesen a nagy számok prímtényezős felbontásának nehézsége – adja az algoritmus biztonságát. Az alábbiakban részletesen bemutatjuk az algoritmus működését és a hozzá kapcsolódó matematikai alapokat.

##### Alapvető Matematikai Fogalmak

Az RSA algoritmus működésének megértéséhez először néhány alapvető matematikai fogalmat kell ismertetni:

1. **Prímszámok és Kompozit Számok**:
    - Egy **prímszám** olyan természetes szám, amelynek két pozitív osztója van: 1 és saját maga. Például a \(2, 3, 5, 7\) prímszámok.
    - Egy **kompozit szám** olyan természetes szám, amely két vagy több prímszám szorzataként írható fel. Például a \(6 = 2 \times 3\).

2. **Euler-féle Totient Függvény (\(\varphi\))**:
    - Az \(\varphi(n)\) függvény az \(n\)-el relatív prím pozitív egész számok számát adja meg, ahol két szám relatív prím, ha legnagyobb közös osztójuk 1. Ha \(n = p_1^{e_1} p_2^{e_2} ... p_k^{e_k}\) a prímtényezős felbontás, akkor \(\varphi(n) = n \cdot (1 - \frac{1}{p_1})(1 - \frac{1}{p_2}) ... (1 - \frac{1}{p_k})\).

3. **Moduláris Aritmetika**:
    - Az RSA algoritmusban a számításokat egy modul \((n)\) alatti maradékrendszerben végezik. Például egy \(x \equiv a \pmod{n}\) kijelentés azt jelenti, hogy az \(x\) osztva \(n\)-el, az \(a\) maradékot adja.

4. **Nagy Prímszámok Prímtényezős Felbontása**:
    - Az RSA biztonsága azon alapul, hogy egy nagy szám két nagy prímszámra való gyors prímtényezős felbontása rendkívül nehéz és számításigényes feladat.

##### Algoritmus Leírása

Az RSA algoritmus három fő lépésből áll: kulcsgenerálás, titkosítás és dekódolás. Az alábbiakban részletesen ismertetjük ezeket a lépéseket.

###### 1. Kulcsgenerálás

**Kulcsgenerálás lépései**:

1. **Két nagy prímszám kiválasztása \(p\) és \(q\)**:
   Válasszunk két nagy méretű prímszámot \(p\) és \(q\).

2. **Nagy egész szám \(n\) kiszámítása**:
   \(n = p \cdot q\)

3. **Euler-féle Totient függvény (\(\varphi\)) kiszámítása**:
   \(\varphi(n) = (p - 1) \cdot (q - 1)\)

4. **Nyilvános kitevő (\(e\)) kiválasztása**:
   Válasszunk egy \(e\) egész számot úgy, hogy \(1 < e < \varphi(n)\) és \(\gcd(e, \varphi(n)) = 1\). Az \(e\) gyakran a 65537-et választják, ami egy közismert és gyakran használt nyilvános kitevő a gyakorlati alkalmazásban.

5. **Privát kitevő (\(d\)) kiszámítása**:
   Számítsuk ki a \(d\) privát kitevőt, mint az \(e\) moduláris inverzét \(\varphi(n)\)-nel szemben, azaz: \(d \cdot e \equiv 1 \pmod{\varphi(n)}\). Ezt az egyenletet a bővített euklideszi algoritmus segítségével oldhatjuk meg.

A nyilvános kulcs a \((e, n)\) pár, míg a privát kulcs a \((d, n)\) pár.

###### 2. Titkosítás

A titkosítás során a nyilvános kulcsot használjuk. Tegyük fel, hogy \(M\) az üzenet, amelyet titkosítani szeretnénk, ahol \(M\) egy egész szám és \(0 \le M < n\).

**Titkosítási lépés**:
- Számítsuk ki a titkosított üzenetet \(C\) az alábbi módon:
  \[
  C = M^e \pmod{n}
  \]

###### 3. Dekódolás

A dekódolás során a privát kulcsot használjuk. Tegyük fel, hogy \(C\) a kapott titkosított üzenet.

**Dekódolási lépés**:
- Számítsuk ki a visszafejtett üzenetet \(M\) az alábbi módon:
  \[
  M = C^d \pmod{n}
  \]

##### Matematika mögött

Az RSA algoritmus érvényessége és biztonságossága az alábbi matematikai tételeken alapul:

1. **Euler-tétel**:
   Ha \(a\) és \(n\) relatív prímek, akkor \(a^{\varphi(n)} \equiv 1 \pmod{n}\).

2. **Tétel**:
   A titkosított üzenet visszafejtésekor \(M = (M^e)^d \pmod{n}\) egyenlőséget két esetben tekintjük:
    - \(M\) és \(n\) relatív prímek: \(M^{ed} \equiv M \pmod{n}\) az azonos kombináció eredményeként.
    - \(M\) legalább egy prímtényező, p-t tartalmaz \(q\)-ból: \(M^{ed} \equiv M \pmod{q}\) hasonlóképpen.

Ezek által az alapvető matematikai eredmények, amikor mindkét esetet figyelembe vesszük, az RSA algoritmus biztosítja a helyes visszafejtést.

Az RSA algoritmus a modern kriptográfia alappillére. Alapvető matematikai tulajdonságok, úgymint prímszámok, Euler-féle Totient függvény és moduláris aritmetika, alapozzák meg. Az algoritmus három szakaszra bontható: kulcsgenerálás, titkosítás és dekódolás. A kulcsgenerálás során két nagy prímszám segítségével hozzuk létre a nyilvános és privát kulcspárokat, amelyek biztonsága a prímtényezős felbontás nehézségéből fakad. A titkosítás és dekódolás moduláris hatványozáson alapul, biztosítva az üzenetek biztonságos továbbítását és visszafejtését.

##### Matematikai Alapok

Az RSA algoritmus a következő alapelvekre épül:
1. **Nagy prímek szorzata**: Az algoritmus két nagy prímszám (\(p\) és \(q\)) szorzatát (\(n = pq\)) használja.
2. **Euler-féle totient függvény**: Az RSA algoritmus működése Euler-féle totient függvényen alapul (\(\varphi(n)\)), amely a szám faktoriálisának egy speciális változata. Mivel \(n\) két prím szorzata (\(n = pq\)), az Euler-totient függvény az RSA esetében így néz ki: \(\varphi(n) = (p-1)(q-1)\).
3. **Nyilvános és privát kulcs**: Az algoritmus nyilvános kulcsa egy (\(e, n\)) pár, ahol \(e\) egy olyan szám, amelynek \(\gcd(e, \varphi(n)) = 1\), azaz \(e\) és \(\varphi(n)\) relatív prímek. A privát kulcs (\(d, n\)) kiszámolásához \(d\) olyan szám, amely kielégíti az \(ed \equiv 1 (\text{mod} \varphi(n))\) kongruenciát, azaz \(\frac{1}{e} \ (\text{mod} \ \varphi(n))\) értéket ad.

RSA másik alapeleme, hogy a megfelelő nagy \((p \text{ és } q\)) prímek választása esetén az \(n\)-t rendkívül nehéz faktorálni \(p\) és \(q\)-ra, ami az algoritmust erőssé teszi a modern kriptográfiai alkalmazásokban.

#### Kulcsgenerálás és titkosítási-dekódolási folyamat

##### Kulcsgenerálás

Az RSA algoritmus kulcsgenerálási folyamata a következő lépésekből áll:

1. **Kiválasztás két nagyméretű, különböző prímszámot**, \(p\)-t és \(q\)-t.
2. **Számítsd ki \(n\)-t**: \(n = p \cdot q\).
3. **Számítsd ki \(\varphi(n)\)-t**: \(\varphi(n) = (p-1)(q-1)\).
4. **Válaszd ki az \(e\) számot**: ahol \(1 < e < \varphi(n)\) és \(\gcd(e, \varphi(n)) = 1\).
5. **Számítsd ki \(d\)-t**: ahol \(d \equiv e^{-1} (\text{mod} \ \varphi(n))\).

**Példa C++ Kódrészlet:**
```cpp
#include <iostream>

#include <cmath>
#include <cstdlib>

#include <ctime>

// Funkció a legnagyobb közös osztó meghatározására
long gcd(long a, long b) {
    while (b != 0) {
        long t = b;
        b = a % b;
        a = t;
    }
    return a;
}

// Funkció az euklideszi kiterjesztett algoritmussal d kiszámolására
long modInverse(long e, long phi) {
    long t = 0, newt = 1;
    long r = phi, newr = e;

    while (newr != 0) {
        long quotient = r / newr;
        t = newt;
        newt = t - quotient * newt;
        r = newr;
        newr = r - quotient * newr;
    }

    if (t < 0) t += phi;

    return t;
}

// Kulcs generálás
void generateRSAKeys(long& n, long& e, long& d) {
    srand(time(0));

    long p = 61; // Prímszám
    long q = 53; // Prímszám
    n = p * q;

    long phi = (p - 1) * (q - 1);

    do {
        e = rand() % phi;
    } while (gcd(e, phi) != 1);

    d = modInverse(e, phi);
}

int main() {
    long n, e, d;
    generateRSAKeys(n, e, d);
    std::cout << "Public Key: (e: " << e << ", n: " << n << ")\n";
    std::cout << "Private Key: (d: " << d << ", n: " << n << ")\n";
    return 0;
}
```

##### Titkosítási Folyamat

Az üzenet (\(m\)), amit titkosítani szeretnénk, először egy számjegysorozatra (\(m < n\)) kell konvertálni. A titkosított üzenet (\(c\)) így értendő:
\[ c \equiv m^e (\text{mod} \ n) \]

**Példa C++ Kódrészlet:**
```cpp
#include <iostream>

#include <cmath>

// Titkosítási funkció
long encrypt(long m, long e, long n) {
    long c = 1;
    for (int i = 0; i < e; ++i) {
        c = (c * m) % n;
    }
    return c;
}

int main() {
    long n = 3233; // n, p * q
    long e = 17; // Public exponent

    long m = 65; // Message to encrypt
    long c = encrypt(m, e, n);

    std::cout << "Encrypted message: " << c << "\n";
    return 0;
}
```

##### Dekódolási Folyamat

A dekódolási folyamat során a titkosított üzenetet (\(c\)) a privát kulccsal (\(d\)) vissza kell fejteni az eredeti üzenetre (\(m\)):
\[ m \equiv c^d (\text{mod} \ n) \]

**Példa C++ Kódrészlet:**
```cpp
#include <iostream>

#include <cmath>

// Dekódolási funkció
long decrypt(long c, long d, long n) {
    long m = 1;
    for (int i = 0; i < d; ++i) {
        m = (m * c) % n;
    }
    return m;
}

int main() {
    long n = 3233; // n, p * q
    long d = 2753; // Private exponent

    long c = 2790; // Encrypted message
    long m = decrypt(c, d, n);

    std::cout << "Decrypted message: " << m << "\n";
    return 0;
}
```

#### Összegzés

Az RSA algoritmus alapjaiban erős matematikai elveken alapul, amelyek biztonságot garantálnak a modern kriptográfiai rendszerekben. A nagy prímszámok választása és az Euler-féle totient függvény használata az alapja annak, hogy a titkosítási algoritmus erős és ellenáll a modern dekriptanalízisnek. Az RSA hatékony megvalósítása megfelelően finom és optimalizált kódolási kérdésekkel is jár, amelyek a gyakorlatban biztosítják a kívánt biztonságot.

### 3.2.1 Diffie-Hellman kulcscsere

A Diffie-Hellman kulcscsere az aszimmetrikus kriptográfia egyik alapvető és forradalmi algoritmusa, amely lehetővé teszi két fél számára, hogy biztonságosan megosszák egymással egy titkos kulcsot, még akkor is, ha a kommunikáció egy nyilvános csatornán zajlik. Az 1976-ban Whitfield Diffie és Martin Hellman által bemutatott módszer áttörést jelentett a kriptográfia világában, mivel megszüntette a korábbi szimmetrikus kulcscsere módszerek legnagyobb gyengeségét: a biztonságos kulcsszállítást. Az alábbiakban részletesen megvizsgáljuk a Diffie-Hellman algoritmus működését, annak alapelveit, valamint a különféle gyakorlati alkalmazási területeit, amelyek magukban foglalják az internetes biztonságot is.

#### Algoritmus működése és alkalmazások

A Diffie-Hellman kulcscsere az egyik legfontosabb és legelterjedtebb aszimmetrikus kriptográfiai algoritmus, amelyet Whitfield Diffie és Martin Hellman publikáltak 1976-ban. Ez az algoritmus lehetővé teszi két fél számára, hogy biztonságosan megosszák egy titkos kulcsot egy nyilvános csatornán keresztül, anélkül, hogy valaha is kicserélnének ténylegesen titkos adatokat. Ez a közös titkos kulcs aztán használható szimmetrikus titkosítási algoritmusokban adatok kódolására és dekódolására.

Az algoritmus alapvető működési elve a szorzóhatványozás (exponenciáció) és a moduláris aritmetika egy erős kombinációján alapul, amelyeket együtt használnak a diszkrét logaritmus probléma nehézsége kihasználására. A diszkrét logaritmus probléma kifejezetten az, hogy egy véletlenszerű alap (generátor), egy modulus és egy kitevő tekintetében nehéz visszafejteni az alapot és a modust ismerve az eredeti exponens értékét.

Tegyük fel, hogy Aliz és Bob szeretnének megosztani egy titkos kulcsot. Az algoritmus a következő lépéseken keresztül működik:

1. Közös paraméterek kiválasztása:
   Aliz és Bob közösen megegyeznek két nyilvános paraméterben: egy nagy prímszám \( p \) és egy primitív gyök (generátor) \( g \) modulo \( p \). Ezeket az értékeket nyilvánosan bejelentik.

2. Privát kulcsok kiválasztása:
    - Aliz választ egy véletlenszerű privát számot \( a \) (0 < a < p-1), és kiszámítja a nyilvános értékét \( A \), ahol \( A = g^a \mod p \).
    - Bob választ egy véletlenszerű privát számot \( b \) (0 < b < p-1), és kiszámítja a nyilvános értékét \( B \), ahol \( B = g^b \mod p \).

3. Publikus kulcsok kicserélése:
    - Aliz elküldi \( A \)-t Bobnak.
    - Bob elküldi \( B \)-t Aliznak.

4. Közös titkos kulcs kiszámítása:
    - Aliz kiszámítja a közös titkos kulcsot \( s \), ahol \( s = B^a \mod p \).
    - Bob kiszámítja a közös titkos kulcsot \( s \), ahol \( s = A^b \mod p \).

Mivel \( s = (g^b)^a \mod p = g^{ba} \mod p = g^{ab} \mod p = (g^a)^b \mod p \), Aliz és Bob ugyanazt a közös titkot fogják megkapni. Ennek köszönhetően a kulcscsere biztonságos anélkül, hogy bárki a nyilvános kommunikáción keresztül megtudná a tényleges titkos kulcsot.

#### Matematikai háttér és biztonság

A Diffie-Hellman kulcscsere biztonsága a diszkrét logaritmus problémán alapul. Az algoritmus értékállandóságát a következő tulajdonságaival lehet igazolni:
1. **Nehézség visszafejtés**: A diszkrét logaritmus probléma nehézsége miatt egyelőre nincs hatékony mód a privát kulcsok visszanyerésére a nyilvános kulcsok ismeretében.
2. **Man-in-the-middle támadás elleni védelem**: Bár alapszinten a Diffie-Hellman sérülékeny lehet egy man-in-the-middle támadással szemben, ha nincs autentikációs mechanizmus, a gyakorlatban ezt különféle módokon (pl. autentikációs protokollok alkalmazásával) lehet kezelni és megvédeni.
3. **Nagy prímszámok és primitív gyökök**: A biztonság növelése érdekében a \( p \) prímszámot elég nagyra kell választani, hogy a hatékonyság és a biztonság is megfelelő legyen.

#### C++ példakód

Az alábbiakban egy egyszerű C++ implementáció található a Diffie-Hellman kulcscserére:

```cpp
#include <iostream>

#include <cmath>
#include <cstdlib>

#include <ctime>

// Function to return (a^b) % c
long long mod_exp(long long a, long long b, long long c) {
    long long result = 1;
    a = a % c;
    
    while (b > 0) {
        if (b % 2 == 1) {
            result = (result * a) % c;
        }
        b = b >> 1;
        a = (a * a) % c;
    }
    return result;
}

int main() {
    // Public parameters
    long long p = 23;   // A large prime number
    long long g = 5;    // Primitive root modulo p
    
    // Private keys
    srand(time(0));
    long long a = rand() % (p - 1) + 1;   // Aliz's private key
    long long b = rand() % (p - 1) + 1;   // Bob's private key
    
    // Public keys
    long long A = mod_exp(g, a, p);  // Aliz's public key
    long long B = mod_exp(g, b, p);  // Bob's public key
    
    // Exchange public keys (A and B)
    // Compute shared secret
    long long s_alice = mod_exp(B, a, p); // Aliz computes shared secret
    long long s_bob = mod_exp(A, b, p);   // Bob computes shared secret
    
    // Both should have the same shared secret 's'
    std::cout << "Aliz's computed shared secret: " << s_alice << std::endl;
    std::cout << "Bob's computed shared secret: " << s_bob << std::endl;
    
    return 0;
}
```

### Alkalmazások

A Diffie-Hellman kulcscserét széleskörben használják különböző biztonságos kommunikációs rendszerekben. Ezek közé tartoznak:

1. **VPN (Virtual Private Networks)**: VPN-ek gyakran használják a Diffie-Hellman kulcscserét titkosítási kulcsok cseréjére, hogy biztonságos kapcsolatot teremtsenek távoli hálózatok között.

2. **TLS/SSL protokollok**: A diffie–hellman kulcscsere része az SSL/TLS protokolloknak, amelyeket az interneten történő biztonságos kommunikációhoz használnak, mint például a HTTPS.

3. **Kriptográfiai protokollok**: Számos kriptográfiai protokollban alkalmazzák, például az SSH-ban és az IPsec-ben biztosítják ezzel a titkosított adatcsatornák létrehozását.

4. **P2P hálózatok**: Peer-to-peer rendszerekben a Diffie-Hellman segít biztonságos adatcsatornák létrehozásában és a hitelesítési folyamatban.

### Összegzés

A Diffie-Hellman kulcscsere kulcsszerepet játszik a modern kriptográfiában és az információbiztonságban, lehetővé téve biztonságos kulcscserét nyilvános kommunikációs csatornákon keresztül. Biztonsági erősségét a diszkrét logaritmus problémán alapul ötletes matematikai megoldása adja, amely a jelenlegi és közeli jövőben alkalmazott számítástechnikai erőforrásokat meghaladó kihívást jelent visszafejtés szempontjából. Az algoritmus rugalmassága és hatékonysága miatt elengedhetetlen alkotóeleme az információvédelmi rendszereknek és a titkosított kommunikációs protokolloknak.### 6.2.3. LZW (Lempel-Ziv-Welch)






