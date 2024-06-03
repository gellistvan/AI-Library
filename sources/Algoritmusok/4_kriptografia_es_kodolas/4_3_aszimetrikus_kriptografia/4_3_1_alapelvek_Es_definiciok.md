\newpage

# 3. Aszimmetrikus kulcsú kriptográfia

Az aszimmetrikus kulcsú kriptográfia, más néven nyilvános kulcsú kriptográfia, a modern titkosítás rendkívül fontos és széles körben használt ágazata, amely forradalmasította a biztonságos adatkommunikációt. Míg a hagyományos, szimmetrikus titkosítási módszerek egy egyetlen közös kulcs használatán alapulnak a titkosításhoz és a visszafejtéshez, az aszimmetrikus kriptográfia két külön kulcsot alkalmaz: egy nyilvános kulcsot, amelyet bárki láthat és felhasználhat, valamint egy privát kulcsot, amelyet csak a címzett ismer. Ez a megközelítés számos biztonsági előnyt nyújt, többek között lehetővé teszi a digitális aláírások használatát, amelyek garantálják az üzenetek hitelességét és integritását, valamint támogatják a kulcsmegosztási mechanizmusokat anélkül, hogy titkos csatornákat kellene kialakítani a kulcsok továbbításához. Az alábbiakban részletesen ismertetjük az aszimmetrikus kriptográfia alapelveit, működési mechanizmusait és a gyakorlatban alkalmazott legnépszerűbb algoritmusokat, mint például az RSA, a Diffie-Hellman és az Elliptikus Görbe Kriptográfia (ECC).


## 3.1. Alapelvek és definíciók

Az aszimmetrikus kulcsú kriptográfia, más néven nyilvános kulcsú kriptográfia, a modern információbiztonság egyik legfontosabb és leginnovatívabb területe. Ezen módszerek alapelve az, hogy két különböző, de matematikailag összefüggő kulcsot használnak: egy nyilvános kulcsot a titkosításhoz, amelyet bárki megismerhet, és egy privát kulcsot a visszafejtéshez, amelyet csak a címzett ismer. Ez a megközelítés számos előnyt kínál a hagyományos szimmetrikus kulcsú kriptográfiához képest, különösen a kulcsok kezelésének és terjesztésének szempontjából. Az aszimmetrikus kulcsú kriptográfia lehetővé teszi a biztonságos kommunikációt és hitelesítést olyan nyílt rendszerekben is, ahol korábban nem találkoztak a résztvevők, forradalmasítva ezzel az információs társadalom működését. E fejezet célja, hogy bemutassa az aszimmetrikus kulcsú kriptográfia alapelveit, működését és jelentőségét, valamint hogy megismertesse az olvasót a legfontosabb algoritmusokkal, mint amilyen az RSA és az elliptikus görbéken alapuló kriptográfia (ECC).

### Nyilvános kulcsú titkosítás és dekódolás

A nyilvános kulcsú titkosítás két kulcspár használatán alapul: egy nyilvános kulcs és egy titkos (privát) kulcs. Ezeket a kulcsokat úgy tervezték, hogy matematikailag összefüggenek, de az egyikről a másikat kiszámítani gyakorlatilag lehetetlen. Az ilyen rendszerek alapvető működési elve a következő:

1. **Kulcspár generálás (Key Generation):** Egy algoritmus előállít egy pár kulcsot, egy nyilvános és egy privát kulcsot. A nyilvános kulcsot mindenki számára elérhetővé teszik, míg a privát kulcsot csak a tulajdonosa tartja titokban.

2. **Titkosítás (Encryption):** Az üzenet küldője (Alice) a címzett (Bob) nyilvános kulcsát használja az üzenet titkosítására. Mivel a nyilvános kulcs mindenki számára elérhető, bárki képes titkosítani az üzenetet, amelyet csak a címzett tud majd dekódolni.

3. **Dekódolás (Decryption):** A címzett (Bob) a saját privát kulcsát használja a titkosított üzenet dekódolására. Ez biztosítja, hogy csak Bob tudja elolvasni az eredeti üzenetet.

Így az aszimmetrikus kulcsú kriptográfia biztosítja a bizalmasságot, mivel csak a titkos kulcs birtokosa (Bob) képes dekódolni a titkosított üzenetet.

#### Matematikai alapelvek

Az aszimmetrikus kriptográfia különböző matematikai problémákat használ, amelyek nehezen megoldhatók, például a nagy számok prímtényezős felbontását vagy az elliptikus görbék problémáit. Az RSA algoritmus alapvető műveletei például a következőképpen néznek ki:

- **Kulcspár generálás:**
    1. Válasszunk két nagy prímszámot: \( p \) és \( q \).
    2. Számítsuk ki az \( n \) értéket, amely ezeknek a prímeknek a szorzata: \( n = pq \).
    3. Számítsuk ki \( \phi(n) \)-t, ahol \( \phi(n) = (p-1)(q-1) \).
    4. Válasszunk egy nyilvános exponens \( e \)-t, amely kisebb \( \phi(n) \)-nél, és többszöröse \( n \)-nek, tehát a legkisebb közös többszöröse \( 1 \), gcd\((e, \phi(n))=1\).
    5. Számítsuk ki a privát kulcsot \( d \)-t, amely az \( e \) inverze mod \( \phi(n) \), tehát \( e \cdot d \equiv 1 \pmod{\phi(n)} \).

- **Titkosítás:**
  A küldő „Alice” kódolja az üzenetet \( m \)-t Bob nyilvános kulcsával (e,n) az alábbiak szerint: \( c = m^e \mod n \), ahol \( c \) a titkosított szöveg.

- **Dekódolás:**
  Bob dekódolja az üzenetet a privát kulcsával (d,n) az alábbi módon: \( m = c^d \mod n \), így \( m \) visszaállítja az eredeti üzenetet.

#### Példa C++ kódban

Az alábbiakban egy egyszerű RSA titkosítási/dekódolási példa C++ nyelven:

```cpp
#include <iostream>
#include <cmath>
#include <utility>
#include <vector>

// Function to compute (base^exp) % mod
long long modExp(long long base, long long exp, long long mod) {
    long long result = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp % 2 == 1)
            result = (result * base) % mod;
        exp = exp >> 1;
        base = (base * base) % mod;
    }
    return result;
}

// Function to compute GCD
long long gcd(long long a, long long b) {
    while (b != 0) {
        long long t = b;
        b = a % b;
        a = t;
    }
    return a;
}

// Function to find modular inverse
long long modInverse(long long e, long long phi) {
    for (long long x = 1; x < phi; ++x) {
        if ((e * x) % phi == 1)
            return x;
    }
    return -1;
}

int main() {
    // Step 1: Choose two prime numbers (for simplicity, we'll use small primes here)
    long long p = 61, q = 53;
    long long n = p * q;
    long long phi = (p - 1) * (q - 1);

    // Step 2: Choose an integer e such that 1 < e < phi and gcd(e, phi) = 1
    long long e = 17;

    // Step 3: Compute d, the modular inverse of e
    long long d = modInverse(e, phi);

    std::cout << "Public Key (e, n): (" << e << ", " << n << ")\n";
    std::cout << "Private Key (d, n): (" << d << ", " << n << ")\n";

    // Example message
    long long message = 42;

    // Step 4: Encrypt the message
    long long encrypted = modExp(message, e, n);
    std::cout << "Encrypted message: " << encrypted << "\n";

    // Step 5: Decrypt the message
    long long decrypted = modExp(encrypted, d, n);
    std::cout << "Decrypted message: " << decrypted << "\n";

    return 0;
}
```

Ebben a C++ programban megtalálhatóak az RSA kulcsgenerálás, titkosítás és dekódolás alapjai. Természetesen a p és q választása éppen csak illusztrációs célú; valósághű környezetben ezek nagyságrendileg nagy prímek lennének, hogy megfeleljenek a biztonsági elvárásoknak.

### Kulcspárok és kulcsmenedzsment

A nyilvános kulcsú kriptográfia alapvető része a kulcspárok és azok megfelelő menedzsmentje. A kulcsmenedzsment kritikus fontosságú a rendszer biztonságának és megbízhatóságának fenntartása érdekében:

1. **Kulcsgenerálás:** Mint ahogy korábban tárgyaltuk, a kulcsokat úgy generálják, hogy azok megfeleljenek a rendszer matematikai követelményeinek. A generált kulcspárok minősége és a kulcsok tényleges véletlenszerűsége alapvető fontosságú.

2. **Kulcstárolás:** Nyilvános és titkos kulcsokat biztonságosan kell tárolni. A nyilvános kulcsokat biztonságos szervereken vagy adatbázisokban tartják, hogy könnyen hozzáférhetők legyenek, míg a titkos kulcsokat titkosított formában kell tárolni és védeni a jogosulatlan hozzáféréstől.

3. **Kulcsnyilvántartás:** A nyilvános kulcsokat gyakran digitális tanúsítványokkal validálják, amelyeket hitelesítő hatóságok (CA-k) bocsátanak ki. Ezek a tanúsítványok tartalmazzák a felhasználók hitelesített nyilvános kulcsait és garantálják azok hitelességét.

4. **Kulcsinaktiválás:** Ha egy kulcs kompromittálódik, azt azonnal érvényteleníteni kell (revocation). Ehhez használatosak a kulcs visszavonási listák (CRL-ek) vagy az Online Certificate Status Protocol (OCSP).

5. **Kulcscsere:** A rendszeres kulcscsere segíti a hosszú távú biztonság fenntartását. Kerülni kell a kulcsok hosszú ideig való használatát a potenciális kompromittálódások minimalizálása érdekében.

Az aszimmetrikus kriptográfia helyes és hatékony alkalmazásához elengedhetetlen a megfelelő kulcsmenedzsment gyakorlatok követése, amely biztosítja az adatbiztonságot, integritást és bizalmasságot.

### Kulcspárok és kulcsmenedzsment

A kulcspárok és a kulcsmenedzsment alapvető fontosságúak az aszimmetrikus kulcsú kriptográfiában. A kulcspárok generálása, tárolása, megosztása és visszavonása mind lényeges lépések a biztonságos kommunikáció és az adatok védelme érdekében. Ebben a fejezetben részletesen bemutatjuk a kulcspárok létrehozását, a kulcspár tárolásával és kezelésével járó kihívásokat, valamint a kulcsmegosztás és -visszavonás módszereit.

#### Kulcspárok generálása

A kulcspárok generálása magában foglalja a nyilvános és privát kulcsok matematikailag összefüggő párjának előállítását. A kulcsoknak meg kell felelniük bizonyos biztonsági követelményeknek, például megfelelő hosszúságúnak kell lenniük, hogy ellenálljanak a kriptanalízisnek. A kulcsok generálásához gyakran használt algoritmusok közé tartozik az RSA (Rivest-Shamir-Adleman) és az ECC (Elliptic Curve Cryptography).

##### RSA algoritmus

Az RSA algoritmus az egyik legszélesebb körben alkalmazott nyilvános kulcsú algoritmus. Az alábbiakban bemutatjuk az RSA kulcspár generálásának lépéseit.

1. **Két nagy prím szám**:

Válasszunk két különálló nagy prím számot, általában \( p \) és \( q \).

```c++
#include <iostream>
#include <cmath>

bool isPrime(long number) {
    if (number <= 1) return false;
    if (number <= 3) return true;

    if (number % 2 == 0 || number % 3 == 0) return false;

    for (long i = 5; i * i <= number; i += 6) {
        if (number % i == 0 || number % (i + 2) == 0)
            return false;
    }
    return true;
}

int main() {
    long p = 101, q = 103; // Example primes
    std::cout << "p is prime: " << isPrime(p) << std::endl;
    std::cout << "q is prime: " << isPrime(q) << std::endl;
    return 0;
}
```

2. **Modulus**:

Számítsuk ki a modulus értéket \( n \), amely \( p \times q \)-val egyenlő.

```c++
long n = p * q;
std::cout << "Modulus n: " << n << std::endl;
```

3. **Euler's Totient**:

Számítsuk ki Euler's totient (φ) értéket, amely \( (p-1) \times (q-1) \).

```c++
long phi = (p - 1) * (q - 1);
std::cout << "Euler's Totient φ: " << phi << std::endl;
```

4. **Nyilvános kulcs**:

Válasszuk ki az "e" nyilvános kulcsot, amely 1 és φ között található, és relatív prím φ-hoz.

```c++
long e = 3; // Often chosen e is 3, 17, or 65537
while(std::__gcd(e, phi) != 1) {
    e += 2;
}
std::cout << "Public key e: " << e << std::endl;
```

5. **Privát kulcs**:

Számítsuk ki az "d" privát kulcsot, amely az "e" multiplikatív inverze modulo φ-hoz.

```c++
long d = 0;
long k = 1;
while((1 + k * phi) % e != 0) {
    k++;
}
d = (1 + k * phi) / e;
std::cout << "Private key d: " << d << std::endl;
```

##### ECC algoritmus

Az elliptikus görbéken alapuló kriptográfia (ECC) kisebb kulcsméretekkel éri el ugyanazt a biztonsági szintet, mint az RSA, ami különösen hasznos az erőforrásokban korlátozott környezetekben. Az ECC kulcspár generálási folyamat rövidebb és hatékonyabb lehet a fenti RSA példa ellenében.

```
to be continued with ECC example and additional subsections covering key storage, sharing, and revocation.
```

