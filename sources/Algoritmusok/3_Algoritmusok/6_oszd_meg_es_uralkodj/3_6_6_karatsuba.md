\newpage

## 6.6. Karatsuba algoritmus (Nagy számok szorzása)

A nagyméretű számok hatékony szorzása különösen fontos szerepet játszik a számítástudomány és a kriptográfia különféle területein. A tradicionális, iskola szintű szorzási módszer, amely $O(n^2)$ időkomplexitású, gyorsan válik néhány nagyságrenddel lassabbá, amikor a számok nagysága növekszik. Itt lép be a képbe a Karatsuba algoritmus, amely az "oszd-meg-és-uralkodj" stratégia egy elegáns alkalmazása és képes a szorzási feladatot $O(n^{\log_2 3})$ időkomplexitásra redukálni. Ebben a fejezetben bemutatjuk a Karatsuba algoritmus működési elvét, lépésről lépésre megvizsgáljuk annak implementációját, valamint teljesítmény elemzést és gyakorlati alkalmazásokat is tárgyalunk. Az olvasó nem csupán betekintést nyer az algoritmus részletes működésébe, hanem megértheti, hogyan nyújt jelentős előnyöket a nagyméretű számok kezelésében.

### 6.6.1 Algoritmus és implementáció

#### Algoritmus

A Karatsuba algoritmus az oszd meg és uralkodj elvet alkalmazva gyorsítja fel a nagy számok szorzását. Az alapötlet abban áll, hogy a hagyományos szorzási módszer több részre bontásával lehetővé válik a szorzások számának csökkentése. Az algoritmus lépései a következőek:

1. **A számok szétosztása:**
    - Két nagy számot $X$ és $Y$ osztunk két részre.
    - Tegyük fel, hogy $X$ és $Y$ mindegyike $n$ számjegyű, és ezt felírjuk $X$ = $10^{m}A + B$, $Y$ = $10^{m}C + D$ formában, ahol $A$, $B$, $C$ és $D$ körülbelül $n/2$ számjegyű részek.

2. **Köztes értékek kiszámítása:**
    - Határozzuk meg $AC$-t, $BD$-t és $(A+B)(C+D)$-t.
    - Ismert, hogy $X \times Y = AC \times 10^{2m} + ((A+B)(C+D) - AC - BD) \times 10^{m} + BD$.

3. **Rekurzív műveletek:**
    - Az eredeti $O(n^2)$ szorzás helyett három kisebb $O(n/2)$ műveletet hajtunk végre.

Az algebrai kifejezések alapján, a végső szorzás $X \times Y$ három kisebb szorzásra bontható, amelyeket a következőképpen írunk fel:
- $z0 = BD$
- $z1 = (A+B)(C+D)$
- $z2 = AC$

Az eredmények összeállítása:
$$
X \times Y = z2 \times 10^{2m} + (z1 - z2 - z0) \times 10^m + z0
$$

#### Implementáció

Az alábbiakban bemutatom, hogyan implementálható a Karatsuba algoritmus C++ nyelven.

```cpp
#include <iostream>

#include <string>
#include <cmath>

using namespace std;

// Helper function to add leading zeros to make the lengths of two strings equal
void equalizeLength(string &num1, string &num2) {
    if (num1.length() > num2.length())
        num2.insert(0, num1.length() - num2.length(), '0');
    else
        num1.insert(0, num2.length() - num1.length(), '0');
}

// Function to add two numbers represented as strings
string addStrings(string num1, string num2) {
    equalizeLength(num1, num2);
    string result = "";
    int carry = 0;
    int n = num1.length();

    for (int i = n - 1; i >= 0; i--) {
        int sum = (num1[i] - '0') + (num2[i] - '0') + carry;
        carry = sum / 10;
        result.insert(0, 1, (sum % 10) + '0');
    }

    if (carry)
        result.insert(0, 1, carry + '0');

    return result;
}

// Function to multiply two numbers represented as strings using the traditional algorithm
string multiplyStrings(string num1, string num2) {
    int n = num1.length();
    string result(n * 2, '0');

    for (int i = n - 1; i >= 0; i--) {
        int carry = 0;
        for (int j = n - 1; j >= 0; j--) {
            int mul = (num1[i] - '0') * (num2[j] - '0') + carry + (result[i + j + 1] - '0');
            carry = mul / 10;
            result[i + j + 1] = (mul % 10) + '0';
        }
        result[i] += carry;
    }

    // Remove leading zeros
    size_t pos = result.find_first_not_of("0");
    if (pos != string::npos)
        return result.substr(pos);

    return "0";
}

// Function to subtract two numbers represented as strings
string subtractStrings(string num1, string num2) {
    equalizeLength(num1, num2);
    string result = "";
    int n = num1.length();
    int borrow = 0;

    for (int i = n - 1; i >= 0; i--) {
        int sub = (num1[i] - '0') - (num2[i] - '0') - borrow;
        if (sub < 0) {
            sub += 10;
            borrow = 1;
        } else {
            borrow = 0;
        }
        result.insert(0, 1, sub + '0');
    }

    // Remove leading zeros
    size_t pos = result.find_first_not_of("0");
    if (pos != string::npos)
        return result.substr(pos);

    return "0";
}

// Karatsuba algorithm for large number multiplication
string karatsubaMultiply(string num1, string num2) {
    int n = num1.length();

    // Base case for recursion: if the numbers are small enough, use traditional multiplication
    if (n == 1)
        return to_string((num1[0] - '0') * (num2[0] - '0'));

    // Make the numbers of equal length and pad them if necessary
    equalizeLength(num1, num2);

    int m = n / 2;

    // Split the numbers into two halves
    string high1 = num1.substr(0, m);
    string low1 = num1.substr(m);
    string high2 = num2.substr(0, m);
    string low2 = num2.substr(m);

    // 3 recursive calls made to numbers approximately half the size
    string z0 = karatsubaMultiply(low1, low2);
    string z1 = karatsubaMultiply(addStrings(low1, high1), addStrings(low2, high2));
    string z2 = karatsubaMultiply(high1, high2);

    // Combine the results
    string result = addStrings(addStrings(z2 + string(n, '0'), subtractStrings(subtractStrings(z1, z2), z0) + string(m, '0')), z0);

    // Remove leading zeros
    size_t pos = result.find_first_not_of("0");
    if (pos != string::npos)
        return result.substr(pos);

    return "0";
}

int main() {
    string num1 = "123456789012345678901234567890";
    string num2 = "987654321098765432109876543210";

    cout << "Number 1: " << num1 << endl;
    cout << "Number 2: " << num2 << endl;

    string result = karatsubaMultiply(num1, num2);

    cout << "Product: " << result << endl;

    return 0;
}
```

A fenti megvalósításban:
- Az `equalizeLength` funkciót használjuk a két számjegyű lépések kiegyenlítésére.
- Az `addStrings` és `subtractStrings` funkciók a számok összegzésére és kivonására szolgálnak.
- A `multiplyStrings` funkció hagyományos módszerrel végzi el a kis számok szorzását.
- A `karatsubaMultiply` függvény maga valósítja meg a Karatsuba algoritmust rekurzív hívások segítségével.

### 6.6.2 Teljesítmény elemzés és alkalmazások

#### Teljesítmény elemzés

A Karatsuba algoritmus lényege a problémák kisebb részekre bontása, és ezek önálló megoldása kevesebb szorzási művelettel. A teljesítmény elemzése során figyelembe kell venni a szorzások és az összegzések, illetve kivonások számát.

Az időkomplexitás elemzése a következőképpen végezhető:

- Az algoritmus rekurzív karaktere miatt a futási idő kielégíti a következő egyenletet:
  $$
  T(n) = 3 T(\frac{n}{2}) + O(n)
  $$
- Megoldva ezt a master tétel segítségével, azt kapjuk, hogy:
  $$
  T(n) = O(n^{\log_2 3}) \approx O(n^{1.585})
  $$

Ez az időkomplexitás jelentősen javítja a hagyományos szorzás $O(n^2)$ komplexitását, különösen nagy $n$ esetén.

#### Alkalmazások

A Karatsuba algoritmus különösen hatékony nagy számjegyű számok szorzásakor, ami gyakori például az alábbi területeken:

- **Kriptográfia:** Hatalmas számokkal végzett műveletek, mint például a RSA kulcsok generálása és digitális aláírások létrehozása.
- **Numerikus számítások:** Tudományos számítások során, különösen szimulációkban és modellezésekben, ahol sok számjegyű precízió szükséges.
- **Számítógépes algebra rendszerek:** Matematikai szoftverek, mint például a Mathematica, MATLAB, amelyek nagy precíziójú aritmetikai műveleteket végeznek.

#### Algoritmikai átfutási idő elemzése

A hagyományos iskolai szorzás algoritmus esetében két $n$ számjegyből álló szám szorzásának időbonyolultsága $O(n^2)$. Ezzel szemben a Karatsuba algoritmus jelentősen javít ezen az időbonyolultságon.

**Karatsuba Szorzás Időbonyolultságának Elemzése:**

A Karatsuba algoritmus az alábbi rekurzív szerkezetet alkalmazza két $n$-számjegyű szám, $X$ és $Y$ szorzásakor:

1. Osszuk $X$-et és $Y$-t két részre:
   $$
   X = X_1 \cdot 10^{m} + X_0
   $$
   $$
   Y = Y_1 \cdot 10^{m} + Y_0
   $$
   ahol $X_0$ és $Y_0$ az alsó $m$ számjegy, $X_1$ és $Y_1$ a felső $m$ számjegy ($m \approx n/2$).

2. Számítsuk ki a következő köztes értékeket:
   $$
   Z_0 = X_0 \cdot Y_0
   $$
   $$
   Z_1 = X_1 \cdot Y_1
   $$
   $$
   Z_2 = (X_0 + X_1) \cdot (Y_0 + Y_1)
   $$

   Ezek köztes értékek rekurzív szorzásokkal számítanak.

3. Az eredmény a következőképpen áll össze:
   $$
   XY = Z_1 \cdot 10^{2m} + (Z_2 - Z_1 - Z_0) \cdot 10^{m} + Z_0
   $$

A Karatsuba algoritmus időbonyolultsága az alábbi T(n)-rekurzív relációval írható le:
$$
T(n) = 3T(n/2) + O(n)
$$

A Master tétel alkalmazásával meghatározhatjuk az időbonyolultságot:
$$
T(n) = O(n^{\log_2 3}) \approx O(n^{1.585})
$$

Ezen elemzés azt mutatja, hogy a Karatsuba algoritmus hatékonyabb, mint a hagyományos $O(n^2)$ szorzás, különösen nagy számok esetén.

#### Karatsuba Algoritmus Implementációja C++ Nyelven

```cpp
#include <iostream>

#include <vector>
#include <string>

#include <cmath>

// Function to add two large numbers
std::string add(const std::string &num1, const std::string &num2) {
    std::string result = "";
    int carry = 0;
    int n1 = num1.size();
    int n2 = num2.size();
    
    for (int i = 0; i < std::max(n1, n2); i++) {
        int digit1 = (i < n1) ? num1[n1 - 1 - i] - '0' : 0;
        int digit2 = (i < n2) ? num2[n2 - 1 - i] - '0' : 0;
        int sum = digit1 + digit2 + carry;
        result = std::to_string(sum % 10) + result;
        carry = sum / 10;
    }
    
    if (carry) {
        result = std::to_string(carry) + result;
    }
    
    return result;
}

// Function to subtract two large numbers (assuming num1 >= num2)
std::string subtract(const std::string &num1, const std::string &num2) {
    std::string result = "";
    int borrow = 0;
    int n1 = num1.size();
    int n2 = num2.size();
    
    for (int i = 0; i < n1; i++) {
        int digit1 = num1[n1 - 1 - i] - '0';
        int digit2 = (i < n2) ? num2[n2 - 1 - i] - '0' : 0;
        digit1 = digit1 - digit2 - borrow;
        if (digit1 < 0) {
            digit1 += 10;
            borrow = 1;
        } else {
            borrow = 0;
        }
        result = std::to_string(digit1) + result;
    }
    
    // Remove leading zeros
    size_t pos = result.find_first_not_of("0");
    if (pos != std::string::npos) {
        result = result.substr(pos);
    } else {
        return "0";
    }
    
    return result;
}

// Function to pad zeros to the left of a number
std::string padZeros(const std::string &num, int zeros) {
    std::string result = num;
    for (int i = 0; i < zeros; i++) {
        result = "0" + result;
    }
    return result;
}

// Karatsuba multiplication function
std::string karatsuba(const std::string &num1, const std::string &num2) {
    int n = std::max(num1.size(), num2.size());
    
    // Base case
    if (n == 1) {
        return std::to_string((num1[0] - '0') * (num2[0] - '0'));
    }
    
    // Pad the numbers with leading zeros
    std::string num1Padded = padZeros(num1, n - num1.size());
    std::string num2Padded = padZeros(num2, n - num2.size());
    
    int mid = n / 2;
    
    // Split the numbers into halves
    std::string X1 = num1Padded.substr(0, mid);
    std::string X0 = num1Padded.substr(mid);
    std::string Y1 = num2Padded.substr(0, mid);
    std::string Y0 = num2Padded.substr(mid);
    
    // Recursively calculate the three products
    std::string P1 = karatsuba(X1, Y1);
    std::string P2 = karatsuba(X0, Y0);
    std::string P3 = karatsuba(add(X1, X0), add(Y1, Y0));
    
    // Combine the results
    std::string result = add(add(padZeros(P1, 2 * (n - mid)), padZeros(subtract(P3, add(P1, P2)), n - mid)), P2);
    
    return result;
}

int main() {
    std::string num1 = "12345678901234567890";
    std::string num2 = "98765432109876543210";

    std::string product = karatsuba(num1, num2);
    std::cout << "Product: " << product << std::endl;

    return 0;
}
```

#### Teljesítmény összehasonlítása

A Karatsuba algoritmus hatékonyságát bizonyítja az, hogy nagy számok szorzása esetén jelentősen kevesebb rekurzív hívást és alapműveletet igényel. A gyakorlati alkalmazásokban gyakran megfigyelhető, hogy míg az egyes rekurzív lépések végrehajtási ideje nő, a teljes algoritmus kompenzál az alacsonyabb időbonyolultságnak köszönhetően.

**Összehasonlítás a Naiv Szorzási Algoritmussal:**

- **Naiv szorzás:** $O(n^2)$
- **Karatsuba szorzás:** $O(n^{1.585})$

Az alábbiakban bemutatjuk az elméleti és gyakorlati futási idő összehasonlítását három különböző számjegy-hossz esetén:

- **Számok hosszúsága $n = 10^3$**:
    - Naiv szorzás: $10^6$ művelet
    - Karatsuba: $10^4 * 2^{14.55} \approx 2.06 \times 10^4$ művelet

- **Számok hosszúsága $n = 10^6$**:
    - Naiv szorzás: $10^{12}$ művelet
    - Karatsuba: $10^6 * 2^{19.17} \approx 4.97 \times 10^7$ művelet

- **Számok hosszúsága $n = 10^9$**:
    - Naiv szorzás: $10^{18}$ művelet
    - Karatsuba: $10^9 * 2^{23.79} \approx 1.20 \times 10^{11}$ művelet

A fenti példák világosan mutatják, hogy a Karatsuba algoritmus mérsékelten vagy lényegesen csökkentheti a műveletek számát a növekvő bemeneti méret esetén.

#### Gyakorlati alkalmazások

A Karatsuba algoritmus hatékonyan használható különböző területeken:

1. **Nagy számokkal való számítások:**
   Banki rendszerek, kriptográfiai műveletek, pénzügyi modellek és tudományos kutatások során, ahol gyakran szükség van óriási számok pontos kezelésére.

2. **Kriptográfia:**
   A nagy számokkal való műveletek, mint például a nyilvános kulcsú titkosítási rendszerek (pl. RSA) alapvető eleme, ahol a nagy prímek szorzására és osztására van szükség.

3. **Matematikai Könyvtárak és Számológépek:**
   Számológépek és matematikai szoftverek, mint például MATLAB, Mathematica, vagy a Python GMP könytára (GNU Multiprecision Library) alkalmazzák a Karatsuba algoritmust nagy számok műveleteinek hatékony kezelésére.

4. **Tudományos Számítások:**
   Asztrofizikai szimulációk, számítógépes grafika és más nagy adatmennyiségű számítási feladatok esetén, ahol a nagy számú integrált és differenciálszámítások szükségesek.

5. **Adatbázisok és Nagy Adatmennyiségek:**
   Nagyméretű adatbázisok kezelése és kérések optimális végrehajtása során, ahol nagy volumenű numerikus adatok rendezése és műveletei gyakoriak.

Ezek a példák bemutatják, hogy a Karatsuba algoritmus széles körben alkalmazható, különösen olyan területeken, ahol a nagy számok számításának pontossága és gyorsasága kulcsfontosságú. Az algoritmus optimális választás nagy méretű numerikus adatok kezelése esetén, és jelentős hatással bír a számítástechnika és a matematika különböző területein.
