\newpage

## 1.9. Sztring keresési algoritmusok

A sztring keresési algoritmusok kulcsszerepet játszanak az informatikában, hiszen számos alkalmazási területen, például szövegszerkesztőkben, adatbázis-kezelőkben és bioinformatikai elemzésekben is használatosak. E fejezet célja, hogy bemutassa a legfontosabb sztring keresési algoritmusokat, amelyek hatékony megoldásokat nyújtanak különböző problémákra. Elsőként a Boyer-Moore algoritmus lesz tárgyalva, mely az egyik leghatékonyabb mintaillesztő eljárás hosszú sztringek esetén. Ezt követi a Knuth-Morris-Pratt (KMP) algoritmus, amely előfeldolgozási fázisa révén minimalizálja a szükséges összehasonlítások számát. Harmadikként a Rabin-Karp algoritmusról lesz szó, amely hashelési technikát alkalmazva teszi lehetővé a gyors keresést. Végül a Trie adatszerkezetek segítségével történő sztring keresés kerül bemutatásra, amely különösen hatékony tud lenni szótárak és előtag-keresések esetén. Ezek az algoritmusok különböző előnyökkel és hátrányokkal rendelkeznek, amelyek megértése hozzájárulhat a megfelelő módszer kiválasztásához az adott probléma megoldására.

### 1.9.1. Boyer-Moore algoritmus

A Boyer-Moore algoritmus az egyik legismertebb és leghatékonyabb sztring keresési módszer, különösen hosszú szövegek esetén. Robert S. Boyer és J Strother Moore által 1977-ben kifejlesztett algoritmus alapvető újítása, hogy a szöveget jobbról balra olvassa, ami lehetővé teszi a mintával nem egyező szakaszok gyors átugrását. Ezen átugrások révén az algoritmus lényegesen gyorsabb lehet az egyes betűk balról jobbra történő összehasonlítására épülő módszereknél, mint például a naiv keresési algoritmusnál.

#### Az algoritmus alapjai

A Boyer-Moore algoritmus két fő heurisztikára épít: a "Bad Character" és a "Good Suffix" heurisztikákra. Ezek a heurisztikák segítenek meghatározni, hogy a szövegben milyen mértékben lehet előrelépni, amikor egy összehasonlítás során nem találnak egyezést.

**Bad Character Heurisztika:**
A "Bad Character" heurisztika a minta azon karakterére koncentrál, amelyik nem egyezik meg a szöveg megfelelő karakterével. Ha a minta egy karaktere nem egyezik meg a szöveg egy karakterével, az algoritmus megvizsgálja, hogy hol fordul elő legközelebb a minta bal oldalán az adott karakter. Ha a karakter nem található meg a minta bal oldalán, az algoritmus a minta teljes hosszával előreléphet. Ellenkező esetben, az algoritmus a minta azon pozíciójára lép előre, ahol az adott karakter legközelebb előfordul.

**Good Suffix Heurisztika:**
A "Good Suffix" heurisztika a minta azon részére koncentrál, amely megegyezik a szöveg egy részével. Ha a minta vége és a szöveg egy szakasza megegyezik, de az összehasonlítás korábbi része eltér, az algoritmus megvizsgálja, hogy a minta bal oldali része tartalmazza-e ezt a szuffixet (vagy annak egy részét). Ha igen, akkor az algoritmus a minta azon pozíciójára lép előre, ahol a szuffix ismét előfordul. Ha nem, akkor a minta a szuffix teljes hosszával lép előre.

#### Az algoritmus lépései

1. **Előfeldolgozás:**
    - Készítünk egy "bad character" táblázatot, amely meghatározza, hogy egy adott karakter milyen mértékben mozdítható el a minta különböző pozícióiban.
    - Készítünk egy "good suffix" táblázatot, amely meghatározza, hogy a minta különböző szuffixei hogyan fordulnak elő a minta bal oldalán.

2. **Keresési folyamat:**
    - A minta végétől kezdve összehasonlítjuk a minta karaktereit a szöveg megfelelő karaktereivel.
    - Ha egy karakter nem egyezik, alkalmazzuk a "bad character" és "good suffix" heurisztikákat, hogy meghatározzuk, milyen mértékben léphetünk előre.
    - Az előrelépés után folytatjuk az összehasonlítást az új pozícióból.

#### Részletes magyarázat példával

Vegyünk egy egyszerű példát, hogy bemutassuk az algoritmus működését. Tegyük fel, hogy a minta a "EXAMPLE" és a szöveg az "HERE IS A SIMPLE EXAMPLE".

1. **Előfeldolgozás:**
    - A "bad character" táblázatot az alábbiak szerint hozzuk létre:
      ```
      E: 5, X: 4, A: 3, M: 2, P: 1, L: 0
      ```
      Ez azt jelenti, hogy például ha az "E" karakter nem egyezik, akkor 5 pozícióval lépünk előre, mivel az "E" legutolsó előfordulása 5 pozícióra van a minta végétől.
    - A "good suffix" táblázatot a következőképpen hozzuk létre:
      ```
      E: 1, LE: 2, PLE: 3, MPLE: 4, AMPLE: 5, XAMPLE: 6, EXAMPLE: 7
      ```
      Ez azt jelenti, hogy ha például a "LE" szuffix nem egyezik, akkor 2 pozícióval lépünk előre.

2. **Keresési folyamat:**
    - Kezdjük az összehasonlítást a minta végéről:
      ```
      Szöveg:  HERE IS A SIMPLE EXAMPLE
      Minta:                   EXAMPLE
      ```
    - A "E" és "E" egyezik, "L" és "L" egyezik, stb. Amíg el nem érjük az első eltérést.
    - Ha eltérést találunk, alkalmazzuk a "bad character" és "good suffix" heurisztikákat, hogy meghatározzuk az előrelépés mértékét.
    - Folytatjuk az összehasonlítást az új pozícióból.

#### C++ Implementáció

Az alábbiakban bemutatunk egy egyszerű C++ implementációt a Boyer-Moore algoritmusra:

```cpp
#include <iostream>

#include <vector>
#include <string>

#include <unordered_map>
using namespace std;

void preprocessBadCharacterHeuristic(const string &pattern, unordered_map<char, int> &badCharTable) {
    int m = pattern.size();
    for (int i = 0; i < m - 1; ++i) {
        badCharTable[pattern[i]] = m - i - 1;
    }
}

void preprocessGoodSuffixHeuristic(const string &pattern, vector<int> &goodSuffixTable) {
    int m = pattern.size();
    vector<int> suffix(m, 0);
    suffix[m - 1] = m;
    int g = m - 1;
    for (int i = m - 2; i >= 0; --i) {
        if (i > g && suffix[i + m - 1 - f] < i - g) {
            suffix[i] = suffix[i + m - 1 - f];
        } else {
            if (i < g) g = i;
            f = i;
            while (g >= 0 && pattern[g] == pattern[g + m - 1 - f]) {
                --g;
            }
            suffix[i] = f - g;
        }
    }
    for (int i = 0; i < m; ++i) {
        goodSuffixTable[i] = m;
    }
    int j = 0;
    for (int i = m - 1; i >= 0; --i) {
        if (suffix[i] == i + 1) {
            for (; j < m - 1 - i; ++j) {
                if (goodSuffixTable[j] == m) {
                    goodSuffixTable[j] = m - 1 - i;
                }
            }
        }
    }
    for (int i = 0; i <= m - 2; ++i) {
        goodSuffixTable[m - 1 - suffix[i]] = m - 1 - i;
    }
}

vector<int> boyerMooreSearch(const string &text, const string &pattern) {
    int n = text.size();
    int m = pattern.size();
    unordered_map<char, int> badCharTable;
    preprocessBadCharacterHeuristic(pattern, badCharTable);
    vector<int> goodSuffixTable(m);
    preprocessGoodSuffixHeuristic(pattern, goodSuffixTable);
    vector<int> result;
    int s = 0;
    while (s <= n - m) {
        int j = m - 1;
        while (j >= 0 && pattern[j] == text[s + j]) {
            --j;
        }
        if (j < 0) {
            result.push_back(s);
            s += goodSuffixTable[0];
        } else {
            s += max(goodSuffixTable[j], badCharTable.find(text[s + j]) != badCharTable.end() ? badCharTable[text[s + j]] : m);
        }
    }
    return result;
}

int main() {
    string text = "HERE IS A SIMPLE EXAMPLE";
    string pattern = "EXAMPLE";
    vector<int> result = boyerMooreSearch(text, pattern);
    for (int pos : result) {
        cout << "Pattern found at index: " << pos << endl;
    }
    return 0;
}
```

Ez az implementáció először előállítja a "bad character" és a "good suffix" táblázatokat a minta előfeldolgozásával. Ezután a szövegben végrehajtja a keresést a Boyer-Moore algoritmus segítségével, és visszaadja az összes találat pozícióját.

#### Összegzés

A Boyer-Moore algoritmus hatékony és széles körben alkalmazott módszer a sztring keresés területén. A két heurisztika (Bad Character és Good Suffix) lehetővé teszi a szöveg gyors átvizsgálását, különösen hosszú minták és szövegek esetén. Az algoritmus erősségei közé tartozik a gyors futási idő a legtöbb gyakorlati esetben, míg gyengesége lehet, hogy bizonyos specifikus minták esetén az átlagos futási idő közelíthet a naiv algoritmuséhoz. Mindazonáltal a Boyer-Moore algoritmus az egyik legfontosabb eszköz a hatékony sztring keresés eszköztárában.

### 1.9.2. Knuth-Morris-Pratt (KMP) algoritmus

A Knuth-Morris-Pratt (KMP) algoritmus egy hatékony sztring keresési algoritmus, amelyet Donald Knuth, Vaughan Pratt és James H. Morris fejlesztettek ki 1977-ben. Az algoritmus fő célja, hogy minimalizálja a szükséges összehasonlítások számát a szöveg és a minta között azáltal, hogy kihasználja az előző összehasonlítások eredményeit. Ez az algoritmus különösen akkor hasznos, amikor sok mintaillesztést kell végrehajtani nagy szövegállományokon.

#### Az algoritmus alapjai

A KMP algoritmus két fő szakaszra osztható: az előfeldolgozásra és a keresési folyamatra. Az előfeldolgozás során a minta egy "prefix-funkció" nevű táblázatát hozzuk létre, amely segít meghatározni, hogy egy adott karakter nem egyezése esetén mennyivel léphetünk előre a szövegben anélkül, hogy újra meg kellene vizsgálni az előzőleg összehasonlított karaktereket.

**Prefix-funkció:**
A prefix-funkció ($\pi$) egy olyan táblázat, amely minden pozícióhoz (indexhez) egy értéket rendel, jelezve, hogy a minta adott pozíciójáig tartó részének legnagyobb prefixe, amely egyben szuffix is, milyen hosszú. Ez a táblázat segít abban, hogy ha a minta egy karaktere nem egyezik a szöveg megfelelő karakterével, akkor a minta melyik pozíciójából kell folytatni az összehasonlítást.

#### Az algoritmus lépései

1. **Előfeldolgozás:**
    - Készítünk egy prefix-funkció táblázatot a minta számára. Ez a táblázat meghatározza, hogy a minta adott pozíciójáig tartó részének legnagyobb prefixe, amely egyben szuffix is, milyen hosszú.

2. **Keresési folyamat:**
    - A szöveg első karakterétől kezdve összehasonlítjuk a minta karaktereit a szöveg megfelelő karaktereivel.
    - Ha egy karakter nem egyezik, a prefix-funkció táblázat segítségével meghatározzuk, hogy mennyivel léphetünk előre a mintában, anélkül, hogy újra meg kellene vizsgálnunk az előzőleg összehasonlított karaktereket.
    - Folytatjuk az összehasonlítást az új pozícióból.

#### Részletes magyarázat példával

Vegyünk egy példát, hogy bemutassuk az algoritmus működését. Tegyük fel, hogy a minta a "ABABAC" és a szöveg az "ABABABCABABABCABABABC".

1. **Előfeldolgozás:**
    - Készítünk egy prefix-funkció táblázatot a minta számára:
      ```
      Pozíció:  0 1 2 3 4 5
      Minta:    A B A B A C
      pi érték:  0 0 1 2 3 0
      ```
      Ez azt jelenti, hogy például a 4. pozícióig (nulla indextől) tartó rész (ABABA) legnagyobb prefixe, amely egyben szuffix is, hossza 3 (ABA).

2. **Keresési folyamat:**
    - Kezdjük az összehasonlítást a szöveg első karakterétől:
      ```
      Szöveg:  ABABABCABABABCABABABC
      Minta:   ABABAC
      ```
    - Az első hat karakter összehasonlítása után az "ABABAC" minta nem egyezik az "ABABAB" szövegrésszel.
    - A prefix-funkció táblázat segítségével megállapítjuk, hogy az 5. pozícióban lévő "C" nem egyezik az "B"-vel, így 3 pozícióval lépünk előre a mintában a $\pi$ táblázat alapján.
    - Folytatjuk az összehasonlítást a szöveg 3. pozíciójától kezdve.
    - Ezt a folyamatot addig folytatjuk, amíg a teljes szöveget át nem vizsgáltuk.

#### Prefix-funkció előállítása

A prefix-funkció előállításához a következő lépéseket követjük:

1. Kezdjük az első karakterrel, amely mindig 0 értékű lesz.
2. Haladunk végig a mintán, és minden pozíciónál meghatározzuk a leghosszabb prefixet, amely egyben szuffix is.
3. Ha találunk egy karaktert, amely megegyezik a minta előző részének karaktereivel, növeljük az értéket.
4. Ha nem találunk egyezést, az érték 0 lesz.

#### C++ Implementáció

Az alábbiakban bemutatunk egy egyszerű C++ implementációt a KMP algoritmusra:

```cpp
#include <iostream>

#include <vector>
#include <string>

using namespace std;

void computePrefixFunction(const string &pattern, vector<int> &prefix) {
    int m = pattern.size();
    int k = 0;
    prefix[0] = 0;
    for (int i = 1; i < m; ++i) {
        while (k > 0 && pattern[k] != pattern[i]) {
            k = prefix[k - 1];
        }
        if (pattern[k] == pattern[i]) {
            ++k;
        }
        prefix[i] = k;
    }
}

vector<int> KMPsearch(const string &text, const string &pattern) {
    int n = text.size();
    int m = pattern.size();
    vector<int> prefix(m);
    computePrefixFunction(pattern, prefix);
    vector<int> result;
    int q = 0;
    for (int i = 0; i < n; ++i) {
        while (q > 0 && pattern[q] != text[i]) {
            q = prefix[q - 1];
        }
        if (pattern[q] == text[i]) {
            ++q;
        }
        if (q == m) {
            result.push_back(i - m + 1);
            q = prefix[q - 1];
        }
    }
    return result;
}

int main() {
    string text = "ABABABCABABABCABABABC";
    string pattern = "ABABAC";
    vector<int> result = KMPsearch(text, pattern);
    for (int pos : result) {
        cout << "Pattern found at index: " << pos << endl;
    }
    return 0;
}
```

Ez az implementáció először előállítja a prefix-funkció táblázatot a minta előfeldolgozásával. Ezután a szövegben végrehajtja a keresést a KMP algoritmus segítségével, és visszaadja az összes találat pozícióját.

#### Előnyök és hátrányok

**Előnyök:**
- A KMP algoritmus garantált lineáris időben fut, ami azt jelenti, hogy a legrosszabb esetben is O(n + m) lépésszámot igényel, ahol n a szöveg hossza és m a minta hossza.
- Az előfeldolgozási lépések egyszerűek és gyorsak, ami lehetővé teszi az algoritmus hatékony alkalmazását nagy szövegekben is.

**Hátrányok:**
- Az algoritmus bonyolultabb, mint a naiv keresési algoritmusok, ami megnehezítheti a megértést és az implementálást.
- A prefix-funkció táblázat előállítása bizonyos esetekben további memóriát igényelhet, ami problémát jelenthet nagyon nagy minták esetén.

#### Összegzés

A Knuth-Morris-Pratt algoritmus egy rendkívül hatékony és széles körben alkalmazott sztring keresési algoritmus, amely minimalizálja a szükséges összehasonlítások számát a szöveg és a minta között. Az algoritmus előfeldolgozási lépéseinek köszönhetően képes a korábbi összehasonlítások eredményeit felhasználva gyorsítani a keresési folyamatot, ami különösen nagy szövegek és gyakori mintaillesztés esetén előnyös. A KMP algoritmus méltán vált az egyik alapvető eszközzé a sztring keresés területén, és számos gyakorlati alkalmazása bizonyítja hatékonyságát és megbízhatóságát.

### 1.9.3. Rabin-Karp algoritmus

A Rabin-Karp algoritmus egy hatékony és széles körben alkalmazott sztring keresési algoritmus, amelyet Richard M. Karp és Michael O. Rabin fejlesztett ki 1987-ben. Az algoritmus különlegessége, hogy hashelési technikát alkalmaz a minta és a szöveg szakaszainak gyors összehasonlítására, ami különösen hasznos többszörös mintaillesztés és nagy adatállományok esetén.

#### Az algoritmus alapjai

A Rabin-Karp algoritmus fő gondolata az, hogy a minta és a szöveg azonos hosszúságú részszövegeit (ablakait) hasheljük, és az összehasonlítást a hashek segítségével végezzük el. Ha a hashek egyeznek, akkor további ellenőrzést végzünk a tényleges karakterek összehasonlításával, hogy elkerüljük a hashelési ütközésekből adódó hamis pozitív találatokat.

**Hash-függvény:**
A hashelés lényege, hogy a szöveg egy szakaszát egyetlen számként reprezentáljuk. A Rabin-Karp algoritmusban gyakran egy gördülő hash-függvényt használnak, amely lehetővé teszi, hogy az egyik ablak hash-értékéből gyorsan kiszámítsuk a következő ablak hash-értékét.

A leggyakrabban alkalmazott hash-függvény a következő formát ölti:
$\text{hash}(S) = (S[0] \cdot d^{m-1} + S[1] \cdot d^{m-2} + \ldots + S[m-1]) \mod q$
ahol:
- $S$ a sztring,
- $m$ a minta hossza,
- $d$ az alkalmazott alap (általában a karakterkészlet mérete, pl. 256 az ASCII karakterkészlet esetén),
- $q$ egy nagy prímszám, amely a hashelési ütközések minimalizálására szolgál.

**Gördülő hash:**
A gördülő hash-függvény lehetővé teszi, hogy a következő ablak hash-értékét az előző ablak hash-értékéből számoljuk ki, csökkentve ezzel a számítási komplexitást. Az új hash-értéket az alábbi képlet segítségével határozhatjuk meg:
$\text{hash}_{\text{new}} = (d \cdot (\text{hash}_{\text{old}} - S[i-1] \cdot d^{m-1}) + S[i+m-1]) \mod q$

#### Az algoritmus lépései

1. **Előfeldolgozás:**
    - Számítsuk ki a minta hash-értékét a fenti hash-függvény segítségével.
    - Számítsuk ki a szöveg első ablakának hash-értékét, amely ugyanakkora hosszúságú, mint a minta.

2. **Keresési folyamat:**
    - Hasonlítsuk össze a minta hash-értékét a szöveg aktuális ablakának hash-értékével.
    - Ha a hash-értékek egyeznek, végezzünk karakterenkénti összehasonlítást az ablak és a minta között, hogy megerősítsük a találatot.
    - Ha egyezést találunk, jegyezzük fel a találat helyét.
    - Gördítsük az ablakot egy karakterrel jobbra, és számítsuk ki az új ablak hash-értékét a gördülő hash-függvény segítségével.
    - Folytassuk a folyamatot a szöveg végéig.

#### Részletes magyarázat példával

Vegyünk egy példát, hogy bemutassuk az algoritmus működését. Tegyük fel, hogy a minta a "ABC" és a szöveg az "ABABDABC".

1. **Előfeldolgozás:**
    - Válasszunk egy megfelelő értéket az alap (d) és a prímszám (q) számára, például d = 256 és q = 101.
    - Számítsuk ki a minta hash-értékét:
      ```
      hash("ABC") = (65 * 256^2 + 66 * 256^1 + 67) mod 101 = 4
      ```
    - Számítsuk ki a szöveg első ablakának hash-értékét (hossz 3):
      ```
      hash("ABA") = (65 * 256^2 + 66 * 256^1 + 65) mod 101 = 1
      ```

2. **Keresési folyamat:**
    - Hasonlítsuk össze a minta hash-értékét (4) a szöveg aktuális ablakának hash-értékével (1). Mivel nem egyeznek, gördítsük az ablakot jobbra.
    - Számítsuk ki az új ablak hash-értékét:
      ```
      hash("BAB") = (256 * (1 - 65 * 256^2) + 66) mod 101 = 90
      ```
    - Hasonlítsuk össze a minta hash-értékét (4) az új ablak hash-értékével (90). Mivel nem egyeznek, gördítsük az ablakot tovább.
    - Folytassuk a folyamatot a szöveg végéig.

#### C++ Implementáció

Az alábbiakban bemutatunk egy egyszerű C++ implementációt a Rabin-Karp algoritmusra:

```cpp
#include <iostream>

#include <string>
using namespace std;

#define d 256

void search(string pat, string txt, int q) {
    int M = pat.size();
    int N = txt.size();
    int i, j;
    int p = 0; // hash value for pattern
    int t = 0; // hash value for text
    int h = 1;

    // The value of h would be "pow(d, M-1)%q"
    for (i = 0; i < M - 1; i++)
        h = (h * d) % q;

    // Calculate the hash value of pattern and first window of text
    for (i = 0; i < M; i++) {
        p = (d * p + pat[i]) % q;
        t = (d * t + txt[i]) % q;
    }

    // Slide the pattern over text one by one
    for (i = 0; i <= N - M; i++) {
        // Check the hash values of current window of text and pattern.
        // If the hash values match then only check for characters one by one
        if (p == t) {
            // Check for characters one by one
            for (j = 0; j < M; j++) {
                if (txt[i + j] != pat[j])
                    break;
            }

            // if p == t and pat[0...M-1] = txt[i, i+1, ...i+M-1]
            if (j == M)
                cout << "Pattern found at index " << i << endl;
        }

        // Calculate hash value for next window of text: Remove leading digit,
        // add trailing digit
        if (i < N - M) {
            t = (d * (t - txt[i] * h) + txt[i + M]) % q;

            // We might get negative value of t, converting it to positive
            if (t < 0)
                t = (t + q);
        }
    }
}

int main() {
    string txt = "ABABDABC";
    string pat = "ABC";
    int q = 101; // A prime number
    search(pat, txt, q);
    return 0;
}
```

Ez az implementáció először előállítja a minta és a szöveg első ablakának hash-értékeit, majd végrehajtja a keresést a Rabin-Karp algoritmus segítségével, és visszaadja az összes találat pozícióját.

#### Előnyök és hátrányok

**Előnyök:**
- A Rabin-Karp algoritmus nagyon hatékony, ha egyszerre több minta keresése szükséges, mivel minden minta hash-értéke előre kiszámítható és összehasonlítható a szöveg hash-értékeivel.
- Az algoritmus egyszerű és könnyen implementálható.

**Hátrányok:**
- Az algoritmus érzékeny a hashelési ütközésekre, amelyek hamis pozitív találatokat eredményezhetnek, és ezek további karakterenkénti összehasonlítást igényelnek.
- A legrosszabb esetben az algoritmus futási ideje $O(nm)$, ahol n a szöveg hossza és m a minta hossza, ami előfordulhat, ha sok a hashelési ütközés.

#### Összegzés

A Rabin-Karp algoritmus egy hatékony és sokoldalú sztring keresési algoritmus, amely hashelési technikát alkalmaz a minta és a szöveg szakaszainak gyors összehasonlítására. Az algoritmus különösen hasznos, ha egyszerre több minta keresése szükséges, mivel minden minta hash-értéke előre kiszámítható és gyorsan összehasonlítható a szöveg hash-értékeivel. Bár az algoritmus érzékeny a hashelési ütközésekre, megfelelő választott hash-függvénnyel és nagy prímszámmal a legtöbb gyakorlati esetben hatékonyan alkalmazható.

### 1.9.4. Sztring keresés Trie adatszerkezetekben

A Trie adatszerkezet, más néven prefix fa, egy speciális fa adatszerkezet, amelyet elsősorban sztring keresési feladatokra használnak. A Trie hatékonyan támogatja az előtag-alapú kereséseket és sztring tárolásokat, lehetővé téve a gyors keresést, beszúrást és törlést. Ezt az adatszerkezetet leggyakrabban szótárakban, automatikus kiegészítésben és egyéb alkalmazásokban használják, ahol nagy mennyiségű sztring kezelésére van szükség.

#### Az adatszerkezet alapjai

A Trie egy fa alapú adatszerkezet, ahol minden csomópont egy karaktert képvisel, és a gyökértől egy levélig vezető út egy sztringet határoz meg. A Trie különleges tulajdonsága, hogy a közös előtagok egyetlen ágat használnak, ami jelentős memóriatakarékosságot eredményezhet.

**Struktúra:**
- **Gyökér:** A Trie gyökércsomópontja nem tartalmaz karaktert, hanem a gyermek csomópontokba vezet, amelyek az összes lehetséges kezdő karaktert tartalmazzák.
- **Belső csomópontok:** Minden belső csomópont egy karaktert és egy vagy több gyermek csomópontot tartalmaz.
- **Levélek:** A levélcsomópontok sztringek végét jelzik, és gyakran tartalmaznak egy jelzőt, hogy a fa adott pontján véget ér egy sztring.

#### Alapműveletek a Trie-ban

1. **Beszúrás:**
   A sztring beszúrása során a Trie gyökerétől indulva haladunk lefelé, karakterenként. Ha egy karakterhez tartozó csomópont nem létezik, létrehozzuk azt. Az utolsó karakterhez érve egy jelzőt helyezünk el, amely jelzi a sztring végét.

2. **Keresés:**
   A keresési művelet hasonló a beszúráshoz. A gyökértől indulva karakterenként haladunk lefelé. Ha elérjük a sztring végét és a megfelelő jelzőt találjuk, a sztring létezik a Trie-ban. Ha bármelyik karakter nem található, a keresés sikertelen.

3. **Törlés:**
   A törlési művelet bonyolultabb, mivel gondoskodni kell arról, hogy ne hagyjunk felesleges csomópontokat a fában. A törlés során visszafelé haladva töröljük a csomópontokat, ha azok nem részét képezik más sztringeknek.

#### Előtag keresés Trie-ban

Az előtag keresés különösen hatékony a Trie-ban. Az előtag keresés során a gyökértől indulva haladunk lefelé az előtag karakterei szerint. Ha az előtag utolsó karakteréig elérünk, akkor minden ebből a pontból elérhető levél egy olyan sztringet jelöl, amely az adott előtaggal kezdődik.

#### Példa a Trie-ra

Tekintsük az alábbi sztringeket: "CAT", "CAR", "DOG", "DO". Az ezekből a sztringekből felépített Trie a következőképpen néz ki:

```
         (root)
         /  |  \
        C   D   O
       /|    \   
      A R     O
     /      /  
    T      G
            \
             (end)
```

1. **Beszúrás:**
    - "CAT" beszúrása: C → A → T
    - "CAR" beszúrása: C → A → R
    - "DOG" beszúrása: D → O → G
    - "DO" beszúrása: D → O

2. **Keresés:**
    - "CAT" keresése: C → A → T → (end)
    - "DOG" keresése: D → O → G → (end)
    - "COW" keresése: C → O → W (nem található)

3. **Előtag keresés:**
    - "DO" előtag keresése: D → O (minden további csomópont és levél a "DO"-val kezdődő sztringeket jelöli)

#### C++ Implementáció

Az alábbiakban bemutatunk egy egyszerű C++ implementációt a Trie adatszerkezetre, amely tartalmazza a beszúrás, keresés és előtag keresés műveleteket:

```cpp
#include <iostream>

#include <unordered_map>
using namespace std;

class TrieNode {
public:
    unordered_map<char, TrieNode*> children;
    bool isEndOfWord;
    
    TrieNode() : isEndOfWord(false) {}
};

class Trie {
private:
    TrieNode* root;
    
public:
    Trie() {
        root = new TrieNode();
    }
    
    void insert(const string &word) {
        TrieNode* node = root;
        for (char ch : word) {
            if (node->children.find(ch) == node->children.end()) {
                node->children[ch] = new TrieNode();
            }
            node = node->children[ch];
        }
        node->isEndOfWord = true;
    }
    
    bool search(const string &word) {
        TrieNode* node = root;
        for (char ch : word) {
            if (node->children.find(ch) == node->children.end()) {
                return false;
            }
            node = node->children[ch];
        }
        return node->isEndOfWord;
    }
    
    bool startsWith(const string &prefix) {
        TrieNode* node = root;
        for (char ch : prefix) {
            if (node->children.find(ch) == node->children.end()) {
                return false;
            }
            node = node->children[ch];
        }
        return true;
    }
};

int main() {
    Trie trie;
    trie.insert("CAT");
    trie.insert("CAR");
    trie.insert("DOG");
    trie.insert("DO");
    
    cout << "Search for 'CAT': " << (trie.search("CAT") ? "Found" : "Not Found") << endl;
    cout << "Search for 'CAR': " << (trie.search("CAR") ? "Found" : "Not Found") << endl;
    cout << "Search for 'COW': " << (trie.search("COW") ? "Found" : "Not Found") << endl;
    cout << "Starts with 'DO': " << (trie.startsWith("DO") ? "Yes" : "No") << endl;
    
    return 0;
}
```

#### Előnyök és hátrányok

**Előnyök:**
- A Trie adatszerkezet hatékonyan kezeli az előtag kereséseket és a sztringek gyors beszúrását.
- Képes tárolni és kezelni nagy mennyiségű sztringet memóriahatékony módon, különösen akkor, ha sok sztring osztozik közös előtagon.
- Alkalmas szótárak és nyelvi feldolgozási alkalmazások számára.

**Hátrányok:**
- A Trie memóriakövetelménye nagy lehet, különösen akkor, ha a sztringek sok különböző karaktert tartalmaznak és kevés közös előtagot osztanak meg.
- A Trie-ban történő keresési műveletek időigényesebbek lehetnek, ha a fa mély és sok csomópontot tartalmaz.

#### Optimális alkalmazások

A Trie adatszerkezet különösen hasznos a következő alkalmazásokban:
- **Szótárak és keresőmotorok:** Gyors előtag keresés és automatikus kiegészítés.
- **Szövegfeldolgozás:** Szöveges adatok hatékony kezelése és keresése.
- **DNS szekvencia elemzés:** Nagy mennyiségű génszekvencia gyors keresése és tárolása.
- **Fordítóprogramok és nyelvi elemzők:** Tokenek és kulcsszavak gyors keresése és osztályozása.

#### Összegzés

A Trie adatszerkezet egy erőteljes eszköz a sztring keresési feladatok hatékony megoldására. Az előtag alapú keresések támogatása és a sztringek közös előtagjainak kihasználása révén a Trie képes gyorsan és hatékonyan kezelni nagy mennyiségű sztringet. Az alkalmazások széles skáláján bizonyítja hasznosságát, különösen olyan területeken, ahol a gyors keresési műveletek és a memóriahatékonyság kiemelt fontosságúak. Az algoritmusok és adatszerkezetek területén a Trie alapvető eszközként szolgál, és jelentős szerepet játszik a sztring keresési problémák megoldásában.