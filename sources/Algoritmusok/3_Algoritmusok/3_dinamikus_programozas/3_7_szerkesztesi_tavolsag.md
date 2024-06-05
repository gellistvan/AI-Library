## 3.6. Szerkesztési távolság (Edit Distance)

A szerkesztési távolság az algoritmusok egyik alapvető problémája, amely a két karakterlánc közötti különbség mérésére szolgál. Az ilyen típusú problémák megoldása kulcsfontosságú számos területen, beleértve a szövegfeldolgozást, a genomikai elemzést és a gépi fordítást. A leggyakrabban használt mérőszámok közé tartozik a Levenshtein-távolság, amely a minimális szerkesztési műveletek számát jelenti, amelyek szükségesek az egyik karakterlánc másikká alakításához. Ebben a fejezetben megvizsgáljuk a Levenshtein-távolság meghatározásának alapelveit, az alkalmazási területeit és példákat mutatunk be. Továbbá, részletesen tárgyaljuk a probléma megoldását rekurzív megközelítéssel, majd bemutatjuk, hogyan lehet hatékonyabb algoritmusokat kialakítani dinamikus programozás alkalmazásával.

### 3.6.1. Levenshtein távolság

A Levenshtein-távolság, más néven szerkesztési távolság, egy fontos mérőszám a karakterláncok összehasonlításában. Az algoritmus nevét Vlagyimir Levenshtein-ről kapta, aki 1965-ben vezette be ezt a fogalmat. A Levenshtein-távolság két sztring közötti különbséget méri azzal, hogy megszámolja a szükséges szerkesztési műveletek (beszúrás, törlés, csere) minimális számát, amelyek az egyik sztringet a másikká alakítják. Ez az alapfogalom számos alkalmazási területen hasznos, beleértve a szövegfeldolgozást, bioinformatikát, gépi fordítást és még sok mást.

#### Matematikai Meghatározás

A Levenshtein-távolság $d(s, t)$ két sztring, $s$ és $t$ között a következő módon definiálható:

1. **Beszúrás (Insert):** Egy karakter beszúrása az egyik sztringbe.
2. **Törlés (Delete):** Egy karakter törlése az egyik sztringből.
3. **Csere (Substitution):** Egy karakter kicserélése egy másik karakterre.

Formálisan, ha $|s|$ és $|t|$ az $s$ és $t$ hossza, akkor $d(s, t)$ meghatározható a következő rekurzív képlettel:

$$
d(i, j) =
\begin{cases}
j & \text{ha } i = 0 \\
i & \text{ha } j = 0 \\
d(i-1, j) + 1 & \text{ha } i, j > 0 \text{ és } s[i] \neq t[j] \\
d(i, j-1) + 1 & \text{ha } i, j > 0 \text{ és } s[i] \neq t[j] \\
d(i-1, j-1) + 1 & \text{ha } i, j > 0 \text{ és } s[i] \neq t[j] \\
d(i-1, j-1) & \text{ha } s[i] = t[j]
\end{cases}
$$

#### Rekurzív Megközelítés

A Levenshtein-távolság kiszámításának egyik legegyszerűbb módja a rekurzió használata. Azonban ez a módszer nem hatékony, mivel ugyanazokat a számításokat többször is elvégzi, ami exponenciális időbonyolultsághoz vezet.

```cpp
int levenshtein_recursive(const std::string &s, const std::string &t, int i, int j) {
    if (i == 0) return j;
    if (j == 0) return i;
    
    int cost = (s[i - 1] == t[j - 1]) ? 0 : 1;
    
    return std::min({
        levenshtein_recursive(s, t, i - 1, j) + 1, // Törlés
        levenshtein_recursive(s, t, i, j - 1) + 1, // Beszúrás
        levenshtein_recursive(s, t, i - 1, j - 1) + cost // Csere
    });
}
```

#### Dinamikus Programozás

A dinamikus programozás jelentős mértékben javítja a Levenshtein-távolság számításának hatékonyságát azáltal, hogy tárolja az alproblémák megoldásait és újra felhasználja azokat. Ezzel az optimalizálással a számítás lineáris időben történik, ami lényegesen gyorsabb a rekurzív megoldásnál.

##### Algoritmus Lépései

1. **Táblázat Felépítése:** Készítsünk egy $(m+1) \times (n+1)$ méretű mátrixot, ahol $m$ és $n$ az $s$ és $t$ hosszai.
2. **Inicializálás:** A mátrix első sorát és oszlopát inicializáljuk a szegélyfeltételeknek megfelelően.
3. **Kitöltés:** Töltsük ki a mátrixot alulról felfelé és balról jobbra, a fenti rekurzív képlet alapján.
4. **Visszakövetés:** Az utolsó cellában lévő érték adja a Levenshtein-távolságot.

### 3.6.2. Megoldás bemutatása rekurzióval

A rekurzív megközelítés alapvetően természetes módon adódik a Levenshtein-távolság meghatározására, mivel a probléma természeténél fogva kisebb alproblémákra bontható. A rekurzív megközelítés lényege, hogy a nagyobb problémát kisebb, könnyebben kezelhető részekre bontja, amelyeket aztán összegezve adja a teljes megoldást.

#### Rekurzív Definíció

Mint korábban említettük, a Levenshtein-távolság $d(s, t)$ két sztring $s$ és $t$ között a minimális számú szerkesztési művelet, amely az egyik sztringet a másikká alakítja. Az $s$ és $t$ sztringek közötti Levenshtein-távolság rekurzív definíciója:

$$
d(i, j) =
\begin{cases}
j & \text{ha } i = 0 \\
i & \text{ha } j = 0 \\
\min \left\{
\begin{array}{ll}
d(i-1, j) + 1, & \text{(törlés)} \\
d(i, j-1) + 1, & \text{(beszúrás)} \\
d(i-1, j-1) + \text{cost}(i, j), & \text{(csere vagy egyezés)}
\end{array}
\right. & \text{egyébként}
\end{cases}
$$

A fenti képletben a $\text{cost}(i, j)$ a csere művelet költsége, amely 0, ha az $i$-edik karakter az $s$ sztringben megegyezik a $j$-edik karakterrel a $t$ sztringben, különben 1.

#### Rekurzív Algoritmus

A rekurzív algoritmus közvetlenül követi a fenti definíciót. Az algoritmus minden lépésben három lehetőséget mérlegel:

1. **Törlés:** Eltávolítjuk az aktuális karaktert az egyik sztringből, és így a probléma az egyik sztring rövidített verziójára csökken.
2. **Beszúrás:** Beszúrunk egy karaktert a másik sztringbe, és így a probléma az egyik sztring hosszabbított verziójára csökken.
3. **Csere vagy Egyezés:** Megvizsgáljuk az aktuális karaktereket mindkét sztringben. Ha megegyeznek, nincs költség, különben van.

##### Példa Kód C++ Nyelven

Az alábbiakban egy példa a Levenshtein-távolság rekurzív meghatározására C++ nyelven:

```cpp
int levenshtein_recursive(const std::string &s, const std::string &t, int i, int j) {
    // Base cases
    if (i == 0) return j; // If s is empty, the cost is the length of t
    if (j == 0) return i; // If t is empty, the cost is the length of s
    
    // Cost of substituting s[i-1] with t[j-1]
    int cost = (s[i - 1] == t[j - 1]) ? 0 : 1;
    
    // Recur for the three operations
    return std::min({
        levenshtein_recursive(s, t, i - 1, j) + 1, // Deletion
        levenshtein_recursive(s, t, i, j - 1) + 1, // Insertion
        levenshtein_recursive(s, t, i - 1, j - 1) + cost // Substitution or match
    });
}
```

Ez a függvény a $s$ és $t$ sztringek minden lehetséges állapotára újra és újra meghívja önmagát, amíg el nem éri az alap eseteket. Minden hívásban három másik hívást végez, ami azt eredményezi, hogy a teljes időbonyolultság $O(3^{\min(m, n)})$, ahol $m$ és $n$ a sztringek hossza. Ez a komplexitás rendkívül hatékonytalan nagyobb sztringek esetében, mivel sok redundáns számítást végez.

#### Hátrányok és Optimalizálási Lehetőségek

A rekurzív megoldás legnagyobb hátránya, hogy sok redundáns számítást végez. Ugyanazokat az alproblémákat többször is kiszámítja, ami jelentős időbeli pazarlást eredményez. Például, ha $s = "kitten"$ és $t = "sitting"$, a rekurzív megközelítés többször is kiszámítja az olyan alproblémákat, mint $d("kitte", "sittin")$ vagy $d("kitt", "sitt")$.

##### Memorization

A rekurzív algoritmus hatékonyságának javítása érdekében alkalmazható a memorizáció, amely egyfajta dinamikus programozás. Ebben a megközelítésben egy táblázatot használunk a már kiszámított alproblémák eredményeinek tárolására, így azokat nem kell újra kiszámítani. A memorizáció használatával a rekurzív algoritmus időbonyolultsága $O(m \cdot n)$-re csökken, ami sokkal hatékonyabb a tisztán rekurzív megközelítésnél.

```cpp
int levenshtein_recursive_memo(const std::string &s, const std::string &t, int i, int j, std::vector<std::vector<int>> &memo) {
    // Check if result is already computed
    if (memo[i][j] != -1) return memo[i][j];
    
    if (i == 0) return j;
    if (j == 0) return i;
    
    int cost = (s[i - 1] == t[j - 1]) ? 0 : 1;
    
    memo[i][j] = std::min({
        levenshtein_recursive_memo(s, t, i - 1, j, memo) + 1,
        levenshtein_recursive_memo(s, t, i, j - 1, memo) + 1,
        levenshtein_recursive_memo(s, t, i - 1, j - 1, memo) + cost
    });
    
    return memo[i][j];
}
```

Ebben a változatban egy $(m+1) \times (n+1)$ méretű mátrixot használunk a memoizáláshoz, ahol $m$ és $n$ az $s$ és $t$ sztringek hosszai. A memo mátrix inicializálásakor minden értéket -1-re állítunk, jelezve, hogy az adott alprobléma még nincs kiszámítva. A memoizált rekurzív függvény minden hívásnál ellenőrzi, hogy az eredmény már el van-e mentve, és ha igen, akkor azt visszaadja, így elkerüli a redundáns számításokat.

#### Összefoglalás

A rekurzív megoldás egy intuitív megközelítés a Levenshtein-távolság kiszámítására, amely a probléma természetes alstruktúráit használja ki. Azonban a tisztán rekurzív megközelítés nem hatékony nagy sztringek esetében, mivel sok redundáns számítást végez. A memorizáció alkalmazásával jelentősen javítható az algoritmus hatékonysága, mivel elkerülhetővé válnak az ismétlődő számítások.

A következő fejezetben bemutatjuk a dinamikus programozás módszerét, amely tovább optimalizálja a Levenshtein-távolság számítását, és részletesen megvizsgáljuk ennek előnyeit és implementációs részleteit.

### 3.6.3. Megoldás bemutatása dinamikus programozással

A dinamikus programozás egy hatékony módszer a Levenshtein-távolság kiszámítására, amely jelentősen csökkenti az időbonyolultságot a rekurzív megközelítéshez képest. A dinamikus programozás lényege, hogy a problémát kisebb, átfedő alproblémákra bontja, és azokat táblázatos formában tárolja, így elkerülve az ismétlődő számításokat. Ebben a fejezetben részletesen bemutatjuk, hogyan alkalmazható a dinamikus programozás a Levenshtein-távolság kiszámítására, és elemezzük ennek a megközelítésnek az előnyeit.

#### Dinamikus Programozás Elve

A dinamikus programozás alapötlete az, hogy egy táblázatban (mátrixban) tároljuk a részproblémák megoldásait, és ezeket újra felhasználjuk a nagyobb probléma megoldása során. A Levenshtein-távolság esetében egy $(m+1) \times (n+1)$ méretű mátrixot hozunk létre, ahol $m$ és $n$ az összehasonlítandó sztringek hosszai. Minden mátrix cella $dp[i][j]$ tartalmazza az $s[0:i]$ és $t[0:j]$ részsztringek közötti szerkesztési távolságot.

#### Algoritmus Lépései

1. **Mátrix Inicializálása:** Kezdetben inicializáljuk a mátrix első sorát és oszlopát. Az első sorban az értékek az üres sztring és az $t$ részsztringek közötti távolságot mutatják (ami az összes karakter beszúrásának költsége), az első oszlopban pedig az $s$ részsztringek és az üres sztring közötti távolságot (ami az összes karakter törlésének költsége).

2. **Mátrix Kitöltése:** A mátrixot alulról felfelé és balról jobbra haladva töltjük ki. Minden cella értéke a három lehetséges szerkesztési művelet minimuma:
    - **Törlés:** $dp[i-1][j] + 1$
    - **Beszúrás:** $dp[i][j-1] + 1$
    - **Csere:** $dp[i-1][j-1] + \text{cost}(i, j)$, ahol a $\text{cost}(i, j)$ 0, ha az $s[i-1]$ és $t[j-1]$ karakterek megegyeznek, különben 1.

3. **Eredmény Kiolvasása:** A mátrix utolsó cellája, $dp[m][n]$, tartalmazza a két teljes sztring közötti Levenshtein-távolságot.

##### Példa Kód C++ Nyelven

Az alábbi kód a Levenshtein-távolság dinamikus programozás segítségével történő kiszámítását mutatja be:

```cpp
int levenshtein_dp(const std::string &s, const std::string &t) {
    int m = s.length();
    int n = t.length();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));

    // Initialize the base cases
    for (int i = 0; i <= m; ++i) {
        dp[i][0] = i;
    }
    for (int j = 0; j <= n; ++j) {
        dp[0][j] = j;
    }

    // Fill the DP table
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            int cost = (s[i - 1] == t[j - 1]) ? 0 : 1;
            dp[i][j] = std::min({ 
                dp[i - 1][j] + 1, // Deletion
                dp[i][j - 1] + 1, // Insertion
                dp[i - 1][j - 1] + cost // Substitution
            });
        }
    }

    return dp[m][n];
}
```

#### Mátrix Inicializálása

A mátrix inicializálása a dinamikus programozás első lépése, amely biztosítja, hogy az alap esetek megfelelően legyenek kezelve. Az $dp[i][0] = i$ és $dp[0][j] = j$ inicializálása biztosítja, hogy az algoritmus megfelelően számolja ki a sztring és az üres sztring közötti távolságot. Ez az alapvető lépés elengedhetetlen a helyes működéshez.

#### Mátrix Kitöltése

A mátrix kitöltése az algoritmus fő része, ahol az egyes cellák értékei a korábbi cellák értékeiből kerülnek kiszámításra. A kitöltési folyamat során három művelet közül választjuk a minimumot:
- **Törlés:** Ha egy karaktert törlünk az $s$ sztringből, a költség $dp[i-1][j] + 1$.
- **Beszúrás:** Ha egy karaktert beszúrunk az $t$ sztringbe, a költség $dp[i][j-1] + 1$.
- **Csere vagy Egyezés:** Ha egy karaktert cserélünk vagy egyeztetünk, a költség $dp[i-1][j-1] + \text{cost}(i, j)$.

A $\text{cost}(i, j)$ függvény a karakterek összehasonlításának költségét adja meg: 0, ha az $s[i-1]$ és $t[j-1]$ karakterek megegyeznek, különben 1.

#### Idő- és Térbonyolultság

A dinamikus programozás jelentősen javítja az algoritmus időbonyolultságát a rekurzív megközelítéshez képest. Az időbonyolultság $O(m \cdot n)$, mivel a mátrix minden celláját egyszer számítjuk ki. Az algoritmus térbonyolultsága szintén $O(m \cdot n)$, mivel a teljes mátrixot tároljuk a memória.

Ez az idő- és térbonyolultság azonban nagy sztringek esetében is kezelhető, különösen akkor, ha a sztringek hossza nem extrém nagy. Azonban, ha további optimalizálásra van szükség, a térbonyolultság csökkenthető úgy, hogy csak az aktuális és az előző sorokat tároljuk, mivel a számítás során csak ezekre van szükségünk.

##### Térbonyolultság Csökkentése

A mátrix térbonyolultságát tovább lehet csökkenteni az alábbi módon:

```cpp
int levenshtein_dp_optimized(const std::string &s, const std::string &t) {
    int m = s.length();
    int n = t.length();
    std::vector<int> prev(n + 1), curr(n + 1);

    // Initialize the base case
    for (int j = 0; j <= n; ++j) {
        prev[j] = j;
    }

    // Fill the DP table
    for (int i = 1; i <= m; ++i) {
        curr[0] = i;
        for (int j = 1; j <= n; ++j) {
            int cost = (s[i - 1] == t[j - 1]) ? 0 : 1;
            curr[j] = std::min({ 
                prev[j] + 1, // Deletion
                curr[j - 1] + 1, // Insertion
                prev[j - 1] + cost // Substitution
            });
        }
        std::swap(prev, curr);
    }

    return prev[n];
}
```

Ez az optimalizált megközelítés csak két sort tárol, csökkentve a térbonyolultságot $O(n)$-re.

#### Összegzés

A dinamikus programozás hatékony és elegáns megoldást nyújt a Levenshtein-távolság kiszámítására, jelentősen javítva a rekurzív megközelítés időbonyolultságát. Az algoritmus alkalmazható különféle szövegfeldolgozási feladatokban, beleértve a helyesírás-ellenőrzést, szövegkeresést, bioinformatikai szekvencia-összehasonlítást és gépi fordítást. Az optimalizált változat további előnyt jelent nagy sztringek esetében, mivel csökkenti a memóriaigényt. A következő fejezetekben a dinamikus programozás alkalmazásának további részleteit és gyakorlati példáit tárgyaljuk, bemutatva, hogyan használhatók ezek a technikák valós világban előforduló problémák megoldására.




### 3.6.4. Alkalmazások és példák

A Levenshtein-távolság, más néven szerkesztési távolság, egy sokoldalú és hasznos eszköz számos alkalmazási területen. A különböző szövegfeldolgozási feladatoktól kezdve a bioinformatikai szekvenciák összehasonlításán át egészen a gépi fordításig és az adatbázisok karbantartásáig, a Levenshtein-távolság különböző feladatokban nyújt hatékony megoldást. Ebben a fejezetben részletesen bemutatjuk a Levenshtein-távolság különféle alkalmazási területeit és példáit, valamint azok megvalósítási módját.

#### Szövegfeldolgozás és Helyesírás-ellenőrzés

A helyesírás-ellenőrzés az egyik legelterjedtebb alkalmazási területe a Levenshtein-távolságnak. Egy helyesírás-ellenőrző program összehasonlítja a felhasználó által bevitt szavakat egy szótárral, és meghatározza, hogy a bevitt szó helyes-e. Ha a szó helytelen, a program javaslatokat kínál, amelyek a legkisebb Levenshtein-távolságra vannak a bevitt szótól. Ezáltal könnyen felismerhetők a gépelési hibák, és hasznos javaslatokat lehet tenni.

Például, ha a felhasználó a "recieve" szót írja be, a helyesírás-ellenőrző összehasonlítja azt a szótárban lévő szavakkal, és megtalálja, hogy a "receive" szó csak egy csere műveletnyire van, így azt javasolja javításként.

#### Bioinformatika

A Levenshtein-távolságot széles körben alkalmazzák a bioinformatikában, különösen a DNS és fehérjeszekvenciák összehasonlításában. A szekvenciák közötti hasonlóság mérése alapvető fontosságú a genetikai kutatásokban, evolúciós kapcsolatok vizsgálatában és betegségek diagnosztizálásában.

Például két DNS szekvencia összehasonlítása során a Levenshtein-távolság megmutatja, hogy hány nukleotid beszúrására, törlésére vagy cseréjére van szükség az egyik szekvencia másikká alakításához. Ez segít azonosítani a genetikai variációkat és mutációkat.

##### Példa Kód C++ Nyelven

```cpp
#include <iostream>

#include <string>
#include <vector>

// Function to calculate Levenshtein distance
int levenshtein_dp(const std::string &s, const std::string &t) {
    int m = s.length();
    int n = t.length();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));

    for (int i = 0; i <= m; ++i) {
        for (int j = 0; j <= n; ++j) {
            if (i == 0) {
                dp[i][j] = j;
            } else if (j == 0) {
                dp[i][j] = i;
            } else {
                int cost = (s[i - 1] == t[j - 1]) ? 0 : 1;
                dp[i][j] = std::min({ 
                    dp[i - 1][j] + 1, // Deletion
                    dp[i][j - 1] + 1, // Insertion
                    dp[i - 1][j - 1] + cost // Substitution
                });
            }
        }
    }

    return dp[m][n];
}

int main() {
    std::string seq1 = "GATTACA";
    std::string seq2 = "GCATGCU";
    std::cout << "Levenshtein distance between " << seq1 << " and " << seq2 << " is " << levenshtein_dp(seq1, seq2) << std::endl;
    return 0;
}
```

#### Gépi Fordítás és Szövegösszehasonlítás

A gépi fordítás és szövegösszehasonlítás másik fontos alkalmazási területe a Levenshtein-távolságnak. A gépi fordító rendszerek gyakran hasonlítják össze a forrásnyelvi szöveget a célnyelvi szövegekkel, hogy azonosítsák a hasonló szerkezeteket és kifejezéseket. A Levenshtein-távolság segít abban, hogy a fordító rendszer felismerje a hasonló szövegrészeket, és javításokat végezzen a fordításokban.

#### Adattisztítás és Deduplikáció

Az adattisztítás és deduplikáció során a Levenshtein-távolságot arra használják, hogy azonosítsák a hasonló, de nem teljesen megegyező rekordokat egy adatbázisban. Az adatbázisok gyakran tartalmaznak duplikált vagy hasonló bejegyzéseket, amelyek pontatlanságokat okozhatnak az elemzések során. A Levenshtein-távolság segítségével azonosíthatók és eltávolíthatók ezek a duplikátumok.

Például egy ügyféladatbázisban a "John Smith" és "Jon Smith" bejegyzések hasonlóak, de nem azonosak. A Levenshtein-távolság segítségével ezek a bejegyzések könnyen összehasonlíthatók és deduplikálhatók.

#### Dokumentum Összehasonlítás

A Levenshtein-távolság hasznos eszköz a dokumentumok összehasonlításában is. Az irodalmi és tudományos kutatások során gyakran szükség van két dokumentum hasonlóságának mérésére. A Levenshtein-távolság segítségével megmérhető, hogy mennyire különböznek a dokumentumok egymástól, és azonosíthatók a plágium esetek.

#### Számítógépes Látás

A számítógépes látás területén a Levenshtein-távolságot használják a képfelismerési algoritmusokban is, különösen az optikai karakterfelismerés (OCR) során. Az OCR rendszerek a képeken lévő szövegeket digitális szöveggé alakítják, és a Levenshtein-távolság segít az OCR hibáinak javításában, például a karakterek helytelen felismerésének korrigálásában.

#### Fonetikus Hasonlóság

A Levenshtein-távolság alkalmazható a fonetikus hasonlóság mérésére is, ami különösen hasznos a beszédfelismerési rendszerekben. A beszédfelismerő rendszerek gyakran hibáznak a hasonló hangzású szavak felismerésekor, és a Levenshtein-távolság segíthet az ilyen hibák csökkentésében és a felismerési pontosság növelésében.

#### Alkalmazási Példák

##### Példa 1: Helyesírás-ellenőrzés

Képzeljük el, hogy egy helyesírás-ellenőrző rendszert kell implementálnunk, amely a felhasználó által bevitt szavakat egy szótárhoz hasonlítja, és javaslatokat kínál a helytelenül írt szavakra.

```cpp
#include <iostream>

#include <string>
#include <vector>

#include <algorithm>

// Function to calculate Levenshtein distance
int levenshtein_dp(const std::string &s, const std::string &t) {
    int m = s.length();
    int n = t.length();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));

    for (int i = 0; i <= m; ++i) {
        for (int j = 0; j <= n; ++j) {
            if (i == 0) {
                dp[i][j] = j;
            } else if (j == 0) {
                dp[i][j] = i;
            } else {
                int cost = (s[i - 1] == t[j - 1]) ? 0 : 1;
                dp[i][j] = std::min({ 
                    dp[i - 1][j] + 1, // Deletion
                    dp[i][j - 1] + 1, // Insertion
                    dp[i - 1][j - 1] + cost // Substitution
                });
            }
        }
    }

    return dp[m][n];
}

std::vector<std::string> spell_check(const std::string &word, const std::vector<std::string> &dictionary) {
    std::vector<std::string> suggestions;
    int min_distance = INT_MAX;

    for (const std::string &dict_word : dictionary) {
        int distance = levenshtein_dp(word, dict_word);
        if (distance < min_distance) {
            min_distance = distance;
            suggestions.clear();
            suggestions.push_back(dict_word);
        } else if (distance == min_distance) {
            suggestions.push_back(dict_word);
        }
    }

    return suggestions;
}

int main() {
    std::vector<std::string> dictionary = {"receive", "deceive", "perceive", "believe"};
    std::string word = "recieve";

    std::vector<std::string> suggestions = spell_check(word, dictionary);
    std::cout << "Suggestions for \"" << word << "\": ";
    for (const std::string &suggestion : suggestions) {
        std::cout << suggestion << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

Ez a példa bemutatja, hogyan lehet egy helyesírás-ellenőrző rendszert implementálni a Levenshtein-távolság segítségével. A program összehasonlítja a bevitt szót a szótár szavaival, és javaslatokat tesz a legkisebb távolságra lévő szavak alapján.

##### Példa 2: DNS Szekvenciák Összehasonlítása

A DNS szekvenciák összehasonlítása alapvető fontosságú a genetikai kutatásokban. Az alábbi példa bemutatja, hogyan lehet a Levenshtein-távolságot alkalmazni két DNS szekvencia összehasonlítására.

```cpp
#include <iostream>

#include <string>
#include <vector>

int levenshtein_dp(const std::string &s, const std::string &t) {
    int m = s.length();
    int n = t.length();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));

    for (int i = 0; i <= m; ++i) {
        for (int j = 0; j <= n; ++j) {
            if (i == 0) {
                dp[i][j] = j;
            } else if (j == 0) {
                dp[i][j] = i;
            } else {
                int cost = (s[i - 1] == t[j - 1]) ? 0 : 1;
                dp[i][j] = std::min({ 
                    dp[i - 1][j] + 1, 
                    dp[i][j - 1] + 1, 
                    dp[i - 1][j - 1] + cost 
                });
            }
        }
    }

    return dp[m][n];
}

int main() {
    std::string seq1 = "GATTACA";
    std::string seq2 = "GCATGCU";
    std::cout << "Levenshtein distance between " << seq1 << " and " << seq2 << " is " << levenshtein_dp(seq1, seq2) << std::endl;
    return 0;
}
```

Ez a példa bemutatja, hogyan használható a Levenshtein-távolság a DNS szekvenciák összehasonlítására, és segít azonosítani a genetikai variációkat és mutációkat.

#### Összefoglalás

A Levenshtein-távolság egy rendkívül hasznos eszköz, amely számos alkalmazási területen nyújt hatékony megoldást. A helyesírás-ellenőrzéstől és szövegfeldolgozástól kezdve a bioinformatikai szekvenciák összehasonlításán át egészen a gépi fordításig és az adattisztításig, a Levenshtein-távolság segít a szövegek és szekvenciák közötti különbségek mérésében és elemzésében. Az algoritmus hatékony implementációja dinamikus programozással lehetővé teszi a nagyobb sztringek és szekvenciák gyors és pontos összehasonlítását, ami elengedhetetlen a modern számítástechnikai és biológiai kutatásokban.

