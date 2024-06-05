\newpage

## 3.7. Érmék összegzése (Coin Change)

Az érmék összegzése probléma egy klasszikus példa a kombinatorikus optimalizáció területén, amely számos valós életbeli alkalmazással rendelkezik. Legyen adott egy véges halmaz különböző címletű érmékből, és egy adott összeg, amit ezekből az érmékből kell kifizetni. Az egyik leggyakoribb kérdés az, hogy mi a legkisebb számú érme, amellyel pontosan ki lehet fizetni ezt az összeget. Ez a probléma számos különböző megközelítéssel megoldható, beleértve a rekurzív módszereket és a dinamikus programozást. Az alábbiakban részletesen megvizsgáljuk mindkét megoldási technikát, kiemelve azok előnyeit és hátrányait. Az első részben a minimális érmék számának meghatározásával foglalkozunk, majd áttérünk a rekurzív megoldásra, végül pedig a dinamikus programozási megközelítést tárgyaljuk, amely hatékonyabb megoldást kínál nagyobb problémák esetén.

### 3.8.1. Minimum érmék számának meghatározása

Az érmék összegzése (Coin Change) probléma egyik központi kérdése az, hogy hogyan lehet egy adott összeget minimális számú érmével kifizetni, adott címletek mellett. Ez a kérdés számos valós alkalmazási területen felmerül, például pénzügyi tranzakciók optimalizálása, készletgazdálkodás, valamint különféle erőforrások hatékony elosztása során. Ebben a fejezetben részletesen megvizsgáljuk, hogyan határozható meg a minimális érmék száma, amely szükséges egy adott összeg pontos kifizetéséhez.

#### A probléma definíciója

Az érmék összegzése probléma formálisan az alábbiak szerint definiálható: Legyen adott egy véges halmaz különböző címletű érmékből $C = \{c_1, c_2, \ldots, c_n\}$, valamint egy $S$ egész szám, amely az összeg, amit ki kell fizetni. A cél az, hogy megtaláljuk a legkisebb $k$ számot, amelyre igaz, hogy $S$ összeget pontosan $k$ darab érméből lehet kifizetni a halmaz elemeit felhasználva.

#### Probléma analízise

A probléma megoldásának többféle megközelítése is létezik, beleértve a rekurzív és dinamikus programozási technikákat. Az alapvető kihívás az, hogy olyan kombinációt találjunk az érmék közül, amely minimális számú érmét tartalmaz, de az összeg pontosan $S$.

Az érmék összegzése probléma egy NP-nehéz probléma, ami azt jelenti, hogy a probléma pontos megoldása hosszú számítási időt igényelhet nagyobb bemenetek esetén. Ezért különösen fontos hatékony algoritmusokat kifejleszteni a probléma megoldására.

#### Általános megközelítés

Az érmék összegzése probléma megoldásához szükséges egy jól strukturált megközelítés. Az alábbi lépések általánosan alkalmazhatók a probléma megoldása során:

1. **Érmék kiválasztása és sorrendjük:** Először is érdemes lehet az érméket csökkenő sorrendbe rendezni, hogy először a legnagyobb címletekkel próbálkozzunk. Ez azonban nem mindig vezet optimális megoldáshoz.

2. **Kombinációk vizsgálata:** Minden lehetséges kombinációt meg kell vizsgálni, hogy megtaláljuk a minimális számú érmét. Ez történhet rekurzív vagy iteratív módon.

3. **Részproblémák megoldása:** Az érmék összegzése probléma felbontható kisebb részproblémákra. Például, ha ismerjük a minimális érmék számát egy kisebb összeg kifizetésére, ezt az információt felhasználhatjuk nagyobb összegek esetén.

4. **Optimalizálás:** A részproblémák megoldásainak tárolása és újrafelhasználása jelentősen csökkentheti a számítási időt. Ezt a megközelítést használja a dinamikus programozás.

#### Gyakorlati alkalmazások

A minimális érmék számának meghatározása számos gyakorlati alkalmazási területtel rendelkezik:

- **Pénzügyi tranzakciók:** A probléma közvetlenül alkalmazható pénzügyi tranzakciók optimalizálására, például amikor egy pénzváltó automata vagy kassza a legkevesebb érme kiadására törekszik.
- **Készletgazdálkodás:** A különböző erőforrások hatékony elosztása során gyakran felmerül a minimális egységek használatának kérdése.
- **Kombinatorikus optimalizálás:** Az érmék összegzése probléma számos más kombinatorikus optimalizálási probléma alapjául szolgál, és azok megoldásában is hasznos technikákat biztosít.

#### Következtetés

Az érmék összegzése probléma, különösen a minimális érmék számának meghatározása, egy alapvető és gyakran vizsgált probléma az algoritmusok területén. A probléma megértése és hatékony megoldása lehetővé teszi a különböző gyakorlati alkalmazások optimalizálását, és alapvető fontosságú a számítástudomány és az operációkutatás számos területén. A következő alfejezetekben részletesen megvizsgáljuk a probléma rekurzív és dinamikus programozási megközelítéseit, valamint bemutatjuk azok előnyeit és hátrányait.

### 3.8.2. Megoldás rekurzióval

A rekurzív megközelítés az érmék összegzése (Coin Change) problémára egy intuitív és közvetlen módszer, amely egyszerűen megvalósítható, de nem mindig a leghatékonyabb megoldás. Ebben a fejezetben részletesen bemutatjuk a rekurzív algoritmus működését, előnyeit és korlátait.

#### Rekurzív algoritmus elve

A rekurzív megoldás alapja a probléma felbontása kisebb, hasonló részekre, majd ezen részproblémák megoldása. Az érmék összegzése probléma esetén ez azt jelenti, hogy minden egyes címlet kiválasztásával csökkentjük az összeget, majd rekurzívan hívjuk a függvényt a maradék összegre. A rekurzió mélyülésével a maradék összeg egyre kisebb lesz, amíg el nem érjük az alap esetet, amikor az összeg 0, és nincs szükség több érmére.

#### Rekurzív megközelítés működése

A rekurzív megközelítés három alapvető lépésből áll:
1. **Alap eset kezelése:** Ha az összeg 0, akkor nincs szükség több érmére, így a megoldás 0 érmét igényel.
2. **Érmék kiválasztása:** Minden lehetséges címletet kiválasztunk, és kivonjuk azt az aktuális összegből.
3. **Rekurzív hívás:** Rekurzívan meghívjuk a függvényt a maradék összegre, majd összehasonlítjuk a különböző címletekből kapott eredményeket, hogy megtaláljuk a minimális érmék számát.

A következő pszeudokód szemlélteti a rekurzív megközelítést:

```cpp
int coinChangeRec(vector<int>& coins, int amount) {
    if (amount == 0) return 0; // Alap eset: nincs szükség több érmére
    int minCoins = INT_MAX; // Kezdeti érték a minimális érmék számának
    for (int coin : coins) { // Végigmegyünk az összes címleten
        if (amount - coin >= 0) { // Ha a címlet kisebb vagy egyenlő az aktuális összegnél
            int result = coinChangeRec(coins, amount - coin); // Rekurzív hívás a maradék összegre
            if (result != INT_MAX && result + 1 < minCoins) { // Ellenőrzés és minimális érték frissítése
                minCoins = result + 1;
            }
        }
    }
    return minCoins;
}
```

#### Rekurzív algoritmus részletei

**Alap eset:**
Az alap eset a rekurzió legmélyebb pontja, ahol az összeg 0. Ebben az esetben nincs szükség több érmére, így a függvény visszatérési értéke 0.

**Érmék kiválasztása és rekurzív hívás:**
Minden lehetséges címletet kiválasztunk, és kivonjuk azt az aktuális összegből. Ezután a maradék összegre rekurzívan meghívjuk a függvényt. Ha a maradék összegre kapott megoldás nem végtelen (ami azt jelenti, hogy létezik megoldás), akkor frissítjük a minimális érmék számát az aktuális címlet figyelembevételével.

**Eredmény visszaadása:**
A függvény visszaadja a minimális érmék számát, amely szükséges az adott összeg kifizetéséhez. Ha nem található megoldás (azaz a minimális érmék száma végtelen marad), akkor nincs lehetséges kombináció a címletekből az adott összeg kifizetésére.

#### Előnyök és hátrányok

**Előnyök:**
- **Egyszerű implementáció:** A rekurzív megoldás könnyen megérthető és implementálható, különösen kisebb problémaméretek esetén.
- **Intuitív megközelítés:** A probléma természetes felbontása kisebb részekre jól illeszkedik a rekurzív technikához.

**Hátrányok:**
- **Exponenciális időkomplexitás:** A rekurzív megoldás időkomplexitása exponenciális lehet a problémaméret növekedésével, mivel sokszor ugyanazokat a részproblémákat számítjuk újra és újra.
- **Memóriahasználat:** A rekurzív hívások mély hívási veremhez vezethetnek, ami nagy memóriaigényt jelenthet nagyobb összeg esetén.
- **Hatékonyság hiánya:** A rekurzív megoldás nem hatékony nagyobb problémák esetén, mivel sok redundáns számítást végez.

#### Optimalizációs lehetőségek

A rekurzív megközelítés optimalizálására több módszer is létezik:

- **Memoizáció:** Az egyik leghatékonyabb technika a memoizáció, amely a dinamikus programozás alapelveit használja fel. Ebben az esetben a részproblémák megoldásait egy táblában tároljuk, és újrafelhasználjuk, amikor ugyanaz a részprobléma ismét előfordul. Ez jelentősen csökkenti az időkomplexitást.
- **Dinamikus programozás:** Teljesen elkerülve a rekurziót, a dinamikus programozás iteratív megközelítést alkalmaz, amely szintén eltárolja és újrafelhasználja a részeredményeket, így elkerülve a redundáns számításokat.

#### Összegzés

A rekurzív megközelítés az érmék összegzése problémára egyszerű és könnyen érthető, azonban nagyobb problémaméretek esetén hatékonysági problémákba ütközik. Az idő- és memóriaigények exponenciális növekedése miatt a rekurzív megoldás nem alkalmas nagyobb összegű problémák kezelésére. Ennek ellenére a rekurzív technika fontos szerepet játszik az algoritmusok tervezésében és megértésében, mivel alapvető betekintést nyújt a probléma természetébe és a lehetséges megoldások szerkezetébe. Az optimalizációs technikák, mint a memoizáció és a dinamikus programozás, jelentősen javíthatják a megoldás hatékonyságát, és ezek a módszerek részletesebben tárgyalásra kerülnek a következő alfejezetekben.


### 3.8.3. Megoldás dinamikus programozással

A dinamikus programozás egy hatékony módszer a kombinatorikus optimalizálási problémák megoldására, különösen akkor, ha a probléma kisebb, egymást átfedő részproblémákra bontható. Az érmék összegzése (Coin Change) probléma esetében a dinamikus programozás használata jelentősen csökkentheti a számítási időt és a memóriahasználatot azáltal, hogy eltárolja és újrafelhasználja a korábban kiszámított részproblémák eredményeit. Ebben a fejezetben részletesen bemutatjuk a dinamikus programozási megközelítést az érmék összegzése probléma megoldására.

#### Dinamikus programozási megközelítés alapelvei

A dinamikus programozás két fő technikát alkalmaz: a memoizációt és a tabulációt. Mindkét módszer lényege, hogy a részproblémák eredményeit eltároljuk, így elkerülve az ismételt számításokat. Az érmék összegzése probléma esetén az alábbi lépések követésével oldjuk meg a feladatot:

1. **Állapot definiálása:** Határozzuk meg a problémát állapotokként, ahol minden állapot egy adott összeg minimális érmék számát jelenti.
2. **Állapotátmenet:** Határozzuk meg az állapotok közötti átmeneteket, azaz hogyan számíthatjuk ki az adott összeg minimális érmék számát a kisebb összegek minimális érmék számából.
3. **Alap eset:** Definiáljuk az alap esetet, amelyet közvetlenül meg tudunk oldani, és amely nem igényel további részproblémák megoldását.
4. **Optimalizáció:** Iteratívan oldjuk meg a részproblémákat, és tároljuk el az eredményeket, hogy a fő problémát hatékonyan meg tudjuk oldani.

#### Az érmék összegzése probléma megoldása dinamikus programozással

Az érmék összegzése probléma megoldása során egy tömböt használunk, ahol a tömb indexe az adott összeget jelöli, és az értéke a minimális érmék számát tartalmazza, amellyel az adott összeget ki lehet fizetni.

1. **Állapot definiálása:**
   Legyen $dp[i]$ az a minimális érmék száma, amellyel az $i$ összeget ki lehet fizetni. Kezdetben minden $dp[i]$ értéket végtelenre (vagy egy nagyon nagy számra) inicializálunk, kivéve $dp[0]$-t, amely 0, mivel a 0 összeg kifizetéséhez 0 érme szükséges.

2. **Állapotátmenet:**
   Minden érmecímletre $c$ és minden összegre $i$ az 1-től $S$-ig terjedő tartományban (ahol $S$ a kívánt összeg), a következő átmenetet alkalmazzuk:
   $$
   dp[i] = \min(dp[i], dp[i - c] + 1)
   $$
   Ez azt jelenti, hogy ha az aktuális címlet $c$ kisebb vagy egyenlő az aktuális összegnél $i$, akkor frissítjük $dp[i]$-t a $dp[i - c] + 1$ értékre, ha az kisebb, mint a jelenlegi $dp[i]$.

3. **Alap eset:**
   Az alap eset $dp[0] = 0$, mivel a 0 összeg kifizetéséhez 0 érme szükséges.

4. **Optimalizáció:**
   Iteratívan számítjuk ki az összes $dp[i]$ értéket az 1-től $S$-ig terjedő tartományban. Az iterációk során minden címletet felhasználunk az összes lehetséges összeg kifizetéséhez.

Az alábbi C++ kód bemutatja a dinamikus programozás alkalmazását az érmék összegzése problémára:

```cpp
#include <vector>

#include <algorithm>
#include <climits>

int coinChangeDP(const std::vector<int>& coins, int amount) {
    std::vector<int> dp(amount + 1, INT_MAX);
    dp[0] = 0;
    for (int i = 1; i <= amount; ++i) {
        for (int coin : coins) {
            if (i - coin >= 0 && dp[i - coin] != INT_MAX) {
                dp[i] = std::min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    return dp[amount] == INT_MAX ? -1 : dp[amount];
}
```

#### Idő- és térkomplexitás elemzése

A dinamikus programozás algoritmusának időkomplexitása $O(S \cdot n)$, ahol $S$ a kívánt összeg, és $n$ az elérhető érmecímletek száma. Ez az időkomplexitás abból adódik, hogy minden egyes összeget 1-től $S$-ig kiszámítunk, és minden egyes összeg esetén végigmegyünk az összes címleten.

A térkomplexitás $O(S)$, mivel egy $S+1 \$ elemű tömböt használunk az összes részprobléma eredményének tárolására.

#### Előnyök és hátrányok

**Előnyök:**
- **Hatékonyság:** A dinamikus programozás jelentősen hatékonyabb, mint a rekurzív megközelítés, különösen nagyobb problémaméretek esetén.
- **Egyértelmű megoldás:** Az algoritmus garantáltan megtalálja a minimális érmék számát, ha létezik megoldás.

**Hátrányok:**
- **Memóriahasználat:** A térkomplexitás $O(S)$, ami nagyobb összegek esetén jelentős memóriahasználatot jelenthet.
- **Nem mindig optimális minden helyzetben:** Bár az algoritmus hatékony, más megközelítések, például a különböző heurisztikus algoritmusok, bizonyos esetekben gyorsabbak lehetnek.

#### Gyakorlati alkalmazások és kiterjesztések

A dinamikus programozási megközelítést számos gyakorlati alkalmazásban használják:
- **Pénzügyi rendszerek:** A minimális érmék számának meghatározása segíthet a pénzváltók és automaták működésének optimalizálásában.
- **Logisztikai optimalizálás:** Az erőforrások elosztásának hatékony kezelése hasonló problémákra bontható le.
- **Általános optimalizáció:** Sok kombinatorikus optimalizálási probléma hasonló megoldási technikákat igényel.

A dinamikus programozási technikák további finomításai, mint például a különböző memoizációs technikák, javíthatják az algoritmus hatékonyságát és alkalmazhatóságát. Az érmék összegzése probléma megértése és hatékony megoldása alapvető fontosságú a számítástudomány és az operációkutatás különböző területein, és alapvető eszközt biztosít a komplex problémák kezelésére.

#### Következtetés

A dinamikus programozás erőteljes és hatékony megközelítést kínál az érmék összegzése probléma megoldására. Míg a rekurzív megoldások egyszerűek és intuitívak, a dinamikus programozás jelentősen csökkenti a számítási időt és a memóriahasználatot azáltal, hogy eltárolja és újrafelhasználja a korábban kiszámított részproblémák eredményeit. Az ilyen optimalizációs technikák alapvető fontosságúak a modern számítástechnikai alkalmazásokban, és lehetővé teszik a komplex problémák hatékony megoldását.

