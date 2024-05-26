\newpage

## 7. Fenwick fa (Binary Indexed Tree)

A Fenwick fa, más néven Binary Indexed Tree (BIT), egy hatékony adatszerkezet, amely lehetővé teszi a prefix összegek gyors lekérdezését és frissítését. Ez a szerkezet különösen hasznos olyan feladatoknál, ahol gyakoriak az összeg lekérdezések és az adatok módosításai. A Fenwick fa kombinálja a dinamizmus és az egyszerűség előnyeit, miközben kiváló teljesítményt biztosít mind az összegek lekérdezésénél, mind az értékek frissítésénél. A következő alfejezetekben megismerkedünk a Fenwick fa alapfogalmaival és alapvető műveleteivel, majd bemutatjuk, hogyan alkalmazhatjuk ezt az adatszerkezetet a prefix összeg és az intervallum frissítés problémáinak hatékony megoldására.

## 7.1. Alapfogalmak és műveletek

### Bevezetés

A Fenwick fa, más néven Binary Indexed Tree (BIT), egy különleges adatszerkezet, amely lehetővé teszi bizonyos típusú lekérdezések és frissítések hatékony végrehajtását. Elsősorban prefix összegek és intervallum frissítések kezelésére használják. A Fenwick fa fő előnye, hogy mind a lekérdezések, mind a frissítések időkomplexitása $O(\log n)$, ahol $n$ az elemek száma. Ezáltal ideális megoldást nyújt nagy méretű adathalmazok esetén, ahol a műveletek hatékonysága kulcsfontosságú.

### Alapfogalmak

#### Prefix összeg

A prefix összeg egy olyan művelet, amely egy tömb első $i$ elemének összegét adja meg. Formálisan, ha van egy $a$ tömbünk, akkor az $i$-edik prefix összeg:

$\text{prefix\_sum}(i) = \sum_{k=1}^{i} a[k]$

#### Frissítés

A frissítés egy olyan művelet, amely a tömb egy adott elemének értékét módosítja. Ez a művelet közvetlenül befolyásolja az összes, az adott elemre épülő prefix összeget.

### A Fenwick fa alapelvei

A Fenwick fa egy tömb alapú adatszerkezet, amely hatékonyan támogatja a prefix összeg és frissítés műveleteket. A fa egy olyan tömbön alapul, amelynek elemei részösszegeket tartalmaznak, így lehetővé téve a gyors összegzést és módosítást.

#### Struktúra

A Fenwick fa egy olyan tömböt használ, amelyet $BIT$-nek nevezünk. Az $BIT[i]$ elem tartalmazza az $a$ tömb azon elemeinek összegét, amelyek hatással vannak a $i$-edik pozíció prefix összegére.

#### Konstrukció

A Fenwick fa építése egy üres $BIT$ tömbből indul, amelyet fokozatosan töltünk fel az $a$ tömb elemeinek megfelelően.

#### Indexelési szabályok

A Fenwick fa indexelése egy speciális szabályt követ, amely a legkisebb változó biten alapul. Az $i$-edik elem által reprezentált intervallum a következőképpen határozható meg:

$\text{Intervallum}(i) = [i - \text{LSB}(i) + 1, i]$

ahol $\text{LSB}(i)$ az $i$-edik szám legkisebb változó bitjének értéke.

### Műveletek

#### Inicializálás

Az inicializálás során a Fenwick fa minden eleme nullára van állítva. Ez a művelet $O(n)$ időben történik, ahol $n$ a tömb mérete.

```cpp
void initializeBIT(int BIT[], int n) {
    for (int i = 1; i <= n; i++) {
        BIT[i] = 0;
    }
}
```

#### Frissítés

A frissítés során egy adott pozíció értékét növeljük egy adott $\Delta$ értékkel. Ennek során a BIT tömb megfelelő elemeit is frissítjük.

```cpp
void updateBIT(int BIT[], int n, int index, int delta) {
    while (index <= n) {
        BIT[index] += delta;
        index += index & (-index);
    }
}
```

A frissítési művelet során a legkisebb változó bitet használjuk annak meghatározására, hogy mely elemeket kell frissíteni. Ez biztosítja, hogy minden frissítés $O(\log n)$ időben történjen.

#### Prefix összeg lekérdezés

A prefix összeg lekérdezés során az $i$-edik pozícióig terjedő elemek összegét számoljuk ki.

```cpp
int getPrefixSum(int BIT[], int index) {
    int sum = 0;
    while (index > 0) {
        sum += BIT[index];
        index -= index & (-index);
    }
    return sum;
}
```

A művelet során az indexet fokozatosan csökkentjük a legkisebb változó bit segítségével, ami lehetővé teszi, hogy csak azokat az elemeket adjuk össze, amelyek szükségesek a prefix összeghez. Ennek köszönhetően a művelet időkomplexitása szintén $O(\log n)$.

#### Teljes tömb frissítése

Ha az egész tömböt egy adott értékkel szeretnénk frissíteni, akkor minden elemen külön-külön végre kell hajtani a frissítési műveletet. Ez az eljárás szintén $O(n \log n)$ időben fut le.

```cpp
void updateArray(int BIT[], int n, int delta) {
    for (int i = 1; i <= n; i++) {
        updateBIT(BIT, n, i, delta);
    }
}
```

### Elméleti alapok és analízis

#### Időkomplexitás

A Fenwick fa használata jelentős időmegtakarítást eredményez a hagyományos tömbök használatához képest. Mind a frissítési, mind a lekérdezési műveletek időkomplexitása $O(\log n)$, szemben a lineáris $O(n)$ megoldásokkal.

#### Memóriakomplexitás

A Fenwick fa memóriakomplexitása $O(n)$, mivel egy $n$ elemű tömböt használ a prefix összegek tárolására. Ezáltal a Fenwick fa nem igényel jelentős mennyiségű extra memóriát a hagyományos tömbökhöz képest.

### Összegzés

A Fenwick fa egy hatékony és könnyen implementálható adatszerkezet, amely kiválóan alkalmas prefix összegek és intervallum frissítések kezelésére. Az alapvető műveletek - mint az inicializálás, frissítés és prefix összeg lekérdezés - mind $O(\log n)$ időben futnak, ami jelentős teljesítménynövekedést eredményez nagy méretű adathalmazok esetén. Az egyszerű implementáció és a hatékony működés teszi a Fenwick fát népszerű választássá számos alkalmazási területen, beleértve a számítógépes algoritmusok és az adatbázis-kezelő rendszerek területét is.

## 7.2. Alkalmazások: prefix összeg, intervallum frissítés

### Bevezetés

A Fenwick fa, vagy Binary Indexed Tree (BIT), egy sokoldalú adatszerkezet, amely különböző típusú lekérdezések és frissítések hatékony végrehajtását teszi lehetővé. A leggyakoribb alkalmazásai közé tartozik a prefix összegek számítása és az intervallum frissítések kezelése. Ezek az alkalmazások különösen fontosak számos számítógépes algoritmusban, adatbázis-kezelő rendszerekben és egyéb informatika területeken. Ebben az alfejezetben részletesen bemutatjuk, hogyan használható a Fenwick fa a prefix összeg és az intervallum frissítés problémáinak megoldására.

### Prefix összeg

#### Definíció és jelentőség

A prefix összeg egy adott tömb első $i$ elemének összegét jelenti. Formálisan, ha van egy $a$ tömbünk, akkor az $i$-edik prefix összeg:

$\text{prefix\_sum}(i) = \sum_{k=1}^{i} a[k]$

Ez a művelet különösen hasznos lehet, amikor gyakran kell lekérdeznünk egy tömb bizonyos részének összegét, például pénzügyi adatok vagy statisztikai elemzések során.

#### Megvalósítás Fenwick fával

A Fenwick fa hatékonyan támogatja a prefix összeg lekérdezését $O(\log n)$ időben. A Fenwick fa használatával a prefix összeg lekérdezése a következő lépésekből áll:

1. **Inicializálás:** Egy BIT tömb létrehozása, amely a részösszegeket tartalmazza.
2. **Frissítés:** Az eredeti tömb értékeinek beállítása és a BIT tömb frissítése.
3. **Lekérdezés:** A kívánt prefix összeg kiszámítása a BIT tömb segítségével.

A prefix összeg lekérdezésének algoritmusa:

```cpp
int getPrefixSum(int BIT[], int index) {
    int sum = 0;
    while (index > 0) {
        sum += BIT[index];
        index -= index & (-index);
    }
    return sum;
}
```

#### Példa

Tegyük fel, hogy van egy tömbünk: $[3, 2, -1, 6, 5, 4, -3, 3, 7, 2]$. A Fenwick fa segítségével gyorsan kiszámíthatjuk a prefix összegeket, például az első 5 elem összegét:

$\text{prefix\_sum}(5) = 3 + 2 - 1 + 6 + 5 = 15$

### Intervallum frissítés

#### Definíció és jelentőség

Az intervallum frissítés egy olyan művelet, amely egy adott intervallumban növeli meg az elemek értékét egy adott $\Delta$ értékkel. Ez a művelet különösen fontos olyan feladatoknál, ahol gyakoriak az értékek módosításai egy adott tartományban, például hőmérsékleti adatok frissítése vagy adatbázisokban végzett csoportos frissítések esetén.

#### Megvalósítás Fenwick fával

A Fenwick fa két külön BIT tömböt használ az intervallum frissítések kezelésére: az egyik tömb a közvetlen értékeket, míg a másik a prefix összegek frissítéséhez szükséges korrekciókat tartalmazza.

1. **Inicializálás:** Két BIT tömb létrehozása, egy az értékekhez és egy a korrekciókhoz.
2. **Intervallum frissítés:** Az adott intervallum frissítése a két BIT tömb megfelelő módosításával.
3. **Lekérdezés:** A kívánt elem értékének lekérdezése a két BIT tömb kombinációjával.

Az intervallum frissítés és lekérdezés algoritmusa:

```cpp
void updateRange(int BIT1[], int BIT2[], int n, int l, int r, int delta) {
    updateBIT(BIT1, n, l, delta);
    updateBIT(BIT1, n, r + 1, -delta);
    updateBIT(BIT2, n, l, delta * (l - 1));
    updateBIT(BIT2, n, r + 1, -delta * r);
}

int query(int BIT1[], int BIT2[], int index) {
    return (getPrefixSum(BIT1, index) * index) - getPrefixSum(BIT2, index);
}
```

#### Példa

Tegyük fel, hogy frissíteni szeretnénk a 2. és 5. indexek közötti elemeket 3-mal. Az intervallum frissítés után az értékek módosulnak:
```
a[2] & = a[2] + 3
a[3] & = a[3] + 3
a[4] & = a[4] + 3
a[5] & = a[5] + 3
```


### Komplexitás analízis

A Fenwick fa mind a prefix összeg, mind az intervallum frissítés problémáira hatékony megoldást kínál. Az időkomplexitás mindkét művelet esetén $O(\log n)$, ami jelentős javulást jelent a naiv $O(n)$ megoldásokhoz képest. A memóriakomplexitás $O(n)$, mivel a BIT tömb mérete az eredeti tömb méretével arányos.

### Összegzés

A Fenwick fa alkalmazása a prefix összeg és az intervallum frissítés problémáinak megoldására jelentős előnyöket kínál mind az idő-, mind a memóriakomplexitás tekintetében. Az egyszerű implementáció és a hatékony működés miatt a Fenwick fa széles körben alkalmazható különböző számítástechnikai és adatbázis-kezelési feladatokban. Az itt bemutatott algoritmusok és példák segítenek megérteni és alkalmazni a Fenwick fa alapelveit a gyakorlatban, biztosítva a gyors és hatékony adatkezelést.

