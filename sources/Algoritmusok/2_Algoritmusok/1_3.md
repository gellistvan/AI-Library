## 1.3. Interpolációs keresés

Az interpolációs keresés egy hatékony keresési algoritmus, amelyet gyakran használnak rendezett listákban. Ezen módszer a bináris keresés egyfajta kiterjesztése, ahol a középső elem kiválasztása helyett az aktuális keresett érték elhelyezkedésének becslésére használják az adatok eloszlását. Az interpolációs keresés különösen jól működik egyenletes eloszlású adatstruktúrák esetében, ahol az elemek közel egyenlő távolságra vannak egymástól. Ebben a fejezetben részletesen megvizsgáljuk az interpolációs keresés alapelveit és feltételeit, bemutatjuk az algoritmus implementációját és gyakorlati alkalmazásait, valamint összehasonlítjuk a bináris kereséssel, hogy rávilágítsunk előnyeire és hátrányaira.

### 1.3.1. Alapelvek és feltételek

Az interpolációs keresés egy hatékony keresési algoritmus, amelyet rendezett listákban alkalmaznak. Az algoritmus különösen akkor hasznos, amikor az adatok egyenletesen oszlanak el, mivel ilyenkor a keresési idő jelentősen csökkenthető a hagyományos bináris kereséssel szemben. Az interpolációs keresés alapelve az, hogy a keresett elem pozícióját becsüljük meg az adatok eloszlása alapján, és nem egyszerűen a lista középső eleménél kezdjük a keresést, mint a bináris keresés esetében. E fejezetben részletesen bemutatjuk az interpolációs keresés alapelveit, a működéséhez szükséges feltételeket, valamint a módszer hatékonyságát befolyásoló tényezőket.

#### Alapelvek

Az interpolációs keresés azon az elven alapul, hogy egy rendezett listában az elemek közel egyenlő távolságra vannak egymástól. Ha egy adott keresett értéket (kulcsot) szeretnénk megtalálni a listában, akkor a kulcs helyét a következő interpolációs formula segítségével becsüljük meg:

$\mathrm{pozíció} = \mathrm{alsó\_index} + \left( \frac{\mathrm{kulcs} - \mathrm{lista}[\mathrm{alsó\_index}]}{\mathrm{lista}[\mathrm{felső\_index}] - \mathrm{lista}[\mathrm{alsó\_index}]} \right) \times (\mathrm{felső\_index} - \mathrm{alsó\_index})$


Itt:
- $\text{kulcs}$ a keresett érték.
- $\text{lista}$ az a rendezett lista, amelyben keresünk.
- $\text{alsó\_index}$ és $\text{felső\_index}$ a lista jelenlegi alsó és felső határa.

Az interpolációs keresés lényege, hogy a kulcs várható helyét becsüli meg az alsó és felső határok között, figyelembe véve a kulcs és az adott határok értékeinek különbségét. Ha a kulcs értéke a lista közepén van, akkor az interpolációs keresés hasonlóan működik, mint a bináris keresés. Azonban, ha a kulcs közelebb van az alsó vagy felső határhoz, akkor az interpolációs keresés gyorsabban elérheti a célját.

#### Feltételek

Az interpolációs keresés hatékony működéséhez több feltételnek is teljesülnie kell:

1. **Rendezett lista:** Az interpolációs keresés csak rendezett listákon működik, mivel az algoritmus a kulcs helyének becslésére használja az elemek sorrendjét.
2. **Egyenletes eloszlás:** Az algoritmus hatékonysága akkor a legnagyobb, ha az elemek közel egyenlő távolságra vannak egymástól. Ha az elemek eloszlása egyenetlen, az interpolációs keresés teljesítménye jelentősen romolhat.
3. **Lineáris elérési idő:** Az interpolációs keresés minden egyes lépésben hozzáfér az elemekhez, ezért az elérési időnek lineárisnak kell lennie. Olyan adatszerkezetekben működik jól, ahol az elemek közvetlen elérése gyors.

#### Hatékonyság és időbeli komplexitás

Az interpolációs keresés hatékonyságát leginkább az elemek eloszlása befolyásolja. Ideális esetben, egyenletes eloszlású elemek esetén az interpolációs keresés időbeli komplexitása közelítőleg $O(\log \log n)$. Ez jelentős javulást jelenthet a bináris keresés $O(\log n)$ komplexitásához képest. Azonban ha az elemek eloszlása nagyon egyenetlen, az interpolációs keresés legrosszabb esetben $O(n)$ időben is futtatható, ami kevésbé hatékony, mint a bináris keresés.

Az algoritmus hatékonysága szempontjából fontos megemlíteni, hogy az interpolációs keresés iteratív és rekurzív módon is megvalósítható. A rekurzív megvalósítás hátránya, hogy mély rekurzív hívások esetén a verem túlcsordulásához vezethet, míg az iteratív megközelítés általában biztonságosabb és stabilabb.

#### Interpolációs keresés implementációja C++ nyelven

Az alábbiakban bemutatjuk az interpolációs keresés egy egyszerű iteratív megvalósítását C++ nyelven:

```cpp
#include <iostream>
#include <vector>

int interpolacioKereses(const std::vector<int>& lista, int kulcs) {
    int also = 0;
    int felso = lista.size() - 1;

    while (also <= felso && kulcs >= lista[also] && kulcs <= lista[felso]) {
        if (also == felso) {
            if (lista[also] == kulcs) return also;
            return -1;
        }

        int pozicio = also + ((double)(felso - also) / (lista[felso] - lista[also])) * (kulcs - lista[also]);

        if (lista[pozicio] == kulcs) return pozicio;
        if (lista[pozicio] < kulcs) also = pozicio + 1;
        else felso = pozicio - 1;
    }
    return -1;
}

int main() {
    std::vector<int> lista = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    int kulcs = 70;
    int index = interpolacioKereses(lista, kulcs);

    if (index != -1) {
        std::cout << "A kulcs megtalálható a(z) " << index << " indexen." << std::endl;
    } else {
        std::cout << "A kulcs nincs a listában." << std::endl;
    }

    return 0;
}
```

#### Az algoritmus működése

Az algoritmus az alábbi lépések szerint működik:

1. **Inicializálás:** Az algoritmus meghatározza a lista alsó ($\text{also}$) és felső ($\text{felso}$) indexét.
2. **Pozíció becslése:** Az aktuális alsó és felső határok alapján az algoritmus becsli a kulcs várható helyét a fentebb bemutatott interpolációs formula segítségével.
3. **Elem összehasonlítása:** Az algoritmus összehasonlítja a becsült pozíción lévő elemet a keresett kulccsal:
    - Ha a kulcs megtalálható a becsült pozíción, az algoritmus visszatér a pozícióval.
    - Ha a becsült pozíció eleme kisebb a kulcsnál, az alsó határt frissítjük a becsült pozíció utáni elemre.
    - Ha a becsült pozíció eleme nagyobb a kulcsnál, a felső határt frissítjük a becsült pozíció előtti elemre.
4. **Ismétlés:** Az algoritmus az előző lépéseket ismétli, amíg meg nem találja a kulcsot, vagy az alsó és felső határ össze nem záródik.

#### Összegzés

Az interpolációs keresés egy hatékony keresési algoritmus, amely különösen jól működik egyenletes eloszlású rendezett listákban. Az algoritmus becsli a keresett kulcs helyét, figyelembe véve az elemek eloszlását, így gyorsabban megtalálhatja a kulcsot, mint a bináris keresés. Az interpolációs keresés hatékonysága azonban erősen függ az adatok eloszlásától, és egyenetlen eloszlás esetén a teljesítménye jelentősen romolhat. Az algoritmus iteratív és rekurzív megvalósítása is lehetséges, és az implementáció során fontos figyelembe venni az adatok szerkezetét és eloszlását a legjobb eredmények elérése érdekében.

### 1.3.2. Implementáció és alkalmazások

Az interpolációs keresés implementációja és gyakorlati alkalmazásai fontos szerepet játszanak az algoritmus hatékonyságának és hasznosságának megértésében. Az alábbiakban részletesen tárgyaljuk az interpolációs keresés különböző implementációs aspektusait, beleértve az iteratív és rekurzív megközelítéseket, valamint az algoritmus gyakorlati alkalmazási területeit, előnyeit és hátrányait.

#### Implementáció

Az interpolációs keresés algoritmusa többféleképpen megvalósítható. Az iteratív megközelítés általában előnyösebb, mivel elkerüli a rekurzív hívások okozta verem túlcsordulást, különösen nagyobb adathalmazok esetén. Az alábbiakban bemutatjuk mind az iteratív, mind a rekurzív megvalósítást C++ nyelven.

##### Iteratív implementáció

Az iteratív implementáció egyértelmű és hatékony módja az interpolációs keresés megvalósításának. Az alábbi C++ kód bemutatja, hogyan lehet az algoritmust iteratív módon megvalósítani:

```cpp
#include <iostream>
#include <vector>

int interpolacioKereses(const std::vector<int>& lista, int kulcs) {
    int also = 0;
    int felso = lista.size() - 1;

    while (also <= felso && kulcs >= lista[also] && kulcs <= lista[felso]) {
        if (also == felso) {
            if (lista[also] == kulcs) return also;
            return -1;
        }

        int pozicio = also + ((double)(felso - also) / (lista[felso] - lista[also])) * (kulcs - lista[also]);

        if (lista[pozicio] == kulcs) return pozicio;
        if (lista[pozicio] < kulcs) also = pozicio + 1;
        else felso = pozicio - 1;
    }
    return -1;
}
```

##### Rekurzív implementáció

A rekurzív megközelítés szintén népszerű, de nagyobb adathalmazok esetén a rekurzív hívások miatt a verem túlcsordulásának veszélye fennállhat. Az alábbi C++ kód a rekurzív megvalósítást mutatja be:

```cpp
#include <iostream>
#include <vector>

int interpolacioKeresesRekurziv(const std::vector<int>& lista, int kulcs, int also, int felso) {
    if (also <= felso && kulcs >= lista[also] && kulcs <= lista[felso]) {
        if (also == felso) {
            if (lista[also] == kulcs) return also;
            return -1;
        }

        int pozicio = also + ((double)(felso - also) / (lista[felso] - lista[also])) * (kulcs - lista[also]);

        if (lista[pozicio] == kulcs) return pozicio;
        if (lista[pozicio] < kulcs) return interpolacioKeresesRekurziv(lista, kulcs, pozicio + 1, felso);
        return interpolacioKeresesRekurziv(lista, kulcs, also, pozicio - 1);
    }
    return -1;
}

int main() {
    std::vector<int> lista = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    int kulcs = 70;
    int index = interpolacioKeresesRekurziv(lista, kulcs, 0, lista.size() - 1);

    if (index != -1) {
        std::cout << "A kulcs megtalálható a(z) " << index << " indexen." << std::endl;
    } else {
        std::cout << "A kulcs nincs a listában." << std::endl;
    }

    return 0;
}
```

#### Alkalmazások

Az interpolációs keresés alkalmazása számos területen előnyös lehet, különösen akkor, ha a keresési területen az elemek egyenletes eloszlásúak. Az alábbiakban bemutatjuk az interpolációs keresés néhány gyakorlati alkalmazási területét:

##### 1. Nagy adatbázisok

Nagy adatbázisokban, ahol az adatok egyenletes eloszlásúak, az interpolációs keresés hatékonyabban működik, mint a bináris keresés. Például egy adatbázisban, amely pénzügyi tranzakciókat vagy egyéb numerikus adatokat tartalmaz, az interpolációs keresés gyorsabban megtalálhatja a keresett rekordot.

##### 2. Számítógépes grafika

A számítógépes grafikában, különösen a textúrázási eljárások során, az interpolációs keresés hasznos lehet a textúrázási minták gyors keresésére és az adatok közötti interpolációra. Ez különösen akkor fontos, amikor a textúrák egyenletes eloszlásúak és a hozzáférési idő kritikus.

##### 3. Pénzügyi elemzés

A pénzügyi elemzés területén az interpolációs keresés hatékonyan alkalmazható pénzügyi időszakok, például árfolyamok vagy egyéb piaci adatok keresésére. Az algoritmus lehetővé teszi az adatok gyors keresését és elemzését, különösen akkor, ha az adatok egyenletesen oszlanak el időben.

##### 4. Rendszermonitorozás

A rendszermonitorozás során, ahol a rendszerteljesítmény-adatok egyenletes időintervallumokban kerülnek rögzítésre, az interpolációs keresés segítségével gyorsan megtalálhatók a releváns adatok. Ez lehetővé teszi a rendszermérnökök számára, hogy gyorsan és hatékonyan azonosítsák a problémákat és trendeket.

#### Előnyök és hátrányok

Az interpolációs keresés alkalmazásának számos előnye és hátránya van, amelyeket érdemes figyelembe venni az algoritmus kiválasztásakor:

##### Előnyök

1. **Gyors keresés egyenletes eloszlás esetén:** Az algoritmus ideális esetben $O(\log \log n)$ időbeli komplexitással rendelkezik, ami gyorsabb lehet, mint a bináris keresés $O(\log n)$ komplexitása.
2. **Hatékonyság nagy adathalmazok esetén:** Nagy adathalmazokban, ahol az adatok egyenletesen oszlanak el, az interpolációs keresés jelentős teljesítményjavulást eredményezhet.
3. **Rugalmas alkalmazhatóság:** Az algoritmus különböző területeken alkalmazható, ahol az adatok eloszlása közel egyenletes.

##### Hátrányok

1. **Egyenetlen eloszlás esetén romló teljesítmény:** Ha az adatok eloszlása egyenetlen, az algoritmus teljesítménye jelentősen romolhat, és akár $O(n)$ időbeli komplexitásúvá is válhat.
2. **Korlátozott alkalmazhatóság:** Az algoritmus csak rendezett listákon működik, így nem alkalmazható rendezetlen adathalmazok esetén.
3. **Rekurzív megvalósítás korlátai:** A rekurzív megvalósítás verem túlcsordulásához vezethet nagy adathalmazok esetén, ami a rendszer stabilitását veszélyeztetheti.

#### Összegzés

Az interpolációs keresés egy hatékony algoritmus, amely különösen jól működik egyenletes eloszlású rendezett listákban. Az algoritmus implementálása iteratív és rekurzív módon is lehetséges, bár az iteratív megközelítés általában stabilabb és biztonságosabb. Az interpolációs keresés számos gyakorlati alkalmazási területtel rendelkezik, beleértve a nagy adatbázisokat, a számítógépes grafikát, a pénzügyi elemzést és a rendszermonitorozást. Az algoritmus előnyei közé tartozik a gyors keresési idő egyenletes eloszlás esetén és a rugalmas alkalmazhatóság, míg hátrányai közé sorolható a romló teljesítmény egyenetlen eloszlás esetén és a rekurzív megvalósítás korlátai. Az interpolációs keresés hatékonysága és alkalmazhatósága nagyban függ az adatok eloszlásától és a

konkrét alkalmazási körülményektől, ezért fontos mérlegelni az algoritmus előnyeit és hátrányait a megfelelő keresési módszer kiválasztásakor.

### 1.3.3. Összehasonlítás a bináris kereséssel

Az interpolációs keresés és a bináris keresés két széles körben használt algoritmus rendezett listákban történő keresésre. Bár mindkettő hatékonyabb, mint a lineáris keresés, alapelveik és hatékonyságuk jelentősen különböznek egymástól. Ebben az alfejezetben részletesen összehasonlítjuk az interpolációs keresést és a bináris keresést, kiemelve előnyeiket, hátrányaikat, alkalmazhatóságukat és teljesítményüket különböző körülmények között.

#### Alapelvek és működési mechanizmusok

##### Bináris keresés

A bináris keresés egy egyszerű és hatékony keresési algoritmus, amely rendezett listákban használható. Az algoritmus az alábbi lépések szerint működik:

1. **Kezdőpont kijelölése:** Az algoritmus a lista középső eleménél kezd.
2. **Elem összehasonlítása:** Összehasonlítja a keresett kulcsot a középső elemmel:
    - Ha a kulcs kisebb, mint a középső elem, a keresést a bal oldali részlistában folytatja.
    - Ha a kulcs nagyobb, a jobb oldali részlistában folytatja a keresést.
    - Ha a kulcs megegyezik a középső elemmel, az algoritmus megtalálta a kulcsot és visszatér a pozícióval.
3. **Ismétlés:** Az algoritmus addig ismétli a fenti lépéseket, amíg meg nem találja a kulcsot vagy a részlista üressé nem válik.

A bináris keresés időbeli komplexitása $O(\log n)$, mivel minden lépésben a keresési térfelet felezi.

##### Interpolációs keresés

Az interpolációs keresés szintén rendezett listákban használható, de az alapelve eltér a bináris kereséstől:

1. **Pozíció becslése:** Az algoritmus becsüli a keresett kulcs helyét az alsó és felső határok közötti távolság arányában:

   $$
   \mathrm{pozíció} = \mathrm{alsó\_index} + \left( \frac{\mathrm{kulcs} - \mathrm{lista}[\mathrm{alsó\_index}]}{\mathrm{lista}[\mathrm{felső\_index}] - \mathrm{lista}[\mathrm{alsó\_index}]} \right) \times (\mathrm{felső\_index} - \mathrm{alsó\_index})
   $$

2. **Elem összehasonlítása:** Összehasonlítja a becsült pozíción lévő elemet a kulccsal:
    - Ha a kulcs kisebb, a bal oldali részlistában folytatja a keresést.
    - Ha a kulcs nagyobb, a jobb oldali részlistában folytatja a keresést.
    - Ha a kulcs megegyezik, az algoritmus megtalálta a kulcsot és visszatér a pozícióval.
3. **Ismétlés:** Az algoritmus addig ismétli a lépéseket, amíg meg nem találja a kulcsot vagy a részlista üressé nem válik.

Az interpolációs keresés időbeli komplexitása ideális esetben $O(\log \log n)$, ha az adatok egyenletesen oszlanak el.

#### Teljesítmény összehasonlítás

##### Időbeli komplexitás

- **Bináris keresés:** Az időbeli komplexitása $O(\log n)$. Ez független az adatok eloszlásától, és mindig ugyanaz marad, amennyiben az adatok rendezettek.
- **Interpolációs keresés:** Az időbeli komplexitása ideális esetben $O(\log \log n)$, ha az adatok egyenletesen oszlanak el. Azonban ha az adatok eloszlása egyenetlen, a legrosszabb esetben $O(n)$ is lehet.

##### Teljesítmény és hatékonyság

- **Bináris keresés:** Konzisztens teljesítményt nyújt, mivel minden lépésben a keresési térfelet felezi. Az algoritmus hatékony, függetlenül az adatok eloszlásától.
- **Interpolációs keresés:** Gyorsabb lehet a bináris keresésnél egyenletes eloszlású adatok esetén. Azonban az adatok egyenetlen eloszlása jelentősen ronthatja a teljesítményt.

##### Adatszerkezet követelményei

- **Bináris keresés:** Csak rendezett listát igényel, függetlenül az adatok eloszlásától.
- **Interpolációs keresés:** Rendezett és egyenletesen eloszló adatokra van szükség a maximális hatékonyság eléréséhez.

#### Gyakorlati alkalmazások és korlátok

##### Gyakorlati alkalmazások

**Bináris keresés:**

1. **Adatbázisok:** Gyakran használják adatbázisokban indexek keresésére.
2. **Számítógépes tudomány:** Széles körben alkalmazzák algoritmusokban és adatszerkezetekben, például a bináris keresőfákban.
3. **Rendszerprogramozás:** Használják rendszerszintű programozásban, például kernel adatstruktúrákban.

**Interpolációs keresés:**

1. **Nagy adatbázisok:** Használható nagy adatbázisokban, ahol az adatok egyenletes eloszlásúak.
2. **Pénzügyi elemzés:** Alkalmazzák pénzügyi időszakok adatelemzésében, ahol az adatok időben egyenletesen oszlanak el.
3. **Számítógépes grafika:** Használható textúrázási eljárásokban, ahol a textúrák egyenletesen oszlanak el.

##### Korlátok

**Bináris keresés:**

1. **Csak rendezett adatok:** Csak rendezett adatokon működik.
2. **Korlátozott adaptivitás:** Nem használja ki az adatok eloszlását a teljesítmény javítására.

**Interpolációs keresés:**

1. **Egyenetlen eloszlás:** Rosszul teljesít egyenetlen eloszlású adatok esetén.
2. **Komplexitás:** Az algoritmus bonyolultabb, mint a bináris keresés, és nehezebb implementálni.

#### Példa C++ implementációra

Az alábbiakban bemutatjuk mindkét algoritmus egyszerű implementációját C++ nyelven.

**Bináris keresés:**

```cpp
#include <iostream>
#include <vector>

int binarisKereses(const std::vector<int>& lista, int kulcs) {
    int also = 0;
    int felso = lista.size() - 1;
    
    while (also <= felso) {
        int kozep = also + (felso - also) / 2;
        
        if (lista[kozep] == kulcs) return kozep;
        if (lista[kozep] < kulcs) also = kozep + 1;
        else felso = kozep - 1;
    }
    
    return -1;
}
```

**Interpolációs keresés:**

```cpp
#include <iostream>
#include <vector>

int interpolacioKereses(const std::vector<int>& lista, int kulcs) {
    int also = 0;
    int felso = lista.size() - 1;

    while (also <= felso && kulcs >= lista[also] && kulcs <= lista[felso]) {
        if (also == felso) {
            if (lista[also] == kulcs) return also;
            return -1;
        }

        int pozicio = also + ((double)(felso - also) / (lista[felso] - lista[also])) * (kulcs - lista[also]);

        if (lista[pozicio] == kulcs) return pozicio;
        if (lista[pozicio] < kulcs) also = pozicio + 1;
        else felso = pozicio - 1;
    }
    return -1;
}
```

#### Összegzés

Az interpolációs keresés és a bináris keresés közötti választás nagyban függ az adott probléma sajátosságaitól és az adatok eloszlásától. A bináris keresés egy általános célú, hatékony algoritmus, amely kon

zisztens teljesítményt nyújt rendezett adathalmazok esetén. Az interpolációs keresés viszont különösen akkor hatékony, ha az adatok egyenletes eloszlásúak, de teljesítménye jelentősen romolhat egyenetlen eloszlás esetén.

Mindkét algoritmusnak megvannak a maga előnyei és korlátai, és a megfelelő keresési módszer kiválasztása során fontos figyelembe venni az adott alkalmazási környezetet és a keresett adatok jellemzőit. A bináris keresés egyszerűsége és stabilitása miatt széles körben alkalmazott, míg az interpolációs keresés előnyeit főként specifikus, egyenletes eloszlású adathalmazok esetén lehet kihasználni.

