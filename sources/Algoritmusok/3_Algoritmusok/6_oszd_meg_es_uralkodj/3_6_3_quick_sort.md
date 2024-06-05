\newpage

## 6.3. Gyorsrendezés (Quick Sort)

A gyorsrendezés, vagy angolul Quick Sort, a számítástudomány egyik legismertebb és legszélesebb körben használt rendezési algoritmusa. Az Oszd-meg-és-uralkodj módszert alkalmazva hatékonyan rendezi az adatokat, méghozzá átlagosan O(n log n) idő alatt. Ebben a fejezetben részletesen megvizsgáljuk, hogyan működik a Quick Sort algoritmus, bemutatjuk annak lépéseit és különböző megvalósítási módjait, majd alapos teljesítmény elemzést végzünk, hogy megértsük erősségeit és gyengeségeit más rendezési algoritmusokkal szemben. Többek között megismerhetjük majd a legjobb és legrosszabb esetekben előforduló időbeli komplexitást, és rávilágítunk arra, miért maradt a Quick Sort hosszú évek óta a fejlesztők és kutatók kedvence.

## 6.3. Gyorsrendezés (Quick Sort)

### 6.3.1 Algoritmus és implementáció

A gyorsrendezés, vagy más néven Quick Sort, a számítástechnika és algoritmusok terén ismert talán legszélesebb körben használt rendezési algoritmus. Az eljárás alapötlete az oszd-meg-és-uralkodj (divide-and-conquer) stratégia alkalmazása, amely hatékonyan kezeli a nagy méretű adatstruktúrák rendezését. A Quick Sort algoritmus lényege, hogy kiválaszt egy úgynevezett pivot elemet, majd az adatokat két részre osztja, és rekurzív módon rendezi ezeket a részeket. Most nézzük meg az algoritmus működését lépésről lépésre, és hogy hogyan implementálhatjuk ezt a gyakorlatban.

#### Algoritmus Lépései

1. **Pivot kiválasztása:** Az algoritmus első lépése egy pivot elem kiválasztása a rendezendő adatelemek közül. Ez az elem később a rendezési folyamat során központi szerepet játszik, mivel körülötte osztjuk két részre az adathalmazt. A pivot elem kiválasztásának több módszere létezik, például választhatjuk az első, utolsó vagy középső elemet, illetve alkalmazhatunk véletlenszerű kiválasztást is.

2. **Elemek partícionálása:** Miután kiválasztottuk a pivot elemet, a következő lépésben partícionáljuk az adathalmazt két részre: egy olyan részre, amely kisebb vagy egyenlő a pivot elemnél, és egy olyan részre, amely nagyobb a pivot elemnél. Ezzel a lépéssel biztosítjuk, hogy minden elem a pivot elem bal oldalán kisebb vagy egyenlő a pivot elemnél, míg minden elem a jobb oldalán nagyobb a pivot elemnél.

3. **Rekurzív rendezés:** Miután a partícionálás megtörtént, a Quick Sort algoritmus rekurzív módon alkalmazza önmagát mindkét részre, azaz a pivot elemtől balra eső, illetve jobbra eső részekre. A rekurzív alkalmazás mindaddig folytatódik, amíg a részek egyetlen elemre nem redukálódnak, ami nyilvánvalóan rendezett.

4. **Kombináció:** Mivel a rekurzív alatt az adathalmaz részekre történt felosztása, illetve ezek önálló rendezése történt meg, az egyes részek kombinálásával áll elő a teljesen rendezett adathalmaz.

#### Algoritmus Implementáció

A Quick Sort algoritmusának implementációjára számos nyelvet lehet felhasználni, itt azonban C++ nyelven mutatjuk be a koncepciót. Nemcsak a gyorsaság, hanem a programozási nyelv eleganciája miatt is érdemes C++-ban gondolkodni, különösen olyan problémák esetén, ahol a memória kezelés és az optimalizálás kulcsfontosságú.

```cpp
#include <iostream>

#include <vector>

// Segédfüggvény a partícionáláshoz
int partition(std::vector<int>& arr, int low, int high) {
    int pivot = arr[high]; // A pivot elemet az utolsó elemnek választjuk
    int i = low - 1; // A kisebb elemek indexe

    for (int j = low; j <= high - 1; j++) {
        // Ha az aktuális elem kisebb vagy egyenlő a pivot elemnél
        if (arr[j] <= pivot) {
            i++; // Növeljük a kisebb elemek indexét
            std::swap(arr[i], arr[j]); // Cseréljük az elemeket
        }
    }
    std::swap(arr[i + 1], arr[high]); // Cseréljük a pivot elemet a megfelelő helyére
    return (i + 1);
}

// A Quick Sort algoritmus rekurzív függvénye
void quickSort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        // Partícionáljuk az adatokat
        int pi = partition(arr, low, high);

        // Rekurzív módon rendezzük a partíciókat
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int main() {
    std::vector<int> data = {10, 7, 8, 9, 1, 5};
    int n = data.size();

    // Meghívjuk a Quick Sort algoritmust
    quickSort(data, 0, n - 1);

    std::cout << "Sorted array: ";
    for (int i = 0; i < n; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

#### Algoritmus Optimalizálása és Variációk

A Quick Sort alapvető implementációjának több optimalizációs és variációs lehetősége is van. Ezek célja, hogy az algoritmus teljesítményét és hatékonyságát növeljék különböző helyzetekben.

- **Pivot Kiválasztás:** Az egyszerűbb módszerek mellett, mint a bal, jobb vagy középső elem kiválasztása, léteznek bonyolultabb technikák is, például a medián-of-three (három elem mediánjának kiválasztása) vagy véletlenszerű kiválasztás, amelyek javíthatják a pivot kiválasztásának hatékonyságát, csökkentve ezáltal a legrosszabb esetek előfordulásának valószínűségét.

- **Tail Rekurzió Optimalizálás:** A gyorsrendezés algoritmusban jellemzően két rekurzív hívás van. A tail rekurzió optimalizálás lényege, hogy az egyik rekurzív hívást iterációként kezeljük, ezáltal csökkentve a rekurzív hívások számát és a veremmélységet.

- **Iteratív Implementáció:** Bizonyos esetekben az iteratív megközelítés hatékonyabb lehet, különösen, ha mély rekurzív hívások számát minimalizálni kell. Ezen változatok általában egy explicit verem használnak a rekurzív hívások felváltására.

- **Kis Méretű Adatok Rendezése:** Kis méretű adatstruktúrák esetén a Quick Sort helyett más rendezési algoritmusok, például az insertion sort (beszúró rendezés) alkalmazása is szóba jöhet, mivel ezek az algoritmusok kis adatstruktúráknál hatékonyabbak lehetnek.

### 6.3.2 Teljesítményelemzés és -összehasonlítás

#### Bevezetés

A Gyorsrendezés az egyik leggyakrabban használt rendezési algoritmus, kiemelkedő teljesítményjellemzői miatt az átlagos esetben, valamint elegáns és egyértelmű rekurzív jellege miatt. Azonban a teljesítménye jelentősen változhat az elfogadott báziselem és a bemeneti adatok jellege függvényében. Ez a szakasz részletesen elemzi a Gyorsrendezés teljesítményjellemzőit, megvitatva az elméleti és gyakorlati szempontokat, valamint összehasonlítva más jól ismert rendezési algoritmusokkal.

#### Időbonyolultság elemzése

A Gyorsrendezés időbonyolultsága három fő esetre oszlik: legjobb eset, átlagos eset és legrosszabb eset. Ez a szakasz részletes elemzést nyújt minden egyes esetre.

##### Legjobb eset

A legjobb esetben a partícionálási folyamat mindig két egyenlő részre osztja a listát. Így az `n` elemű rekurziós fa mélysége `log(n)` lesz. Minden rekurziós fa szintje magába foglalja a lista partícionálását, ami `O(n)` időt igényel. Ezért a Gyorsrendezés legjobb esetének időbonyolultsága:

$$
T(n) = O(n \\log n)
$$

A legjobb eset általában egy jó báziselem kiválasztási stratégiával érhető el, mint például a tömb mediánjának kiválasztása vagy véletlenszerű báziselem használata.

##### Átlagos eset

A Gyorsrendezés átlagos esetének bonyolultsága szintén $O(n \\log n)$. Átlagosan minden partícionálási lépés nagyjából két egyenlő részre osztja a tömböt. Így a rekurziós fa mélysége ismét `log(n)` lesz, és a szinteken elvégzett munka `O(n)`.

Az átlagos eset matematikai elemzése magában foglalja a következő rekurziós reláció megoldását:

$$
T(n) = 2T(n/2) + O(n)
$$

Az Osztó-tétel alkalmazásával az oszd-meg-és-uralkodj rekurziókra:

$$
T(n) = O(n \\log n)
$$

A báziselem választásától való függése ellenére a Gyorsrendezés fenntartja az átlagos eset bonyolultságát.

##### Legrosszabb eset

A Gyorsrendezés legrosszabb esetének bonyolultsága akkor következik be, amikor a báziselem választás nagyon kiegyensúlyozatlan partíciókhoz vezet. Ez előfordulhat, ha a báziselem mindig a legkisebb vagy legnagyobb elemet választja, ami egy nagyon aszimmetrikus felosztást eredményez. Ebben az esetben a rekurziós fa mélysége `n` lesz, és minden szinten elvégzett munka `O(n)`, ami a legrosszabb eset időbonyolultságát eredményezi:

$$
T(n) = O(n^2)
$$

A legrosszabb eset elkerülése érdekében gyakran alkalmaznak különböző báziselem választási stratégiákat, mint például a véletlenszerű báziselem kiválasztás vagy a medián-3 módszer.

#### Tárbonyolultság elemzése

A Gyorsrendezés nagyrészt in-place működik, csak egy kis segédveremterületet igényel a rekurzióhoz. A tárbonyolultságot a rekurziós fa maximális mélysége határozza meg.

##### Legjobb és átlagos esetek

Mind a legjobb, mind az átlagos esetekben a rekurziós fa mélysége `log(n)`, ami a következő tárbonyolultságot eredményezi:

$$
S(n) = O(\log n)
$$

##### Legrosszabb eset

A legrosszabb esetben, amikor a rekurziós fa mélysége `n` lehet:

$$
S(n) = O(n)
$$

Ez a tárbonyolultság a rekurzív hívási verem tárolási szükségleteiből adódik.

#### Gyakorlati szempontok

Bár az elméleti elemzés szilárd alapot nyújt a Gyorsrendezés teljesítményének megértéséhez, a gyakorlati teljesítmény különbözhet különböző tényezők alapján, beleértve:

- **Báziselem kiválasztása**: Egy jó báziselem kiválasztása kulcsfontosságú. Stratégiák, mint például a három elem mediánjának használata vagy a véletlenszerű báziselem választás segíthet elkerülni a legrosszabb esetet.
- **Bemeneti adatok eloszlása**: A bemeneti adatok jellege (például már rendezett, fordított sorrendű, véletlenszerű) jelentősen befolyásolhatja a teljesítményt.
- **Hibrid algoritmusok**: A Gyorsrendezést gyakran kombinálják más algoritmusokkal, mint például a Beszúró rendezéssel kis részhalmazok esetén, ami javítja a teljesítményt.

#### Összehasonlítás más rendezési algoritmusokkal

A Gyorsrendezés teljesítményének megértése érdekében hasznos összehasonlítani más népszerű rendezési algoritmusokkal, mint például a Beszúró rendezés, a Buborékrendezés és az Összefésülő rendezés.

##### Beszúró rendezés

A Beszúró rendezés egyszerű és hatékony kis vagy majdnem rendezett adathalmazok esetén, de időbonyolultsága `O(n^2)` az általános esetben, ami sokkal lassabb, mint a Gyorsrendezés átlagos esetben nyújtott `O(n \\log n)` bonyolultsága.

##### Buborékrendezés

A Buborékrendezés szintén `O(n^2)` időbonyolultságú az átlagos és legrosszabb esetben, és általában kevésbé hatékony, mint a Gyorsrendezés. Bár a Buborékrendezés egyszerűen megvalósítható, nagyobb adathalmazok esetén nem praktikus.

##### Összefésülő rendezés

Az Összefésülő rendezés egy másik hatékony rendezési algoritmus, amely stabil és időbonyolultsága mindig `O(n \\log n)`, függetlenül a bemeneti adatok rendezésétől. Azonban az Összefésülő rendezés `O(n)` extra tárhelyet igényel, míg a Gyorsrendezés in-place, azaz nem igényel extra memóriát.
