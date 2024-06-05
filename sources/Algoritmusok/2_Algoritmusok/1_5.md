\newpage

## 1.5. Ugrásos keresés (Jump Search)

Az ugrásos keresés (Jump Search) egy hatékony keresési algoritmus, amelyet rendezett tömbökben alkalmaznak. Az algoritmus különlegessége, hogy nem lineárisan, hanem meghatározott lépésekben (ugrásokban) halad előre, így csökkentve a keresési időt. Az ugrásos keresés lényege, hogy előre meghatározott blokkokban ellenőrzi az elemeket, és ha megtalálja a keresett értéket egy adott blokkban, akkor lineáris kereséssel folytatja az adott blokkon belül. Ez az eljárás különösen hatékony nagy méretű, rendezett tömbök esetén, ahol a keresés gyorsabb lehet, mint a hagyományos lineáris keresés, de egyszerűbb és könnyebben implementálható, mint a bináris keresés. A következő alfejezetekben részletesen bemutatjuk az ugrásos keresés alapelveit, implementációját, optimalizálási lehetőségeit, valamint teljesítményének összehasonlítását más keresési algoritmusokkal.

### 1.5.1. Alapelvek és feltételek

Az ugrásos keresés (Jump Search) egy rendezett tömbökben történő keresési algoritmus, amely kombinálja a lineáris keresés egyszerűségét és a bináris keresés hatékonyságát. Az algoritmus a rendezett tömb egyes elemeit bizonyos lépésközönként vizsgálja, majd ha megtalálja az intervallumot, amelybe a keresett elem tartozik, lineáris kereséssel folytatja az adott intervallumban.

#### Alapelvek

Az ugrásos keresés alapelvei az alábbi lépésekben foglalhatók össze:

1. **Lépésköz meghatározása**: A keresési lépések távolságát általában a tömb méretének négyzetgyökeként határozzuk meg. Ez a lépésköz a tömb átlépésének optimális mérete, mivel minimalizálja az összehasonlítások számát. Ha $n$ a tömb mérete, akkor a lépésköz $\sqrt{n}$.

2. **Intervallum keresése**: Az algoritmus a tömb elemeit lépésközönként vizsgálja. Ha a keresett elem kisebb a jelenlegi elemnél, akkor az algoritmus visszalép egy intervallumnyi távolságra.

3. **Lineáris keresés az intervallumban**: Miután megtalálta az intervallumot, amelybe a keresett elem tartozik, az algoritmus lineárisan végigmegy az intervallum elemein, hogy megtalálja a pontos pozíciót.

#### Feltételek

Az ugrásos keresés megfelelő működéséhez bizonyos feltételeknek kell teljesülniük:

1. **Rendezett tömb**: Az ugrásos keresés csak rendezett tömbök esetén alkalmazható, mivel az algoritmus lényegében a rendezett elemek tulajdonságaira épít.

2. **Ismert tömbméret**: A tömb méretének ismertnek kell lennie, mivel a lépésközt a tömb méretének négyzetgyöke alapján határozzuk meg.

3. **Egyenletes eloszlás**: Bár az ugrásos keresés bármilyen rendezett tömb esetén használható, a legjobban egyenletesen elosztott elemekkel működik, ahol az elemek közötti különbségek közel azonosak.

#### Algoritmus részletei

Az ugrásos keresés algoritmusa a következő lépésekben hajtható végre:

1. Határozzuk meg a lépésközt: $\text{step} = \sqrt{n}$.
2. Kezdjük a keresést a tömb első elemétől, és minden lépésnél ugorjunk előre $\text{step}$ elemmel.
3. Ha a keresett elem kisebb, mint a jelenlegi elem, ugorjunk vissza egy lépésköznyit, és végezzünk lineáris keresést az adott intervallumban.
4. Ha a keresett elem nagyobb, mint a jelenlegi elem, folytassuk a keresést a következő lépésközön belül.
5. Ha elértük a tömb végét, és még mindig nem találtuk meg az elemet, akkor az elem nincs a tömbben.

#### Implementáció

Az alábbiakban bemutatjuk az ugrásos keresés egy C++ nyelvű implementációját:

```cpp
#include <iostream>
#include <cmath>

int jumpSearch(int arr[], int n, int x) {
    // Determine the optimal step size
    int step = sqrt(n);
    int prev = 0;

    // Find the block where element is present
    while (arr[std::min(step, n)-1] < x) {
        prev = step;
        step += sqrt(n);
        if (prev >= n) return -1; // If we reach the end of array
    }

    // Perform linear search within the block
    for (int i = prev; i < std::min(step, n); i++) {
        if (arr[i] == x) return i;
    }

    return -1; // Element not found
}

int main() {
    int arr[] = {0, 1, 2, 4, 6, 7, 9, 12, 14, 17, 19, 23, 26, 29, 32};
    int n = sizeof(arr) / sizeof(arr[0]);
    int x = 19;

    int index = jumpSearch(arr, n, x);

    if (index != -1) {
        std::cout << "Element " << x << " is at index " << index << std::endl;
    } else {
        std::cout << "Element " << x << " not found in array" << std::endl;
    }

    return 0;
}
```

#### Optimalizálás

Az ugrásos keresés hatékonysága a lépésköz megválasztásától függ. A négyzetgyök alapú lépésköz általában optimális, de egyes esetekben szükség lehet a lépésköz finomhangolására. Például, ha a tömb mérete ismert, és az elemek eloszlása egyenletes, akkor a négyzetgyök alapú lépésköz valóban optimális. Azonban, ha az elemek eloszlása nem egyenletes, akkor a lépésköz megválasztása módosítható a keresési hatékonyság javítása érdekében.

#### Összegzés

Az ugrásos keresés egy egyszerű, mégis hatékony algoritmus rendezett tömbök keresésére, amely egyesíti a lineáris és a bináris keresés előnyeit. Az algoritmus hatékonysága a lépésköz optimális megválasztásán alapul, amely a tömb méretének négyzetgyökeként van meghatározva. Az ugrásos keresés különösen hasznos nagy méretű rendezett tömbök esetén, ahol gyors és hatékony keresési lehetőséget biztosít.

### 1.5.2. Implementáció és optimalizálás

Az ugrásos keresés (Jump Search) implementációja és optimalizálása részletesen magában foglalja az algoritmus lépéseinek megértését, a kódolási folyamatot, valamint a teljesítmény maximalizálásának módszereit. Ebben az alfejezetben bemutatjuk az ugrásos keresés részletes implementációját C++ nyelven, és megvizsgáljuk az optimalizálás különböző technikáit.

#### Implementáció lépései

Az ugrásos keresés algoritmusa négy fő lépésből áll:

1. **Lépésköz meghatározása**: Az algoritmus első lépése a lépésköz (step size) meghatározása. Ez a lépésköz általában a tömb méretének négyzetgyöke. Ha $n$ a tömb mérete, akkor a lépésköz $\sqrt{n}$.

2. **Intervallum keresése**: Az algoritmus a tömb elemeit a meghatározott lépésközönként vizsgálja. Amíg a vizsgált elem kisebb a keresett elemtől, addig az algoritmus tovább lép. Ha a keresett elem kisebb, mint a vizsgált elem, akkor az algoritmus visszalép egy lépésköznyit.

3. **Lineáris keresés az intervallumban**: Ha megtaláltuk az intervallumot, ahol a keresett elem lehet, az algoritmus lineárisan végigmegy az intervallum elemein, hogy megtalálja a pontos helyet.

4. **Elem megtalálása vagy hiánya**: Ha az elem megtalálható az intervallumban, az algoritmus visszaadja az elem indexét. Ha az intervallum végéig nem találja meg, az algoritmus jelzi, hogy az elem nincs a tömbben.

#### Részletes implementáció C++ nyelven

Az alábbiakban bemutatjuk az ugrásos keresés algoritmusának részletes implementációját C++ nyelven:

```cpp
#include <iostream>
#include <cmath>
#include <algorithm>

int jumpSearch(int arr[], int n, int x) {
    // Determine the optimal step size
    int step = sqrt(n);
    int prev = 0;

    // Find the block where element is present (if it is present)
    while (arr[std::min(step, n) - 1] < x) {
        prev = step;
        step += sqrt(n);
        if (prev >= n) return -1; // If we've reached the end of array
    }

    // Linear search within the identified block
    for (int i = prev; i < std::min(step, n); i++) {
        if (arr[i] == x) return i;
    }

    return -1; // Element not found
}

int main() {
    int arr[] = {0, 1, 2, 4, 6, 7, 9, 12, 14, 17, 19, 23, 26, 29, 32};
    int n = sizeof(arr) / sizeof(arr[0]);
    int x = 19;

    int index = jumpSearch(arr, n, x);

    if (index != -1) {
        std::cout << "Element " << x << " is at index " << index << std::endl;
    } else {
        std::cout << "Element " << x << " not found in array" << std::endl;
    }

    return 0;
}
```

#### Optimalizálási technikák

Az ugrásos keresés optimalizálása számos tényezőtől függ, beleértve a lépésköz meghatározását, a tömb elrendezését, valamint a hardveres sajátosságokat.

1. **Lépésköz meghatározása**: Az algoritmus lépésköze általában a tömb méretének négyzetgyöke. Ez a választás az algoritmus optimális teljesítményét biztosítja, mivel minimalizálja az összehasonlítások számát. Azonban egyes esetekben a lépésköz finomhangolása szükséges lehet. Például, ha a tömb elemei nem egyenletesen oszlanak el, a lépésköz módosítása javíthatja a teljesítményt.

2. **Cache optimalizálás**: A modern számítógépek cache memóriája jelentős hatással lehet az algoritmus teljesítményére. Az ugrásos keresés során a cache memória hatékony kihasználása érdekében fontos, hogy az algoritmus a tömb elemeit együttesen vizsgálja. A cache-felhasználás optimalizálása érdekében célszerű a tömb elemeit úgy elrendezni, hogy a cache memóriában minél több elem férjen el egyszerre.

3. **Parallelizálás**: Az ugrásos keresés algoritmusának párhuzamosítása további teljesítménynövekedést eredményezhet. A keresési folyamat párhuzamos végrehajtása több processzormag segítségével lehetővé teszi az algoritmus gyorsabb végrehajtását. A párhuzamosítás során a tömb különböző részein történő keresést külön szálakra osztva végezhetjük el.

4. **Intervallum optimalizálás**: Az algoritmus intervallum keresési részének optimalizálása is fontos. Az intervallumon belüli lineáris keresés helyett alkalmazhatunk például bináris keresést az intervallumban, ha az intervallum elég nagy ahhoz, hogy a bináris keresés hatékonyabb legyen.

#### Összegzés

Az ugrásos keresés egy hatékony algoritmus, amely különösen jól működik rendezett tömbök esetén. Az algoritmus lépésközének megfelelő meghatározása, a cache memória hatékony kihasználása, a párhuzamos végrehajtás és az intervallum optimalizálás mind hozzájárulhatnak a keresési folyamat gyorsításához. Az alapos megértés és a különböző optimalizálási technikák alkalmazása lehetővé teszi az ugrásos keresés algoritmusának hatékony használatát a gyakorlatban.

