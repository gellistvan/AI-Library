\newpage

## 2.7. Shell Sort

A Shell Sort egy hatékony és rugalmas rendezési algoritmus, amelyet Donald Shell fejlesztett ki 1959-ben. Az algoritmus az egyszerűbb inzerciós rendezés általánosítása, amely a bemeneti tömb elemeit először nagyobb lépésekben rendezi, majd fokozatosan csökkenti ezeket a lépéseket, amíg el nem éri az alapvető inzerciós rendezést. Ezzel a módszerrel a Shell Sort képes a kezdetben távoli elemek közötti rendetlenséget csökkenteni, mielőtt a kisebb lépésekkel finomítaná a rendezést, így hatékonyabban oldja meg a nagyobb adathalmazok rendezését. A következő alfejezetekben megvizsgáljuk a Shell Sort alapelveit és implementációját, különböző hézag sorozatokat, mint például a Knuth és Sedgewick sorozatok, az algoritmus teljesítményének és komplexitásának elemzését, valamint gyakorlati alkalmazásait és példáit.

### 2.7.1. Alapelvek és implementáció

A Shell Sort, amelyet Donald Shell mutatott be 1959-ben, az egyik legrégebbi és legfontosabb adaptív rendezési algoritmus. Az algoritmus az egyszerű inzerciós rendezés továbbfejlesztése, és célja a rendezés hatékonyságának növelése azáltal, hogy csökkenti az elemek közötti áthelyezések számát. Az alábbiakban részletesen tárgyaljuk a Shell Sort alapelveit és implementációját.

#### Alapelvek

A Shell Sort alapötlete az, hogy a bemeneti tömb elemeit először nagyobb lépésekben, úgynevezett "hézagok" (gaps) szerint rendezi. Ez azt jelenti, hogy az algoritmus először a távoli elemeket rendezi, majd fokozatosan csökkenti a hézagok méretét, míg végül a hézag mérete el nem éri az 1-et. Amikor a hézag mérete 1, az algoritmus az inzerciós rendezést használja, amely ekkor már hatékonyabb lesz, mivel a tömb nagyrészt rendezett állapotban van.

Az algoritmus egyik kulcsfontosságú eleme a megfelelő hézagsorozat kiválasztása. A hézagsorozat meghatározza, hogy az algoritmus milyen lépésekben csökkenti a távolságot az elemek között. Több különböző hézagsorozat létezik, amelyek különböző teljesítményt nyújtanak különböző adathalmazokon. Néhány ismert hézagsorozat a következő:

- **Original Shell Sequence**: Az eredeti hézagsorozat, amelyet Shell javasolt, a $\left\lfloor \frac{n}{2} \right\rfloor$, $\left\lfloor \frac{n}{4} \right\rfloor$, ..., 1 sorozatot követi.
- **Knuth Sequence**: Ez a sorozat a $1, 4, 13, 40, 121, ...$ alakot követi, ahol a hézagokat a $h_i = 3h_{i-1} + 1$ képlet határozza meg.
- **Sedgewick Sequence**: Ez a sorozat a $1, 5, 19, 41, 109, ...$ alakot követi, és két különböző képletet használ a hézagok meghatározásához: $4^i + 3 \times 2^{i-1} + 1$ és $9 \times 4^{i-1} - 9 \times 2^{i-1} + 1$.

#### Implementáció

Az alábbiakban egy részletes implementáció található a Shell Sort algoritmusra C++ nyelven, amely az eredeti Shell által javasolt hézagsorozatot használja.

```cpp
#include <iostream>

#include <vector>

// Shell Sort function
void shellSort(std::vector<int>& arr) {
    int n = arr.size();
    
    // Start with a big gap, then reduce the gap
    for (int gap = n / 2; gap > 0; gap /= 2) {
        // Do a gapped insertion sort for this gap size.
        // The first gap elements arr[0..gap-1] are already in gapped order
        // keep adding one more element until the entire array is gap sorted
        for (int i = gap; i < n; i += 1) {
            // add arr[i] to the elements that have been gap sorted
            // save arr[i] in temp and make a hole at position i
            int temp = arr[i];

            // shift earlier gap-sorted elements up until the correct location for arr[i] is found
            int j;
            for (j = i; j >= gap && arr[j - gap] > temp; j -= gap)
                arr[j] = arr[j - gap];

            // put temp (the original arr[i]) in its correct location
            arr[j] = temp;
        }
    }
}

// Utility function to print an array
void printArray(const std::vector<int>& arr) {
    for (int i = 0; i < arr.size(); i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

// Driver code
int main() {
    std::vector<int> arr = {12, 34, 54, 2, 3};
    std::cout << "Array before sorting: \n";
    printArray(arr);

    shellSort(arr);

    std::cout << "Array after sorting: \n";
    printArray(arr);
    return 0;
}
```

#### Hézag sorozatok

Mint korábban említettük, a hézagsorozat kiválasztása kulcsfontosságú a Shell Sort hatékonysága szempontjából. Az eredeti Shell sorozat egyszerű és intuitív, de modern kutatások azt mutatják, hogy más sorozatok, mint például a Knuth vagy Sedgewick sorozatok, gyakran jobb teljesítményt nyújtanak.

- **Knuth sorozat**: A Knuth sorozat használatakor a hézagok sorozatát a $h = 1$ értéktől kezdve a $h = 3h + 1$ képlettel generáljuk, amíg a hézag értéke kisebb lesz, mint a tömb mérete. Ennek eredményeként a sorozat növekvő sorrendben lesz: 1, 4, 13, 40, 121, stb. A nagyobb hézagok gyorsabban csökkentik a rendetlenséget, míg a kisebb hézagok finomítják a rendezést.

- **Sedgewick sorozat**: A Sedgewick által javasolt sorozat két különböző képletet használ a hézagok meghatározásához. Az első képlet $h_k = 4^k + 3 \times 2^{k-1} + 1$ generálja a sorozatot, míg a második képlet $h_k = 9 \times 4^{k-1} - 9 \times 2^{k-1} + 1$ egy alternatív sorozatot ad. A Sedgewick sorozat előnye, hogy jól kiegyensúlyozza a nagy és kis hézagok közötti különbséget, így hatékonyabb rendezést tesz lehetővé.

#### Teljesítmény és komplexitás

A Shell Sort teljesítménye nagymértékben függ a választott hézagsorozattól. Az általános esetben a Shell Sort legrosszabb esetbeni időbonyolultsága $O(n^2)$, ami rosszabb, mint a gyorsabb rendezési algoritmusok, mint például a quicksort vagy a mergesort. Azonban megfelelő hézagsorozat kiválasztásával a Shell Sort gyakorlati teljesítménye lényegesen jobb lehet.

Az eredeti Shell által javasolt sorozat esetében a legrosszabb esetbeni időbonyolultság $O(n^2)$. A Knuth sorozat esetében a legrosszabb esetbeni időbonyolultság $O(n^{3/2})$, ami jelentős javulást jelent. A Sedgewick sorozat még tovább csökkenti a legrosszabb esetbeni időbonyolultságot, $O(n^{4/3})$-

re, ami még hatékonyabbá teszi az algoritmust bizonyos típusú adathalmazok esetén.

#### Algoritmus leírása

A Shell Sort algoritmus lépései a következők:

1. **Hézag sorozat előállítása**: A kiválasztott hézagsorozat alapján előállítjuk a különböző hézagokat. Például az eredeti Shell sorozat esetében a hézagok $n/2, n/4, ..., 1$.

2. **Hézagolt rendezés**: Minden hézagérték esetén a következőket tesszük:
    - **Inzerciós rendezés alkalmazása**: Az aktuális hézagnak megfelelően a tömb elemeit hézagolt inzerciós rendezéssel rendezzük. Ez azt jelenti, hogy minden hézagra külön-külön alkalmazzuk az inzerciós rendezést.

3. **Hézag csökkentése**: Csökkentjük a hézag méretét, és visszatérünk a hézagolt rendezés lépéshez. Addig folytatjuk, amíg a hézag mérete el nem éri az 1-et.

4. **Végső rendezés**: Amikor a hézag mérete 1, az algoritmus egyszerű inzerciós rendezést alkalmaz a teljes tömbre, amely ekkorra már nagyrészt rendezett állapotban van, így az inzerciós rendezés hatékonyabb lesz.

A fenti algoritmus végrehajtása során az elemek nagy távolságokra történő cseréje csökkenti az elemek közötti rendetlenséget, és segít minimalizálni a kisebb hézagokkal végzett rendezési lépések számát. Ez a módszer hatékonyabbá teszi a rendezést, különösen nagyobb adathalmazok esetében.

#### Példakód C++ nyelven (Knuth sorozattal)

Az alábbiakban egy Shell Sort implementáció található C++ nyelven, amely a Knuth sorozatot használja.

```cpp
#include <iostream>

#include <vector>

// Generate Knuth sequence
std::vector<int> generateKnuthSequence(int size) {
    std::vector<int> sequence;
    int h = 1;
    while (h < size) {
        sequence.push_back(h);
        h = 3 * h + 1;
    }
    return sequence;
}

// Shell Sort function with Knuth sequence
void shellSort(std::vector<int>& arr) {
    int n = arr.size();
    std::vector<int> gaps = generateKnuthSequence(n);

    for (int gap : gaps) {
        for (int i = gap; i < n; ++i) {
            int temp = arr[i];
            int j;
            for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
                arr[j] = arr[j - gap];
            }
            arr[j] = temp;
        }
    }
}

// Utility function to print an array
void printArray(const std::vector<int>& arr) {
    for (int i = 0; i < arr.size(); i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

// Driver code
int main() {
    std::vector<int> arr = {12, 34, 54, 2, 3};
    std::cout << "Array before sorting: \n";
    printArray(arr);

    shellSort(arr);

    std::cout << "Array after sorting: \n";
    printArray(arr);
    return 0;
}
```

Ez az implementáció a Knuth sorozatot használja a hézagok generálásához. Az `generateKnuthSequence` függvény előállítja a Knuth sorozatot a megadott méret alapján, majd az `shellSort` függvény ezt a sorozatot használja a rendezéshez.

#### Összefoglalás

A Shell Sort egy hatékony rendezési algoritmus, amely az inzerciós rendezés adaptációján alapul. Az algoritmus teljesítménye nagymértékben függ a választott hézagsorozattól. Bár a legrosszabb esetbeni időbonyolultsága általában rosszabb, mint a modern algoritmusoké, a gyakorlatban a megfelelő hézagsorozatok használatával a Shell Sort hatékony és praktikus megoldás lehet nagy adathalmazok rendezésére. A különböző hézagsorozatok, mint például az eredeti Shell, a Knuth és a Sedgewick sorozatok, különböző teljesítményt nyújtanak, és az adathalmaz specifikus jellemzőitől függően választhatók ki.

### 2.7.2. Hézag sorozatok (pl. Knuth, Sedgewick)

A Shell Sort hatékonyságát jelentősen befolyásolja a választott hézagsorozat, amely meghatározza, hogy milyen lépésekkel csökkentjük az elemek közötti távolságot a rendezés során. A megfelelő hézagsorozat kiválasztása kulcsfontosságú ahhoz, hogy az algoritmus hatékonyan működjön különböző adathalmazokon. Ebben az alfejezetben részletesen bemutatjuk a legismertebb hézagsorozatokat, beleértve a Knuth és Sedgewick sorozatokat, valamint más jelentős sorozatokat is.

#### Alapfogalmak és követelmények

A hézagsorozatokat többféle szempontból lehet értékelni, mint például a rendezési teljesítmény, az implementáció egyszerűsége, és az elméleti tulajdonságok. Egy jó hézagsorozatnak az alábbi követelményeknek kell megfelelnie:

- **Csökkenő sorrend**: A sorozat elemei csökkenő sorrendben kell legyenek, hogy az algoritmus a nagy hézagoktól haladjon a kisebbek felé.
- **Hézagok lefedése**: A sorozatnak le kell fednie minden pozitív egészet a legnagyobb hézagtól az 1-ig.
- **Növekvő hatékonyság**: A sorozatnak biztosítania kell, hogy a rendezés hatékonysága növekedjen, ahogy a hézag mérete csökken.

#### Knuth sorozat

A Knuth sorozat, amelyet Donald E. Knuth javasolt, az egyik legismertebb és leghatékonyabb hézagsorozat a Shell Sort algoritmushoz. A sorozat elemei a következő képlet szerint generálhatók:

$$
h_k = 3^k - 1
$$

Az első néhány elem a következőképpen alakul: 1, 4, 13, 40, 121, stb. Ez a sorozat geometriai növekedést mutat, amely biztosítja, hogy a rendezés során a nagy hézagok gyorsan csökkentsék a rendetlenséget, míg a kisebb hézagok finomítják a rendezést.

A Knuth sorozat előnyei:

- **Hatékonyság**: A nagyobb hézagok gyorsabban csökkentik a rendetlenséget, így kevesebb lépés szükséges a rendezéshez.
- **Egyszerű implementáció**: A sorozat generálása és alkalmazása egyszerű és gyors.

#### Sedgewick sorozat

A Sedgewick sorozatot Robert Sedgewick fejlesztette ki, és két különböző képletet használ a hézagok generálásához:

$$
h_k = 4^k + 3 \times 2^{k-1} + 1 \newline
h_k = 9 \times 4^{k-1} - 9 \times 2^{k-1} + 1
$$

Ez a sorozat különösen hatékony nagyobb adathalmazok esetében, mivel jól kiegyensúlyozza a nagy és kis hézagok közötti különbségeket. Az első néhány elem a következőképpen alakul: 1, 5, 19, 41, 109, stb.

A Sedgewick sorozat előnyei:

- **Kiegyensúlyozottság**: A sorozat kiegyensúlyozza a nagy és kis hézagokat, ami hatékonyabb rendezést tesz lehetővé.
- **Jobb teljesítmény**: Az algoritmus jobb teljesítményt nyújt a gyakorlatban, különösen nagy adathalmazok esetében.

#### Tokuda sorozat

A Tokuda sorozat, amelyet Norihiro Tokuda javasolt, egy másik hatékony hézagsorozat, amely a következő képlet alapján generálható:

$$
h_k = \left\lceil \frac{9 \times (9/4)^k}{5} \right\rceil
$$

Az első néhány elem: 1, 4, 9, 20, 46, stb. Ez a sorozat gyorsan növekvő hézagokat biztosít, amelyek hatékonyan csökkentik a rendetlenséget.

A Tokuda sorozat előnyei:

- **Nagy hézagok**: A sorozat gyorsan növekvő hézagokat biztosít, amelyek hatékonyan csökkentik a kezdeti rendetlenséget.
- **Hatékony rendezés**: Az algoritmus gyorsan rendez nagy adathalmazokat.

#### Hibbard sorozat

A Hibbard sorozat, amelyot T. N. Hibbard javasolt, egy egyszerűbb hézagsorozat, amely a következőképpen generálható:

$$
h_k = 2^k - 1
$$

Az első néhány elem: 1, 3, 7, 15, 31, stb. Bár ez a sorozat egyszerű és könnyen implementálható, nem mindig nyújt optimális teljesítményt.

A Hibbard sorozat előnyei:

- **Egyszerűség**: A sorozat generálása és implementálása egyszerű.
- **Gyors kezdeti csökkenés**: A sorozat gyorsan csökkenti a kezdeti rendetlenséget.

#### Praktikus megfontolások

A különböző hézagsorozatok közötti választás során figyelembe kell venni az adott alkalmazás sajátosságait és a rendezendő adathalmaz jellemzőit. Néhány praktikus megfontolás:

- **Adathalmaz mérete**: Nagyobb adathalmazok esetében a Sedgewick és Tokuda sorozatok általában jobb teljesítményt nyújtanak.
- **Adathalmaz szerkezete**: Az adathalmaz kezdeti rendezettsége befolyásolhatja a hézagsorozat hatékonyságát. Például, ha az adathalmaz kezdetben nagyrészt rendezett, a kisebb hézagokkal rendelkező sorozatok hatékonyabbak lehetnek.
- **Implementációs egyszerűség**: Egyes sorozatok egyszerűbben implementálhatók, ami fontos lehet, ha az implementáció gyors és egyszerű megvalósítása a cél.

#### Példakód C++ nyelven (Sedgewick sorozattal)

Az alábbiakban egy Shell Sort implementáció található C++ nyelven, amely a Sedgewick sorozatot használja.

```cpp
#include <iostream>

#include <vector>

// Generate Sedgewick sequence
std::vector<int> generateSedgewickSequence(int size) {
    std::vector<int> sequence;
    int k = 0;
    int h;
    while (true) {
        if (k % 2 == 0) {
            h = 9 * (1 << (2 * k)) - 9 * (1 << k) + 1;
        } else {
            h = (1 << (2 * k + 1)) + 3 * (1 << (k + 1)) - 1;
        }
        if (h >= size) break;
        sequence.push_back(h);
        k++;
    }
    return sequence;
}

// Shell Sort function with Sedgewick sequence
void shellSort(std::vector<int>& arr) {
    int n = arr.size();
    std::vector<int> gaps = generateSedgewickSequence(n);

    for (int gap : gaps) {
        for (int i = gap; i < n; ++i) {
            int temp = arr[i];
            int j;
            for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
                arr[j] = arr[j - gap];
            }
            arr[j] = temp;
        }
    }
}

// Utility function to print an array
void printArray(const std::vector<int>& arr) {
    for (int i = 0; i < arr.size(); i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

// Driver code
int main() {
    std::vector<int> arr = {12, 34, 54, 2, 3};
    std::cout << "Array before sorting: \n";
    printArray(arr);

    shellSort(arr);

    std::cout << "Array after sorting: \n";
    printArray(arr);
    return 0;
}
```

Ez az implementáció a Sedgewick sorozatot használja a hézagok generálásához. Az `generateSedgewickSequence` függvény előállítja a Sedgewick sorozatot a megadott méret alapján, majd az `shellSort` függvény ezt a sorozatot használja a rendezéshez.

#### Összefoglalás

A hézagsorozatok kiválasztása alapvetően meghatározza a Shell Sort algoritmus hatékonyságát. A különböző sorozatok különböző előnyökkel és hátrányokkal rendelkeznek, és a választás során figyelembe kell venni az adathalmaz sajátosságait és a kívánt teljesítményt. A Knuth, Sedgewick, Tokuda és Hibbard sorozatok mind jól ismertek és széles körben használtak, de a legjobb eredmény elérése érdekében érdemes kísérletezni többféle sorozattal is.

### 2.7.3. Teljesítmény és komplexitás elemzése

A Shell Sort algoritmus teljesítménye és komplexitása az egyik legérdekesebb és legösszetettebb témakör a rendezési algoritmusok között. Az algoritmus hatékonysága nagymértékben függ a választott hézagsorozattól, és bár az átlagos esetben gyorsabb lehet, mint az egyszerűbb rendezési algoritmusok, a legrosszabb esetben mégis alulmaradhat más modern algoritmusokkal szemben. Ebben a fejezetben részletesen megvizsgáljuk a Shell Sort teljesítményét, időbonyolultságát, és a különböző hézagsorozatok hatását.

#### Alapfogalmak

Mielőtt a részletekbe mennénk, fontos megérteni néhány alapvető fogalmat:

- **Időbonyolultság**: Az algoritmus futási idejének mértéke a bemenet méretének függvényében. Az időbonyolultságot gyakran aszimptotikus notációban, például $O(n^2)$, $O(n \log n)$ stb. formában adjuk meg.
- **Hézagsorozat**: A hézagsorozat határozza meg, hogy a Shell Sort milyen lépésekben rendezi a tömb elemeit. A sorozat elemei csökkenő sorrendben vannak, és az algoritmus ezen hézagok szerint rendezi a tömböt.
- **Átlagos eset**: Az algoritmus futási ideje egy tipikus bemenetre.
- **Legrosszabb eset**: Az algoritmus futási ideje a lehető legkedvezőtlenebb bemenetre.

#### Hézagsorozatok hatása

A Shell Sort teljesítményének elemzése során kiemelten fontos szerepet játszik a hézagsorozat kiválasztása. Az alábbiakban néhány ismert hézagsorozat időbonyolultságát és teljesítményét vizsgáljuk meg.

##### Eredeti Shell sorozat

Az eredeti Shell által javasolt hézagsorozat a következőképpen néz ki: $\left\lfloor \frac{n}{2} \right\rfloor, \left\lfloor \frac{n}{4} \right\rfloor, ..., 1$. Ennek a sorozatnak a legrosszabb esetbeni időbonyolultsága $O(n^2)$. Ez azt jelenti, hogy az eredeti sorozat nem nyújt jelentős javulást az egyszerű inzerciós rendezéshez képest, különösen nagyobb adathalmazok esetében.

##### Knuth sorozat

A Knuth sorozat, amely a $h_k = 3h_{k-1} + 1$ képlet szerint generálódik, jelentős javulást nyújt az eredeti Shell sorozathoz képest. Ennek a sorozatnak a legrosszabb esetbeni időbonyolultsága $O(n^{3/2})$, ami lényegesen jobb, mint az eredeti sorozat $O(n^2)$ időbonyolultsága. Az átlagos esetben a Knuth sorozat hatékonyabb rendezést tesz lehetővé, mivel a nagyobb hézagok gyorsabban csökkentik a rendetlenséget.

##### Sedgewick sorozat

A Sedgewick sorozat két különböző képletet használ a hézagok generálásához: $h_k = 4^k + 3 \times 2^{k-1} + 1$ és $h_k = 9 \times 4^{k-1} - 9 \times 2^{k-1} + 1$. Ez a sorozat tovább javítja a Shell Sort teljesítményét, és a legrosszabb esetbeni időbonyolultsága $O(n^{4/3})$. Az átlagos esetben a Sedgewick sorozat még jobb teljesítményt nyújt, különösen nagyobb adathalmazok esetében.

##### Tokuda sorozat

A Tokuda sorozat a $h_k = \left\lceil \frac{9 \times (9/4)^k}{5} \right\rceil$ képlet alapján generálódik, és a legrosszabb esetbeni időbonyolultsága $O(n^{4/3})$. Ez a sorozat hasonló teljesítményt nyújt, mint a Sedgewick sorozat, de bizonyos esetekben még hatékonyabb lehet a gyakorlati alkalmazásokban.

#### Teljesítmény és komplexitás elemzése

A különböző hézagsorozatok teljesítményének elemzése során az alábbi szempontokat kell figyelembe venni:

##### Időbonyolultság

Az időbonyolultság az egyik legfontosabb mérőszám az algoritmusok hatékonyságának értékelésekor. A Shell Sort időbonyolultsága nagymértékben függ a választott hézagsorozattól. Az alábbi táblázat összefoglalja néhány ismert hézagsorozat időbonyolultságát:

| Hézagsorozat | Legrosszabb esetbeni időbonyolultság | Átlagos esetbeni időbonyolultság |
|--------------|--------------------------------------|---------------------------------|
| Eredeti Shell | $O(n^2)$ | $O(n \log^2 n)$ |
| Knuth | $O(n^{3/2})$ | $O(n \log n)$ |
| Sedgewick | $O(n^{4/3})$ | $O(n \log n)$ |
| Tokuda | $O(n^{4/3})$ | $O(n \log n)$ |

##### Térbeli komplexitás

A Shell Sort térbeli komplexitása általában $O(1)$, mivel az algoritmus nem igényel jelentős mennyiségű extra memóriát a rendezés során. Az összes művelet in-place történik, ami azt jelenti, hogy az algoritmus csak egy állandó mennyiségű extra memóriát használ, függetlenül a bemenet méretétől.

##### Gyakorlati teljesítmény

A gyakorlati teljesítmény értékelése során figyelembe kell venni az algoritmus viselkedését különböző típusú és méretű adathalmazok esetében. A hézagsorozat kiválasztása befolyásolja az algoritmus stabilitását, robusztusságát és általános hatékonyságát.

- **Kis adathalmazok**: Kis méretű adathalmazok esetében az algoritmus minden hézagsorozat mellett hatékonyan működik. Azonban a Sedgewick és Tokuda sorozatok általában jobb teljesítményt nyújtanak, mivel gyorsabban csökkentik a rendetlenséget.
- **Nagy adathalmazok**: Nagy méretű adathalmazok esetében a Knuth, Sedgewick és Tokuda sorozatok jelentős előnyt nyújtanak az eredeti Shell sorozathoz képest. Ezek a sorozatok hatékonyabban csökkentik a kezdeti rendetlenséget, és kevesebb lépés szükséges a teljes rendezéshez.
- **Különböző adathalmaz szerkezetek**: A hézagsorozat kiválasztása befolyásolja az algoritmus viselkedését különböző adathalmaz szerkezetek esetében. Például, ha az adathalmaz kezdetben nagyrészt rendezett, a kisebb hézagokkal rendelkező sorozatok hatékonyabbak lehetnek.

#### Példakód C++ nyelven (Tokuda sorozattal)

Az alábbiakban egy Shell Sort implementáció található C++ nyelven, amely a Tokuda sorozatot használja.

```cpp
#include <iostream>

#include <vector>
#include <cmath>

// Generate Tokuda sequence
std::vector<int> generateTokudaSequence(int size) {
    std::vector<int> sequence;
    int k = 0;
    while (true) {
        int h = std::ceil((9 * std::pow(9.0 / 4.0, k)) / 5);
        if (h >= size) break;
        sequence.push_back(h);
        k++;
    }
    return sequence;
}

// Shell Sort function with Tokuda sequence
void shellSort(std::vector<int>& arr) {
    int n = arr.size();
    std::vector<int> gaps = generateTokudaSequence(n);

    for (int gap : gaps) {
        for (int i = gap; i < n; ++i) {
            int temp = arr[i];
            int j;
            for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
                arr[j] = arr[j - gap];
            }
            arr[j] = temp;
        }
    }
}

// Utility function to print an array
void printArray(const std::vector<int>& arr) {
    for (int i = 0; i < arr.size(); i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

// Driver code
int main() {
    std::vector<int> arr = {12, 34, 54, 2, 3};
    std::cout << "Array before sorting: \n";
    printArray(arr);

    shellSort(arr);

    std::cout << "Array after sorting: \n";
    printArray(arr);
    return 0;
}
```

Ez az implementáció a Tokuda sorozatot használja a hézagok generálásához. Az `generateTokudaSequence` függvény előállítja a Tokuda sorozatot a megadott méret alapján, majd az `shellSort` függvény ezt a sorozatot használja a rendezéshez.

#### Összefoglalás

A Shell Sort teljesítménye és komplexitása nagymértékben függ a választott hézagsorozattól. Az eredeti Shell sorozat viszonylag egyszerű, de nem nyújt optimális teljesítményt. A Knuth, Sedgewick és Tokuda sorozatok jelentős javulást hoznak az algoritmus hatékonyságában, különösen nagyobb adathalmazok esetében. Az időbonyolultság és a gyakorlati teljesítmény elemzése során figyelembe kell venni az adathalmaz méretét és szerkezetét, valamint az implementáció egyszerűségét és hatékonyságát. A megfelelő hézagsorozat kiválasztásával a Shell Sort hatékony és praktikus rendezési algoritmus lehet számos alkalmazásban.

### 2.7.4. Gyakorlati alkalmazások és példák

A Shell Sort, mint hatékony és adaptív rendezési algoritmus, számos gyakorlati alkalmazási területtel rendelkezik. Ez az alfejezet részletesen bemutatja a Shell Sort gyakorlati felhasználási módjait, valamint példákon keresztül szemlélteti, hogyan alkalmazható az algoritmus különböző helyzetekben. Emellett ismertetünk néhány példakódot is C++ nyelven, hogy jobban megértsük az algoritmus gyakorlati implementációját.

#### Gyakorlati alkalmazások

A Shell Sort előnyei különösen akkor mutatkoznak meg, ha a rendezendő adathalmaz mérete közepes nagyságú, vagy ha a kezdeti rendetlenség mértéke nem túl nagy. Az alábbiakban bemutatunk néhány gyakorlati alkalmazási területet:

##### Beágyazott rendszerek

Beágyazott rendszerekben gyakran szükség van hatékony és memóriahatékony rendezési algoritmusokra. A Shell Sort alacsony térbeli komplexitása miatt (O(1) extra memóriát használ) kiválóan alkalmas ilyen környezetekben. Például, amikor egy mikrokontrolleren futó programnak adatokat kell rendeznie valós időben, a Shell Sort hatékony és megbízható megoldást nyújt.

##### Szoftverfejlesztés

Számos szoftverfejlesztési feladat során szükség van rendezési műveletekre, például adatok előkészítése jelentésekhez, listák rendezése felhasználói interfészekben, vagy háttérfolyamatok optimalizálása érdekében. A Shell Sort gyorsabb, mint az egyszerű inzerciós rendezés, különösen akkor, ha a bemeneti adathalmaz nagymértékben rendezett. Ezért gyakran használják olyan szoftverkomponensekben, ahol a teljesítmény kritikus fontosságú.

##### Oktatás

Az algoritmusok és adatszerkezetek oktatásában a Shell Sort fontos szerepet játszik. Az algoritmus egyszerűsége és hatékonysága miatt könnyen érthető és jól demonstrálható a diákok számára. Emellett a különböző hézagsorozatok vizsgálata lehetőséget ad arra, hogy a diákok mélyebb ismereteket szerezzenek az algoritmusok teljesítményének és komplexitásának elemzéséről.

#### Példák

Az alábbiakban néhány gyakorlati példát mutatunk be a Shell Sort algoritmus alkalmazására, valamint C++ nyelvű kódpéldákat is mellékelünk a szemléltetés érdekében.

##### 1. Adatok rendezése beágyazott rendszerben

Képzeljünk el egy beágyazott rendszert, amely hőmérséklet-érzékelő adatokat gyűjt. Az adatokat rendezni kell, mielőtt elküldenénk őket egy központi feldolgozó egységnek. A Shell Sort alacsony memóriaigénye miatt ideális választás erre a feladatra.

```cpp
#include <iostream>

#include <vector>

// Shell Sort function using a simple gap sequence
void shellSort(std::vector<float>& arr) {
    int n = arr.size();
    for (int gap = n / 2; gap > 0; gap /= 2) {
        for (int i = gap; i < n; ++i) {
            float temp = arr[i];
            int j;
            for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
                arr[j] = arr[j - gap];
            }
            arr[j] = temp;
        }
    }
}

int main() {
    std::vector<float> temperatures = {23.4, 22.1, 25.6, 21.9, 24.3};
    std::cout << "Temperatures before sorting: ";
    for (float temp : temperatures) {
        std::cout << temp << " ";
    }
    std::cout << std::endl;

    shellSort(temperatures);

    std::cout << "Temperatures after sorting: ";
    for (float temp : temperatures) {
        std::cout << temp << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

##### 2. Felhasználói interfész lista rendezése

Egy alkalmazásban, amelyben a felhasználóknak rendezett listákat kell megjeleníteni, a Shell Sort gyors és hatékony megoldást nyújt. Képzeljünk el egy kontaktlistát, amelyet a felhasználónak név szerint kell rendezni.

```cpp
#include <iostream>

#include <vector>
#include <string>

// Shell Sort function for strings
void shellSort(std::vector<std::string>& arr) {
    int n = arr.size();
    for (int gap = n / 2; gap > 0; gap /= 2) {
        for (int i = gap; i < n; ++i) {
            std::string temp = arr[i];
            int j;
            for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
                arr[j] = arr[j - gap];
            }
            arr[j] = temp;
        }
    }
}

int main() {
    std::vector<std::string> contacts = {"Alice", "Bob", "Charlie", "David", "Eve"};
    std::cout << "Contacts before sorting: ";
    for (const std::string& contact : contacts) {
        std::cout << contact << " ";
    }
    std::cout << std::endl;

    shellSort(contacts);

    std::cout << "Contacts after sorting: ";
    for (const std::string& contact : contacts) {
        std::cout << contact << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

##### 3. Nagy adatbázis rendezése

Amikor egy nagy adatbázis rekordjait kell rendezni, például egy könyvtár katalógusát, a Shell Sort hatékonyan alkalmazható. A következő példa bemutatja, hogyan rendezhetjük egy könyvtár könyveit a címük alapján.

```cpp
#include <iostream>

#include <vector>
#include <string>

// Shell Sort function for library books
void shellSort(std::vector<std::string>& books) {
    int n = books.size();
    for (int gap = n / 2; gap > 0; gap /= 2) {
        for (int i = gap; i < n; ++i) {
            std::string temp = books[i];
            int j;
            for (j = i; j >= gap && books[j - gap] > temp; j -= gap) {
                books[j] = books[j - gap];
            }
            books[j] = temp;
        }
    }
}

int main() {
    std::vector<std::string> library = {
        "The Catcher in the Rye", "To Kill a Mockingbird", "1984",
        "Pride and Prejudice", "The Great Gatsby"
    };
    std::cout << "Library before sorting: ";
    for (const std::string& book : library) {
        std::cout << book << " | ";
    }
    std::cout << std::endl;

    shellSort(library);

    std::cout << "Library after sorting: ";
    for (const std::string& book : library) {
        std::cout << book << " | ";
    }
    std::cout << std::endl;

    return 0;
}
```

#### Hézagsorozatok alkalmazása a gyakorlatban

A gyakorlatban a hézagsorozatok megválasztása jelentős hatással lehet a Shell Sort hatékonyságára. Az alábbiakban bemutatunk néhány gyakorlati példát, amelyek különböző hézagsorozatokat alkalmaznak.

##### Knuth sorozat alkalmazása

A Knuth sorozat használata hatékony lehet közepes és nagy méretű adathalmazok esetében.

```cpp
#include <iostream>

#include <vector>

// Generate Knuth sequence
std::vector<int> generateKnuthSequence(int size) {
    std::vector<int> sequence;
    int h = 1;
    while (h < size) {
        sequence.push_back(h);
        h = 3 * h + 1;
    }
    return sequence;
}

// Shell Sort function with Knuth sequence
void shellSortKnuth(std::vector<int>& arr) {
    int n = arr.size();
    std::vector<int> gaps = generateKnuthSequence(n);

    for (int gap : gaps) {
        for (int i = gap; i < n; ++i) {
            int temp = arr[i];
            int j;
            for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
                arr[j] = arr[j - gap];
            }
            arr[j] = temp;
        }
    }
}

int main() {
    std::vector<int> data = {45, 23, 53, 12, 34, 6, 78, 33};
    std::cout << "Data before sorting: ";
    for (int val : data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    shellSortKnuth(data);

    std::cout << "Data after sorting: ";
    for (int val : data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

##### Sedgewick sorozat alkalmazása

A Sedgewick sorozat különösen hatékony nagy adathalmazok esetén, mivel jobban kiegyensúlyozza a nagy és kis hézagok közötti különbséget.

```cpp
#include <iostream>

#include <vector>

// Generate Sedgewick sequence
std::vector<int> generateSedgewickSequence(int size) {
    std::vector<int> sequence;
    int k = 0;
    int h;
    while (true) {
        if (k % 2 == 0) {
            h = 9 * (1 << (2 * k)) - 9 * (1 << k) + 1;
        } else {
            h = (1 << (2 * k + 1)) + 3 * (1 << (k + 1)) - 1;
        }
        if (h >= size) break;
        sequence.push_back(h);
        k++;
    }
    return sequence;
}

// Shell Sort function with Sedgewick sequence
void shellSortSedgewick(std::vector<int>& arr) {
    int n = arr.size();
    std::vector<int> gaps = generateSedgewickSequence(n);

    for (int gap : gaps) {
        for (int i = gap; i < n; ++i) {
            int temp = arr[i];
            int j;
            for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
                arr[j] = arr[j - gap];
            }
            arr[j] = temp;
        }
    }
}

int main() {
    std::vector<int> data = {45, 23, 53, 12, 34, 6, 78, 33};
    std::cout << "Data before sorting: ";
    for (int val : data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    shellSortSedgewick(data);

    std::cout << "Data after sorting: ";
    for (int val : data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

#### Összefoglalás

A Shell Sort egy hatékony és adaptív rendezési algoritmus, amely számos gyakorlati alkalmazási területtel rendelkezik. A beágyazott rendszerektől kezdve a szoftverfejlesztésen át az oktatásig, a Shell Sort sokféle környezetben nyújt megbízható és gyors rendezési megoldást. A hézagsorozatok megválasztása jelentős hatással van az algoritmus teljesítményére, és a különböző sorozatok különböző előnyöket kínálnak. A gyakorlatban a Knuth és Sedgewick sorozatok gyakran jobb teljesítményt nyújtanak, különösen nagyobb adathalmazok esetében. A példakódok bemutatják, hogyan lehet a Shell Sortot különböző helyzetekben alkalmazni, és hogyan lehet különböző hézagsorozatokat használni a rendezés hatékonyságának növelése érdekében.
