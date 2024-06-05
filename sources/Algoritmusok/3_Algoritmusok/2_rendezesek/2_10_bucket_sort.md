\newpage

## 2.10. Bucket Sort

A Bucket Sort egy hatékony és gyakran használt rendezési algoritmus, amely különösen jól teljesít bizonyos típusú adatoknál. Az algoritmus alapvető elképzelése, hogy az adatokat több kisebb részre, úgynevezett "vödrökre" osztjuk, majd ezeket a vödröket külön-külön rendezzük. A Bucket Sort sajátossága, hogy nem minden esetben használható optimálisan, de bizonyos feltételek mellett, például egyenletes eloszlású adatok esetén, kimagasló teljesítményt nyújt. Ebben a fejezetben részletesen bemutatjuk a Bucket Sort alapelveit és implementációját, megvizsgáljuk, hogyan kezel különböző adateloszlásokat, elemezzük a teljesítményét és komplexitását, valamint gyakorlati alkalmazásokon keresztül szemléltetjük a használatát.

### 2.10.1. Alapelvek és implementáció

A Bucket Sort, más néven Bin Sort, egy disztribúciós rendezési algoritmus, amely kifejezetten jól teljesít egyenletesen eloszlott adatok esetén. Az algoritmus fő gondolata az, hogy az adatokat kisebb csoportokra, vödrökre (bucketekre) bontjuk, majd ezeket a vödröket külön-külön rendezzük, végül pedig összeillesztjük a rendezett vödröket egy végleges, rendezett listába.

#### Alapelvek

A Bucket Sort algoritmus lépései a következők:

1. **Vödrök létrehozása:** Meghatározzuk a megfelelő számú vödröt. A vödrök száma gyakran az adatok számának egy közelítő értéke vagy az adatok eloszlásától függ.
2. **Adatok elosztása a vödrökbe:** Minden adatot a megfelelő vödörbe helyezünk egy hash függvény vagy más elosztási stratégia segítségével. Az ideális eset az, hogy minden vödörbe hasonló számú elem kerül.
3. **Vödrök rendezése:** Minden egyes vödröt külön-külön rendezünk. Ehhez használhatunk más rendezési algoritmusokat, például gyorsrendezést (Quick Sort) vagy beszúrásos rendezést (Insertion Sort).
4. **Rendezett vödrök összefűzése:** A rendezett vödrök tartalmát összefűzzük, hogy megkapjuk a végleges rendezett listát.

#### Implementáció

A következő implementáció bemutatja a Bucket Sort algoritmust C++ nyelven:

```cpp
#include <iostream>

#include <vector>
#include <algorithm>

void bucketSort(std::vector<float>& arr) {
    int n = arr.size();
    
    // 1. Vödrök létrehozása
    std::vector<std::vector<float>> buckets(n);

    // 2. Adatok elosztása a vödrökbe
    for (int i = 0; i < n; i++) {
        int bucketIndex = n * arr[i]; // Assumption: arr[i] in range [0, 1)
        buckets[bucketIndex].push_back(arr[i]);
    }

    // 3. Vödrök rendezése
    for (int i = 0; i < n; i++) {
        std::sort(buckets[i].begin(), buckets[i].end());
    }

    // 4. Rendezett vödrök összefűzése
    int index = 0;
    for (int i = 0; i < n; i++) {
        for (size_t j = 0; j < buckets[i].size(); j++) {
            arr[index++] = buckets[i][j];
        }
    }
}

int main() {
    std::vector<float> arr = {0.42, 0.32, 0.23, 0.52, 0.25, 0.47, 0.51};
    
    std::cout << "Original array: ";
    for (float num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    bucketSort(arr);
    
    std::cout << "Sorted array: ";
    for (float num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

#### Alapelvek részletes magyarázata

1. **Vödrök létrehozása:** A vödrök számának megválasztása kritikus lépés a Bucket Sort algoritmus hatékonysága szempontjából. Általános gyakorlat, hogy a vödrök számát az adatok számával egyezőnek választjuk, de ezt az adatok eloszlása is befolyásolhatja.

2. **Adatok elosztása a vödrökbe:** Minden adatot egy megfelelő vödörbe helyezünk egy egyszerű hash függvény segítségével, amely meghatározza, hogy az adott adat melyik vödörbe kerüljön. Például, ha az adatok a [0, 1) intervallumban helyezkednek el, az `n * arr[i]` képlet alapján helyezzük el az adatokat a vödrökbe, ahol `n` a vödrök száma.

3. **Vödrök rendezése:** Miután az adatokat a vödrökbe helyeztük, minden egyes vödröt külön rendezünk. Az, hogy milyen rendezési algoritmust használunk, a vödörökben található adatok számától és eloszlásától függ. Kis mennyiségű adat esetén a beszúrásos rendezés (Insertion Sort) hatékony lehet, míg nagyobb mennyiségű adat esetén más, gyorsabb algoritmusok, mint a gyorsrendezés (Quick Sort), alkalmazása lehet előnyös.

4. **Rendezett vödrök összefűzése:** Végül a rendezett vödrök tartalmát összefűzzük egy végleges, rendezett listába. Ez a lépés egyszerűen az összes vödör tartalmának egyesítését jelenti.

#### Az algoritmus hatékonysága és komplexitása

A Bucket Sort algoritmus időbeli komplexitása a vödrök számától és az egyes vödrökben található elemek számától függ. Ideális esetben, ha az adatok egyenletesen oszlanak el a vödrök között, az időbeli komplexitás O(n + k), ahol n az adatok száma, k pedig a vödrök száma. A legrosszabb esetben azonban, ha az adatok nem oszlanak el egyenletesen, az időbeli komplexitás $O(n^2)$ is lehet.

Az algoritmus térbeli komplexitása O(n + k), mivel szükség van a vödrök számára és az adatok tárolására szolgáló helyre.

#### Példa és gyakorlati alkalmazások

A Bucket Sort algoritmus gyakran alkalmazható olyan esetekben, amikor az adatok eloszlása ismert és egyenletes, például:
- Floating-point számok rendezése, amelyek egy adott intervallumban egyenletesen oszlanak el.
- Nagy mennyiségű adatok rendezése adatbázisokban, ahol az adatok előzetesen szűrhetők vagy csoportosíthatók.

Összefoglalva, a Bucket Sort egy hatékony és jól alkalmazható rendezési algoritmus, amely különösen hasznos lehet bizonyos típusú adatstruktúrák esetén. Az alapelvek és az implementáció megértése segíthet abban, hogy a megfelelő körülmények között optimálisan alkalmazzuk ezt az algoritmust.

### 2.10.2. Különböző adateloszlások kezelése

A Bucket Sort algoritmus hatékonysága nagymértékben függ az adatok eloszlásától. Ideális esetben az adatok egyenletesen oszlanak el a vödrök között, azonban a gyakorlatban különböző eloszlásokkal találkozhatunk, amelyek befolyásolják az algoritmus teljesítményét. Ebben az alfejezetben részletesen megvizsgáljuk, hogyan kezeli a Bucket Sort különböző adateloszlásokat, és milyen technikákat alkalmazhatunk az algoritmus optimalizálására.

#### Egyenletes eloszlás

Az egyenletes eloszlás az, amikor az adatok egyenletesen oszlanak el egy adott intervallumban. Ebben az esetben a Bucket Sort optimálisan működik, mivel minden vödörbe nagyjából azonos számú elem kerül.

Példa: Tegyük fel, hogy az adatok a [0, 1) intervallumban egyenletesen oszlanak el. Ha 10 vödröt hozunk létre, akkor minden vödörbe várhatóan körülbelül azonos számú adat kerül, ami biztosítja a rendezési lépések hatékonyságát.

#### Normális (Gauss-) eloszlás

Normális eloszlás esetén az adatok nagy része az átlag körül koncentrálódik, és a szórás határozza meg, hogy az adatok milyen széles sávban oszlanak el. Ebben az esetben érdemes az adatok eloszlásának megfelelően meghatározni a vödrök határait, hogy elkerüljük a vödrök túltöltését az átlag környékén.

Technika: Ha tudjuk, hogy az adatok normális eloszlásúak, érdemes több vödröt létrehozni az átlag körül, míg a távolabbi régiókban kevesebb vödör is elegendő lehet. Ezzel csökkenthetjük az átlag környékén található vödrök méretét és a rendezési lépések időigényét.

#### Torzított eloszlás

Torzított eloszlás esetén az adatok egy része egy adott intervallumra koncentrálódik, míg a többi rész sokkal ritkábban fordul elő. Ez az eloszlás nehezíti a vödrök egyenletes feltöltését, ami az algoritmus hatékonyságának csökkenéséhez vezethet.

Technika: Ilyen esetekben alkalmazhatunk adaptív vödörméreteket, ahol a vödrök méretét az adatok sűrűsége alapján határozzuk meg. Azaz, ahol sűrűbben vannak az adatok, ott kisebb méretű vödröket használunk, míg a ritkábban előforduló adatoknál nagyobb méretű vödröket alkalmazunk.

#### Egyenlőtlen eloszlás

Egyenlőtlen eloszlásnál az adatok jelentős része néhány érték körül csoportosul, míg a többi érték alig van jelen. Ez az eloszlás a legnagyobb kihívás a Bucket Sort számára, mivel néhány vödör nagyon sok elemet tartalmazhat, ami lelassítja a rendezést.

Technika: Ebben az esetben hatékony megoldás lehet a vödrök dinamikus bővítése. Azaz, amikor egy vödör megtelik, új vödröt hozunk létre, amely a túltöltött vödör elemeit is tartalmazza. Ezzel biztosíthatjuk, hogy egyetlen vödör se tartalmazzon túl sok elemet, és a rendezési lépések hatékonyak maradjanak.

#### Kvantilis-alapú vödrözés

Az egyik leghatékonyabb módszer különböző eloszlások kezelésére a kvantilis-alapú vödrözés. Ennek a technikának a lényege, hogy az adatokat kvantilisek alapján osztjuk fel vödrökre, így minden vödörbe hasonló számú elem kerül. Ez a megközelítés különösen hasznos egyenlőtlen vagy torzított eloszlás esetén.

Példa: Ha a teljes adatállomány 1000 elem, és 10 vödröt akarunk létrehozni, akkor minden vödörbe 100 elem kerül. Az egyes vödrök határait úgy határozzuk meg, hogy minden vödör pontosan 100 elemet tartalmazzon, figyelembe véve az adatok eloszlását.

#### Példa kvantilis-alapú vödrözésre C++ nyelven

```cpp
#include <iostream>

#include <vector>
#include <algorithm>

void bucketSortQuantile(std::vector<float>& arr) {
    int n = arr.size();
    if (n <= 1) return;

    // Rendezés a kvantilisek meghatározásához
    std::sort(arr.begin(), arr.end());

    // 1. Vödrök létrehozása kvantilisek alapján
    std::vector<std::vector<float>> buckets(n / 10); // 10 vödröt hozunk létre

    // 2. Adatok elosztása a vödrökbe kvantilisek alapján
    for (int i = 0; i < n; i++) {
        int bucketIndex = i / (n / 10);
        buckets[bucketIndex].push_back(arr[i]);
    }

    // 3. Vödrök rendezése
    for (auto& bucket : buckets) {
        std::sort(bucket.begin(), bucket.end());
    }

    // 4. Rendezett vödrök összefűzése
    int index = 0;
    for (auto& bucket : buckets) {
        for (float num : bucket) {
            arr[index++] = num;
        }
    }
}

int main() {
    std::vector<float> arr = {0.92, 0.12, 0.85, 0.45, 0.27, 0.78, 0.95, 0.32, 0.73, 0.53};
    
    std::cout << "Original array: ";
    for (float num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    bucketSortQuantile(arr);
    
    std::cout << "Sorted array: ";
    for (float num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

#### Adateloszlások elemzése és adaptív vödörméretezés

Az adatok eloszlásának pontos ismerete lehetőséget ad arra, hogy az algoritmust optimálisan adaptáljuk. Az adateloszlások elemzése során figyelembe vehetjük a következőket:
- **Átlag és szórás:** Ezek az értékek segíthetnek meghatározni az adatok koncentrációját.
- **Percentilisek és kvantilisek:** Ezek az értékek megmutatják, hogyan oszlanak el az adatok az intervallumok között.
- **Histogramos elemzés:** Egy hisztogram segíthet az adatok eloszlásának vizuális megjelenítésében és a megfelelő vödörstratégia kiválasztásában.

Az adaptív vödörméretezés során dinamikusan változtathatjuk a vödrök számát és méretét az adatok aktuális eloszlásának megfelelően. Ez magában foglalhatja:
- **Dinamikus vödörbővítés:** Amikor egy vödör megtelik, további vödröket hozunk létre az elemek szétosztásához.
- **Vödrök újrakalibrálása:** Az adatok eloszlásának figyelembevételével a vödrök határait újra meghatározhatjuk a rendezés során.

#### Konklúzió

A Bucket Sort algoritmus hatékonysága nagyban függ az adatok eloszlásától. Különböző eloszlások különböző megközelítéseket igényelnek a vödrök létrehozásában és kezelésében. Az egyenletes eloszlás esetén az algoritmus optimálisan működik, míg normális, torzított és egyenlőtlen eloszlások esetén speciális technikák alkalmazására van szükség. Az adaptív vödörméretez

és és a kvantilis-alapú vödrözés olyan hatékony módszerek, amelyek lehetővé teszik az algoritmus teljesítményének optimalizálását különböző adateloszlások mellett.

### 2.10.3. Teljesítmény és komplexitás elemzése

A Bucket Sort egy hatékony rendezési algoritmus, amely bizonyos feltételek mellett kimagasló teljesítményt nyújthat. A teljesítmény és komplexitás elemzése során figyelembe vesszük az algoritmus időbeli és térbeli komplexitását, valamint a legjobb, átlagos és legrosszabb esetekben nyújtott teljesítményét. Emellett megvizsgáljuk azokat a tényezőket is, amelyek befolyásolják az algoritmus hatékonyságát.

#### Időbeli komplexitás

A Bucket Sort algoritmus időbeli komplexitása három fő lépésre bontható: a vödrök létrehozása és az elemek elosztása, a vödrök rendezése, valamint a rendezett vödrök összefűzése.

1. **Vödrök létrehozása és elemek elosztása:** Az adatok vödrökbe való elosztása O(n) időt igényel, ahol n az adatok száma. Minden egyes elem esetén egy egyszerű hash függvényt alkalmazunk, amely meghatározza, hogy az elem melyik vödörbe kerüljön.

2. **Vödrök rendezése:** Minden vödröt külön rendezünk. Ha az adatok egyenletesen oszlanak el, és minden vödörben kb. n/k elem található (ahol k a vödrök száma), akkor az egyes vödrök rendezése O((n/k) log(n/k)) időt igényel. Az összes vödör rendezési ideje így O(n log(n/k)) lesz.

3. **Rendezett vödrök összefűzése:** A rendezett vödrök összefűzése szintén O(n) időt igényel, mivel minden egyes vödör elemeit egyesítjük egy végleges rendezett listába.

Az algoritmus teljes időbeli komplexitása tehát:
$$
O(n) + O(n \log(n/k)) + O(n) = O(n \log(n/k))
$$

Legjobb esetben, ha a vödrök száma megegyezik az elemek számával (k = n), az időbeli komplexitás O(n) lesz.

#### Térbeli komplexitás

A Bucket Sort térbeli komplexitása két fő tényezőből áll: a vödrök tárolásához szükséges hely és az eredeti adatstruktúra tárolása.

1. **Vödrök tárolása:** Az algoritmusnak k vödörre van szüksége, amelyek mindegyike legfeljebb n elemet tartalmazhat. Az egyes vödrök különböző mennyiségű elemet tartalmazhatnak az adatok eloszlásától függően. Az összes vödör tárolásához szükséges hely O(n + k).

2. **Eredeti adatstruktúra:** Az eredeti adatokat is tárolni kell, ami O(n) helyet igényel.

A teljes térbeli komplexitás tehát:
$$
O(n + k)
$$

#### Legjobb eset

A legjobb eset akkor valósul meg, ha az adatok egyenletesen oszlanak el a vödrök között, és minden vödörbe közel azonos számú elem kerül. Ebben az esetben a vödrök rendezése gyorsan megtörténik, és az algoritmus időbeli komplexitása O(n).

Példa: Ha 100 elemet rendezünk 10 vödörbe, és minden vödörbe pontosan 10 elem kerül, akkor minden vödör rendezése O(10 log 10) időt igényel, azaz a teljes időbeli komplexitás O(100) lesz.

#### Átlagos eset

Az átlagos esetben az adatok valamilyen ismert eloszlás szerint oszlanak el, például egyenletes vagy normális eloszlásban. Ebben az esetben a vödrök közel egyenletesen töltődnek fel, de lehet némi variancia a vödrök méretében. Az időbeli komplexitás ebben az esetben O(n log(n/k)) lesz, ami k = O(n) esetén O(n) időt igényel.

Példa: Ha az adatok normális eloszlásúak, és 100 elemet rendezünk 10 vödörbe, akkor néhány vödör kicsit több, mások kicsit kevesebb elemet tartalmazhatnak, de a rendezési lépések összességében még mindig hatékonyak lesznek.

#### Legrosszabb eset

A legrosszabb eset akkor következik be, ha az adatok nem egyenletesen oszlanak el, és néhány vödör sokkal több elemet tartalmaz, mint a többi. Ez a helyzet jelentős időbeli és térbeli többletköltséget okoz.

Példa: Ha 100 elemből 90 egyetlen vödörbe kerül, és a maradék 10 elem oszlik el a többi 9 vödör között, akkor a legnagyobb vödör rendezése O(90 log 90) időt igényel, ami jelentősen megnöveli az algoritmus teljes futási idejét.

A legrosszabb eset időbeli komplexitása így $O(n^2)$ is lehet, ha minden elem egyetlen vödörbe kerül.

#### Teljesítményoptimalizálás

A Bucket Sort teljesítményének optimalizálása érdekében különböző technikákat alkalmazhatunk:

1. **Vödrök számának és méretének megfelelő meghatározása:** Az optimális teljesítmény elérése érdekében a vödrök számát és méretét az adatok eloszlásához kell igazítani. Ha az adatok eloszlása ismert, akkor adaptív vödörméreteket használhatunk.

2. **Vödrök dinamikus bővítése:** Ha egy vödör túl sok elemet tartalmaz, dinamikusan további vödröket hozhatunk létre az elemek szétosztásához. Ezzel elkerülhető a vödrök túlterhelése és a rendezési lépések lelassulása.

3. **Kvantilis-alapú vödrözés:** Az adatok eloszlásának figyelembevételével kvantilisek alapján oszthatjuk fel a vödröket, így minden vödörbe hasonló számú elem kerül. Ez különösen hasznos egyenlőtlen vagy torzított eloszlás esetén.

4. **Paralelizáció:** A Bucket Sort algoritmus jól párhuzamosítható, mivel a vödrök külön-külön rendezhetők. A párhuzamos feldolgozás jelentősen javíthatja az algoritmus teljesítményét nagy adathalmazok esetén.

#### Példa kód optimalizált Bucket Sort implementációra C++ nyelven

Az alábbi példa kód egy optimalizált Bucket Sort implementációt mutat be, amely figyelembe veszi az adatok eloszlását és kvantilis-alapú vödrözést alkalmaz:

```cpp
#include <iostream>

#include <vector>
#include <algorithm>

#include <thread>

// Rendezési függvény egy-egy vödörre
void sortBucket(std::vector<float>& bucket) {
    std::sort(bucket.begin(), bucket.end());
}

void bucketSortOptimized(std::vector<float>& arr) {
    int n = arr.size();
    if (n <= 1) return;

    // Kvantilisek alapján vödrök létrehozása
    int numBuckets = 10; // Például 10 vödör
    std::vector<std::vector<float>> buckets(numBuckets);

    // Adatok elosztása a vödrökbe kvantilisek alapján
    for (int i = 0; i < n; i++) {
        int bucketIndex = std::min(numBuckets - 1, static_cast<int>(numBuckets * arr[i]));
        buckets[bucketIndex].push_back(arr[i]);
    }

    // Vödrök párhuzamos rendezése
    std::vector<std::thread> threads;
    for (int i = 0; i < numBuckets; i++) {
        threads.emplace_back(sortBucket, std::ref(buckets[i]));
    }

    for (auto& th : threads) {
        th.join();
    }

    // Rendezett vödrök összefűzése
    int index = 0;
    for (auto& bucket : buckets) {
        for (float num : bucket) {
            arr[index++] = num;
        }
    }
}

int main() {
    std::vector<float> arr = {0.92, 0.12, 0.85, 0.45, 0.27, 0.78, 0.95, 0.32, 0.73, 0.53};
    
    std::cout << "Original array: ";
    for (float num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    bucketSortOptimized(arr);
    
    std::cout << "Sorted array: ";
    for (float num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

#### Összegzés

A Bucket Sort algoritmus teljesítményét és komplexitását számos tényező befolyásolja, beleértve az adatok eloszlását, a vödrök számát és méretét, valamint az alkalmazott rendezési technikákat. Az algoritmus ideális esetben lineáris időben fut, azonban nem megfelelő adateloszlás vagy vödrözési stratégia esetén a teljesítmény jelentősen romolhat. Az optimalizálási technikák, mint az adaptív vödörméretezés, kvantilis-alapú vödrözés és paralelizáció, segíthetnek az algoritmus hatékonyságának javításában különböző adateloszlások mellett.

### 2.10.4. Gyakorlati alkalmazások és példák

A Bucket Sort algoritmus számos gyakorlati alkalmazási területtel rendelkezik, különösen olyan esetekben, amikor az adatok egyenletes eloszlásúak, vagy ahol a rendezés hatékonysága kiemelten fontos. Ebben az alfejezetben részletesen bemutatjuk a Bucket Sort alkalmazásait különböző területeken, és gyakorlati példákon keresztül szemléltetjük a használatát.

#### Alkalmazások

##### 1. Lebegőpontos számok rendezése

A Bucket Sort különösen hatékony lebegőpontos számok rendezésénél, amelyek egy adott intervallumban, például [0, 1), egyenletesen oszlanak el. A lebegőpontos számok rendezése gyakori feladat tudományos számítások és szimulációk során.

**Példa:** Egy fizikai szimuláció során keletkező adatok, amelyek pontos időbélyegeket tartalmaznak, és szükséges ezek rendezése időrend szerint a további feldolgozáshoz.

##### 2. Hash alapú adatbázis rendezés

Adatbázisokban gyakran használnak hash alapú technikákat az adatok gyors kereséséhez és rendezéséhez. A Bucket Sort jól illeszkedik az ilyen technikákhoz, mivel az adatokat hash függvények segítségével osztja el vödrökbe.

**Példa:** Egy nagy adatbázisban tárolt tranzakciós adatok rendezése az időbélyegek vagy azonosítók alapján, hogy gyorsan lehessen hozzáférni az adatokhoz időrendben vagy csoportosítva.

##### 3. Digitális képfeldolgozás

A képfeldolgozás során gyakran szükséges a pixelek intenzitásának rendezése különböző műveletekhez, például histogram kiegyenlítéshez vagy szegmensek azonosításához. A Bucket Sort hatékonyan alkalmazható ilyen feladatokhoz, különösen ha az intenzitás értékek egyenletesen oszlanak el.

**Példa:** Egy digitális kép histogram kiegyenlítése során a pixelek intenzitás értékeinek rendezése, hogy az egyes intenzitás tartományok kiegyenlítését elvégezhessük.

##### 4. Párhuzamos számítások és adatfeldolgozás

A Bucket Sort algoritmus jól párhuzamosítható, mivel a vödrök külön-külön rendezhetők. Ez különösen előnyös nagy méretű adathalmazok esetén, ahol a párhuzamos feldolgozás jelentősen csökkentheti a futási időt.

**Példa:** Nagy méretű adatfájlok rendezése egy elosztott rendszerben, ahol minden csomópont külön rendezi a saját vödrét, majd az eredmények összefűzése után kapjuk meg a végleges rendezett adatállományt.

##### 5. Grafikus megjelenítés és adatelemzés

A Bucket Sort alkalmazható grafikus megjelenítési rendszerekben is, ahol a különböző objektumokat vagy elemeket kell rendezni az adott kritériumok szerint, hogy hatékonyan lehessen őket megjeleníteni vagy elemezni.

**Példa:** Egy játék motorban az objektumok Z tengely mentén történő rendezése, hogy a renderelési sorrend helyes legyen és a mélységi rétegek megfelelően jelenjenek meg.

#### Példák és implementációk

##### Példa 1: Lebegőpontos számok rendezése

Tegyük fel, hogy egy szimuláció eredményeként kapunk egy lebegőpontos számokból álló listát, amelyeket növekvő sorrendbe kell rendezni. A következő C++ kód egy egyszerű implementációt mutat be a Bucket Sort használatával.

```cpp
#include <iostream>

#include <vector>
#include <algorithm>

// Rendezési függvény egy-egy vödörre
void sortBucket(std::vector<float>& bucket) {
    std::sort(bucket.begin(), bucket.end());
}

void bucketSort(std::vector<float>& arr) {
    int n = arr.size();
    if (n <= 1) return;

    // Vödrök létrehozása
    std::vector<std::vector<float>> buckets(n);

    // Adatok elosztása a vödrökbe
    for (int i = 0; i < n; i++) {
        int bucketIndex = n * arr[i];
        buckets[bucketIndex].push_back(arr[i]);
    }

    // Vödrök rendezése
    for (auto& bucket : buckets) {
        sortBucket(bucket);
    }

    // Rendezett vödrök összefűzése
    int index = 0;
    for (auto& bucket : buckets) {
        for (float num : bucket) {
            arr[index++] = num;
        }
    }
}

int main() {
    std::vector<float> arr = {0.78, 0.12, 0.23, 0.56, 0.34, 0.89, 0.45, 0.67, 0.99, 0.01};
    
    std::cout << "Original array: ";
    for (float num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    bucketSort(arr);
    
    std::cout << "Sorted array: ";
    for (float num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

##### Példa 2: Hash alapú adatbázis rendezés

Egy adatbázisban tárolt tranzakciók rendezése időbélyegek szerint. A következő C++ kód egy egyszerű implementációt mutat be, amely hash függvényt használ a tranzakciók vödrökbe történő elosztására, majd rendezésére.

```cpp
#include <iostream>

#include <vector>
#include <algorithm>

// Tranzakció struktúra
struct Transaction {
    int id;
    float timestamp;
};

// Rendezési függvény egy-egy vödörre
void sortBucket(std::vector<Transaction>& bucket) {
    std::sort(bucket.begin(), bucket.end(), [](const Transaction& a, const Transaction& b) {
        return a.timestamp < b.timestamp;
    });
}

void bucketSort(std::vector<Transaction>& arr) {
    int n = arr.size();
    if (n <= 1) return;

    // Vödrök létrehozása
    std::vector<std::vector<Transaction>> buckets(n);

    // Adatok elosztása a vödrökbe
    for (int i = 0; i < n; i++) {
        int bucketIndex = n * arr[i].timestamp;
        buckets[bucketIndex].push_back(arr[i]);
    }

    // Vödrök rendezése
    for (auto& bucket : buckets) {
        sortBucket(bucket);
    }

    // Rendezett vödrök összefűzése
    int index = 0;
    for (auto& bucket : buckets) {
        for (const Transaction& txn : bucket) {
            arr[index++] = txn;
        }
    }
}

int main() {
    std::vector<Transaction> arr = {
        {1, 0.78}, {2, 0.12}, {3, 0.23}, {4, 0.56}, {5, 0.34},
        {6, 0.89}, {7, 0.45}, {8, 0.67}, {9, 0.99}, {10, 0.01}
    };
    
    std::cout << "Original array: ";
    for (const auto& txn : arr) {
        std::cout << "{" << txn.id << ", " << txn.timestamp << "} ";
    }
    std::cout << std::endl;
    
    bucketSort(arr);
    
    std::cout << "Sorted array: ";
    for (const auto& txn : arr) {
        std::cout << "{" << txn.id << ", " << txn.timestamp << "} ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

##### Példa 3: Digitális képfeldolgozás

Egy kép pixeleinek intenzitásértékeit rendezni kell, hogy egyenletes histogram kiegyenlítést lehessen végezni. A következő C++ kód egy egyszerű implementációt mutat be, amely a Bucket Sort algoritmust használja a pixel intenzitásértékek rendezésére.

```cpp
#include <iostream>

#include <vector>
#include <algorithm>

// Rendezési függvény egy-egy vödörre
void sortBucket(std::vector<int>& bucket) {
    std::sort(bucket.begin(), bucket.end());
}

void bucketSort(std::vector<int>& arr, int maxValue) {
    int n = arr.size();
    if (n <= 1) return;

    // Vödrök létrehozása
    std::vector<std::vector<int>> buckets(maxValue + 1);

    // Adatok elosztása a vödrökbe
    for (int i = 0; i < n; i++) {
        buckets[arr[i]].push_back(arr[i]);
    }

    // Vödrök rendezése
    for (auto& bucket : buckets) {
        sortBucket(bucket);
    }

    // Rendezett vödrök összefűzése
    int index = 0;
    for (auto& bucket : buckets) {
        for (int num : bucket) {
            arr[index++] = num;
        }
    }
}

int main() {
    std::vector<int> arr = {78, 12, 23, 56, 34, 89, 45, 67, 99, 1};
    int maxValue = 100;
    
    std::cout << "Original array: ";
    for (int num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    bucketSort(arr, maxValue);
    
    std::cout << "Sorted array: ";
    for (int num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

##### Példa 4: Párhuzamos számítások és adatfeldolgozás

Nagy adatállományok párhuzamos rendezése elosztott rendszerekben. A következő C++ kód egy egyszerű implementációt mutat be, amely párhuzamosan rendezi az adatokat.

```cpp
#include <iostream>

#include <vector>
#include <algorithm>

#include <thread>

// Rendezési függvény egy-egy vödörre
void sortBucket(std::vector<float>& bucket) {
    std::sort(bucket.begin(), bucket.end());
}

void bucketSortParallel(std::vector<float>& arr) {
    int n = arr.size();
    if (n <= 1) return;

    // Vödrök létrehozása
    int numBuckets = 10; // Például 10 vödör
    std::vector<std::vector<float>> buckets(numBuckets);

    // Adatok elosztása a vödrökbe
    for (int i = 0; i < n; i++) {
        int bucketIndex = std::min(numBuckets - 1, static_cast<int>(numBuckets * arr[i]));
        buckets[bucketIndex].push_back(arr[i]);
    }

    // Vödrök párhuzamos rendezése
    std::vector<std::thread> threads;
    for (int i = 0; i < numBuckets; i++) {
        threads.emplace_back(sortBucket, std::ref(buckets[i]));
    }

    for (auto& th : threads) {
        th.join();
    }

    // Rendezett vödrök összefűzése
    int index = 0;
    for (auto& bucket : buckets) {
        for (float num : bucket) {
            arr[index++] = num;
        }
    }
}

int main() {
    std::vector<float> arr = {0.92, 0.12, 0.85, 0.45, 0.27, 0.78, 0.95, 0.32, 0.73, 0.53};
    
    std::cout << "Original array: ";
    for (float num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    bucketSortParallel(arr);
    
    std::cout << "Sorted array: ";
    for (float num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

#### Összegzés

A Bucket Sort algoritmus számos gyakorlati alkalmazási területtel rendelkezik, beleértve a lebegőpontos számok rendezését, adatbázisok hash alapú rendezését, digitális képfeldolgozást, párhuzamos számításokat és adatfeldolgozást, valamint grafikus megjelenítést és adatelemzést. Az algoritmus hatékonysága különösen kiemelkedő, ha az adatok egyenletesen oszlanak el, és megfelelően optimalizált vödörstratégiát alkalmazunk. Az implementációk és példák bemutatják, hogyan használható a Bucket Sort különböző gyakorlati feladatok megoldására, és hogyan érhetjük el a legjobb teljesítményt különböző körülmények között.