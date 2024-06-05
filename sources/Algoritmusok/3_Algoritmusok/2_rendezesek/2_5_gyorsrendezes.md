\newpage

## 2.5. Merge Sort

A Merge Sort egy hatékony, oszd meg és uralkodj (divide and conquer) elven alapuló rendezési algoritmus, amelyet John von Neumann fejlesztett ki 1945-ben. Az algoritmus a probléma rekurzív felosztásával működik, ahol az elemeket kisebb részekre bontjuk, majd azokat rendezzük és végül egyesítjük (merge). A Merge Sort stabil rendezési algoritmus, ami azt jelenti, hogy megőrzi az azonos kulcsú elemek eredeti sorrendjét. Az algoritmus teljesítménye a legrosszabb esetben is $O(n \log n)$, így különösen előnyös nagy adatállományok rendezésekor. Ebben a fejezetben részletesen megvizsgáljuk a Merge Sort alapelveit és rekurzív megvalósítását, a bottom-up megközelítést, valamint az algoritmus teljesítményét és komplexitását. Emellett gyakorlati alkalmazásokkal és példákkal is illusztráljuk a Merge Sort működését.

### 2.5.1. Alapelvek és rekurzív megvalósítás

A Merge Sort egy klasszikus példa az oszd meg és uralkodj (divide and conquer) elvű algoritmusokra, amelyek különösen hatékonyak nagy adatstruktúrák rendezésére. A Merge Sort két fő fázisból áll: a problémát kisebb részekre bontjuk (divide), majd ezeket a részeket rendezzük és egyesítjük (conquer és merge). Ez a megközelítés garantálja, hogy az algoritmus teljesítménye még a legrosszabb esetben is $O(n \log n)$, ahol $n$ a rendezendő elemek száma.

#### Alapelvek

A Merge Sort alapelvei az alábbiakban foglalhatók össze:

1. **Bontás (Divide)**: Az eredeti tömböt két egyenlő részre osztjuk. Ez a rekurzív lépés addig folytatódik, amíg az összes alrész legfeljebb egy elemből áll. Egyetlen elemről feltételezzük, hogy rendezett.
2. **Egyesítés (Merge)**: Az így kapott rendezett részeket páronként összehasonlítjuk és egyesítjük úgy, hogy az eredmény ismét rendezett legyen. Ez a fázis garantálja, hogy a rendezett részek egyesítésével végül az egész tömb rendezett lesz.

#### Rekurzív megvalósítás

A rekurzív megvalósítás a következő lépésekből áll:

1. **Alap eset**: Ha a tömb mérete 0 vagy 1, akkor a tömb már rendezett, így a rendezés befejeződik.
2. **Rekurzív bontás**: A tömböt két részre osztjuk, majd rekurzívan alkalmazzuk a Merge Sort algoritmust mindkét részre.
3. **Egyesítés**: A rendezett részeket egyesítjük egy segéd tömb segítségével.

Az alábbiakban bemutatjuk a rekurzív Merge Sort algoritmus C++ nyelvű implementációját:

```cpp
#include <iostream>

#include <vector>

// Merge két rendezett alrész összevonása
void merge(std::vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Két segédtömb létrehozása
    std::vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; ++i)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; ++j)
        R[j] = arr[mid + 1 + j];

    // Indexek a két alrészhez és az egyesített tömbhöz
    int i = 0, j = 0, k = left;

    // Az alrészek összehasonlítása és összevonása
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // A fennmaradó elemek másolása, ha van
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// Rekurzív Merge Sort
void mergeSort(std::vector<int>& arr, int left, int right) {
    if (left < right) {
        // Középső index kiszámítása
        int mid = left + (right - left) / 2;

        // Rekurzív rendezés az alrészekre
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        // Két rendezett alrész egyesítése
        merge(arr, left, mid, right);
    }
}

int main() {
    std::vector<int> arr = {12, 11, 13, 5, 6, 7};
    int arr_size = arr.size();

    std::cout << "Given array is \n";
    for (int i = 0; i < arr_size; i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;

    mergeSort(arr, 0, arr_size - 1);

    std::cout << "\nSorted array is \n";
    for (int i = 0; i < arr_size; i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
    return 0;
}
```

#### Részletezés

1. **Segédtömbök használata**: A merge fázis során két segédtömböt használunk az elemek másolására, így biztosítva az eredeti tömb rendezett egyesítését. Ez a megközelítés stabilitást biztosít az algoritmusnak, mivel az azonos értékű elemek eredeti sorrendje megmarad.

2. **Rekurzív felosztás**: Az algoritmus rekurzív természete azt jelenti, hogy a tömböt folyamatosan félbevágjuk, amíg el nem érjük az alap esetet, ahol a tömb mérete 1. Ezután a rekurzív hívások visszafordulnak, és a rendezett részek egyesítése történik meg.

3. **Komplexitás**: A Merge Sort algoritmus időbeli komplexitása minden esetben $O(n \log n)$. Az osztás fázis $\log n$ lépést igényel, mivel minden lépésben kettéosztjuk a tömböt. Az egyesítés fázis pedig $O(n)$ lépést igényel, mivel minden elemet egyszer kell másolni. A térbeli komplexitás $O(n)$, mivel az egyesítés során szükség van egy segédtömbre, amely az összes elemet tartalmazza.

4. **Optimális felhasználás**: A Merge Sort algoritmus ideális olyan helyzetekben, ahol stabil rendezésre van szükség, vagy amikor nagy adatstruktúrákat kell rendezni, amelyek nem férnek el a memóriában. Az algoritmus hatékonyan használható külső rendezési technikákban is, ahol az adatok lemezen tárolódnak.

5. **Stabilitás és párhuzamosíthatóság**: A Merge Sort stabil, ami azt jelenti, hogy megőrzi az azonos kulcsú elemek sorrendjét. Továbbá jól párhuzamosítható, mivel az alrészek rendezése független egymástól, így több processzoron párhuzamosan futtatható.

Összefoglalva, a Merge Sort egy rendkívül hatékony és stabil rendezési algoritmus, amelynek rekurzív megvalósítása jól illusztrálja az oszd meg és uralkodj elv működését. Az algoritmus ideális választás lehet nagy adatstruktúrák rendezésére, különösen akkor, ha stabil rendezésre van szükség.

### 2.5.2. Bottom-up merge sort

A Bottom-up merge sort egy alternatív megközelítése a Merge Sort algoritmusnak, amely a rekurzív megoldással szemben iteratív módon működik. Ez a módszer különösen előnyös lehet olyan esetekben, amikor a rekurzió által okozott memóriaterhelést szeretnénk elkerülni, vagy egyszerűen csak egy másik nézőpontból kívánjuk megérteni az algoritmus működését.

#### Alapelvek

A Bottom-up merge sort az oszd meg és uralkodj (divide and conquer) elvén alapul, de a tömböt nem rekurzív módon osztja fel, hanem iteratív lépésekben, egyre nagyobb részleteket rendezve. Az algoritmus az alábbi alapelvekre épül:

1. **Kezdeti kis részek**: Az algoritmus azzal kezdődik, hogy minden egyes elemet külön-külön tekintjük, mint egy-egy egyetlen elemű rendezett tömböt.
2. **Iteratív összevonás**: Ezután az egyes elemeket párokba rendezzük és összevonjuk őket, hogy két elemű rendezett tömböket kapjunk. Ezt a folyamatot folytatjuk, egyre nagyobb és nagyobb rendezett részeket egyesítve, amíg végül az egész tömb rendezetté válik.
3. **Lépésenkénti növelés**: Minden iterációban a rendezett részek méretét kettővel szorozzuk, így a kezdeti egy elemű részekből kettő, majd négy, nyolc és így tovább elemű rendezett részek lesznek.

#### Iteratív megvalósítás

A Bottom-up merge sort algoritmus C++ nyelvű implementációja az alábbiakban látható:

```cpp
#include <iostream>

#include <vector>

// Merge két rendezett alrész összevonása
void merge(std::vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    std::vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; ++i)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; ++j)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// Iteratív Bottom-up Merge Sort
void mergeSort(std::vector<int>& arr) {
    int n = arr.size();

    for (int curr_size = 1; curr_size <= n - 1; curr_size = 2 * curr_size) {
        for (int left_start = 0; left_start < n - 1; left_start += 2 * curr_size) {
            int mid = std::min(left_start + curr_size - 1, n - 1);
            int right_end = std::min(left_start + 2 * curr_size - 1, n - 1);

            merge(arr, left_start, mid, right_end);
        }
    }
}

int main() {
    std::vector<int> arr = {12, 11, 13, 5, 6, 7};
    int arr_size = arr.size();

    std::cout << "Given array is \n";
    for (int i = 0; i < arr_size; i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;

    mergeSort(arr);

    std::cout << "\nSorted array is \n";
    for (int i = 0; i < arr_size; i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
    return 0;
}
```

#### Részletezés

1. **Iteratív megközelítés**: A Bottom-up merge sort algoritmus az iteratív módszer miatt elkerüli a rekurzió által okozott memóriaterhelést. Ahelyett, hogy a tömböt rekurzívan osztanánk fel kisebb részekre, az algoritmus az elemeket párokba rendezi és iteratív módon egyesíti őket.

2. **Rendezett részek fokozatos növelése**: Minden iterációban a rendezett részek mérete megduplázódik. Kezdetben minden egyes elem saját rendezett tömböt alkot. Az első iterációban ezeket az egy elemből álló rendezett tömböket páronként egyesítjük, hogy két elemből álló rendezett tömböket kapjunk. A következő iterációban ezek a két elemű rendezett tömbök négy elemű rendezett tömbökké egyesülnek, és így tovább, amíg az egész tömb rendezetté nem válik.

3. **Merge művelet**: A merge művelet két rendezett alrész egyesítését végzi el. Az algoritmus minden iterációban két rendezett részt egyesít úgy, hogy egy segéd tömb segítségével az elemeket összehasonlítja és sorrendbe állítja. Ez a művelet garantálja, hogy a rendezett részek egyesítése után az eredmény ismét rendezett lesz.

4. **Komplexitás**: A Bottom-up merge sort algoritmus időbeli komplexitása $O(n \log n)$, hasonlóan a rekurzív Merge Sorthoz. Az algoritmus minden iterációja $O(n)$ időt igényel, mivel minden elemet egyszer kell összehasonlítani és másolni. Az iterációk száma $\log n$, mivel minden iterációban a rendezett részek mérete megduplázódik. A térbeli komplexitás $O(n)$, mivel a merge művelet során segéd tömbre van szükség, amely az összes elemet tartalmazza.

5. **Stabilitás**: A Bottom-up merge sort stabil algoritmus, mivel megőrzi az azonos kulcsú elemek eredeti sorrendjét. Ez különösen fontos olyan alkalmazásokban, ahol az elemek relatív sorrendje is számít, például adatbázisok rendezésénél.

6. **Praktikus alkalmazások**: A Bottom-up merge sort algoritmus jól alkalmazható olyan helyzetekben, ahol nagy mennyiségű adatot kell rendezni és a rekurzió használata nem kívánatos vagy nem lehetséges. Az algoritmus hatékonyan működik külső rendezési technikákban is, ahol az adatok lemezen tárolódnak, és memóriakorlátok miatt iteratív megközelítés szükséges.

#### Összegzés

A Bottom-up merge sort egy hatékony és stabil rendezési algoritmus, amely iteratív megközelítéssel éri el a rekurzív Merge Sorthoz hasonló teljesítményt. Az algoritmus előnyei közé tartozik a memóriaterhelés csökkentése és a stabilitás megőrzése, amely különösen fontos nagy adatstruktúrák rendezésekor. A Bottom-up merge sort ideális választás lehet olyan alkalmazásokban, ahol a rekurzió használata nem lehetséges vagy nem kívánatos, és stabil rendezésre van szükség.

### 2.5.3. Teljesítmény és komplexitás elemzése

A Merge Sort, legyen az rekurzív vagy bottom-up megközelítésű, az egyik leghatékonyabb általános rendezési algoritmus. Ebben az alfejezetben részletesen elemezzük az algoritmus teljesítményét és komplexitását különböző szempontokból, beleértve az időbeli és térbeli komplexitást, a stabilitást, valamint a gyakorlati alkalmazhatóságot.

#### Időbeli komplexitás

A Merge Sort időbeli komplexitása minden esetben $O(n \log n)$, ami az egyik legjobb időbeli komplexitás a rendezési algoritmusok között. Az időbeli komplexitás analízise két fő szakaszra oszlik: a felosztási (divide) és az egyesítési (merge) fázisra.

1. **Felosztás (Divide)**: Az algoritmus a tömböt folyamatosan két részre osztja, amíg minden rész egyetlen elemből áll. Mivel minden felosztás lépésben a tömb mérete felére csökken, a felosztások száma $\log n$ lesz, ahol $n$ a tömb mérete.

2. **Egyesítés (Merge)**: Az egyesítés során minden egyes elem egyszer kerül összehasonlításra és másolásra. Ez a művelet lineáris időben történik, azaz $O(n)$.

Ezek alapján a teljes időbeli komplexitás így alakul:
$$
T(n) = O(n \log n)
$$
Ez azt jelenti, hogy a Merge Sort teljesítménye skálázható és nagy adatmennyiség esetén is hatékonyan működik.

#### Térbeli komplexitás

A Merge Sort térbeli komplexitása $O(n)$, mivel az algoritmus során szükség van egy segédtömbre, amely a rendezett részek egyesítéséhez szükséges elemeket tartalmazza. A segédtömb mérete megegyezik az eredeti tömb méretével, így az algoritmus extra $O(n)$ memóriát igényel.

1. **Rekurzív megközelítés**: A rekurzív Merge Sort esetében az extra memóriaigény tartalmazza a segédtömböt és a rekurzív hívások miatt a verem (stack) által használt memóriát is. A veremmélység maximálisan $\log n$ szintű, így a teljes térbeli komplexitás:
   $$
   O(n) + O(\log n) \approx O(n)
   $$
   Mivel az $O(n)$ dominál, az extra $O(\log n)$ nem befolyásolja jelentősen a térbeli komplexitást.

2. **Bottom-up megközelítés**: Az iteratív Bottom-up merge sort esetében a segédtömbön kívül nincs szükség további memóriára, mivel nem használ rekurziót. Így a térbeli komplexitás itt is:
   $$
   O(n)
   $$

#### Stabilitás

A Merge Sort egy stabil rendezési algoritmus, ami azt jelenti, hogy az azonos értékű elemek sorrendje nem változik meg a rendezés során. Ez különösen fontos olyan alkalmazásokban, ahol az elemek sorrendje is információt hordoz, például adatbázisokban, ahol a rekordok rendezése közben meg kell őrizni az eredeti sorrendet az azonos kulcsú rekordok esetében.

#### Gyakorlati alkalmazhatóság

A Merge Sort széles körben alkalmazható különböző területeken, beleértve a következőket:

1. **Nagy adatmennyiségek rendezése**: Az $O(n \log n)$ időbeli komplexitás miatt a Merge Sort különösen jól teljesít nagy adatmennyiségek esetén, ahol a kevésbé hatékony algoritmusok, mint a Bubble Sort vagy a Selection Sort, nem lennének megfelelőek.

2. **Külső rendezés**: A Merge Sort jól alkalmazható külső rendezési feladatokban, ahol az adatok nem férnek el a memóriában és lemezen tárolódnak. Ilyen esetekben az algoritmus segítségével az adatok kisebb rendezett részekre bonthatók, amelyeket aztán egyesítéssel rendezünk.

3. **Stabil rendezés igénye**: Olyan helyzetekben, ahol az adatok stabil rendezése elengedhetetlen, a Merge Sort megbízható választás. Ez különösen fontos adatbázisok rendezésekor vagy olyan algoritmusok esetében, amelyek több lépésben rendezik az adatokat.

#### Elméleti megfontolások

1. **Worst-case és Average-case időbeli komplexitás**: A Merge Sort időbeli komplexitása mind a legrosszabb, mind az átlagos esetben $O(n \log n)$. Ez azt jelenti, hogy az algoritmus mindig garantáltan hatékonyan működik, függetlenül az adatok kezdeti elrendezésétől.

2. **Térbeli hatékonyság javítása**: Bár a Merge Sort alapvetően $O(n)$ térbeli komplexitású, léteznek olyan optimalizációk, amelyek csökkenthetik az extra memóriahasználatot. Például, ha az algoritmus in-place módon működik, akkor a segédtömb használata nélkül is végrehajtható a rendezés, bár ez a stabilitás feladását jelentheti.

3. **Párhuzamosíthatóság**: A Merge Sort jól párhuzamosítható algoritmus, mivel az alrészek rendezése független egymástól. Ez lehetővé teszi, hogy az algoritmus különböző részeit párhuzamosan futtassuk több processzoron vagy több szálon, tovább javítva a teljesítményt.

#### C++ Implementáció példa

Az alábbiakban bemutatjuk a Merge Sort algoritmus C++ nyelvű implementációját, amely mind a rekurzív, mind a bottom-up megközelítést tartalmazza.

**Rekurzív megközelítés:**

```cpp
#include <iostream>

#include <vector>

void merge(std::vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    std::vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; ++i)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; ++j)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(std::vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        merge(arr, left, mid, right);
    }
}

int main() {
    std::vector<int> arr = {12, 11, 13, 5, 6, 7};
    int arr_size = arr.size();

    std::cout << "Given array is \n";
    for (int i = 0; i < arr_size; i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;

    mergeSort(arr, 0, arr_size - 1);

    std::cout << "\nSorted array is \n";
    for (int i = 0; i < arr_size; i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
    return 0;
}
```

**Bottom-up megközelítés:**

```cpp
#include <iostream>

#include <vector>

void merge(std::vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    std::vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; ++i)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; ++j)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(std::vector<int>& arr) {
    int n = arr.size();

    for (int curr_size = 1; curr_size <= n - 1; curr_size = 2 * curr_size) {
        for (int left_start = 0; left_start < n - 1; left_start += 2 * curr_size) {
            int mid = std::min(left_start + curr_size - 1, n - 1);
            int right_end = std::min(left_start + 2 * curr_size - 1, n - 1);

            merge(arr, left_start, mid, right_end);
        }
    }
}

int main() {
    std::vector<int> arr = {12, 11, 13, 5, 6, 7};
    int arr_size = arr.size();

    std::cout << "Given array is \n";
    for (int i = 0; i < arr_size; i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;

    mergeSort(arr);

    std::cout << "\nSorted array is \n";
    for (int i = 0; i < arr_size; i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
    return 0;
}
```

#### Összegzés

A Merge Sort algoritmus, mind a rekurzív, mind a bottom-up változatban, kiváló példája az oszd meg és uralkodj elv alkalmazásának. Az $O(n \log n)$ időbeli komplexitása és az $O(n)$ térbeli komplexitása miatt az algoritmus hatékonyan használható nagy adatmennyiségek rendezésére. Stabilitása és párhuzamosíthatósága további előnyöket biztosít, különösen adatbázisok és külső rendezési technikák alkalmazásakor. A különböző megközelítések közötti választás az adott alkalmazás követelményeitől és a rendelkezésre álló erőforrásoktól függ, de mindkét megközelítés biztosítja a rendezési feladatok hatékony megoldását.

### 2.5.4. Gyakorlati alkalmazások és példák

A Merge Sort egy sokoldalú és hatékony rendezési algoritmus, amely számos gyakorlati alkalmazásban és példában megállja a helyét. Az algoritmus stabilitása és $O(n \log n)$ időbeli komplexitása miatt különösen hasznos nagy adathalmazok rendezésekor és olyan helyzetekben, ahol az adatok stabilitása kritikus fontosságú. Ebben az alfejezetben részletesen tárgyaljuk a Merge Sort különböző gyakorlati alkalmazásait és bemutatjuk néhány konkrét példán keresztül, hogyan használható az algoritmus a valós világban.

#### Nagy méretű adathalmazok rendezése

A Merge Sort kiválóan alkalmas nagy méretű adathalmazok rendezésére, különösen akkor, ha a memória korlátozott. Az algoritmus külső rendezési technikákban való alkalmazása különösen előnyös, amikor az adatok nem férnek el a memóriában és lemezre kell támaszkodni. A következő lépések illusztrálják a külső rendezés folyamatát a Merge Sort segítségével:

1. **Adatok felosztása**: Az adathalmazt kisebb részekre osztjuk, amelyek elférnek a memóriában.
2. **Rendezés**: Minden egyes részt külön-külön rendezünk a memóriában a Merge Sort használatával.
3. **Írás a lemezre**: A rendezett részeket visszaírjuk a lemezre.
4. **Egyesítés**: Az összes rendezett részt egyesítjük egy végső rendezett adathalmazzá. Ez a lépés szintén a Merge Sort elvén alapul.

Ez a módszer biztosítja, hogy a nagy adathalmazok hatékonyan rendezhetők anélkül, hogy a memória kapacitása szűk keresztmetszetet jelentene.

#### Adatbázis műveletek

Az adatbázis rendszerekben gyakran szükség van rekordok rendezésére különböző szempontok szerint. A Merge Sort stabilitása és hatékonysága miatt ideális választás adatbázis rekordok rendezésére. A stabil rendezés különösen fontos, ha azonos kulcsú rekordok esetén meg kell őrizni az eredeti sorrendet, például időbélyeg alapján történő rendezéskor.

**Példa:**

Tegyük fel, hogy egy adatbázisban található rekordokat kell rendezni név szerint. A Merge Sort alkalmazása biztosítja, hogy az azonos nevű rekordok időbélyeg alapján megőrzik az eredeti sorrendet.

```cpp
#include <iostream>

#include <vector>
#include <string>

struct Record {
    std::string name;
    int timestamp;
};

void merge(std::vector<Record>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    std::vector<Record> L(n1), R(n2);

    for (int i = 0; i < n1; ++i)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; ++j)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2) {
        if (L[i].name <= R[j].name) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(std::vector<Record>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        merge(arr, left, mid, right);
    }
}

int main() {
    std::vector<Record> records = {
        {"Alice", 3}, {"Bob", 2}, {"Alice", 1}, {"Bob", 4}, {"Charlie", 5}
    };

    mergeSort(records, 0, records.size() - 1);

    for (const auto& record : records) {
        std::cout << record.name << " " << record.timestamp << std::endl;
    }
    return 0;
}
```

Ebben a példában az adatbázis rekordokat név szerint rendezzük, miközben megőrizzük az időbélyeg szerinti sorrendet az azonos nevű rekordok esetében.

#### Párhuzamos feldolgozás

A Merge Sort jól párhuzamosítható, mivel az alrészek rendezése egymástól függetlenül történik. Ez lehetővé teszi, hogy a rendezési feladatot több processzorra vagy szálra osszuk szét, növelve ezzel a teljesítményt. Párhuzamos környezetben a Merge Sort implementálása a következőképpen történhet:

1. **Párhuzamos felosztás**: Az adathalmazt párhuzamosan felosztjuk kisebb részekre.
2. **Párhuzamos rendezés**: Minden részt párhuzamosan rendezünk különböző szálakon vagy processzorokon.
3. **Párhuzamos egyesítés**: Az egyesítés szintén párhuzamosan történik, így több szálon vagy processzoron végezhető.

A párhuzamos Merge Sort implementálása C++ nyelven:

```cpp
#include <iostream>

#include <vector>
#include <thread>

void merge(std::vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    std::vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; ++i)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; ++j)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void parallelMergeSort(std::vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        std::thread t1([&] { parallelMergeSort(arr, left, mid); });
        std::thread t2([&] { parallelMergeSort(arr, mid + 1, right); });

        t1.join();
        t2.join();

        merge(arr, left, mid, right);
    }
}

int main() {
    std::vector<int> arr = {12, 11, 13, 5, 6, 7};
    int arr_size = arr.size();

    std::cout << "Given array is \n";
    for (int i = 0; i < arr_size; i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;

    parallelMergeSort(arr, 0, arr_size - 1);

    std::cout << "\nSorted array is \n";
    for (int i = 0; i < arr_size; i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
    return 0;
}
```

#### Gyakorlati példák

1. **Web szerver naplófájlok rendezése**: Egy web szerver naplófájljait gyakran kell időbélyeg szerint rendezni, hogy elemezni lehessen a látogatók tevékenységét. A Merge Sort használatával a naplófájlok hatékonyan rendezhetők, különösen akkor, ha a naplófájlok mérete meghaladja a memória kapacitását.

2. **Nagy adathalmazok előfeldolgozása**: Adatbányászati vagy gépi tanulási feladatok esetén gyakran elő kell dolgozni az adatokat, ami rendezést igényelhet. A Merge Sort segítségével az adatok hatékonyan rendezhetők elő a további feld

olgozási lépések előtt.

3. **Adatfolyam rendezése**: Streaming adatok esetén, ahol az adatok folyamatosan érkeznek, a Merge Sort használható az adatok rendezett állapotban történő fenntartására. Az algoritmus képes folyamatosan rendezni az adatokat, miközben új adatok érkeznek.

#### Összegzés

A Merge Sort egy rendkívül hatékony és sokoldalú rendezési algoritmus, amely számos gyakorlati alkalmazásban bizonyította hasznosságát. Legyen szó nagy adathalmazok rendezéséről, adatbázis rekordok stabil rendezéséről, párhuzamos feldolgozásról vagy más speciális feladatokról, a Merge Sort kiválóan alkalmas ezeknek a kihívásoknak a kezelésére. Az algoritmus stabilitása, hatékonysága és párhuzamosíthatósága miatt továbbra is az egyik legfontosabb eszköz marad a rendezési algoritmusok között.

