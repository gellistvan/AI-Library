\newpage

## 2.3.   Beszúró rendezés (Insertion Sort)

A beszúró rendezés egy egyszerű és hatékony algoritmus, amely különösen jól működik kisebb adathalmazok rendezésekor vagy majdnem rendezett listák esetén. Az algoritmus lényege, hogy az elemeket egymás után veszi sorra, és minden egyes elemet a helyére szúr be a már részben rendezett listában. Ez a módszer az emberek által is gyakran alkalmazott rendezési technikát imitálja, például amikor kártyákat rendezünk a kezünkben. A következő alfejezetekben részletesen megismerkedünk a beszúró rendezés alapelveivel és implementációjával, valamint annak egy optimalizált változatával, a bináris beszúró rendezéssel. Ezen túlmenően, elemezzük az algoritmus teljesítményét és komplexitását, és gyakorlati példákon keresztül bemutatjuk, hogyan alkalmazható hatékonyan különböző feladatok megoldására.

### 2.3.1. Alapelvek és implementáció

A beszúró rendezés (Insertion Sort) az egyik legegyszerűbb és legintuitívabb rendezési algoritmus. Bár gyakran kevésbé hatékony, mint a bonyolultabb algoritmusok, mint például a gyorsrendezés (Quicksort) vagy a halmazrendezés (Merge Sort), számos előnye miatt mégis figyelemre méltó. Ebben az alfejezetben részletesen megvizsgáljuk a beszúró rendezés alapelveit, az algoritmus működését, annak implementációját, valamint erősségeit és gyengeségeit.

#### Az algoritmus alapelvei

A beszúró rendezés alapelve a következőképpen foglalható össze:
1. Az algoritmus a bemeneti lista elemeit egyesével veszi sorra.
2. Minden egyes elem beillesztésre kerül a már rendezett részlistába, a megfelelő helyre.
3. Ezáltal az aktuálisan vizsgált elemmel bővített részlista mindig rendezett marad.

Az algoritmus során az aktuálisan kiválasztott elemet "beszúrjuk" a már rendezett elemek közé, úgy, hogy a rendezett rész továbbra is rendezett maradjon.

#### Az algoritmus lépései

Az algoritmus lépései a következők:
1. Kezdjük az első elemmel, amely önmagában egy rendezett lista.
2. A második elemet hasonlítsuk össze az elsővel, és ha szükséges, cseréljük meg őket, hogy rendezett legyen.
3. Folytassuk a harmadik elemmel: hasonlítsuk össze az előzőekkel, és illesszük be a megfelelő helyre.
4. Ismételjük meg ezt a folyamatot a lista minden elemére, amíg az egész lista rendezetté nem válik.

#### Példák

Vegyük például a következő nem rendezett listát: [5, 2, 9, 1, 5, 6].

1. Kezdjük az első elemmel: [5]
2. Vegyük a második elemet (2) és illesszük be az első elé: [2, 5]
3. Vegyük a harmadik elemet (9) és illesszük be a megfelelő helyre: [2, 5, 9]
4. Folytassuk a negyedik elemmel (1), illesszük be az elejére: [1, 2, 5, 9]
5. Az ötödik elemet (5) illesszük be a megfelelő helyre: [1, 2, 5, 5, 9]
6. Végül a hatodik elemet (6) illesszük be: [1, 2, 5, 5, 6, 9]

#### Implementáció

Az alábbiakban bemutatunk egy C++ nyelvű implementációt a beszúró rendezéshez:

```cpp
#include <iostream>
#include <vector>

void insertionSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 1; i < n; ++i) {
        int key = arr[i];
        int j = i - 1;
        
        // Move elements of arr[0..i-1], that are greater than key,
        // to one position ahead of their current position
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

void printArray(const std::vector<int>& arr) {
    for (int i = 0; i < arr.size(); i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

int main() {
    std::vector<int> arr = {5, 2, 9, 1, 5, 6};
    insertionSort(arr);
    printArray(arr);
    return 0;
}
```

#### Az algoritmus hatékonysága

A beszúró rendezés időkomplexitása legrosszabb esetben $O(n^2)$, ahol $n$ a rendezendő elemek száma. Ez akkor fordul elő, amikor a bemeneti lista elemei fordított sorrendben vannak rendezve. Az algoritmus legjobb esetben $O(n)$ időkomplexitású, amikor a lista már majdnem rendezett. Az átlagos időkomplexitás szintén $O(n^2)$.

Az algoritmus helyigénye $O(1)$, mivel csak egy konstans mennyiségű extra memóriát igényel a cserék végrehajtásához.

#### Előnyök és hátrányok

A beszúró rendezés számos előnnyel rendelkezik:
- Egyszerűen implementálható és érthető.
- Hatékony kisebb vagy majdnem rendezett listák esetén.
- Stabil rendezési algoritmus, mivel nem cseréli fel az egyenlő értékű elemek sorrendjét.

Ugyanakkor hátrányai is vannak:
- Nagyobb adathalmazok esetén kevésbé hatékony, mint más rendezési algoritmusok.
- Quadratikus időkomplexitása miatt nem alkalmas nagy adathalmazok rendezésére.

#### Összegzés

A beszúró rendezés egy alapvető, mégis fontos algoritmus, amelyet érdemes ismerni, különösen kisebb adathalmazok vagy speciális esetek rendezésére. Bár nem a leghatékonyabb rendezési módszer, az egyszerűsége és az intuitív működése miatt számos gyakorlati alkalmazásban megállja a helyét. A következő alfejezetben a bináris beszúró rendezést tárgyaljuk, amely a beszúró rendezés optimalizált változata, további előnyöket nyújtva bizonyos körülmények között.

### 2.3.2. Bináris beszúró rendezés

A beszúró rendezés (Insertion Sort) egy intuitív és egyszerű rendezési algoritmus, amely különösen hatékony kisebb vagy részben rendezett adathalmazok esetén. Azonban a hagyományos beszúró rendezés időkomplexitása legrosszabb esetben $O(n^2)$, amely jelentős hátrány lehet nagyobb adathalmazok rendezésekor. A bináris beszúró rendezés (Binary Insertion Sort) ezen a problémán kíván javítani azáltal, hogy a beszúró pozíció keresését hatékonyabbá teszi bináris keresés (Binary Search) alkalmazásával. Ebben az alfejezetben részletesen megvizsgáljuk a bináris beszúró rendezés alapelveit, működését és implementációját, valamint annak hatékonysági elemzését.

#### Az algoritmus alapelvei

A bináris beszúró rendezés alapötlete a következő:
1. Minden elem beszúrásakor nem lineáris kereséssel találjuk meg a helyes pozíciót, hanem bináris kereséssel.
2. A bináris keresés alkalmazásával a beszúró pozíció keresésének időkomplexitása $O(\log n)$-re csökken.
3. Az elemeket ezután a megfelelő pozícióba szúrjuk be, amely egy eltolási műveletet igényel, hasonlóan a hagyományos beszúró rendezéshez.

#### Az algoritmus lépései

A bináris beszúró rendezés algoritmusa a következő lépésekből áll:
1. Kezdjük az első elemmel, amely önmagában egy rendezett lista.
2. A második elemtől kezdve minden egyes elemet bináris kereséssel helyezünk be a megfelelő pozícióba a már rendezett részlistában.
3. Az elemeket a megtalált pozícióba szúrjuk be, amely az eltolási műveletek miatt továbbra is $O(n)$ időkomplexitású.

#### Az algoritmus implementációja

Az alábbiakban bemutatunk egy C++ nyelvű implementációt a bináris beszúró rendezéshez:

```cpp
#include <iostream>
#include <vector>

// Function to perform binary search
int binarySearch(const std::vector<int>& arr, int item, int low, int high) {
    if (high <= low)
        return (item > arr[low]) ? (low + 1) : low;

    int mid = (low + high) / 2;

    if (item == arr[mid])
        return mid + 1;

    if (item > arr[mid])
        return binarySearch(arr, item, mid + 1, high);
    return binarySearch(arr, item, low, mid - 1);
}

// Function to perform insertion sort using binary search
void binaryInsertionSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 1; i < n; ++i) {
        int key = arr[i];
        int j = i - 1;

        // Find location where key should be inserted
        int loc = binarySearch(arr, key, 0, j);

        // Move all elements after location to create space
        while (j >= loc) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

void printArray(const std::vector<int>& arr) {
    for (int i = 0; i < arr.size(); i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

int main() {
    std::vector<int> arr = {5, 2, 9, 1, 5, 6};
    binaryInsertionSort(arr);
    printArray(arr);
    return 0;
}
```

#### Az algoritmus hatékonysága

A bináris beszúró rendezés időkomplexitása az elemek közötti keresés szempontjából $O(\log n)$, mivel bináris keresést alkalmazunk. Azonban az elemek eltolási műveletei továbbra is $O(n)$ időigényűek. Így az algoritmus teljes időkomplexitása legrosszabb esetben $O(n \log n)$, amely jelentős javulást jelent a hagyományos beszúró rendezés $O(n^2)$ időkomplexitásához képest.

Az algoritmus helyigénye továbbra is $O(1)$, mivel csak egy konstans mennyiségű extra memóriát igényel a cserék végrehajtásához és a bináris kereséshez.

#### Előnyök és hátrányok

A bináris beszúró rendezés előnyei:
- Hatékonyabb, mint a hagyományos beszúró rendezés, különösen nagyobb adathalmazok esetén.
- Stabil rendezési algoritmus, amely nem változtatja meg az egyenlő értékű elemek sorrendjét.
- Egyszerűen implementálható és könnyen érthető.

Hátrányai:
- Bár hatékonyabb, mint a hagyományos beszúró rendezés, a bináris beszúró rendezés még mindig nem olyan hatékony, mint más modern rendezési algoritmusok, mint például a gyorsrendezés vagy a halmazrendezés.
- Az elemek eltolási műveletei miatt továbbra is jelentős időigényű lehet nagyobb adathalmazok esetén.

#### Összegzés

A bináris beszúró rendezés a hagyományos beszúró rendezés optimalizált változata, amely bináris keresést alkalmaz a beszúró pozíció hatékony megtalálására. Bár az algoritmus jelentős javulást mutat a hagyományos beszúró rendezéshez képest, különösen nagyobb adathalmazok esetén, még mindig nem éri el a modern rendezési algoritmusok hatékonysági szintjét. Mindazonáltal, a bináris beszúró rendezés egy fontos és hasznos algoritmus, amelyet érdemes ismerni és alkalmazni kisebb vagy részben rendezett adathalmazok esetén. A következő alfejezetben az algoritmus teljesítményének és komplexitásának részletes elemzését végezzük el.

### 2.3.3. Teljesítmény és komplexitás elemzése

A beszúró rendezés (Insertion Sort) és annak optimalizált változata, a bináris beszúró rendezés (Binary Insertion Sort) hatékonysága és komplexitása több szempontból is vizsgálható. Az algoritmusok teljesítményének elemzése során figyelembe vesszük az időkomplexitást, a helyigényt, a stabilitást és a gyakorlati alkalmazhatóságot. Ebben az alfejezetben részletesen megvizsgáljuk mindkét algoritmus teljesítményét és komplexitását.

#### Hagyományos beszúró rendezés teljesítménye

##### Időkomplexitás

A hagyományos beszúró rendezés időkomplexitása az alábbi módon alakul:

- **Legrosszabb eset (Worst-case):** Az algoritmus legrosszabb esetben $O(n^2)$ időkomplexitású, amely akkor fordul elő, amikor a bemeneti lista elemei fordított sorrendben vannak rendezve. Ilyenkor minden egyes elem beszúrásakor az összes korábbi elemet át kell mozgatni, ami $n-1$ műveletet jelent az első elem után, $n-2$ műveletet a második elem után, és így tovább, egészen $1$ műveletig az utolsó elem előtt. Az összeg: $\sum_{i=1}^{n-1} i = \frac{n(n-1)}{2}$, ami $O(n^2)$ növekedést jelent.

- **Legjobb eset (Best-case):** A legjobb esetben a bemeneti lista már rendezett, így az algoritmusnak csak egyszer kell végigmennie a listán, hogy minden elemet ellenőrizzen, és egyetlen eltolási műveletet sem kell végrehajtania. Ekkor az időkomplexitás $O(n)$.

- **Átlagos eset (Average-case):** Az átlagos esetben a bemeneti lista elemei véletlenszerűen vannak elosztva. Az időkomplexitás ekkor is $O(n^2)$, mivel az átlagos esetben is sok eltolási műveletre van szükség.

##### Helyigény (Space Complexity)

A beszúró rendezés helyigénye $O(1)$, mivel az algoritmus csak egy konstans mennyiségű extra memóriát igényel a cserék végrehajtásához és az aktuálisan vizsgált elem tárolásához. Az algoritmus in-place működik, vagyis a rendezést a bemeneti listán belül végzi el anélkül, hogy jelentős további memóriát használna.

##### Stabilitás

A hagyományos beszúró rendezés stabil rendezési algoritmus, mivel nem változtatja meg az egyenlő értékű elemek sorrendjét. Ez azt jelenti, hogy ha két elem egyenlő értékű, akkor a rendezett lista ugyanabban a sorrendben tartalmazza őket, mint a bemeneti lista.

#### Bináris beszúró rendezés teljesítménye

##### Időkomplexitás

A bináris beszúró rendezés időkomplexitása az alábbi módon alakul:

- **Legrosszabb eset (Worst-case):** Az algoritmus legrosszabb esetben $O(n \log n)$ időkomplexitású, mivel a beszúró pozíció keresése bináris kereséssel történik, amely $O(\log n)$ időigényű. Azonban az elemek eltolási műveletei továbbra is $O(n)$ időigényűek, így az összes művelet $O(n \log n)$.

- **Legjobb eset (Best-case):** A legjobb esetben a bemeneti lista már rendezett. Ebben az esetben is $O(n \log n)$ időkomplexitású az algoritmus, mivel minden egyes elem bináris keresést igényel a pozíciójának meghatározásához, bár eltolási műveletre nincs szükség.

- **Átlagos eset (Average-case):** Az átlagos esetben a bemeneti lista elemei véletlenszerűen vannak elosztva. Az időkomplexitás ekkor is $O(n \log n)$, mivel a bináris keresés $O(\log n)$ időigényű, és az átlagos eltolási műveletek száma $O(n)$.

##### Helyigény (Space Complexity)

A bináris beszúró rendezés helyigénye megegyezik a hagyományos beszúró rendezés helyigényével, azaz $O(1)$. Az algoritmus in-place működik, és csak egy konstans mennyiségű extra memóriát igényel a bináris keresés végrehajtásához és az aktuálisan vizsgált elem tárolásához.

##### Stabilitás

A bináris beszúró rendezés is stabil rendezési algoritmus, mivel a bináris keresés során az egyenlő értékű elemek nem változtatják meg sorrendjüket, és az eltolási műveletek is megőrzik az elemek eredeti sorrendjét.

#### Összehasonlítás más rendezési algoritmusokkal

- **Gyorsrendezés (Quicksort):** A gyorsrendezés átlagos időkomplexitása $O(n \log n)$, de legrosszabb esetben $O(n^2)$ lehet. Helyigénye általában $O(\log n)$, de nem stabil rendezési algoritmus. A gyorsrendezés általában gyorsabb, mint a beszúró rendezés nagyobb adathalmazok esetén.

- **Halmazrendezés (Merge Sort):** A halmazrendezés időkomplexitása mindig $O(n \log n)$, és helyigénye $O(n)$. Stabil rendezési algoritmus, és gyakran használják nagyobb adathalmazok rendezésére.

- **Kupac rendezés (Heap Sort):** A kupac rendezés időkomplexitása $O(n \log n)$, de nem stabil rendezési algoritmus. Helyigénye $O(1)$.

#### Gyakorlati alkalmazhatóság

A hagyományos beszúró rendezés és a bináris beszúró rendezés is jól alkalmazható kisebb adathalmazok rendezésére, különösen olyan esetekben, amikor a bemeneti lista majdnem rendezett. Ezek az algoritmusok gyakran használatosak beágyazott rendszerekben és alacsony memóriaigényű alkalmazásokban, ahol a helyigény és az egyszerű implementáció kulcsfontosságú.

Mindkét algoritmus hatékony lehet hibrid rendezési algoritmusok részeként, ahol nagyobb adathalmazok durva rendezését más algoritmus végzi, majd a finomhangolást a beszúró rendezés vagy bináris beszúró rendezés végzi el.

#### Összegzés

A hagyományos beszúró rendezés és a bináris beszúró rendezés egyszerű és hatékony algoritmusok kisebb vagy majdnem rendezett adathalmazok esetén. Míg a hagyományos beszúró rendezés időkomplexitása legrosszabb esetben $O(n^2)$, a bináris beszúró rendezés javított változata $O(n \log n)$ időkomplexitást kínál. Mindkét algoritmus stabil és alacsony helyigényű, ezért jól alkalmazhatók bizonyos gyakorlati feladatokban. A következő alfejezetben gyakorlati alkalmazásokkal és példákkal foglalkozunk, amelyek bemutatják, hogyan használhatók ezek az algoritmusok különböző környezetekben.

### 2.3.4. Gyakorlati alkalmazások és példák

A beszúró rendezés (Insertion Sort) és a bináris beszúró rendezés (Binary Insertion Sort) algoritmusai számos gyakorlati alkalmazásban hasznosak. Bár ezek az algoritmusok időkomplexitásuk miatt kevésbé hatékonyak nagyobb adathalmazok esetén, bizonyos speciális esetekben és kisebb adathalmazoknál kifejezetten előnyösek. Ebben az alfejezetben részletesen megvizsgáljuk, milyen területeken alkalmazhatóak ezek az algoritmusok, és bemutatunk néhány gyakorlati példát.

#### Gyakorlati alkalmazások

##### Kisebb adathalmazok rendezése

A beszúró rendezés hatékonyan működik kisebb adathalmazok esetén, ahol az $O(n^2)$ időkomplexitás nem jelent komoly hátrányt. Ezek az algoritmusok különösen hasznosak olyan esetekben, amikor az adathalmaz mérete néhány száz elem alatt van. Például beágyazott rendszerekben és mikrokontrollerekben, ahol az erőforrások korlátozottak, a beszúró rendezés egyszerűsége és alacsony memóriaigénye előnyt jelenthet.

##### Majdnem rendezett listák

Majdnem rendezett listák esetén a beszúró rendezés kifejezetten hatékony, mivel kevés elemcsere szükséges a lista rendezéséhez. Sok valós világban előforduló adat gyakran majdnem rendezett, például egy könyvtárban az új könyvek hozzáadása az előző hónapban már rendezett könyvekhez. Ilyen esetekben a beszúró rendezés közel lineáris időkomplexitással fut, ami jelentős előnyt jelenthet.

##### Online rendezés

Az online rendezés olyan helyzetekben hasznos, amikor az adatokat folyamatosan kell rendezni, ahogy azok beérkeznek. A beszúró rendezés ideális erre a célra, mivel egyszerűen integrálható olyan rendszerekbe, ahol az adatok időben érkeznek, és azonnal rendezni kell őket. Például egy élő adatokkal dolgozó rendszerben, mint amilyen egy tőzsdei adatokat kezelő alkalmazás, ahol az új adatok folyamatosan érkeznek és integrálódnak a már meglévő rendezett adathalmazba.

##### Hibakeresés és oktatás

A beszúró rendezés egyszerűsége és intuitív működése miatt kiválóan alkalmas oktatási célokra. A programozás és algoritmusok tanításakor gyakran használják példaként, hogy bemutassák az alapvető rendezési elveket és technikákat. Emellett hibakeresési (debugging) feladatokhoz is hasznos, mivel könnyen követhető és ellenőrizhető lépései vannak.

#### Példák

##### Példa 1: Rendezett listába beszúrás

Képzeljük el, hogy van egy alkalmazásunk, amely folyamatosan kap új adatokat, és ezeket be kell szúrnia egy már rendezett listába. Tegyük fel, hogy egy könyvtár rendszerében dolgozunk, ahol az új könyveket folyamatosan hozzá kell adnunk a már rendezett könyvek listájához cím szerint.

```cpp
#include <iostream>
#include <vector>
#include <string>

struct Book {
    std::string title;
    std::string author;
    int year;
};

int binarySearch(const std::vector<Book>& books, const Book& newBook, int low, int high) {
    if (high <= low)
        return (newBook.title > books[low].title) ? (low + 1) : low;

    int mid = (low + high) / 2;

    if (newBook.title == books[mid].title)
        return mid + 1;

    if (newBook.title > books[mid].title)
        return binarySearch(books, newBook, mid + 1, high);
    return binarySearch(books, newBook, low, mid - 1);
}

void insertBook(std::vector<Book>& books, const Book& newBook) {
    int n = books.size();
    int pos = binarySearch(books, newBook, 0, n - 1);
    books.insert(books.begin() + pos, newBook);
}

void printBooks(const std::vector<Book>& books) {
    for (const auto& book : books) {
        std::cout << book.title << " by " << book.author << " (" << book.year << ")\n";
    }
}

int main() {
    std::vector<Book> books = {
        {"A Tale of Two Cities", "Charles Dickens", 1859},
        {"Moby Dick", "Herman Melville", 1851},
        {"Pride and Prejudice", "Jane Austen", 1813},
        {"The Great Gatsby", "F. Scott Fitzgerald", 1925}
    };

    Book newBook = {"1984", "George Orwell", 1949};
    insertBook(books, newBook);

    printBooks(books);
    return 0;
}
```

##### Példa 2: Online rendezés

Egy online adatokat kezelő rendszer, például egy tőzsdei alkalmazás, ahol az új tranzakciók folyamatosan érkeznek, és azokat azonnal be kell szúrni a már rendezett tranzakciós listába.

```cpp
#include <iostream>
#include <vector>

struct Transaction {
    int id;
    double amount;
    std::string date;
};

int binarySearch(const std::vector<Transaction>& transactions, const Transaction& newTransaction, int low, int high) {
    if (high <= low)
        return (newTransaction.amount > transactions[low].amount) ? (low + 1) : low;

    int mid = (low + high) / 2;

    if (newTransaction.amount == transactions[mid].amount)
        return mid + 1;

    if (newTransaction.amount > transactions[mid].amount)
        return binarySearch(transactions, newTransaction, mid + 1, high);
    return binarySearch(transactions, newTransaction, low, mid - 1);
}

void insertTransaction(std::vector<Transaction>& transactions, const Transaction& newTransaction) {
    int n = transactions.size();
    int pos = binarySearch(transactions, newTransaction, 0, n - 1);
    transactions.insert(transactions.begin() + pos, newTransaction);
}

void printTransactions(const std::vector<Transaction>& transactions) {
    for (const auto& transaction : transactions) {
        std::cout << "ID: " << transaction.id << ", Amount: " << transaction.amount << ", Date: " << transaction.date << "\n";
    }
}

int main() {
    std::vector<Transaction> transactions = {
        {1, 1000.50, "2024-05-20"},
        {2, 250.75, "2024-05-21"},
        {3, 1500.00, "2024-05-22"}
    };

    Transaction newTransaction = {4, 500.00, "2024-05-23"};
    insertTransaction(transactions, newTransaction);

    printTransactions(transactions);
    return 0;
}
```

#### Gyakorlati hatékonyság és optimalizálás

Az algoritmusok gyakorlati alkalmazhatósága szempontjából fontos megérteni, hogy mikor érdemes ezeket használni, és milyen optimalizálási lehetőségek állnak rendelkezésre. Például, a bináris beszúró rendezés hatékonysága növelhető az eltolási műveletek optimalizálásával, például blokkokban történő áthelyezéssel.

#### Összegzés

A beszúró rendezés és a bináris beszúró rendezés algoritmusai számos gyakorlati alkalmazásban hasznosak, különösen kisebb adathalmazok rendezésekor, majdnem rendezett listák esetén, online rendezési feladatoknál, valamint oktatási célokra. Az algoritmusok egyszerűsége és stabilitása előnyt jelenthet bizonyos speciális esetekben, ahol a nagyobb komplexitású rendezési algoritmusok nem szükségesek. A következő fejezetben további rendezési algoritmusokat tárgyalunk, és összehasonlítjuk azok hatékonyságát és alkalmazhatóságát a beszúró rendezéssel.

