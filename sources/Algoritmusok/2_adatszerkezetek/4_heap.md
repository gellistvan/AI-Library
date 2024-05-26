\newpage 
# 4. Heaps (Halmok)

A halmok (heaps) olyan speciális fa alapú adatszerkezetek, amelyek kulcsfontosságú szerepet játszanak számos algoritmus és adatszerkezet működésében. Különösen előnyösek prioritási sorok kezelésében és rendezési algoritmusok, mint például a heapsort megvalósításában. A halmok leggyakoribb típusai a Min-Heap és a Max-Heap, amelyek eltérő rendezési tulajdonságokkal rendelkeznek: a Min-Heap-ben a szülő csomópontok mindig kisebbek vagy egyenlőek a gyermekeiknél, míg a Max-Heap-ben a szülő csomópontok mindig nagyobbak vagy egyenlőek a gyermekeiknél. Ebben a fejezetben részletesen bemutatjuk a halmok alapfogalmait, tömbökkel való implementációját, valamint az alapvető műveleteket, mint a beszúrás, törlés és a heapify. Végül megismerkedünk a heapsort algoritmussal és a prioritási sorok alkalmazásával, amelyek széles körben használatosak az informatika különböző területein.

## 4.1 Min-Heap és Max-Heap alapfogalmak

A halmok, vagy angolul heaps, olyan speciális bináris fák, amelyek számos algoritmus és adatszerkezet alapjául szolgálnak, különösen a prioritási sorok és a rendezési algoritmusok terén. Két alapvető típusa van a halmoknak: a Min-Heap és a Max-Heap. Ezek a halmok különböző rendezési tulajdonságokkal rendelkeznek, amelyek meghatározzák, hogyan szerveződnek a csomópontok az adatstruktúrán belül.

### Min-Heap

A Min-Heap egy bináris fa, amelyben minden egyes csomópont értéke kisebb vagy egyenlő a gyermekeinek értékével. Ez azt jelenti, hogy a fa gyökerében található a legkisebb elem. A Min-Heap egyik legfontosabb tulajdonsága, hogy a legkisebb elem könnyen elérhető, ami különösen hasznos prioritási sorok megvalósításánál.

**Tulajdonságok:**
- A gyökér mindig a legkisebb elem.
- Bármely csomópont értéke kisebb vagy egyenlő a gyermekeinek értékénél.
- A fa magassága logaritmikus az elemek számához képest, ami biztosítja az egyensúlyi állapotot.

### Max-Heap

A Max-Heap egy bináris fa, amelyben minden egyes csomópont értéke nagyobb vagy egyenlő a gyermekeinek értékével. Ebben a struktúrában a gyökér a legnagyobb elem, ami szintén hasznos lehet számos alkalmazásban, például prioritási sorok megvalósításánál, ahol a legmagasabb prioritású elemet szeretnénk gyorsan elérni.

**Tulajdonságok:**
- A gyökér mindig a legnagyobb elem.
- Bármely csomópont értéke nagyobb vagy egyenlő a gyermekeinek értékénél.
- A fa magassága itt is logaritmikus az elemek számához képest.

### Min-Heap és Max-Heap Implementációja Tömbökkel

A halmok implementálása gyakran tömbökkel történik, mivel ez egyszerű és hatékony módja az adatok tárolásának és kezelésének. A fa szerkezete jól leképezhető egy lineáris tömbben, ahol a csomópontok szülő-gyermek kapcsolatai könnyen kezelhetők.

**Szülő-gyermek kapcsolat tömbben:**
- Ha egy csomópont indexe `i`, akkor:
    - Bal gyermeke az indexen: `2*i + 1`
    - Jobb gyermeke az indexen: `2*i + 2`
    - Szülője az indexen: `(i - 1) / 2`

Ez az elrendezés lehetővé teszi, hogy a csomópontok közötti kapcsolatok gyorsan és hatékonyan elérhetők legyenek, ami különösen fontos a heap műveletek, mint a beszúrás, törlés és a heapify esetében.

### Példakód C++ nyelven: Min-Heap Implementáció

Az alábbiakban bemutatunk egy egyszerű Min-Heap implementációt C++ nyelven, amely tartalmazza a beszúrás és törlés műveleteket, valamint a heapify eljárást.

```cpp
#include <iostream>
#include <vector>
#include <stdexcept>

class MinHeap {
private:
    std::vector<int> heap;

    void heapifyDown(int index) {
        int smallest = index;
        int leftChild = 2 * index + 1;
        int rightChild = 2 * index + 2;

        if (leftChild < heap.size() && heap[leftChild] < heap[smallest]) {
            smallest = leftChild;
        }

        if (rightChild < heap.size() && heap[rightChild] < heap[smallest]) {
            smallest = rightChild;
        }

        if (smallest != index) {
            std::swap(heap[index], heap[smallest]);
            heapifyDown(smallest);
        }
    }

    void heapifyUp(int index) {
        int parent = (index - 1) / 2;
        if (index && heap[parent] > heap[index]) {
            std::swap(heap[index], heap[parent]);
            heapifyUp(parent);
        }
    }

public:
    void insert(int key) {
        heap.push_back(key);
        int index = heap.size() - 1;
        heapifyUp(index);
    }

    void deleteMin() {
        if (heap.empty()) {
            throw std::out_of_range("Heap is empty");
        }

        heap[0] = heap.back();
        heap.pop_back();
        heapifyDown(0);
    }

    int getMin() {
        if (heap.empty()) {
            throw std::out_of_range("Heap is empty");
        }
        return heap.front();
    }

    void printHeap() {
        for (int i : heap) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }
};

int main() {
    MinHeap heap;
    heap.insert(3);
    heap.insert(2);
    heap.insert(15);
    heap.insert(5);
    heap.insert(4);
    heap.insert(45);

    std::cout << "Min-Heap array: ";
    heap.printHeap();

    std::cout << "The minimum element is " << heap.getMin() << std::endl;

    heap.deleteMin();
    std::cout << "After deleting the minimum element, the heap is: ";
    heap.printHeap();

    return 0;
}
```

### Magyarázat a C++ Kódhoz

1. **Osztály Deklaráció**: A `MinHeap` osztály egy privát adattagot tartalmaz, ami egy vektor, amely a heap elemeit tárolja.
2. **heapifyDown() és heapifyUp()**: Ezek a privát metódusok biztosítják a heap tulajdonságok fenntartását. A `heapifyDown()` a törlés művelet után használatos, míg a `heapifyUp()` a beszúrás művelet után.
3. **insert()**: Ez a metódus új elemet ad a heap-hez, majd a `heapifyUp()` segítségével helyreállítja a heap tulajdonságokat.
4. **deleteMin()**: Ez a metódus eltávolítja a legkisebb elemet (a gyökeret) és a `heapifyDown()` segítségével helyreállítja a heap tulajdonságokat.
5. **getMin()**: Ez a metódus visszaadja a legkisebb elemet a heap-ből.
6. **printHeap()**: Ez a metódus kiírja a heap elemeit.

Ez a példa jól szemlélteti a Min-Heap alapvető működését, és hogyan használható hatékonyan prioritási sorok megvalósítására. Hasonló módon, egy Max-Heap implementáció is megvalósítható, ahol a `heapifyDown()` és `heapifyUp()` metódusokban a kisebb-nagyobb összehasonlításokat felcseréljük.

## 4.2 Implementáció tömbökkel

A halmok (heaps) implementálása tömbökkel egy hatékony és egyszerű módja annak, hogy egy logikai fastruktúrát lineáris adatstruktúrában valósítsunk meg. A tömbös megvalósítás előnyei közé tartozik a helytakarékosság és a gyors hozzáférés a csomópontokhoz. Ebben az alfejezetben részletesen megvizsgáljuk, hogyan lehet a Min-Heap és a Max-Heap struktúrákat tömbökkel megvalósítani, beleértve az alapvető műveleteket és azok hatékonyságát. A bemutatott példakódok C++ nyelven íródtak.

### Tömbös Megvalósítás Általános Elvei

A bináris halmok, legyenek azok Min-Heap-ek vagy Max-Heap-ek, könnyen ábrázolhatók egy tömbben. A tömb indexei és a fa csomópontjai közötti kapcsolatok meghatározott szabályokat követnek, amelyek lehetővé teszik a gyors hozzáférést és módosítást.

**Szülő-gyermek kapcsolat tömbben:**
- Ha egy csomópont indexe `i`, akkor:
    - Bal gyermeke az indexen: `2*i + 1`
    - Jobb gyermeke az indexen: `2*i + 2`
    - Szülője az indexen: `(i - 1) / 2`

Ez a kapcsolat biztosítja, hogy a csomópontok közötti kapcsolatok könnyen kezelhetők legyenek a tömbön belül.

### Min-Heap Tömbös Megvalósítása

A Min-Heap egy olyan bináris fa, amelyben minden csomópont értéke kisebb vagy egyenlő a gyermekeinek értékével. Az alábbiakban bemutatjuk egy Min-Heap tömbös megvalósítását C++ nyelven.

```cpp
#include <iostream>
#include <vector>
#include <stdexcept>

class MinHeap {
private:
    std::vector<int> heap;

    void heapifyDown(int index) {
        int smallest = index;
        int leftChild = 2 * index + 1;
        int rightChild = 2 * index + 2;

        if (leftChild < heap.size() && heap[leftChild] < heap[smallest]) {
            smallest = leftChild;
        }

        if (rightChild < heap.size() && heap[rightChild] < heap[smallest]) {
            smallest = rightChild;
        }

        if (smallest != index) {
            std::swap(heap[index], heap[smallest]);
            heapifyDown(smallest);
        }
    }

    void heapifyUp(int index) {
        int parent = (index - 1) / 2;
        if (index && heap[parent] > heap[index]) {
            std::swap(heap[index], heap[parent]);
            heapifyUp(parent);
        }
    }

public:
    void insert(int key) {
        heap.push_back(key);
        int index = heap.size() - 1;
        heapifyUp(index);
    }

    void deleteMin() {
        if (heap.empty()) {
            throw std::out_of_range("Heap is empty");
        }

        heap[0] = heap.back();
        heap.pop_back();
        heapifyDown(0);
    }

    int getMin() {
        if (heap.empty()) {
            throw std::out_of_range("Heap is empty");
        }
        return heap.front();
    }

    void printHeap() {
        for (int i : heap) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }
};

int main() {
    MinHeap heap;
    heap.insert(3);
    heap.insert(2);
    heap.insert(15);
    heap.insert(5);
    heap.insert(4);
    heap.insert(45);

    std::cout << "Min-Heap array: ";
    heap.printHeap();

    std::cout << "The minimum element is " << heap.getMin() << std::endl;

    heap.deleteMin();
    std::cout << "After deleting the minimum element, the heap is: ";
    heap.printHeap();

    return 0;
}
```

**Magyarázat a Kódhoz:**

1. **Osztály Deklaráció**: A `MinHeap` osztály egy privát adattagot tartalmaz, amely egy vektor, amely a heap elemeit tárolja.
2. **heapifyDown() és heapifyUp()**: Ezek a privát metódusok biztosítják a heap tulajdonságok fenntartását. A `heapifyDown()` a törlés művelet után használatos, míg a `heapifyUp()` a beszúrás művelet után.
3. **insert()**: Ez a metódus új elemet ad a heap-hez, majd a `heapifyUp()` segítségével helyreállítja a heap tulajdonságokat.
4. **deleteMin()**: Ez a metódus eltávolítja a legkisebb elemet (a gyökeret) és a `heapifyDown()` segítségével helyreállítja a heap tulajdonságokat.
5. **getMin()**: Ez a metódus visszaadja a legkisebb elemet a heap-ből.
6. **printHeap()**: Ez a metódus kiírja a heap elemeit.

### Max-Heap Tömbös Megvalósítása

A Max-Heap egy olyan bináris fa, amelyben minden csomópont értéke nagyobb vagy egyenlő a gyermekeinek értékével. Az alábbiakban bemutatjuk egy Max-Heap tömbös megvalósítását C++ nyelven.

```cpp
#include <iostream>
#include <vector>
#include <stdexcept>

class MaxHeap {
private:
    std::vector<int> heap;

    void heapifyDown(int index) {
        int largest = index;
        int leftChild = 2 * index + 1;
        int rightChild = 2 * index + 2;

        if (leftChild < heap.size() && heap[leftChild] > heap[largest]) {
            largest = leftChild;
        }

        if (rightChild < heap.size() && heap[rightChild] > heap[largest]) {
            largest = rightChild;
        }

        if (largest != index) {
            std::swap(heap[index], heap[largest]);
            heapifyDown(largest);
        }
    }

    void heapifyUp(int index) {
        int parent = (index - 1) / 2;
        if (index && heap[parent] < heap[index]) {
            std::swap(heap[index], heap[parent]);
            heapifyUp(parent);
        }
    }

public:
    void insert(int key) {
        heap.push_back(key);
        int index = heap.size() - 1;
        heapifyUp(index);
    }

    void deleteMax() {
        if (heap.empty()) {
            throw std::out_of_range("Heap is empty");
        }

        heap[0] = heap.back();
        heap.pop_back();
        heapifyDown(0);
    }

    int getMax() {
        if (heap.empty()) {
            throw std::out_of_range("Heap is empty");
        }
        return heap.front();
    }

    void printHeap() {
        for (int i : heap) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }
};

int main() {
    MaxHeap heap;
    heap.insert(3);
    heap.insert(2);
    heap.insert(15);
    heap.insert(5);
    heap.insert(4);
    heap.insert(45);

    std::cout << "Max-Heap array: ";
    heap.printHeap();

    std::cout << "The maximum element is " << heap.getMax() << std::endl;

    heap.deleteMax();
    std::cout << "After deleting the maximum element, the heap is: ";
    heap.printHeap();

    return 0;
}
```

**Magyarázat a Kódhoz:**

1. **Osztály Deklaráció**: A `MaxHeap` osztály egy privát adattagot tartalmaz, amely egy vektor, amely a heap elemeit tárolja.
2. **heapifyDown() és heapifyUp()**: Ezek a privát metódusok biztosítják a heap tulajdonságok fenntartását. A `heapifyDown()` a törlés művelet után használatos, míg a `heapifyUp()` a beszúrás művelet után.
3. **insert()**: Ez a metódus új elemet ad a heap-hez, majd a `heapifyUp()` segítségével helyreállítja a heap tulajdonságokat.
4. **deleteMax()**: Ez a metódus eltávolítja a legnagyobb elemet (a gyökeret) és a `heapifyDown()` segítségével helyreállítja a heap tulajdonságokat.
5. **getMax()**: Ez a metódus visszaadja a legnagyobb elemet a heap-ből.
6. **printHeap()**: Ez a metódus kiírja a heap elemeit.

### Hatékonyság

A halmok műveletei tömbös megvalósításban hatékonyak. A beszúrás és törlés műveletek időbeli komplexitása `O(log n)`, mivel a műveletek során legfeljebb a fa magasságának megfelelő számú lépést kell végrehajtani. A legkisebb vagy legnagyobb elem elérése pedig `O(1)` időben történik, mivel ezek az elemek mindig a tömb első helyén találhatók.

### Összegzés

A halmok tömbös megvalósítása egy hatékony és elegáns módja annak, hogy bináris fastruktúrákat kezeljünk. Mind a Min-Heap, mind a Max-Heap esetében a tömbös megvalósítás lehetővé teszi a gyors hozzáférést és a hatékony műveletek végrehajtását. A bemutatott példakódok C++ nyelven jól szemléltetik, hogyan valósíthatók meg ezek az adatszerkezetek és műveleteik a gyakorlatban.

## 4.3 Műveletek: beszúrás, törlés, heapify

A halmok (heaps) alapvető műveletei közé tartozik a beszúrás (insert), a törlés (delete) és a heapify. Ezek a műveletek biztosítják a halom tulajdonságainak fenntartását és az adatok hatékony kezelését. Ebben az alfejezetben részletesen bemutatjuk ezeket a műveleteket, a hozzájuk kapcsolódó algoritmusokat és példakódokat C++ nyelven.

### Beszúrás (Insert)

A beszúrás művelet célja egy új elem hozzáadása a halomhoz úgy, hogy a halom tulajdonságai érvényben maradjanak. A beszúrás során az új elemet először a halom végére helyezzük, majd felfelé haladva helyreállítjuk a halom tulajdonságokat a `heapifyUp` művelettel.

**Algoritmus:**
1. Helyezzük az új elemet

### Beszúrás (Insert)

A beszúrás művelet célja egy új elem hozzáadása a halomhoz úgy, hogy a halom tulajdonságai érvényben maradjanak. A beszúrás során az új elemet először a halom végére helyezzük, majd felfelé haladva helyreállítjuk a halom tulajdonságokat a `heapifyUp` művelettel.

**Algoritmus:**
1. Helyezzük az új elemet a halom végére.
2. Hasonlítsuk össze az új elemet a szülőjével.
3. Ha az új elem kisebb (Min-Heap esetén) vagy nagyobb (Max-Heap esetén) a szülőnél, cseréljük meg őket.
4. Ismételjük ezt a folyamatot, amíg a halom tulajdonságai helyre nem állnak, vagy el nem érjük a gyökércsomópontot.

**Példakód C++ nyelven:**

```cpp
#include <iostream>
#include <vector>
#include <stdexcept>

class MinHeap {
private:
    std::vector<int> heap;

    void heapifyUp(int index) {
        int parent = (index - 1) / 2;
        if (index && heap[parent] > heap[index]) {
            std::swap(heap[index], heap[parent]);
            heapifyUp(parent);
        }
    }

public:
    void insert(int key) {
        heap.push_back(key);
        int index = heap.size() - 1;
        heapifyUp(index);
    }

    void printHeap() {
        for (int i : heap) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }
};

int main() {
    MinHeap heap;
    heap.insert(3);
    heap.insert(2);
    heap.insert(15);
    heap.insert(5);
    heap.insert(4);
    heap.insert(45);

    std::cout << "Min-Heap array after insertions: ";
    heap.printHeap();

    return 0;
}
```

### Törlés (Delete)

A törlés művelet célja egy elem eltávolítása a halomból. A leggyakrabban a halom gyökerének (legkisebb elem a Min-Heap-ben vagy legnagyobb elem a Max-Heap-ben) törlése történik. A törlés után a halom tulajdonságait a `heapifyDown` művelettel állítjuk helyre.

**Algoritmus:**
1. Cseréljük ki a gyökércsomópontot a halom utolsó elemével.
2. Távolítsuk el az utolsó elemet.
3. Hasonlítsuk össze az új gyökércsomópontot a gyermekeivel.
4. Ha szükséges, cseréljük meg a gyökércsomópontot a kisebbik (Min-Heap) vagy nagyobbik (Max-Heap) gyermekével.
5. Ismételjük ezt a folyamatot, amíg a halom tulajdonságai helyre nem állnak, vagy el nem érjük a levélcsomópontot.

**Példakód C++ nyelven:**

```cpp
#include <iostream>
#include <vector>
#include <stdexcept>

class MinHeap {
private:
    std::vector<int> heap;

    void heapifyDown(int index) {
        int smallest = index;
        int leftChild = 2 * index + 1;
        int rightChild = 2 * index + 2;

        if (leftChild < heap.size() && heap[leftChild] < heap[smallest]) {
            smallest = leftChild;
        }

        if (rightChild < heap.size() && heap[rightChild] < heap[smallest]) {
            smallest = rightChild;
        }

        if (smallest != index) {
            std::swap(heap[index], heap[smallest]);
            heapifyDown(smallest);
        }
    }

public:
    void insert(int key) {
        heap.push_back(key);
        int index = heap.size() - 1;
        heapifyUp(index);
    }

    void deleteMin() {
        if (heap.empty()) {
            throw std::out_of_range("Heap is empty");
        }

        heap[0] = heap.back();
        heap.pop_back();
        heapifyDown(0);
    }

    int getMin() {
        if (heap.empty()) {
            throw std::out_of_range("Heap is empty");
        }
        return heap.front();
    }

    void printHeap() {
        for (int i : heap) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }
};

int main() {
    MinHeap heap;
    heap.insert(3);
    heap.insert(2);
    heap.insert(15);
    heap.insert(5);
    heap.insert(4);
    heap.insert(45);

    std::cout << "Min-Heap array: ";
    heap.printHeap();

    std::cout << "The minimum element is " << heap.getMin() << std::endl;

    heap.deleteMin();
    std::cout << "After deleting the minimum element, the heap is: ";
    heap.printHeap();

    return 0;
}
```

### Heapify

A heapify művelet célja egy csomópont és annak gyermekei közötti helyes halom tulajdonságok helyreállítása. Két változata létezik: `heapifyUp` és `heapifyDown`.

**heapifyUp**: Ez a művelet akkor szükséges, amikor egy új elemet adunk a halomhoz. A `heapifyUp` során az új elemet a megfelelő helyére mozgatjuk a fa gyökeréig, amíg a halom tulajdonságai érvényben maradnak.

**heapifyDown**: Ez a művelet akkor szükséges, amikor a gyökércsomópontot eltávolítjuk vagy egy elem törlése után a halom tulajdonságai megsérülnek. A `heapifyDown` során az adott csomópontot a megfelelő helyére mozgatjuk a fa leveleibe, amíg a halom tulajdonságai érvényben maradnak.

**Példakód C++ nyelven:**

```cpp
#include <iostream>
#include <vector>
#include <stdexcept>

class MinHeap {
private:
    std::vector<int> heap;

    void heapifyDown(int index) {
        int smallest = index;
        int leftChild = 2 * index + 1;
        int rightChild = 2 * index + 2;

        if (leftChild < heap.size() && heap[leftChild] < heap[smallest]) {
            smallest = leftChild;
        }

        if (rightChild < heap.size() && heap[rightChild] < heap[smallest]) {
            smallest = rightChild;
        }

        if (smallest != index) {
            std::swap(heap[index], heap[smallest]);
            heapifyDown(smallest);
        }
    }

    void heapifyUp(int index) {
        int parent = (index - 1) / 2;
        if (index && heap[parent] > heap[index]) {
            std::swap(heap[index], heap[parent]);
            heapifyUp(parent);
        }
    }

public:
    void insert(int key) {
        heap.push_back(key);
        int index = heap.size() - 1;
        heapifyUp(index);
    }

    void deleteMin() {
        if (heap.empty()) {
            throw std::out_of_range("Heap is empty");
        }

        heap[0] = heap.back();
        heap.pop_back();
        heapifyDown(0);
    }

    int getMin() {
        if (heap.empty()) {
            throw std::out_of_range("Heap is empty");
        }
        return heap.front();
    }

    void printHeap() {
        for (int i : heap) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }
};

int main() {
    MinHeap heap;
    heap.insert(3);
    heap.insert(2);
    heap.insert(15);
    heap.insert(5);
    heap.insert(4);
    heap.insert(45);

    std::cout << "Min-Heap array: ";
    heap.printHeap();

    std::cout << "The minimum element is " << heap.getMin() << std::endl;

    heap.deleteMin();
    std::cout << "After deleting the minimum element, the heap is: ";
    heap.printHeap();

    return 0;
}
```

**Magyarázat a Kódhoz:**

1. **Osztály Deklaráció**: A `MinHeap` osztály egy privát adattagot tartalmaz, amely egy vektor, amely a heap elemeit tárolja.
2. **heapifyDown() és heapifyUp()**: Ezek a privát metódusok biztosítják a heap tulajdonságok fenntartását. A `heapifyDown()` a törlés művelet után használatos, míg a `heapifyUp()` a beszúrás művelet után.
3. **insert()**: Ez a metódus új elemet ad a heap-hez, majd a `heapifyUp()` segítségével helyreállítja a heap tulajdonságokat.
4. **deleteMin()**: Ez a metódus eltávolítja a legkisebb elemet (a gyökeret) és a `heapifyDown()` segítségével helyreállítja a heap tulajdonságokat.
5. **getMin()**: Ez a metódus visszaadja a legkisebb elemet a heap-ből.
6. **printHeap()**: Ez a metódus kiírja a heap elemeit.

### Hatékonyság

A halom műveletek, mint a beszúrás és törlés, időbeli komplexitása `O(log n)`, mivel legfeljebb a fa magasságának megfelelő számú lépést kell végrehajtani. A `heapifyUp` és `heapifyDown` műveletek is hasonló időkomplexitással bírnak. A legkisebb vagy legnagyobb elem elérése `O(1)` időben történik, mivel ezek az elemek mindig a tömb első helyén találhatók.

### Összegzés

A halmok alapvető műveletei, mint a beszúrás, törlés és heapify, elengedhetetlenek a halom tulajdonságainak fenntartásához és az adatok hatékony kezeléséhez. A bemutatott példakódok C++ nyelven jól szemléltetik, hogyan valósíthatók meg ezek a műveletek a gyakorlatban, biztosítva a halom szerkezetének és tulajdonságainak megőrzését.

## 4.4 Heapsort algoritmus

A Heapsort algoritmus egy hatékony és stabil rendezési algoritmus, amely a halom (heap) adatszerkezet tulajdonságait használja ki. A Heapsort időbeli komplexitása `O(n log n)`, ami garantálja, hogy nagy mennyiségű adat rendezése esetén is hatékonyan működik. Ebben az alfejezetben részletesen bemutatjuk a Heapsort algoritmus működését, a szükséges műveleteket, és példakódot adunk C++ nyelven.

### Az algoritmus működése

A Heapsort két fő fázisból áll:
1. **Halomépítés (Heapify)**: Az eredeti tömbből egy Max-Heap vagy Min-Heap építése.


A Heapsort algoritmus egy hatékony és stabil rendezési algoritmus, amely a halom (heap) adatszerkezet tulajdonságait használja ki. A Heapsort időbeli komplexitása `O(n log n)`, ami garantálja, hogy nagy mennyiségű adat rendezése esetén is hatékonyan működik. Ebben az alfejezetben részletesen bemutatjuk a Heapsort algoritmus működését, a szükséges műveleteket, és példakódot adunk C++ nyelven.

### Az algoritmus működése

A Heapsort két fő fázisból áll:
1. **Halomépítés (Heapify)**: Az eredeti tömbből egy Max-Heap vagy Min-Heap építése.
2. **Rendezett sorozat előállítása**: A halomból egy rendezett sorozat kinyerése úgy, hogy az első elemet (a legnagyobb elemet a Max-Heap-ben vagy a legkisebb elemet a Min-Heap-ben) kivesszük, majd a halom tulajdonságokat helyreállítjuk.

**Max-Heap esetén a Heapsort lépései:**
1. Az eredeti tömbből Max-Heap-et építünk.
2. Az első elemet (a legnagyobb elemet) kicseréljük a tömb utolsó elemével, majd csökkentjük a halom méretét.
3. A halom tulajdonságait helyreállítjuk a `heapifyDown` művelettel.
4. Ismételjük a 2-3 lépéseket, amíg a halom mérete nagyobb, mint 1.

### Halomépítés (Heapify)

A halomépítés során az eredeti tömbből Max-Heap vagy Min-Heap épül. Ezt a `heapifyDown` művelettel érjük el, amely biztosítja, hogy minden csomópont megfeleljen a halom tulajdonságainak.

**Példakód C++ nyelven:**

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

// Segédfüggvény, amely biztosítja, hogy a csomópont megfeleljen a Max-Heap tulajdonságainak
void heapifyDown(std::vector<int>& heap, int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && heap[left] > heap[largest])
        largest = left;

    if (right < n && heap[right] > heap[largest])
        largest = right;

    if (largest != i) {
        std::swap(heap[i], heap[largest]);
        heapifyDown(heap, n, largest);
    }
}

// Heapsort algoritmus implementációja
void heapSort(std::vector<int>& arr) {
    int n = arr.size();

    // Max-Heap építése
    for (int i = n / 2 - 1; i >= 0; i--)
        heapifyDown(arr, n, i);

    // Rendezett sorozat előállítása
    for (int i = n - 1; i > 0; i--) {
        std::swap(arr[0], arr[i]);
        heapifyDown(arr, i, 0);
    }
}

// Tömb kiíratása
void printArray(const std::vector<int>& arr) {
    for (int i : arr)
        std::cout << i << " ";
    std::cout << std::endl;
}

int main() {
    std::vector<int> arr = {12, 11, 13, 5, 6, 7};

    std::cout << "Original array: ";
    printArray(arr);

    heapSort(arr);

    std::cout << "Sorted array: ";
    printArray(arr);

    return 0;
}
```

**Magyarázat a Kódhoz:**

1. **heapifyDown**: Ez a függvény biztosítja, hogy az adott csomópont és gyermekei megfeleljenek a Max-Heap tulajdonságainak. Ha egy csomópont értéke kisebb, mint bármelyik gyermeke, akkor megcseréljük őket, majd rekurzívan meghívjuk a `heapifyDown` függvényt az érintett gyermekre.
2. **heapSort**: A Heapsort algoritmus megvalósítása. Először Max-Heap-et építünk az eredeti tömbből, majd a halom legnagyobb elemét kicseréljük a tömb utolsó elemével, és csökkentjük a halom méretét. A halom tulajdonságait helyreállítjuk a `heapifyDown` művelettel, majd ismételjük a folyamatot, amíg az összes elem rendezetté nem válik.
3. **printArray**: Ez a segédfüggvény kiírja a tömb elemeit.

### Példa a Heapsort működésére

Vegyünk egy példatömböt: `{12, 11, 13, 5, 6, 7}`

1. **Max-Heap építése**:
    - A `heapifyDown` függvényt az utolsó nem levélcsomóponttól (n/2 - 1) kezdve alkalmazzuk minden csomópontra.
    - Az eredeti tömbből épített Max-Heap: `{13, 11, 12, 5, 6, 7}`

2. **Rendezett sorozat előállítása**:
    - Cseréljük az első elemet (13) az utolsóval (7), majd alkalmazzuk a `heapifyDown` függvényt a gyökércsomópontra.
    - Az első csere után: `{7, 11, 12, 5, 6, 13}`
    - Az új Max-Heap: `{12, 11, 7, 5, 6, 13}`
    - Ismételjük a cserét és a `heapifyDown` műveletet, amíg az összes elem rendezetté nem válik.

### Összegzés

A Heapsort algoritmus egy hatékony rendezési módszer, amely a halom adatszerkezet tulajdonságait használja ki. Az algoritmus két fő fázisban működik: először Max-Heap-et épít az eredeti tömbből, majd rendezett sorozatot állít elő a halom tulajdonságainak helyreállításával. A bemutatott példakód C++ nyelven jól szemlélteti az algoritmus működését, és biztosítja, hogy az adatok hatékonyan és stabilan rendeződjenek.

## 4.5 Prioritási sorok

A prioritási sorok (priority queues) olyan adatszerkezetek, amelyek lehetővé teszik az elemek prioritás szerinti kezelését. A prioritási sorok különösen hasznosak számos algoritmusban és alkalmazásban, mint például a grafalgoritmusokban, a feladatütemezésben, és a hálózati útvonalválasztásban. Ebben az alfejezetben részletesen bemutatjuk a prioritási sorok működését, a halmok segítségével történő megvalósításukat, és C++ példakódokkal illusztráljuk a gyakorlati használatukat.

### Prioritási sorok működése

A prioritási sorok két alapvető műveletet támogatnak:
1. **Insert (beszúrás)**: Új elem hozzáadása a prioritási sorhoz a megfelelő prioritással.
2. **Extract-Min / Extract-Max (kivétel)**: A legmagasabb prioritású elem eltávolítása és visszaadása a prioritási sorból.

A prioritási sorok két típusa:
- **Min-prioritási sor**: A legkisebb értékű elem rendelkezik a legmagasabb prioritással.
- **Max-prioritási sor**: A legnagyobb értékű elem rendelkezik a legmagasabb prioritással.

A halmok kiválóan alkalmasak prioritási sorok megvalósítására, mivel garantálják a hatékony beszúrást és kivételt.

### Prioritási sor megvalósítása halmokkal

A halmokkal megvalósított prioritási sorok lehetővé teszik a beszúrás és kivétel műveletek hatékony végrehajtását `O(log n)` időben, ahol `n` a prioritási sorban lévő elemek száma.

**Példakód C++ nyelven:**

```cpp
#include <iostream>
#include <vector>
#include <stdexcept>

class MinPriorityQueue {
private:
    std::vector<int> heap;

    void heapifyDown(int index) {
        int smallest = index;
        int leftChild = 2 * index + 1;
        int rightChild = 2 * index + 2;

        if (leftChild < heap.size() && heap[leftChild] < heap[smallest]) {
            smallest = leftChild;
        }

        if (rightChild < heap.size() && heap[rightChild] < heap[smallest]) {
            smallest = rightChild;
        }

        if (smallest != index) {
            std::swap(heap[index], heap[smallest]);
            heapifyDown(smallest);
        }
    }

    void heapifyUp(int index) {
        int parent = (index - 1) / 2;
        if (index && heap[parent] > heap[index]) {
            std::swap(heap[index], heap[parent]);
            heapifyUp(parent);
        }
    }

public:
    void insert(int key) {
        heap.push_back(key);
        int index = heap.size() - 1;
        heapifyUp(index);
    }

    int extractMin() {
        if (heap.empty()) {
            throw std::out_of_range("Priority queue is empty");
        }

        int root = heap.front();
        heap[0] = heap.back();
        heap.pop_back();
        heapifyDown(0);

        return root;
    }

    int getMin() const {
        if (heap.empty()) {
            throw std::out_of_range("Priority queue is empty");
        }
        return heap.front();
    }

    bool isEmpty() const {
        return heap.empty();
    }
};

int main() {
    MinPriorityQueue pq;
    pq.insert(3);
    pq.insert(2);
    pq.insert(15);
    pq.insert(5);
    pq.insert(4);
    pq.insert(45);

    std::cout << "Min element: " << pq.getMin() << std::endl;
    std::cout << "Extracted min element: " << pq.extractMin() << std::endl;
    std::cout << "Min element after extraction: " << pq.getMin() << std::endl;

    while (!pq.isEmpty()) {
        std::cout << pq.extractMin() << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

**Magyarázat a kódhoz:**

1. **Osztály deklaráció**: A `MinPriorityQueue` osztály egy vektort használ a heap tárolására, és privát metódusokat tartalmaz a `heapifyDown` és `heapifyUp` műveletekhez.
2. **heapifyDown és heapifyUp**: Ezek a metódusok biztosítják a halom tulajdonságok fenntartását a beszúrás és kivétel műveletek során.
3. **insert**: Ez a metódus új elemet ad a prioritási sorhoz, és a `heapifyUp` metódussal helyreállítja a halom tulajdonságokat.
4. **extractMin**: Ez a metódus eltávolítja és visszaadja a legkisebb elemet a prioritási sorból, majd a `heapifyDown` metódussal helyreállítja a halom tulajdonságokat.
5. **getMin**: Ez a metódus visszaadja a legkisebb elemet a prioritási sorból anélkül, hogy eltávolítaná azt.
6. **isEmpty**: Ez a metódus ellenőrzi, hogy a prioritási sor üres-e.
7. **main**: A `main` függvény bemutatja a prioritási sor használatát különböző műveleteken keresztül.

### Max-prioritási sor megvalósítása

A Max-prioritási sor megvalósítása hasonló a Min-prioritási soréhoz, de a `heapifyDown` és `heapifyUp` műveletek során a nagyobb-kisebb összehasonlításokat használjuk.

**Példakód C++ nyelven:**

```cpp
#include <iostream>
#include <vector>
#include <stdexcept>

class MaxPriorityQueue {
private:
    std::vector<int> heap;

    void heapifyDown(int index) {
        int largest = index;
        int leftChild = 2 * index + 1;
        int rightChild = 2 * index + 2;

        if (leftChild < heap.size() && heap[leftChild] > heap[largest]) {
            largest = leftChild;
        }

        if (rightChild < heap.size() && heap[rightChild] > heap[largest]) {
            largest = rightChild;
        }

        if (largest != index) {
            std::swap(heap[index], heap[largest]);
            heapifyDown(largest);
        }
    }

    void heapifyUp(int index) {
        int parent = (index - 1) / 2;
        if (index && heap[parent] < heap[index]) {
            std::swap(heap[index], heap[parent]);
            heapifyUp(parent);
        }
    }

public:
    void insert(int key) {
        heap.push_back(key);
        int index = heap.size() - 1;
        heapifyUp(index);
    }

    int extractMax() {
        if (heap.empty()) {
            throw std::out_of_range("Priority queue is empty");
        }

        int root = heap.front();
        heap[0] = heap.back();
        heap.pop_back();
        heapifyDown(0);

        return root;
    }

    int getMax() const {
        if (heap.empty()) {
            throw std::out_of_range("Priority queue is empty");
        }
        return heap.front();
    }

    bool isEmpty() const {
        return heap.empty();
    }
};

int main() {
    MaxPriorityQueue pq;
    pq.insert(3);
    pq.insert(2);
    pq.insert(15);
    pq.insert(5);
    pq.insert(4);
    pq.insert(45);

    std::cout << "Max element: " << pq.getMax() << std::endl;
    std::cout << "Extracted max element: " << pq.extractMax() << std::endl;
    std::cout << "Max element after extraction: " << pq.getMax() << std::endl;

    while (!pq.isEmpty()) {
        std::cout << pq.extractMax() << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

**Magyarázat a kódhoz:**

1. **Osztály deklaráció**: A `MaxPriorityQueue` osztály egy vektort használ a heap tárolására, és privát metódusokat tartalmaz a `heapifyDown` és `heapifyUp` műveletekhez.
2. **heapifyDown és heapifyUp**: Ezek a metódusok biztosítják a halom tulajdonságok fenntartását a beszúrás és kivétel műveletek során.
3. **insert**: Ez a metódus új elemet ad a prioritási sorhoz, és a `heapifyUp` metódussal helyreállítja a halom tulajdonságokat.
4. **extractMax**: Ez a metódus eltávolítja és visszaadja a legnagyobb elemet a prioritási sorból, majd a `heapifyDown` metódussal helyreállítja a halom tulajdonságokat.
5. **getMax**: Ez a metódus visszaadja a legnagyobb elemet a prioritási sorból anélkül, hogy eltávolítaná azt.
6. **isEmpty**: Ez a metódus ellenőrzi, hogy a prioritási sor üres-e.
7. **main**: A `main` függvény bemutatja a prioritási sor használatát különböző műveleteken keresztül.

### Hatékonyság

A prioritási sorok halmokkal történő megvalósítása hatékony, mivel a beszúrás és a kivétel műveletek mind `O(log n)` időkomplexitásúak. A halomépítés `O(n)` idő alatt elvégezhető, így nagy mennyiségű adat kezelése esetén is jól teljesítenek. A prioritási sorok különösen hasznosak olyan feladatoknál, ahol gyakran szükséges a legnagyobb vagy legkisebb elem gyors elérése és kezelése.

### Összegzés

A prioritási sorok kulcsfontosságú adatszerkezetek számos algoritmusban és alkalmazásban. A halmok segítségével történő megvalósításuk biztosítja a beszúrás és kivétel műveletek hatékonyságát, így ezek az adatszerkezetek széles körben alkalmazhatók. A bemutatott példakódok C++ nyelven jól szemléltetik a prioritási sorok gyakorlati használatát, és biztosítják, hogy a felhasználók könnyen implementálhassák és alkalmazhassák ezeket az adatszerkezeteket saját projektjeikben.


