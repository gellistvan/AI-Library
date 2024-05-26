\newpage

# II. Adatszerkezetek

\newpage 
# 1. Alapvető adatszerkezetek

Az adatszerkezetek a számítástechnika és programozás alapvető építőkövei, amelyek lehetővé teszik az adatok hatékony tárolását, kezelését és elérését. Az első fejezet célja, hogy bevezesse az olvasót a legfontosabb alapvető adatszerkezetek világába, amelyek az összetettebb adatszerkezetek és algoritmusok megértésének alapját képezik. Ebben a fejezetben megismerkedünk a tömbökkel, amelyek fix méretű, egymást követő memóriaterületeken tárolt elemekből állnak; a linked listákkal, amelyek dinamikus memóriafelhasználásukkal rugalmasabb adattárolást tesznek lehetővé; valamint a veremmel (stack) és a sorral (queue), amelyek speciális szabályok szerint hozzáférhető és kezelhető adatszerkezetek, gyakran használtak például algoritmusok végrehajtásának irányításában és adatok ideiglenes tárolásában. Ezek az alapvető adatszerkezetek nélkülözhetetlenek minden programozó eszköztárában, és megértésük elengedhetetlen a hatékony és optimalizált kód írásához.

## 1.1 Tömbök

A tömbök (angolul arrays) az egyik legegyszerűbb és leggyakrabban használt adatszerkezetek a programozásban. Egy tömb olyan adatszerkezet, amely azonos típusú elemek sorozatát tartalmazza, és az elemeket egymás után, egyetlen memóriaterületen tárolja. Ez a szerkezet különösen hasznos, amikor előre tudjuk, hogy pontosan hány elemet kell tárolnunk, és ezekhez az elemekhez gyorsan, hatékonyan szeretnénk hozzáférni.

### A tömbök szerkezete és működése

A tömbök legfőbb jellemzője, hogy fix méretűek, ami azt jelenti, hogy a létrehozásuk során meg kell adnunk a méretüket, és ez a méret nem változtatható meg később. Minden elem egyedi indexszel rendelkezik, amely alapján közvetlenül elérhetjük őket. Az indexelés általában 0-tól kezdődik, tehát az első elem indexe 0, a másodiké 1, és így tovább.

A következő példa C++ nyelven mutatja be egy egyszerű tömb deklarációját és inicializálását:

```cpp
#include <iostream>

int main() {
    // Egy 5 elemű tömb deklarációja és inicializálása
    int numbers[5] = {1, 2, 3, 4, 5};

    // Az elemek elérése és kiíratása
    for(int i = 0; i < 5; ++i) {
        std::cout << "Element at index " << i << " : " << numbers[i] << std::endl;
    }

    return 0;
}
```

Ebben a példában a `numbers` nevű tömb 5 egész számot tartalmaz. Az elemek eléréséhez egyszerűen használjuk az indexeket a `numbers[i]` szintaxis segítségével.

### Tömbök használati módjai

A tömbök számos feladatra használhatók, többek között:

1. **Adatok tárolása és rendezése**: A tömbök lehetővé teszik, hogy nagyszámú adatot tároljunk és rendezzünk. Számos rendezési algoritmus, mint például a buborékrendezés (bubble sort) vagy a gyorsrendezés (quick sort), hatékonyan működik tömbökkel.

2. **Keresés**: A tömbök lehetővé teszik az elemek gyors keresését. Például a lineáris keresés és a bináris keresés algoritmusai jól alkalmazhatók tömbökön.

3. **Matematikai és statisztikai műveletek**: Gyakran használjuk a tömböket matematikai és statisztikai számításokhoz, például átlag, medián, szórás stb. számítására.

4. **Adatok tárolása más adatszerkezetek számára**: A tömbök gyakran más adatszerkezetek, például verem (stack) és sor (queue) alapját képezik.

### Előnyök

A tömböknek számos előnyük van:

1. **Gyors hozzáférés**: Mivel az elemek egyfolytában, egymás után kerülnek tárolásra, az indexek segítségével közvetlenül elérhetjük őket. Az elérési idő konstans, O(1) időbonyolultságú.

2. **Egyszerűség**: A tömbök könnyen érthetők és használhatók, ami ideális választássá teszi őket a kezdő programozók számára.

3. **Alacsony memóriaigény**: Az elemek egymás utáni tárolása és az azonos típusú elemek tárolása miatt a tömbök memóriahasználata optimalizált.

### Hátrányok

Az előnyök mellett a tömböknek vannak hátrányaik is:

1. **Fix méret**: A tömbök méretét előre meg kell határozni, és később nem módosítható. Ha nem ismerjük előre a szükséges méretet, vagy ha a méret dinamikusan változik, a tömbök használata problémás lehet.

2. **Lassú beszúrás és törlés**: Az elemek beszúrása és törlése a tömb közepén lassú műveletek lehetnek, mivel a többi elemet el kell mozdítani. Ezek a műveletek O(n) időbonyolultságúak, ahol n a tömb mérete.

### Hatékonyság és időbonyolultság

Az alábbiakban összefoglaljuk a tömbök hatékonyságát különböző műveletek esetén:

- **Hozzáférés**: O(1) – Az elemekhez közvetlenül, index alapján férünk hozzá.
- **Keresés**: O(n) – Lineáris keresés esetén, a legrosszabb esetben minden elemet át kell nézni.
- **Beszúrás (általában a végén)**: O(1) – Ha a tömb nincs tele, az új elemet egyszerűen hozzáadjuk a végére.
- **Beszúrás (középen)**: O(n) – Az elemek elmozdítására van szükség.
- **Törlés (általában a végén)**: O(1) – Az utolsó elem eltávolítása gyors.
- **Törlés (középen)**: O(n) – Az elemek elmozdítására van szükség.

### Példák különböző műveletekre

Nézzünk néhány példát a fent említett műveletek végrehajtására C++ nyelven.

**Hozzáférés egy elemhez:**

```cpp
#include <iostream>

int main() {
    int numbers[5] = {10, 20, 30, 40, 50};

    // Hozzáférés a harmadik elemhez (index 2)
    int thirdElement = numbers[2];
    std::cout << "Third element: " << thirdElement << std::endl;

    return 0;
}
```

**Lineáris keresés:**

```cpp
#include <iostream>

int main() {
    int numbers[5] = {10, 20, 30, 40, 50};
    int target = 30;
    bool found = false;

    // Lineáris keresés
    for (int i = 0; i < 5; ++i) {
        if (numbers[i] == target) {
            found = true;
            std::cout << "Element found at index: " << i << std::endl;
            break;
        }
    }

    if (!found) {
        std::cout << "Element not found." << std::endl;
    }

    return 0;
}
```

**Beszúrás a végére (amennyiben a tömb nem tele):**

```cpp
#include <iostream>

int main() {
    int numbers[5] = {10, 20, 30, 40}; // Az ötödik elem nem inicializált
    int size = 4;

    // Új elem beszúrása a végére
    if (size < 5) {
        numbers[size] = 50;
        size++;
    }

    // A tömb elemeinek kiíratása
    for (int i = 0; i < size; ++i) {
        std::cout << numbers[i] << " ";
    }

    return 0;
}
```

**Elem törlése középről:**

```cpp
#include <iostream>

int main() {
    int numbers[5] = {10, 20, 30, 40, 50};
    int size = 5;
    int indexToDelete = 2; // A harmadik elem törlése

    // Az elem törlése
    for (int i = indexToDelete; i < size - 1; ++i) {
        numbers[i] = numbers[i + 1];
    }
    size--;

    // A tömb elemeinek kiíratása
    for (int i = 0; i < size; ++i) {
        std::cout << numbers[i] << " ";
    }

    return 0;
}
```

### Összefoglalás

A tömbök alapvető és nélkülözhetetlen eszközei a programozásnak, különösen akkor, amikor fix méretű és azonos típusú elemeket kell tárolnunk és gyorsan elérnünk. Bár vannak bizonyos korlátaik, mint például a fix méret és a lassú beszúrási és törlési műveletek, sok alkalmazásban ideálisak, és gyakran használják őket más adatszerkezetek alapjaként is. Az alapvető működésük és hatékonyságuk megértése kulcsfontosságú minden programozó számára.

## 1.2 Láncolt listák

A láncolt listák (angolul linked lists) az egyik legfontosabb és leggyakrabban használt dinamikus adatszerkezetek a programozásban. Ellentétben a tömbökkel, a láncolt listákban az elemek nem egymás után, egy összefüggő memóriaterületen helyezkednek el, hanem minden egyes elem (csomópont vagy node) egy mutatót tartalmaz, amely a következő elem helyét jelöli ki a memóriában. Ez a struktúra nagy rugalmasságot biztosít az adatok tárolásában és kezelésében.

### A láncolt listák szerkezete és működése

Egy láncolt lista alapvetően egy sor egymáshoz láncolt csomópontból áll, ahol minden csomópont két részből áll: az adatot tartalmazó mezőből és a következő csomópontra mutató mutatóból. Az első csomópontot hívjuk a lista fejének (head), az utolsó csomópont pedig nullára (nullptr) mutat, jelezve a lista végét.

A következő C++ kód egy egyszerű egyszeresen láncolt lista (singly linked list) létrehozását és néhány alapvető műveletet mutat be:

```cpp
#include <iostream>

// Egy láncolt lista csomópontjának definíciója
struct Node {
    int data;
    Node* next;
};

// Egy láncolt lista osztálya
class LinkedList {
public:
    LinkedList() : head(nullptr) {}

    // Új elem hozzáadása a lista elejéhez
    void insertAtBeginning(int value) {
        Node* newNode = new Node();
        newNode->data = value;
        newNode->next = head;
        head = newNode;
    }

    // Egy elem törlése a listából
    void deleteNode(int key) {
        Node* temp = head;
        Node* prev = nullptr;

        // Ha a fej csomópontot kell törölni
        if (temp != nullptr && temp->data == key) {
            head = temp->next; // A fej mutatóját átállítjuk
            delete temp;       // A régi fej csomópont felszabadítása
            return;
        }

        // Más csomópont keresése a listában
        while (temp != nullptr && temp->data != key) {
            prev = temp;
            temp = temp->next;
        }

        // Ha az elemet nem találtuk meg a listában
        if (temp == nullptr) return;

        // A csomópont törlése
        prev->next = temp->next;
        delete temp;
    }

    // A lista elemeinek kiíratása
    void printList() {
        Node* temp = head;
        while (temp != nullptr) {
            std::cout << temp->data << " -> ";
            temp = temp->next;
        }
        std::cout << "nullptr" << std::endl;
    }

private:
    Node* head;
};

int main() {
    LinkedList list;
    list.insertAtBeginning(10);
    list.insertAtBeginning(20);
    list.insertAtBeginning(30);

    std::cout << "Initial list: ";
    list.printList();

    list.deleteNode(20);

    std::cout << "After deleting 20: ";
    list.printList();

    return 0;
}
```

Ebben a példában a `LinkedList` osztály a láncolt lista alapvető műveleteit valósítja meg: új csomópont hozzáadása a lista elejéhez, csomópont törlése és a lista elemeinek kiíratása.

### Láncolt listák használati módjai1 

A láncolt listákat számos különböző helyzetben használhatjuk:

1. **Dinamikus memória kezelés**: A láncolt listák dinamikusan foglalják a memóriát, ami azt jelenti, hogy a lista mérete rugalmas és a futási idő alatt változhat. Ez különösen hasznos, ha nem tudjuk előre a szükséges elemek számát.

2. **Stack és Queue megvalósítás**: A láncolt listák könnyen alkalmazhatók verem (stack) és sor (queue) adatszerkezetek megvalósítására. A verem LIFO (Last In, First Out), a sor pedig FIFO (First In, First Out) elven működik.

3. **Gyakori beszúrás és törlés**: Ha gyakran kell elemeket beszúrni vagy törölni a lista közepén, a láncolt lista sokkal hatékonyabb lehet, mint a tömb, mivel a beszúrás és törlés csak a mutatók átállítását igényli.

### Előnyök

A láncolt listáknak számos előnyük van:

1. **Dinamikus méret**: A láncolt lista mérete rugalmasan változhat a futási idő alatt, ami előnyös, ha a szükséges elem szám előre nem ismert.

2. **Hatékony beszúrás és törlés**: Az elemek beszúrása és törlése gyors és hatékony lehet, különösen a lista elején vagy közepén, mivel csak a mutatók átállítását igényli, O(1) időbonyolultsággal.

3. **Memóriahasználat**: Az elemek csak akkor foglalnak memóriát, amikor ténylegesen szükség van rájuk, ami optimális memóriahasználatot eredményez.

### Hátrányok

A láncolt listáknak vannak hátrányaik is:

1. **Lassú hozzáférés**: A láncolt lista elemeinek elérése lassabb, mint a tömbé, mivel a csomópontokat sorban kell bejárni, O(n) időbonyolultsággal.

2. **Többlet memória**: Minden csomópont tartalmaz egy mutatót is az adatok mellett, ami extra memóriahasználatot jelent.

### Hatékonyság és időbonyolultság

Az alábbiakban összefoglaljuk a láncolt lista hatékonyságát különböző műveletek esetén:

- **Hozzáférés**: O(n) – Az elemekhez sorban, egymás után férünk hozzá.
- **Keresés**: O(n) – A legrosszabb esetben minden elemet át kell nézni.
- **Beszúrás (általában az elején)**: O(1) – Az új csomópont hozzáadása gyors.
- **Törlés**: O(n) – A megfelelő csomópont megtalálása időigényes lehet, de a törlés művelete maga O(1).

### Példák különböző műveletekre

Nézzünk néhány példát a fent említett műveletek végrehajtására C++ nyelven.

**Új csomópont beszúrása a lista elejére:**

```cpp
#include <iostream>

struct Node {
    int data;
    Node* next;
};

class LinkedList {
public:
    LinkedList() : head(nullptr) {}

    void insertAtBeginning(int value) {
        Node* newNode = new Node();
        newNode->data = value;
        newNode->next = head;
        head = newNode;
    }

    void printList() {
        Node* temp = head;
        while (temp != nullptr) {
            std::cout << temp->data << " -> ";
            temp = temp->next;
        }
        std::cout << "nullptr" << std::endl;
    }

private:
    Node* head;
};

int main() {
    LinkedList list;
    list.insertAtBeginning(10);
    list.insertAtBeginning(20);
    list.insertAtBeginning(30);

    list.printList();

    return 0;
}
```

**Egy csomópont törlése a listából:**

```cpp
#include <iostream>

struct Node {
    int data;
    Node* next;
};

class LinkedList {
public:
    LinkedList() : head(nullptr) {}

    void insertAtBeginning(int value) {
        Node* newNode = new Node();
        newNode->data = value;
        newNode->next = head;
        head = newNode;
    }

    void deleteNode(int key) {
        Node* temp = head;
        Node* prev = nullptr;

        if (temp != nullptr && temp->data == key) {
            head = temp->next;
            delete temp;
            return;
        }

        while (temp != nullptr && temp->data != key) {
            prev = temp;
            temp = temp->next;
        }

        if (temp == nullptr) return;

        prev->next = temp->next;
        delete temp;
    }

    void printList() {
        Node* temp = head;
        while (temp != nullptr) {
            std::cout << temp->data << " -> ";
            temp = temp->next;
        }
        std::cout << "nullptr" << std::endl;
    }

private:
    Node* head;
};

int main() {
    LinkedList list;
    list.insertAtBeginning(10);
    list.insertAtBeginning(20);
    list.insertAtBeginning(30);

    std::cout << "Initial list: ";
    list.printList();

    list.deleteNode(20);

    std::cout << "After deleting 20: ";
    list.printList();

    return 0;
}
```

### Összefoglalás

A láncolt listák rendkívül hasznos és sokoldalú adatszerkezetek, amelyek lehetővé teszik a dinamikus memóriafelhasználást és a rugalmas adatkezelést. Bár hozzáférésük lassabb, mint a tömböké, a beszúrás és törlés hatékonysága sok esetben előnyösebbé teszi őket, különösen akkor, ha az elemek számát nem ismerjük előre vagy gyakran változtatnunk kell a lista tartalmát. A láncolt listák alapvető megértése elengedhetetlen minden programozó számára, mivel ezek az adatszerkezetek számos komplexebb algoritmus és adatkezelési technika alapját képezik.

## 1.3 Verem (Stack)

A verem (angolul stack) egy olyan lineáris adatszerkezet, amely az elemeket a LIFO (Last In, First Out) elv alapján kezeli. Ez azt jelenti, hogy az utoljára beszúrt elem kerül először kivételre. A verem olyan adatszerkezet, amely számos számítástechnikai és programozási probléma megoldásában játszik kulcsszerepet, és gyakran használják például algoritmusok vezérlésében, visszalépési műveletek megvalósításában és kifejezések kiértékelésében.

### A verem szerkezete és működése

A verem két alapvető műveletet támogat:
- **Push**: Egy elem hozzáadása a verem tetejére.
- **Pop**: A legfelső elem eltávolítása a verem tetejéről.

Ezen kívül gyakran van egy **Top** művelet is, amely a verem tetején lévő elemet adja vissza anélkül, hogy eltávolítaná azt.

A következő példa C++ nyelven mutatja be egy egyszerű verem megvalósítását tömb segítségével:

```cpp
#include <iostream>
#include <stdexcept>

class Stack {
private:
    int* arr;
    int top;
    int capacity;

public:
    Stack(int size) {
        arr = new int[size];
        capacity = size;
        top = -1;
    }

    ~Stack() {
        delete[] arr;
    }

    // Elem hozzáadása a verem tetejére
    void push(int x) {
        if (isFull()) {
            throw std::overflow_error("Stack Overflow");
        }
        arr[++top] = x;
    }

    // Elem eltávolítása a verem tetejéről
    int pop() {
        if (isEmpty()) {
            throw std::underflow_error("Stack Underflow");
        }
        return arr[top--];
    }

    // A verem tetején lévő elem lekérdezése
    int peek() const {
        if (!isEmpty()) {
            return arr[top];
        } else {
            throw std::underflow_error("Stack is empty");
        }
    }

    // Ellenőrzés, hogy a verem üres-e
    bool isEmpty() const {
        return top == -1;
    }

    // Ellenőrzés, hogy a verem tele van-e
    bool isFull() const {
        return top == capacity - 1;
    }
};

int main() {
    Stack stack(5);

    stack.push(10);
    stack.push(20);
    stack.push(30);

    std::cout << "Top element is: " << stack.peek() << std::endl;

    std::cout << "Stack elements: ";
    while (!stack.isEmpty()) {
        std::cout << stack.pop() << " ";
    }

    std::cout << std::endl;

    return 0;
}
```

Ebben a példában a `Stack` osztály egy verem alapvető műveleteit valósítja meg: elemek beszúrása (`push`), elemek eltávolítása (`pop`), a verem tetején lévő elem lekérdezése (`peek`), valamint ellenőrzi, hogy a verem üres (`isEmpty`) vagy tele (`isFull`) van-e.

### A verem használati módjai

A verem számos felhasználási módja közül néhány:

1. **Függvényhívások kezelésése**: A programozási nyelvek gyakran használják a veremet a függvényhívások kezelésére. Minden függvényhíváskor a paraméterek és a lokális változók a verembe kerülnek, és a függvény befejeztével eltávolításra kerülnek onnan.

2. **Visszalépési műveletek (Backtracking)**: A verem segítségével könnyen megvalósíthatók visszalépési algoritmusok, például labirintus megoldása, királynék problémája stb.

3. **Kifejezések kiértékelése**: A verem gyakran használatos matematikai kifejezések értékeléséhez, különösen a postfix (fordított lengyel) jelölésben.

4. **Történet kezelés (History management)**: Böngészők és egyéb alkalmazások a verem segítségével kezelik a történetet, ahol az utolsó látogatott oldal kerül először visszahívásra.

### Előnyök

A verem használatának számos előnye van:

1. **Egyszerűség**: A verem működése egyszerű és könnyen érthető, ami megkönnyíti az implementációt és a használatot.

2. **Hatékonyság**: A verem műveletei (push, pop, peek) mind O(1) időbonyolultságúak, ami nagyon gyors adatkezelést tesz lehetővé.

3. **Rendezett adatkezelés**: A LIFO elv biztosítja, hogy az adatok rendezett módon kerüljenek feldolgozásra, ami különösen hasznos olyan feladatoknál, mint a függvényhívások kezelése.

### Hátrányok

A veremnek vannak korlátai is:

1. **Fix kapacitás**: Ha a verem tömb alapú megvalósítást használ, akkor a kapacitását előre meg kell határozni, ami memória pazarlást vagy túlcsordulást eredményezhet.

2. **Korlátozott hozzáférés**: A verem csak a legfelső elemhez biztosít közvetlen hozzáférést, így a közbenső elemek elérése nehézkes.

### Hatékonyság és időbonyolultság

Az alábbiakban összefoglaljuk a verem hatékonyságát különböző műveletek esetén:

- **Beszúrás (push)**: O(1) – Az új elem hozzáadása a verem tetejére gyors.
- **Eltávolítás (pop)**: O(1) – Az elem eltávolítása a verem tetejéről gyors.
- **Lekérdezés (peek)**: O(1) – A legfelső elem lekérdezése gyors.

### Példák különböző műveletekre

Nézzünk néhány további példát a verem különböző műveleteire C++ nyelven.

**Új elem hozzáadása a veremhez (push):**

```cpp
#include <iostream>
#include <stdexcept>

class Stack {
private:
    int* arr;
    int top;
    int capacity;

public:
    Stack(int size) {
        arr = new int[size];
        capacity = size;
        top = -1;
    }

    ~Stack() {
        delete[] arr;
    }

    void push(int x) {
        if (isFull()) {
            throw std::overflow_error("Stack Overflow");
        }
        arr[++top] = x;
    }

    bool isFull() const {
        return top == capacity - 1;
    }
};

int main() {
    Stack stack(5);
    stack.push(10);
    stack.push(20);
    stack.push(30);

    std::cout << "Elements pushed to stack." << std::endl;

    return 0;
}
```

**Elem eltávolítása a veremből (pop):**

```cpp
#include <iostream>
#include <stdexcept>

class Stack {
private:
    int* arr;
    int top;
    int capacity;

public:
    Stack(int size) {
        arr = new int[size];
        capacity = size;
        top = -1;
    }

    ~Stack() {
        delete[] arr;
    }

    void push(int x) {
        if (isFull()) {
            throw std::overflow_error("Stack Overflow");
        }
        arr[++top] = x;
    }

    int pop() {
        if (isEmpty()) {
            throw std::underflow_error("Stack Underflow");
        }
        return arr[top--];
    }

    bool isEmpty() const {
        return top == -1;
    }
};

int main() {
    Stack stack(5);
    stack.push(10);
    stack.push(20);
    stack.push(30);

    std::cout << "Popped element: " << stack.pop() << std::endl;

    return 0;
}
```

### Összefoglalás

A verem egy egyszerű, de hatékony adatszerkezet, amely számos algoritmus és programozási feladat megoldásában játszik kulcsszerepet. A LIFO elv alapján működő verem lehetővé teszi az adatok rendezett és gyors kezelését, különösen olyan helyzetekben, ahol az utolsóként hozzáadott elemeket kell először feldolgozni. A verem alapvető megértése elengedhetetlen minden programozó számára, mivel ez az adatszerkezet számos gyakorlati alkalmazás alapját képezi.

## 1.4 Sor (Queue)

A sor (angolul queue) egy olyan lineáris adatszerkezet, amely az elemeket a FIFO (First In, First Out) elv alapján kezeli. Ez azt jelenti, hogy az elsőként beszúrt elem kerül először kivételre, hasonlóan ahhoz, ahogyan egy sorban állás működik. A sorok számos alkalmazási területen használatosak, különösen akkor, amikor az adatok rendezett feldolgozására van szükség.

### A sor szerkezete és működése

A sor két alapvető műveletet támogat:
- **Enqueue**: Egy elem hozzáadása a sor végéhez.
- **Dequeue**: A legelső elem eltávolítása a sor elejéről.

Ezen kívül gyakran van egy **Front** művelet is, amely a sor elején lévő elemet adja vissza anélkül, hogy eltávolítaná azt, és egy **Rear** művelet, amely a sor végén lévő elemet adja vissza.

A következő példa C++ nyelven mutatja be egy egyszerű sor megvalósítását tömb segítségével:

```cpp
#include <iostream>
#include <stdexcept>

class Queue {
private:
    int* arr;
    int front;
    int rear;
    int capacity;
    int count;

public:
    Queue(int size) {
        arr = new int[size];
        capacity = size;
        front = 0;
        rear = -1;
        count = 0;
    }

    ~Queue() {
        delete[] arr;
    }

    // Elem hozzáadása a sor végéhez
    void enqueue(int x) {
        if (isFull()) {
            throw std::overflow_error("Queue Overflow");
        }
        rear = (rear + 1) % capacity;
        arr[rear] = x;
        count++;
    }

    // Elem eltávolítása a sor elejéről
    int dequeue() {
        if (isEmpty()) {
            throw std::underflow_error("Queue Underflow");
        }
        int item = arr[front];
        front = (front + 1) % capacity;
        count--;
        return item;
    }

    // A sor elején lévő elem lekérdezése
    int peek() const {
        if (isEmpty()) {
            throw std::underflow_error("Queue is empty");
        }
        return arr[front];
    }

    // Ellenőrzés, hogy a sor üres-e
    bool isEmpty() const {
        return count == 0;
    }

    // Ellenőrzés, hogy a sor tele van-e
    bool isFull() const {
        return count == capacity;
    }

    // A sor méretének lekérdezése
    int size() const {
        return count;
    }
};

int main() {
    Queue queue(5);

    queue.enqueue(10);
    queue.enqueue(20);
    queue.enqueue(30);
    queue.enqueue(40);

    std::cout << "Front element is: " << queue.peek() << std::endl;
    std::cout << "Queue size is " << queue.size() << std::endl;

    queue.dequeue();
    queue.dequeue();

    queue.enqueue(50);

    std::cout << "Queue elements: ";
    while (!queue.isEmpty()) {
        std::cout << queue.dequeue() << " ";
    }

    std::cout << std::endl;

    return 0;
}
```

Ebben a példában a `Queue` osztály a sor alapvető műveleteit valósítja meg: elemek hozzáadása (`enqueue`), elemek eltávolítása (`dequeue`), a sor elején lévő elem lekérdezése (`peek`), valamint ellenőrzi, hogy a sor üres (`isEmpty`) vagy tele (`isFull`) van-e, és lekérdezi a sor aktuális méretét (`size`).

### A sor használati módjai

A sor számos felhasználási módja közül néhány:

1. **Ütemezés (Scheduling)**: A sorokat gyakran használják ütemezőkben, ahol a folyamatok vagy feladatok időrendben kerülnek feldolgozásra. Például a CPU ütemező algoritmusok a feladatok sorban történő kezelésére épülnek.

2. **Adatfolyam-kezelés (Stream Processing)**: A sorok ideálisak adatfolyamok kezelésére, ahol az adatok érkezési sorrendben kerülnek feldolgozásra. Például a nyomtatási sorokban a dokumentumok nyomtatási sorrendjének kezelésére használják.

3. **Szélességi keresés (Breadth-First Search)**: A gráf algoritmusokban, mint például a szélességi keresés, a sorokat használják a csúcsok felfedezésére és kezelésére.

4. **Közlekedési rendszerek szimulációja**: A sorokat használják olyan rendszerek szimulációjára, ahol az entitások érkezési sorrendben kerülnek feldolgozásra, mint például a repülőgépek leszállása vagy az ügyfelek kiszolgálása egy pénztárnál.

### Előnyök

A sor használatának számos előnye van:

1. **Egyszerűség**: A sorok működése egyszerű és könnyen érthető, ami megkönnyíti az implementációt és a használatot.

2. **Rendezett adatkezelés**: A FIFO elv biztosítja, hogy az adatok érkezési sorrendben kerüljenek feldolgozásra, ami ideális számos ütemezési és adatfeldolgozási feladatnál.

3. **Hatékony adatfeldolgozás**: Az elemek hozzáadása és eltávolítása a sor végéről és elejéről gyors és hatékony műveletek.

### Hátrányok

A soroknak vannak korlátai is:

1. **Fix kapacitás**: Ha a sor tömb alapú megvalósítást használ, akkor a kapacitását előre meg kell határozni, ami memória pazarlást vagy túlcsordulást eredményezhet.

2. **Korlátozott hozzáférés**: A sor csak a legelső elemhez biztosít közvetlen hozzáférést, így a közbenső elemek elérése nehézkes.

### Hatékonyság és időbonyolultság

Az alábbiakban összefoglaljuk a sor hatékonyságát különböző műveletek esetén:

- **Beszúrás (enqueue)**: O(1) – Az új elem hozzáadása a sor végéhez gyors.
- **Eltávolítás (dequeue)**: O(1) – Az elem eltávolítása a sor elejéről gyors.
- **Lekérdezés (peek)**: O(1) – A legelső elem lekérdezése gyors.

### Példák különböző műveletekre

Nézzünk néhány további példát a sor különböző műveleteire C++ nyelven.

**Új elem hozzáadása a sorhoz (enqueue):**

```cpp
#include <iostream>
#include <stdexcept>

class Queue {
private:
    int* arr;
    int front;
    int rear;
    int capacity;
    int count;

public:
    Queue(int size) {
        arr = new int[size];
        capacity = size;
        front = 0;
        rear = -1;
        count = 0;
    }

    ~Queue() {
        delete[] arr;
    }

    void enqueue(int x) {
        if (isFull()) {
            throw std::overflow_error("Queue Overflow");
        }
        rear = (rear + 1) % capacity;
        arr[rear] = x;
        count++;
    }

    bool isFull() const {
        return count == capacity;
    }
};

int main() {
    Queue queue(5);
    queue.enqueue(10);
    queue.enqueue(20);
    queue.enqueue(30);

    std::cout << "Elements enqueued to queue." << std::endl;

    return 0;
}
```

**Elem eltávolítása a sorból (dequeue):**

```cpp
#include <iostream>
#include <stdexcept>

class Queue {
private:
    int* arr;
    int front;
    int rear;
    int capacity;
    int count;

public:
    Queue(int size) {
        arr = new int[size];
        capacity = size;
        front = 0;
        rear = -1;
        count = 0;
    }

    ~Queue() {
        delete[] arr;
    }

    void enqueue(int x) {
        if (isFull()) {
            throw std::overflow_error("Queue Overflow");
        }
        rear = (rear + 1) % capacity;
        arr[rear] = x;
        count++;
    }

    int dequeue() {
        if (isEmpty()) {
            throw std::underflow_error("Queue Underflow");
        }
        int item = arr[front];
        front = (front + 1) % capacity;
        count--;
        return item;


    }

    bool isEmpty() const {
        return count == 0;
    }
};

int main() {
    Queue queue(5);
    queue.enqueue(10);
    queue.enqueue(20);
    queue.enqueue(30);

    std::cout << "Dequeued element: " << queue.dequeue() << std::endl;

    return 0;
}
```

### Összefoglalás

A sor egy egyszerű, de hatékony adatszerkezet, amely számos algoritmus és programozási feladat megoldásában játszik kulcsszerepet. A FIFO elv alapján működő sor lehetővé teszi az adatok rendezett és gyors feldolgozását, különösen olyan helyzetekben, ahol az elsőként érkezett elemeket kell először feldolgozni. A sor alapvető megértése elengedhetetlen minden programozó számára, mivel ez az adatszerkezet számos gyakorlati alkalmazás alapját képezi.

