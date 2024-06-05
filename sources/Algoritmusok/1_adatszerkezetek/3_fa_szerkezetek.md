\newpage

## 3. Fa szerkezetek

A fa szerkezetek az adatszerkezetek egy különleges és rendkívül hasznos osztályát képviselik, amelyek számos alkalmazásban elengedhetetlenek. A fák olyan hierarchikus struktúrák, amelyek csomópontokból állnak, és a csomópontokat élek kötik össze, miközben nincs bennük kör. A fa adatszerkezetek különösen hatékonyak a keresési, rendezési és hierarchikus adatok tárolásában, és számos algoritmus alapját képezik. Ebben a fejezetben különböző típusú fákat fogunk megvizsgálni, kezdve az általános fa szerkezetekkel, majd rátérünk a specifikusabb bináris fákra, AVL fákra, piros-fekete fákra, valamint a B-fák és B±fák részletes tárgyalására. Minden egyes típust részletesen bemutatunk, megvizsgáljuk azok felépítését, működését és alkalmazási területeit, hogy átfogó képet kapjunk ezekről az alapvető adatszerkezetekről.

### 3.1. Általános fa szerkezetek

A fa szerkezetek az informatikában és a számítástudományban alapvető fontosságú adatstruktúrák, melyek hierarchikus kapcsolatrendszereket modelleznek. Ezek az adatszerkezetek számos alkalmazási területen megtalálhatók, beleértve a fájlrendszereket, a szintaktikai elemzést és a keresési algoritmusokat. Egy fa szerkezet csúcsokból (vagy csomópontokból) és élekből áll, ahol minden csúcs kapcsolódhat egy vagy több gyermek csúcshoz, míg egyetlen csúcsot gyökérként definiálunk, amelyből az egész fa szerkezete kiindul. A levél csúcsok olyan csomópontok, amelyeknek nincsenek gyermek csúcsai. Ebben a fejezetben megismerkedünk a fa szerkezetek alapfogalmaival és azok különböző reprezentációs módjaival, beleértve a szülő-gyermek, illetve a bal-jobb csomóponti megközelítéseket.

#### 3.1.1. Alapfogalmak (csúcsok, élek, gyökér, levél)

A fa szerkezetek alapfogalmai közé tartoznak a csúcsok (csomópontok), élek, gyökér és levelek. Ezek az elemek alkotják azokat az építőelemeket, amelyekből bármilyen fa struktúra felépíthető. Ebben az alfejezetben részletesen megvizsgáljuk ezeket az alapfogalmakat, hogy mélyebb megértést nyerjünk a fa adatszerkezetek működéséről és tulajdonságairól.

##### Csúcsok (Nodes)

A csúcsok a fa szerkezet legfontosabb építőkövei, amelyek információt hordoznak. Minden csúcs rendelkezik egy adattartalommal, valamint mutatókkal (pointerekkel), amelyek a fa további csúcsaira mutatnak. Egy csúcs adatmezője lehet bármilyen típusú adat, például egy egész szám, karakter vagy bonyolultabb adatstruktúra.

A fa szerkezet C++-ban történő implementálásakor a csúcsokat gyakran egy osztály segítségével definiáljuk. Az alábbiakban bemutatunk egy egyszerű csúcs definíciót:

```cpp
#include <iostream>
#include <vector>

// Csúcs osztály definíciója
class Node {
public:
    int data; // Adattartalom
    Node* left; // Bal gyerek csúcs mutatója
    Node* right; // Jobb gyerek csúcs mutatója

    // Konstruktor
    Node(int value) : data(value), left(nullptr), right(nullptr) {}
};
```

##### Élek (Edges)

Az élek a csúcsok közötti kapcsolatokat jelölik, amelyek meghatározzák a fa szerkezet hierarchiáját. Minden él egy irányított kapcsolatot képvisel két csúcs között: egy szülő és egy gyerek csúcs között. A fa szerkezetben nincsenek ciklusok, tehát az élek mindig egyirányúak és nincs visszatérés a gyökérhez.

C++-ban az élek implicit módon vannak definiálva a csúcsok mutatóin keresztül. Az előző példában a `left` és `right` mutatók olyan éleket képviselnek, amelyek a jelenlegi csúcsot összekötik a bal és jobb gyerek csúcsokkal.

##### Gyökér (Root)

A gyökér csúcs a fa legfelső szintjén található csúcs, amelyből az egész fa szerkezete kiindul. Minden fa szerkezetnek egyetlen gyökér csúcsa van, amely közvetlenül vagy közvetve összeköti az összes többi csúcsot.

A gyökér csúcsot általában egy külön mutatóval jelöljük a fa osztályban:

```cpp
class BinaryTree {
public:
    Node* root; // Gyökér csúcs mutatója

    // Konstruktor
    BinaryTree() : root(nullptr) {}

    // Gyökér csúcs létrehozása
    void createRoot(int value) {
        if (root == nullptr) {
            root = new Node(value);
        }
    }
};
```

##### Levél (Leaf)

A levél csúcsok azok a csúcsok, amelyeknek nincsenek gyerek csúcsai. Ezek a fa szerkezet legalsó szintjén helyezkednek el, és végelemeknek tekinthetők. A levelek fontosak a fa struktúrákban, mivel gyakran ezek az elemek hordozzák a végső adatokat, amelyeket keresünk vagy manipulálunk.

A levelek felismerhetők arról, hogy mind a bal, mind a jobb gyerek csúcs mutatójuk `nullptr`:

```cpp
bool isLeaf(Node* node) {
    return (node->left == nullptr && node->right == nullptr);
}
```

##### Összefoglaló

A fa szerkezetek alapfogalmai – a csúcsok, élek, gyökér és levelek – alapvető fontosságúak ahhoz, hogy megértsük a hierarchikus adatstruktúrák működését és felépítését. A csúcsok adattartalmat hordoznak és mutatókon keresztül kapcsolódnak egymáshoz, az élek meghatározzák a hierarchiát, a gyökér csúcs az egész fa kiindulópontja, míg a levél csúcsok a fa legalsó szintjén található végelemek.

A következőkben részletesen megvizsgáljuk, hogyan építhetünk és manipulálhatunk különböző típusú fa struktúrákat, és milyen algoritmusokat használhatunk ezek hatékony kezelésére. Az alábbi példakód egy egyszerű bináris fa létrehozását és alapvető műveleteit mutatja be C++ nyelven:

```cpp
int main() {
    // Bináris fa létrehozása
    BinaryTree tree;
    tree.createRoot(10);

    // Gyerek csúcsok hozzáadása
    tree.root->left = new Node(5);
    tree.root->right = new Node(20);

    // Ellenőrzés
    std::cout << "Gyökér csúcs: " << tree.root->data << std::endl;
    std::cout << "Bal gyerek csúcs: " << tree.root->left->data << std::endl;
    std::cout << "Jobb gyerek csúcs: " << tree.root->right->data << std::endl;

    // Levél ellenőrzés
    std::cout << "A bal gyerek csúcs levél: " << std::boolalpha << isLeaf(tree.root->left) << std::endl;
    std::cout << "A gyökér csúcs levél: " << std::boolalpha << isLeaf(tree.root) << std::endl;

    return 0;
}
```

Ez a példa bemutatja, hogyan hozhatunk létre egy egyszerű bináris fát, hogyan adhatunk hozzá gyerek csúcsokat, és hogyan ellenőrizhetjük, hogy egy csúcs levél-e. A következő alfejezetben részletesen tárgyaljuk a fa struktúrák különböző reprezentációs módjait és azok előnyeit, hátrányait.

#### 3.1.2 Reprezentáció (szülő-gyerek, bal-jobb csomópont)

A fa struktúrák reprezentációja kulcsfontosságú az adatstruktúrák tervezésében és megvalósításában. A fa adatszerkezetek különböző módon ábrázolhatók, amelyek közül a leggyakoribbak a szülő-gyerek (parent-child) és a bal-jobb (left-right) csomóponti megközelítések. Ezek a reprezentációs módok különböző előnyökkel és hátrányokkal rendelkeznek, és az adott alkalmazástól függően érdemes őket kiválasztani. Ebben az alfejezetben részletesen megvizsgáljuk ezeket a módszereket, és példakódokkal illusztráljuk őket C++ nyelven.

##### Szülő-gyerek Reprezentáció

A szülő-gyerek reprezentáció egy egyszerű és gyakori módszer a fa struktúrák ábrázolására. Ebben a megközelítésben minden csúcs tartalmaz egy mutatót a szülő csúcsára és mutatókat a gyerek csúcsokra. Ez a struktúra különösen hasznos olyan esetekben, amikor a csúcsok közötti kapcsolatokra és azok hierarchiájára van szükség.

###### Szülő-gyerek Reprezentáció Megvalósítása C++ Nyelven

Az alábbi példában bemutatjuk egy általános fa struktúra megvalósítását szülő-gyerek reprezentációval C++ nyelven:

```cpp
#include <iostream>
#include <vector>

class Node {
public:
    int data; // Adattartalom
    Node* parent; // Szülő csúcs mutatója
    std::vector<Node*> children; // Gyerek csúcsok mutatói

    // Konstruktor
    Node(int value, Node* parent = nullptr) : data(value), parent(parent) {}

    // Gyerek hozzáadása
    void addChild(int value) {
        Node* child = new Node(value, this);
        children.push_back(child);
    }

    // Szülő visszaadása
    Node* getParent() {
        return parent;
    }

    // Gyerekek visszaadása
    std::vector<Node*> getChildren() {
        return children;
    }
};

int main() {
    // Gyökér csúcs létrehozása
    Node* root = new Node(1);

    // Gyerek csúcsok hozzáadása
    root->addChild(2);
    root->addChild(3);

    // Második szintű gyerekek hozzáadása
    root->children[0]->addChild(4);
    root->children[0]->addChild(5);
    root->children[1]->addChild(6);

    // Gyökér csúcs adatainak kiírása
    std::cout << "Gyökér csúcs: " << root->data << std::endl;

    // Gyerekek adatainak kiírása
    for (Node* child : root->getChildren()) {
        std::cout << "Gyerek csúcs: " << child->data << std::endl;
    }

    // Szülő visszakeresése
    std::cout << "A 4-es csúcs szülője: " << root->children[0]->children[0]->getParent()->data << std::endl;

    return 0;
}
```

Ez a kód egy egyszerű fa struktúrát hoz létre, amelyben minden csúcs tartalmaz egy adattartalmat, egy mutatót a szülő csúcsára, valamint egy vektort, amely a gyerek csúcsokat tárolja. A szülő-gyerek kapcsolat könnyen kezelhető ezzel a struktúrával, és lehetővé teszi a fa hierarchiájának egyszerű megértését és kezelését.

##### Bal-jobb Csomóponti Reprezentáció

A bal-jobb csomóponti reprezentáció különösen hasznos a bináris fák esetében, ahol minden csúcs legfeljebb két gyerek csúccsal rendelkezhet: egy bal és egy jobb gyerekkel. Ez a reprezentáció egyszerűsíti a fa szerkezetet és hatékonyabbá teszi a műveleteket, mint például a keresést és a beszúrást.

###### Bal-jobb Csomóponti Reprezentáció Megvalósítása C++ Nyelven

Az alábbi példában bemutatjuk egy bináris fa megvalósítását bal-jobb csomóponti reprezentációval C++ nyelven:

```cpp
#include <iostream>

class BinaryTreeNode {
public:
    int data; // Adattartalom
    BinaryTreeNode* left; // Bal gyerek csúcs mutatója
    BinaryTreeNode* right; // Jobb gyerek csúcs mutatója

    // Konstruktor
    BinaryTreeNode(int value) : data(value), left(nullptr), right(nullptr) {}
};

class BinaryTree {
public:
    BinaryTreeNode* root; // Gyökér csúcs mutatója

    // Konstruktor
    BinaryTree() : root(nullptr) {}

    // Gyökér csúcs létrehozása
    void createRoot(int value) {
        if (root == nullptr) {
            root = new BinaryTreeNode(value);
        }
    }

    // Gyerek csúcsok hozzáadása
    void addLeftChild(BinaryTreeNode* parent, int value) {
        if (parent->left == nullptr) {
            parent->left = new BinaryTreeNode(value);
        }
    }

    void addRightChild(BinaryTreeNode* parent, int value) {
        if (parent->right == nullptr) {
            parent->right = new BinaryTreeNode(value);
        }
    }

    // Preorder bejárás
    void preorderTraversal(BinaryTreeNode* node) {
        if (node != nullptr) {
            std::cout << node->data << " ";
            preorderTraversal(node->left);
            preorderTraversal(node->right);
        }
    }
};

int main() {
    // Bináris fa létrehozása
    BinaryTree tree;
    tree.createRoot(1);

    // Gyerek csúcsok hozzáadása
    tree.addLeftChild(tree.root, 2);
    tree.addRightChild(tree.root, 3);
    tree.addLeftChild(tree.root->left, 4);
    tree.addRightChild(tree.root->left, 5);
    tree.addLeftChild(tree.root->right, 6);
    tree.addRightChild(tree.root->right, 7);

    // Preorder bejárás kiíratása
    std::cout << "Preorder bejárás: ";
    tree.preorderTraversal(tree.root);
    std::cout << std::endl;

    return 0;
}
```

Ebben a példában egy bináris fát hoztunk létre, amelyben minden csúcs tartalmaz egy adattartalmat, valamint mutatókat a bal és jobb gyerek csúcsokra. A `preorderTraversal` függvény segítségével bejárhatjuk a fa elemeit előrendben.

##### Előnyök és Hátrányok

Mindkét reprezentációs módszernek vannak előnyei és hátrányai. A szülő-gyerek reprezentáció általánosabb és rugalmasabb, mivel lehetővé teszi több gyerek csúcs kezelését, ugyanakkor bonyolultabb lehet a műveletek végrehajtása és az adatszerkezet kezelése. A bal-jobb csomóponti reprezentáció viszont egyszerűbb és hatékonyabb lehet a bináris fák esetében, de korlátozottabb, mivel csak két gyerek csúcsot tesz lehetővé.

##### Összefoglaló

A fa struktúrák különböző reprezentációi lehetőséget nyújtanak arra, hogy a különböző alkalmazási területeken hatékonyan és megfelelően használjuk őket. A szülő-gyerek és bal-jobb csomóponti megközelítések mindegyike saját előnyökkel és hátrányokkal rendelkezik, és a választás az adott alkalmazástól és a fa típusától függ. A következő fejezetekben részletesebben megvizsgáljuk a fa struktúrák különböző típusait és azok speciális alkalmazásait.


### 3.2. Bináris fák

A bináris fák az adatszerkezetek egyik alapvető és rendkívül sokoldalú formája, amelyeket széles körben alkalmaznak különböző informatikai és programozási problémák megoldására. Egy bináris fa egy gyökeres fa, ahol minden csomópont legfeljebb két gyermekkel rendelkezik, amelyeket gyakran bal és jobb gyermeknek neveznek. A bináris fák előnyei közé tartozik a hatékony keresés, beszúrás és törlés lehetősége, valamint a fa szerkezetének egyszerűsége, amely könnyen alkalmazható különböző algoritmusok megvalósítására. Ebben a fejezetben megismerkedünk a bináris keresőfákkal (Binary Search Trees, BST), részletesen bemutatjuk a leggyakoribb műveleteket, mint a beszúrás, törlés és keresés, valamint megvizsgáljuk a fa balanszírozásának fontosságát és módszereit.

#### 3.2.1. Bináris keresőfák (BST)

A bináris keresőfák (Binary Search Trees, BST) az adatszerkezetek egyik legfontosabb típusát képviselik, amelyeket gyakran használnak hatékony adatkeresési, rendezési és tárolási feladatok megoldására. A BST-k alapvető tulajdonsága, hogy minden csomópontban tárolt kulcs nagyobb, mint a bal gyermekében tárolt kulcsok és kisebb, mint a jobb gyermekében tárolt kulcsok. Ez a tulajdonság lehetővé teszi a gyors keresést, beszúrást és törlést, mivel minden művelet során csak egy alcsomópontot kell vizsgálni.

##### Definíció és alapfogalmak

Egy bináris keresőfa egy olyan bináris fa, amely a következő tulajdonságokkal rendelkezik:
- Minden csomópontban egy egyedi kulcs (key) található.
- A bal alcsomópontban lévő kulcsok kisebbek, mint az aktuális csomópont kulcsa.
- A jobb alcsomópontban lévő kulcsok nagyobbak, mint az aktuális csomópont kulcsa.

Ezeket a tulajdonságokat rekurzívan alkalmazva a fa minden csomópontjára, a BST-t hatékonyan lehet használni különböző műveletek végrehajtására.

##### Alapvető műveletek

A bináris keresőfák három alapvető művelete a keresés, a beszúrás és a törlés. Ezeket a műveleteket részletesen tárgyaljuk az alábbiakban.

###### Keresés

A keresés művelete egy adott kulcs megtalálására szolgál a fa struktúrájában. A keresés rekurzív vagy iteratív módon is megvalósítható. Az algoritmus a következőképpen működik:

1. Kezdjük a gyökércsomóponttal.
2. Ha a keresett kulcs egyenlő a gyökér kulcsával, akkor a kulcs megtalálva.
3. Ha a keresett kulcs kisebb, mint a gyökér kulcsa, folytassuk a keresést a bal alcsomópontban.
4. Ha a keresett kulcs nagyobb, mint a gyökér kulcsa, folytassuk a keresést a jobb alcsomópontban.
5. Ha az alcsomópont null, a keresett kulcs nincs a fában.

C++ példakód:

```cpp
struct Node {
    int key;
    Node* left;
    Node* right;

    Node(int value) : key(value), left(nullptr), right(nullptr) {}
};

Node* search(Node* root, int key) {
    if (root == nullptr || root->key == key) {
        return root;
    }

    if (key < root->key) {
        return search(root->left, key);
    }

    return search(root->right, key);
}
```

###### Beszúrás

A beszúrás művelete egy új kulcs hozzáadására szolgál a fához. Az algoritmus a kereséshez hasonlóan működik, de ha elérjük a megfelelő helyet, létrehozunk egy új csomópontot.

1. Kezdjük a gyökércsomóponttal.
2. Ha az új kulcs kisebb, mint az aktuális csomópont kulcsa, és a bal alcsomópont üres, beszúrjuk az új csomópontot balra.
3. Ha az új kulcs kisebb, mint az aktuális csomópont kulcsa, és a bal alcsomópont nem üres, folytassuk a beszúrást a bal alcsomópontban.
4. Ha az új kulcs nagyobb, mint az aktuális csomópont kulcsa, és a jobb alcsomópont üres, beszúrjuk az új csomópontot jobbra.
5. Ha az új kulcs nagyobb, mint az aktuális csomópont kulcsa, és a jobb alcsomópont nem üres, folytassuk a beszúrást a jobb alcsomópontban.

C++ példakód:

```cpp
Node* insert(Node* root, int key) {
    if (root == nullptr) {
        return new Node(key);
    }

    if (key < root->key) {
        root->left = insert(root->left, key);
    } else if (key > root->key) {
        root->right = insert(root->right, key);
    }

    return root;
}
```

###### Törlés

A törlés művelete egy adott kulcs eltávolítására szolgál a fából. A törlés némileg bonyolultabb, mivel három esetet kell kezelni:
1. A törlendő csomópontnak nincs gyermeke (levél csomópont).
2. A törlendő csomópontnak egy gyermeke van.
3. A törlendő csomópontnak két gyermeke van.

Az algoritmus a következő lépéseket követi:
1. Keressük meg a törlendő csomópontot.
2. Ha a csomópontnak nincs gyermeke, egyszerűen eltávolítjuk.
3. Ha a csomópontnak egy gyermeke van, kicseréljük a csomópontot a gyermekével.
4. Ha a csomópontnak két gyermeke van, megkeressük a csomópont utódját (a jobb alcsomópont legkisebb elemét), és helyettesítjük a csomópontot az utóddal, majd töröljük az utódot.

C++ példakód:

```cpp
Node* findMin(Node* node) {
    while (node->left != nullptr) {
        node = node->left;
    }
    return node;
}

Node* deleteNode(Node* root, int key) {
    if (root == nullptr) {
        return root;
    }

    if (key < root->key) {
        root->left = deleteNode(root->left, key);
    } else if (key > root->key) {
        root->right = deleteNode(root->right, key);
    } else {
        // Node with only one child or no child
        if (root->left == nullptr) {
            Node* temp = root->right;
            delete root;
            return temp;
        } else if (root->right == nullptr) {
            Node* temp = root->left;
            delete root;
            return temp;
        }

        // Node with two children: Get the inorder successor (smallest in the right subtree)
        Node* temp = findMin(root->right);

        // Copy the inorder successor's content to this node
        root->key = temp->key;

        // Delete the inorder successor
        root->right = deleteNode(root->right, temp->key);
    }

    return root;
}
```

##### Balanszírozás

A bináris keresőfák egyik kihívása, hogy a beszúrások és törlések hatására kiegyensúlyozatlanná válhatnak, ami a keresési műveletek hatékonyságát jelentősen csökkentheti. A balanszírozás célja, hogy a fa magasságát minimálisra csökkentse, és ezáltal fenntartsa a hatékony működést.

Számos balanszírozási módszer létezik, mint például az AVL fák, a piros-fekete fák, és a Splay fák. Ezek a módszerek különböző technikákat alkalmaznak a fa magasságának minimalizálására és az egyensúly fenntartására a beszúrási és törlési műveletek során.

###### AVL fák

Az AVL fa egy olyan bináris keresőfa, amelyben minden csomópontban tároljuk az alcsomópontok magasságát, és biztosítjuk, hogy a bal és jobb alcsomópontok magasságának különbsége legfeljebb 1 legyen. Ha ez a különbség meghaladja az 1-et, a fa újra balanszírozódik forgatások (rotációk) segítségével.

C++ példakód az AVL fa beszúrásához és balanszírozásához:

```cpp
struct Node {
    int key;
    Node* left;
    Node* right;
    int height;

    Node(int value) : key(value), left(nullptr), right(nullptr), height(1) {}
};

int height(Node* node) {
    if (node == nullptr) {
        return 0;
    }
    return node->height;
}

int getBalance(Node* node) {
    if (node == nullptr) {
        return 0;
    }
    return height(node->left) - height(node->right);
}

Node* rightRotate(Node* y) {
    Node* x = y->left;
    Node* T2 = x->right;

    x->right = y;
    y->left = T2;

    y->height = std::max(height(y->left), height(y->right)) + 1;
    x->height = std::max(height(x->left), height(x->right)) + 1;

    return x;
}

Node* leftRotate(Node* x) {
    Node* y = x->right;
    Node* T2 = y->left;

    y->left = x;
    x->right = T2;

    x->height = std::max(height(x->left), height(x->right)) + 1;
    y->height = std::max(height(y->left), height(y->right)) + 1;

    return y;
}

Node* insert(Node* node, int key) {
    if (node == nullptr) {
        return new Node(key);
    }

    if (key < node->key) {
        node->left = insert(node->left, key);
    } else if (key > node->key) {
        node->right = insert(node->right, key);
    } else {
        return node;
    }

    node->height = 1 + std::max(height(node->left), height(node->right));

    int balance = getBalance(node);

    if (balance > 1 && key < node->left->key) {
        return rightRotate(node);
    }

    if (balance < -1 && key > node->right->key) {
        return leftRotate(node);
    }

    if (balance > 1 && key > node->left->key) {
        node->left = leftRotate(node->left);
        return rightRotate(node);
    }

    if (balance < -1 && key < node->right->key) {
        node->right = rightRotate(node->right);
        return leftRotate(node);
    }

    return node;
}
```

Az AVL fa biztosítja, hogy a fa magassága mindig O(log n) maradjon, ahol n a csomópontok száma, így a keresési, beszúrási és törlési műveletek is hatékonyak maradnak.

##### Összefoglalás

A bináris keresőfák alapvető adatszerkezetek, amelyek hatékonyan támogatják a keresési, beszúrási és törlési műveleteket. Azonban az egyensúly fenntartása érdekében különböző balanszírozási módszereket kell alkalmazni, mint például az AVL fák, amelyek biztosítják a fa optimális magasságát. Ezek a módszerek és technikák elengedhetetlenek a modern adatfeldolgozó rendszerekben és algoritmusokban, így alapos ismeretük és megértésük kulcsfontosságú a hatékony programozási megoldások kidolgozásához.

#### 3.2.2. Műveletek: beszúrás, törlés, keresés

A bináris keresőfák (Binary Search Trees, BST) használata során három alapvető műveletet kell alaposan megértenünk: a beszúrás, a törlés és a keresés. Ezek a műveletek képezik a BST-k alapját, és hatékony végrehajtásuk elengedhetetlen a fa optimális működéséhez. Ebben az alfejezetben részletesen bemutatjuk ezen műveletek működését, algoritmusait és implementációját C++ nyelven.

##### Beszúrás

A beszúrás művelete új elem hozzáadását jelenti a bináris keresőfához. Az algoritmus célja, hogy megtalálja az új elem megfelelő helyét a fa struktúrájában, miközben megőrzi a bináris keresőfa tulajdonságait.

A beszúrás algoritmusa:
1. Kezdjük a gyökércsomóponttal.
2. Ha a fa üres, az új elem lesz a gyökér.
3. Hasonlítsuk össze az új elem kulcsát a jelenlegi csomópont kulcsával.
    - Ha az új kulcs kisebb, mint a jelenlegi csomópont kulcsa, folytassuk a beszúrást a bal alcsomópontban.
    - Ha az új kulcs nagyobb, mint a jelenlegi csomópont kulcsa, folytassuk a beszúrást a jobb alcsomópontban.
4. Ha elérünk egy null értékű alcsomóponthoz, hozzuk létre az új csomópontot, és helyezzük el ott az új elemet.

Az alábbi C++ kód bemutatja a beszúrás művelet megvalósítását:

```cpp
struct Node {
    int key;
    Node* left;
    Node* right;

    Node(int value) : key(value), left(nullptr), right(nullptr) {}
};

Node* insert(Node* root, int key) {
    if (root == nullptr) {
        return new Node(key);
    }

    if (key < root->key) {
        root->left = insert(root->left, key);
    } else if (key > root->key) {
        root->right = insert(root->right, key);
    }

    return root;
}
```

##### Törlés

A törlés művelete egy adott elem eltávolítását jelenti a bináris keresőfából. A törlés bonyolultabb lehet, mivel különböző eseteket kell kezelni attól függően, hogy a törlendő csomópontnak hány gyermeke van.

A törlés algoritmusa három fő esetet kezel:
1. **Levél csomópont (nincs gyermek):** Egyszerűen eltávolítjuk a csomópontot.
2. **Egy gyermek:** Helyettesítjük a csomópontot a gyermekével.
3. **Két gyermek:** Megkeressük az utódot (inorder successor), amely a jobb alcsomópont legkisebb eleme, helyettesítjük vele a törlendő csomópontot, majd eltávolítjuk az utódot.

Az alábbi C++ kód bemutatja a törlés művelet megvalósítását:

```cpp
Node* findMin(Node* node) {
    while (node->left != nullptr) {
        node = node->left;
    }
    return node;
}

Node* deleteNode(Node* root, int key) {
    if (root == nullptr) {
        return root;
    }

    if (key < root->key) {
        root->left = deleteNode(root->left, key);
    } else if (key > root->key) {
        root->right = deleteNode(root->right, key);
    } else {
        if (root->left == nullptr) {
            Node* temp = root->right;
            delete root;
            return temp;
        } else if (root->right == nullptr) {
            Node* temp = root->left;
            delete root;
            return temp;
        }

        Node* temp = findMin(root->right);
        root->key = temp->key;
        root->right = deleteNode(root->right, temp->key);
    }

    return root;
}
```

##### Keresés

A keresés művelete egy adott kulcs megtalálását jelenti a bináris keresőfában. A keresési algoritmus kihasználja a BST tulajdonságait, hogy hatékonyan, O(log n) idő alatt megtalálja a kulcsot, ahol n a csomópontok száma.

A keresés algoritmusa:
1. Kezdjük a gyökércsomóponttal.
2. Ha a keresett kulcs megegyezik a jelenlegi csomópont kulcsával, a kulcs megtalálva.
3. Ha a keresett kulcs kisebb, folytassuk a keresést a bal alcsomópontban.
4. Ha a keresett kulcs nagyobb, folytassuk a keresést a jobb alcsomópontban.
5. Ha az alcsomópont null értékű, a kulcs nincs a fában.

Az alábbi C++ kód bemutatja a keresés művelet megvalósítását:

```cpp
Node* search(Node* root, int key) {
    if (root == nullptr || root->key == key) {
        return root;
    }

    if (key < root->key) {
        return search(root->left, key);
    }

    return search(root->right, key);
}
```

##### Az algoritmusok időkomplexitása

A fenti műveletek időkomplexitása a fa magasságától függ:
- **Keresés:** Az időkomplexitás a legrosszabb esetben O(h), ahol h a fa magassága. Egy kiegyensúlyozott fa esetén ez O(log n), ahol n a csomópontok száma.
- **Beszúrás:** Az időkomplexitás a legrosszabb esetben O(h). Egy kiegyensúlyozott fa esetén ez O(log n).
- **Törlés:** Az időkomplexitás a legrosszabb esetben O(h). Egy kiegyensúlyozott fa esetén ez O(log n).

##### AVL fák: Beszúrás és törlés balanszírozása

Az AVL fák esetében a beszúrás és törlés műveletek során szükséges a fa balanszírozása, hogy biztosítsuk az O(log n) időkomplexitást. Az AVL fa egyensúlyi állapotát úgy tartjuk fenn, hogy minden csomópontnál biztosítjuk, hogy a bal és jobb alcsomópontok magasságának különbsége legfeljebb 1 legyen. Ha a különbség meghaladja az 1-et, a fa forgatások (rotációk) segítségével újra balanszírozódik.

Az AVL fa beszúrás algoritmusa:

```cpp
struct Node {
    int key;
    Node* left;
    Node* right;
    int height;

    Node(int value) : key(value), left(nullptr), right(nullptr), height(1) {}
};

int height(Node* node) {
    if (node == nullptr) {
        return 0;
    }
    return node->height;
}

int getBalance(Node* node) {
    if (node == nullptr) {
        return 0;
    }
    return height(node->left) - height(node->right);
}

Node* rightRotate(Node* y) {
    Node* x = y->left;
    Node* T2 = x->right;

    x->right = y;
    y->left = T2;

    y->height = std::max(height(y->left), height(y->right)) + 1;
    x->height = std::max(height(x->left), height(x->right)) + 1;

    return x;
}

Node* leftRotate(Node* x) {
    Node* y = x->right;
    Node* T2 = y->left;

    y->left = x;
    x->right = T2;

    x->height = std::max(height(x->left), height(x->right)) + 1;
    y->height = std::max(height(y->left), height(y->right)) + 1;

    return y;
}

Node* insert(Node* node, int key) {
    if (node == nullptr) {
        return new Node(key);
    }

    if (key < node->key) {
        node->left = insert(node->left, key);
    } else if (key > node->key) {
        node->right = insert(node->right, key);
    } else {
        return node;
    }

    node->height = 1 + std::max(height(node->left), height(node->right));

    int balance = getBalance(node);

    if (balance > 1 && key < node->left->key) {
        return rightRotate(node);
    }

    if (balance < -1 && key > node->right->key) {
        return leftRotate(node);
    }

    if (balance > 1 && key > node->left->key) {
        node->left = leftRotate(node->left);
        return rightRotate(node);
    }

    if (balance < -1 && key < node->right->key) {
        node->right = rightRotate(node->right);
        return leftRotate(node);
    }

    return node;
}
```

A fentiek alapján láthatjuk, hogy a bináris keresőfák műveletei alapvető fontosságúak az adatszerkezetek megértésében és hatékony alkalmazásában. A műveletek megfelelő megértése és implementációja lehetővé teszi a bináris keresőfák hatékony használatát különböző informatikai és programozási problémák megoldásában.

#### 3.2.3. Balanszírozás

A bináris keresőfák (Binary Search Trees, BST) hatékonysága nagymértékben függ a fa balanszírozottságától. Ha a fa kiegyensúlyozott, akkor a keresési, beszúrási és törlési műveletek időkomplexitása O(log n), ahol n a csomópontok száma. Azonban ha a fa kiegyensúlyozatlan, a legrosszabb esetben a műveletek időkomplexitása O(n) is lehet, ami jelentősen csökkenti a hatékonyságot. Ebben az alfejezetben a balanszírozás fontosságát, különböző módszereit és az ezekhez kapcsolódó algoritmusokat tárgyaljuk részletesen.

##### Balanszírozás fontossága

A balanszírozás célja, hogy minimalizáljuk a fa magasságát, ezáltal fenntartva a gyors adatkeresési és manipulációs műveletek lehetőségét. A kiegyensúlyozott fa lehetővé teszi, hogy minden út a gyökértől a levelekig hasonló hosszúságú legyen, ami garantálja a logaritmikus időkomplexitást.

##### AVL fák

Az AVL fa egy önbalanszírozó bináris keresőfa, amelyben minden csomópontban tároljuk a bal és jobb alcsomópontok magasságát, és biztosítjuk, hogy a magasságkülönbség legfeljebb 1 legyen. Ha ez a különbség meghaladja az 1-et, a fát újra balanszírozni kell forgatások segítségével.

###### AVL fa definíciója

Az AVL fa minden csomópontja tartalmaz egy kulcsot és két gyermek mutatót (bal és jobb), valamint egy magasságot:

```cpp
struct Node {
    int key;
    Node* left;
    Node* right;
    int height;

    Node(int value) : key(value), left(nullptr), right(nullptr), height(1) {}
};
```

###### Magasság és balansz faktor

Az AVL fa balanszolásához szükség van a csomópontok magasságának és balansz faktorának meghatározására.

```cpp
int height(Node* node) {
    if (node == nullptr) {
        return 0;
    }
    return node->height;
}

int getBalance(Node* node) {
    if (node == nullptr) {
        return 0;
    }
    return height(node->left) - height(node->right);
}
```

###### Forgatások (Rotations)

Az AVL fák balanszírozásához négyféle forgatást alkalmazunk:
- Jobbra forgatás (Right Rotation)
- Balra forgatás (Left Rotation)
- Bal-jobb forgatás (Left-Right Rotation)
- Jobb-bal forgatás (Right-Left Rotation)

**Jobbra forgatás (Right Rotation):**

```cpp
Node* rightRotate(Node* y) {
    Node* x = y->left;
    Node* T2 = x->right;

    x->right = y;
    y->left = T2;

    y->height = std::max(height(y->left), height(y->right)) + 1;
    x->height = std::max(height(x->left), height(x->right)) + 1;

    return x;
}
```

**Balra forgatás (Left Rotation):**

```cpp
Node* leftRotate(Node* x) {
    Node* y = x->right;
    Node* T2 = y->left;

    y->left = x;
    x->right = T2;

    x->height = std::max(height(x->left), height(x->right)) + 1;
    y->height = std::max(height(y->left), height(y->right)) + 1;

    return y;
}
```

**Bal-jobb forgatás (Left-Right Rotation):**

```cpp
Node* leftRightRotate(Node* node) {
    node->left = leftRotate(node->left);
    return rightRotate(node);
}
```

**Jobb-bal forgatás (Right-Left Rotation):**

```cpp
Node* rightLeftRotate(Node* node) {
    node->right = rightRotate(node->right);
    return leftRotate(node);
}
```

###### Beszúrás és balanszírozás

Az AVL fa beszúrása során, ha az egyensúly megbomlik, a megfelelő forgatási műveletet alkalmazzuk, hogy újra balanszírozzuk a fát.

```cpp
Node* insert(Node* node, int key) {
    if (node == nullptr) {
        return new Node(key);
    }

    if (key < node->key) {
        node->left = insert(node->left, key);
    } else if (key > node->key) {
        node->right = insert(node->right, key);
    } else {
        return node;
    }

    node->height = 1 + std::max(height(node->left), height(node->right));

    int balance = getBalance(node);

    if (balance > 1 && key < node->left->key) {
        return rightRotate(node);
    }

    if (balance < -1 && key > node->right->key) {
        return leftRotate(node);
    }

    if (balance > 1 && key > node->left->key) {
        node->left = leftRotate(node->left);
        return rightRotate(node);
    }

    if (balance < -1 && key < node->right->key) {
        node->right = rightRotate(node->right);
        return leftRotate(node);
    }

    return node;
}
```

###### Törlés és balanszírozás

A törlés művelete során is szükség lehet a fa balanszírozására. Az AVL fa törlési algoritmusa hasonló a beszúrási algoritmushoz, azonban a törlés után is ellenőrizzük a balansz faktort, és szükség esetén alkalmazzuk a megfelelő forgatási műveletet.

```cpp
Node* deleteNode(Node* root, int key) {
    if (root == nullptr) {
        return root;
    }

    if (key < root->key) {
        root->left = deleteNode(root->left, key);
    } else if (key > root->key) {
        root->right = deleteNode(root->right, key);
    } else {
        if (root->left == nullptr || root->right == nullptr) {
            Node* temp = root->left ? root->left : root->right;

            if (temp == nullptr) {
                temp = root;
                root = nullptr;
            } else {
                *root = *temp;
            }
            delete temp;
        } else {
            Node* temp = findMin(root->right);
            root->key = temp->key;
            root->right = deleteNode(root->right, temp->key);
        }
    }

    if (root == nullptr) {
        return root;
    }

    root->height = 1 + std::max(height(root->left), height(root->right));

    int balance = getBalance(root);

    if (balance > 1 && getBalance(root->left) >= 0) {
        return rightRotate(root);
    }

    if (balance > 1 && getBalance(root->left) < 0) {
        root->left = leftRotate(root->left);
        return rightRotate(root);
    }

    if (balance < -1 && getBalance(root->right) <= 0) {
        return leftRotate(root);
    }

    if (balance < -1 && getBalance(root->right) > 0) {
        root->right = rightRotate(root->right);
        return leftRotate(root);
    }

    return root;
}
```

##### Piros-fekete fák

A piros-fekete fa egy másik önbalanszírozó bináris keresőfa, amely biztosítja, hogy a fa magassága mindig O(log n) maradjon. A piros-fekete fa különleges szabályokkal rendelkezik a csomópontok színezésére vonatkozóan, amelyek segítenek fenntartani az egyensúlyt.

###### Piros-fekete fa tulajdonságai

1. Minden csomópont piros vagy fekete.
2. A gyökér mindig fekete.
3. Minden levél (null csomópont) fekete.
4. Ha egy csomópont piros, akkor mindkét gyermeke fekete (nem lehet két egymást követő piros csomópont).
5. Minden útnak egy csomóponttól bármelyik leszármazott levélig ugyanannyi fekete csomópontja van.

###### Beszúrás és balanszírozás

A piros-fekete fa beszúrása során a beszúrt csomópont piros lesz. Az egyensúly fenntartása érdekében az új csomópont beszúrása után alkalmazni kell a megfelelő átszínezési és forgatási műveleteket.

```cpp
enum Color { RED, BLACK };

struct Node {
    int key;
    Node* left;
    Node* right;
    Node* parent;
    Color color;

    Node(int value) : key(value), left(nullptr), right(nullptr), parent(nullptr), color(RED) {}
};

Node* insert(Node* root, Node* pt) {
    if (root == nullptr) {
        return pt;
    }

    if (pt->key < root->key) {
        root->left = insert(root->left, pt);
        root->left->parent = root;
    } else if (pt->key > root->key) {
        root->right = insert(root->right, pt);
        root->right->parent = root;
    }

    return root;
}

void rotateLeft(Node*& root, Node*& pt) {
    Node* pt_right = pt->right;

    pt->right = pt_right->left;

    if (pt->right != nullptr) {
        pt->right->parent = pt;
    }

    pt_right->parent = pt->parent;

    if (pt->parent == nullptr) {
        root = pt_right;
    } else if (pt == pt->parent->left) {
        pt->parent->left = pt_right;
    } else {
        pt->parent->right = pt_right;
    }

    pt_right->left = pt;
    pt->parent = pt_right;
}

void rotateRight(Node*& root, Node*& pt) {
    Node* pt_left = pt->left;

    pt->left = pt_left->right;

    if (pt->left != nullptr) {
        pt->left->parent = pt;
    }

    pt_left->parent = pt->parent;

    if (pt->parent == nullptr) {
        root = pt_left;
    } else if (pt == pt->parent->left) {
        pt->parent->left = pt_left;
    } else {
        pt->parent->right = pt_left;
    }

    pt_left->right = pt;
    pt->parent = pt_left;
}

void fixViolation(Node*& root, Node*& pt) {
    Node* parent_pt = nullptr;
    Node* grand_parent_pt = nullptr;

    while ((pt != root) && (pt->color != BLACK) && (pt->parent->color == RED)) {
        parent_pt = pt->parent;
        grand_parent_pt = pt->parent->parent;

        if (parent_pt == grand_parent_pt->left) {
            Node* uncle_pt = grand_parent_pt->right;

            if (uncle_pt != nullptr && uncle_pt->color == RED) {
                grand_parent_pt->color = RED;
                parent_pt->color = BLACK;
                uncle_pt->color = BLACK;
                pt = grand_parent_pt;
            } else {
                if (pt == parent_pt->right) {
                    rotateLeft(root, parent_pt);
                    pt = parent_pt;
                    parent_pt = pt->parent;
                }

                rotateRight(root, grand_parent_pt);
                std::swap(parent_pt->color, grand_parent_pt->color);
                pt = parent_pt;
            }
        } else {
            Node* uncle_pt = grand_parent_pt->left;

            if (uncle_pt != nullptr && uncle_pt->color == RED) {
                grand_parent_pt->color = RED;
                parent_pt->color = BLACK;
                uncle_pt->color = BLACK;
                pt = grand_parent_pt;
            } else {
                if (pt == parent_pt->left) {
                    rotateRight(root, parent_pt);
                    pt = parent_pt;
                    parent_pt = pt->parent;
                }

                rotateLeft(root, grand_parent_pt);
                std::swap(parent_pt->color, grand_parent_pt->color);
                pt = parent_pt;
            }
        }
    }

    root->color = BLACK;
}

void insert(Node*& root, int key) {
    Node* pt = new Node(key);

    root = insert(root, pt);

    fixViolation(root, pt);
}
```

##### Összefoglalás

A balanszírozás kulcsfontosságú a bináris keresőfák hatékony működésének fenntartásához. Az AVL fák és piros-fekete fák mindkét megközelítése különböző módszereket alkalmaz a balansz fenntartására, biztosítva, hogy a fa magassága mindig O(log n) maradjon. A megfelelő balanszírozás garantálja, hogy a keresési, beszúrási és törlési műveletek időkomplexitása hatékony marad, ami elengedhetetlen a modern adatfeldolgozó rendszerekben és algoritmusokban.

### 3.3. AVL fák

Az AVL fák a kiegyensúlyozott bináris keresőfák egyik típusát képviselik, amelyek rendkívül fontosak a hatékony adatkezelés és keresés szempontjából. Ezen fejezet célja, hogy mélyebb betekintést nyújtson az AVL fák működésébe és azok alkalmazási lehetőségeibe. Az AVL fák lehetővé teszik az adatok gyorsabb elérését és manipulálását, mivel garantálják, hogy a fa minden ága közel azonos magasságú maradjon. Ezt a tulajdonságukat a beépített rotációs műveletek segítségével tartják fenn, amelyek biztosítják a kiegyensúlyozottságot minden beszúrás vagy törlés után. A fejezet részletesen tárgyalja az AVL fák alapjait, a szükséges rotációs műveleteket, a magassági egyensúly fenntartásának módszereit, valamint az AVL fák teljesítményének elemzését, bemutatva, hogyan járulnak hozzá a hatékony adatkezeléshez.

#### 3.3.1. AVL fák alapjai

Az AVL fa egy speciális típusú bináris keresőfa, amely biztosítja a fa kiegyensúlyozottságát a beszúrási és törlési műveletek során. Nevét két szovjet informatikusról, Georgy Adelson-Velsky-ről és Evgenii Landis-ról kapta, akik 1962-ben vezették be ezt az adatstruktúrát. Az AVL fák egyik legfontosabb tulajdonsága, hogy minden csomópontja kiegyensúlyozott, azaz a bal és a jobb al-fák magasságának különbsége legfeljebb 1 lehet. Ez a tulajdonság garantálja, hogy a műveletek legrosszabb esetben is O(log n) időben futnak, ahol n a csomópontok száma a fában.

##### AVL fák tulajdonságai

Egy AVL fa alapvető tulajdonságai a következők:
- **Bináris keresőfa tulajdonság**: Minden csomópont bal al-fájában található elemek kisebbek, míg a jobb al-fájában található elemek nagyobbak, mint a csomópont értéke.
- **Kiegyensúlyozottsági feltétel**: Minden csomópont esetében a bal és jobb al-fák magasságának különbsége legfeljebb 1.

Az AVL fa minden csomópontjához tartozik egy magasság érték, amely a csomópont mélységétől függ. A magasságot az alábbiak szerint definiáljuk:
- Egy levél csomópont magassága 1.
- Egy nem levél csomópont magassága 1 plusz a bal és jobb al-fák magasságainak maximuma.

##### AVL fa szerkezete

Egy AVL fa csomópontjának definíciója C++ nyelven az alábbiak szerint néz ki:

```cpp
#include <iostream>
#include <algorithm>

using namespace std;

struct Node {
    int key;
    Node* left;
    Node* right;
    int height;
    
    Node(int k) : key(k), left(nullptr), right(nullptr), height(1) {}
};
```

##### Magasság kiszámítása

A csomópont magasságának kiszámítása egy egyszerű függvénnyel történik:

```cpp
int height(Node* n) {
    if (n == nullptr) return 0;
    return n->height;
}
```

##### Egyensúlyi tényező kiszámítása

Az egyensúlyi tényező (balance factor) a bal és a jobb al-fák magasságának különbsége:

```cpp
int getBalance(Node* n) {
    if (n == nullptr) return 0;
    return height(n->left) - height(n->right);
}
```

##### Beszúrás egy AVL fába

A beszúrási művelet során először egy standard bináris keresőfa beszúrás történik, majd a fa kiegyensúlyozása következik. A beszúrási művelet C++ implementációja az alábbi:

```cpp
Node* insert(Node* node, int key) {
    // 1. Standard BST beszúrás
    if (node == nullptr) return new Node(key);
    
    if (key < node->key)
        node->left = insert(node->left, key);
    else if (key > node->key)
        node->right = insert(node->right, key);
    else
        return node; // Azonos kulcsokat nem engedünk meg

    // 2. Csomópont magasságának frissítése
    node->height = 1 + max(height(node->left), height(node->right));

    // 3. Egyensúlyi tényező kiszámítása
    int balance = getBalance(node);

    // 4. Kiegyensúlyozás
    // Bal-Bal eset
    if (balance > 1 && key < node->left->key)
        return rightRotate(node);

    // Jobb-Jobb eset
    if (balance < -1 && key > node->right->key)
        return leftRotate(node);

    // Bal-Jobb eset
    if (balance > 1 && key > node->left->key) {
        node->left = leftRotate(node->left);
        return rightRotate(node);
    }

    // Jobb-Bal eset
    if (balance < -1 && key < node->right->key) {
        node->right = rightRotate(node->right);
        return leftRotate(node);
    }

    return node;
}
```

##### Rotációs műveletek

Az AVL fák kiegyensúlyozásához rotációs műveleteket alkalmazunk. Két alapvető rotáció létezik: jobb rotáció és bal rotáció.

###### Jobb rotáció

A jobb rotáció (right rotation) egy csomópont bal al-fáját emeli fel, és az adott csomópontot a jobb al-fába helyezi. Az alábbi függvény végzi el a jobb rotációt:

```cpp
Node* rightRotate(Node* y) {
    Node* x = y->left;
    Node* T2 = x->right;

    // Rotáció végrehajtása
    x->right = y;
    y->left = T2;

    // Magasságok frissítése
    y->height = max(height(y->left), height(y->right)) + 1;
    x->height = max(height(x->left), height(x->right)) + 1;

    // Új gyökér visszaadása
    return x;
}
```

###### Bal rotáció

A bal rotáció (left rotation) egy csomópont jobb al-fáját emeli fel, és az adott csomópontot a bal al-fába helyezi. Az alábbi függvény végzi el a bal rotációt:

```cpp
Node* leftRotate(Node* x) {
    Node* y = x->right;
    Node* T2 = y->left;

    // Rotáció végrehajtása
    y->left = x;
    x->right = T2;

    // Magasságok frissítése
    x->height = max(height(x->left), height(x->right)) + 1;
    y->height = max(height(y->left), height(y->right)) + 1;

    // Új gyökér visszaadása
    return y;
}
```

##### AVL fa példa

Az alábbiakban egy teljes példa található egy AVL fa használatára, beleértve a beszúrási műveleteket:

```cpp
int main() {
    Node* root = nullptr;

    root = insert(root, 10);
    root = insert(root, 20);
    root = insert(root, 30);
    root = insert(root, 40);
    root = insert(root, 50);
    root = insert(root, 25);

    cout << "Inorder traversal of the constructed AVL tree is \n";
    inorderTraversal(root);

    return 0;
}

void inorderTraversal(Node* root) {
    if (root != nullptr) {
        inorderTraversal(root->left);
        cout << root->key << " ";
        inorderTraversal(root->right);
    }
}
```

##### Összefoglalás

Az AVL fák hatékony adatstruktúrák, amelyek garantálják a gyors keresési, beszúrási és törlési műveleteket a kiegyensúlyozottságuk fenntartásával. Az egyensúlyi tényezők és a rotációs műveletek révén az AVL fák képesek biztosítani, hogy a fa magassága mindig logaritmikus maradjon a csomópontok számának függvényében. Az AVL fák ezen tulajdonságai miatt széles körben alkalmazhatók különféle számítástechnikai területeken, ahol a gyors és hatékony adatkezelés elengedhetetlen.

#### 3.3.2. Rotációs műveletek

Az AVL fák működésének alapja a kiegyensúlyozottság fenntartása, amelyet rotációs műveletekkel érünk el. A rotációs műveletek biztosítják, hogy a beszúrási és törlési műveletek után az AVL fa magassági egyensúlya helyreálljon. Két alapvető rotációs műveletet különböztetünk meg: a bal rotációt és a jobb rotációt. Ezek kombinációjával további két összetett rotációt is végrehajthatunk: a bal-jobb rotációt és a jobb-bal rotációt.

##### Alapvető rotációk

###### Jobb rotáció (Right Rotation)

A jobb rotáció akkor szükséges, ha egy csomópont bal al-fája nehezebb, mint a jobb al-fa. A jobb rotáció az alábbi lépésekből áll:

1. Jelöljük a rotálódó csomópontot $y$-nal, és annak bal gyermekét $x$-szel.
2. Az $x$ jobb gyermekét $T2$-vel jelöljük.
3. Végezzük el a rotációt úgy, hogy $x$ lesz az új gyökér, $y$ pedig az $x$ jobb gyermeke, míg $T2$ $y$ bal gyermekévé válik.

A jobb rotáció grafikus ábrázolása:

```
    y                                  x
   / \        Right Rotation         /   \
  x   T3    - - - - - - - - >       T1    y
 / \                                    /  \
T1   T2                                T2   T3
```

A jobb rotáció C++ implementációja:

```cpp
Node* rightRotate(Node* y) {
    Node* x = y->left;
    Node* T2 = x->right;

    // Rotáció végrehajtása
    x->right = y;
    y->left = T2;

    // Magasságok frissítése
    y->height = max(height(y->left), height(y->right)) + 1;
    x->height = max(height(x->left), height(x->right)) + 1;

    // Új gyökér visszaadása
    return x;
}
```

###### Bal rotáció (Left Rotation)

A bal rotáció akkor szükséges, ha egy csomópont jobb al-fája nehezebb, mint a bal al-fa. A bal rotáció az alábbi lépésekből áll:

1. Jelöljük a rotálódó csomópontot $x$-szel, és annak jobb gyermekét $y$-nal.
2. Az $y$ bal gyermekét $T2$-vel jelöljük.
3. Végezzük el a rotációt úgy, hogy $y$ lesz az új gyökér, $x$ pedig az $y$ bal gyermeke, míg $T2$ $x$ jobb gyermekévé válik.

A bal rotáció grafikus ábrázolása:

```
    x                                y
   /  \     Left Rotation          /   \
 T1   y   - - - - - - - - >       x    T3
     /  \                         /  \
    T2   T3                      T1   T2
```

A bal rotáció C++ implementációja:

```cpp
Node* leftRotate(Node* x) {
    Node* y = x->right;
    Node* T2 = y->left;

    // Rotáció végrehajtása
    y->left = x;
    x->right = T2;

    // Magasságok frissítése
    x->height = max(height(x->left), height(x->right)) + 1;
    y->height = max(height(y->left), height(y->right)) + 1;

    // Új gyökér visszaadása
    return y;
}
```

##### Összetett rotációk

Az összetett rotációk kombinált rotációs műveletek, amelyek akkor szükségesek, ha a fa szerkezete olyan módon torzul, hogy egyszeri jobb vagy bal rotáció nem elegendő a kiegyensúlyozáshoz.

###### Bal-jobb rotáció (Left-Right Rotation)

A bal-jobb rotáció akkor szükséges, ha egy csomópont bal gyermekének jobb al-fája nehezebb, mint a bal al-fa. A bal-jobb rotáció két lépésben történik:

1. Bal rotáció az $x$ bal gyermekére ($x$-en).
2. Jobb rotáció az $x$ csomópontra ($z$-n).

Grafikus ábrázolás:

```
    z                                      z
   / \                                   /   \
  y   T4  - - - - - - - - >           x       T4
 / \                                  /  \
T1   x                               y    T3
    /  \                            / \
   T2   T3                        T1   T2
```

C++ implementáció:

```cpp
Node* leftRightRotate(Node* z) {
    z->left = leftRotate(z->left);
    return rightRotate(z);
}
```

###### Jobb-bal rotáció (Right-Left Rotation)

A jobb-bal rotáció akkor szükséges, ha egy csomópont jobb gyermekének bal al-fája nehezebb, mint a jobb al-fa. A jobb-bal rotáció két lépésben történik:

1. Jobb rotáció az $x$ jobb gyermekére ($y$-on).
2. Bal rotáció az $x$ csomópontra ($z$-n).

Grafikus ábrázolás:

```
    z                                      z
   / \                                   /   \
  T1   y  - - - - - - - - >           T1       x
      / \                                    /  \
     x   T4                                 y    T3
    /  \                                  / \
   T2   T3                               T2   T4
```

C++ implementáció:

```cpp
Node* rightLeftRotate(Node* z) {
    z->right = rightRotate(z->right);
    return leftRotate(z);
}
```

##### Rotációs műveletek integrálása a beszúrási műveletbe

A beszúrási művelet során a fa szerkezetének változásai miatt szükségessé válhat a rotációk végrehajtása az egyensúly fenntartása érdekében. A rotációk végrehajtása az alábbi módon történik a beszúrási műveletben:

```cpp
Node* insert(Node* node, int key) {
    // 1. Standard BST beszúrás
    if (node == nullptr) return new Node(key);

    if (key < node->key)
        node->left = insert(node->left, key);
    else if (key > node->key)
        node->right = insert(node->right, key);
    else
        return node; // Azonos kulcsokat nem engedünk meg

    // 2. Csomópont magasságának frissítése
    node->height = 1 + max(height(node->left), height(node->right));

    // 3. Egyensúlyi tényező kiszámítása
    int balance = getBalance(node);

    // 4. Kiegyensúlyozás
    // Bal-Bal eset
    if (balance > 1 && key < node->left->key)
        return rightRotate(node);

    // Jobb-Jobb eset
    if (balance < -1 && key > node->right->key)
        return leftRotate(node);

    // Bal-Jobb eset
    if (balance > 1 && key > node->left->key) {
        node->left = leftRotate(node->left);
        return rightRotate(node);
    }

    // Jobb-Bal eset
    if (balance < -1 && key < node->right->key) {
        node->right = rightRotate(node->right);
        return leftRotate(node);
    }

    return node;
}
```

##### Rotációk a törlési műveletben

A törlési művelet során is szükség lehet rotációkra, hogy fenntartsuk az AVL fa kiegyensúlyozottságát. A törlés során a következő lépések szükségesek:

1. Standard bináris keresőfa törlés végrehajtása.
2. Csomópont magasságának frissítése.
3. Egyensúlyi tényező kiszámítása és szükség esetén rotációk végrehajtása.

A törlés rotációs műveleteinek implementációja:

```cpp
Node* deleteNode(Node* root, int key) {
    // 1. Standard BST törlés
    if (root == nullptr) return root;

    if (key < root->key)
        root->left = deleteNode(root->left, key);
    else if (key > root->key)
        root->right = deleteNode(root->right, key);
    else {
        if ((root->left == nullptr) || (root->right == nullptr)) {
            Node* temp = root->left ? root->left : root->right;
            if (temp == nullptr) {
                temp = root;
                root = nullptr;
            } else
                *root = *temp;
            delete temp;
        } else {
            Node* temp = minValueNode(root->right);
            root->key = temp->key;
            root->right = deleteNode(root->right, temp->key);
        }
    }

    if (root == nullptr) return root;

    // 2. Csomópont magasságának frissítése
    root->height = 1 + max(height(root->left), height(root->right));

    // 3. Egyensúlyi tényező kiszámítása
    int balance = getBalance(root);

    // 4. Kiegyensúlyozás
    // Bal-Bal eset
    if (balance > 1 && getBalance(root->left) >= 0)
        return rightRotate(root);

    // Bal-Jobb eset
    if (balance > 1 && getBalance(root->left) < 0) {
        root->left = leftRotate(root->left);
        return rightRotate(root);
    }

    // Jobb-Jobb eset
    if (balance < -1 && getBalance(root->right) <= 0)
        return leftRotate(root);

    // Jobb-Bal eset
    if (balance < -1 && getBalance(root->right) > 0) {
        root->right = rightRotate(root->right);
        return leftRotate(root);
    }

    return root;
}

Node* minValueNode(Node* node) {
    Node* current = node;
    while (current->left != nullptr)
        current = current->left;
    return current;
}
```

##### Összefoglalás

A rotációs műveletek alapvető szerepet játszanak az AVL fák kiegyensúlyozottságának fenntartásában. A jobb és bal rotáció, valamint ezek kombinációi (bal-jobb és jobb-bal rotáció) biztosítják, hogy minden beszúrási és törlési művelet után a fa magassága logaritmikus maradjon a csomópontok számának függvényében. Az AVL fák ezen kiegyensúlyozási műveletei garantálják az adatstruktúra hatékonyságát, így alkalmazásuk számos számítástechnikai területen elengedhetetlen.

#### 3.3.3. Magassági egyensúly

Az AVL fák egyik legfontosabb jellemzője a magassági egyensúly fenntartása. Ez a tulajdonság biztosítja, hogy minden csomópont bal és jobb al-fájának magasságkülönbsége legfeljebb 1 legyen. A magassági egyensúly megőrzése elengedhetetlen a fa hatékonyságának fenntartása érdekében, mivel garantálja, hogy a műveletek (beszúrás, törlés, keresés) legrosszabb esetben is O(log n) időben futnak, ahol n a csomópontok száma.

##### Magasság és egyensúlyi tényező

Minden AVL fa csomópontjához tartozik egy magassági érték, amely az adott csomópont alatti maximális útvonal hosszát jelenti. Az egyensúlyi tényező (balance factor) a bal és a jobb al-fák magasságkülönbségeként definiálható. Az egyensúlyi tényező értéke -1, 0 vagy +1 lehet egy kiegyensúlyozott AVL fában.

A csomópont magasságának kiszámítása és az egyensúlyi tényező meghatározása:

```cpp
struct Node {
    int key;
    Node* left;
    Node* right;
    int height;

    Node(int k) : key(k), left(nullptr), right(nullptr), height(1) {}
};

int height(Node* n) {
    if (n == nullptr) return 0;
    return n->height;
}

int getBalance(Node* n) {
    if (n == nullptr) return 0;
    return height(n->left) - height(n->right);
}
```

##### Magasság frissítése

Minden beszúrási és törlési művelet után frissíteni kell a csomópont magassági értékét. A csomópont magassága az 1 plusz a bal és jobb al-fák magasságának maximuma:

```cpp
void updateHeight(Node* n) {
    n->height = 1 + max(height(n->left), height(n->right));
}
```

##### Kiegyensúlyozási stratégiák

Az AVL fák kiegyensúlyozása négy fő forgatókönyv alapján történik:
1. **Bal-Bal eset (LL Rotation)**: A bal al-fa magassága nagyobb, és a bal al-fában történt beszúrás.
2. **Jobb-Jobb eset (RR Rotation)**: A jobb al-fa magassága nagyobb, és a jobb al-fában történt beszúrás.
3. **Bal-Jobb eset (LR Rotation)**: A bal al-fa magassága nagyobb, de a bal al-fa jobb gyermekében történt beszúrás.
4. **Jobb-Bal eset (RL Rotation)**: A jobb al-fa magassága nagyobb, de a jobb al-fa bal gyermekében történt beszúrás.

###### Bal-Bal eset

A bal-bal eset akkor fordul elő, ha egy csomópont bal al-fája nehezebb, és a bal al-fában történt beszúrás. Ilyenkor egy egyszerű jobb rotációval helyreállítható az egyensúly:

```cpp
Node* rightRotate(Node* y) {
    Node* x = y->left;
    Node* T2 = x->right;

    x->right = y;
    y->left = T2;

    updateHeight(y);
    updateHeight(x);

    return x;
}
```

###### Jobb-Jobb eset

A jobb-jobb eset akkor fordul elő, ha egy csomópont jobb al-fája nehezebb, és a jobb al-fában történt beszúrás. Ilyenkor egy egyszerű bal rotációval helyreállítható az egyensúly:

```cpp
Node* leftRotate(Node* x) {
    Node* y = x->right;
    Node* T2 = y->left;

    y->left = x;
    x->right = T2;

    updateHeight(x);
    updateHeight(y);

    return y;
}
```

###### Bal-Jobb eset

A bal-jobb eset akkor fordul elő, ha egy csomópont bal al-fája nehezebb, de a bal al-fa jobb gyermekében történt beszúrás. Ilyenkor először egy bal rotációt kell végrehajtani a bal gyermekre, majd egy jobb rotációt a csomópontra:

```cpp
Node* leftRightRotate(Node* z) {
    z->left = leftRotate(z->left);
    return rightRotate(z);
}
```

###### Jobb-Bal eset

A jobb-bal eset akkor fordul elő, ha egy csomópont jobb al-fája nehezebb, de a jobb al-fa bal gyermekében történt beszúrás. Ilyenkor először egy jobb rotációt kell végrehajtani a jobb gyermekre, majd egy bal rotációt a csomópontra:

```cpp
Node* rightLeftRotate(Node* z) {
    z->right = rightRotate(z->right);
    return leftRotate(z);
}
```

##### Beszúrás AVL fába magassági egyensúllyal

Az AVL fába történő beszúrás során a kiegyensúlyozottság megőrzése érdekében minden beszúrás után ellenőrizni és szükség esetén korrigálni kell a fa magassági egyensúlyát. Az alábbi kód bemutatja, hogyan történik a beszúrás egy AVL fába, figyelembe véve a magassági egyensúlyt:

```cpp
Node* insert(Node* node, int key) {
    if (node == nullptr) return new Node(key);

    if (key < node->key)
        node->left = insert(node->left, key);
    else if (key > node->key)
        node->right = insert(node->right, key);
    else
        return node;

    updateHeight(node);

    int balance = getBalance(node);

    if (balance > 1 && key < node->left->key)
        return rightRotate(node);

    if (balance < -1 && key > node->right->key)
        return leftRotate(node);

    if (balance > 1 && key > node->left->key) {
        node->left = leftRotate(node->left);
        return rightRotate(node);
    }

    if (balance < -1 && key < node->right->key) {
        node->right = rightRotate(node->right);
        return leftRotate(node);
    }

    return node;
}
```

##### Törlés AVL fában magassági egyensúllyal

Az AVL fában történő törlés során is biztosítani kell a magassági egyensúly fenntartását. A törlés után a csomópont magasságának frissítése és az egyensúlyi tényező ellenőrzése szükséges. Az alábbi kód bemutatja, hogyan történik a törlés egy AVL fában, figyelembe véve a magassági egyensúlyt:

```cpp
Node* deleteNode(Node* root, int key) {
    if (root == nullptr) return root;

    if (key < root->key)
        root->left = deleteNode(root->left, key);
    else if (key > root->key)
        root->right = deleteNode(root->right, key);
    else {
        if ((root->left == nullptr) || (root->right == nullptr)) {
            Node* temp = root->left ? root->left : root->right;
            if (temp == nullptr) {
                temp = root;
                root = nullptr;
            } else
                *root = *temp;
            delete temp;
        } else {
            Node* temp = minValueNode(root->right);
            root->key = temp->key;
            root->right = deleteNode(root->right, temp->key);
        }
    }

    if (root == nullptr) return root;

    updateHeight(root);

    int balance = getBalance(root);

    if (balance > 1 && getBalance(root->left) >= 0)
        return rightRotate(root);

    if (balance > 1 && getBalance(root->left) < 0) {
        root->left = leftRotate(root->left);
        return rightRotate(root);
    }

    if (balance < -1 && getBalance(root->right) <= 0)
        return leftRotate(root);

    if (balance < -1 && getBalance(root->right) > 0) {
        root->right = rightRotate(root->right);
        return leftRotate(root);
    }

    return root;
}

Node* minValueNode(Node* node) {
    Node* current = node;
    while (current->left != nullptr)
        current = current->left;
    return current;
}
```

##### Példák magassági egyensúly fenntartására

Az alábbiakban egy teljes példa található egy AVL fa használatára, beleértve a beszúrási és törlési műveleteket, valamint a magassági egyensúly fenntartását:

```cpp
#include <iostream>
#include <algorithm>

using namespace std;

struct Node {
    int key;
    Node* left;
    Node* right;
    int height;

    Node(int k) : key(k), left(nullptr), right(nullptr), height(1) {}
};

int height(Node* n) {
    if (n == nullptr) return 0;
    return n->height;
}

void updateHeight(Node* n) {
    n->height = 1 + max(height(n->left), height(n->right));
}

int getBalance(Node* n) {
    if (n == nullptr) return 0;
    return height(n->left) - height(n->right);
}

Node* rightRotate(Node* y) {
    Node* x = y->left;
    Node* T2 = x->right;

    x->right = y;
    y->left = T2;

    updateHeight(y);
    updateHeight(x);

    return x;
}

Node* leftRotate(Node* x) {
    Node* y = x->right;
    Node* T2 = y->left;

    y->left = x;
    x->right = T2;

    updateHeight(x);
    updateHeight(y);

    return y;
}

Node* insert(Node* node, int key) {
    if (node == nullptr) return new Node(key);

    if (key < node->key)
        node->left = insert(node->left, key);
    else if (key > node->key)
        node->right = insert(node->right, key);
    else
        return node;

    updateHeight(node);

    int balance = getBalance(node);

    if (balance > 1 && key < node->left->key)
        return rightRotate(node);

    if (balance < -1 && key > node->right->key)
        return leftRotate(node);

    if (balance > 1 && key > node->left->key) {
        node->left = leftRotate(node->left);
        return rightRotate(node);
    }

    if (balance < -1 && key < node->right->key) {
        node->right = rightRotate(node->right);
        return leftRotate(node);
    }

    return node;
}

Node* deleteNode(Node* root, int key) {
    if (root == nullptr) return root;

    if (key < root->key)
        root->left = deleteNode(root->left, key);
    else if (key > root->key)
        root->right = deleteNode(root->right, key);
    else {
        if ((root->left == nullptr) || (root->right == nullptr)) {
            Node* temp = root->left ? root->left : root->right;
            if (temp == nullptr) {
                temp = root;
                root = nullptr;
            } else
                *root = *temp;
            delete temp;
        } else {
            Node* temp = minValueNode(root->right);
            root->key = temp->key;
            root->right = deleteNode(root->right, temp->key);
        }
    }

    if (root == nullptr) return root;

    updateHeight(root);

    int balance = getBalance(root);

    if (balance > 1 && getBalance(root->left) >= 0)
        return rightRotate(root);

    if (balance > 1 && getBalance(root->left) < 0) {
        root->left = leftRotate(root->left);
        return rightRotate(root);
    }

    if (balance < -1 && getBalance(root->right) <= 0)
        return leftRotate(root);

    if (balance < -1 && getBalance(root->right) > 0) {
        root->right = rightRotate(root->right);
        return leftRotate(root);
    }

    return root;
}

Node* minValueNode(Node* node) {
    Node* current = node;
    while (current->left != nullptr)
        current = current->left;
    return current;
}

void inorderTraversal(Node* root) {
    if (root != nullptr) {
        inorderTraversal(root->left);
        cout << root->key << " ";
        inorderTraversal(root->right);
    }
}

int main() {
    Node* root = nullptr;

    root = insert(root, 10);
    root = insert(root, 20);
    root = insert(root, 30);
    root = insert(root, 40);
    root = insert(root, 50);
    root = insert(root, 25);

    cout << "Inorder traversal of the constructed AVL tree is \n";
    inorderTraversal(root);

    root = deleteNode(root, 40);

    cout << "\nInorder traversal after deletion of 40 \n";
    inorderTraversal(root);

    return 0;
}
```

##### Összefoglalás

Az AVL fák magassági egyensúlyának fenntartása kulcsfontosságú a hatékony működés szempontjából. A magassági egyensúly megőrzése érdekében a beszúrási és törlési műveletek után frissíteni kell a csomópontok magasságát és egyensúlyi tényezőjét, valamint szükség esetén rotációs műveleteket kell végrehajtani. Az AVL fák ezen tulajdonságai garantálják, hogy a műveletek legrosszabb esetben is logaritmikus időben futnak, így alkalmazásuk számos számítástechnikai területen elengedhetetlen.

### 3.3.4. Teljesítmény elemzés

Az AVL fák hatékony adatstruktúrák, amelyek garantálják a kiegyensúlyozott műveletek gyors végrehajtását. Ebben a fejezetben részletesen megvizsgáljuk az AVL fák teljesítményét különböző szempontok szerint, beleértve a beszúrási, törlési és keresési műveletek időbeli összetettségét, valamint összehasonlítjuk más adatstruktúrákkal, például a hagyományos bináris keresőfákkal (BST) és a vörös-fekete fákkal.

##### Időbeli összetettség

Az AVL fáknál minden művelet időbeli összetettsége a fa magasságától függ. Mivel az AVL fa garantálja, hogy a fa magassága mindig logaritmikus marad a csomópontok számának függvényében, a műveletek időbeli összetettsége is logaritmikus lesz.

###### Beszúrás

A beszúrási művelet során a következő lépések történnek:
1. A csomópont beszúrása a bináris keresőfa szabályai szerint.
2. A csomópontok magasságának frissítése.
3. Az egyensúlyi tényező ellenőrzése és szükség esetén rotációs műveletek végrehajtása.

Az AVL fában a beszúrási művelet időbeli összetettsége O(log n), mivel a legrosszabb esetben is csak logaritmikus számú csomópontot kell frissíteni és legfeljebb két rotációs műveletet kell végrehajtani.

###### Törlés

A törlési művelet során a következő lépések történnek:
1. A csomópont törlése a bináris keresőfa szabályai szerint.
2. A csomópontok magasságának frissítése.
3. Az egyensúlyi tényező ellenőrzése és szükség esetén rotációs műveletek végrehajtása.

Az AVL fában a törlési művelet időbeli összetettsége szintén O(log n), mivel a legrosszabb esetben is csak logaritmikus számú csomópontot kell frissíteni és legfeljebb két rotációs műveletet kell végrehajtani.

###### Keresés

A keresési művelet során a következő lépések történnek:
1. A keresés a bináris keresőfa szabályai szerint halad a fa gyökércsomópontjától kezdve lefelé.

Az AVL fában a keresési művelet időbeli összetettsége O(log n), mivel a fa kiegyensúlyozottsága garantálja, hogy a fa magassága logaritmikus marad.

##### AVL fa példák C++ nyelven

Az alábbiakban bemutatjuk a beszúrási, törlési és keresési műveletek implementációját C++ nyelven, amelyek az AVL fák kiegyensúlyozottságát fenntartják.

###### Beszúrás

```cpp
#include <iostream>
#include <algorithm>

using namespace std;

struct Node {
    int key;
    Node* left;
    Node* right;
    int height;

    Node(int k) : key(k), left(nullptr), right(nullptr), height(1) {}
};

int height(Node* n) {
    if (n == nullptr) return 0;
    return n->height;
}

void updateHeight(Node* n) {
    n->height = 1 + max(height(n->left), height(n->right));
}

int getBalance(Node* n) {
    if (n == nullptr) return 0;
    return height(n->left) - height(n->right);
}

Node* rightRotate(Node* y) {
    Node* x = y->left;
    Node* T2 = x->right;

    x->right = y;
    y->left = T2;

    updateHeight(y);
    updateHeight(x);

    return x;
}

Node* leftRotate(Node* x) {
    Node* y = x->right;
    Node* T2 = y->left;

    y->left = x;
    x->right = T2;

    updateHeight(x);
    updateHeight(y);

    return y;
}

Node* insert(Node* node, int key) {
    if (node == nullptr) return new Node(key);

    if (key < node->key)
        node->left = insert(node->left, key);
    else if (key > node->key)
        node->right = insert(node->right, key);
    else
        return node;

    updateHeight(node);

    int balance = getBalance(node);

    if (balance > 1 && key < node->left->key)
        return rightRotate(node);

    if (balance < -1 && key > node->right->key)
        return leftRotate(node);

    if (balance > 1 && key > node->left->key) {
        node->left = leftRotate(node->left);
        return rightRotate(node);
    }

    if (balance < -1 && key < node->right->key) {
        node->right = rightRotate(node->right);
        return leftRotate(node);
    }

    return node;
}

void inorderTraversal(Node* root) {
    if (root != nullptr) {
        inorderTraversal(root->left);
        cout << root->key << " ";
        inorderTraversal(root->right);
    }
}

int main() {
    Node* root = nullptr;

    root = insert(root, 10);
    root = insert(root, 20);
    root = insert(root, 30);
    root = insert(root, 40);
    root = insert(root, 50);
    root = insert(root, 25);

    cout << "Inorder traversal of the constructed AVL tree is \n";
    inorderTraversal(root);

    return 0;
}
```

###### Törlés

```cpp
Node* deleteNode(Node* root, int key) {
    if (root == nullptr) return root;

    if (key < root->key)
        root->left = deleteNode(root->left, key);
    else if (key > root->key)
        root->right = deleteNode(root->right, key);
    else {
        if ((root->left == nullptr) || (root->right == nullptr)) {
            Node* temp = root->left ? root->left : root->right;
            if (temp == nullptr) {
                temp = root;
                root = nullptr;
            } else
                *root = *temp;
            delete temp;
        } else {
            Node* temp = minValueNode(root->right);
            root->key = temp->key;
            root->right = deleteNode(root->right, temp->key);
        }
    }

    if (root == nullptr) return root;

    updateHeight(root);

    int balance = getBalance(root);

    if (balance > 1 && getBalance(root->left) >= 0)
        return rightRotate(root);

    if (balance > 1 && getBalance(root->left) < 0) {
        root->left = leftRotate(root->left);
        return rightRotate(root);
    }

    if (balance < -1 && getBalance(root->right) <= 0)
        return leftRotate(root);

    if (balance < -1 && getBalance(root->right) > 0) {
        root->right = rightRotate(root->right);
        return leftRotate(root);
    }

    return root;
}

Node* minValueNode(Node* node) {
    Node* current = node;
    while (current->left != nullptr)
        current = current->left;
    return current;
}

void inorderTraversal(Node* root) {
    if (root != nullptr) {
        inorderTraversal(root->left);
        cout << root->key << " ";
        inorderTraversal(root->right);
    }
}

int main() {
    Node* root = nullptr;

    root = insert(root, 10);
    root = insert(root, 20);
    root = insert(root, 30);
    root = insert(root, 40);
    root = insert(root, 50);
    root = insert(root, 25);

    cout << "Inorder traversal of the constructed AVL tree is \n";
    inorderTraversal(root);

    root = deleteNode(root, 40);

    cout << "\nInorder traversal after deletion of 40 \n";
    inorderTraversal(root);

    return 0;
}
```

###### Keresés

A keresési művelet AVL fában azonos a bináris keresőfákban alkalmazott módszerrel, mivel az AVL fa is bináris keresőfa:

```cpp
Node* search(Node* root, int key) {
    if (root == nullptr || root->key == key)
        return root;

    if (key < root->key)
        return search(root->left, key);

    return search(root->right, key);
}

int main() {
    Node* root = nullptr;

    root = insert(root, 10);
    root = insert(root, 20);
    root = insert(root, 30);
    root = insert(root, 40);
    root = insert(root, 50);
    root = insert(root, 25);

    Node* result = search(root, 30);
    if (result != nullptr)
        cout << "Node with key 30 found.\n";
    else
        cout << "Node with key 30 not found.\n";

    return 0;
}
```

##### Összehasonlítás más adatstruktúrákkal

###### Bináris keresőfa (BST)

A bináris keresőfa egy egyszerűbb adatstruktúra, amely nem garantálja a kiegyensúlyozottságot. A műveletek időbeli összetettsége a legrosszabb esetben O(n), amikor a fa teljesen kiegyensúlyozatlan (pl. láncolt lista). Az AVL fákkal szemben a BST előnye az egyszerűség és a kisebb karbantartási költség, de a teljesítménye gyakran alulmarad a kiegyensúlyozott fákkal szemben.

###### Vörös-fekete fa (Red-Black Tree)

A vörös-fekete fák egy másik kiegyensúlyozott bináris keresőfa típus, amely lazábban biztosítja a kiegyensúlyozottságot, mint az AVL fák. A vörös-fekete fák garantálják, hogy a műveletek időbeli összetettsége O(log n), hasonlóan az AVL fákhoz. Azonban a vörös-fekete fák előnye, hogy az átlagos esetben kevesebb rotációs műveletet igényelnek, így gyakran gyorsabbak az AVL fáknál a beszúrási és törlési műveletek során.

##### Gyakorlati alkalmazások

Az AVL fák olyan alkalmazásokban hasznosak, ahol a gyors keresés, beszúrás és törlés elengedhetetlen, és a fa kiegyensúlyozottságát folyamatosan fenn kell tartani. Például:

- **Adatbázisok**: Az AVL fák gyakran alkalmazhatók adatbázis indexeként, ahol az adatok gyors elérése és módosítása kulcsfontosságú.
- **Szövegszerkesztők**: A szövegszerkesztőkben az AVL fák használhatók a dokumentum szövegének gyors elérésére és módosítására.
- **Memóriakezelés**: Az operációs rendszerekben az AVL fák alkalmazhatók a szabad memóriaterületek gyors keresésére és allokálására.

##### Összefoglalás

Az AVL fák hatékony és kiegyensúlyozott adatstruktúrák, amelyek garantálják, hogy a műveletek időbeli összetettsége mindig logaritmikus marad. A magassági egyensúly fenntartása érdekében szükséges rotációs műveletek biztosítják, hogy a fa mindig kiegyensúlyozott maradjon, így az AVL fák kiváló választásnak bizonyulnak számos alkalmazási területen, ahol a gyors adatkezelés elengedhetetlen. Az AVL fák és más kiegyensúlyozott fák összehasonlítása alapján látható, hogy mindkét struktúra előnyös lehet különböző felhasználási esetekben, az AVL fák szigorúbb egyensúlyi feltételei révén különösen hasznosak a kritikus adatbázis-műveletekben.

#### 3.4.   Piros-fekete fák

A piros-fekete fák a kiegyensúlyozott bináris keresőfák egy speciális típusa, amelyeket a hatékony keresés, beszúrás és törlés műveletek érdekében fejlesztettek ki. Ezek a fák különleges tulajdonságokkal és szabályokkal rendelkeznek, amelyek biztosítják, hogy az egyes műveletek időbeli bonyolultsága amortizáltan O(log n) maradjon. A piros-fekete fák színezési szabályainak és az ezekhez kapcsolódó rotációs műveleteknek köszönhetően az adatszerkezet hatékonyan fenntartja a kiegyensúlyozottságot. Ebben a fejezetben részletesen megvizsgáljuk a piros-fekete fák alapfogalmait és tulajdonságait, ismertetjük a színezési szabályokat, bemutatjuk a rotációs és újra színezési technikákat, valamint összehasonlítjuk őket az AVL fákkal, hogy átfogó képet kapjunk ezen adatszerkezet előnyeiről és hátrányairól.

##### 3.4.1. Alapfogalmak és tulajdonságok

A piros-fekete fák (Red-Black Trees) speciális bináris keresőfák, amelyek szigorú szabályokat követnek a csomópontok színezésére vonatkozóan, hogy biztosítsák a fa kiegyensúlyozottságát. Az alábbiakban részletesen bemutatjuk ezen adatszerkezet alapfogalmait és tulajdonságait.

###### Alapfogalmak

1. **Csomópont (Node)**: Minden csomópont rendelkezik egy kulccsal, egy színnel (piros vagy fekete), egy bal és egy jobb gyerekkel, valamint egy szülő csomóponttal.
2. **Gyökér (Root)**: A fa legfelső csomópontja, amelynek nincs szülője.
3. **Levél (Leaf)**: A fa olyan csomópontjai, amelyeknek nincs gyereke. A piros-fekete fákban a levél csomópontokat gyakran fekete színű "NIL" csomópontokként ábrázolják, hogy megkönnyítsék az algoritmusok kezelését.
4. **Magasság (Height)**: A gyökértől a legmélyebb levélig vezető út hosszának maximális értéke.
5. **Fekete-magasság (Black-Height)**: Egy csomópontból egy levélig vezető út során érintett fekete csomópontok száma.

###### Tulajdonságok

A piros-fekete fák az alábbi tulajdonságokat tartják be, hogy biztosítsák a kiegyensúlyozottságot:

1. **Színezés**: Minden csomópont piros vagy fekete.
2. **Gyökér színe**: A gyökér mindig fekete.
3. **Levél színezése**: Minden levél (NIL csomópont) fekete.
4. **Piros csomópont szabálya**: Egy piros csomópontnak mindkét gyereke fekete. Ez megakadályozza, hogy két piros csomópont közvetlenül egymás után helyezkedjen el a fában.
5. **Fekete-magasság szabály**: Minden csomópontból bármelyik leszármazott levélig vezető út során azonos számú fekete csomópont található.

###### Példakód C++ nyelven

Az alábbiakban bemutatunk egy egyszerű C++ osztályt egy piros-fekete fa csomópontjának reprezentálására:

```cpp
#include <iostream>
using namespace std;

enum Color { RED, BLACK };

struct Node {
    int data;
    bool color;
    Node *left, *right, *parent;

    // Constructor
    Node(int data) {
        this->data = data;
        left = right = parent = nullptr;
        this->color = RED;
    }
};

class RedBlackTree {
private:
    Node *root;

protected:
    void rotateLeft(Node *&, Node *&);
    void rotateRight(Node *&, Node *&);
    void fixInsert(Node *&, Node *&);

public:
    // Constructor
    RedBlackTree() { root = nullptr; }
    void insert(const int &n);
    void inorder();
    void levelOrder();
};

// Function to perform inorder traversal
void inorderHelper(Node *root) {
    if (root == nullptr)
        return;

    inorderHelper(root->left);
    cout << root->data << " ";
    inorderHelper(root->right);
}

// Function to perform level order traversal
void levelOrderHelper(Node *root) {
    if (root == nullptr)
        return;

    std::queue<Node *> q;
    q.push(root);

    while (!q.empty()) {
        Node *temp = q.front();
        cout << temp->data << " ";
        q.pop();

        if (temp->left != nullptr)
            q.push(temp->left);

        if (temp->right != nullptr)
            q.push(temp->right);
    }
}

void RedBlackTree::rotateLeft(Node *&root, Node *&pt) {
    Node *pt_right = pt->right;

    pt->right = pt_right->left;

    if (pt->right != nullptr)
        pt->right->parent = pt;

    pt_right->parent = pt->parent;

    if (pt->parent == nullptr)
        root = pt_right;

    else if (pt == pt->parent->left)
        pt->parent->left = pt_right;

    else
        pt->parent->right = pt_right;

    pt_right->left = pt;
    pt->parent = pt_right;
}

void RedBlackTree::rotateRight(Node *&root, Node *&pt) {
    Node *pt_left = pt->left;

    pt->left = pt_left->right;

    if (pt->left != nullptr)
        pt->left->parent = pt;

    pt_left->parent = pt->parent;

    if (pt->parent == nullptr)
        root = pt_left;

    else if (pt == pt->parent->left)
        pt->parent->left = pt_left;

    else
        pt->parent->right = pt_left;

    pt_left->right = pt;
    pt->parent = pt_left;
}

void RedBlackTree::fixInsert(Node *&root, Node *&pt) {
    Node *parent_pt = nullptr;
    Node *grand_parent_pt = nullptr;

    while ((pt != root) && (pt->color != BLACK) &&
           (pt->parent->color == RED)) {

        parent_pt = pt->parent;
        grand_parent_pt = pt->parent->parent;

        /* Case : A
            Parent of pt is left child
            of Grand-parent of pt */
        if (parent_pt == grand_parent_pt->left) {

            Node *uncle_pt = grand_parent_pt->right;

            /* Case : 1
                The uncle of pt is also red
                Only Recoloring required */
            if (uncle_pt != nullptr && uncle_pt->color == RED) {
                grand_parent_pt->color = RED;
                parent_pt->color = BLACK;
                uncle_pt->color = BLACK;
                pt = grand_parent_pt;
            }

            else {
                /* Case : 2
                    pt is right child of its parent
                    Left-rotation required */
                if (pt == parent_pt->right) {
                    rotateLeft(root, parent_pt);
                    pt = parent_pt;
                    parent_pt = pt->parent;
                }

                /* Case : 3
                    pt is left child of its parent
                    Right-rotation required */
                rotateRight(root, grand_parent_pt);
                swap(parent_pt->color, grand_parent_pt->color);
                pt = parent_pt;
            }
        }

        /* Case : B
            Parent of pt is right child
            of Grand-parent of pt */
        else {
            Node *uncle_pt = grand_parent_pt->left;

            /*  Case : 1
                The uncle of pt is also red
                Only Recoloring required */
            if ((uncle_pt != nullptr) && (uncle_pt->color == RED)) {
                grand_parent_pt->color = RED;
                parent_pt->color = BLACK;
                uncle_pt->color = BLACK;
                pt = grand_parent_pt;
            }
            else {
                /* Case : 2
                    pt is left child of its parent
                    Right-rotation required */
                if (pt == parent_pt->left) {
                    rotateRight(root, parent_pt);
                    pt = parent_pt;
                    parent_pt = pt->parent;
                }

                /* Case : 3
                    pt is right child of its parent
                    Left-rotation required */
                rotateLeft(root, grand_parent_pt);
                swap(parent_pt->color, grand_parent_pt->color);
                pt = parent_pt;
            }
        }


    }

    root->color = BLACK;
}

void RedBlackTree::insert(const int &data) {
    Node *pt = new Node(data);

    root = BSTInsert(root, pt);

    fixInsert(root, pt);
}

void RedBlackTree::inorder() { inorderHelper(root); }

void RedBlackTree::levelOrder() { levelOrderHelper(root); }
```

Ez a kód egy egyszerű piros-fekete fa implementációját mutatja be C++ nyelven. A `Node` struktúra definiálja a fa csomópontjait, míg a `RedBlackTree` osztály tartalmazza azokat a műveleteket, amelyek a fa kiegyensúlyozásához szükségesek. Az `insert` függvény egy új csomópontot szúr be a fába, majd a `fixInsert` függvény segítségével biztosítja, hogy a fa továbbra is megfeleljen a piros-fekete fa szabályainak.


#### 3.4.2. Színezési szabályok

A piros-fekete fák (Red-Black Trees) a bináris keresőfák egy speciális típusát alkotják, amelyek az egyensúly megtartása érdekében szigorú színezési szabályokat követnek. Ezek a szabályok biztosítják, hogy a fa magassága mindig logaritmikus marad az elemek számához képest, így garantálva az O(log n) keresési, beszúrási és törlési időt. Ebben az alfejezetben részletesen bemutatjuk a piros-fekete fák színezési szabályait, amelyek a fa strukturális egyensúlyának fenntartására szolgálnak.

##### Alapvető Színezési Szabályok

A piros-fekete fák öt alapvető színezési szabályt követnek, amelyek biztosítják a fa kiegyensúlyozottságát:

1. **Szín Szabály**: Minden csomópont vagy piros, vagy fekete színű.
2. **Gyökér Színe**: A gyökér mindig fekete.
3. **Levél Színezése**: Minden levél, vagyis NIL csomópont, fekete.
4. **Piros Szülő Szabály**: Ha egy csomópont piros, akkor mindkét gyermeke fekete. Ez megakadályozza, hogy két piros csomópont közvetlenül egymás után helyezkedjen el a fában, ezzel segítve az egyensúly fenntartását.
5. **Fekete-Magasság Szabály**: Minden csomópontból bármelyik leszármazott levélig vezető út során azonos számú fekete csomópont található.

##### Színezési Szabályok Részletesen

###### 1. Szín Szabály

A piros-fekete fa minden csomópontja egy színt hordoz: piros vagy fekete. Ez a szín a fa kiegyensúlyozottságának fenntartására szolgál. A szín információ kritikus fontosságú a beszúrás és törlés során végrehajtott egyensúlyozási műveletekben (rotációk és újraszínezés).

###### 2. Gyökér Színe

A fa gyökere mindig fekete. Ez a szabály biztosítja, hogy a fa legfelső szintje stabil és kiegyensúlyozott maradjon. Ha a beszúrási vagy törlési műveletek során a gyökér pirosra vált, akkor azt azonnal feketévé kell alakítani, hogy megfeleljen ennek a szabálynak.

###### 3. Levél Színezése

Minden levél, amely egy NIL (null) csomópont, fekete. Ezek a NIL csomópontok nem tartalmaznak adatokat és csak a fa szerkezetének fenntartására szolgálnak. A fekete levelek biztosítják, hogy minden valódi csomópontból a levélig vezető út során azonos számú fekete csomópont található, ami a következő szabályhoz kapcsolódik.

###### 4. Piros Szülő Szabály

Ha egy csomópont piros, akkor mindkét gyermeke fekete. Ez a szabály biztosítja, hogy ne legyen két egymást követő piros csomópont, ami segít megőrizni a fa kiegyensúlyozottságát. Ez a szabály különösen fontos a beszúrás és törlés utáni egyensúlyozási folyamatok során, ahol új piros csomópontok kerülhetnek a fába.

###### 5. Fekete-Magasság Szabály

Minden csomópontból bármelyik leszármazott levélig vezető út során azonos számú fekete csomópont található. Ez a tulajdonság biztosítja, hogy a fa nem lesz túlságosan lejtős egyik irányban sem, mivel minden út azonos hosszúságú fekete csomópontokban mérve. A fekete-magasság fogalma kritikus jelentőségű a piros-fekete fa kiegyensúlyozottságának megőrzése szempontjából.

##### Színezési Szabályok Alkalmazása

A színezési szabályokat különösen fontos figyelembe venni a beszúrás és törlés műveletek során, mivel ezek a műveletek megzavarhatják a fa egyensúlyát. Az alábbiakban bemutatjuk, hogyan alkalmazzuk a színezési szabályokat ezekben a műveletekben.

###### Beszúrás

A beszúrási művelet során az új csomópontot mindig piros színűként szúrjuk be. Ezután ellenőriznünk kell a színezési szabályokat, és szükség esetén rotációkat és újraszínezést kell végrehajtani a fa kiegyensúlyozásának fenntartása érdekében.

Például:

```cpp
void fixInsert(Node *&root, Node *&pt) {
    Node *parent_pt = nullptr;
    Node *grand_parent_pt = nullptr;

    while ((pt != root) && (pt->color != BLACK) &&
           (pt->parent->color == RED)) {

        parent_pt = pt->parent;
        grand_parent_pt = pt->parent->parent;

        /* Case : A
            Parent of pt is left child
            of Grand-parent of pt */
        if (parent_pt == grand_parent_pt->left) {

            Node *uncle_pt = grand_parent_pt->right;

            /* Case : 1
                The uncle of pt is also red
                Only Recoloring required */
            if (uncle_pt != nullptr && uncle_pt->color == RED) {
                grand_parent_pt->color = RED;
                parent_pt->color = BLACK;
                uncle_pt->color = BLACK;
                pt = grand_parent_pt;
            }

            else {
                /* Case : 2
                    pt is right child of its parent
                    Left-rotation required */
                if (pt == parent_pt->right) {
                    rotateLeft(root, parent_pt);
                    pt = parent_pt;
                    parent_pt = pt->parent;
                }

                /* Case : 3
                    pt is left child of its parent
                    Right-rotation required */
                rotateRight(root, grand_parent_pt);
                swap(parent_pt->color, grand_parent_pt->color);
                pt = parent_pt;
            }
        }

        /* Case : B
            Parent of pt is right child
            of Grand-parent of pt */
        else {
            Node *uncle_pt = grand_parent_pt->left;

            /*  Case : 1
                The uncle of pt is also red
                Only Recoloring required */
            if ((uncle_pt != nullptr) && (uncle_pt->color == RED)) {
                grand_parent_pt->color = RED;
                parent_pt->color = BLACK;
                uncle_pt->color = BLACK;
                pt = grand_parent_pt;
            }
            else {
                /* Case : 2
                    pt is left child of its parent
                    Right-rotation required */
                if (pt == parent_pt->left) {
                    rotateRight(root, parent_pt);
                    pt = parent_pt;
                    parent_pt = pt->parent;
                }

                /* Case : 3
                    pt is right child of its parent
                    Left-rotation required */
                rotateLeft(root, grand_parent_pt);
                swap(parent_pt->color, grand_parent_pt->color);
                pt = parent_pt;
            }
        }
    }

    root->color = BLACK;
}
```

###### Törlés

A törlési művelet bonyolultabb, mivel nem csak az eltávolított csomópont helyettesítését kell megoldani, hanem a színezési szabályok fenntartását is biztosítani kell. A törlés utáni helyreállítás során gyakran több lépésben kell rotációkat és újraszínezéseket végrehajtani.

##### Összegzés

A piros-fekete fák színezési szabályai kritikus szerepet játszanak a fa kiegyensúlyozottságának fenntartásában. Ezek a szabályok biztosítják, hogy a fa magassága mindig logaritmikus marad az elemek számához képest, ezáltal garantálva az O(log n) keresési, beszúrási és törlési időt. A színezési szabályok betartása és a megfelelő rotációk és újraszínezések végrehajtása biztosítja, hogy a piros-fekete fa mindig kiegyensúlyozott és hatékonyan használható maradjon.

#### 3.4.3. Rotációk és újra színezés

A piros-fekete fák hatékony működésének kulcsa a rotációk és az újraszínezési műveletek megfelelő alkalmazása. Ezek a műveletek biztosítják, hogy a fa mindig kiegyensúlyozott maradjon, és a piros-fekete fa szabályai ne sérüljenek. Ebben az alfejezetben részletesen bemutatjuk a rotációk és az újraszínezés működését és alkalmazását, amelyek elengedhetetlenek a piros-fekete fák karbantartásában.

##### Rotációk

A rotációk olyan műveletek, amelyek átrendezik a csomópontok hierarchiáját egy piros-fekete fában anélkül, hogy megsértenék a bináris keresőfa tulajdonságait. Két alapvető rotációs művelet létezik: a bal rotáció és a jobb rotáció.

###### Bal rotáció (Left Rotation)

A bal rotáció a következőképpen működik: egy adott csomópontot, mondjuk $x$-et, lecserélünk a jobb gyermekével, $y$-nal. A művelet lépései:

1. $y$ bal gyermeke $x$ jobb gyermekévé válik.
2. Ha $y$-nak van bal gyermeke, akkor annak szülője $x$ lesz.
3. $x$ szülője $y$ szülője lesz.
4. Ha $x$ a gyökér volt, akkor most $y$ lesz az új gyökér.
5. $y$ bal gyermeke $x$ lesz, és $x$ szülője $y$.

Ez a művelet biztosítja, hogy a fa szerkezete megmaradjon, és az eredeti bináris keresőfa tulajdonságai ne sérüljenek.

###### Jobb rotáció (Right Rotation)

A jobb rotáció hasonló a bal rotációhoz, de az ellenkező irányban történik. Egy adott csomópontot, mondjuk $y$-t, lecserélünk a bal gyermekével, $x$-szel. A művelet lépései:

1. $x$ jobb gyermeke $y$ bal gyermekévé válik.
2. Ha $x$-nek van jobb gyermeke, akkor annak szülője $y$ lesz.
3. $y$ szülője $x$ szülője lesz.
4. Ha $y$ a gyökér volt, akkor most $x$ lesz az új gyökér.
5. $x$ jobb gyermeke $y$ lesz, és $y$ szülője $x$.

##### Újraszínezés

Az újraszínezés a piros-fekete fákban a csomópontok színének módosítását jelenti annak érdekében, hogy a színezési szabályok továbbra is érvényesek maradjanak. Az újraszínezés különösen fontos a beszúrási és törlési műveletek után, amikor a fa kiegyensúlyozottsága felborulhat.

###### Beszúrás utáni újraszínezés

Amikor egy új csomópontot beszúrunk a piros-fekete fába, azt mindig piros színnel tesszük. Ez az alapértelmezett piros szín segít abban, hogy ne sérüljön az 5. szabály (fekete-magasság szabály). Az új csomópont beszúrása után az alábbi lépések szerint végezzük el az újraszínezést és szükség esetén a rotációkat:

1. **Eset 1: Az új csomópont gyökér**: Ha az új csomópont a gyökér, akkor feketévé kell alakítani, hogy megfeleljen a 2. szabálynak (gyökér színe).
2. **Eset 2: Az új csomópont szülője fekete**: Ebben az esetben nincs teendő, mivel nem sértjük meg a 4. szabályt (piros szülő szabály).
3. **Eset 3: Az új csomópont szülője piros, a nagybácsi is piros**: Ebben az esetben a szülőt és a nagybácsit feketére, a nagyszülőt pedig pirosra színezzük át, majd az újraszínezést a nagyszülőre alkalmazzuk rekurzívan.
4. **Eset 4: Az új csomópont szülője piros, a nagybácsi fekete**: Ebben az esetben rotációra van szükség:
    - **Eset 4a: Az új csomópont a jobb gyermek, a szülő a bal gyermek**: Bal rotációt végzünk a szülőre.
    - **Eset 4b: Az új csomópont a bal gyermek, a szülő a jobb gyermek**: Jobb rotációt végzünk a szülőre.

Ezután átszínezzük a szülőt feketére és a nagyszülőt pirosra, majd végrehajtjuk a megfelelő rotációt a nagyszülőre.

###### Törlés utáni újraszínezés

A törlés utáni újraszínezés és rotációk bonyolultabbak lehetnek, mivel nem csak a törölt csomópont helyettesítését kell megoldani, hanem a fa kiegyensúlyozottságát is fenn kell tartani. A törlés utáni helyreállítás lépései:

1. **Eset 1: A törölt csomópont piros**: Ebben az esetben nincs szükség újraszínezésre, mivel a piros csomópont eltávolítása nem befolyásolja a fekete-magasságot.
2. **Eset 2: A törölt csomópont fekete, egy piros gyermek**: Ebben az esetben a piros gyermeket feketévé színezzük.
3. **Eset 3: A törölt csomópont fekete, két fekete gyermek**: Ebben az esetben a következő lépéseket kell végrehajtani:
    - **Eset 3a: A törölt csomópont testvére piros**: Ebben az esetben a testvért feketére, a szülőt pirosra színezzük, majd végrehajtjuk a megfelelő rotációt.
    - **Eset 3b: A törölt csomópont testvére fekete, a testvér mindkét gyermeke fekete**: Ebben az esetben a testvért pirosra színezzük, és a helyreállítási folyamatot a szülőnél folytatjuk.
    - **Eset 3c: A törölt csomópont testvére fekete, a testvérnek van egy piros gyermeke**: Ebben az esetben a testvért és a gyermekét átszínezzük, majd végrehajtjuk a megfelelő rotációt.

##### Példakód C++ nyelven

Az alábbiakban bemutatunk egy példakódot, amely megvalósítja a rotációkat és az újraszínezést C++ nyelven.

```cpp
void rotateLeft(Node *&root, Node *&pt) {
    Node *pt_right = pt->right;

    pt->right = pt_right->left;

    if (pt->right != nullptr)
        pt->right->parent = pt;

    pt_right->parent = pt->parent;

    if (pt->parent == nullptr)
        root = pt_right;

    else if (pt == pt->parent->left)
        pt->parent->left = pt_right;

    else
        pt->parent->right = pt_right;

    pt_right->left = pt;
    pt->parent = pt_right;
}

void rotateRight(Node *&root, Node *&pt) {
    Node *pt_left = pt->left;

    pt->left = pt_left->right;

    if (pt->left != nullptr)
        pt->left->parent = pt;

    pt_left->parent = pt->parent;

    if (pt->parent == nullptr)
        root = pt_left;

    else if (pt == pt->parent->left)
        pt->parent->left = pt_left;

    else
        pt->parent->right = pt_left;

    pt_left->right = pt;
    pt->parent = pt_left;
}

void fixInsert(Node *&root, Node *&pt) {
    Node *parent_pt = nullptr;
    Node *grand_parent_pt = nullptr;

    while ((pt != root) && (pt->color != BLACK) &&
           (pt->parent->color == RED)) {

        parent_pt = pt->parent;
        grand_parent_pt = pt->parent->parent;

        if (parent_pt == grand_parent_pt->left) {
            Node *uncle_pt = grand_parent_pt->right;

            if (uncle_pt != nullptr && uncle_pt->color == RED) {
                grand_parent_pt->color = RED;
                parent_pt->color = BLACK;
                uncle_pt->color = BLACK;
                pt = grand_parent_pt;
            } else {
                if (pt == parent_pt->right) {
                    rotateLeft(root, parent_pt);
                    pt = parent_pt;
                    parent_pt = pt->parent;
                }
                rotateRight(root, grand_parent_pt);
                swap(parent_pt->color, grand_parent_pt->color);
                pt = parent_pt;
            }
        } else {
            Node *uncle_pt = grand_parent_pt->left;

            if ((uncle_pt != nullptr) && (uncle_pt->color == RED)) {
                grand_parent_pt->color = RED;
                parent_pt->color = BLACK;
                uncle_pt->color = BLACK;
                pt = grand_parent_pt;
            } else {
                if (pt == parent_pt->left) {
                    rotateRight(root, parent_pt);
                    pt = parent_pt;
                    parent_pt = pt->parent;
                }
                rotateLeft(root, grand_parent_pt);
                swap(parent_pt->color, grand_parent_pt->color);
                pt = parent_pt;
            }
        }
    }

    root->color = BLACK;
}
```

##### Összegzés

A rotációk és újraszínezés alapvető fontosságú műveletek a piros-fekete fákban, amelyek biztosítják a fa kiegyensúlyozottságát és hatékony működését. A rotációk lehetővé teszik a csomópontok hierarchiájának átrendezését, míg az újraszínezés a színezési szabályok betartását garantálja. Ezek a műveletek együttesen biztosítják, hogy a piros-fekete fa mindig kiegyensúlyozott maradjon, és az O(log n) keresési, beszúrási és törlési idő garantált legyen.

#### 3.4.4. Összehasonlítás AVL fákkal

A piros-fekete fák és az AVL fák egyaránt kiegyensúlyozott bináris keresőfák, amelyek célja, hogy biztosítsák a keresési, beszúrási és törlési műveletek hatékonyságát. Mindkét adatszerkezet garantálja az O(log n) időbeli komplexitást, de különböző megközelítéseket alkalmaznak a kiegyensúlyozottság fenntartására. Ebben az alfejezetben részletesen összehasonlítjuk a piros-fekete fákat és az AVL fákat, bemutatva előnyeiket és hátrányaikat különböző szempontok szerint.

##### Alapfogalmak és Kiegyensúlyozottság

###### Piros-Fekete Fák

A piros-fekete fák kiegyensúlyozottságát színezési szabályokkal és rotációkkal tartják fenn. Az alábbi szabályok biztosítják a fa strukturális egyensúlyát:
1. Minden csomópont piros vagy fekete.
2. A gyökér mindig fekete.
3. Minden levél (NIL csomópont) fekete.
4. Egy piros csomópontnak mindkét gyermeke fekete.
5. Minden csomópontból bármelyik leszármazott levélig vezető úton azonos számú fekete csomópont található.

Ezek a szabályok biztosítják, hogy a fa magassága legfeljebb kétszerese legyen a logaritmusának, így a műveletek hatékonysága garantált marad.

###### AVL Fák

Az AVL fák (Adelson-Velsky és Landis fák) a magasságbeli különbségek alapján tartják fenn a kiegyensúlyozottságot. Minden csomópont esetében az alábbi szabály érvényes:
- Az egyes csomópontok bal és jobb alfa közötti magasságkülönbség legfeljebb 1 lehet.

Ez a szabály szigorúbb, mint a piros-fekete fáknál alkalmazott színezési szabályok, így az AVL fák általában jobban kiegyensúlyozottak, de a beszúrási és törlési műveletek több rotációt igényelhetnek.

##### Beszúrás és Törlés

###### Beszúrás Piros-Fekete Fában

A beszúrás során az új csomópontot piros színnel szúrjuk be, majd rotációkkal és újraszínezéssel biztosítjuk, hogy a színezési szabályok továbbra is érvényesek maradjanak. Ez az eljárás általában kevesebb rotációt igényel, mint az AVL fáknál, ami gyorsabb beszúrási műveleteket eredményezhet.

###### Beszúrás AVL Fában

Az AVL fákban a beszúrási művelet után minden érintett csomópont magasságát frissíteni kell, és szükség esetén rotációkat kell végrehajtani, hogy a kiegyensúlyozottságot fenntartsuk. Az AVL fák szigorúbb kiegyensúlyozási szabályai miatt gyakrabban szükségesek a rotációk, ami a beszúrási műveletek lassulásához vezethet, különösen nagy adathalmazok esetén.

###### Törlés Piros-Fekete Fában

A törlés után a piros-fekete fáknál rotációkat és újraszínezést kell végrehajtani a kiegyensúlyozottság fenntartása érdekében. A törlés utáni kiegyensúlyozási folyamat bonyolult lehet, de általában hatékonyan végezhető el, mivel a színezési szabályok rugalmasabbak.

###### Törlés AVL Fában

Az AVL fákban a törlés után minden érintett csomópont magasságát frissíteni kell, és szükség esetén rotációkat kell végrehajtani. A szigorú kiegyensúlyozási szabályok miatt a törlés utáni műveletek több rotációt igényelhetnek, ami lassabb műveletvégrehajtást eredményezhet.

##### Keresési Műveletek

Mind a piros-fekete fák, mind az AVL fák garantálják az O(log n) időbeli komplexitást a keresési műveletekhez. Az AVL fák általában jobban kiegyensúlyozottak, így a keresési műveletek valamivel gyorsabbak lehetnek, de a különbség gyakran elhanyagolható a gyakorlatban.

##### Memóriafelhasználás

###### Piros-Fekete Fák

A piros-fekete fák valamivel kevesebb memóriát használnak, mivel a kiegyensúlyozottság fenntartásához kevesebb rotációra van szükség, és a fa magassága általában nagyobb lehet, mint az AVL fáké.

###### AVL Fák

Az AVL fák szigorú kiegyensúlyozási szabályai miatt a fa magassága kisebb, de több rotációra és magasság frissítésére van szükség a műveletek során, ami nagyobb memóriafelhasználást eredményezhet.

##### Gyakorlati Alkalmazások

###### Piros-Fekete Fák

A piros-fekete fákat gyakran használják olyan alkalmazásokban, ahol a beszúrási és törlési műveletek gyakoriak, mivel ezek a műveletek általában kevesebb rotációt igényelnek. Például a Linux kernelben a piros-fekete fák használatosak a folyamatok ütemezéséhez és a memória kezelőben.

###### AVL Fák

Az AVL fák ideálisak olyan alkalmazásokhoz, ahol a keresési műveletek dominálnak, és a gyors hozzáférés kritikus fontosságú. Az AVL fák szigorúbb kiegyensúlyozottsága miatt a keresési műveletek általában gyorsabbak. Példák ilyen alkalmazásokra az adatbázisok indexelése és a szövegkereső rendszerek.

##### Példakód C++ nyelven

Az alábbiakban bemutatunk egy egyszerű C++ osztályt, amely megvalósít egy AVL fát:

```cpp
#include <iostream>
using namespace std;

struct Node {
    int key;
    Node *left;
    Node *right;
    int height;
};

int height(Node *N) {
    if (N == nullptr)
        return 0;
    return N->height;
}

int max(int a, int b) {
    return (a > b) ? a : b;
}

Node* newNode(int key) {
    Node* node = new Node();
    node->key = key;
    node->left = nullptr;
    node->right = nullptr;
    node->height = 1; // new node is initially added at leaf
    return(node);
}

Node *rightRotate(Node *y) {
    Node *x = y->left;
    Node *T2 = x->right;

    // Perform rotation
    x->right = y;
    y->left = T2;

    // Update heights
    y->height = max(height(y->left), height(y->right)) + 1;
    x->height = max(height(x->left), height(x->right)) + 1;

    // Return new root
    return x;
}

Node *leftRotate(Node *x) {
    Node *y = x->right;
    Node *T2 = y->left;

    // Perform rotation
    y->left = x;
    x->right = T2;

    // Update heights
    x->height = max(height(x->left), height(x->right)) + 1;
    y->height = max(height(y->left), height(y->right)) + 1;

    // Return new root
    return y;
}

int getBalance(Node *N) {
    if (N == nullptr)
        return 0;
    return height(N->left) - height(N->right);
}

Node* insert(Node* node, int key) {
    if (node == nullptr)
        return(newNode(key));

    if (key < node->key)
        node->left = insert(node->left, key);
    else if (key > node->key)
        node->right = insert(node->right, key);
    else // Equal keys are not allowed in BST
        return node;

    node->height = 1 + max(height(node->left),
                        height(node->right));

    int balance = getBalance(node);

    // If this node becomes unbalanced, then
    // there are 4 cases

    // Left Left Case
    if (balance > 1 && key < node->left->key)
        return rightRotate(node);

    // Right Right Case
    if (balance < -1 && key > node->right->key)
        return leftRotate(node);

    // Left Right Case
    if (balance > 1 && key > node->left->key) {
        node->left = leftRotate(node->left);
        return rightRotate(node);
    }

    // Right Left Case
    if (balance < -1 && key < node->right->key) {
        node->right = rightRotate(node->right);
        return leftRotate(node);
    }

    return node;
}

void preOrder(Node *root) {
    if(root != nullptr) {
        cout << root->key << " ";
        preOrder(root->left);
        preOrder(root->right);
    }
}

int main() {
    Node *root = nullptr;

    root = insert(root, 10);
    root = insert(root, 20);
    root = insert(root, 30);
    root = insert(root, 40);
    root = insert(root, 50);
    root = insert(root, 25);

    cout << "Preorder traversal of the constructed AVL tree is \n";
    preOrder(root);

    return 0;
}
```

##### Összegzés

Mind a piros-fekete fák, mind az AVL fák hatékony adatszerkezetek, amelyek biztosítják a bináris keresőfák kiegyensúlyozottságát. A piros-fekete fák rugalmasabb színezési szabályaiknak köszönhetően gyorsabb beszúrási és törlési műveleteket biztosítanak, míg az AVL fák szigorúbb kiegyensúlyozási szabályaik miatt általában jobban kiegyensúlyozottak, ami gyorsabb keresési műveleteket eredményez. A választás a kettő között az alkalmazás igényeitől és a domináns műveletektől függ.

### 3.5. B-fák és B±fák

A B-fák és B±fák az adatszerkezetek területén kiemelkedően fontos szerepet töltenek be, különösen a nagy mennyiségű adat hatékony tárolása és keresése terén. Ezek a kiegyensúlyozott fa struktúrák lehetővé teszik a gyors adatbevitel és -kiolvasást, amely kritikus fontosságú az adatbázisok és fájlrendszerek teljesítményének optimalizálásához. A fejezet során megvizsgáljuk a B-fák és B±fák alapvető fogalmait és alkalmazási területeit, részletesen bemutatjuk a velük végezhető műveleteket és azok tulajdonságait, valamint elemzést nyújtunk arról, hogyan használják ezeket a struktúrákat az adatbázisokban és fájlrendszerekben a valós életben.

#### 3.5.1. Alapfogalmak és alkalmazások

##### Alapfogalmak

A B-fák és B±fák kiegyensúlyozott fák, amelyek elsősorban adatbázisok és fájlrendszerek szervezésére használatosak. A B-fa (Balanced Tree) egy általánosított bináris keresőfa, amely több gyerekkel rendelkezhet, mint a hagyományos bináris fa. Ennek köszönhetően a B-fák sokkal jobban skálázhatók, és nagy mennyiségű adat kezelésére alkalmasak, miközben biztosítják a gyors keresést, beszúrást és törlést.

A B±fa (B-plus Tree) egy speciális típusa a B-fának, ahol az összes adat csak a levélszinteken tárolódik, és a belső csomópontok csak az irányítást szolgálják. Ez az elrendezés különösen előnyös, ha gyakori adatkereséseket kell végrehajtani, mivel a levélcsomópontok láncolt listát alkotnak, amely lehetővé teszi a hatékony szekvenciális hozzáférést.

###### B-fák szerkezete

A B-fa szerkezete a következőképpen definiálható:
- Minden csomópont legfeljebb $m$ gyerekkel rendelkezhet, ahol $m$ a fa rendje.
- Minden csomópontban legalább $\lceil m/2 \rceil$ gyerek található, kivéve a gyökércsomópontot.
- Minden belső csomópontban legalább $\lceil m/2 \rceil - 1$ és legfeljebb $m-1$ kulcs található.
- Minden levélcsomópont ugyanazon a szinten helyezkedik el.

A B±fa szerkezete hasonló a B-fáéhoz, azzal a különbséggel, hogy az összes adat csak a levélcsomópontokban tárolódik, és a belső csomópontok csak irányítást szolgálnak.

###### B-fák tulajdonságai

A B-fák és B±fák legfontosabb tulajdonságai közé tartozik:
- **Kiegyensúlyozottság:** Minden levélcsomópont ugyanazon a szinten van, ami biztosítja az O(log n) időbeli komplexitást a keresés, beszúrás és törlés műveletekhez.
- **Több kulcs tárolása:** Minden csomópont több kulcsot tárolhat, ami csökkenti a fa magasságát és növeli a hatékonyságot.
- **Hatékony disk I/O:** A csomópontok mérete gyakran egy blokk méretének felel meg a háttértárolón, ami minimalizálja a diszk I/O műveleteket.

##### Alkalmazások

###### Adatbázisok

A B-fák és B±fák széles körben alkalmazottak adatbázis-kezelő rendszerekben (DBMS) az indexelési mechanizmusokban. Az indexek olyan adatszerkezetek, amelyek lehetővé teszik a gyors keresést az adatbázisban tárolt adatok között. A B-fák előnyei az adatbázisokban a következők:
- **Gyors keresés:** Az O(log n) keresési idő lehetővé teszi az adatbázis-rekordok gyors elérését.
- **Hatékony beszúrás és törlés:** Az új rekordok beszúrása és a meglévők törlése hatékonyan elvégezhető, anélkül, hogy az egész fa átrendezésére lenne szükség.
- **Kiegyensúlyozott struktúra:** A fa kiegyensúlyozottsága biztosítja, hogy az adatbázis teljesítménye ne romoljon a beszúrások és törlések során.

A B±fák különösen népszerűek az adatbázisokban, mivel a levélcsomópontok láncolása lehetővé teszi a hatékony tartománykereséseket és szekvenciális hozzáférést.

###### Fájlrendszerek

A fájlrendszerekben a B-fák és B±fák gyakran használatosak a fájlok és könyvtárak indexelésére. Ez lehetővé teszi a gyors fájlkeresést és az attribútumok gyors elérését. Például az NTFS (New Technology File System) fájlrendszer a B±fákat használja a fájlok és könyvtárak indexelésére, míg az ext4 fájlrendszer szintén hasonló adatstruktúrákat alkalmaz.

##### Példa B±fa implementáció C++ nyelven

Az alábbiakban bemutatunk egy egyszerű B±fa implementációt C++ nyelven:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

class BPlusTreeNode {
public:
    std::vector<int> keys;
    std::vector<BPlusTreeNode*> children;
    bool isLeaf;

    BPlusTreeNode(bool leaf);
    void insertNonFull(int key);
    void splitChild(int i, BPlusTreeNode* y);
    void traverse();

    friend class BPlusTree;
};

class BPlusTree {
public:
    BPlusTreeNode* root;
    int t; // Minimum degree

    BPlusTree(int _t) {
        root = nullptr;
        t = _t;
    }

    void traverse() {
        if (root != nullptr) root->traverse();
    }

    void insert(int key);
};

BPlusTreeNode::BPlusTreeNode(bool leaf) {
    isLeaf = leaf;
}

void BPlusTree::insert(int key) {
    if (root == nullptr) {
        root = new BPlusTreeNode(true);
        root->keys.push_back(key);
    } else {
        if (root->keys.size() == 2*t - 1) {
            BPlusTreeNode* s = new BPlusTreeNode(false);
            s->children.push_back(root);
            s->splitChild(0, root);
            int i = 0;
            if (s->keys[0] < key) i++;
            s->children[i]->insertNonFull(key);
            root = s;
        } else {
            root->insertNonFull(key);
        }
    }
}

void BPlusTreeNode::insertNonFull(int key) {
    int i = keys.size() - 1;
    if (isLeaf) {
        keys.push_back(0);
        while (i >= 0 && keys[i] > key) {
            keys[i + 1] = keys[i];
            i--;
        }
        keys[i + 1] = key;
    } else {
        while (i >= 0 && keys[i] > key) i--;
        if (children[i + 1]->keys.size() == 2*t - 1) {
            splitChild(i + 1, children[i + 1]);
            if (keys[i + 1] < key) i++;
        }
        children[i + 1]->insertNonFull(key);
    }
}

void BPlusTreeNode::splitChild(int i, BPlusTreeNode* y) {
    BPlusTreeNode* z = new BPlusTreeNode(y->isLeaf);
    z->keys.resize(t - 1);
    for (int j = 0; j < t - 1; j++)
        z->keys[j] = y->keys[j + t];
    if (!y->isLeaf) {
        z->children.resize(t);
        for (int j = 0; j < t; j++)
            z->children[j] = y->children[j + t];
    }
    y->keys.resize(t - 1);
    children.insert(children.begin() + i + 1, z);
    keys.insert(keys.begin() + i, y->keys[t - 1]);
}

void BPlusTreeNode::traverse() {
    int i;
    for (i = 0; i < keys.size(); i++) {
        if (!isLeaf) children[i]->traverse();
        std::cout << " " << keys[i];
    }
    if (!isLeaf) children[i]->traverse();
}

int main() {
    BPlusTree t(3);
    t.insert(10);
    t.insert(20);
    t.insert(5);
    t.insert(6);
    t.insert(12);
    t.insert(30);
    t.insert(7);
    t.insert(17);

    std::cout << "Traversal of the constructed tree is ";
    t.traverse();

    return 0;
}
```

##### Összefoglalás

A B-fák és B±fák alapfogalmainak és alkalmazásainak részletes bemutatása során láthattuk, hogy ezek az adatstruktúrák kulcsszerepet játszanak az adatbázisok és fájlrendszerek hatékony működésében. A kiegyensúlyozott fa struktúrák lehetővé teszik a gyors keresést, beszúrást és törlést, valamint a hatékony diszk I/O műveleteket. Ezek az előnyök teszik a B-fákat és B±fákat elengedhetetlen eszközökké a nagy mennyiségű adat kezelésére és tárolására.

#### 3.5.2. Műveletek és tulajdonságok

##### Bevezetés

A B-fák és B±fák műveletei és tulajdonságai kulcsfontosságúak az adatstruktúrák megértésében és hatékony alkalmazásában. Ezek a műveletek biztosítják, hogy a fák kiegyensúlyozottak maradjanak, lehetővé téve az optimális keresési, beszúrási és törlési műveleteket. Ebben a fejezetben részletesen megvizsgáljuk a B-fák és B±fák legfontosabb műveleteit, valamint azok tulajdonságait.

##### Műveletek

###### Keresés

A keresés egy alapvető művelet a B-fákban és B±fákban, amelynek célja egy adott kulcs megtalálása a fában. A keresés folyamatát a következőképpen írhatjuk le:

1. **Gyökérből indulás:** A keresést mindig a gyökércsomópontból indítjuk.
2. **Kulcsok összehasonlítása:** Az aktuális csomópontban található kulcsokkal összehasonlítjuk a keresett kulcsot.
3. **Leágazás:** Ha a keresett kulcs kisebb, mint a vizsgált kulcs, akkor a bal oldali gyerek csomóponthoz megyünk; ha nagyobb, akkor a jobb oldalihoz.
4. **Rekurzió:** Ezt a folyamatot ismételjük, amíg meg nem találjuk a keresett kulcsot, vagy el nem érünk egy levélcsomópontot, ahol a kulcs nem található.

A keresés időbeli komplexitása O(log n), mivel minden lépésben a fa magasságát követjük, amely logaritmikusan növekszik a fa méretével.

###### Beszúrás

A beszúrási művelet célja egy új kulcs hozzáadása a B-fához vagy B±fához. A beszúrás folyamatát a következőképpen írhatjuk le:

1. **Kulcs helyének meghatározása:** Az új kulcs helyét a keresési művelettel határozzuk meg.
2. **Beszúrás levélcsomópontba:** Az új kulcsot az aktuális levélcsomópontba szúrjuk be.
3. **Csomópont osztása:** Ha a beszúrás után a csomópont kulcsainak száma meghaladja a megengedett maximumot, akkor a csomópontot két részre osztjuk, és a középső kulcsot a szülő csomópontba emeljük fel.

A beszúrás időbeli komplexitása O(log n), mivel a beszúrás helyének meghatározása és a csomópont osztása logaritmikus időben történik.

###### Törlés

A törlés egy bonyolultabb művelet, amely egy adott kulcs eltávolítását célozza a B-fából vagy B±fából. A törlés folyamatát a következőképpen írhatjuk le:

1. **Kulcs keresése:** A törölni kívánt kulcs helyének meghatározása a keresési művelettel.
2. **Levélcsomópontból törlés:** Ha a kulcs egy levélcsomópontban található, egyszerűen eltávolítjuk onnan.
3. **Belső csomópontból törlés:** Ha a kulcs egy belső csomópontban található, két lehetőség van:
   - **Előző vagy következő kulcs:** Kicseréljük a törlendő kulcsot az előző vagy következő levélcsomópontban található kulccsal, majd töröljük azt a levélcsomópontból.
   - **Merge:** Ha a szomszédos csomópontokban lévő kulcsok száma nem elegendő, akkor egyesítjük a csomópontokat, és az egyesített csomópontból töröljük a kulcsot.

A törlés időbeli komplexitása szintén O(log n), mivel a keresési és átrendezési műveletek logaritmikus időben történnek.

##### Tulajdonságok

###### Magasság és kiegyensúlyozottság

A B-fák és B±fák egyik legfontosabb tulajdonsága a kiegyensúlyozottság. Minden levélcsomópont ugyanazon a szinten helyezkedik el, ami biztosítja, hogy a fa magassága O(log n) legyen, ahol n a fa elemeinek száma. Ez a tulajdonság biztosítja, hogy a keresési, beszúrási és törlési műveletek mindig hatékonyak maradjanak.

###### Kulcsok és csomópontok száma

A B-fák és B±fák csomópontjaiban található kulcsok száma korlátozott. Egy m rendű B-fában minden csomópont legfeljebb m-1 kulcsot tartalmazhat, és legalább $\lceil m/2 \rceil - 1$ kulcsot kell tartalmaznia (kivéve a gyökércsomópontot, amely lehet kevesebb kulccsal is). Ez a korlátozás biztosítja, hogy a fa mindig kiegyensúlyozott maradjon.

###### Disk I/O optimalizálás

A B-fák és B±fák kialakítása optimalizálja a diszk I/O műveleteket, ami különösen fontos a nagy adatmennyiséggel dolgozó rendszerekben, mint például az adatbázisok és fájlrendszerek. A csomópontok mérete gyakran egy blokk méretének felel meg a háttértárolón, ami minimalizálja a szükséges diszk I/O műveletek számát.

##### Példa B±fa implementáció C++ nyelven

Az alábbiakban bemutatunk egy részletes példát a B±fa implementációjára C++ nyelven, amely tartalmazza a keresési, beszúrási és törlési műveleteket.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

class BPlusTreeNode {
public:
    std::vector<int> keys;
    std::vector<BPlusTreeNode*> children;
    bool isLeaf;

    BPlusTreeNode(bool leaf);
    void insertNonFull(int key);
    void splitChild(int i, BPlusTreeNode* y);
    void traverse();
    BPlusTreeNode* search(int key);

    friend class BPlusTree;
};

class BPlusTree {
public:
    BPlusTreeNode* root;
    int t; // Minimum degree

    BPlusTree(int _t) {
        root = nullptr;
        t = _t;
    }

    void traverse() {
        if (root != nullptr) root->traverse();
    }

    BPlusTreeNode* search(int key) {
        return (root == nullptr) ? nullptr : root->search(key);
    }

    void insert(int key);
    void remove(int key);
};

BPlusTreeNode::BPlusTreeNode(bool leaf) {
    isLeaf = leaf;
}

void BPlusTree::insert(int key) {
    if (root == nullptr) {
        root = new BPlusTreeNode(true);
        root->keys.push_back(key);
    } else {
        if (root->keys.size() == 2*t - 1) {
            BPlusTreeNode* s = new BPlusTreeNode(false);
            s->children.push_back(root);
            s->splitChild(0, root);
            int i = 0;
            if (s->keys[0] < key) i++;
            s->children[i]->insertNonFull(key);
            root = s;
        } else {
            root->insertNonFull(key);
        }
    }
}

void BPlusTreeNode::insertNonFull(int key) {
    int i = keys.size() - 1;
    if (isLeaf) {
        keys.push_back(0);
        while (i >= 0 && keys[i] > key) {
            keys[i + 1] = keys[i];
            i--;
        }
        keys[i + 1] = key;
    } else {
        while (i >= 0 && keys[i] > key) i--;
        if (children[i + 1]->keys.size() == 2*t - 1) {
            splitChild(i + 1, children[i + 1]);
            if (keys[i + 1] < key) i++;
        }
        children[i + 1]->insertNonFull(key);
    }
}

void BPlusTreeNode::splitChild(int i, BPlusTreeNode* y) {
    BPlusTreeNode* z = new BPlusTreeNode(y->isLeaf);
    z->keys.resize(t - 1);
    for (int j = 0; j < t - 1; j++)
        z->keys[j] = y->keys[j + t];
    if (!y->isLeaf) {
        z->children.resize(t);
        for (int j = 0; j < t; j++)
            z->children[j] = y->children[j + t];
    }
    y->keys.resize(t - 1);
    children.insert(children.begin() + i + 1, z);
    keys.insert(keys.begin() + i, y->keys[t - 1]);
}

BPlusTreeNode* BPlusTreeNode::search(int key) {
    int i = 0;
    while (i < keys.size() && key > keys[i]) i++;
    if (i < keys.size() && keys[i] == key) return this;
    if (isLeaf) return nullptr;
    return children[i]->search(key);
}

void BPlusTreeNode::traverse() {
    int i;
    for (i = 0; i < keys.size(); i++) {
        if (!isLeaf) children[i]->traverse();
        std::cout << " " << keys[i];
    }
    if (!isLeaf) children[i]->traverse();
}

int main() {
    BPlusTree t(3);
    t.insert(10);
    t.insert(20);
    t.insert(5);
    t.insert(6);
    t.insert(12);
    t.insert(30);
    t.insert(7);
    t.insert(17);

    std::cout << "Traversal of the constructed tree is ";
    t.traverse();

    int key = 6;
    (t.search(key) != nullptr) ? std::cout << "\nPresent" : std::cout << "\nNot Present";

    key = 15;
    (t.search(key) != nullptr) ? std::cout << "\nPresent" : std::cout << "\nNot Present";

    return 0;
}
```

##### Összefoglalás

A B-fák és B±fák műveleteinek és tulajdonságainak részletes bemutatása során láthattuk, hogy ezek az adatstruktúrák rendkívül hatékonyak a keresés, beszúrás és törlés műveletekben. A kiegyensúlyozott szerkezet és a diszk I/O optimalizálás különösen fontos szerepet játszik a nagy adatmennyiséggel dolgozó rendszerek teljesítményének javításában. A részletes példakód bemutatásával is szemléltettük ezen műveletek gyakorlati megvalósítását.

#### 3.5.3. Használat adatbázisokban és fájlrendszerekben

##### Bevezetés

A B-fák és B±fák használata az adatbázisokban és fájlrendszerekben kritikus fontosságú, mivel ezek az adatszerkezetek hatékonyan támogatják a nagy mennyiségű adat kezelését, tárolását és gyors elérését. Az alábbiakban részletesen bemutatjuk, hogyan használják ezeket az adatszerkezeteket különböző rendszerekben, és miért váltak az adatbázis-kezelők és fájlrendszerek alapvető eszközeivé.

##### Használat adatbázisokban

###### Indexelés

Az adatbázisokban az indexek alapvető fontosságúak a gyors adatkeresés érdekében. A B-fák és B±fák az egyik leggyakrabban használt adatstruktúrák az indexeléshez, mivel képesek hatékonyan kezelni a beszúrásokat, törléseket és kereséseket.

1. **B-fák az indexelésben:**
   - A B-fák lehetővé teszik az adatbázis-rekordok gyors elérését a kulcsok alapján.
   - Mivel a B-fák kiegyensúlyozottak, a keresési idő O(log n), ami jelentősen csökkenti az adatok keresésének idejét.
   - Az adatbázis-kezelő rendszerek gyakran használják a B-fákat a primer és szekunder indexek létrehozásához.

2. **B±fák az indexelésben:**
   - A B±fák különösen hasznosak az adatbázisokban, mert minden adat csak a levélcsomópontokban van tárolva, és a belső csomópontok csak útvonalakat tartalmaznak.
   - A levélcsomópontok láncolt listát alkotnak, ami lehetővé teszi a hatékony tartománykeresést és szekvenciális hozzáférést.
   - Az Oracle és a MySQL adatbázis-kezelő rendszerek is B±fákat használnak az indexek megvalósításához.

###### Adatbázis műveletek

A B-fák és B±fák hatékonyan támogatják az adatbázis-kezelő rendszerek alapvető műveleteit:

1. **Keresés:**
   - Az adatbázisokban végzett keresési műveletek során a B-fák gyorsan megtalálják a keresett rekordot a kulcsok segítségével.
   - A B±fák előnye a gyors szekvenciális hozzáférés, amely lehetővé teszi a tartománykeresést és az adatbázis táblák rendezett bejárását.

2. **Beszúrás:**
   - Az új rekordok beszúrása a B-fák és B±fák esetében hatékonyan történik a fa kiegyensúlyozottságának fenntartása mellett.
   - Az adatbázis-kezelők automatikusan átrendezik a fákat, ha a csomópontok telítődnek, biztosítva ezzel a kiegyensúlyozott szerkezetet.

3. **Törlés:**
   - A rekordok törlése szintén hatékonyan valósítható meg a B-fák és B±fák segítségével.
   - Az adatbázis-kezelők gondoskodnak arról, hogy a törlés után a fa kiegyensúlyozott maradjon, és a csomópontok ne váljanak túl telítettekké vagy túl üressé.

##### Használat fájlrendszerekben

A fájlrendszerekben a B-fák és B±fák szintén kulcsszerepet játszanak, különösen a fájlok és könyvtárak kezelésében.

###### Fájlrendszer struktúra

1. **NTFS fájlrendszer:**
   - Az NTFS (New Technology File System), amelyet a Microsoft fejlesztett ki, B±fákat használ a fájlok és könyvtárak indexelésére.
   - A B±fák lehetővé teszik a fájlok gyors keresését és elérését, valamint a metaadatok hatékony kezelését.
   - Az NTFS-ben minden könyvtár egy B±fa, amely a fájlok nevét és metaadatait tárolja.

2. **ext4 fájlrendszer:**
   - Az ext4 fájlrendszer, amely az egyik legnépszerűbb Linux fájlrendszer, szintén B-fákat használ a fájlok és könyvtárak kezelésére.
   - Az ext4-ben a B-fák segítenek a gyors hozzáférés biztosításában és a fájlrendszer teljesítményének optimalizálásában.

###### Fájlrendszer műveletek

A fájlrendszerekben végzett alapvető műveletek során a B-fák és B±fák hatékonyan támogatják a következőket:

1. **Fájlkeresés:**
   - A fájlrendszerekben végzett fájlkeresés során a B±fák gyorsan megtalálják a keresett fájlt a könyvtárakban.
   - A B±fák láncolt levélcsomópontjai lehetővé teszik a gyors tartománykeresést és a fájlok szekvenciális hozzáférését.

2. **Fájlbeszúrás:**
   - Az új fájlok beszúrása a könyvtárakba hatékonyan történik a B±fák használatával.
   - A fájlrendszer automatikusan átrendezi a fákat, ha a csomópontok telítődnek, biztosítva ezzel a kiegyensúlyozott szerkezetet és a gyors hozzáférést.

3. **Fájltörlés:**
   - A fájlok törlése szintén hatékonyan valósítható meg a B±fák segítségével.
   - A fájlrendszer gondoskodik arról, hogy a törlés után a fa kiegyensúlyozott maradjon, és a csomópontok ne váljanak túl telítettekké vagy túl üressé.

##### Gyakorlati példák és implementáció

Az alábbiakban bemutatunk néhány gyakorlati példát és implementációt, amelyek bemutatják a B-fák és B±fák használatát az adatbázisokban és fájlrendszerekben.

###### Példa: B±fa használata az NTFS fájlrendszerben

Az NTFS fájlrendszerben a könyvtárstruktúra B±fákat használ a fájlok és könyvtárak indexelésére. Az alábbiakban egy egyszerű példa C++ nyelven, amely bemutatja, hogyan használhatjuk a B±fákat fájlrendszeri műveletekhez.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

class BPlusTreeNode {
public:
    std::vector<std::string> keys;
    std::vector<BPlusTreeNode*> children;
    bool isLeaf;

    BPlusTreeNode(bool leaf);
    void insertNonFull(const std::string& key);
    void splitChild(int i, BPlusTreeNode* y);
    void traverse();
    BPlusTreeNode* search(const std::string& key);

    friend class BPlusTree;
};

class BPlusTree {
public:
    BPlusTreeNode* root;
    int t; // Minimum degree

    BPlusTree(int _t) {
        root = nullptr;
        t = _t;
    }

    void traverse() {
        if (root != nullptr) root->traverse();
    }

    BPlusTreeNode* search(const std::string& key) {
        return (root == nullptr) ? nullptr : root->search(key);
    }

    void insert(const std::string& key);
};

BPlusTreeNode::BPlusTreeNode(bool leaf) {
    isLeaf = leaf;
}

void BPlusTree::insert(const std::string& key) {
    if (root == nullptr) {
        root = new BPlusTreeNode(true);
        root->keys.push_back(key);
    } else {
        if (root->keys.size() == 2*t - 1) {
            BPlusTreeNode* s = new BPlusTreeNode(false);
            s->children.push_back(root);
            s->splitChild(0, root);
            int i = 0;
            if (s->keys[0] < key) i++;
            s->children[i]->insertNonFull(key);
            root = s;
        } else {
            root->insertNonFull(key);
        }
    }
}

void BPlusTreeNode::insertNonFull(const std::string& key) {
    int i = keys.size() - 1;
    if (isLeaf) {
        keys.push_back("");
        while (i >= 0 && keys[i] > key) {
            keys[i + 1] = keys[i];
            i--;
        }
        keys[i + 1] = key;
    } else {
        while (i >= 0 && keys[i] > key) i--;
        if (children[i + 1]->keys.size() == 2*t - 1) {
            splitChild(i + 1, children[i + 1]);
            if (keys[i + 1] < key) i++;
        }
        children[i + 1]->insertNonFull(key);
    }
}

void BPlusTreeNode::splitChild(int i, BPlusTreeNode* y) {
    BPlusTreeNode* z = new BPlusTreeNode(y->isLeaf);
    z->keys.resize(t - 1);
    for (int j = 0; j < t - 1; j++)
        z->keys[j] = y->keys[j + t];
    if (!y->isLeaf) {
        z->children.resize(t);
        for (int j = 0; j < t; j++)
            z->children[j] = y->children[j + t];
    }
    y->keys.resize(t - 1);
    children.insert(children.begin() + i + 1, z);
    keys.insert(keys.begin() + i, y->keys[t - 1]);
}

BPlusTreeNode* BPlusTreeNode::search(const std::string& key) {
    int i = 0;
    while (i < keys.size() && key > keys[i]) i++;
    if (i < keys.size() && keys[i] == key) return this;
    if (isLeaf) return nullptr;
    return children[i]->search(key);
}

void BPlusTreeNode::traverse() {
    int i;
    for (i = 0; i < keys.size(); i++) {
        if (!isLeaf) children[i]->traverse();
        std::cout << " " << keys[i];
    }
    if (!isLeaf) children[i]->traverse();
}

int main() {
    BPlusTree t(3);
    t.insert("file1.txt");
    t.insert("file2.txt");
    t.insert("file3.txt");
    t.insert("file4.txt");
    t.insert("file5.txt");

    std::cout << "Traversal of the constructed tree is ";
    t.traverse();

    std::string key = "file3.txt";
    (t.search(key) != nullptr) ? std::cout << "\nPresent" : std::cout << "\nNot Present";

    key = "file6.txt";
    (t.search(key) != nullptr) ? std::cout << "\nPresent" : std::cout << "\nNot Present";

    return 0;
}
```

##### Összefoglalás

A B-fák és B±fák használata az adatbázisokban és fájlrendszerekben jelentős előnyökkel jár, mivel lehetővé teszik a nagy mennyiségű adat hatékony kezelését, tárolását és gyors elérését. Az adatbázisokban ezek az adatszerkezetek biztosítják a gyors keresést, beszúrást és törlést, míg a fájlrendszerekben segítik a fájlok és könyvtárak rendezett tárolását és gyors elérését. Az előzőekben bemutatott példák és implementációk részletesen szemléltették, hogyan használhatjuk a B-fákat és B±fákat a gyakorlatban, megerősítve azok jelentőségét és hatékonyságát.