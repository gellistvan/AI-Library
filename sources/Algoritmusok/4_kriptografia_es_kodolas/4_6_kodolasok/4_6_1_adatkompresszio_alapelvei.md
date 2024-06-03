\newpage

# 4.6. Kódolási algoritmusok

A kódolási algoritmusok világa kifinomult technikák és eljárások sokaságát öleli fel, amelyek célja az adatok hatékony és biztonságos átalakítása. E szekcióban bemutatjuk a legfontosabb kódolási algoritmusokat, amelyek a modern kommunikáció és információ-technológia alapját képezik. Az általánosan ismert eljárásoktól, mint például a Huffman-kódolás és a Run-Length Encoding (RLE), egészen a bonyolultabb adaptív és paritásellenőrző kódokig, átfogó képet kaphatunk ezen algoritmusok működéséről és alkalmazási területeiről. A kódolási technikák megismerése elengedhetetlen ahhoz, hogy értékelni tudjuk, miként járulnak hozzá a hatékony adatátvitelhez, a tárhely optimalizálásához és az adatbiztonság megőrzéséhez a digitális korban.

## 4.6.1. Adatkompresszió alapelvei

Adatkompresszióval nap mint nap találkozunk, legyen szó fájlok tárolásáról, adatátvitelről vagy éppen multimédiás tartalmak feldolgozásáról. A kompresszió célja, hogy az eredeti adatokat olyan formátumra alakítsa, amely kevesebb tárhelyet igényel, illetve gyorsabban továbbítható. Ebben a fejezetben megismerkedünk az adatkompresszió alapelveivel, beleértve a veszteségmentes és veszteséges kompressziós technikák közötti különbségeket. A továbbiakban feltárjuk az entropia és az információelmélet alapjait, amelyek kulcsfontosságú szerepet játszanak a hatékony kompressziós algoritmusok megértésében és fejlesztésében. Az alapelvek tisztázása után rátérünk néhány elterjedt kódolási algoritmus részleteire is, bemutatva azok működési mechanizmusait és alkalmazási területeit.

### Veszteségmentes és veszteséges kompresszió

Adatkompresszió a digitális adatok tárolásának és továbbításának egyik alapvető technikája. A cél a redundancia csökkentése a kódolt adatokban, miközben az eredeti információ minél nagyobb arányban megőrizhető. A kompresszió két alapvető kategóriába sorolható: veszteségmentes és veszteséges.

#### Veszteségmentes kompresszió

A veszteségmentes kompresszió célja, hogy az eredeti adatokat úgy tömörítse, hogy azok a dekódolás után pontosan visszaállíthatók legyenek. Ez különösen fontos a különböző típusú adatfájlok, például szöveges dokumentumok, programfájlok, illetve egyes adatbázisok esetében, ahol az adat integritása kritikus.

Az egyik leggyakrabban használt veszteségmentes kompressziós algoritmus a Huffman-kódolás, amelyet David A. Huffman fejlesztett ki 1952-ben. A Huffman-kódolás egy karakter alapú kódolási eljárás, amely a ritkábban előforduló karakterekhez hosszabb kódokat, míg a gyakrabban előfordulókhoz rövidebb kódokat rendel.

Példa Huffman-kódolásra C++ nyelven:

```cpp
#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>

using namespace std;

// Trie node
struct Node {
    char ch;
    int freq;
    Node* left;
    Node* right;
};

// Comparison object to be used to order the heap
struct comp {
    bool operator()(Node* l, Node* r) {
        return l->freq > r->freq;
    }
};

// Utility function to check if Huffman Tree contains only a single node
bool isLeaf(Node* root) {
    return root->left == nullptr && root->right == nullptr;
}

// Traverse the Huffman Tree and store Huffman Codes in a map
void encode(Node* root, string str, unordered_map<char, string>& huffmanCode) {
    if (root == nullptr)
        return;

    // found a leaf node
    if (isLeaf(root)) {
        huffmanCode[root->ch] = str;
    }

    encode(root->left, str + "0", huffmanCode);
    encode(root->right, str + "1", huffmanCode);
}

// Build Huffman Tree and decode given input text
void buildHuffmanTree(string text) {
    // count frequency of appearance of each character
    // and store it in a map
    unordered_map<char, int> freq;
    for (char ch : text) {
        freq[ch]++;
    }

    // Create a priority queue to store live nodes of Huffman Tree
    priority_queue<Node*, vector<Node*>, comp> pq;

    // Create a leaf node for each character and add it to the priority queue.
    for (auto pair : freq) {
        pq.push(new Node{pair.first, pair.second, nullptr, nullptr});
    }

    // do till there is more than one node in the queue
    while (pq.size() != 1) {
        // Remove the two nodes of highest priority
        // (lowest frequency) from the queue
        Node* left = pq.top(); pq.pop();
        Node* right = pq.top(); pq.pop();

        // Create a new internal node with these two nodes as children
        // and with frequency equal to the sum of the two nodes' frequencies.
        pq.push(new Node{'\0', left->freq + right->freq, left, right});
    }

    // `root` stores pointer to root of Huffman Tree
    Node* root = pq.top();

    // Traverse the Huffman Tree and store Huffman Codes
    // in a map. Also print them
    unordered_map<char, string> huffmanCode;
    encode(root, "", huffmanCode);

    cout << "Huffman Codes are :\n" << '\n';
    for (auto pair : huffmanCode) {
        cout << pair.first << " " << pair.second << '\n';
    }

    cout << "\nOriginal string was :\n" << text << '\n';
}

// Huffman coding algorithm implementation in C++ 
int main() {
    string text = "Huffman coding is a lossless data compression algorithm.";
    buildHuffmanTree(text);

    return 0;
}
```

Másik példa veszteségmentes tömörítésre a Run-Length Encoding (RLE), amely különösen hatékony az ismétlődő karakterekből álló adatok esetén.

#### Veszteséges kompresszió

A veszteséges kompresszió lényege, hogy az eredeti adatot úgy tömöríti, hogy egyes részei elhagyhatók vagy leegyszerűsíthetők legyenek a tárolási vagy továbbítási igények csökkentése céljából. Ez a fajta kompresszió elfogadható bizonyos médiatípusoknál, mint például a képek, videók és hangok, ahol az emberi érzékelés toleránsabb bizonyos fokú torzításokra vagy részletvesztésre.

A JPEG egy elterjedt veszteséges képkompressziós algoritmus, amely a képadatokat diszkrét koszinusz transzformációval (DCT) alakítja át. A DCT transzformáció segítségével a kép energiája egy kis számban koefficiensekben koncentrálódik, és ezeket a koefficienseket kvantálják, majd végül entropikus kódolással tömörítik.

A veszteséges kompressziós algoritmusok egyik kihívása az 'Artifact', azaz a nem kívánt, a tömörítésből adódó torzítások megjelenése. Például:

- Blokkolás: Négyzetes minták a JPEG képeknél, alacsony bitráták esetén.
- Csengés: Visszhangszerű artefakt a hangoknál.

Ezek az artefaktok különösen megjelenhetnek nagyon alacsony kompresszió használatával és általában elkerülhetők bizonyos paraméterek finombeállításával.

### Entropia és információelmélet alapjai

Az információelmélet az a tudományág, amely az információ mértékeinek matematikai modellezésével, átvitelével és tárolásával foglalkozik. Az információelmélet alapjait Claude Shannon rakta le 1948-ban publikált híres cikkében, amelyben bevezette az entropia és az információ fogalmát az adatkommunikáció kontextusába. Az entropia, mint központi fogalom, a rendezetlenség vagy a bizonytalanság mértékét jelenti egy információs forrásban. Ebben a fejezetben alaposan megvizsgáljuk az entropia és az információelmélet alapjait, és hogyan alkalmazhatók ezek az elvek adatkompresszióban.

#### Entropia definíciója

Az entropia egy valószínűségi változó bizonytalanságának mértéke. Matematikailag az entropia \(H(X)\) egy diszkrét valószínűségi változó \(X\) esetén a következőképpen definiálható:

\[ H(X) = - \sum_{i=1}^n p(x_i) \log_2 p(x_i) \]

ahol \(p(x_i)\) az \(x_i\) esemény bekövetkezésének valószínűsége, és a logaritmus bázisa 2, ami bitben mérhető információt ad meg. Az entropia tehát azt méri, hogy átlagosan mennyi információ szükséges egy esemény kimenetelének meghatározásához.

#### Az entropia tulajdonságai

1. **Non-negativitás**: Az entropia mindig nem negatív, \(H(X) \geq 0\). A minimum értéket, nullát, akkor éri el, ha az esemény kimenetele teljesen determinisztikus, azaz nincs bizonytalanság (például egy érme, amely mindig fej lenne).

2. **Maximális entropia**: Az entropia maximális, ha az események egyenlő valószínűségűek. Például egy tökéletesen kiegyensúlyozott érme dobásakor, ahol a fej és az írás valószínűsége egyaránt 0,5, az entropia maximális értéke 1 bit.

3. **Additivitás**: Két független valószínűségi változó esetén az összetett rendszer entropiája az egyes változók entropiáinak összege:

\[ H(X, Y) = H(X) + H(Y) \]

#### Közös entropia és feltételes entropia

Az entropia fogalmát kiterjeszthetjük két vagy több valószínűségi változóra is. A közös entropia \(H(X, Y)\) méri a bizonytalanságot mindkét változó együttes figyelembevételével. Matematikailag:

\[ H(X, Y) = - \sum_{x \in X} \sum_{y \in Y} p(x, y) \log_2 p(x, y) \]

A feltételes entropia \(H(Y|X)\) az \(Y\) bizonytalanságát méri, feltételezve, hogy \(X\) ismert. Ez kifejezhető a következőképpen:

\[ H(Y|X) = - \sum_{x \in X} p(x) \sum_{y \in Y} p(y|x) \log_2 p(y|x) \]

A feltételes entropia mutatja, hogy mennyi információ szükséges \(Y\) ismeretéhez, ha \(X\)-et már ismertnek tekintjük.

#### Kölcsönös információ és Kullback-Leibler divergencia

A kölcsönös információ \(I(X; Y)\) két változó közötti információ megosztásának mértéke. Ez azt méri, hogy mennyi újdonságot nyújt \(X\) ismerete \(Y\)-ról, és viszont:

\[ I(X; Y) = H(X) + H(Y) - H(X, Y) \]

A Kullback-Leibler divergencia vagy relatív entropia \(D_{KL}(P||Q)\) egy mérték, amely a \(P\) valószínűségi eloszlás és a \(Q\) valószínűségi eloszlás közötti különbséget méri:

\[ D_{KL}(P||Q) = \sum_{i} P(i) \log_2 \left( \frac{P(i)}{Q(i)} \right) \]

Ez a mérőszám fontos szerepet játszik az információelmélet és a gyakorlati alkalmazások, például a gépi tanulás és a statisztika területén.

#### Adatkompresszió Shannon-tételének fényében

Claude Shannon híres második tétele, más néven Shannon's Source Coding Theorem, az adatkompresszió elméleti alapjait fektette le. A tétel kimondja, hogy egy diszkrét forrásból származó információt hatékonyan lehet kódolni egy olyan átlagos bitsorozathosszal, ami megközelíti a forrás entropiáját.

Formálisan, ha \(X\) egy valószínűségi változó, amely \(n\) különböző kimenetellel rendelkezik, és minden kimenetel valószínűsége \(p(x_i)\), akkor az ezekhez tartozó optimális kódszavak átlagos hossza (L) kielégíti az alábbi egyenlőséget:

\[ H(X) \leq L < H(X) + 1 \]

Más szóval az entropia szolgálhat az indikátoraként annak, hogy milyen mértékben tudjuk tömöríteni a forrás kimenetét veszteségmentesen.

#### Gyakorlati kódolási algoritmusok és az entropia

A gyakorlatban számos kódolási algoritmus használja az entropia elveit a tömörítésre:

1. **Huffman-kódolás**: Egy gyakran használt veszteségmentes kódolási módszer, amely egy bináris fát épít a forrás eloszlása alapján, és a leggyakrabban előforduló szimbólumokhoz rövidebb kódszavakat rendel, míg a ritkábbakhoz hosszabbakat. Az optimális Huffman-kódolás átlagos hossza nagyon közel van a forrás entropiájához.

2. **Arithmetic Coding**: Egy másik veszteségmentes tömörítési technika, amellyel egy valós számot rendelünk az olvasandó szimbólumok sorozatához. Ezzel az eljárással nagyon magas tömörítési arány érhető el, különösen akkor, ha a szimbólumok valószínűségi eloszlása egyenletes.

3. **Lempel-Ziv-Welch (LZW)**: Az LZW tömörítési algoritmus adaptív és veszteségmentes, amely növekvő méretű szótárakat használ, hogy a szimbólum sorozatokat rövidebb kódszavakkal helyettesítse.

#### Összefoglalás

Az entropia és az információelmélet alapelvei kulcsfontosságúak az adatkompresszió megértéséhez és alkalmazásához. Az entropia méri a bizonytalanság mértékét egy forrásban, és a Shannon's Source Coding Theorem elméleti felső korlátot ad arra nézve, hogy mennyire lehet hatékonyan tömöríteni az információt. A gyakorlati kódolási algoritmusok, mint a Huffman-kódolás, az Arithmetic Coding, és az LZW, ezen elvek felhasználásával képesek hatékony adatkompresszióra. Az entropia és a hozzá kapcsolódó fogalmak, mint a közös entropia, a feltételes entropia és a kölcsönös információ, további mélységet adnak, és lehetővé teszik az adatok finomabb analizálását és feldolgozását.

Az entropia és az információelmélet alapos megértése, valamint ezek alkalmazása a kódolási algoritmusokban döntő fontosságú mind az elméleti kutatók, mind a gyakorlati mérnökök számára, akik hatékony és eredményes adatkompressziós rendszereket kívánnak kialakítani.

