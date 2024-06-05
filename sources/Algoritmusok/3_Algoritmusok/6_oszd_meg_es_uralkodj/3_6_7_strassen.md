\newpage

## 6.7. Strassen algoritmus (Mátrix szorzás)

Az Oszd-meg-és-uralkodj módszerek családjában különösen kiemelkedő helyet foglal el a Strassen algoritmus, amely forradalmasította a mátrixok szorzásának területét. A hagyományos módszerekhez képest jelentősége abban rejlik, hogy képes csökkenteni a szükséges számítási lépések számát, ezzel gyorsítva a folyamatot nagyobb méretű mátrixok esetén. Ez a rész mélyebb betekintést nyújt a Strassen algoritmus működésébe, annak lépéseibe, és könnyen érthető példákkal illusztrálja az alkalmazását. Ezen felül részletesen elemzi az algoritmus teljesítményét és bemutatja, milyen gyakorlati problémák megoldásában juthat fontos szerephez. A Strassen algoritmus megértése hozzásegíti az olvasót ahhoz, hogy jobban lássa, hogyan használható az Oszd-meg-és-uralkodj paradigmája a hatékonyság növelésére a mátrix műveletekben.

### 6.7.1 Algoritmus és implementáció

Az Oszd-meg-és-uralkodj algoritmusok egyik kiemelkedő példája a Strassen algoritmus, amelyet Volker Strassen dolgozott ki 1969-ben. A Strassen algoritmus jelentős szerepet játszik a mátrixszorzási eljárások optimalizálásában, mivel csökkenti a műveletek komplexitását a hagyományos módszerekhez képest. Míg a hagyományos algoritmusok O(n³) időbonyolultságúak, a Strassen algoritmus $O(n^log7) \approx O(n^2.81)$ komplexitású, mely nagyobb mátrixok esetén komoly gyorsulást eredményez.

#### Az algoritmus alapelvei

A Strassen algoritmus a mátrixszorzást 2x2-es mátrixokra vonatkozó eljárásra egyszerűsíti, majd az eredményeket kombinálja. Ennek alapelve a mátrixokat kisebb blokkokra bontja, majd ezeket a blokk-mátrixokat szorozza össze speciális szabályok szerint.

Tekintsünk két $A$ és $B$ mátrixot, melyek mindegyike nxn méretű, és n hatványai kettő. A Strassen algoritmus lépései a következők:

1. **Mátrix blokkokra bontása**: Mindkét mátrixot (A és B) fel kell osztani négy egyenlő részre (mindegyik n/2 x n/2 méretű):
   $$
   A = \begin{pmatrix}
   A_{11} & A_{12} \\
   A_{21} & A_{22}
   \end{pmatrix}
   $$
   és
   $$
   B = \begin{pmatrix}
   B_{11} & B_{12} \\
   B_{21} & B_{22}
   \end{pmatrix}
   $$

2. **Köztes mátrixok kiszámítása**: Speciális lineáris kombinációk alkalmazásával hét köztes mátrixot kell számítani:
   $$
   M_1 = (A_{11} + A_{22})(B_{11} + B_{22})
   $$
   $$
   M_2 = (A_{21} + A_{22})B_{11}
   $$
   $$
   M_3 = A_{11}(B_{12} - B_{22})
   $$
   $$
   M_4 = A_{22}(B_{21} - B_{11})
   $$
   $$
   M_5 = (A_{11} + A_{12})B_{22}
   $$
   $$
   M_6 = (A_{21} - A_{11})(B_{11} + B_{12})
   $$
   $$
   M_7 = (A_{12} - A_{22})(B_{21} + B_{22})
   $$

3. **Eredő mátrix kiszámítása**: A köztes mátrixok felhasználásával számítjuk az eredmény mátrix blokkjait:
   $$
   C_{11} = M_1 + M_4 - M_5 + M_7
   $$
   $$
   C_{12} = M_3 + M_5
   $$
   $$
   C_{21} = M_2 + M_4
   $$
   $$
   C_{22} = M_1 - M_2 + M_3 + M_6
   $$

4. **Blokkok kombinálása**: Az eredmény mátrix $C$ összeállítása a négy blokkból:
   $$
   C = \begin{pmatrix}
   C_{11} & C_{12} \\
   C_{21} & C_{22}
   \end{pmatrix}
   $$

#### Rekurzív alkalmazás

A leírt folyamat önmagában csak 2x2-es blokkokra vonatkozik, de rekurzív módon alkalmazva nagyobb méretű mátrixok esetén is hatékony eredményt ér el. Amint a mátrixok mérete eléri a legkisebb kívánt méretet (általában 1x1), a szorzás közvetlenül elvégezhető.

#### Implementáció C++ nyelven

Az alábbiakban bemutatjuk a Strassen algoritmus egy C++ nyelvű implementációját. Az egyszerűség kedvéért feltételezzük, hogy a mátrixok mérete hatványai kettőnek, és a végrehajtott műveletek nem optimalizáltak cache szempontjából. Egy általánosabb megvalósítás tovább fejleszthető optimalizációkkal és különböző mátrixméretek kezelésére is.

```cpp
#include <iostream>

#include <vector>

// Define a type for the matrix.
typedef std::vector<std::vector<int>> Matrix;

// Matrix addition
Matrix add(const Matrix &A, const Matrix &B) {
    int n = A.size();
    Matrix C(n, std::vector<int>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

// Matrix subtraction
Matrix subtract(const Matrix &A, const Matrix &B) {
    int n = A.size();
    Matrix C(n, std::vector<int>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}

// Strassen's algorithm for matrix multiplication
Matrix strassen(const Matrix &A, const Matrix &B) {
    int n = A.size();
    if (n == 1) {
        return Matrix{{A[0][0] * B[0][0]}};
    }

    int k = n / 2;
    Matrix A11(k, std::vector<int>(k)), A12(k, std::vector<int>(k)),
           A21(k, std::vector<int>(k)), A22(k, std::vector<int>(k)),
           B11(k, std::vector<int>(k)), B12(k, std::vector<int>(k)),
           B21(k, std::vector<int>(k)), B22(k, std::vector<int>(k));

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + k];
            A21[i][j] = A[i + k][j];
            A22[i][j] = A[i + k][j + k];
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + k];
            B21[i][j] = B[i + k][j];
            B22[i][j] = B[i + k][j + k];
        }
    }

    Matrix M1 = strassen(add(A11, A22), add(B11, B22));
    Matrix M2 = strassen(add(A21, A22), B11);
    Matrix M3 = strassen(A11, subtract(B12, B22));
    Matrix M4 = strassen(A22, subtract(B21, B11));
    Matrix M5 = strassen(add(A11, A12), B22);
    Matrix M6 = strassen(subtract(A21, A11), add(B11, B12));
    Matrix M7 = strassen(subtract(A12, A22), add(B21, B22));

    Matrix C11 = add(subtract(add(M1, M4), M5), M7);
    Matrix C12 = add(M3, M5);
    Matrix C21 = add(M2, M4);
    Matrix C22 = add(subtract(add(M1, M3), M2), M6);

    Matrix C(n, std::vector<int>(n));
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            C[i][j] = C11[i][j];
            C[i][j + k] = C12[i][j];
            C[i + k][j] = C21[i][j];
            C[i + k][j + k] = C22[i][j];
        }
    }
    return C;
}

// Main function for testing
int main() {
    Matrix A = { {1, 2}, {3, 4} };
    Matrix B = { {5, 6}, {7, 8} };

    Matrix C = strassen(A, B);
    
    for (const auto &row : C) {
        for (const auto &elem : row) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
```

Ez a kód egy egyszerű implementációja a Strassen algoritmusnak, amely bemutatja a mátrixok feldarabolását és az egyes lépések implementációját. A kódot optimalizálhatjuk különböző módszerekkel, például cache-optimalizációval vagy a legkisebb mátrixméret küszöbének beállításával, hogy igazán nagy mátrixok esetén gyorsabb legyen az eljárás.

#### Teljesítményelemzés

A Strassen algoritmus esetén az időkomplexitás $O(n^log2(7)) \approx O(n^2.81)$, ami kevesebb mint a hagyományos O(n³). Egyéb előnyök közé tartozik, hogy egyszerre kevesebb szorzást igényel (7 a korábbi 8 helyett), amely különösen nagy mátrixok esetén lehet hasznos. Az optimális implementációra vonatkozó további részletek a következő, 6.7.2 Teljesítmény elemzés és alkalmazások alfejezetben szerepelnek.

A Strassen algoritmus gyakorlati alkalmazásai közé tartoznak a tudományos számítások, a grafika, a gépi tanulás és minden olyan terület, ahol nagy méretű mátrixok műveletei kritikusak. A tényleges gyorsulás azonban az adott mátrix mérettől és az implementáció minőségétől is függ.

### 6.7.2 Teljesítmény elemzés és alkalmazások

#### 6.7.2.1 Teljesítmény Elemzés

A Strassen algoritmus, amely Volker Strassen német matematikus nevéhez fűződik, egy hatékony oszd-meg-és-uralkodj módszert kínál a négyzetes mátrixok szorzására. Az algoritmus az $O(n^3)$ időbeli bonyolultsággal dolgozó hagyományos mátrixszorzás helyett $O(n^{\log_2 7})$, azaz nagyjából $O(n^{2.81})$ időbeli bonyolultságot kínál. Mielőtt az algoritmus teljesítményének elemzésébe mélyülnénk, vegyük át lépésről lépésre, hogyan működik, és mik a kulcsfontosságú részei.

#### Az Algoritmus Működése

A Strassen algoritmus a $2 \times 2$ méretű mátrixokra alapozza az oszd-meg-és-uralkodj stratégiát. Az algoritmus a mátrixot kisebb részekre bontja, majd a részeket külön számítja ki. A $2 \times 2$ mátrixok esetében a következőképpen írható fel:

Legyenek A és B két $2 \times 2$ mátrix:
$$
A = \begin{pmatrix}
a & b \\
c & d
\end{pmatrix},
B = \begin{pmatrix}
e & f \\
g & h
\end{pmatrix}
$$

Az A és B mátrixok szorzata egy $2 \times 2$ mátrix lesz, ahol:
$$
C = A \times B = \begin{pmatrix}
p_1 + p_4 - p_5 + p_7 & p_3 + p_5 \\
p_2 + p_4 & p_1 - p_2 + p_3 + p_6
\end{pmatrix}
$$

Ahol a $p_i$ értékek a Strassen-féle részletek:
- $p_1 = a(e-f)$
- $p_2 = (a+b)h$
- $p_3 = (c+d)e$
- $p_4 = d(g-h)$
- $p_5 = (a+d)(e+h)$
- $p_6 = (b-d)(g+h)$
- $p_7 = (a-c)(e+f)$

A Strassen-féle szorzás helyettesíti a hagyományos nyolc szorzási műveletet hét szorzással és több (négyzetesen növekvő) összeadással és kivonással.

#### Rekurzió

Az algoritmus általában rekurzió révén működik, nagyobb mátrixokat $2 \times 2$-es blokkokra bontva:

1. Ha az input mátrixok mérete nem $2^k$, akkor a mátrixokat megfelelően kibővítjük.
2. A mátrixokat négy blokkra bontjuk: A-t és B-t.
3. Alkalmazzuk a Strassen algoritmust a blokkokra.
4. Összeállítjuk a végeredményt az egyes $p_i$ szorzatokból.

#### Teljesítmény elemzés

A Strassen algoritmus bonyolultságát úgy állapítjuk meg, hogy kiszámoljuk a maximális méretű műveletek számát:

1. **Oszd-meg-és-uralkodj lépések:**
    - Az A és B mátrixok szétbontása 4 részre.
    - 7 rekurzív hívás a $2 \times 2$ al-mátrixokra.

2. **Összegzés és kivonás:**
    - Kisebb mátrixok összegei és különbségei.

Ez alapján a T(n) időkomplexitás rekurzív egyenlete a Strassen-féle algoritmusra a következőképpen írható:
$$
T(n) = 7T\left(\frac{n}{2}\right) + O(n^2)
$$

Ez a kapcsolat gyorsan megmutatja, hogy a Strassen algoritmus bonyolultsága $O(n^{\log_2 7})$, ami hozzávetőleg $O(n^{2.81})$.

#### Empirikus Megközelítések

Az empirikus teljesítmény analízis az algoritmus gyakorlatban történő futtatásával és eredményeinek mérésével történik. Az empirikus vizsgálat kifejezetten fontos mivel a Strassen algoritmus nem mindig a leghatékonyabb kis méretű mátrixok esetében vagy erőforrás-korlátos környezetben. Gyakorlati tapasztalatok szerint az algoritmus előnyei főként nagyobb mátrixok esetén érvényesülnek, ahol az időkomplexitásbeli különbségek nagyobb hatást gyakorolnak a futásidőre.

#### 6.7.2.2 Alkalmazások

A Strassen algoritmust számos területen alkalmazzák, különösen ott, ahol nagy méretű mátrixok szorzására van szükség. Néhány példa:

1. **Numerikus analízis:**
   Nagy rendszerek megoldása, például a lineáris egyenletrendszerek esetén.

2. **Grafikus számítások:**
   Kép-alapú műveletek, valamint grafikai transzformációk során a mátrixszorzás kiemelkedően fontos.

3. **Adatbázisok:**
   Nagyméretű adatok műveleteinek gyorsítása és mátrix alapú adatelemzés.

4. **Fizikai szimulációk és modellezések:**
   A mátrix-alapú modelleket széles körben használják a fizikai szimulációkban, például a mechanikai ingás rendszerekben.

Ezen alkalmazási területeken a Strassen algoritmus hatékonyságának köszönhetően a számítási idő jelentősen csökkenthető, amely különösen fontos lehet nagy adatmennyiségek esetén.

#### Optimalizációk és Hibrid Megközelítések

Gyakorlati rendszerekben gyakran alkalmaznak hibrid megközelítéseket, amelyek kombinálják a Strassen algoritmust más módszerekkel. Például kis méretű mátrixok esetén a hagyományos $O(n^3)$ algoritmus rugalmasabb és könnyebben paraméterezhető, míg nagyobb mátrixok esetén a Strassen algoritmus futási ideje jelentősen gyorsabb lehet.

A hibrid algoritmusok optimalizálására számos stratégia létezik, beleértve a hatékony memóriaelérést, cache-optimalizált implementációkat, valamint a párhuzamosítási technikákat (parallel computing).

#### Konklúzió

A Strassen algoritmus kulcsfontosságú eredmény a mátrixszorzás optimalizálásában, különösen akkor, amikor a műveleti idő kulcsfontosságú. Az algoritmus nem csak matematikai szépségével, hanem gyakorlati hasznosságával is kiemelkedik a számítástudomány területén. Azt azonban fontos megjegyezni, hogy az optimalizálás és a helyes alkalmazási környezet meghatározása nélkül az algoritmus nem mindig a leghatékonyabb választás.

A fentiek minden nagyméretű mátrix szorzása során fontos iránymutatást adhatnak, és a Strassen algoritmus helyes megválasztása sok esetben a teljesítmény növekedéshez vezethet, amely elengedhetetlen a modern számítógépes rendszerekben.

