\newpage

## 5.2.   N-királynő probléma

A N-királynő probléma az egyik legismertebb és legnépszerűbb példa a backtrack algoritmusok alkalmazására. A probléma lényege, hogy egy $N \times N$ méretű sakktáblán úgy helyezzünk el $N$ királynőt, hogy azok ne támadhassák meg egymást. Ez azt jelenti, hogy egyetlen királynő sem állhat ugyanazon a soron, oszlopon vagy átlón. Ez a probléma nemcsak a matematika és a számítástudomány területén bír nagy jelentőséggel, hanem számos gyakorlati alkalmazása is van, például az optimalizálás és a mesterséges intelligencia területén. A következő szakaszokban részletesen megvizsgáljuk a probléma definícióját és lehetséges megoldásait, bemutatjuk a vizuális megjelenítést és az alkalmazási lehetőségeket, valamint áttekintjük az implementációs lépéseket és az optimalizálási technikákat.

### 5.2.1. Probléma definíció és megoldás

A N-királynő probléma egy klasszikus kombinatorikai probléma, amely mély matematikai gyökerekkel rendelkezik és jelentősége van a számítástudományban, különösen az algoritmusok és a mesterséges intelligencia területén. A probléma célja, hogy egy $N \times N$ méretű sakktáblán $N$ darab királynőt helyezzünk el úgy, hogy azok ne támadhassák meg egymást. Ez azt jelenti, hogy egyetlen királynő sem állhat ugyanazon a soron, oszlopon vagy átlón.

#### Probléma Definíció

Formálisan a N-királynő probléma a következőképpen definiálható:

- **Input**: Egy egész szám $N$, amely meghatározza a sakktábla méretét ($N \times N$) és a királynők számát.
- **Output**: Az összes olyan konfiguráció, amelyben $N$ királynőt helyezünk el a táblán úgy, hogy semelyik kettő ne álljon ugyanazon a soron, oszlopon vagy átlón.

Ez a probléma különösen érdekes, mert a megoldások száma gyorsan nő az $N$ növekedésével, és egyre bonyolultabbá válik megtalálni az összes érvényes elrendezést.

#### Matematikai Modell

A probléma matematikai modellje a következő feltételeket tartalmazza:

1. **Sor és oszlop feltételek**: Minden királynő különböző sorokban és oszlopokban kell, hogy legyen. Ez azt jelenti, hogy ha $Q_i$ az $i$-edik királynő helyzetét jelöli, akkor $Q_i \neq Q_j$ minden $i \neq j$ esetén.
2. **Átlók feltételei**: Minden királynő különböző átlókon kell, hogy legyen. Ez két feltételre bontható:
    - Főátlók: Az $i$-edik királynő a $(i, Q_i)$ pozícióban van. Az átló feltétele azt jelenti, hogy $|Q_i - Q_j| \neq |i - j|$ minden $i \neq j$ esetén.
    - Mellékátlók: Hasonlóan, $|Q_i + Q_j| \neq |i + j|$ minden $i \neq j$ esetén.

#### Megoldási Stratégia: Backtracking

A backtracking algoritmus az egyik legelterjedtebb módszer a N-királynő probléma megoldására. Ez az algoritmus egy rekurzív keresési technika, amely fokozatosan épít fel egy megoldást, és ha egy részmegoldás nem vezet célhoz, visszalép és más utakat próbál ki.

##### Backtracking Algoritmus Lépései:

1. **Kezdeti lépés**: Kezdjük az első sorral.
2. **Elhelyezés**: Helyezzünk el egy királynőt az aktuális sorban egy olyan oszlopban, ahol az nem támadható meg más királynők által.
3. **Rekurzív lépés**: Lépjünk a következő sorra és ismételjük meg az elhelyezést.
4. **Visszalépés**: Ha nem találunk érvényes helyet az aktuális sorban, lépjünk vissza az előző sorra és próbáljunk meg másik oszlopot.
5. **Megoldás tárolása**: Ha elértük az utolsó sort és sikeresen elhelyeztünk egy királynőt, tároljuk a megoldást.
6. **Visszalépés és folytatás**: Folytassuk a keresést, hogy megtaláljuk az összes lehetséges megoldást.

##### Pseudocode

Az alábbi pseudocode leírja a backtracking megközelítést:

```
function solveNQueens(N):
    solutions = []
    board = createEmptyBoard(N)
    placeQueens(board, 0, N, solutions)
    return solutions

function placeQueens(board, row, N, solutions):
    if row == N:
        solutions.append(copy(board))
        return
    for col in range(0, N):
        if isSafe(board, row, col, N):
            board[row][col] = 'Q'
            placeQueens(board, row + 1, N, solutions)
            board[row][col] = '.'

function isSafe(board, row, col, N):
    for i in range(0, row):
        if board[i][col] == 'Q':
            return False
    for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
        if board[i][j] == 'Q':
            return False
    for i, j in zip(range(row-1, -1, -1), range(col+1, N)):
        if board[i][j] == 'Q':
            return False
    return True
```

##### C++ Implementáció

Az alábbi C++ kód bemutatja a backtracking algoritmus egy lehetséges implementációját a N-királynő probléma megoldására:

```cpp
#include <iostream>
#include <vector>
using namespace std;

bool isSafe(vector<string> &board, int row, int col, int N) {
    for (int i = 0; i < row; ++i) {
        if (board[i][col] == 'Q') return false;
    }
    for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; --i, --j) {
        if (board[i][j] == 'Q') return false;
    }
    for (int i = row - 1, j = col + 1; i >= 0 && j < N; --i, ++j) {
        if (board[i][j] == 'Q') return false;
    }
    return true;
}

void solveNQueensUtil(vector<string> &board, int row, int N, vector<vector<string>> &solutions) {
    if (row == N) {
        solutions.push_back(board);
        return;
    }
    for (int col = 0; col < N; ++col) {
        if (isSafe(board, row, col, N)) {
            board[row][col] = 'Q';
            solveNQueensUtil(board, row + 1, N, solutions);
            board[row][col] = '.';
        }
    }
}

vector<vector<string>> solveNQueens(int N) {
    vector<vector<string>> solutions;
    vector<string> board(N, string(N, '.'));
    solveNQueensUtil(board, 0, N, solutions);
    return solutions;
}

int main() {
    int N = 8; // Példa: 8-királynő probléma
    vector<vector<string>> solutions = solveNQueens(N);
    for (auto solution : solutions) {
        for (auto row : solution) {
            cout << row << endl;
        }
        cout << endl;
    }
    return 0;
}
```

#### Alternatív Megoldások és Optimalizálás

A backtracking algoritmus hatékonyságának növelése érdekében számos optimalizálási technika alkalmazható, mint például a memoizáció, a bit-manipuláció, és a heurisztikus megközelítések.

##### Bit-Manipuláció

A bit-manipuláció egy hatékony módja annak, hogy csökkentsük az idő- és memóriaköltséget a N-királynő probléma megoldása során. Ebben a megközelítésben a bit-műveleteket használjuk a királynők helyzetének és az átlók ellenőrzésére.

##### Heurisztikus Megközelítések

Heurisztikus algoritmusok, mint például a genetikus algoritmusok vagy a mesterséges méh kolónia algoritmusok, szintén alkalmazhatók a N-királynő probléma megoldására. Ezek a módszerek gyakran gyorsabbak lehetnek nagy $N$ értékek esetén, bár nem garantálnak optimális megoldást.

### 5.2.2. Optimalizálási technikák

A hagyományos backtracking algoritmus önmagában hatékony, de jelentős számítási erőforrásokat igényelhet, különösen nagy $N$ értékek esetén. Az alábbiakban részletesen tárgyaljuk a különböző optimalizálási technikákat, amelyek segítségével javíthatjuk a backtracking algoritmus teljesítményét a N-királynő probléma megoldása során. Ezek közé tartoznak a heurisztikák, prunning technikák és más fejlett módszerek.

#### Heurisztikák

Heurisztikák alkalmazása a backtracking algoritmusban segíthet csökkenteni a keresési tér méretét, ezáltal javítva az algoritmus hatékonyságát. Az alábbiakban néhány heurisztikát mutatunk be:

1. **Minimum Remaining Values (MRV)**: Mindig azt az oszlopot válasszuk, ahol a legkevesebb lehetőség van a királynő elhelyezésére. Ez a heurisztika minimalizálja a hátralévő lehetőségeket, így gyorsabban felismerhetjük a zsákutcákat.

2. **Least Constraining Value (LCV)**: Azokat az oszlopokat részesítjük előnyben, amelyek a legkevesebb korlátozást jelentik a többi királynő számára. Ez a heurisztika lehetővé teszi, hogy több lehetőség maradjon a későbbi királynők elhelyezésére.

#### Pruning Technika

A prunning technika, más néven metszés, azt jelenti, hogy megszabadulunk azoktól a részmegoldásoktól, amelyek biztosan nem vezetnek teljes megoldáshoz. Ezáltal jelentősen csökkenthetjük a keresési tér méretét és gyorsíthatjuk a megoldási folyamatot. Az alábbiakban néhány prunning technikát mutatunk be:

1. **Forward Checking**: Minden királynő elhelyezése után frissítjük az összes többi királynő lehetséges pozícióinak listáját. Ha bármelyik királynő számára nincs érvényes pozíció, azonnal visszalépünk (backtrack).

2. **Constraint Propagation**: Kiterjesztett forward checking, ahol a korlátozások propagálása során tovább szűkítjük a lehetséges pozíciók listáját a következő lépésekhez.

##### Példa: Forward Checking Implementáció C++ nyelven

```cpp
#include <iostream>
#include <vector>
using namespace std;

bool isSafe(int row, int col, vector<int> &solution) {
    for (int i = 0; i < row; ++i) {
        if (solution[i] == col || abs(solution[i] - col) == abs(i - row))
            return false;
    }
    return true;
}

void forwardChecking(int row, vector<int> &solution, vector<vector<int>> &solutions, int N, vector<bool> &columns, vector<bool> &d1, vector<bool> &d2) {
    if (row == N) {
        solutions.push_back(solution);
        return;
    }
    for (int col = 0; col < N; ++col) {
        if (!columns[col] && !d1[row - col + N - 1] && !d2[row + col]) {
            solution[row] = col;
            columns[col] = d1[row - col + N - 1] = d2[row + col] = true;
            forwardChecking(row + 1, solution, solutions, N, columns, d1, d2);
            columns[col] = d1[row - col + N - 1] = d2[row + col] = false;
        }
    }
}

vector<vector<int>> solveNQueens(int N) {
    vector<vector<int>> solutions;
    vector<int> solution(N, -1);
    vector<bool> columns(N, false);
    vector<bool> d1(2 * N - 1, false);
    vector<bool> d2(2 * N - 1, false);
    forwardChecking(0, solution, solutions, N, columns, d1, d2);
    return solutions;
}

int main() {
    int N = 8; // Példa: 8-királynő probléma
    vector<vector<int>> solutions = solveNQueens(N);
    for (auto &solution : solutions) {
        for (int row : solution) {
            for (int col = 0; col < N; ++col) {
                cout << (col == row ? 'Q' : '.') << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    return 0;
}
```

#### Constraint Propagation

A constraint propagation egy továbbfejlesztett technika, amelyben az aktuális döntés után minden további változó tartományát frissítjük, hogy tükrözze az új korlátozásokat. Ez csökkenti a lehetséges megoldások számát és lehetővé teszi a korai visszalépést.

##### Példa: Constraint Propagation

1. **Előre látás**: A királynő elhelyezése után frissítjük a lehetséges pozíciók listáját minden sorban és átlóban.
2. **Döntési pontok**: Ha bármelyik változó (sor) tartománya üressé válik, azonnal visszalépünk.

#### Bit-Manipuláció

A bit-manipuláció hatékony módszer a támadási pozíciók nyilvántartására. Három bitmaszkot használhatunk: egyet az oszlopok, egyet a főátlók, és egyet a mellékátlók számára. Ez jelentősen csökkentheti az ellenőrzési időt.

##### Bit-Manipuláció Implementáció

```cpp
#include <iostream>
#include <vector>
using namespace std;

void solveNQueens(int row, int cols, int diags1, int diags2, vector<int> &solution, vector<vector<int>> &solutions, int N) {
    if (row == N) {
        solutions.push_back(solution);
        return;
    }
    for (int col = 0; col < N; ++col) {
        int diag1 = row - col + N - 1;
        int diag2 = row + col;
        if (!(cols & (1 << col)) && !(diags1 & (1 << diag1)) && !(diags2 & (1 << diag2))) {
            solution[row] = col;
            solveNQueens(row + 1, cols | (1 << col), diags1 | (1 << diag1), diags2 | (1 << diag2), solution, solutions, N);
        }
    }
}

vector<vector<int>> solveNQueens(int N) {
    vector<vector<int>> solutions;
    vector<int> solution(N, -1);
    solveNQueens(0, 0, 0, 0, solution, solutions, N);
    return solutions;
}

int main() {
    int N = 8;
    vector<vector<int>> solutions = solveNQueens(N);
    for (auto &solution : solutions) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                cout << (solution[i] == j ? 'Q' : '.') << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    return 0;
}
```

### 5.2.3. Vizuális megjelenítés és alkalmazások

A N-királynő probléma nemcsak algoritmusok és matematikai modellezés szempontjából érdekes, hanem vizuálisan is izgalmas kihívás. A probléma vizuális megjelenítése segít megérteni a megoldások szerkezetét, valamint az alkalmazási területek széles skáláját nyitja meg. Ebben a fejezetben részletesen tárgyaljuk a vizuális megjelenítési módszereket, valamint a N-királynő probléma gyakorlati alkalmazásait.

#### Vizuális Megjelenítés

A N-királynő probléma megoldásainak vizuális megjelenítése során a cél, hogy a sakktáblán elhelyezett királynők pozícióját szemléltessük úgy, hogy azok ne támadhassák meg egymást. A vizuális megjelenítés eszközei közé tartoznak a grafikus ábrázolások, animációk és interaktív megoldások.

##### Sakktábla Ábrázolása

A sakktábla vizuális megjelenítése egyszerű négyzetrácsos ábrázolás, ahol az egyes mezők vagy üresek, vagy egy királynőt tartalmaznak. Az ábrázolás során figyelembe kell venni a következő szempontokat:

1. **Méret**: A sakktábla $N \times N$ méretű.
2. **Kirakodás**: A királynők elhelyezése az egyes sorokban és oszlopokban.
3. **Színezés**: A sakktábla mezőit általában fekete és fehér színekkel szokás ábrázolni, hasonlóan a hagyományos sakktáblához.

##### Animációk és Interaktív Megoldások

Az animációk és interaktív megoldások lehetővé teszik a N-királynő probléma dinamikus bemutatását. Az animációk során lépésről lépésre követhetjük a királynők elhelyezésének folyamatát, amely különösen hasznos az algoritmusok működésének szemléltetéséhez.

**Interaktív vizualizáció**: Egy interaktív vizualizációs eszköz lehetővé teszi a felhasználók számára, hogy saját maguk próbálják ki a királynők elhelyezését a sakktáblán, és valós időben láthatják az eredményeket.

##### Példa C++ Implementáció: Grafikus Megjelenítés

Az alábbi C++ kód egy egyszerű konzolos megjelenítést mutat be, amely vizualizálja a megoldást egy $N \times N$ sakktáblán.

```cpp
#include <iostream>
#include <vector>
using namespace std;

void printBoard(vector<int> &solution, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (solution[i] == j) cout << 'Q' << " ";
            else cout << '.' << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void solveNQueens(int row, int cols, int diags1, int diags2, vector<int> &solution, vector<vector<int>> &solutions, int N) {
    if (row == N) {
        solutions.push_back(solution);
        printBoard(solution, N);
        return;
    }
    for (int col = 0; col < N; ++col) {
        int diag1 = row - col + N - 1;
        int diag2 = row + col;
        if (!(cols & (1 << col)) && !(diags1 & (1 << diag1)) && !(diags2 & (1 << diag2))) {
            solution[row] = col;
            solveNQueens(row + 1, cols | (1 << col), diags1 | (1 << diag1), diags2 | (1 << diag2), solution, solutions, N);
        }
    }
}

int main() {
    int N = 8; // Példa: 8-királynő probléma
    vector<vector<int>> solutions;
    vector<int> solution(N, -1);
    solveNQueens(0, 0, 0, 0, solution, solutions, N);
    return 0;
}
```

#### Alkalmazások

A N-királynő probléma számos gyakorlati alkalmazási területtel rendelkezik, különösen az optimalizálás és a mesterséges intelligencia területén. Az alábbiakban néhány konkrét alkalmazási példát mutatunk be.

##### Optimalizálási Feladatok

1. **Erőforrás-ütemezés**: A N-királynő probléma hasonló az erőforrás-ütemezési problémákhoz, ahol több feladatot kell úgy elosztani, hogy ne legyenek ütközések. Például, egyetemek órarendjének összeállításánál, ahol a tanárokat és tantermeket úgy kell elosztani, hogy ne legyenek átfedések.

2. **Számítógépes hálózatok**: A N-királynő probléma megoldása hasznos lehet számítógépes hálózatok optimalizálásában is, ahol a csomópontok (királynők) közötti kommunikációt kell úgy irányítani, hogy minimális legyen az ütközés és a késleltetés.

##### Mesterséges Intelligencia és Robotika

1. **Térbeli tervezés**: A robotok mozgásának és elhelyezkedésének tervezése során hasznos lehet a N-királynő probléma, különösen, ha a robotok útvonalát és pozícióját úgy kell meghatározni, hogy ne akadályozzák egymást.

2. **Játéktervezés**: A N-királynő probléma megoldása felhasználható a mesterséges intelligencia alapú játékokban, például sakkszimulátorokban, ahol a különböző figurák mozgását és elhelyezkedését kell optimalizálni.

##### Kutatási és Oktatási Eszköz

1. **Algoritmusok tanítása**: A N-királynő probléma kiválóan alkalmas az algoritmusok és optimalizálási technikák oktatására. A probléma egyszerűsége és a megoldási technikák széles skálája lehetővé teszi a diákok számára, hogy mélyebben megismerkedjenek a backtracking, heurisztikák, és prunning technikák alkalmazásával.

2. **Kutatási területek**: A N-királynő probléma számos kutatási területen hasznos, különösen az algoritmusok optimalizálása és a kombinatorikus problémák megoldása terén. A probléma komplexitása és a megoldási stratégiák variációi új kutatási irányokat nyithatnak meg.

