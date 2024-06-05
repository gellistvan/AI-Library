\newpage

## 5.4. Sudoku megoldása

A Sudoku rejtvények megoldása kiváló példája a backtrack algoritmusok alkalmazásának. Ez a népszerű logikai játék, amely egy 9x9-es rács kitöltését igényli számokkal 1-től 9-ig, úgy, hogy minden sorban, oszlopban és 3x3-as alrácsban minden szám csak egyszer szerepelhet, nemcsak a szórakozást szolgálja, hanem a kombinatorikus problémák megértésének hatékony eszköze is. A következő alfejezetekben bemutatjuk a Sudoku megoldásának algoritmusát és annak implementációját, majd feltárjuk azokat a heurisztikákat és optimalizálási technikákat, amelyekkel az algoritmus hatékonysága tovább növelhető. Az elméleti háttér és a gyakorlati példák segítségével az olvasók átfogó képet kapnak arról, hogyan alkalmazható a backtracking módszer e konkrét probléma megoldására, és hogyan lehet általánosítani más hasonló kombinatorikus kihívásokra.

### 5.4.1. Algoritmus és implementáció

A Sudoku megoldása során a backtracking algoritmus egy robusztus megközelítést kínál a probléma összetettségének kezelésére. Ez az algoritmus olyan problémák megoldására alkalmas, amelyekben a megoldás egy részleges megoldások sorozatának keresésével található meg, és a rossz irányokból való visszatérés (backtracking) szükséges lehet a helyes megoldás megtalálásához.

#### A Sudoku probléma leírása

A Sudoku egy 9x9-es rács, amely 81 cellából áll. A cél az, hogy minden cellát számokkal töltsünk ki 1-től 9-ig úgy, hogy minden sorban, oszlopban és 3x3-as alrácsban minden szám pontosan egyszer szerepeljen. A rács részben kitöltött, és az üres cellák megfelelő számokkal való kitöltése a feladat.

#### Backtracking algoritmus

A backtracking algoritmus lépései a Sudoku megoldására a következők:

1. **Keresd meg az első üres cellát a rácsban.**
2. **Próbálj ki egy számot 1-től 9-ig az üres cellában.**
3. **Ellenőrizd, hogy az aktuális szám érvényes-e az aktuális cellában, azaz nem ütközik-e a sorban, oszlopban vagy a 3x3-as alrácsban lévő többi számmal.**
4. **Ha az aktuális szám érvényes, lépj a következő üres cellára, és ismételd meg a folyamatot.**
5. **Ha egy szám érvénytelen, próbáld ki a következőt a 1-től 9-ig.**
6. **Ha egy cellába nem helyezhető érvényes szám, akkor lépj vissza az előző cellához (backtrack), és próbálj ki egy másik számot.**
7. **Folytasd a folyamatot, amíg a rács teljesen ki nem töltött vagy minden lehetőséget ki nem próbáltál.**

#### Algoritmus lépései részletesen

**1. Üres cella keresése:**
Az algoritmus először megkeresi az első üres cellát. Ez lehet egy egyszerű iteráció a rács celláin keresztül.

**2. Szám kipróbálása:**
Az algoritmus megpróbál minden lehetséges számot (1-9) az üres cellában.

**3. Érvényesség ellenőrzése:**
Az érvényesség ellenőrzése három fő feltétel alapján történik:
- **Sor érvényessége:** Az adott szám nem lehet jelen az aktuális sorban.
- **Oszlop érvényessége:** Az adott szám nem lehet jelen az aktuális oszlopban.
- **Alrács érvényessége:** Az adott szám nem lehet jelen az aktuális 3x3-as alrácsban.

**4. Következő üres cellára lépés:**
Ha az aktuális szám érvényes, az algoritmus a következő üres cellára lép, és megismétli a folyamatot.

**5. Backtracking:**
Ha egy szám érvénytelen, az algoritmus a következő számot próbálja ki. Ha egyik szám sem érvényes, visszalép az előző cellára, és ott próbálkozik egy másik számmal.

#### Implementáció C++ nyelven

Az alábbiakban bemutatunk egy példaimplementációt C++ nyelven a Sudoku megoldására backtracking algoritmussal:

```cpp
#include <iostream>

#include <vector>

using namespace std;

#define N 9

bool isValid(vector<vector<int>>& board, int row, int col, int num) {
    // Check if the number is already in the row
    for (int x = 0; x < N; x++)
        if (board[row][x] == num)
            return false;

    // Check if the number is already in the column
    for (int x = 0; x < N; x++)
        if (board[x][col] == num)
            return false;

    // Check if the number is already in the 3x3 subgrid
    int startRow = row - row % 3, startCol = col - col % 3;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (board[i + startRow][j + startCol] == num)
                return false;

    return true;
}

bool solveSudoku(vector<vector<int>>& board, int row, int col) {
    // If we have reached the end of the board
    if (row == N - 1 && col == N)
        return true;
    
    // If column value becomes 9, we move to the next row
    if (col == N) {
        row++;
        col = 0;
    }

    // If the current position is already filled, move to the next column
    if (board[row][col] != 0)
        return solveSudoku(board, row, col + 1);

    // Try placing numbers 1 to 9 in the current empty cell
    for (int num = 1; num <= 9; num++) {
        // Check if placing num at board[row][col] is valid
        if (isValid(board, row, col, num)) {
            board[row][col] = num;

            // Recursively try filling the rest of the board
            if (solveSudoku(board, row, col + 1))
                return true;
        }

        // If num is not valid or if solving further does not lead to a solution
        // then we reset the cell and backtrack
        board[row][col] = 0;
    }

    // If no number from 1 to 9 can solve the board, return false
    return false;
}

void printBoard(const vector<vector<int>>& board) {
    for (int r = 0; r < N; r++) {
        for (int d = 0; d < N; d++) {
            cout << board[r][d] << " ";
        }
        cout << endl;
    }
}

int main() {
    vector<vector<int>> board = {
        {5, 3, 0, 0, 7, 0, 0, 0, 0},
        {6, 0, 0, 1, 9, 5, 0, 0, 0},
        {0, 9, 8, 0, 0, 0, 0, 6, 0},
        {8, 0, 0, 0, 6, 0, 0, 0, 3},
        {4, 0, 0, 8, 0, 3, 0, 0, 1},
        {7, 0, 0, 0, 2, 0, 0, 0, 6},
        {0, 6, 0, 0, 0, 0, 2, 8, 0},
        {0, 0, 0, 4, 1, 9, 0, 0, 5},
        {0, 0, 0, 0, 8, 0, 0, 7, 9}
    };

    if (solveSudoku(board, 0, 0))
        printBoard(board);
    else
        cout << "No solution exists" << endl;

    return 0;
}
```

#### Algoritmus analízise és optimalizálási lehetőségek

A bemutatott backtracking algoritmus egy egyszerű és érthető megközelítés a Sudoku megoldására. Azonban, mivel minden lehetséges számot kipróbál minden üres cellában, az algoritmus időigénye exponenciálisan növekszik a rács méretével. Ezért gyakran szükséges optimalizálási technikák alkalmazása a hatékonyság javítása érdekében.

**Heurisztikák:**
- **Minimum Remaining Value (MRV):** Válasszuk azt az üres cellát, amelynek a legkevesebb érvényes lehetősége van. Ez csökkenti a backtracking szükségességét, mivel az elsőként megoldandó cellák gyorsabban tölthetők ki.
- **Degree Heuristic:** Ha több cellának is ugyanannyi érvényes lehetősége van, válasszuk azt, amelyik a legtöbb meg nem oldott cellához kapcsolódik. Ez maximalizálja az újonnan felmerülő kényszerek számát.
- **Least Constraining Value:** Válasszuk azt a számot, amelyik a legkevesebb korlátozást okozza a többi még meg nem oldott cellában.

**Optimalizálási technikák:**
- **Constraint Propagation:** Ha egy számot egy cellába helyezünk, azonnal frissítsük a többi cella lehetséges értékeit, hogy minimalizáljuk a jövőbeli ellenőrzések számát.
- **Dancing Links és Exact Cover:** Használhatók speciális adatstruktúrák és algoritmusok, mint például a Dancing Links (DLX) és az Exact Cover, amelyek hatékonyan kezelik a kombinatorikus problémákat, mint a Sudoku.

A bemutatott algoritmus és a kapcsolódó optimalizálási technikák alapvető eszközöket nyújtanak a Sudoku és más hasonló kombinatorikus problémák megoldására, lehetővé téve a backtracking módszer hatékony alkalmazását.

### 5.4.2. Heurisztikák és optimalizálás

A Sudoku megoldása backtracking algoritmussal hatékony lehet, de a problémák összetettségével és a potenciális megoldások nagy számával az algoritmus teljesítménye drasztikusan csökkenhet. Ezért számos heurisztika és optimalizálási technika létezik, amelyek javíthatják a backtracking algoritmus hatékonyságát. Ezek a módszerek minimalizálják a szükséges lépések számát és gyorsítják a megoldás folyamatát. Ebben a fejezetben részletesen bemutatjuk ezeket a technikákat és azok alkalmazását a Sudoku megoldásában.

#### Minimum Remaining Value (MRV) Heurisztika

Az MRV heurisztika az egyik leghatékonyabb módszer a keresési tér szűkítésére. Az MRV elve alapján mindig azt az üres cellát választjuk, amelyiknek a legkevesebb lehetséges értéke van. Ezzel minimalizáljuk a további döntési lehetőségek számát és maximalizáljuk a megoldáshoz vezető út gyors megtalálásának esélyét.

**MRV implementációja:**

```cpp
pair<int, int> findEmptyCellWithMRV(const vector<vector<int>>& board) {
    int minOptions = 10; // Maximum 9 lehetőség van (1-9)
    pair<int, int> bestCell = {-1, -1};

    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            if (board[row][col] == 0) {
                int options = 0;
                for (int num = 1; num <= 9; num++) {
                    if (isValid(board, row, col, num)) {
                        options++;
                    }
                }
                if (options < minOptions) {
                    minOptions = options;
                    bestCell = {row, col};
                }
            }
        }
    }
    return bestCell;
}
```

#### Degree Heurisztika

A Degree Heurisztika akkor használatos, ha több cellának ugyanannyi érvényes lehetősége van. Ebben az esetben azt a cellát választjuk, amelyik a legtöbb meg nem oldott cellához kapcsolódik (azaz amelyik a legtöbb korlátozást hozza létre). Ezzel maximalizáljuk a kényszerek érvényesítésének számát, ami további cellák érvényes értékeinek csökkentéséhez vezethet.

#### Least Constraining Value (LCV) Heurisztika

Az LCV heurisztika alapján mindig azt a számot választjuk, amelyik a legkevesebb korlátozást okozza a többi még meg nem oldott cellában. Ezzel minimalizáljuk a jövőbeli konfliktusok lehetőségét és növeljük a megoldás gyors megtalálásának esélyét.

**LCV implementációja:**

```cpp
vector<int> findLeastConstrainingValues(const vector<vector<int>>& board, int row, int col) {
    vector<int> values;
    vector<int> constraints(10, 0);

    for (int num = 1; num <= 9; num++) {
        if (isValid(board, row, col, num)) {
            values.push_back(num);
            // Count how many times this number can be placed in other empty cells
            for (int r = 0; r < N; r++) {
                for (int c = 0; c < N; c++) {
                    if (board[r][c] == 0 && isValid(board, r, c, num)) {
                        constraints[num]++;
                    }
                }
            }
        }
    }

    // Sort values based on constraints in ascending order
    sort(values.begin(), values.end(), [&](int a, int b) {
        return constraints[a] < constraints[b];
    });

    return values;
}
```

#### Constraint Propagation

A Constraint Propagation (kényszerek terjesztése) egy hatékony technika, amely azonnal frissíti a rács állapotát, amikor egy számot egy cellába helyezünk. Ez a technika minimalizálja a jövőbeli ellenőrzések számát azáltal, hogy a lehetséges értékek számát csökkenti a többi cellában.

Egy egyszerű példa a Constraint Propagation-ra a Forward Checking, ahol minden lépés után frissítjük a lehetséges értékek halmazát az üres cellákban. Ha egy cellának nincs több érvényes lehetősége, azonnal visszalépünk (backtrack).

**Forward Checking implementációja:**

```cpp
bool forwardCheck(vector<vector<int>>& board, vector<vector<set<int>>>& options, int row, int col, int num) {
    // Place the number and update the options
    board[row][col] = num;
    options[row][col].clear();

    // Update options for related cells
    for (int x = 0; x < N; x++) {
        options[row][x].erase(num);
        options[x][col].erase(num);
    }
    int startRow = row - row % 3, startCol = col - col % 3;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            options[i + startRow][j + startCol].erase(num);
        }
    }

    // Check for any empty cells with no options left
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            if (board[r][c] == 0 && options[r][c].empty()) {
                return false;
            }
        }
    }

    return true;
}

bool solveSudokuWithForwardChecking(vector<vector<int>>& board, vector<vector<set<int>>>& options, int row, int col) {
    // Find the next empty cell with MRV
    pair<int, int> cell = findEmptyCellWithMRV(board);
    row = cell.first;
    col = cell.second;

    if (row == -1) {
        return true; // No empty cell left
    }

    // Try least constraining values
    vector<int> values = findLeastConstrainingValues(board, row, col);
    for (int num : values) {
        if (isValid(board, row, col, num)) {
            // Temporarily place the number and forward check
            vector<vector<set<int>>> backupOptions = options;
            if (forwardCheck(board, options, row, col, num)) {
                if (solveSudokuWithForwardChecking(board, options, row, col)) {
                    return true;
                }
            }
            // Backtrack
            board[row][col] = 0;
            options = backupOptions;
        }
    }

    return false;
}
```

#### Dancing Links és Exact Cover problémák

A Dancing Links (DLX) és az Exact Cover problémák Donald Knuth által kidolgozott megközelítései. Ezek a technikák hatékony adatstruktúrákat és algoritmusokat kínálnak a kombinatorikus problémák, például a Sudoku megoldására. Az Exact Cover probléma egy olyan probléma, amelyben a bemeneti halmazt az alhalmazok egy olyan kiválasztott családjával kell lefedni, amelyben minden elem pontosan egyszer szerepel.

A Dancing Links egy hatékony adatstruktúra az Exact Cover problémák megoldására. A DLX a linkelt listák speciális változata, ahol a csomópontok eltávolítása és visszahelyezése gyorsan és hatékonyan végezhető.

**DLX implementációja:**

A DLX implementációja meglehetősen összetett, de az alapelvek a következők:
1. A Sudoku rácsot egy bináris mátrixként modellezzük, ahol minden sor egy lehetséges számelhelyezést képvisel.
2. A Dancing Links adatstruktúrát használjuk a mátrix manipulálására és a lefedési probléma megoldására.
3. Az algoritmus iteratív módon eltávolítja és visszahelyezi a lehetséges megoldásokat, miközben keresi a teljes lefedést.

A fenti optimalizálási technikák és heurisztikák alkalmazásával a backtracking algoritmus hatékonysága jelentősen növelhető. Ezek a módszerek minimalizálják a keresési tér nagyságát, gyorsítják a megoldási folyamatot és növelik a sikeres megoldás megtalálásának valószínűségét. A Sudoku megoldása így nemcsak gyorsabbá, hanem megbízhatóbbá is válik, lehetővé téve a nagyobb és összetettebb rácsok hatékony kezelését is.