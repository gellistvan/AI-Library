## 5.6.   Labirintus megoldása

A labirintusok megoldása az algoritmusok világában egy klasszikus probléma, amely kiválóan demonstrálja a backtrack algoritmusok hatékonyságát és alkalmazhatóságát. Ez a fejezet bemutatja, hogyan lehet egy labirintust algoritmikusan megoldani, lépésről lépésre feltárva az ehhez szükséges adatszerkezeteket és algoritmusokat. A 5.6.1 részben részletesen megvizsgáljuk a labirintus reprezentálására használt adatszerkezeteket, valamint az alapvető backtrack algoritmus működését és implementációját. A 5.6.2 részben pedig különböző heurisztikákat és optimalizálási technikákat tárgyalunk, amelyek javíthatják a megoldás hatékonyságát, és segítenek megtalálni a lehető leggyorsabb útvonalat a labirintusban.

### 5.6.1. Adatszerkezet, algoritmus és implementáció

A labirintus megoldása, mint klasszikus backtracking probléma, különösen hasznos az algoritmusok oktatásában és kutatásában. Ebben az alfejezetben részletesen bemutatjuk a labirintus megoldásához szükséges adatszerkezeteket, az alapvető algoritmust, valamint annak implementációját.

#### Adatszerkezetek

A labirintus megoldásához szükséges alapvető adatszerkezetek a következők:

1. **Labirintus Reprezentáció**:
   A labirintust általában egy kétdimenziós mátrixként ábrázoljuk, ahol az egyes cellák lehetnek járhatóak vagy akadályok. Egy tipikus reprezentáció a következőképpen néz ki:

   ```cpp
   const int N = 5;
   int maze[N][N] = {
       {1, 0, 0, 0, 0},
       {1, 1, 0, 1, 1},
       {0, 1, 0, 0, 1},
       {1, 1, 1, 0, 1},
       {0, 0, 1, 1, 1}
   };
   ```
   Ebben a példában az `1` értékű cellák járhatóak, míg a `0` értékűek akadályokat jelentenek.

2. **Látogatottsági Mátrix**:
   A visszalépéses (backtracking) algoritmus során szükséges lehet egy látogatottsági mátrix használata, amely nyomon követi, hogy mely cellákat látogattuk meg korábban, hogy elkerüljük az ismételt látogatást és az esetleges végtelen ciklusokat:

   ```cpp
   bool visited[N][N] = {false};
   ```

#### Algoritmus

A labirintus megoldásához használt visszalépéses algoritmus lépései a következők:

1. **Kezdőpont kijelölése**:
   Kiindulási pontként általában a labirintus bal felső sarkát (0,0) választjuk, de bármely más kezdőpont is megadható. A cél a labirintus jobb alsó sarka (N-1,N-1).

2. **Lépési lehetőségek**:
   Négy irányba lehet lépni: fel, le, balra, jobbra. Ezeket az irányokat irányvektorokkal lehet reprezentálni:

   ```cpp
   int rowNum[] = {-1, 1, 0, 0};
   int colNum[] = {0, 0, -1, 1};
   ```

3. **Visszalépéses keresés (Backtracking)**:
   A visszalépéses algoritmus rekurzív módon próbálja meg felfedezni a labirintus összes lehetséges útvonalát, amíg el nem éri a célt vagy ki nem derül, hogy az adott út nem vezet célhoz.

   ```cpp
   bool isValid(int maze[N][N], bool visited[N][N], int x, int y) {
       return (x >= 0 && x < N && y >= 0 && y < N && maze[x][y] == 1 && !visited[x][y]);
   }

   bool solveMazeUtil(int maze[N][N], int x, int y, int sol[N][N]) {
       if (x == N - 1 && y == N - 1) {
           sol[x][y] = 1;
           return true;
       }

       if (isValid(maze, visited, x, y)) {
           sol[x][y] = 1;
           visited[x][y] = true;

           for (int k = 0; k < 4; k++) {
               int newX = x + rowNum[k];
               int newY = y + colNum[k];
               if (solveMazeUtil(maze, newX, newY, sol))
                   return true;
           }

           sol[x][y] = 0;
           visited[x][y] = false;
       }

       return false;
   }

   bool solveMaze(int maze[N][N]) {
       int sol[N][N] = {0};

       if (!solveMazeUtil(maze, 0, 0, sol)) {
           cout << "No solution found" << endl;
           return false;
       }

       printSolution(sol);
       return true;
   }

   void printSolution(int sol[N][N]) {
       for (int i = 0; i < N; i++) {
           for (int j = 0; j < N; j++)
               cout << sol[i][j] << " ";
           cout << endl;
       }
   }
   ```

#### Implementáció

Az algoritmus megvalósítása során a következő lépéseket kell követnünk:

1. **Érvényes lépés ellenőrzése**:
   Az `isValid` függvény ellenőrzi, hogy az adott lépés érvényes-e, azaz a koordináták a labirintus határain belül vannak, a cella járható és még nem látogattuk meg.

2. **Rekurzív keresés**:
   A `solveMazeUtil` függvény rekurzívan próbál lépni minden lehetséges irányba, amíg el nem éri a célt. Ha egy lépés nem vezet célhoz, a visszalépés történik (backtracking), és a következő lehetséges irány próbálkozik.

3. **Eredmény kiírása**:
   Ha a megoldás megtalálható, a `printSolution` függvény kiírja a megoldást, amely megmutatja a helyes útvonalat a labirintusban.

Az algoritmus hatékonysága nagyban függ a labirintus méretétől és a benne található akadályok elhelyezkedésétől. Az egyszerűség és a viszonylagos könnyű implementáció miatt a backtracking algoritmus széles körben használt, de nagyobb méretű és bonyolultabb labirintusok esetén előfordulhat, hogy más megközelítések, például heurisztikák vagy optimalizálási technikák alkalmazása szükséges. Ezeket a technikákat a következő alfejezetben tárgyaljuk.

