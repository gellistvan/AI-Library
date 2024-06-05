\newpage

## 6.5. Legkisebb és legnagyobb elem keresése

Az algoritmusok világában az egyik leggyakoribb probléma az adathalmazok szélsőségeinek, azaz a legkisebb és legnagyobb elemeinek megtalálása. Ezek az értékek kulcsszerepet játszanak számos alkalmazásban, a statisztikai elemzésektől kezdve a különféle optimalizálási algoritmusokig. Ebben a fejezetben az oszd-meg-és-uralkodj paradigma alkalmazását vizsgáljuk a legkisebb és legnagyobb elemek keresésére. Megismerkedünk az alapvető algoritmussal és annak implementációs részleteivel, majd rátérünk a teljesítmény elemzésére és más meglévő módszerekkel való összehasonlítására. Az algoritmus részleteinek feltárása lehetőséget kínál arra, hogy megértsük és értékeljük az oszd-meg-és-uralkodj módszerek előnyeit és kihívásait ezekben a gyakran előforduló feladatokban.

## 6.5. Legkisebb és legnagyobb elem keresése

### 6.5.1 Algoritmus és implementáció

Az optimális megoldások keresése a legkisebb és legnagyobb elem megtalálására egy tömbben az Oszd-meg-és-uralkodj (Divide-and-Conquer) stratégia alapján nemcsak hogy érdekes, de elengedhetetlen is egy algoritmusokat részletező könyv szempontjából. Az ilyen algoritmusok különösen érdekesek, mivel híresek hatékonyságukról és elegáns megközelítésükről.

Az Oszd-meg-és-uralkodj stratégia három fő lépésből áll:
1. **Felbontás (Divide)**: Az eredeti problémát kisebb, hasonló jellegű részekre bontjuk.
2. **Feloldás (Conquer)**: A kisebb részeket feloldjuk rekurzív módon.
3. **Egyesítés (Combine)**: Az alproblémák megoldásait összekombináljuk, hogy az eredeti problémára kapjunk választ.

#### Algoritmus

1. **Felbontás (Divide)**: Egészen addig bontsuk a tömböt, amíg egyetlen vagy kettő elem marad.
    - Ha a tömb mérete egy, akkor az az egy elem egyszerre a minimum és maximum is.
    - Ha a tömb mérete kettő, akkor egyszerűen összehasonlítjuk a két elemet és meghatározzuk a minimumot és a maximumot.

2. **Feloldás (Conquer)**: Rekurzív módon oldjuk fel az alcsoportokat, találva a minimumot és a maximumot az alkömbökben.

3. **Egyesítés (Combine)**: Az alkömbök eredményeit összehasonlítva meghatározzuk a teljes tömb minimumát és maximumát.

#### Részletes algoritmus

Vizsgáljuk meg részletesen az algoritmus logikáját követve a fenti lépéseket.

##### Algoritmus Leírás

- Tegyük fel, hogy van egy tömb **A[0...n-1]**.
- A célunk az, hogy megtaláljuk a minimum és maximum értéket ebben a tömbben.
- A tömb két részre (vagy több részre) osztható. Tegyük jelenleg egyszerűen két részre, amelyet feloldhatunk rekurzívan.
- A bal oldali al-tömb: **A[0...mid]**
- A jobb oldali al-tömb: **A[mid+1...n-1]**

Minden felbontott al-tömbre megtaláljuk a minimum és a maximum értékeket, majd egyesítjük azokat az eredeti tömb minimum és maximum értékeinek meghatározása érdekében.

##### Implementáció

Most nézzük egy lehetséges C++ nyelven írt példakódot az előzőekben tárgyalt algoritmus alapján:

```cpp
#include <iostream>

#include <climits>
using namespace std;

struct MinMax {
    int min;
    int max;
};

MinMax findMinMax(int arr[], int low, int high) {
    MinMax result, leftResult, rightResult;
    
    // Ha csak egy elem van a tömbben
    if (low == high) {
        result.min = result.max = arr[low];
        return result;
    }
    
    // Ha két elem van a tömbben
    if (high == low + 1) {
        if (arr[low] < arr[high]) {
            result.min = arr[low];
            result.max = arr[high];
        } else {
            result.min = arr[high];
            result.max = arr[low];
        }
        return result;
    }
    
    // Középpont meghatározása
    int mid = (low + high) / 2;
    
    // Rekurzív megoldás a bal és jobb alkömbökben
    leftResult = findMinMax(arr, low, mid);
    rightResult = findMinMax(arr, mid + 1, high);
    
    // Kombinálás: összehasonlítjuk a két alkömbök eredményeit
    result.min = min(leftResult.min, rightResult.min);
    result.max = max(leftResult.max, rightResult.max);
    
    return result;
}

int main() {
    int arr[] = {1000, 11, 445, 1, 330, 3000};
    int arr_size = sizeof(arr)/sizeof(arr[0]);
    
    MinMax result = findMinMax(arr, 0, arr_size - 1);
    
    cout << "Minimum element: " << result.min << endl;
    cout << "Maximum element: " << result.max << endl;
    
    return 0;
}
```

##### További Részletek az Implementációról

A fenti kód egy ```MinMax``` nevű struktúrát használ, hogy együtt tárolja a minimális és maximális értékeket. A `findMinMax` függvény rekurzívan határozza meg a minimumot és a maximumot az adott tömb al-tömbjében. Ahogy ezt az algoritmust végrehajtjuk, minden szinten két rekurzív hívással visszatérünk a két al-tömb minimum és maximum értékeivel, amelyek közül a végső minimumot és maximumot a fő tömbből válaszuk ki.

Ez a megoldás kihasználja az Oszd-meg-és-uralkodj technikát, és az időkomplexitása **O(n)**, ahol **n** a tömb elemeinek száma. Ennek az az oka, hogy minden lépésben a tömb két részre oszlik, és minden egyes szintig minden elem részt vesz pontosan egyszer a minimális és maximális értékek meghatározásában.

Ezért a bemutatott algoritmus hatékony és jól struktúrált módszer a legkisebb és legnagyobb elem keresésére egy tömbben az Oszd-meg-és-uralkodj módszerrel. Ahogy azt a következő fejezetben látjuk, ez a megközelítés nemcsak hatékony, hanem összehasonlításra is alkalmas más hasonló algoritmusokkal, hogy hangsúlyozzuk ennek a technikának a hatékonyságát és alkalmazhatóságát különféle problémák megoldására.

## 6.5.2 Teljesítmény elemzés és összehasonlítás

### Bevezetés

Az "oszd meg és uralkodj" stratégia, vagy angolul "divide and conquer", az algoritmuselmélet egyik központi technikája, amely azzal éri el hatékonyságát, hogy a problémát kisebb részekre bontja, majd ezeket a részproblémákat külön-külön oldja meg, és végül az eredményeket kombinálja az eredeti probléma megoldása érdekében. Ebben a fejezetben bemutatjuk, hogyan lehet a legkisebb és legnagyobb elemet keresni egy tömbben vagy listában az oszd-meg-és-uralkodj módszer segítségével, majd részletesen elemezzük és összehasonlítjuk az algoritmusok teljesítményét más, egyszerűbb megközelítésekkel.

#### Alapvető megközelítések

A legkisebb és legnagyobb elem keresésének legalapvetőbb módja az, hogy az elemeket a tömbben egyenként végigvizsgáljuk, és a legkisebb, illetve legnagyobb elemet egy egyszerű iteratív algoritmussal határozzuk meg. Az egyszerű szekvenciális keresési (brute-force) algoritmusnak az időkomplexitása O(n), ahol n a tömb elemeinek száma. Az oszd-meg-és-uralkodj technikával azonban az időkomplexitást optimalizálhatjuk, különösen nagyobb méretű adathalmazok esetén.

#### Algoritmus

Az oszd-meg-és-uralkodj algoritmus a legkisebb és legnagyobb elem keresése esetében a következő lépéseket tartalmazza:
1. **Felbontás (Divide):** Az eredeti tömböt két részre osztjuk.
2. **Alproblémák megoldása (Conquer):** Rekurzívan meghatározzuk mindkét rész tömb legkisebb és legnagyobb elemét.
3. **Kombinálás (Combine):** A két rész tömb legkisebb és legnagyobb elemei közül kiválasztjuk az abszolút legkisebb és legnagyobb elemeket.

Az algoritmus pseudokódja a következőképpen néz ki:

```cpp
struct MinMax {
    int min;
    int max;
};

MinMax findMinMax(vector<int>& arr, int low, int high) {
    MinMax result;

    // Base case: only one element
    if (low == high) {
        result.min = arr[low];
        result.max = arr[low];
        return result;
    }

    // Base case: only two elements
    if (high == low + 1) {
        if (arr[low] < arr[high]) {
            result.min = arr[low];
            result.max = arr[high];
        } else {
            result.min = arr[high];
            result.max = arr[low];
        }
        return result;
    }

    // Divide
    int mid = (low + high) / 2;
    MinMax left = findMinMax(arr, low, mid);
    MinMax right = findMinMax(arr, mid + 1, high);

    // Conquer
    result.min = min(left.min, right.min);
    result.max = max(left.max, right.max);

    return result;
}
```

Ebben a kódban a `findMinMax` függvény rekurzívan osztja két részre a tömböt, majd a kisebb részeken meghívja magát, és végül összesíti az alsóbb szinteken kapott eredményeket, hogy megkapja az egész tömb legkisebb és legnagyobb elemét.

#### Időkomplexitás

Az időkomplexitás elemzéséhez tekintsük az algoritmus rekurzív természetét. Az algoritmus minden hívásában két részre osztjuk a tömböt, ezért a rekurzív hívások száma logaritmikus (log n). Az összehasonlítások száma viszont minden rekurzív lépésben lineáris, O(1) idő alatt végezhető el. A rekurrencia reláció a következőképpen alakul:

$$
T(n) = 2T\left(\frac{n}{2}\right) + 2
$$

Ez a reláció hasonló a bináris keresés rekurrencia relációjához, amelynek megoldása $T(n) = O(n)$. Az oszd-meg-és-uralkodj módszer tehát ugyanúgy lineáris időben működik, mint a szekvenciális keresési algoritmus, de a konstans időszükséglet kisebb lehet, mert egyszerre több számítást is végrehajthat.

#### Összehasonlítás más megközelítésekkel

A szekvenciális keresési algoritmus, amely egyszerű iterációval határozza meg a legkisebb és legnagyobb elemet, szintén O(n) időkomplexitást mutat, de ebben az esetben minden egyes elem vizsgálata során két összehasonlítást kell végeznünk (egy a minimumra, egy a maximumra). Az oszd-meg-és-uralkodj módszer ezzel szemben hatékonyabb lehet nagyobb adathalmazok esetében, különösen ha a rekurzív hívások párhuzamosan futtathatók.

Az oszd-meg-és-uralkodj algoritmus teljesítménye nagyban függ az implementáció részleteitől és az adathalmaz méretétől. Azon esetekben, amikor a részproblémák párhuzamosan megoldhatók (például párhuzamos feldolgozási környezetben), az oszd-meg-és-uralkodj módszer jelentős előnyökkel járhat.

### Gyakorlati alkalmazások

Gyakorlatban a választás az algoritmusok között nagymértékben a probléma konkrét részleteitől függ. Például:
- Nagy, elosztott rendszerekben gyakran párhuzamosan futtatható algoritmusokkal oldjuk meg a problémákat, így az oszd-meg-és-uralkodj módszer kifejezetten előnyös lehet.
- Kisebb méretű adathalmazok esetében az egyszerű, szekvenciális keresési algoritmus is elegendő lehet.

