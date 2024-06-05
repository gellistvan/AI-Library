\newpage

# 4. Hash függvények és integritás ellenőrzés

A digitális információ korában az adatbiztonság és integritás megőrzése minden eddiginél fontosabb. Ennek egyik alapvető eszköze a hash függvények használata, amelyek elengedhetetlenek mind a kriptográfiában, mind az adatkezelés különböző területein. Ezek a matematikai függvények egy adott bemenő adatból fix hosszúságú, látszólag véletlenszerű karakterláncot generálnak, amit "hash értéknek" vagy "kivonatnak" nevezünk. A hash függvények alkalmazásával lehetőség nyílik az adatok gyors és hatékony azonosítására, valamint integritásuk ellenőrzésére, különösen nagy adatbázisok vagy adatfolyamok esetén. Ezen bevezetésben részletesen megvizsgáljuk, hogyan működnek ezek a függvények, milyen szerepet játszanak a kriptográfiai protokollokban, és miként segítik az adatvédelmet és -integritást a mindennapi számítástechnikai műveletek során.

## 4.1. Alapelvek és definíciók

A modern kriptográfiában a hash függvények központi szerepet játszanak az adatok integritásának és biztonságának megőrzésében. Ezek a függvények olyan algoritmusok, amelyek tetszőleges hosszúságú bemenetet vesznek, és abból egy fix hosszúságú kimenetet, úgynevezett hash értéket generálnak. A hash függvények alapvető tulajdonságai közé tartozik, hogy kis változtatás a bemeneti adatban jelentős változást eredményez a kimenetben, valamint hogy ugyanaz a bemenet mindig ugyanazt a kimenetet eredményezi. E tulajdonságok teszik őket különösen alkalmassá az integritás ellenőrzésére és adatbiztonsági célokra. Ebben a fejezetben részletesen megvizsgáljuk, hogyan működnek a hash függvények, milyen tulajdonságokkal rendelkeznek, és hogyan nyújtanak védelmet az adatütközés, a pre-image támadások és a második pre-image támadások ellen. Az alábbiakban bemutatjuk a hash függvények alapelveit és a velük kapcsolatos leggyakoribb fogalmakat.

### Hash függvények működése

#### Áttekintés

A hash függvények folyamata egy bemeneti adatot (üzenet, fájl, adatstruktúra stb.) vesz, és abból egy fix hosszúságú, általában rövidebb karakterláncot vagy számsorozatot állít elő, amit hash értéknek (vagy egyszerűen hash-nek) nevezünk. A hash függvényeket általában a következőképpen jelöljük:
\[ H: \mathbb{M} \rightarrow \mathbb{D} \]
ahol \( \mathbb{M} \) a bemeneti értékek halmaza, és \(\mathbb{D}\) a lehetséges hash értékek halmaza. A \(|\mathbb{D}|<<|\mathbb{M}|\), ami azt jelenti, hogy a kimeneti tartomány sokkal kisebb, mint a bemeneti tartomány.

#### Főbb lépések

1. **Bemenet előkészítése (Padding):** Az üzenetek gyakran különböző hosszúságúak, ezért szükséges, hogy ezek a bemenetek egy meghatározott blokkmérettel kompatibilisek legyenek. Ez többnyire kiegészítéssel (padding) történik, hogy az üzenet hossza osztható legyen a blokkmérettel.

2. **Belső állapot inicializálása:** Sok modern hash függvény használ egy belső állapotot, amelyet kezdetben egy prédeterminált értékekkel töltenek fel. Ez az állapot értéke minden feldolgozó körben módosul.

3. **Tömörítési függvény:** A bemenetet több blokkba osztják, és mindegyik blokk külön-külön feldolgozásra kerül egy tömörítési függvényen keresztül. Ez a tömörítési függvény a bemeneti blokkot és az aktuális belső állapotot felhasználva egy új belső állapotot generál.

4. **Kimeneti kivonat készítése:** Miután az összes blokk feldolgozásra került, az utolsó belső állapotból képezik a hash értéket, amely a végleges kimeneti érték lesz.

### Hash függvények tulajdonságai

A hatékony hash függvényeknek számos fölöttébb fontos tulajdonsággal kell rendelkezniük, hogy különféle alkalmazásokban megbízhatóan működjenek.

#### Determinizmus

Ez alapvető követelmény, amely szerint egy bemenet (üzenet) mindig ugyanazt a hash értéket adja vissza. Formálisan:
\[ \forall x \in \mathbb{M}, \, H(x) = H(x) \]

#### Gyors számítás

Egy jó hash függvénynek hatékonyan, gyorsan kell működnie, még akkor is, ha a bemeneti adat nagyon hosszú. Ez különösen fontos azokban az alkalmazásokban, amelyek nagy mennyiségű adatot igényelnek, például adatbázisok kezelése vagy fájlok integritásának ellenőrzése során.

#### Ütközésállóság (Collision Resistance)

Az ütközésállóság azt jelenti, hogy nagyon nehéz két különböző bemeneti értéket találni, amelyek ugyanazon hash értéket adják vissza. Formálisan:
\[ H(x) = H(y) \implies x = y \]

#### Gyenge pre-image ellenállás

A gyenge pre-image ellenállás azt biztosítja, hogy egy adott hash értékből visszafejteni a kiinduló bemeneti adatot gyakorlatilag lehetetlen. Formálisan, egy adott \( h \) hash érték esetén:
\[ \text{Given } h, \text{ it is computationally infeasible to find } x \text{ such that } H(x) = h \]

#### Erős pre-image ellenállás (Second Pre-image Resistance)

Az erős pre-image ellenállás hasonló a gyenge pre-image ellenálláshoz, de itt az a követelmény, hogy egy adott bemeneti adat \( x \) és annak hash értéke \( H(x) \) ismeretében gyakorlatilag lehetetlen legyen egy második, különböző bemeneti adat \( x' \) találása, amelyre \( H(x) = H(x') \). Formálisan:
\[ \text{Given } x, \text{ it is computationally infeasible to find } x' \neq x \text{ such that } H(x) = H(x') \]

#### Pseuso-random generáció

Egy jó hash függvény kimenete olyan tulajdonsággal kell rendelkezzen, amely hasonló a véletlenszerű generáláshoz. Ez azt jelenti, hogy kis bemeneti változások jelentős kimeneti változásokat eredményeznek, amely elv a lavina hatás néven ismert.

#### Példa Hash Függvényre (C++)

Bár a kriptográfiai hash függvények implementációja igen bonyolult és nagy szakértelmet igényel, itt egy egyszerűbb, nem-kriptográfiai hash függvény példáját mutatjuk be, hogy képet kapjunk az alapötletekről.

```cpp
#include <iostream>

#include <string>

unsigned int simpleHashFunction(std::string input) {
    unsigned int hash = 0;
    for (char c : input) {
        hash = hash * 31 + c;
    }
    return hash;
}

int main() {
    std::string input = "Hello, world!";
    unsigned int hashValue = simpleHashFunction(input);
    std::cout << "Hash value: " << hashValue << std::endl;
    return 0;
}
```
Ez az egyszerű példa bemutatja, hogy egy hash függvény hogyan hozz létre egy kimeneti értéket az összes karakter ASCII-kódja alapján, és egy egyszerű tömörítési metódussal (multiplikatív hash függvény) állít elő egy hash értéket.

### Ütközés, pre-image és második pre-image ellenállás

A hash függvények elméletében és alkalmazásaiban az egyik legfontosabb követelmény a bizonyos ellenállások megléte a különböző támadási módszerekkel szemben. Három alapvető koncepcióra koncentrálunk ebben a fejezetben: ütközési ellenállás (collision resistance), pre-image ellenállás (pre-image resistance) és második pre-image ellenállás (second pre-image resistance). Ezek az ellenállási tulajdonságok kritikusak a biztonságos kriptográfiai rendszerek megtervezésében és megvalósításában.

#### Ütközés (Collision) és Ütközési Ellenállás (Collision Resistance)

Az ütközés olyan szituáció, amikor egy hash függvény két különböző bemenethez ugyanazt a hash értéket rendeli. Formálisan, ha H egy hash függvény, akkor a H(x) = H(y) igaz, ahol x ≠ y és x, y a bemeneti értékek.

**Ütközési ellenállás** egy hash függvény azon tulajdonsága, hogy nehéz legyen két különböző bemenetet találni, amelyekhez ugyanazt a hash értéket rendeli. Például, egy hash függvény H ütközési ellenálló, ha nincs hatékony algoritmus, amely két különböző x és y elemet talál, amire H(x) = H(y).

A gyakorlatban ez azt jelenti, hogy ha a függvény n-bites hash értéket állít elő, akkor az ütközés megtalálásának költsége exponenciális legyen az n-ben. Például, egy 256 bites hash függvény esetében az ütközések megtalálásának bonyolultsága O(2^128), ami nagyon nagy szám és így a függvény elég biztonságos a gyakorlatban.

#### Pre-image Ellenállás (Pre-image Resistance)

A pre-image a hash függvény eredetéhez tartozó bemeneti értéket jelenti. Ha van egy hash érték y, akkor a pre-image x olyan bemeneti érték, amire a hash függvény H(x) = y.

**Pre-image ellenállás** az a tulajdonság, hogy egy adott hash kimeneti érték y esetén nagyon nehéz megtalálni egy olyan x bemeneti értéket, amelyre H(x) = y. Formálisan, adott y esetén nehéz x-et találni, amelyre H(x) = y.

Az pre-image ellenállásra vonatkozó elvárás az, hogy a bemenet (pre-image) keresése exponenciális legyen a hash függvény bit hosszában. Például, egy 256 bites hash függvény esetén az pre-image megtalálása O(2^256) lépésekben történjen.

#### Második Pre-image Ellenállás (Second Pre-image Resistance)

A második pre-image ellenállás ahhoz kapcsolódik, hogy egy adott bemeneti érték hash képéhez nehéz egy második eltérő bemeneti értéket találni, ami ugyanazt a hash értéket adja. Ha x egy adott bemenet, akkor nehéz másik y értéket találni, amely különbözik x-től (x ≠ y) és H(y) = H(x).

**Második pre-image ellenállás** tehát azt jelenti, hogy egy adott x esetén nagyon nehéz egy második y-t találni, amely különbözik x-től, de H(x) = H(y). Ez különbözik a pre-image ellenállástól, ahol csak az adott kimenethez tartozó bármely bemenet megtalálása a cél.

A második pre-image ellenállás elvárása, hogy a második pre-image hatékony megtalálása költséges legyen, hasonlóan az ütközési ellenálláshoz. Például, egy 256 bites hash függvény esetén a második pre-image megtalálása O(2^256) lépésekben történjen.

#### Példa Hash Függvény Implementáció C++-ban

Az alábbiakban egy egyszerű hash függvény C++ implementációját mutatjuk be szemléltetésként. Ez nem egy kriptográfiailag biztonságos hash függvény, hanem inkább oktatási célokat szolgál.

```cpp
#include <iostream>

#include <string>

class SimpleHash {
public:
    unsigned long long hash(const std::string& input) {
        unsigned long long hashValue = 0;
        for (char c : input) {
            hashValue = (hashValue * 31) + c;
        }
        return hashValue;
    }
};

int main() {
    SimpleHash hasher;
    std::string input1 = "hello";
    std::string input2 = "world";
    std::string input3 = "hello";

    std::cout << "Hash of '" << input1 << "': " << hasher.hash(input1) << std::endl;
    std::cout << "Hash of '" << input2 << "': " << hasher.hash(input2) << std::endl;
    std::cout << "Hash of '" << input3 << "': " << hasher.hash(input3) << std::endl;

    return 0;
}
```

Ez a kód egy nagyon egyszerű hash függvényt definiál, amely egy adott bemenethez több 31 szorzással és összegzéssel hash értéket képez. Habár ez a példa „valami hash” függvényként működik, nem felel meg a kriptográfiai követelményeknek a pre-image, második pre-image és ütközési ellenállás szempontjából.

#### Összefoglalás

A hash függvények kulcsfontosságú szerepet játszanak a kriptográfiai rendszerekben, és megértésük alapvető ahhoz, hogy biztonságos rendszereket tervezhessünk és valósíthassunk meg. Az ütközés, pre-image és második pre-image ellenállás mind lényeges tulajdonságok, amelyek biztosítják, hogy a hash függvények hatékonyak és biztonságosak legyenek a gyakorlatban. Az említett ellenállási szintek nélkül a hash függvények könnyen támadhatóak lennének, veszélyeztetve ezzel a kriptográfiai rendszerek biztonságát.

