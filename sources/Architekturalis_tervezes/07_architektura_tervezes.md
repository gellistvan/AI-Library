\newpage

## 7. **Architektúra tervezés**

Az architektúra tervezése a szoftverfejlesztési folyamat egyik legkritikusabb lépése, amely meghatározza a rendszer szerkezetét, komponenseit és azok interakcióit. Az architektúratípusok, mint például a monolitikus, a mikroszolgáltatások és a rétegezett architektúra, különböző megközelítéseket kínálnak a rendszer felépítésére és skálázására, mindegyik saját előnyökkel és kihívásokkal. Emellett az architektúrális minták, mint az MVC (Model-View-Controller), MVP (Model-View-Presenter), MVVM (Model-View-ViewModel), CQRS (Command Query Responsibility Segregation) és az Event Sourcing, olyan bevált megoldásokat kínálnak, amelyek segítenek az összetett problémák kezelésében és a rendszer karbantarthatóságának javításában. Ebben a fejezetben részletesen megvizsgáljuk ezen architektúratípusok és minták alapelveit, alkalmazási területeit és gyakorlati példáit, hogy átfogó képet nyújtsunk az architektúra tervezésének különböző aspektusairól.

### Architektúratípusok (monolitikus, mikroszolgáltatások, rétegezett architektúra)

Az architektúratípusok megértése és megfelelő alkalmazása kritikus szerepet játszik a szoftverfejlesztés sikerében. Az architektúra típusának megválasztása befolyásolja a rendszer skálázhatóságát, karbantarthatóságát, teljesítményét és a fejlesztési folyamat rugalmasságát. Ebben az alfejezetben három fő architektúratípust vizsgálunk meg részletesen: a monolitikus architektúrát, a mikroszolgáltatás-alapú architektúrát és a rétegezett architektúrát. Mindegyik típus saját előnyökkel és kihívásokkal rendelkezik, és különböző helyzetekben lehetnek optimálisak.

#### Monolitikus architektúra

A monolitikus architektúra egy olyan megközelítés, amelyben a szoftver összes komponense egyetlen, egységes alkalmazásban van összekapcsolva. Minden funkció és szolgáltatás egyetlen kódbázisban van megvalósítva, és az egész alkalmazás együtt kerül telepítésre és futtatásra.

**Előnyök**:

1. **Egyszerű fejlesztési környezet**: A fejlesztők számára könnyebb a kódbázis megértése és a fejlesztési környezet beállítása, mivel minden egy helyen található.
2. **Egyszerű telepítés**: Az alkalmazás telepítése egyszerűbb, mivel csak egyetlen alkalmazást kell kezelni.
3. **Közös memóriaterület**: A komponensek közvetlenül hozzáférhetnek egymás adataihoz, ami egyszerűsíti az adatok megosztását és a kommunikációt.

**Hátrányok**:

1. **Korlátozott skálázhatóság**: Az alkalmazás méretének növekedésével nehézségek merülhetnek fel a skálázásban, mivel minden komponens együtt nő.
2. **Karbantarthatósági problémák**: Nagyobb alkalmazások esetében a kód komplexitása növekszik, ami megnehezíti a karbantartást és a hibakeresést.
3. **Telepítési kockázatok**: Minden módosítás után az egész alkalmazást újra kell telepíteni, ami növeli a telepítési hibák és az üzemzavarok kockázatát.

**Példa**:
Egy e-kereskedelmi alkalmazás, amely egyetlen kódbázisban tartalmazza az összes funkciót, beleértve a termékkatalógust, a kosarat, a fizetési rendszert és a rendeléskezelést. Az alkalmazás egyetlen szerveren fut, és minden funkció közvetlenül kommunikál egymással.

#### Mikroszolgáltatások alapú architektúra

A mikroszolgáltatás-alapú architektúra olyan megközelítés, amelyben az alkalmazás különálló, függetlenül telepíthető szolgáltatásokból áll. Minden mikroszolgáltatás egyetlen üzleti funkciót valósít meg, és saját adatbázissal rendelkezhet. A szolgáltatások közötti kommunikációt tipikusan könnyűsúlyú protokollokkal, például HTTP vagy gRPC segítségével valósítják meg.

**Előnyök**:

1. **Skálázhatóság**: Az egyes szolgáltatások külön-külön skálázhatók, ami nagyobb rugalmasságot biztosít a rendszer erőforrásainak optimalizálásában.
2. **Rugalmas fejlesztés**: A fejlesztők különálló csapatokban dolgozhatnak a különböző szolgáltatásokon, ami gyorsabb fejlesztési ciklusokat és nagyobb autonómiát tesz lehetővé.
3. **Karbantarthatóság**: A kisebb, jól definiált szolgáltatások könnyebben karbantarthatók és tesztelhetők, ami csökkenti a hibakeresés és a javítás komplexitását.

**Hátrányok**:

1. **Komplexitás a kommunikációban**: A szolgáltatások közötti kommunikáció komplexitása növekszik, és szükség lehet robusztus kommunikációs mechanizmusokra és monitoring eszközökre.
2. **Telepítési kihívások**: A különálló szolgáltatások telepítése és kezelése bonyolultabb, különösen nagy méretű rendszerek esetében.
3. **Adatkonzisztencia**: Az adatok megosztása és a konzisztencia fenntartása kihívást jelenthet a különálló adatbázisok miatt.

**Példa**:
Egy e-kereskedelmi platform mikroszolgáltatás-alapú architektúrával, amely különálló szolgáltatásokra bontja a termékkatalógust, a kosarat, a fizetési rendszert és a rendeléskezelést. Minden szolgáltatás külön-külön skálázható, és a szolgáltatások közötti kommunikáció HTTP API-kon keresztül történik.

#### Rétegezett architektúra

A rétegezett architektúra, más néven n-rétegű vagy több rétegű architektúra, az alkalmazás logikáját különálló rétegekre bontja. Ezek a rétegek tipikusan a prezentációs réteg, az üzleti logika rétege és az adat-hozzáférési réteg.

**Előnyök**:

1. **Modularitás**: Az alkalmazás különálló rétegekre bontása elősegíti a modularitást és az újrafelhasználhatóságot.
2. **Karbantarthatóság**: A rétegek közötti világos elkülönítés megkönnyíti a kód karbantartását és tesztelését.
3. **Skálázhatóság**: A különálló rétegek különböző módon skálázhatók, például a prezentációs réteg horizontálisan, míg az adat-hozzáférési réteg vertikálisan.

**Hátrányok**:

1. **Teljesítménybeli korlátok**: A rétegek közötti kommunikáció hozzáadott rétegzéssel jár, ami befolyásolhatja a rendszer teljesítményét.
2. **Komplexitás**: A rétegek közötti függőségek és az adatáramlás komplexitása növekedhet, különösen nagy és összetett rendszerek esetében.
3. **Fejlesztési sebesség**: A rétegek közötti szigorú elkülönítés miatt a fejlesztési sebesség csökkenhet, mivel egy-egy réteg változásai más rétegeket is érinthetnek.

**Példa**:
Egy vállalati CRM rendszer rétegezett architektúrával, ahol a prezentációs réteg felelős a felhasználói felületért, az üzleti logika réteg kezeli az üzleti szabályokat és folyamatokat, míg az adat-hozzáférési réteg felelős az adatbázis kezeléssel kapcsolatos műveletekért.

#### Összehasonlítás és alkalmazási területek

Míg a monolitikus architektúra egyszerűsége miatt ideális lehet kisebb projektek vagy induló vállalkozások számára, ahol a fejlesztési és telepítési folyamatokat gyorsan kell végrehajtani, a mikroszolgáltatás-alapú architektúra nagyobb rendszerek és vállalati szintű alkalmazások esetében lehet előnyös, ahol a skálázhatóság és a fejlesztési autonómia kulcsfontosságú. A rétegezett architektúra pedig olyan alkalmazásoknál lehet optimális, ahol fontos a modularitás és az egyes rétegek külön-külön történő kezelése, például komplex vállalati rendszereknél.

Az architektúratípus megválasztása tehát nagyban függ a projekt méretétől, a szervezet céljaitól, a fejlesztési csapat tapasztalataitól és a technikai követelményektől. Az alapos tervezés és a megfelelő architektúra kiválasztása biztosítja, hogy a rendszer hosszú távon is megfeleljen az üzleti és technikai elvárásoknak, és fenntarthatóan működjön a változó környezetben.

### Architektúrális minták (MVC, MVP, MVVM, CQRS, Event Sourcing)

Az architektúrális minták a szoftverfejlesztés bevált megoldásai, amelyek segítenek az összetett problémák kezelésében és a rendszerek karbantarthatóságának, rugalmasságának és skálázhatóságának javításában. Ezek a minták meghatározott struktúrákat és elveket kínálnak a rendszer komponenseinek elrendezésére és azok közötti interakciók kezelésére. Ebben az alfejezetben részletesen bemutatjuk az MVC (Model-View-Controller), MVP (Model-View-Presenter), MVVM (Model-View-ViewModel), CQRS (Command Query Responsibility Segregation) és Event Sourcing mintákat, azok alkalmazási területeit és gyakorlati példáit.

#### Model-View-Controller (MVC)

Az MVC egy elterjedt architektúrális minta, amely a szoftveralkalmazásokat három fő komponensre bontja: a modellre, a nézetre és a vezérlőre.

**Komponensek**:

- **Modell (Model)**: Kezeli az alkalmazás adatait és az üzleti logikát. A modell felelős az adatok lekérdezéséért és frissítéséért, valamint az üzleti szabályok alkalmazásáért.
- **Nézet (View)**: Felelős az adatok megjelenítéséért a felhasználó számára. A nézet lekéri az adatokat a modelltől és megjeleníti azokat a felhasználói felületen.
- **Vezérlő (Controller)**: Kezeli a felhasználói bemeneteket, és ezek alapján irányítja a modell és a nézet közötti kommunikációt. A vezérlő meghatározza, hogyan reagáljon a rendszer a felhasználói interakciókra.

**Előnyök**:

- Tiszta elkülönítés az üzleti logika és a megjelenítés között.
- Könnyebb karbantarthatóság és tesztelhetőség.
- Egyszerűbb a párhuzamos fejlesztés, mivel a különböző komponensek függetlenül fejleszthetők.

**Példa**:
Egy webes alkalmazás, amely termékeket jelenít meg egy online áruházban. A modell tartalmazza a termékek adatait és az adatbázis kezelést, a nézet a termékeket listázza egy HTML oldalon, míg a vezérlő kezeli a felhasználói keresési lekérdezéseket és a szűrési opciókat.

#### Model-View-Presenter (MVP)

Az MVP egy másik népszerű architektúrális minta, amely hasonló az MVC-hez, de különös hangsúlyt fektet a nézet és az üzleti logika közötti kapcsolatra. Az MVP mintában a vezérlő szerepét a presenter veszi át.

**Komponensek**:

- **Modell (Model)**: Kezeli az adatokat és az üzleti logikát, mint az MVC-ben.
- **Nézet (View)**: Felelős az adatok megjelenítéséért és a felhasználói interakciók fogadásáért. A nézet interfészként szolgál a presenter felé.
- **Presenter (Presenter)**: Kezeli az üzleti logikát és a nézet közötti interakciót. A presenter kommunikál a modellel, és frissíti a nézetet a kapott adatokkal.

**Előnyök**:

- Tiszta elválasztás a nézet és a logika között.
- Könnyebb tesztelhetőség, mivel a nézet könnyen mockolható.
- Rugalmasabb, mint az MVC, mivel a presenter közvetlenül frissítheti a nézetet.

**Példa**:
Egy mobilalkalmazás, amely felhasználói profilokat jelenít meg. A modell tartalmazza a felhasználói adatokat, a nézet a felhasználói felületet jeleníti meg, míg a presenter kezeli az adatok betöltését és a nézet frissítését a felhasználói interakciók alapján.

#### Model-View-ViewModel (MVVM)

Az MVVM minta különösen népszerű a WPF (Windows Presentation Foundation) és más modern felhasználói interfészek fejlesztése során. Az MVVM célja, hogy adatközpontú alkalmazásokat hozzon létre, ahol a nézet és a modell közötti kapcsolatot a ViewModel biztosítja.

**Komponensek**:

- **Modell (Model)**: Az alkalmazás adatait és logikáját kezeli.
- **Nézet (View)**: A felhasználói interfész, amely adatokat jelenít meg.
- **ViewModel (ViewModel)**: A nézet logikáját kezeli és adatközlést biztosít a nézet és a modell között. A ViewModel a nézetet adatközléssel (data binding) frissíti.

**Előnyök**:

- Erős támogatás az adatközléssel (data binding) és a felhasználói felület frissítésével kapcsolatban.
- Könnyen tesztelhető, mivel a ViewModel elkülönül a nézettől.
- Nagyobb modularitás és újrafelhasználhatóság.

**Példa**:
Egy asztali alkalmazás, amely valós idejű pénzügyi adatokat jelenít meg. A modell a pénzügyi adatokat kezeli, a nézet a felhasználói interfészt biztosítja, míg a ViewModel adatközléssel frissíti a nézetet a modellből származó adatok alapján.

#### Command Query Responsibility Segregation (CQRS)

A CQRS egy architektúrális minta, amely elválasztja az adatkezelési műveleteket két különálló részre: a parancsokra (commands) és a lekérdezésekre (queries). Ez a minta különösen hasznos összetett üzleti logikával és nagy teljesítményű rendszerekkel rendelkező alkalmazások esetében.

**Komponensek**:

- **Commands (Parancsok)**: Az adat módosítását végzik. Minden parancs egy konkrét műveletet reprezentál, például "hozzáadás", "frissítés" vagy "törlés".
- **Queries (Lekérdezések)**: Az adat lekérdezését végzik, és nem módosítják az adatokat. Ezek a műveletek csak olvasási műveletek végrehajtására szolgálnak.

**Előnyök**:

- Különállóan optimalizálhatók az írási és olvasási műveletek.
- Támogatja a skálázhatóságot és a teljesítmény optimalizálását.
- Jobb elkülönítést biztosít a komplex üzleti logika kezelésében.

**Példa**:
Egy banki rendszer, amely különálló szolgáltatásokat használ a tranzakciók rögzítésére (parancsok) és a számlaegyenlegek lekérdezésére (lekérdezések). A parancsok felelősek a tranzakciók végrehajtásáért és a számlaadatok frissítéséért, míg a lekérdezések biztosítják a valós idejű adatok elérését az ügyfelek számára.

#### Event Sourcing

Az Event Sourcing egy olyan architektúrális minta, amely az alkalmazás állapotának kezelését események sorozataként valósítja meg. Az események az adatbázisban tárolódnak, és a rendszer állapotának helyreállítása ezen események újrajátszásával történik.

**Komponensek**:

- **Events (Események)**: Az alkalmazásban történt változások reprezentálása. Minden esemény egy specifikus műveletet reprezentál, például "felhasználó létrehozása" vagy "tranzakció végrehajtása".
- **Event Store**: Az események tárolására szolgáló adatbázis.
- **Aggregates**: Az eseményekből rekonstruált objektumok, amelyek az alkalmazás aktuális állapotát reprezentálják.

**Előnyök**:

- Történeti adatok teljes körű nyilvántartása és auditálhatósága.
- Könnyű állapotrekonstrukció és hibakeresés.
- Támogatja a komplex üzleti logika és a tranzakciók kezelését.

**Hátrányok**:

- Az események tárolása és kezelése komplexitást adhat a rendszerhez.
- Az események újrajátszása időigényes lehet nagy adatmennyiség esetén.
- Az események szigorú verziókövetése szükséges a rendszer evolúciója során.

**Példa**:
Egy rendelési rendszer, amely minden rendelés státuszváltozását eseményként rögzíti. Az események magukban foglalhatják a "rendelés létrehozása", "rendelés feladása" és "rendelés kézbesítése" műveleteket. Az Event Store tárolja ezeket az eseményeket, és szükség esetén rekonstruálja az aktuális rendelési állapotot.

#### Összefoglalás

Az architektúrális minták alkalmazása lehetőséget ad a fejlesztőknek, hogy jól bevált megoldásokkal kezeljék az összetett problémákat és javítsák a rendszer karbantarthatóságát, skálázhatóságát és rugalmasságát. Az MVC, MVP és MVVM minták különböző megközelítéseket kínálnak a felhasználói felület és az üzleti logika kezelésére, míg a CQRS és az Event Sourcing a nagy teljesítményű és komplex üzleti logikával rendelkező rendszerek esetében nyújtanak hatékony megoldásokat. A megfelelő minta kiválasztása és alkalmazása biztosítja, hogy a szoftver megfeleljen az üzleti és technikai követelményeknek, és hosszú távon is fenntartható legyen.



