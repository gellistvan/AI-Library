\newpage

## 5. Követelményanalízis

A követelményanalízis a szoftverfejlesztési folyamat egyik legkritikusabb szakasza, amely meghatározza a szoftver által teljesítendő funkciókat és követelményeket. Ez a fejezet részletesen bemutatja a követelménygyűjtés és dokumentáció fontosságát, módszereit és eszközeit, valamint a use case-ek és user story-k szerepét a felhasználói igények és üzleti célok pontos megfogalmazásában. A követelményanalízis révén biztosítható, hogy a fejlesztők és az érintett felek közös megértésre jussanak a szoftver elvárásairól, elősegítve ezzel a projektek sikeres és hatékony megvalósítását.

### Követelménygyűjtés és dokumentáció

A követelménygyűjtés és dokumentáció a szoftverfejlesztési életciklus első és talán legfontosabb lépése, amely meghatározza a projekt sikerét. E folyamat során a fejlesztők és az érintett felek együttműködnek, hogy pontosan megértsék és rögzítsék a szoftverrel szemben támasztott elvárásokat és igényeket. A jól definiált és dokumentált követelmények biztosítják, hogy a fejlesztők pontosan azt szállítsák, amit a felhasználók és a megrendelők igényelnek, minimalizálva ezzel a félreértések és hibák lehetőségét a későbbi fázisokban.

#### Követelménygyűjtés folyamata

A követelménygyűjtés több lépésből álló folyamat, amely magában foglalja az információgyűjtést, az elemzést, az érvényesítést és a dokumentálást. Az alábbiakban részletesen ismertetjük e lépéseket:

1. **Információgyűjtés**: Az első lépés a releváns információk összegyűjtése a szoftver felhasználóitól, megrendelőitől és más érintett felektől. Ez történhet különböző módszerek segítségével, mint például:
    - **Interjúk**: Személyes vagy csoportos interjúk során a fejlesztők közvetlenül kérdezik meg az érintetteket az igényeikről és elvárásaikról.
    - **Kérdőívek és felmérések**: Strukturált kérdőívek segítségével széles körben gyűjthetők össze a követelmények.
    - **Workshopok és megbeszélések**: Közös munkamegbeszélések során az érintettek együttműködve határozzák meg a követelményeket.
    - **Megfigyelés**: A fejlesztők megfigyelhetik a felhasználók munkafolyamatait és tevékenységeit, hogy jobban megértsék a gyakorlati igényeket.

2. **Elemzés**: Az összegyűjtött információk elemzése során a fejlesztők azonosítják a különböző követelményeket, kategorizálják őket és meghatározzák azokat az összefüggéseket és ellentmondásokat, amelyek felmerülhetnek. Az elemzés célja, hogy világos és részletes képet adjon a szoftverrel szemben támasztott elvárásokról.

3. **Érvényesítés**: Az elemzett követelményeket érvényesíteni kell az érintett felekkel, hogy biztosítsák azok helyességét és teljességét. Az érvényesítési folyamat során az érintettek visszajelzéseket adhatnak, és a fejlesztők módosíthatják a követelményeket az észrevételek alapján.

4. **Dokumentálás**: A véglegesített követelményeket részletesen dokumentálni kell. A követelményspecifikáció (Software Requirements Specification, SRS) egy hivatalos dokumentum, amely tartalmazza a szoftverrel szemben támasztott összes funkcionális és nem-funkcionális követelményt. Az SRS szolgál alapul a fejlesztés, tesztelés és karbantartás során.

#### Követelménytípusok

A követelmények két fő kategóriába sorolhatók: funkcionális és nem-funkcionális követelmények.

1. **Funkcionális követelmények**: Ezek a követelmények leírják, hogy a szoftvernek milyen funkciókat és szolgáltatásokat kell biztosítania. Például:
    - A felhasználóknak lehetőséget kell biztosítani a regisztrációra és bejelentkezésre.
    - A rendszernek képesnek kell lennie az adatok importálására és exportálására.
    - A felhasználóknak jelentéseket kell tudniuk generálni a rendszer adatainak alapján.

2. **Nem-funkcionális követelmények**: Ezek a követelmények a szoftver minőségi jellemzőire és teljesítményére vonatkoznak. Például:
    - A rendszernek legalább 1000 egyidejű felhasználót kell kiszolgálnia.
    - Az adatokat titkosítva kell tárolni és továbbítani.
    - A rendszernek 99,9%-os rendelkezésre állással kell működnie.

#### Példák a követelménygyűjtésre és dokumentációra

**Példa 1**: Egy e-kereskedelmi webáruház fejlesztése

- **Információgyűjtés**: Interjúk a potenciális felhasználókkal, workshopok az üzleti partnerekkel, és kérdőívek az online vásárlási szokásokról.
- **Elemzés**: Az összegyűjtött információk alapján azonosítják a követelményeket, mint például a termékkeresés, a kosárkezelés és a fizetési módok.
- **Érvényesítés**: Az üzleti partnerekkel és felhasználókkal történő egyeztetés során pontosítják a követelményeket.
- **Dokumentálás**: Az SRS dokumentum tartalmazza a funkcionális követelményeket (pl. termékkatalógus böngészése) és a nem-funkcionális követelményeket (pl. oldalbetöltési sebesség).

**Példa 2**: Egy vállalati CRM rendszer fejlesztése

- **Információgyűjtés**: Megfigyelés a jelenlegi ügyfélkezelési folyamatokról, interjúk az értékesítési csapattal, és felmérések a felhasználói igényekről.
- **Elemzés**: Az igények és követelmények azonosítása, mint például az ügyféladatok kezelése, az értékesítési folyamatok nyomon követése és a riportok generálása.
- **Érvényesítés**: Visszajelzések begyűjtése az értékesítési csapattól és a menedzsmenttől, majd a követelmények pontosítása.
- **Dokumentálás**: Az SRS dokumentum rögzíti a rendszer funkcionális követelményeit (pl. ügyféladatok szűrése) és a nem-funkcionális követelményeket (pl. adatbiztonság).

##### Következtetés

A követelménygyűjtés és dokumentáció kritikus fontosságú a szoftverfejlesztési folyamatban, mivel meghatározza a szoftverrel szemben támasztott elvárásokat és biztosítja, hogy a fejlesztők pontosan megértsék azokat. A jól végzett követelményanalízis minimalizálja a hibák és félreértések lehetőségét, és elősegíti a projektek sikeres megvalósítását. A követelmények pontos és részletes dokumentálása biztosítja, hogy a fejlesztés, tesztelés és karbantartás során minden érintett fél egyértelmű és egységes információkkal rendelkezzen a szoftver céljairól és funkcióiról.


### Use case-ek és user story-k

A követelményanalízis során az egyik legfontosabb lépés a felhasználói igények és üzleti célok pontos megfogalmazása. A use case-ek (használati esetek) és user story-k (felhasználói történetek) olyan eszközök, amelyek segítenek abban, hogy a fejlesztők és az érintett felek közös megértésre jussanak a szoftver elvárásaival kapcsolatban. Ezen eszközök alkalmazása lehetővé teszi, hogy a követelmények világosak, mérhetőek és nyomon követhetőek legyenek, valamint elősegítik a hatékony kommunikációt a projekt során.

#### Use Case-ek

A use case egy olyan eszköz, amely a rendszer és a felhasználók közötti interakciókat írja le. A használati esetek célja, hogy részletesen bemutassák, hogyan fogják a felhasználók használni a rendszert, és milyen célokat szeretnének elérni vele. A use case-ek általában diagramok és szöveges leírások formájában jelennek meg, és az alábbi elemeket tartalmazzák:

- **Szereplők (Actors)**: Azok a felhasználók vagy más rendszerek, amelyek interakcióba lépnek a rendszerrel. Például egy webáruház esetében az egyik szereplő lehet a vásárló, míg egy másik szereplő lehet a rendszergazda.
- **Forgatókönyv (Scenario)**: Az események sorozata, amely leírja, hogyan ér el egy szereplő egy adott célt a rendszer használatával.
- **Előfeltételek (Preconditions)**: Azok a feltételek, amelyeknek teljesülniük kell a használati eset megkezdése előtt.
- **Utófeltételek (Postconditions)**: Azok a feltételek, amelyeknek teljesülniük kell a használati eset sikeres befejezése után.
- **Fő folyamat (Main Flow)**: Az események alapvető sorrendje, amely leírja a használati eset normál menetét.
- **Alternatív folyamatok (Alternate Flows)**: Azok az eseménysorozatok, amelyek eltérnek a fő folyamattól, de még mindig elérik a célt.

**Példa**: Egy online könyváruház használati esete

- **Cél**: Könyv vásárlása
- **Szereplő**: Vásárló
- **Előfeltételek**: A vásárló regisztrált és bejelentkezett a rendszerbe.
- **Utófeltételek**: A vásárló sikeresen megvásárolta a könyvet, és visszaigazolást kapott.
- **Fő folyamat**:
    1. A vásárló keres egy könyvet a katalógusban.
    2. A vásárló kiválaszt egy könyvet és hozzáadja a kosarához.
    3. A vásárló megtekinti a kosarát és elindítja a fizetési folyamatot.
    4. A vásárló megadja a fizetési adatokat és megerősíti a vásárlást.
    5. A rendszer feldolgozza a fizetést és visszaigazolást küld a vásárlónak.
- **Alternatív folyamatok**:
    - Ha a keresés nem ad eredményt, a vásárló új keresést indít.
    - Ha a fizetési adatok hibásak, a rendszer hibaüzenetet küld és kéri a helyes adatokat.

#### User Story-k

A user story egy rövid, egyszerű leírás egy adott funkcióról a felhasználó szemszögéből. A user story-k az agilis fejlesztési módszertanokban népszerűek, és céljuk, hogy a fejlesztők könnyen megértsék, mit várnak el a felhasználók a rendszertől. A user story-k általában az alábbi sablont követik: "Mint [szereplő], szeretnék [cél], hogy [indoklás]". A user story-k tartalmazhatnak elfogadási kritériumokat is, amelyek meghatározzák, hogy a funkció mikor tekinthető késznek.

**Példa**: Egy online könyváruház user story-ja

- **User Story**: Mint vásárló, szeretnék könyvet keresni a katalógusban, hogy megtaláljam és megvásároljam a kívánt könyvet.
- **Elfogadási kritériumok**:
    - A keresési funkció lehetővé teszi a könyvek keresését cím, szerző és kategória alapján.
    - A keresési eredmények pontosak és relevánsak.
    - A vásárló képes a keresési eredményekből könyvet választani és hozzáadni a kosarához.

#### Use Case-ek és User Story-k összehasonlítása

Bár a use case-ek és a user story-k hasonló célt szolgálnak, vannak fontos különbségek közöttük:

- **Részletesség**: A use case-ek részletesebben írják le a rendszer és a felhasználók közötti interakciókat, beleértve az előfeltételeket, utófeltételeket és alternatív folyamatokat. A user story-k rövidebbek és egyszerűbbek, gyakran egyetlen mondatban foglalják össze a követelményt.
- **Formátum**: A use case-ek gyakran diagramokkal és szöveges leírásokkal jelennek meg, míg a user story-k általában rövid szöveges leírások, amelyeket elfogadási kritériumok egészítenek ki.
- **Alkalmazás**: A use case-ek gyakrabban használatosak a hagyományos fejlesztési módszertanokban, míg a user story-k az agilis fejlesztésben népszerűek.

##### Következtetés

A use case-ek és user story-k alapvető eszközök a követelményanalízis során, amelyek segítenek a fejlesztőknek és az érintett feleknek pontosan meghatározni és megérteni a szoftverrel szemben támasztott elvárásokat. A használati esetek részletes leírásai és forgatókönyvei átfogó képet adnak a rendszer működéséről, míg a user story-k egyszerűbb és gyorsabb megközelítést kínálnak a felhasználói igények megfogalmazására. Mindkét módszer elősegíti a hatékony kommunikációt és együttműködést a projekt során, biztosítva, hogy a szoftver a felhasználók valós igényeit és üzleti céljait tükrözze.