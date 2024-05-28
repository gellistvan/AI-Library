\newpage

## 5. **Nem funkcionális követelmények (NFR)**

A nem funkcionális követelmények (NFR-ek) alapvető szerepet játszanak a szoftverarchitektúra tervezésében és megvalósításában, mivel meghatározzák a rendszer minőségi attribútumait és viselkedését. Ezek a követelmények, mint például a teljesítmény, skálázhatóság, biztonság és rendelkezésre állás, közvetlenül befolyásolják a szoftver működését és felhasználói élményét. Emellett a megbízhatóság, karbantarthatóság és használhatóság biztosítják, hogy a rendszer hosszú távon is fenntartható és könnyen kezelhető legyen. Ebben a fejezetben részletesen tárgyaljuk a nem funkcionális követelmények különböző aspektusait, bemutatva, hogyan lehet ezeket az elveket és gyakorlatokat hatékonyan alkalmazni a szoftverrendszerek tervezése során.

### Teljesítmény, skálázhatóság, biztonság, rendelkezésre állás

A nem funkcionális követelmények (NFR-ek) közül a teljesítmény, skálázhatóság, biztonság és rendelkezésre állás kulcsfontosságú tényezők a szoftverrendszerek tervezése és megvalósítása során. Ezek a követelmények nem közvetlenül a rendszer funkcionalitását határozzák meg, hanem annak minőségi attribútumait, amelyek jelentősen befolyásolják a rendszer használhatóságát, megbízhatóságát és felhasználói elégedettségét. Az alábbiakban részletesen bemutatjuk ezeknek a követelményeknek a jelentőségét, gyakorlati alkalmazásukat és példákat adunk arra, hogyan lehet ezeket az elveket integrálni a szoftverarchitektúrába.

#### Teljesítmény

A teljesítmény azt méri, hogy a rendszer milyen gyorsan és hatékonyan képes végrehajtani a feladatait. A jó teljesítmény elengedhetetlen a felhasználói élmény szempontjából, különösen olyan alkalmazások esetében, amelyek valós idejű válaszidőt igényelnek, mint például a webes alkalmazások, játékok és pénzügyi rendszerek. A teljesítmény optimalizálása érdekében figyelembe kell venni a rendszer válaszidejét, áteresztőképességét és erőforrás-felhasználását.

**Optimalizálási technikák:**
- **Caching (gyorsítótárazás)**: Az adatokat gyakran tárolják gyorsítótárakban, hogy csökkentsék az adatbázis-lekérdezések számát és javítsák a válaszidőt. Például egy webalkalmazásban a felhasználói profilokat gyorsítótárban tárolhatják, hogy gyorsabb hozzáférést biztosítsanak.
- **Load Balancing (terheléselosztás)**: A terheléselosztók egyenletesen osztják el a beérkező kéréseket több szerver között, javítva ezzel a rendszer teljesítményét és megbízhatóságát. Például egy e-kereskedelmi platformon a terheléselosztók segítenek a megnövekedett forgalom kezelésében a nagy leárazások idején.
- **Profiling és monitoring**: A rendszer teljesítményének folyamatos figyelése és elemzése segít azonosítani a szűk keresztmetszeteket és optimalizálni a kódot. Például egy pénzügyi alkalmazás esetében a tranzakciók feldolgozási idejét folyamatosan monitorozzák, és optimalizálják a kritikus útvonalakat.

#### Skálázhatóság

A skálázhatóság azt jelenti, hogy a rendszer képes kezelni a növekvő terhelést anélkül, hogy teljesítménybeli problémák lépnének fel. A skálázhatóság lehet vertikális (az erőforrások növelése egyetlen szerveren) vagy horizontális (több szerver hozzáadása a rendszerhez). A jól skálázható rendszerek képesek alkalmazkodni a változó terhelési körülményekhez és biztosítják a folyamatos működést.

**Skálázhatósági megoldások:**
- **Microservices (mikroszolgáltatások)**: A rendszer funkcióit kisebb, független szolgáltatásokra bontják, amelyek külön-külön skálázhatók. Például egy streaming szolgáltatás esetében a videólejátszást, felhasználói profilkezelést és hirdetési rendszert külön mikroszolgáltatásokként valósítják meg.
- **Containerization (konténerizáció)**: A konténerizáció lehetővé teszi az alkalmazások könnyű telepítését és skálázását különböző környezetekben. Docker és Kubernetes példák olyan technológiákra, amelyek segítik a konténerizált alkalmazások kezelését és skálázását.
- **Serverless Architecture (szerver nélküli architektúra)**: A szerver nélküli architektúrákban az alkalmazások futtatása és skálázása a felhőszolgáltatók feladata, így a fejlesztőknek nem kell aggódniuk az infrastruktúra kezelése miatt. Például egy chat alkalmazás üzenetküldő funkcióit szerver nélküli szolgáltatásokkal valósítják meg, amelyek automatikusan skálázódnak a terhelés függvényében.

#### Biztonság

A biztonság a szoftverrendszer azon képessége, hogy védelmet nyújtson a különböző fenyegetésekkel, mint például a jogosulatlan hozzáférés, adatszivárgás és támadások ellen. A biztonságos rendszer biztosítja az adatok integritását, bizalmasságát és rendelkezésre állását. A biztonsági intézkedések tervezése és megvalósítása kritikus fontosságú minden szoftverrendszer esetében.

**Biztonsági intézkedések:**
- **Hitelesítés és autorizáció**: Azonosítani kell a felhasználókat és biztosítani, hogy csak a megfelelő jogosultságokkal rendelkezők férhessenek hozzá a rendszer erőforrásaihoz. Például egy vállalati alkalmazás kétfaktoros hitelesítést alkalmaz a belépéskor, és a felhasználói szerepkörök alapján szabályozza az erőforrásokhoz való hozzáférést.
- **Adatvédelem**: Az érzékeny adatok titkosítása és védelme kritikus fontosságú a biztonság szempontjából. Például egy pénzügyi alkalmazásban az összes tranzakciós adatot titkosítva tárolják és továbbítják.
- **Támadásmegelőzés**: Az alkalmazásokat védeni kell a különböző támadásokkal szemben, mint például a SQL injection, XSS (Cross-Site Scripting) és DDoS (Distributed Denial of Service) támadások. Például egy webalkalmazás tűzfal (WAF) használata segít megelőzni a webes támadásokat.

#### Rendelkezésre állás

A rendelkezésre állás azt méri, hogy a rendszer milyen mértékben képes folyamatosan működni és elérhető maradni a felhasználók számára. A magas rendelkezésre állású rendszerek minimalizálják a leállási időt és biztosítják a folyamatos szolgáltatást, még hibák vagy karbantartási munkálatok esetén is.

**Rendelkezésre állást növelő technikák:**
- **Redundancia és replikáció**: A rendszer kritikus komponenseinek többszörözése és az adatok replikációja biztosítja, hogy egy komponens meghibásodása esetén is rendelkezésre álljanak alternatív megoldások. Például egy adatbázis replikációs technikával biztosítják, hogy az adatok több szerveren is elérhetők legyenek.
- **Failover Mechanizmusok**: A failover mechanizmusok automatikusan átkapcsolják a szolgáltatást egy tartalék rendszerre vagy komponensre meghibásodás esetén. Például egy webalkalmazás esetében, ha az elsődleges szerver meghibásodik, a forgalmat egy tartalék szerver veszi át.
- **Georedundancy (georedundancia)**: Az adatok és szolgáltatások több földrajzi helyen történő elhelyezése biztosítja, hogy egy helyi meghibásodás vagy katasztrófa esetén is elérhetők maradjanak. Például egy globális felhőszolgáltatás több adatközpontban tárolja az adatokat, biztosítva ezzel a folyamatos rendelkezésre állást.

#### Példák a gyakorlatban

1. **Teljesítmény és skálázhatóság egy e-kereskedelmi platformon**: Egy nagy e-kereskedelmi platform, mint például az Amazon, folyamatosan optimalizálja a teljesítményt és skálázhatóságot. A rendszer különböző komponensei, mint például a keresőmotor, a termékajánló rendszer és a fizetési rendszer, mind mikroszolgáltatásokként vannak megvalósítva, amelyek külön-külön sk

álázhatók és optimalizálhatók. A gyorsítótárazás, terheléselosztás és folyamatos monitorozás mind hozzájárulnak a rendszer kiváló teljesítményéhez és megbízhatóságához.

2. **Biztonság és rendelkezésre állás egy banki rendszerben**: Egy banki rendszer esetében a biztonság és rendelkezésre állás kiemelt fontosságú. A rendszer kétfaktoros hitelesítést alkalmaz a felhasználók azonosítására, és szigorú hozzáférés-szabályozást valósít meg az érzékeny adatok védelme érdekében. Az adatokat titkosítva tárolják és továbbítják, és folyamatosan monitorozzák a rendszer biztonságát, hogy megelőzzék a támadásokat. A rendszer redundanciát és failover mechanizmusokat alkalmaz a magas rendelkezésre állás biztosítása érdekében.

3. **Teljesítmény, skálázhatóság és biztonság egy IoT platformon**: Egy IoT platform, amely milliónyi eszközt kezel, magas teljesítményt és skálázhatóságot igényel. A rendszer mikroszolgáltatás-alapú architektúrát használ, ahol minden szolgáltatás külön skálázható. Az adatokat gyorsítótárazás és terheléselosztás segítségével kezelik, biztosítva a gyors válaszidőt és a megbízható működést. Az eszközök közötti kommunikáció titkosított csatornákon keresztül történik, és folyamatosan figyelik a biztonsági fenyegetéseket, hogy megvédjék az érzékeny adatokat.

### Megbízhatóság, karbantarthatóság, használhatóság

A nem funkcionális követelmények (NFR-ek) között a megbízhatóság, karbantarthatóság és használhatóság különös figyelmet érdemelnek, mivel ezek a tényezők közvetlenül befolyásolják a szoftverrendszerek hosszú távú működését és felhasználói elégedettségét. Az alábbiakban részletesen bemutatjuk e követelmények jelentőségét, gyakorlati alkalmazásukat és konkrét példákat adunk arra, hogyan lehet ezeket az elveket hatékonyan integrálni a szoftverarchitektúrába.

#### Megbízhatóság

A megbízhatóság a szoftverrendszer azon képessége, hogy előre meghatározott időn belül hibamentesen és következetesen működjön. A megbízhatóság fontos tényező a felhasználói bizalom szempontjából, különösen olyan rendszerek esetében, amelyek kritikus feladatokat látnak el, mint például a banki alkalmazások, egészségügyi rendszerek és ipari vezérlőrendszerek.

**Megbízhatósági technikák:**
- **Hibatűrés**: A hibatűrő rendszerek képesek folytatni működésüket még meghibásodás esetén is. Ez redundancia és tartalék komponensek alkalmazásával érhető el. Például egy repülésirányító rendszer redundáns adatkapcsolatokat és tartalék szervereket használ, hogy biztosítsa a folyamatos működést még hardverhibák esetén is.
- **Hibakezelés és -megelőzés**: A rendszer megfelelő hibakezelési mechanizmusokkal rendelkezik, amelyek gyorsan és hatékonyan képesek kezelni a hibákat és megakadályozni azok továbbterjedését. Például egy e-kereskedelmi platformnál a tranzakciók kezelése során alkalmazott tranzakciós naplózás és rollback mechanizmusok biztosítják, hogy az adatok konzisztens állapotban maradjanak hibák esetén is.
- **Tesztelés és verifikáció**: A megbízhatóság növelése érdekében a szoftverrendszereket alaposan tesztelik különböző körülmények között. Automatikus tesztelési keretrendszerek és folyamatos integrációs eszközök segítik a rendszer hibamentes működésének biztosítását. Például egy pénzügyi alkalmazásban a folyamatos integrációs folyamatok minden új kódváltoztatás után lefuttatják a teljes tesztkészletet, hogy biztosítsák a rendszer stabilitását.

#### Karbantarthatóság

A karbantarthatóság azt méri, hogy a szoftverrendszer milyen könnyen módosítható, javítható és bővíthető. A jól karbantartható rendszerek lehetővé teszik a gyors hibajavítást, az új funkciók könnyű hozzáadását és a meglévő funkciók egyszerű módosítását, ezáltal csökkentve a fejlesztési költségeket és növelve a rendszer élettartamát.

**Karbantarthatósági megoldások:**
- **Modularitás**: A rendszer modulokra bontása elősegíti a könnyű karbantarthatóságot, mivel a változtatások és javítások izoláltan, a teljes rendszer befolyásolása nélkül végezhetők el. Például egy vállalati alkalmazásban külön modulok kezelik az ügyféladatokat, a terméknyilvántartást és a rendeléseket, így egy-egy modul módosítása nem érinti a többi modult.
- **Dokumentáció**: A részletes és naprakész dokumentáció megkönnyíti a fejlesztők számára a rendszer megértését és karbantartását. A dokumentáció tartalmazza az architektúra terveit, a komponensek leírását, az interfészek specifikációit és a nem-funkcionális követelményeket. Például egy API dokumentáció tartalmazza az összes elérhető végpontot, a kérések és válaszok formátumát, valamint az engedélyezési követelményeket.
- **Kódminőség és szabványok**: A kódminőség fenntartása érdekében a fejlesztők követik a bevált gyakorlatokat és kódolási szabványokat. Ez magában foglalja a kód áttekintéseket, a statikus kódelemzést és a folyamatos refaktorálást. Például egy agilis fejlesztési környezetben a kódáttekintések rendszeresek, és a csapatok a kódolási szabványokat és best practice-eket követik a minőség fenntartása érdekében.

#### Használhatóság

A használhatóság azt méri, hogy a rendszer milyen mértékben könnyen és hatékonyan használható a felhasználók számára. A jó használhatóság biztosítja, hogy a felhasználók gyorsan és egyszerűen végezhessék el a szükséges műveleteket, csökkenti a hibák előfordulásának valószínűségét és növeli a felhasználói elégedettséget.

**Használhatósági megoldások:**
- **Felhasználói élmény (UX) tervezés**: A felhasználói élmény tervezése során a fejlesztők figyelembe veszik a felhasználók igényeit, viselkedését és céljait, hogy egy intuitív és hatékony felhasználói felületet hozzanak létre. Például egy mobil alkalmazás tervezése során a felhasználói élményre összpontosítva egyszerű és intuitív navigációs rendszert alakítanak ki.
- **Hozzáférhetőség**: A hozzáférhetőség biztosítása érdekében a rendszernek figyelembe kell vennie a különböző felhasználói csoportok, beleértve a fogyatékkal élő személyek igényeit is. Ez magában foglalja a képernyőolvasók támogatását, a nagy kontrasztú színeket és a billentyűzet-navigáció lehetőségét. Például egy webalkalmazás fejlesztése során biztosítják, hogy az oldal minden eleme elérhető és navigálható legyen billentyűzettel, valamint támogassa a képernyőolvasókat.
- **Felhasználói visszajelzések integrálása**: A felhasználói visszajelzések rendszeres gyűjtése és integrálása segít a rendszer használhatóságának folyamatos javításában. Ez magában foglalja az A/B tesztelést, a felhasználói tesztelést és az analitikai eszközök használatát a felhasználói viselkedés megértéséhez. Például egy e-kereskedelmi platform rendszeresen gyűjt visszajelzéseket a felhasználóktól, és ezek alapján finomítja a vásárlási folyamatot.

#### Példák a gyakorlatban

1. **Megbízhatóság egy egészségügyi rendszerben**: Egy egészségügyi rendszer esetében a megbízhatóság kiemelten fontos, mivel az adatok pontossága és elérhetősége közvetlen hatással van a betegek ellátására. A rendszer redundáns adatbázisokkal, automatikus biztonsági mentésekkel és hibatűrő architektúrával biztosítja, hogy az adatok mindig elérhetők legyenek, és a szolgáltatás folyamatosan működjön.

2. **Karbantarthatóság egy pénzügyi alkalmazásban**: Egy pénzügyi alkalmazás esetében a modularitás és a részletes dokumentáció biztosítja a könnyű karbantarthatóságot. A rendszer külön modulokban valósítja meg az ügyfélszolgálati funkciókat, a tranzakciókezelést és a jelentéskészítést. Minden modul részletes dokumentációval rendelkezik, amely megkönnyíti a fejlesztők számára a módosításokat és bővítéseket.

3. **Használhatóság egy e-kereskedelmi platformon**: Egy e-kereskedelmi platform esetében a felhasználói élmény tervezése és a hozzáférhetőség biztosítása kulcsfontosságú. Az egyszer

ű navigációs rendszer, a gyors keresési funkciók és a reszponzív dizájn biztosítják, hogy a felhasználók könnyen megtalálják és megvásárolják a termékeket. A felhasználói visszajelzések rendszeres gyűjtése és elemzése segít a folyamatos fejlesztésekben és a felhasználói élmény javításában.

#### Konklúzió

A teljesítmény, skálázhatóság, biztonság és rendelkezésre állás a szoftverrendszerek alapvető nem funkcionális követelményei, amelyek meghatározzák a rendszer minőségét és felhasználói élményét. Ezeknek a követelményeknek a figyelembe vétele és integrálása a szoftverarchitektúrába biztosítja, hogy a rendszer képes legyen kezelni a növekvő terhelést, megvédje az érzékeny adatokat és folyamatosan elérhető maradjon a felhasználók számára. Az optimalizálási technikák és biztonsági intézkedések gyakorlati alkalmazása különböző szoftverrendszerekben bemutatja ezen elvek univerzális jelentőségét és hasznosságát a szoftverfejlesztésben.

A megbízhatóság, karbantarthatóság és használhatóság alapvető nem funkcionális követelmények, amelyek meghatározzák a szoftverrendszerek hosszú távú sikerét és felhasználói elégedettségét. A megbízható rendszerek biztosítják a hibamentes működést, a jól karbantartható rendszerek könnyen módosíthatók és bővíthetők, míg a használható rendszerek egyszerű és hatékony felhasználói élményt nyújtanak. E követelmények integrálása a szoftverarchitektúrába és a fejlesztési folyamatba elengedhetetlen a magas minőségű, fenntartható és sikeres szoftverrendszerek létrehozásához.
