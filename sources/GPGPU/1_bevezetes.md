\newpage

## 1. Bevezetés

Az általános célú számítások a GPU-kon (GPGPU) egy viszonylag új és gyorsan fejlődő terület, amely az elmúlt évtizedben forradalmasította a nagy teljesítményű számítástechnikai feladatok megoldását. Ebben a könyvben bemutatjuk a GPGPU alapjait, a GPU programozás koncepcióit, és megvizsgáljuk, hogyan lehet hatékonyan alkalmazni a GPU-kat különböző problémák megoldására. Az első fejezet célja, hogy bevezesse az olvasót a GPU-k történetébe, evolúciójába, valamint megismertesse a GPU programozás alapfogalmait és felhasználási területeit.

### 1.1. A GPU története és evolúciója

A grafikus feldolgozó egységek (GPU-k) története és evolúciója a számítógépes grafika fejlődéséhez köthető. A GPU-k fejlődése több évtizedes múltra tekint vissza, amely során a specializált grafikai feladatok végrehajtására tervezett eszközökből általános célú számítási egységekké váltak.

#### A kezdetek

Az első számítógépes grafikus kártyák az 1980-as években jelentek meg, és kezdetben kizárólag 2D grafikai feladatok végrehajtására voltak alkalmasak. Ezek a korai GPU-k, mint például az IBM 8514/A és a VGA (Video Graphics Array) kártyák, alapvető grafikai műveleteket támogattak, mint például a képpontok megjelenítése és a vonalak rajzolása.

#### A 3D grafika megjelenése

Az 1990-es években a számítógépes játékok és a multimédiás alkalmazások iránti igény növekedésével megjelentek az első 3D-s grafikai kártyák. Az olyan vállalatok, mint az NVIDIA és az ATI (ma AMD), elkezdték fejleszteni azokat a hardveres megoldásokat, amelyek képesek voltak komplex 3D grafikai műveletek végrehajtására. Az NVIDIA 1999-ben bemutatta a GeForce 256-ot, amelyet az első GPU-ként reklámoztak, mivel tartalmazta a transform and lighting (T&L) motorokat, amelyek lehetővé tették a 3D-s objektumok valós idejű transzformációját és megvilágítását.

#### Az általános célú számítások felé

A 2000-es évek elején a GPU-k teljesítménye jelentősen megnőtt, és a programozók felismertek, hogy a GPU-k párhuzamos feldolgozási képességei alkalmasak lehetnek nem csak grafikai, hanem egyéb számítási feladatok elvégzésére is. Az NVIDIA 2006-ban bevezette a CUDA (Compute Unified Device Architecture) platformot, amely lehetővé tette a programozók számára, hogy C nyelven írjanak programokat a GPU-k számára. Ez a lépés forradalmasította a GPU-k alkalmazását, mivel lehetővé tette, hogy a tudományos számítások, gépi tanulás, adatfeldolgozás és más területek is kihasználhassák a GPU-k párhuzamos feldolgozási képességeit.

#### Az evolúció folytatása

Azóta a GPU-k folyamatosan fejlődtek, mind teljesítményük, mind programozhatóságuk tekintetében. Az NVIDIA, az AMD és más gyártók folyamatosan újabb és újabb generációkat hoztak létre, amelyek egyre nagyobb teljesítményt és hatékonyságot kínáltak. A GPU-k ma már kulcsszerepet játszanak a mesterséges intelligencia, a mélytanulás, a tudományos kutatás és a szimuláció területén is.

### 1.2. Mi az a GPU programozás?

A GPU programozás olyan technika, amely lehetővé teszi a fejlesztők számára, hogy kihasználják a GPU-k hatalmas párhuzamos feldolgozási képességeit az általános célú számítások végrehajtására. Míg a CPU-k (Central Processing Units) egy vagy néhány erős maggal rendelkeznek, amelyek sorosan hajtják végre a műveleteket, addig a GPU-k sok ezer kisebb maggal rendelkeznek, amelyek párhuzamosan képesek futtatni a számításokat.

#### A GPU architektúra

A GPU-k architektúrája alapvetően különbözik a CPU-kétól. Míg a CPU-k általában néhány magot tartalmaznak, amelyek bonyolult utasításokat képesek végrehajtani, a GPU-k sok ezer egyszerűbb magot tartalmaznak, amelyek egyszerű műveleteket hajtanak végre, de párhuzamosan. Ez az architektúra lehetővé teszi a GPU-k számára, hogy rendkívül nagy számítási teljesítményt nyújtsanak, különösen akkor, ha a feladat párhuzamosítható.

#### A GPU programozási modellek

A GPU programozás során a legelterjedtebb programozási modellek a CUDA (Compute Unified Device Architecture) és az OpenCL (Open Computing Language).

##### CUDA

A CUDA az NVIDIA által kifejlesztett platform, amely lehetővé teszi a fejlesztők számára, hogy C, C++ és Fortran nyelveken írjanak programokat, amelyek közvetlenül futtathatók az NVIDIA GPU-kon. A CUDA használata során a programozók meghatározzák a kernel nevű függvényeket, amelyek a GPU-n futnak, és ezek a kernelek párhuzamosan végzik el a számításokat.

##### OpenCL

Az OpenCL egy nyílt szabvány, amelyet az Apple fejlesztett ki, és amelyet az Khronos Group karbantart. Az OpenCL támogatja a különböző gyártók GPU-it, valamint egyéb párhuzamos feldolgozó egységeket, mint például a CPU-kat és a DSP-ket. Az OpenCL programozási modell hasonló a CUDA-hoz, de szélesebb körű hardver támogatást kínál.

#### A GPU programozás kihívásai

A GPU programozás számos kihívással jár. Az egyik legnagyobb kihívás a párhuzamos programozás komplexitása, mivel a fejlesztőknek biztosítaniuk kell, hogy a párhuzamosan futó szálak megfelelően szinkronizálva legyenek, és elkerüljék a versenyhelyzeteket. Ezenkívül a GPU-k memóriakezelése is különbözik a CPU-kétól, és a hatékony memóriahasználat kulcsfontosságú a jó teljesítmény eléréséhez.

### 1.3. GPU-k felhasználási területei

A GPU-kat ma már számos területen alkalmazzák az általános célú számítások végrehajtására. Az alábbiakban bemutatjuk a legfontosabb felhasználási területeket.

#### Tudományos számítások

A tudományos kutatásokban a GPU-kat gyakran használják nagyméretű adatfeldolgozási feladatok végrehajtására, például molekuláris dinamikai szimulációk, asztrofizikai szimulációk és időjárási modellezés terén. A GPU-k párhuzamos feldolgozási képességei lehetővé teszik, hogy ezek a szimulációk sokkal gyorsabban futtathatók legyenek, mint hagyományos CPU-kkal.

#### Mesterséges intelligencia és gépi tanulás

A mesterséges intelligencia és a gépi tanulás területén a GPU-k kulcsszerepet játszanak a nagy teljesítményű neurális hálózatok tanításában. A mélytanulási algoritmusok, mint például a konvolúciós neurális hálózatok (CNN-ek) és a generatív ellenséges hálózatok (GAN-ek), rendkívül nagy számítási igénnyel rendelkeznek, amelyeket a GPU-k hatékonyan képesek kielégíteni.

#### Képfeldolgozás és számítógépes látás

A képfeldolgozási és számítógépes látási feladatok során a GPU-k gyors és párhuzamos feldolgozási képességei lehetővé teszik a valós idejű képan