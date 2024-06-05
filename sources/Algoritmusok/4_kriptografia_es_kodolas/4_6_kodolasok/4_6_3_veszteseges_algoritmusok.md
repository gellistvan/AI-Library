\newpage

## 6.3. Veszteséges adatkompressziós algoritmusok

Az információs társadalomban a digitális adatok hatékony tárolása és továbbítása kulcsfontosságú szerepet játszik. Az adatkompresszió célja az adatok méretének csökkentése, ami gyorsabb adatátvitelt és kisebb tárolási igényt eredményez. Két fő típusa létezik: veszteségmentes és veszteséges adatkompresszió. Míg a veszteségmentes kompresszió során az eredeti adat teljes mértékben visszaállítható, addig a veszteséges kompresszió esetében az adatok bizonyos része elveszik, cserébe jelentősen kisebb fájlméret érhető el. A veszteséges adatkompressziós algoritmusok különösen hasznosak olyan alkalmazásokban, ahol az emberi érzékelés tolerálja az apróbb adatveszteségeket, például a képek és hangok esetében. Ebben a fejezetben két elterjedt veszteséges adatkompressziós algoritmust, a JPEG-et és az MP3-at mutatjuk be, amelyek napjainkban széles körben használatosak a digitális képek és hanganyagok tömörítésére.

### 6.3.1. JPEG

A JPEG (Joint Photographic Experts Group) egy veszteséges képkompressziós szabvány, amely széles körben használt a digitális képek tömörítésére. Az 1992-ben megjelent szabvány azóta is az egyik legelterjedtebb formátum a digitális képfeldolgozás területén. A JPEG kompresszió alapelve az, hogy az emberi szem érzékelési sajátosságait kihasználva csökkenti a kép méretét úgy, hogy az észlelhető minőségromlás minimális maradjon.

#### A JPEG Kompresszió Folyamata

A JPEG kompresszió több lépésből áll, amelyek közül a legfontosabbak a következők:
1. **Színterek átalakítása**
2. **Blokkokra bontás**
3. **DCT (Diszkrét Koszinusz Transzformáció) alkalmazása**
4. **Kvantalizáció**
5. **Huffman kódolás**

##### Színterek átalakítása

A JPEG algoritmus első lépése a színterek átalakítása. A legtöbb digitális kép RGB (Red, Green, Blue) színtérben van tárolva, de az emberi szem érzékenyebb a fényerősség (luminancia) változásokra, mint a színárnyalat (krominancia) változásokra. Ennek kihasználása érdekében a képeket YCbCr színtérbe alakítják át, ahol az Y komponens a luminancia információt, míg a Cb és Cr komponensek a krominancia információkat tartalmazzák.

##### Blokkokra bontás

Az átalakított képet 8x8 pixeles blokkokra bontják. Ez a blokkokra bontás lehetővé teszi a képfeldolgozás egyszerűbb és hatékonyabb végrehajtását, valamint a lokális frekvencia-információk jobb kihasználását.

##### DCT (Diszkrét Koszinusz Transzformáció) alkalmazása

A következő lépésben minden egyes 8x8-as blokkra Diszkrét Koszinusz Transzformációt (DCT) alkalmaznak. A DCT a blokkot a frekvencia tartományába transzformálja, azaz a blokk pixeleit olyan frekvencia komponensekké alakítja, amelyek megadják a blokk mintázatát. A DCT eredményeként egy 8x8-as mátrixot kapunk, ahol az egyes elemek a különböző frekvenciák amplitúdóit reprezentálják.

Matematikailag a DCT így néz ki:

\[
C(u,v) = \frac{1}{4} \alpha(u) \alpha(v) \sum_{x=0}^{7} \sum_{y=0}^{7} P(x,y) \cos\left(\frac{(2x+1)u\pi}{16}\right) \cos\left(\frac{(2y+1)v\pi}{16}\right)
\]

ahol \(\alpha(u)\) és \(\alpha(v)\) az alábbiak szerint definiáltak:

\[
\alpha(u) = \begin{cases}
\frac{1}{\sqrt{2}} & \text{ha } u = 0 \\
1 & \text{ha } u \neq 0
\end{cases}
\]

##### Kvantalizáció

A DCT mátrix elemeit ezután kvantalizálják, ami a veszteséges kompresszió legfontosabb lépése. A kvantalizáció során a mátrix elemeit egy előre meghatározott kvantalizációs mátrix elemeivel osztják el, majd a kapott eredményeket lekerekítik. Ez a lépés a magasabb frekvenciájú komponensek elhanyagolásához vezet, mivel ezek kevésbé érzékelhetők az emberi szem számára.

Példa kvantalizációs mátrixra:

\[
\begin{bmatrix}
16 & 11 & 10 & 16 & 24 & 40 & 51 & 61 \\
12 & 12 & 14 & 19 & 26 & 58 & 60 & 55 \\
14 & 13 & 16 & 24 & 40 & 57 & 69 & 56 \\
14 & 17 & 22 & 29 & 51 & 87 & 80 & 62 \\
18 & 22 & 37 & 56 & 68 & 109 & 103 & 77 \\
24 & 35 & 55 & 64 & 81 & 104 & 113 & 92 \\
49 & 64 & 78 & 87 & 103 & 121 & 120 & 101 \\
72 & 92 & 95 & 98 & 112 & 100 & 103 & 99
\end{bmatrix}
\]

##### Huffman kódolás

A kvantalizált DCT mátrixot zigzag sorrendben bejárva egy egy dimenziós sorozatot kapunk, amelyet Huffman kódolással tovább tömörítenek. A Huffman kódolás egy veszteségmentes tömörítési technika, amely a gyakrabban előforduló elemekhez rövidebb kódokat, míg a ritkábban előforduló elemekhez hosszabb kódokat rendel. Ezáltal tovább csökkenthető a kép mérete anélkül, hogy további adatvesztés következne be.

#### Dekódolás

A JPEG dekódolási folyamata a fenti lépések megfordítása, azaz először a Huffman dekódolást alkalmazzák, majd a kvantalizált mátrixot visszaalakítják, és végül az inverz DCT-t alkalmazva visszanyerik az eredeti 8x8-as blokkokat. Az így kapott blokkokból pedig összeállítják az eredeti képet.

#### Példakód C++ Nyelven

Az alábbiakban bemutatok egy egyszerű példakódot C++ nyelven, amely egy 8x8-as blokkra alkalmazza a DCT-t és a kvantalizációt:

```cpp
#include <iostream>
#include <cmath>
#include <vector>

const int N = 8;
const double PI = 3.14159265358979323846;

// DCT
void DCT(const std::vector<std::vector<double>>& input, std::vector<std::vector<double>>& output) {
    for (int u = 0; u < N; ++u) {
        for (int v = 0; v < N; ++v) {
            double sum = 0.0;
            for (int x = 0; x < N; ++x) {
                for (int y = 0; y < N; ++y) {
                    sum += input[x][y] * cos((2 * x + 1) * u * PI / (2 * N)) * cos((2 * y + 1) * v * PI / (2 * N));
                }
            }
            double alpha_u = (u == 0) ? sqrt(1.0 / N) : sqrt(2.0 / N);
            double alpha_v = (v == 0) ? sqrt(1.0 / N) : sqrt(2.0 / N);
            output[u][v] = alpha_u * alpha_v * sum;
        }
    }
}

// Kvantalizáció
void Quantize(std::vector<std::vector<double>>& dct, const std::vector<std::vector<int>>& quant_matrix) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            dct[i][j] = round(dct[i][j] / quant_matrix[i][j]);
        }
    }
}

int main() {
    std::vector<std::vector<double>> input = {
        {52, 55, 61, 66, 70, 61, 64, 73},
        {63, 59, 66, 90, 109, 85, 69, 72},
        {62, 59, 68, 113, 144, 104, 66, 73},
        {63, 58, 71, 122, 154, 106, 70, 69},
        {67, 61, 68, 104, 126, 88, 68, 70},
        {79, 65, 60, 70, 77, 68, 58, 75},
        {85, 71, 64, 59, 55, 61, 65, 83},
        {87, 79, 69, 68, 65, 76, 78, 94}
    };

    std::vector<std::vector<double>> dct(N, std::vector<double>(N, 0));
    std::vector<std::vector<int>> quant_matrix = {
        {16, 11, 10, 16, 24, 40, 51, 61},
        {12, 12, 14, 19, 26, 58, 60, 55},
        {14, 13, 16, 24, 40, 57, 69, 56},
        {14, 17, 22, 29, 51, 87, 80, 62},
        {18, 22, 37, 56, 68, 109, 103, 77},
        {24, 35, 55, 64, 81, 104, 113, 92},
        {49, 64, 78, 87, 103, 121, 120, 101},
        {72, 92, 95, 98, 112, 100, 103, 99}
    };

    DCT(input, dct);
    Quantize(dct, quant_matrix);

    // Output the DCT coefficients after quantization
    std::cout << "DCT coefficients after quantization:\n";
    for (const auto& row : dct) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
```

#### Minőség és Fájlméret Trade-off

A JPEG kompresszióban a kvantalizáció mértéke döntően befolyásolja a tömörített kép minőségét és méretét. Minél durvább a kvantalizáció (azaz minél nagyobbak a kvantalizációs mátrix elemei), annál kisebb lesz a fájlméret, de annál nagyobb lesz a veszteség is a képminőségben. Az optimális beállítás megtalálása függ a konkrét alkalmazástól és a felhasználói igényektől.

#### Alkalmazási Területek

A JPEG formátumot számos területen használják, többek között digitális fényképezőgépekben, weboldalakon, és más multimédiás alkalmazásokban. Előnye a kiváló tömörítési arány és a széleskörű támogatottság, hátránya viszont, hogy nem alkalmas a precíziós követelményeket igénylő képfeldolgozási feladatokhoz, mint például az orvosi képek vagy az űrfelvételek.

### 6.3.2. MP3

Az MP3 (MPEG-1 Audio Layer III) egy veszteséges hangkompressziós szabvány, amely a digitális hangfájlok tömörítésére szolgál. Az MP3-at az 1990-es évek elején fejlesztették ki a Moving Picture Experts Group (MPEG) által, és azóta a digitális hangfájlok egyik legelterjedtebb formátumává vált. Az MP3 kompresszió célja az, hogy a hangfájlok méretét jelentősen csökkentse, miközben a hallható minőség romlása minimális marad.

#### Az MP3 Kompresszió Folyamata

Az MP3 kompresszió több lépésből áll, amelyek a következők:
1. **PCM (Pulse Code Modulation) kódolt hangadatok előfeldolgozása**
2. **Pszichoakusztikai modell alkalmazása**
3. **MDCT (Modifikált Diszkrét Koszinusz Transzformáció) alkalmazása**
4. **Kvantalizáció és szűrés**
5. **Huffman kódolás**

##### PCM (Pulse Code Modulation) Kódolt Hangadatok Előfeldolgozása

Az első lépésben az eredeti PCM kódolt hangadatokat elemeire bontják. A PCM egy lineáris kvantalizációs eljárás, amely a hangjelet digitális jellé alakítja. Az MP3 kódolás során az eredeti PCM adatokat rövid, általában 1152 minta hosszúságú szegmensekre bontják, és ezekkel a szegmensekkel külön-külön dolgoznak tovább.

##### Pszichoakusztikai Modell Alkalmazása

Az emberi hallás sajátosságainak kihasználása érdekében az MP3 kompresszió egy pszichoakusztikai modellt alkalmaz. Az emberi fül nem egyenletesen érzékeli a különböző frekvenciákat és hangerősségeket, és bizonyos hangokat elfednek mások (maszkolás). Az MP3 algoritmus ezt a jelenséget kihasználva csökkenti vagy eltávolítja azokat a hanginformációkat, amelyeket az emberi fül valószínűleg nem észlelne.

##### MDCT (Modifikált Diszkrét Koszinusz Transzformáció) Alkalmazása

A PCM adatokból vett blokkokra Modifikált Diszkrét Koszinusz Transzformációt (MDCT) alkalmaznak. Az MDCT egy speciális diszkrét koszinusz transzformáció, amely a blokkot frekvencia tartományba transzformálja. Az MDCT különlegessége, hogy átfedő ablakokat használ, ami csökkenti a blokkhatásokból eredő torzításokat.

Az MDCT képlete:

\[
X_k = \sum_{n=0}^{2N-1} x_n \cos\left[\frac{\pi}{N}\left(n+\frac{1}{2}+\frac{N}{2}\right)\left(k+\frac{1}{2}\right)\right]
\]

ahol \(N\) a blokk mérete, \(x_n\) a bemeneti időtartománybeli minta, és \(X_k\) a frekvencia tartománybeli eredmény.

##### Kvantalizáció és Szűrés

Az MDCT után a frekvencia tartománybeli adatokat kvantalizálják. A kvantalizáció során a frekvencia komponenseket egy előre meghatározott kvantalizációs lépcső szerint osztják el, és az értékeket kerekítik. Ez a lépés szintén veszteséges, mivel a kvantalizációval információk vesznek el. Az MP3 algoritmus dinamikusan állítja be a kvantalizációs lépcsőt a pszichoakusztikai modell alapján, hogy minimalizálja az észlelhető minőségromlást.

##### Huffman Kódolás

A kvantalizált frekvencia adatokat Huffman kódolással tömörítik. A Huffman kódolás egy veszteségmentes tömörítési eljárás, amely rövid kódokat rendel a gyakran előforduló értékekhez, és hosszabb kódokat a ritkábban előforduló értékekhez. Ezzel tovább csökkenthető a hangfájl mérete anélkül, hogy további információ veszne el.

#### Dekódolás

Az MP3 dekódolási folyamata a kódolási lépések megfordításával történik. Először a Huffman kódokat dekódolják, majd a kvantalizált frekvencia komponenseket visszaalakítják. Ezt követően az inverz MDCT-t alkalmazva visszanyerik az időtartománybeli jelet. Végül a pszichoakusztikai modell alapján elvégzett változtatásokat figyelembe véve visszaállítják a PCM adatokat.

#### Példakód C++ Nyelven

Az alábbiakban bemutatok egy egyszerű példakódot C++ nyelven, amely egy 32 minta hosszúságú blokkra alkalmazza az MDCT-t:

```cpp
#include <iostream>
#include <vector>
#include <cmath>

const int N = 32;
const double PI = 3.14159265358979323846;

void MDCT(const std::vector<double>& input, std::vector<double>& output) {
    for (int k = 0; k < N; ++k) {
        double sum = 0.0;
        for (int n = 0; n < 2 * N; ++n) {
            sum += input[n] * cos(PI / N * (n + 0.5 + N / 2.0) * (k + 0.5));
        }
        output[k] = sum;
    }
}

int main() {
    std::vector<double> input = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    };
    std::vector<double> output(N, 0.0);

    MDCT(input, output);

    // Output the MDCT coefficients
    std::cout << "MDCT coefficients:\n";
    for (double val : output) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    return 0;
}
```

#### Bitráta és Minőség

Az MP3 fájlok bitrátája (bitrate) jelentős hatással van a hangminőségre és a fájlméretre. A bitráta azt jelzi, hogy egy másodpercnyi hangadat mekkora helyet foglal el, tipikusan kilobit/másodperc (kbps) egységben mérve. A gyakori bitráták közé tartozik a 128 kbps, 192 kbps és 320 kbps. Általában a magasabb bitráta jobb hangminőséget eredményez, de nagyobb fájlmérettel jár.

#### Alkalmazási Területek

Az MP3 formátumot számos területen használják, beleértve a digitális zenelejátszókat, okostelefonokat, számítógépeket és online streaming szolgáltatásokat. Az MP3 népszerűségét a kiváló tömörítési aránynak, a viszonylag jó hangminőségnek és a széleskörű támogatottságnak köszönheti.


### 6.3.3. Videó kompressziós algoritmusok (pl. H.264)
### Működés és alkalmazások

#### Bevezetés
A videókompresszió modern világunk egyik kulcsfontosságú technológiája, amely lehetővé teszi a nagy felbontású videók hatékony tárolását és továbbítását. Az egyik legelterjedtebb videó kompressziós szabvány, amely széles körben használt mind a professzionális mind az otthoni felhasználásban, a H.264, más néven AVC (Advanced Video Coding). Ez a fejezet mélyreható betekintést nyújt a H.264 működési elveibe, valamint annak gyakorlati alkalmazásaiba.

#### H.264 működési elvei

Az H.264 kompressziós algoritmus számos innovatív technikát alkalmaz a videó adatok hatékony csökkentése érdekében, miközben megőrzi a lehető legjobb vizuális minőséget. A következő szakaszok részletezik ezeket az elveket és technikákat.

##### 1. **Intraframe tömörítés (I-képkockák)**

Az intraframe tömörítés a képkockákon belüli redundancia csökkentésére összpontosít. Az I-képkockák a videó olyan képkockái, amelyeket teljes mértékben, önállóan tömörítenek, minden más képkocka nélkül. Ez általában egy DCT (Discrete Cosine Transform) alapú megközelítést használ, ahol a képkocka blokkjaira (általában 8x8 vagy 16x16 pixeles blokkok) felosztják, majd ezek a blokkok frekvenciatartományba kerülnek átalakításra.

A blokkok átalakítása után kvantálási lépések következnek, amelyek jelentősen csökkentik a kép adatmennyiségét azáltal, hogy a kevésbé fontos frekvenciakomponensek pontosságát csökkentik. Végül, a blokkadatokat entropia kódolással (például Huffman vagy aritmetikai kódolással) tömörítik tovább az adatok mennyiségének minimalizálása érdekében.

##### Példakód C++ nyelven az DCT transzformációra:
```cpp
#include <iostream>
#include <cmath>

const int N = 8;

// DCT Transform function
void dctTransform(double matrix[N][N], double result[N][N]) {
    for (int u = 0; u < N; ++u) {
        for (int v = 0; v < N; ++v) {
            double sum = 0.0;
            for (int x = 0; x < N; ++x) {
                for (int y = 0; y < N; ++y) {
                    sum += matrix[x][y] * 
                           cos((2 * x + 1) * u * M_PI / (2 * N)) * 
                           cos((2 * y + 1) * v * M_PI / (2 * N));
                }
            }
            double Cu = (u == 0) ? sqrt(1.0 / N) : sqrt(2.0 / N);
            double Cv = (v == 0) ? sqrt(1.0 / N) : sqrt(2.0 / N);
            result[u][v] = Cu * Cv * sum;
        }
    }
}

int main() {
    double matrix[N][N] = { /* Initial pixel values of an 8x8 block */ };
    double result[N][N];

    dctTransform(matrix, result);

    // Output the result
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << result[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
```
##### 2. **Interframe tömörítés (P és B-képkockák)**

Az interframe tömörítés a képkockák közötti redundancia csökkentésére összpontosít. A P-képkockák (prediktív képkockák) egy korábbi I vagy P-képkockából való elmozdulás révén kerülnek tömörítésre. A mozgáskompenzációs technikákat alkalmazzák, ahol az aktuális képkocka blokkjait keresik meg a korábbiakban, és kiszámítják a mozgásvektorokat.

A B-képkockák (biprediktív képkockák) két különböző képkocka (egy korábbi és egy jövőbeli) között végzik el a predikciót. Ez jelentős mennyiségű redundancia csökkentési lehetőséget biztosít.

##### 3. **Mozgáskompenzáció és mozgásvektorok**
A mozgáskompenzációs eljárás során a képkockák különböző blokkjait vizsgálják, hogy megállapítsák a blokkok elmozdulását. Ennek eredményeképpen mozgásvektorokat hoznak létre, amelyek a blokkok elmozdulását jelzik.

A mozgásvektorok és a maradék (az a rész amit nem lehetett a mozgáskompenzációval kitalálni) δ-al értéket DCT-n és kvantáláson mennek keresztül, hasonlóan az Intra blokkokhoz.

#### Alkalmazások

A H.264 szabvány széleskörű alkalmazása számos területen megfigyelhető, beleértve a következőket:

##### 1. **Streaming Szolgáltatások**
A H.264 kódolt videók széles körben használják a különböző video streaming platformok, mint például a Netflix, YouTube, és más videószolgáltatások. A H.264 rugalmassága és hatékonysága lehetővé teszi a magas minőségű videók továbbítását az interneten keresztül.

##### 2. **Blu-ray és fizikai média**
A Blu-ray lemez szabvány a H.264 kódolást használja annak érdekében, hogy nagy kapacitású, magas minőségű videókat tárolhasson. Ez lehetővé teszi akár több órás HD videó tárolását egy lemezen.

##### 3. **Biztonsági és megfigyelőrendszerek**
A H.264 videótömörítés megnöveli a biztonsági kamerák és megfigyelőrendszerek hatékonyságát azáltal, hogy csökkenti a tárolási és átviteli sávszélesség igényeket, így lehetővé teszi a hosszabb idejű felvételek tárolását és valós idejű monitorozást.

##### 4. **Videokonferenciák**
A H.264 jelentős szerepet játszik a videokonferencia rendszerekben is, mivel lehetővé teszi a valós idejű, nagy felbontású videó továbbítását alacsony sávszélesség mellett, ezáltal fokozva a kommunikációs élményt.

##### 5. **Mobil alkalmazások és camcorder-ek**
A H.264 széles körben használják mobil eszközökben és digitális kamerákban annak érdekében, hogy hatékonyan tárolják a nagy felbontású, mozgásban lévő tartalmakat, miközben minimalizálják a fájlméretet.
