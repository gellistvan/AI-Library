\newpage

## 1.14. Evolúciós és heurisztikus keresési módszerek

Az evolúciós és heurisztikus keresési módszerek olyan hatékony technikák, amelyek a komplex problémák megoldásában játszanak kulcsszerepet. Ezek a módszerek az optimalizáció és a keresési feladatok különböző területein alkalmazhatók, ahol a hagyományos algoritmusok nem feltétlenül nyújtanak kielégítő eredményeket. Az evolúciós keresési módszerek, mint például a genetikus algoritmusok, a természetes szelekció és az evolúció elvein alapulnak, míg a heurisztikus technikák, mint a Hill Climbing és a Simulated Annealing, gyakorlati megközelítéseket kínálnak a lokális és globális optimumok megtalálására. E fejezet célja, hogy bemutassa ezeket a technikákat, azok alkalmazási területeit és működési elveit, valamint összehasonlítsa hatékonyságukat különböző keresési problémák megoldásában.

### 1.14.1. Genetikus algoritmusok keresési alkalmazásai

A genetikus algoritmusok (GA-k) az evolúciós algoritmusok egy speciális osztályát alkotják, amelyek a természetes szelekció és a genetikai evolúció elvein alapulnak. Az 1970-es években John Holland és munkatársai által kifejlesztett GA-k azóta széles körben alkalmazzák különböző optimalizálási és keresési problémák megoldására. Ebben a fejezetben részletesen tárgyaljuk a genetikus algoritmusok működési elveit, kulcsfontosságú komponenseit, valamint azok alkalmazási területeit és előnyeit.

#### Működési elvek

A genetikus algoritmusok alapja egy populációs megközelítés, amelyben egy populáció egyedek (megoldások) halmazából áll. Az egyedek genotípusok formájában reprezentálják a potenciális megoldásokat. Az algoritmus az alábbi fő lépésekből áll:

1. **Inicializáció:** Az algoritmus egy kezdeti populációval indul, amely véletlenszerűen generált egyedekből áll. Az egyedek genotípusait általában bináris kódolással vagy más, a probléma természetéhez illeszkedő reprezentációval ábrázolják.

2. **Fitness értékelés:** Minden egyedhez fitness értéket rendelünk, amely megméri az adott megoldás minőségét a célfüggvény alapján. A fitness függvény kulcsfontosságú a keresési folyamat irányításában, mivel a szelekciós mechanizmusok ezt az értéket használják a legjobbnak ítélt egyedek kiválasztásához.

3. **Szelekció:** A szelekciós folyamat során a populáció legjobb egyedeit választják ki a reprodukcióra. A szelekciós mechanizmusok közé tartozik a rulettkerék-szelekció, a rangszelekció és a torna-szelekció. Ezek a mechanizmusok biztosítják, hogy a magasabb fitness értékű egyedek nagyobb valószínűséggel kerüljenek kiválasztásra, így növelve a populáció átlagos fitness szintjét a következő generációkban.

4. **Rekombináció (keresztezés):** A kiválasztott egyedekből új utódok (megoldások) jönnek létre a keresztezési operátor alkalmazásával. A keresztezés során az egyedek genetikai információit kombináljuk, hogy új genotípusokat hozzunk létre. A leggyakrabban használt keresztezési módszerek közé tartozik az egypontos és a többpontos keresztezés.

5. **Mutáció:** A mutációs operátor véletlenszerű változtatásokat vezet be az egyedek genotípusában, hogy új genetikai variációkat hozzon létre és megakadályozza a populáció korai konvergenciáját. A mutációs ráta általában alacsony, de kritikus szerepet játszik a keresési tér alapos feltérképezésében.

6. **Utódlás:** Az új generáció létrehozása után az algoritmus visszatér a fitness értékelés lépéséhez, és az egész folyamat addig ismétlődik, amíg el nem érjük a megadott stop feltételt (például egy adott számú generáció elérése, vagy egy bizonyos fitness érték elérése).

Az alábbiakban egy példát mutatunk be egy egyszerű genetikus algoritmusra C++ nyelven.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <ctime>

// Egyed reprezentációja
struct Individual {
    std::vector<int> genes;
    double fitness;

    Individual(int length) : genes(length), fitness(0.0) {}
};

// Fitness függvény (példa célfüggvény)
double evaluateFitness(const Individual& ind) {
    // Itt definiáljuk a célfüggvényt
    double fitness = 0.0;
    for (int gene : ind.genes) {
        fitness += gene; // Egyszerű példa: összegezzük a géneket
    }
    return fitness;
}

// Inicializáció
std::vector<Individual> initializePopulation(int populationSize, int geneLength) {
    std::vector<Individual> population(populationSize, Individual(geneLength));
    std::default_random_engine generator(static_cast<unsigned int>(std::time(0)));
    std::uniform_int_distribution<int> distribution(0, 1);

    for (auto& ind : population) {
        for (int& gene : ind.genes) {
            gene = distribution(generator);
        }
        ind.fitness = evaluateFitness(ind);
    }
    return population;
}

// Szelekció (torna-szelekció)
Individual tournamentSelection(const std::vector<Individual>& population) {
    const int tournamentSize = 3;
    std::vector<Individual> tournament;
    std::default_random_engine generator(static_cast<unsigned int>(std::time(0)));
    std::uniform_int_distribution<int> distribution(0, population.size() - 1);

    for (int i = 0; i < tournamentSize; ++i) {
        tournament.push_back(population[distribution(generator)]);
    }
    return *std::max_element(tournament.begin(), tournament.end(), [](const Individual& a, const Individual& b) {
        return a.fitness < b.fitness;
    });
}

// Keresztezés (egypontos)
Individual crossover(const Individual& parent1, const Individual& parent2) {
    int geneLength = parent1.genes.size();
    Individual offspring(geneLength);
    std::default_random_engine generator(static_cast<unsigned int>(std::time(0)));
    std::uniform_int_distribution<int> distribution(0, geneLength - 1);

    int crossoverPoint = distribution(generator);
    for (int i = 0; i < geneLength; ++i) {
        if (i < crossoverPoint) {
            offspring.genes[i] = parent1.genes[i];
        } else {
            offspring.genes[i] = parent2.genes[i];
        }
    }
    return offspring;
}

// Mutáció
void mutate(Individual& ind, double mutationRate) {
    std::default_random_engine generator(static_cast<unsigned int>(std::time(0)));
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int& gene : ind.genes) {
        if (distribution(generator) < mutationRate) {
            gene = 1 - gene; // Bitflip mutáció
        }
    }
}

// Genetikus algoritmus
void geneticAlgorithm(int populationSize, int geneLength, int generations, double mutationRate) {
    std::vector<Individual> population = initializePopulation(populationSize, geneLength);

    for (int generation = 0; generation < generations; ++generation) {
        std::vector<Individual> newPopulation;
        for (int i = 0; i < populationSize; ++i) {
            Individual parent1 = tournamentSelection(population);
            Individual parent2 = tournamentSelection(population);
            Individual offspring = crossover(parent1, parent2);
            mutate(offspring, mutationRate);
            offspring.fitness = evaluateFitness(offspring);
            newPopulation.push_back(offspring);
        }
        population = newPopulation;

        // Legjobb egyed keresése
        Individual best = *std::max_element(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
            return a.fitness < b.fitness;
        });
        std::cout << "Generation " << generation << " Best Fitness: " << best.fitness << std::endl;
    }
}

int main() {
    const int populationSize = 100;
    const int geneLength = 10;
    const int generations = 50;
    const double mutationRate = 0.01;

    geneticAlgorithm(populationSize, geneLength, generations, mutationRate);
    return 0;
}
```

#### Alkalmazási területek

A genetikus algoritmusokat számos területen alkalmazzák, különösen olyan problémák esetében, ahol a keresési tér nagy és összetett. Az alábbiakban néhány fontos alkalmazási területet mutatunk be:

1. **Optimalizálási problémák:** GA-kat gyakran alkalmaznak nehéz optimalizálási problémák megoldására, mint például az utazó ügynök problémája (TSP), a gráfszínezés, és a menetrend-készítés. E problémákra jellemző, hogy nagyméretű keresési terekkel rendelkeznek, amelyek hagyományos módszerekkel nehezen kezelhetők.

2. **Mérnöki tervezés:** A genetikus algoritmusokat széles körben alkalmazzák mérnöki optimalizálási problémák megoldására, például gépészeti struktúrák tervezésében, ahol a cél a szerkezet tömegének csökkentése, miközben biztosítjuk a szükséges szilárdságot és megbízhatóságot.

3. **Mesterséges intelligencia és gépi tanulás:** GA-kat használnak gépi tanulási algoritmusok hiperparamétereinek optimalizálására, különösen neurális hálózatok esetében. Emellett az adaptív rendszertervezésben is szerepet játszanak, ahol a GA-k segítségével lehet azonosítani és finomítani a rendszerparamétereket.

4. **Bioinformatika:** A genetikus algoritmusokat gyakran használják biológiai szekvenciák, például DNS vagy fehérje szekvenciák optimalizálására és elemzésére. Segítségükkel például optimalizálható a fehérjék hajtogatódási folyamata, ami kulcsfontosságú az új gyógyszerek fejlesztésében.

5. **Gazdasági modellezés:** A GA-k alkalmazása a gazdasági modellezésben és a pénzügyi tervezésben is elterjedt. Segítségükkel például optimalizálható a portfóliók összetétele, figyelembe véve a kockázati tényezőket és a hozamokat.

6. **Robotika és automatizálás:** A genetikus algoritmusokat robotikai alkalmazásokban is használják, például a robotok mozgásának és viselkedésének optimalizálására. A GA-k segítenek megtalálni az optimális útvonalakat és mozgási mintákat, amelyek maximális hatékonyságot és minimális energiaveszteséget biztosítanak.

#### Előnyök és korlátok

A genetikus algoritmusok számos előnyt kínálnak, de néhány korlátot is figyelembe kell venni:

**Előnyök:**

- **Rugalmasság:** A GA-k nem igényelnek differenciálhatósági vagy folytonossági feltételeket a célfüggvényre vonatkozóan, így széles körben alkalmazhatók különféle problémákra.
- **Globális keresés:** A GA-k képesek elkerülni a lokális optimumokban való elakadást, mivel a populációalapú megközelítésük és a mutációs operátor biztosítja a keresési tér alapos feltérképezését.
- **Párhuzamos feldolgozás:** A GA-k párhuzamosan feldolgozhatók, ami növeli a számítási hatékonyságot és csökkenti a futási időt nagy méretű problémák esetén.

**Korlátok:**

- **Paraméterérzékenység:** A GA-k teljesítménye erősen függ a paraméterek (pl. populáció mérete, mutációs ráta, keresztezési arány) megfelelő beállításától. A nem megfelelő paraméterek használata jelentősen ronthatja az algoritmus hatékonyságát.
- **Számítási költség:** A GA-k számításigényesek lehetnek, különösen nagy méretű populációk és generációk esetén. Ezért szükség lehet jelentős számítási erőforrásokra a nagy méretű és komplex problémák megoldásához.
- **Konvergencia sebessége:** A GA-k konvergenciája lassú lehet, különösen, ha a keresési tér nagyon összetett és sok lokális optimumot tartalmaz. Ez hosszabb futási időt eredményezhet a kívánt megoldás eléréséhez.

Összefoglalva, a genetikus algoritmusok hatékony és sokoldalú eszközök, amelyek számos alkalmazási területen nyújtanak megoldást a komplex optimalizálási és keresési problémákra. A megfelelő paraméterezés és az algoritmus gondos implementálása mellett a GA-k képesek kiváló eredményeket elérni számos különféle feladatban.


### 1.14.2. Heurisztikus keresési technikák (pl. Hill Climbing, Simulated Annealing)

A heurisztikus keresési technikák olyan módszerek, amelyek célja a keresési tér gyors és hatékony feltérképezése annak érdekében, hogy jó (bár nem feltétlenül optimális) megoldásokat találjanak komplex problémákra. Ezek a módszerek gyakran olyan környezetekben használatosak, ahol a teljes keresési tér átvizsgálása túlzottan idő- és erőforrás-igényes lenne. Ebben a fejezetben két jelentős heurisztikus keresési technikát, a Hill Climbing és a Simulated Annealing módszereket tárgyaljuk részletesen, bemutatva működési elveiket, előnyeiket, korlátaikat és alkalmazási területeiket.

#### Hill Climbing

A Hill Climbing egy egyszerű és intuitív heurisztikus keresési algoritmus, amelyet gyakran használnak lokális keresési feladatokhoz. A módszer alapgondolata, hogy egy kezdőpontból indulva folyamatosan javítjuk a megoldást, amíg el nem érjük a lokális optimumot. Az algoritmus iteratív módon működik, minden lépésben a jelenlegi megoldás környezetében lévő legjobb szomszédos megoldást választja.

##### Működési elv

1. **Inicializáció:** Az algoritmus egy kezdő megoldással indul, amely lehet véletlenszerűen generált vagy valamilyen heurisztika alapján meghatározott.

2. **Szomszédos megoldások kiértékelése:** A jelenlegi megoldás szomszédos megoldásait kiértékeljük a célfüggvény alapján.

3. **Legjobb szomszéd kiválasztása:** A szomszédos megoldások közül kiválasztjuk a legjobbat. Ha ez a megoldás jobb, mint a jelenlegi megoldás, akkor azt fogadjuk el új jelenlegi megoldásként.

4. **Iteráció:** Az eljárás addig folytatódik, amíg nem találunk olyan szomszédos megoldást, amely jobb, mint a jelenlegi megoldás. Ekkor elérjük a lokális optimumot.

##### Előnyök és hátrányok

A Hill Climbing algoritmus előnyei közé tartozik az egyszerűség és a gyorsaság, mivel kevés memóriát és számítási kapacitást igényel. Azonban a módszer hajlamos a lokális optimumokban való elakadásra, ami azt jelenti, hogy nem mindig találja meg a globális optimumot, különösen nagy és komplex keresési terek esetén.

##### Példa

Az alábbiakban egy egyszerű Hill Climbing algoritmust mutatunk be C++ nyelven.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <ctime>

// Célfüggvény (példa)
double objectiveFunction(const std::vector<int>& solution) {
    double value = 0.0;
    for (int x : solution) {
        value += x; // Egyszerű példa: összegezzük az elemeket
    }
    return value;
}

// Szomszédos megoldások generálása
std::vector<std::vector<int>> generateNeighbors(const std::vector<int>& solution) {
    std::vector<std::vector<int>> neighbors;
    for (size_t i = 0; i < solution.size(); ++i) {
        std::vector<int> neighbor = solution;
        neighbor[i] = 1 - neighbor[i]; // Bitflip
        neighbors.push_back(neighbor);
    }
    return neighbors;
}

// Hill Climbing algoritmus
std::vector<int> hillClimbing(const std::vector<int>& initialSolution) {
    std::vector<int> currentSolution = initialSolution;
    double currentFitness = objectiveFunction(currentSolution);

    while (true) {
        std::vector<std::vector<int>> neighbors = generateNeighbors(currentSolution);
        std::vector<int> bestNeighbor;
        double bestFitness = currentFitness;

        for (const auto& neighbor : neighbors) {
            double fitness = objectiveFunction(neighbor);
            if (fitness > bestFitness) {
                bestFitness = fitness;
                bestNeighbor = neighbor;
            }
        }

        if (bestFitness > currentFitness) {
            currentSolution = bestNeighbor;
            currentFitness = bestFitness;
        } else {
            break; // Lokális optimum
        }
    }
    return currentSolution;
}

int main() {
    const int solutionSize = 10;
    std::vector<int> initialSolution(solutionSize, 0);

    std::default_random_engine generator(static_cast<unsigned int>(std::time(0)));
    std::uniform_int_distribution<int> distribution(0, 1);
    for (int& gene : initialSolution) {
        gene = distribution(generator);
    }

    std::vector<int> bestSolution = hillClimbing(initialSolution);
    std::cout << "Best Solution: ";
    for (int x : bestSolution) {
        std::cout << x << " ";
    }
    std::cout << "\nObjective Value: " << objectiveFunction(bestSolution) << std::endl;

    return 0;
}
```

#### Simulated Annealing

A Simulated Annealing (SA) egy heurisztikus keresési módszer, amely a fizikai hőkezelési folyamatokból merít ihletet, különösen a fémek és egyéb anyagok kristályos szerkezetének lassú hűtéséből. Az SA algoritmus célja a globális optimum megtalálása a keresési térben azáltal, hogy kezdetben nagy mértékű változtatásokat hajt végre, majd fokozatosan csökkenti ezeknek a változtatásoknak a mértékét.

##### Működési elv

1. **Inicializáció:** Az algoritmus egy kezdeti megoldással és egy magas hőmérsékleti értékkel indul.

2. **Szomszédos megoldások generálása:** A jelenlegi megoldás szomszédos megoldásait véletlenszerűen generáljuk.

3. **Elfogadás valószínűsége:** A szomszédos megoldás elfogadása a Metropolis-kritérium alapján történik. Ha az új megoldás jobb, mint a jelenlegi, akkor mindig elfogadjuk. Ha rosszabb, akkor valószínűség alapján döntünk az elfogadásról, amely függ a hőmérséklettől és a megoldások közötti különbségtől.

4. **Hőmérséklet csökkentése:** A hőmérsékletet fokozatosan csökkentjük egy hűtési ütemterv szerint.

5. **Iteráció:** Az eljárás addig folytatódik, amíg a hőmérséklet el nem éri a minimális értéket vagy egy előre meghatározott számú iterációt végre nem hajtunk.

##### Előnyök és hátrányok

A Simulated Annealing algoritmus egyik legnagyobb előnye, hogy képes elkerülni a lokális optimumokban való elakadást, mivel kezdetben nagyobb lépésekben keres és fokozatosan finomítja a megoldást. Azonban a módszernek megfelelő hűtési ütemtervet kell alkalmazni, és a paraméterek beállítása bonyolult lehet.

##### Példa

Az alábbiakban egy egyszerű Simulated Annealing algoritmust mutatunk be C++ nyelven.

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>

// Célfüggvény (példa)
double objectiveFunction(const std::vector<int>& solution) {
    double value = 0.0;
    for (int x : solution) {
        value += x; // Egyszerű példa: összegezzük az elemeket
    }
    return value;
}

// Szomszédos megoldások generálása
std::vector<int> generateNeighbor(const std::vector<int>& solution) {
    std::vector<int> neighbor = solution;
    std::default_random_engine generator(static_cast<unsigned int>(std::time(0)));
    std::uniform_int_distribution<int> distribution(0, solution.size() - 1);

    int index = distribution(generator);
    neighbor[index] = 1 - neighbor[index]; // Bitflip
    return neighbor;
}

// Simulated Annealing algoritmus
std::vector<int> simulatedAnnealing(const std::vector<int>& initialSolution, double initialTemp, double finalTemp, double alpha) {
    std::vector<int> currentSolution = initialSolution;
    double currentFitness = objectiveFunction(currentSolution);
    double temperature = initialTemp;

    std::default_random_engine generator(static_cast<unsigned int>(std::time(0)));
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    while

 (temperature > finalTemp) {
        std::vector<int> neighbor = generateNeighbor(currentSolution);
        double neighborFitness = objectiveFunction(neighbor);
        double deltaFitness = neighborFitness - currentFitness;

        if (deltaFitness > 0 || exp(deltaFitness / temperature) > distribution(generator)) {
            currentSolution = neighbor;
            currentFitness = neighborFitness;
        }

        temperature *= alpha; // Hűtési ütemterv
    }
    return currentSolution;
}

int main() {
    const int solutionSize = 10;
    std::vector<int> initialSolution(solutionSize, 0);

    std::default_random_engine generator(static_cast<unsigned int>(std::time(0)));
    std::uniform_int_distribution<int> distribution(0, 1);
    for (int& gene : initialSolution) {
        gene = distribution(generator);
    }

    double initialTemp = 1000.0;
    double finalTemp = 0.01;
    double alpha = 0.99;

    std::vector<int> bestSolution = simulatedAnnealing(initialSolution, initialTemp, finalTemp, alpha);
    std::cout << "Best Solution: ";
    for (int x : bestSolution) {
        std::cout << x << " ";
    }
    std::cout << "\nObjective Value: " << objectiveFunction(bestSolution) << std::endl;

    return 0;
}
```

#### Alkalmazási területek

Mind a Hill Climbing, mind a Simulated Annealing algoritmusokat széles körben alkalmazzák különféle területeken, különösen ott, ahol a keresési tér nagy és komplex, és a hagyományos módszerek nem nyújtanak hatékony megoldásokat. Az alábbiakban néhány jelentős alkalmazási területet mutatunk be:

1. **Optimalizálási problémák:** Ezek az algoritmusok hatékonyak a kombinatorikus optimalizálási problémák megoldásában, például az utazó ügynök problémájában (TSP), a gráfszínezésben, és a menetrend-készítésben.

2. **Mérnöki tervezés:** A heurisztikus keresési technikákat gyakran alkalmazzák mérnöki optimalizálási problémákban, például mechanikai rendszerek tervezésében és elektromos áramkörök optimalizálásában.

3. **Gépi tanulás:** A Hill Climbing és a Simulated Annealing algoritmusokat használják a gépi tanulási modellek hiperparamétereinek optimalizálására, különösen a neurális hálózatok esetében.

4. **Bioinformatika:** Ezek az algoritmusok hasznosak a biológiai szekvenciák összehasonlításában és a fehérje szerkezetek előrejelzésében.

Összefoglalva, a heurisztikus keresési technikák, mint a Hill Climbing és a Simulated Annealing, hatékony és sokoldalú eszközök, amelyek számos alkalmazási területen nyújtanak megoldást a komplex optimalizálási és keresési problémákra.