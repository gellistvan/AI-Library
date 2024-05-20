\newpage

## 11. Könyvtárak és API-k

A GPGPU (General-Purpose computing on Graphics Processing Units) világában az optimális teljesítmény és hatékonyság elérése érdekében különböző specializált könyvtárak és API-k állnak rendelkezésre. Ezek az eszközök megkönnyítik a fejlesztők számára a komplex számítási feladatok implementálását anélkül, hogy mélyreható ismeretekkel kellene rendelkezniük a GPU-k alacsony szintű programozásáról. Ebben a fejezetben négy fontos könyvtárat és API-t fogunk bemutatni, amelyek jelentős szerepet játszanak a GPU-alapú számítások terén. A Thrust könyvtár a párhuzamos adatstruktúrákat és algoritmusokat kínál, a cuBLAS a lineáris algebrai műveletek gyors végrehajtását teszi lehetővé, a cuFFT a Fourier transzformációk hatékony megvalósítását biztosítja, míg a cuDNN a mélytanulási feladatok optimalizálásában nyújt segítséget. Ezek az eszközök együttesen jelentős mértékben megkönnyítik és felgyorsítják a fejlesztési folyamatokat, lehetővé téve a kutatók és mérnökök számára, hogy teljes mértékben kihasználják a GPU-k hatalmas számítási kapacitását.

### 11. Könyvtárak és API-k

A GPGPU (General-Purpose computing on Graphics Processing Units) területén számos könyvtár és API áll rendelkezésre, hogy megkönnyítse és optimalizálja a különféle számítási feladatokat. Ezek közé tartozik a Thrust, cuBLAS, cuFFT és cuDNN, melyek mindegyike különböző típusú problémák megoldására specializálódott. Ebben a fejezetben ezeket a könyvtárakat és API-kat tárgyaljuk részletesen, különös figyelmet fordítva a Thrust könyvtárra.

#### 11.1 Thrust könyvtár

A Thrust könyvtár egy C++ template könyvtár, amely az STL (Standard Template Library) mintájára készült, de kifejezetten GPU-alapú párhuzamos számításokra optimalizálták. A Thrust segítségével könnyen használhatunk párhuzamos algoritmusokat és adatstruktúrákat, anélkül, hogy mélyebb ismeretekkel rendelkeznénk a CUDA programozásról. Ez a könyvtár ideális eszköz azok számára, akik gyors és hatékony GPU-alapú megoldásokat szeretnének fejleszteni.

##### Adatstruktúrák és algoritmusok

A Thrust könyvtár különféle adatstruktúrákat és algoritmusokat biztosít, amelyekkel könnyedén végezhetünk párhuzamos számításokat. Néhány fontosabb adatstruktúra és algoritmus, amelyeket a Thrust kínál:

- **Adatstruktúrák**:
    - `thrust::host_vector` és `thrust::device_vector`: Ezek az adatstruktúrák hasonlóak az STL `std::vector`-hoz, de az egyik a CPU memóriában (host_vector), míg a másik a GPU memóriában (device_vector) tárolja az adatokat.
    - **Példa**:

      ```cpp
      #include <thrust/host_vector.h>
      #include <thrust/device_vector.h>
      #include <thrust/copy.h>
      #include <iostream>
  
      int main() {
          // Host vektor inicializálása
          thrust::host_vector<int> h_vec(4);
          h_vec[0] = 10;
          h_vec[1] = 20;
          h_vec[2] = 30;
          h_vec[3] = 40;
  
          // Másolás a device vektorba
          thrust::device_vector<int> d_vec = h_vec;
  
          // Device vektor másolása vissza a host vektorba
          thrust::host_vector<int> h_vec_copy = d_vec;
  
          // Eredmények kiíratása
          for (int i = 0; i < h_vec_copy.size(); i++) {
              std::cout << "Element " << i << ": " << h_vec_copy[i] << std::endl;
          }
  
          return 0;
      }
      ```

- **Algoritmusok**:
    - `thrust::sort`: Rendezés algoritmus GPU-n.
    - `thrust::reduce`: Összeadás (reduce) algoritmus GPU-n.
    - `thrust::transform`: Elem szintű transzformációs algoritmus GPU-n.

  **Példa**:

  ```cpp
  #include <thrust/device_vector.h>
  #include <thrust/sort.h>
  #include <thrust/functional.h>
  #include <iostream>

  int main() {
      // Device vektor inicializálása
      thrust::device_vector<int> d_vec(4);
      d_vec[0] = 30;
      d_vec[1] = 10;
      d_vec[2] = 40;
      d_vec[3] = 20;

      // Vektor rendezése növekvő sorrendbe
      thrust::sort(d_vec.begin(), d_vec.end());

      // Eredmények kiíratása
      for (int i = 0; i < d_vec.size(); i++) {
          std::cout << "Element " << i << ": " << d_vec[i] << std::endl;
      }

      return 0;
  }
  ```

    - **Transformálás**:

  ```cpp
  #include <thrust/device_vector.h>
  #include <thrust/transform.h>
  #include <thrust/functional.h>
  #include <iostream>

  int main() {
      // Device vektor inicializálása
      thrust::device_vector<int> d_vec(4);
      d_vec[0] = 1;
      d_vec[1] = 2;
      d_vec[2] = 3;
      d_vec[3] = 4;

      // Vektor transzformálása: minden elem duplázása
      thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), thrust::placeholders::_1 * 2);

      // Eredmények kiíratása
      for (int i = 0; i < d_vec.size(); i++) {
          std::cout << "Element " << i << ": " << d_vec[i] << std::endl;
      }

      return 0;
  }
  ```

##### Thrust és CUDA integráció

A Thrust könyvtár könnyen integrálható a CUDA-val, lehetővé téve a hibrid megoldások fejlesztését, amelyek mind a Thrust, mind a CUDA alacsony szintű képességeit kihasználják. Például, egyedi CUDA kernel-ek használhatók a Thrust által biztosított adatstruktúrákkal és algoritmusokkal kombinálva.

**Példa** egyedi CUDA kernel integrációra:

```cpp
#include <thrust/device_vector.h>
#include <iostream>

// CUDA kernel
__global__ void increment_kernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1;
    }
}

int main() {
    // Device vektor inicializálása
    thrust::device_vector<int> d_vec(4);
    d_vec[0] = 1;
    d_vec[1] = 2;
    d_vec[2] = 3;
    d_vec[3] = 4;

    // Kernel meghívása
    int *raw_ptr = thrust::raw_pointer_cast(d_vec.data());
    increment_kernel<<<1, 4>>>(raw_ptr, d_vec.size());
    cudaDeviceSynchronize();

    // Eredmények kiíratása
    for (int i = 0; i < d_vec.size(); i++) {
        std::cout << "Element " << i << ": " << d_vec[i] << std::endl;
    }

    return 0;
}
```

Ez a példa bemutatja, hogyan használhatunk egyedi CUDA kernel-eket a Thrust könyvtárral együtt, lehetővé téve a még nagyobb rugalmasságot és teljesítményt.

##### Összefoglalás

A Thrust könyvtár egy erőteljes eszköz a GPU-alapú párhuzamos számításokhoz, amely megkönnyíti a fejlesztők számára a párhuzamos algoritmusok és adatstruktúrák használatát. Az egyszerű integráció a CUDA-val és a számos beépített funkció lehetővé teszi a hatékony és gyors alkalmazások fejlesztését, minimalizálva a kód bonyolultságát és fejlesztési időt.

### 11.2 cuBLAS

A cuBLAS (CUDA Basic Linear Algebra Subprograms) könyvtár egy GPU-alapú könyvtár, amely a lineáris algebrai rutinok végrehajtására szolgál. Ez a könyvtár a BLAS (Basic Linear Algebra Subprograms) interfészt implementálja, de a CUDA architektúrához optimalizálva. A cuBLAS segítségével a fejlesztők hatékonyan és gyorsan hajthatnak végre mátrixműveleteket a GPU-n, kihasználva annak párhuzamos feldolgozási képességeit. Ebben a fejezetben részletesen tárgyaljuk a cuBLAS könyvtárat, bemutatva annak főbb funkcióit és használati módjait.

#### cuBLAS alapjai

A cuBLAS könyvtár különféle lineáris algebrai műveleteket támogat, beleértve a vektor- és mátrixműveleteket is. Ezek közé tartoznak a skalárszorzások, vektorszumok, mátrix-skalár szorzások, mátrix-mátrix szorzások, mátrix transzponálások és inverziók. A cuBLAS használatához szükséges a CUDA fejlesztői környezet és a cuBLAS könyvtár telepítése.

#### cuBLAS inicializálása és lezárása

A cuBLAS használatának megkezdése előtt inicializálnunk kell a könyvtárat, és a munka befejeztével le kell zárnunk azt. Az inicializálás és lezárás a következőképpen történik:

```cpp
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

int main() {
    // cuBLAS könyvtár kezelő
    cublasHandle_t handle;

    // cuBLAS inicializálása
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS initialization failed!" << std::endl;
        return EXIT_FAILURE;
    }

    // ...

    // cuBLAS lezárása
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS shutdown failed!" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```

#### Vektor és mátrix műveletek

A cuBLAS számos vektor- és mátrixműveletet támogat. Az alábbiakban néhány gyakran használt műveletet mutatunk be példákkal.

##### Vektor szorzása skalárral

A vektor szorzása egy skalárral egy alapvető művelet, amelyet a `cublasSscal` (float) vagy `cublasDscal` (double) függvényekkel végezhetünk el.

**Példa**:

```cpp
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

int main() {
    // cuBLAS könyvtár kezelő
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Vektor mérete
    int n = 5;
    float alpha = 2.0f;

    // Host vektor
    float h_x[] = {1, 2, 3, 4, 5};

    // Device vektor
    float* d_x;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Vektor szorzása skalárral
    cublasSscal(handle, n, &alpha, d_x, 1);

    // Eredmény másolása vissza a hostra
    cudaMemcpy(h_x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Eredmények kiíratása
    for (int i = 0; i < n; i++) {
        std::cout << "Element " << i << ": " << h_x[i] << std::endl;
    }

    // Memória felszabadítása
    cudaFree(d_x);
    cublasDestroy(handle);

    return 0;
}
```

##### Mátrix-skalár szorzás

A cuBLAS lehetővé teszi mátrixok szorzását skalárokkal is. Ehhez a `cublasSgeam` függvényt használhatjuk, amely általános mátrix-mátrix műveletekre szolgál.

**Példa**:

```cpp
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

int main() {
    // cuBLAS könyvtár kezelő
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Mátrix méretei
    int m = 2, n = 2;
    float alpha = 2.0f;
    float beta = 0.0f;

    // Host mátrix
    float h_A[] = {1, 2, 3, 4};

    // Device mátrix
    float* d_A;
    float* d_C;
    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));
    cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Mátrix-skalár szorzás
    cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &alpha, d_A, m, &beta, d_A, m, d_C, m);

    // Eredmény másolása vissza a hostra
    cudaMemcpy(h_A, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Eredmények kiíratása
    for (int i = 0; i < m * n; i++) {
        std::cout << "Element " << i << ": " << h_A[i] << std::endl;
    }

    // Memória felszabadítása
    cudaFree(d_A);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}
```

##### Mátrix-mátrix szorzás

A mátrix-mátrix szorzás az egyik legfontosabb művelet a lineáris algebrában, amelyet a cuBLAS-ban a `cublasSgemm` (float) vagy `cublasDgemm` (double) függvényekkel hajthatunk végre.

**Példa**:

```cpp
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

int main() {
    // cuBLAS könyvtár kezelő
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Mátrix méretei
    int m = 2, n = 2, k = 2;
    float alpha = 1.0f;
    float beta = 0.0f;

    // Host mátrixok
    float h_A[] = {1, 2, 3, 4};
    float h_B[] = {5, 6, 7, 8};
    float h_C[4];

    // Device mátrixok
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));
    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    // Mátrix-mátrix szorzás: C = alpha * A * B + beta * C
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);

    // Eredmény másolása vissza a hostra
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Eredmények kiíratása
    for (int i = 0; i < m * n; i++) {
        std::cout << "Element " << i << ": " << h_C[i] << std::endl;
    }

    // Memória felszabadítása
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}
```

##### Mátrix transzponálás

A mátrix transzponálás gyakran szükséges művelet, amelyet a `cublasSgeam` függvénnyel végezhetünk el.

**Példa**:

```cpp
#include <cuda_runtime.h>


#include <cublas_v2.h>
#include <iostream>

int main() {
    // cuBLAS könyvtár kezelő
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Mátrix méretei
    int m = 2, n = 3;

    // Host mátrix
    float h_A[] = {1, 2, 3, 4, 5, 6};
    float h_C[6];

    // Device mátrixok
    float* d_A;
    float* d_C;
    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));
    cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Mátrix transzponálás: C = A^T
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &alpha, d_A, m, &beta, d_A, n, d_C, n);

    // Eredmény másolása vissza a hostra
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Eredmények kiíratása
    for (int i = 0; i < m * n; i++) {
        std::cout << "Element " << i << ": " << h_C[i] << std::endl;
    }

    // Memória felszabadítása
    cudaFree(d_A);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}
```

#### Összefoglalás

A cuBLAS könyvtár egy rendkívül hatékony eszköz a lineáris algebrai műveletek GPU-n történő végrehajtásához. A fent bemutatott példák segítségével láthatjuk, hogyan lehet egyszerűen és hatékonyan végrehajtani különféle vektor- és mátrixműveleteket a cuBLAS használatával. A cuBLAS könyvtár lehetővé teszi, hogy a fejlesztők kihasználják a GPU párhuzamos feldolgozási képességeit, így jelentősen megnövelve a számítási feladatok teljesítményét.

### 11.3 cuFFT

A cuFFT (CUDA Fast Fourier Transform) könyvtár a Fourier-transzformációk hatékony végrehajtására szolgál GPU-kon. A Fourier-transzformáció az egyik legfontosabb eszköz a jel- és képfeldolgozásban, tudományos számításokban, és számos mérnöki alkalmazásban. A cuFFT könyvtár lehetővé teszi a fejlesztők számára, hogy gyorsan és hatékonyan hajtsanak végre Fourier-transzformációkat nagy méretű adatokon, kihasználva a GPU párhuzamos feldolgozási képességeit.

#### Fourier-transzformációk alapjai

A Fourier-transzformáció egy matematikai művelet, amely egy idő- vagy térbeli függvényt egy frekvencia-tartománybeli függvénnyé alakít át. A diszkrét Fourier-transzformáció (DFT) a Fourier-transzformáció diszkrét változata, amelyet diszkrét jelek esetén alkalmaznak. A gyors Fourier-transzformáció (FFT) egy hatékony algoritmus a DFT kiszámítására, amely jelentős számítási előnyt biztosít nagy adathalmazok esetén.

#### cuFFT alapjai

A cuFFT könyvtár különféle Fourier-transzformációs műveleteket támogat, beleértve az egy- és többdimenziós transzformációkat is. A cuFFT használatához szükséges a CUDA fejlesztői környezet és a cuFFT könyvtár telepítése.

#### cuFFT inicializálása és használata

A cuFFT használatának megkezdése előtt létre kell hoznunk egy FFT tervet, amely meghatározza a transzformáció paramétereit. A művelet végrehajtása után a tervet meg kell semmisítenünk. Az alábbi példák bemutatják a cuFFT alapvető használatát.

##### 1D FFT végrehajtása

Az egy dimenziós FFT a leggyakoribb művelet, amelyet a cuFFT segítségével végrehajthatunk.

**Példa**:

```cpp
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

int main() {
    // Adat mérete
    int N = 8;
    cufftComplex h_data[N];

    // Adatok inicializálása
    for (int i = 0; i < N; i++) {
        h_data[i].x = i;
        h_data[i].y = 0;
    }

    // Device adat
    cufftComplex* d_data;
    cudaMalloc(&d_data, N * sizeof(cufftComplex));
    cudaMemcpy(d_data, h_data, N * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    // FFT terv létrehozása
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);

    // FFT végrehajtása
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

    // Eredmény másolása vissza a hostra
    cudaMemcpy(h_data, d_data, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // Eredmények kiíratása
    for (int i = 0; i < N; i++) {
        std::cout << "Element " << i << ": (" << h_data[i].x << ", " << h_data[i].y << ")" << std::endl;
    }

    // Memória felszabadítása
    cufftDestroy(plan);
    cudaFree(d_data);

    return 0;
}
```

Ebben a példában egy 1D FFT-t hajtunk végre egy 8 elemű komplex adathalmazon. Az adatok inicializálása után azokat a GPU memóriába másoljuk, létrehozzuk az FFT tervet, majd végrehajtjuk az FFT-t. Az eredményeket visszamásoljuk a host memóriába, és kiíratjuk őket.

##### 2D FFT végrehajtása

A 2D FFT hasonlóan működik az 1D FFT-hez, de kétdimenziós adathalmazokon hajtja végre a transzformációt.

**Példa**:

```cpp
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

int main() {
    // Adat méretei
    int Nx = 4;
    int Ny = 4;
    cufftComplex h_data[Nx * Ny];

    // Adatok inicializálása
    for (int i = 0; i < Nx * Ny; i++) {
        h_data[i].x = i;
        h_data[i].y = 0;
    }

    // Device adat
    cufftComplex* d_data;
    cudaMalloc(&d_data, Nx * Ny * sizeof(cufftComplex));
    cudaMemcpy(d_data, h_data, Nx * Ny * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    // FFT terv létrehozása
    cufftHandle plan;
    cufftPlan2d(&plan, Nx, Ny, CUFFT_C2C);

    // FFT végrehajtása
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

    // Eredmény másolása vissza a hostra
    cudaMemcpy(h_data, d_data, Nx * Ny * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // Eredmények kiíratása
    for (int i = 0; i < Nx * Ny; i++) {
        std::cout << "Element " << i << ": (" << h_data[i].x << ", " << h_data[i].y << ")" << std::endl;
    }

    // Memória felszabadítása
    cufftDestroy(plan);
    cudaFree(d_data);

    return 0;
}
```

Ebben a példában egy 2D FFT-t hajtunk végre egy 4x4-es komplex adathalmazon. Az adatok inicializálása és a GPU memóriába másolása után létrehozzuk az FFT tervet, végrehajtjuk az FFT-t, majd visszamásoljuk és kiíratjuk az eredményeket.

##### 3D FFT végrehajtása

A 3D FFT háromdimenziós adathalmazokon hajt végre Fourier-transzformációt. Ez különösen hasznos volumetrikus adatok esetén, például orvosi képfeldolgozásban vagy folyadékdinamikai szimulációkban.

**Példa**:

```cpp
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

int main() {
    // Adat méretei
    int Nx = 4;
    int Ny = 4;
    int Nz = 4;
    cufftComplex h_data[Nx * Ny * Nz];

    // Adatok inicializálása
    for (int i = 0; i < Nx * Ny * Nz; i++) {
        h_data[i].x = i;
        h_data[i].y = 0;
    }

    // Device adat
    cufftComplex* d_data;
    cudaMalloc(&d_data, Nx * Ny * Nz * sizeof(cufftComplex));
    cudaMemcpy(d_data, h_data, Nx * Ny * Nz * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    // FFT terv létrehozása
    cufftHandle plan;
    cufftPlan3d(&plan, Nx, Ny, Nz, CUFFT_C2C);

    // FFT végrehajtása
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

    // Eredmény másolása vissza a hostra
    cudaMemcpy(h_data, d_data, Nx * Ny * Nz * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // Eredmények kiíratása
    for (int i = 0; i < Nx * Ny * Nz; i++) {
        std::cout << "Element " << i << ": (" << h_data[i].x << ", " << h_data[i].y << ")" << std::endl;
    }

    // Memória felszabadítása
    cufftDestroy(plan);
    cudaFree(d_data);

    return 0;
}
```

Ebben a példában egy 3D FFT-t hajtunk végre egy 4x4x4-es komplex adathalmazon. Az adatok inicializálása és a GPU memóriába másolása után létrehozzuk az FFT tervet, végrehajtjuk az FFT-t, majd visszamásoljuk és kiíratjuk az eredményeket.

#### További funkciók és optimalizálás

A cuFFT könyvtár számos további funkciót és optimalizálási lehetőséget kínál, amelyek lehetővé teszik a fejlesztők számára, hogy még hatékonyabban használják a GPU-t. Például a cuFFT lehetőséget biztosít különböző adatelrendezések (layout) használatára, több GPU támogatására és aszinkron végrehajtásra.

**Példa** több GPU használatára:

```cpp
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

int main() {
    // Adat méretei
    int N = 8;
    cufftComplex h_data[N];

    // Adatok inicializálása
    for (int i = 0; i < N; i++) {
        h_data[i].x = i;
        h_data[i].y = 0;
    }

    // Device adat
    cufftComplex* d_data;
    cudaMalloc(&d_data, N * sizeof(cufftComplex));
    cudaMemcpy(d_data, h_data, N * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    // FFT terv létrehozása több GPU-ra
    cufftHandle plan;
    cufftCreate(&plan);
    cufftXtSetGPUs(plan, 2, {0, 1});
    cufftMakePlan1d(plan, N, CUFFT_C2C, 1, NULL);

    // FFT végrehajtása
    cufftXtExecDescriptorC2C(plan, d_data, d_data, CUFFT_FORWARD);

    // Eredmény másolása vissza a hostra
    cudaMemcpy(h_data, d_data, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // Eredmények kiíratása
    for (int i = 0; i < N; i++) {
        std::cout << "Element " << i << ": (" << h_data[i].x << ", " << h_data[i].y << ")" << std::endl;
    }

    // Memória felszabadítása
    cufftDestroy(plan);
    cudaFree(d_data);

    return 0;
}
```

Ebben a példában egy 1D FFT-t hajtunk végre több GPU használatával. Az FFT tervet úgy konfiguráljuk, hogy több GPU-t is bevonjon a számításba, majd végrehajtjuk az FFT-t és visszamásoljuk az eredményeket a host memóriába.

#### Összefoglalás

A cuFFT könyvtár egy hatékony eszköz a Fourier-transzformációk GPU-n történő végrehajtására. Az itt bemutatott példák segítségével láthattuk, hogyan lehet egyszerűen és hatékonyan végrehajtani 1D, 2D és 3D FFT műveleteket a cuFFT használatával. A cuFFT könyvtár lehetővé teszi, hogy a fejlesztők kihasználják a GPU párhuzamos feldolgozási képességeit, így jelentősen megnövelve a Fourier-transzformációk teljesítményét. A további funkciók és optimalizálási lehetőségek révén a cuFFT könyvtár még nagyobb rugalmasságot és hatékonyságot biztosít a fejlesztők számára.

### 11.4 cuDNN

A cuDNN (CUDA Deep Neural Network) könyvtár egy GPU-gyorsított primitív könyvtár, amely mélytanulási hálózatok implementálásához nyújt optimalizált rutinokat. A cuDNN-t az NVIDIA fejlesztette ki, hogy megkönnyítse a fejlesztők számára a hatékony és gyors mélytanulási algoritmusok implementálását a CUDA-kompatibilis GPU-kon. A könyvtár olyan alapvető mélytanulási műveleteket tartalmaz, mint például a konvolúciós, pooling és normalizációs rétegek, valamint az aktivációs függvények és a visszafelé irányuló gradiens számítások.

#### cuDNN alapjai

A cuDNN egy alacsony szintű API, amely lehetővé teszi a mélytanulási primitívek hatékony végrehajtását a GPU-n. A cuDNN használatának megkezdése előtt inicializálnunk kell a könyvtárat, és a munka befejeztével le kell zárnunk azt. Az alábbiakban bemutatjuk a cuDNN használatának alapvető lépéseit és néhány példakódot.

##### cuDNN inicializálása és lezárása

A cuDNN használatához először inicializálnunk kell a könyvtárat, amihez létre kell hoznunk egy cuDNN kezelőt. A munka befejeztével le kell zárnunk a könyvtárat, és meg kell semmisítenünk a kezelőt.

**Példa** a cuDNN inicializálására és lezárására:

```cpp
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>

int main() {
    // cuDNN könyvtár kezelő
    cudnnHandle_t cudnn;
    
    // cuDNN inicializálása
    cudnnStatus_t status = cudnnCreate(&cudnn);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cuDNN initialization failed: " << cudnnGetErrorString(status) << std::endl;
        return EXIT_FAILURE;
    }

    // ...

    // cuDNN lezárása
    status = cudnnDestroy(cudnn);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cuDNN shutdown failed: " << cudnnGetErrorString(status) << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```

##### Konvolúciós réteg

A konvolúciós réteg az egyik legfontosabb építőeleme a konvolúciós neurális hálózatoknak (CNN). A cuDNN optimalizált rutinokat biztosít a konvolúciós műveletek végrehajtásához, beleértve az előre- és visszafelé irányuló lépéseket is.

**Példa** konvolúciós réteg végrehajtására:

```cpp
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>

int main() {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Bemeneti adat mérete
    int batch_size = 1;
    int channels = 1;
    int height = 5;
    int width = 5;

    // Konvolúciós szűrő mérete
    int filter_count = 1;
    int filter_height = 3;
    int filter_width = 3;

    // Bemeneti és kimeneti adatok
    float input[batch_size * channels * height * width] = {0, 1, 2, 3, 4,
                                                          1, 2, 3, 4, 5,
                                                          2, 3, 4, 5, 6,
                                                          3, 4, 5, 6, 7,
                                                          4, 5, 6, 7, 8};
    float output[batch_size * filter_count * height * width];

    // Szűrő adatok
    float filter[filter_count * channels * filter_height * filter_width] = {1, 1, 1,
                                                                            1, 1, 1,
                                                                            1, 1, 1};

    // Device memória allokálása
    float *d_input, *d_output, *d_filter;
    cudaMalloc(&d_input, sizeof(input));
    cudaMalloc(&d_output, sizeof(output));
    cudaMalloc(&d_filter, sizeof(filter));
    cudaMemcpy(d_input, input, sizeof(input), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, sizeof(filter), cudaMemcpyHostToDevice);

    // Tensor deskriptorok létrehozása
    cudnnTensorDescriptor_t input_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnCreateFilterDescriptor(&filter_descriptor);
    cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, height, width);
    cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, filter_count, height, width);
    cudnnSetFilter4dDescriptor(filter_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filter_count, channels, filter_height, filter_width);

    // Konvolúciós deskriptor
    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnCreateConvolutionDescriptor(&convolution_descriptor);
    cudnnSetConvolution2dDescriptor(convolution_descriptor, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    // Konvolúciós előre lépés
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn, &alpha, input_descriptor, d_input, filter_descriptor, d_filter, convolution_descriptor, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, NULL, 0, &beta, output_descriptor, d_output);

    // Eredmény másolása vissza a hostra
    cudaMemcpy(output, d_output, sizeof(output), cudaMemcpyDeviceToHost);

    // Eredmények kiíratása
    for (int i = 0; i < batch_size * filter_count * height * width; i++) {
        std::cout << "Element " << i << ": " << output[i] << std::endl;
    }

    // Memória felszabadítása
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroy(cudnn);

    return 0;
}
```

Ebben a példában egy egyszerű konvolúciós réteget implementálunk a cuDNN segítségével. A bemeneti adatokat és a szűrőket inicializáljuk, majd ezeket átmásoljuk a GPU memóriába. Létrehozzuk a tensor deskriptorokat és a konvolúciós deskriptort, majd végrehajtjuk a konvolúciós műveletet. Az eredményeket visszamásoljuk a host memóriába, és kiíratjuk őket.

##### Pooling réteg

A pooling réteg egy másik alapvető eleme a konvolúciós neurális hálózatoknak, amely csökkenti a bemeneti adatok méretét, miközben megőrzi a legfontosabb információkat. A cuDNN támogatja a max pooling és az átlag pooling műveleteket is.

**Példa** max pooling réteg végrehajtására:

```cpp
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>

int main() {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Bemeneti adat mérete
    int batch_size = 1;
    int channels = 1;
    int height = 4;
    int width = 4;

    // Bemeneti és kimeneti adatok
    float input[batch_size * channels * height * width] = {1, 2, 3, 4,
                                                           5, 6, 7, 8,
                                                           9, 10, 11, 12,
                                                           13, 14, 15, 16};
    float output[batch_size * channels * (height / 2) * (width / 2)];

    // Device memória allokálása
    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(input));
    cudaMalloc(&d_output, sizeof(output));
    cudaMemcpy(d_input, input, sizeof(input), cudaMemcpyHostToDevice);

    // Tensor deskriptorok létrehozása
    cudnnTensorDescriptor_t input_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, height, width);
    cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, height / 2, width / 2);

    // Pooling deskriptor létrehozása
    cudnnPoolingDescriptor_t pooling_descriptor;
    cudnnCreatePoolingDescriptor(&pooling_descriptor);
    cudnnSetPooling2dDescriptor(pooling_descriptor, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2);

    // Pooling előre lépés
    float alpha = 1.0f, beta = 0.0f;
    cudnnPoolingForward(cudnn, pooling_descriptor, &alpha, input_descriptor, d_input, &beta, output_descriptor, d_output);

    // Eredmény másolása vissza a hostra
    cudaMemcpy(output, d_output, sizeof(output), cudaMemcpyDeviceToHost);

    // Eredmények kiíratása
    for (int i = 0; i < batch_size * channels * (height / 2) * (width / 2); i++) {
        std::cout << "Element " << i << ": " << output[i] << std::endl;
    }

    // Memória felszabadítása
    cudaFree(d_input);
    cudaFree(d_output);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyPoolingDescriptor(pooling_descriptor);
    cudnnDestroy(cudnn);

    return 0;
}
```

Ebben a példában egy max pooling réteget implementálunk a cuDNN segítségével. A bemeneti adatokat inicializáljuk és átmásoljuk a GPU memóriába, majd létrehozzuk a tensor és pooling deskriptorokat. Végrehajtjuk a pooling műveletet, majd az eredményeket visszamásoljuk a host memóriába, és kiíratjuk őket.

##### Aktivációs függvények

Az aktivációs függvények a neurális hálózatok fontos részei, amelyek nemlinearitást visznek a modellbe. A cuDNN támogatja a leggyakrabban használt aktivációs függvényeket, mint például a ReLU, tanh és sigmoid függvényeket.

**Példa** ReLU aktivációs függvény végrehajtására:

```cpp
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>

int main() {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Bemeneti adat mérete
    int batch_size = 1;
    int channels = 1;
    int height = 4;
    int width = 4;

    // Bemeneti és kimeneti adatok
    float input[batch_size * channels * height * width] = {1, -2, 3, -4,
                                                           5, -6, 7, -8,
                                                           9, -10, 11, -12,
                                                           13, -14, 15, -16};
    float output[batch_size * channels * height * width];

    // Device memória allokálása
    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(input));
    cudaMalloc(&d_output, sizeof(output));
    cudaMemcpy(d_input, input, sizeof(input), cudaMemcpyHostToDevice);

    // Tensor deskriptorok létrehozása
    cudnnTensorDescriptor_t tensor_descriptor;
    cudnnCreateTensorDescriptor(&tensor_descriptor);
    cudnnSetTensor4dDescriptor(tensor_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, height, width);

    // Aktivációs deskriptor létrehozása
    cudnnActivationDescriptor_t activation_descriptor;
    cudnnCreateActivationDescriptor(&activation_descriptor);
    cudnnSetActivationDescriptor(activation_descriptor, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);

    // Aktivációs függvény végrehajtása
    float alpha = 1.0f, beta = 0.0f;
    cudnnActivationForward(cudnn, activation_descriptor, &alpha, tensor_descriptor, d_input, &beta, tensor_descriptor, d_output);

    // Eredmény másolása vissza a hostra
    cudaMemcpy(output, d_output, sizeof(output), cudaMemcpyDeviceToHost);

    // Eredmények kiíratása
    for (int i = 0; i < batch_size * channels * height * width; i++) {
        std::cout << "Element " << i << ": " << output[i] << std::endl;
    }

    // Memória felszabadítása
    cudaFree(d_input);
    cudaFree(d_output);
    cudnnDestroyTensorDescriptor(tensor_descriptor);
    cudnnDestroyActivationDescriptor(activation_descriptor);
    cudnnDestroy(cudnn);

    return 0;
}
```

Ebben a példában egy ReLU aktivációs függvényt implementálunk a cuDNN segítségével. A bemeneti adatokat inicializáljuk és átmásoljuk a GPU memóriába, majd létrehozzuk a tensor és aktivációs deszkriptorokat. Végrehajtjuk az aktivációs műveletet, majd az eredményeket visszamásoljuk a host memóriába, és kiíratjuk őket.

##### Visszafelé irányuló gradiens számítás

A visszafelé irányuló gradiens számítás a mélytanulási hálózatok tanításának kulcsfontosságú lépése. A cuDNN optimalizált rutinokat biztosít a visszafelé irányuló gradiens számítások végrehajtásához is.

**Példa** visszafelé irányuló konvolúciós gradiens számításra:

```cpp
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>

int main() {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Bemeneti adat mérete
    int batch_size = 1;
    int channels = 1;
    int height = 5;
    int width = 5;

    // Konvolúciós szűrő mérete
    int filter_count = 1;
    int filter_height = 3;
    int filter_width = 3;

    // Bemeneti és kimeneti adatok
    float input[batch_size * channels * height * width] = {0, 1, 2, 3, 4,
                                                          1, 2, 3, 4, 5,
                                                          2, 3, 4, 5, 6,
                                                          3, 4, 5, 6, 7,
                                                          4, 5, 6, 7, 8};
    float grad_output[batch_size * filter_count * height * width] = {1, 1, 1, 1, 1,
                                                                    1, 1, 1, 1, 1,
                                                                    1, 1, 1, 1, 1,
                                                                    1, 1, 1, 1, 1,
                                                                    1, 1, 1, 1, 1};
    float grad_input[batch_size * channels * height * width];

    // Szűrő adatok
    float filter[filter_count * channels * filter_height * filter_width] = {1, 1, 1,
                                                                            1, 1, 1,
                                                                            1, 1, 1};

    // Device memória allokálása
    float *d_input, *d_grad_output, *d_grad_input, *d_filter;
    cudaMalloc(&d_input, sizeof(input));
    cudaMalloc(&d_grad_output, sizeof(grad_output));
    cudaMalloc(&d_grad_input, sizeof(grad_input));
    cudaMalloc(&d_filter, sizeof(filter));
    cudaMemcpy(d_input, input, sizeof(input), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_output, grad_output, sizeof(grad_output), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, sizeof(filter), cudaMemcpyHostToDevice);

    // Tensor deskriptorok létrehozása
    cudnnTensorDescriptor_t input_descriptor;
    cudnnTensorDescriptor_t grad_output_descriptor;
    cudnnTensorDescriptor_t grad_input_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnCreateTensorDescriptor(&grad_output_descriptor);
    cudnnCreateTensorDescriptor(&grad_input_descriptor);
    cudnnCreateFilterDescriptor(&filter_descriptor);
    cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, height, width);
    cudnnSetTensor4dDescriptor(grad_output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, filter_count, height, width);
    cudnnSetTensor4dDescriptor(grad_input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, height, width);
    cudnnSetFilter4dDescriptor(filter_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filter_count, channels, filter_height, filter_width);

    // Konvolúciós deskriptor
    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnCreateConvolutionDescriptor(&convolution_descriptor);
    cudnnSetConvolution2dDescriptor(convolution_descriptor, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    // Visszafelé irányuló konvolúciós gradiens számítás
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionBackwardData(cudnn, &alpha, filter_descriptor, d_filter, grad_output_descriptor, d_grad_output, convolution_descriptor, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, NULL, 0, &beta, grad_input_descriptor, d_grad_input);

    // Eredmény másolása vissza a hostra
    cudaMemcpy(grad_input, d_grad_input, sizeof(grad_input), cudaMemcpyDeviceToHost);

    // Eredmények kiíratása
    for (int i = 0; i < batch_size * channels * height * width; i++) {
        std::cout << "Element " << i << ": " << grad_input[i] << std::endl;
    }

    // Memória felszabadítása
    cudaFree(d_input);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    cudaFree(d_filter);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(grad_output_descriptor);
    cudnnDestroyTensorDescriptor(grad_input_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroy(cudnn);

    return 0;
}
```

Ebben a példában egy egyszerű visszafelé irányuló konvolúciós gradiens számítást hajtunk végre a cuDNN segítségével. A bemeneti adatokat, a szűrőket és a kimeneti gradienst inicializáljuk, majd ezeket átmásoljuk a GPU memóriába. Létrehozzuk a tensor és konvolúciós deszkriptorokat, majd végrehajtjuk a visszafelé irányuló konvolúciós műveletet. Az eredményeket visszamásoljuk a host memóriába, és kiíratjuk őket.

#### Összefoglalás

A cuDNN könyvtár egy rendkívül hatékony eszköz a mélytanulási algoritmusok GPU-n történő végrehajtásához. Az itt bemutatott példák segítségével láthattuk, hogyan lehet egyszerűen és hatékonyan végrehajtani különféle mélytanulási műveleteket, mint például a konvolúció, pooling, aktivációs függvények és visszafelé irányuló gradiens számítás a cuDNN használatával. A cuDNN könyvtár lehetővé teszi, hogy a fejlesztők kihasználják a GPU párhuzamos feldolgozási képességeit, így jelentősen megnövelve a mélytanulási modellek teljesítményét és hatékonyságát.

