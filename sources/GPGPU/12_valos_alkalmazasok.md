\newpage
## 12. Valós Alkalmazások és Példák

A grafikus feldolgozó egységek (GPU-k) átalakították a számítástechnika számos területét, lehetővé téve az összetett és nagy számításigényű feladatok gyorsabb megoldását. Ebben a fejezetben bemutatjuk, hogyan használhatóak a GPU-k különféle valós alkalmazásokban és példákon keresztül, kihasználva a párhuzamos feldolgozás előnyeit. Elsőként a numerikus számítások világába kalandozunk, ahol a differenciálegyenletek megoldása révén demonstráljuk a GPU-k képességeit. Ezt követően a képfeldolgozás területén mutatjuk be, hogyan valósíthatóak meg a széldetektálás és konvolúciós műveletek GPU-n. A gépi tanulás robbanásszerű fejlődése során a neurális hálózatok GPU-val történő gyorsításának jelentőségét vizsgáljuk. Végül a valós idejű renderelés technikáit tárgyaljuk, különös tekintettel a ray tracing módszerekre, amelyek forradalmasították a számítógépes grafikát. Ezen példák révén bepillantást nyerhetünk abba, hogyan teszik lehetővé a GPU-k a hatékonyság és teljesítmény növelését a különböző tudományos és ipari alkalmazásokban.

### 12.1 Numerikus számítások

A numerikus számítások a matematika olyan területe, amely numerikus módszereket alkalmaz a különféle matematikai problémák megoldására. Ezek a módszerek gyakran nagy mennyiségű számítást igényelnek, ami a hagyományos CPU-k esetében időigényes lehet. A GPU-k azonban, párhuzamos feldolgozási képességeik révén, jelentős teljesítménybeli előnyt kínálnak, különösen a nagyméretű, párhuzamosan végrehajtható feladatok esetében. Ebben a fejezetben a differenciálegyenletek megoldására összpontosítunk, bemutatva, hogyan lehet a GPU-kat hatékonyan alkalmazni ezen a területen.

#### Differenciálegyenletek megoldása GPU-n

A differenciálegyenletek a matematika és a fizika számos területén alapvető fontosságúak. Ezek az egyenletek leírják a változó mennyiségek közötti kapcsolatokat és dinamikákat. Az ilyen egyenletek megoldása gyakran nagy számítási igényt jelent, különösen akkor, ha a megoldásokat nagy felbontású hálón vagy hosszú időintervallumra kell kiszámítani. A GPU-k párhuzamos feldolgozási képességei lehetővé teszik ezen számítások jelentős gyorsítását.

##### Egyszerű példa: Euler-módszer

Az Euler-módszer egy egyszerű és alapvető numerikus módszer az elsőrendű differenciálegyenletek megoldására. Az alábbiakban bemutatjuk, hogyan lehet ezt a módszert implementálni GPU-n CUDA használatával.

```cpp
#include <stdio.h>

#include <cuda.h>

__global__ void euler_step(float *y, float *dydx, float dt, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = y[idx] + dt * dydx[idx];
    }
}

int main() {
    const int N = 1000;
    const float dt = 0.01;
    float *y, *dydx;
    float *d_y, *d_dydx;

    y = (float*)malloc(N * sizeof(float));
    dydx = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        y[i] = 1.0f;
        dydx[i] = -0.5f * y[i];
    }

    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_dydx, N * sizeof(float));

    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dydx, dydx, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    euler_step<<<numBlocks, blockSize>>>(d_y, d_dydx, dt, N);

    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_y);
    cudaFree(d_dydx);
    free(y);
    free(dydx);

    return 0;
}
```

Ebben a példában egy egyszerű differenciálegyenletet oldunk meg az Euler-módszerrel. Az `euler_step` kernel párhuzamosan frissíti az y értékeit a GPU-n, lehetővé téve a nagy számítási teljesítményt.

##### Haladó példa: Runge-Kutta módszer

A Runge-Kutta módszerek a numerikus integrálás pontosabb módszerei közé tartoznak. Az alábbi példa bemutatja a negyedrendű Runge-Kutta módszer (RK4) implementálását CUDA-ban.

```cpp
#include <stdio.h>

#include <cuda.h>

__device__ float dydx(float y) {
    return -0.5f * y;
}

__global__ void runge_kutta_step(float *y, float dt, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float k1 = dt * dydx(y[idx]);
        float k2 = dt * dydx(y[idx] + 0.5f * k1);
        float k3 = dt * dydx(y[idx] + 0.5f * k2);
        float k4 = dt * dydx(y[idx] + k3);
        y[idx] = y[idx] + (k1 + 2.0f * k2 + 2.0f * k3 + k4) / 6.0f;
    }
}

int main() {
    const int N = 1000;
    const float dt = 0.01;
    float *y;
    float *d_y;

    y = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        y[i] = 1.0f;
    }

    cudaMalloc(&d_y, N * sizeof(float));
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    runge_kutta_step<<<numBlocks, blockSize>>>(d_y, dt, N);

    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_y);
    free(y);

    return 0;
}
```

Ebben a példában a `runge_kutta_step` kernel párhuzamosan számítja ki a Runge-Kutta lépéseket, lehetővé téve a differenciálegyenletek pontosabb és gyorsabb megoldását GPU-n.

##### Példa: Hővezetési egyenlet megoldása

A parciális differenciálegyenletek (PDE) megoldása még nagyobb számítási igényt támaszt, amit a GPU-k kiválóan kezelnek. Az alábbi példa bemutatja, hogyan lehet egy egyszerű hővezetési egyenletet megoldani CUDA-val.

```cpp
#include <stdio.h>

#include <cuda.h>

__global__ void heat_equation_step(float *u, float *u_new, int N, float alpha, float dt, float dx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx < N-1) {
        u_new[idx] = u[idx] + alpha * dt / (dx * dx) * (u[idx-1] - 2.0f * u[idx] + u[idx+1]);
    }
}

int main() {
    const int N = 1000;
    const float alpha = 0.01;
    const float dt = 0.1;
    const float dx = 0.1;
    float *u, *u_new;
    float *d_u, *d_u_new;

    u = (float*)malloc(N * sizeof(float));
    u_new = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        u[i] = sinf(i * dx);
    }

    cudaMalloc(&d_u, N * sizeof(float));
    cudaMalloc(&d_u_new, N * sizeof(float));

    cudaMemcpy(d_u, u, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    for (int t = 0; t < 1000; t++) {
        heat_equation_step<<<numBlocks, blockSize>>>(d_u, d_u_new, N, alpha, dt, dx);
        cudaMemcpy(d_u, d_u_new, N * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(u_new, d_u_new, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_u);
    cudaFree(d_u_new);
    free(u);
    free(u_new);

    return 0;
}
```

Ebben a példában egy egyszerű hővezetési egyenletet oldunk meg, amely egy dimenziós térbeli hőmérsékleteloszlást ír le időben. A `heat_equation_step` kernel kiszámítja az új hőmérsékleti értékeket minden időlépésben, párhuzamosan végrehajtva a számításokat a GPU-n.

#### Összegzés

A numerikus számítások területén a GPU-k hatalmas előnyt kínálnak, különösen a nagy számítási igényű feladatok esetében, mint amilyen a differenciálegyenletek megoldása. Az egyszerű Euler-módszertől a bonyolultabb Runge-Kutta módszerekig és a parciális

differenciálegyenletekig a GPU-k párhuzamos feldolgozási képességei lehetővé teszik a számítások jelentős gyorsítását és hatékonyságának növelését. Ezek az előnyök számos tudományos és mérnöki alkalmazásban nyújtanak segítséget, ahol a pontos és gyors numerikus megoldások elengedhetetlenek.

### 12.2 Képfeldolgozás

A képfeldolgozás a digitális képek elemzésének és manipulációjának tudománya, amely számos alkalmazási területen fontos szerepet játszik, beleértve az orvosi képfeldolgozást, a gépi látást, a számítógépes grafikát és még sok mást. A GPU-k jelentős előnyökkel járnak ezen a területen is, mivel a képfeldolgozási feladatok gyakran nagy mennyiségű adatot igényelnek és párhuzamosan végezhetők. Ebben a fejezetben bemutatjuk, hogyan használhatók a GPU-k különböző képfeldolgozási műveletekhez, például a széldetektáláshoz és a konvolúciós műveletekhez.

#### Széldetektálás

A széldetektálás a képfeldolgozás egyik alapvető művelete, amely a képen található élek, azaz a hirtelen intenzitásváltozások detektálását jelenti. Számos algoritmus létezik a széldetektálásra, de az egyik legismertebb a Sobel-operator. A GPU-k lehetővé teszik ezen műveletek hatékony párhuzamos végrehajtását.

##### Sobel-operátor CUDA implementációja

A Sobel-operátor két 3x3-as konvolúciós mátrixot alkalmaz a kép x és y irányú gradiensének meghatározásához. Az alábbi kód bemutatja, hogyan lehet a Sobel-operatorokat implementálni CUDA-ban.

```cpp
#include <stdio.h>

#include <cuda.h>

__global__ void sobel_filter(unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int gx = -input[(y - 1) * width + (x - 1)] - 2 * input[y * width + (x - 1)] - input[(y + 1) * width + (x - 1)] +
                  input[(y - 1) * width + (x + 1)] + 2 * input[y * width + (x + 1)] + input[(y + 1) * width + (x + 1)];
        int gy = -input[(y - 1) * width + (x - 1)] - 2 * input[(y - 1) * width + x] - input[(y - 1) * width + (x + 1)] +
                  input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)];
        int magnitude = min(255, (int)sqrtf(gx * gx + gy * gy));
        output[y * width + x] = magnitude;
    }
}

int main() {
    int width = 1024;
    int height = 768;
    unsigned char *input = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char *output = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char *d_input, *d_output;

    // Fill input with some data
    for (int i = 0; i < width * height; i++) {
        input[i] = rand() % 256;
    }

    cudaMalloc(&d_input, width * height * sizeof(unsigned char));
    cudaMalloc(&d_output, width * height * sizeof(unsigned char));
    cudaMemcpy(d_input, input, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    sobel_filter<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    cudaMemcpy(output, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(output);

    return 0;
}
```

Ebben a kódban a `sobel_filter` kernel kiszámítja az egyes pixelek gradiensét, majd meghatározza az élek intenzitását. A CUDA párhuzamos feldolgozási képességei révén ez a művelet jelentősen felgyorsítható.

#### Konvolúciós műveletek

A konvolúciós műveletek alapvető fontosságúak a képfeldolgozásban, különösen a különféle szűrési technikák és a mély neurális hálózatok esetében. A konvolúció során egy képet egy kernel mátrixszal transzformálunk, amely lehetővé teszi különböző jellemzők kiemelését vagy elnyomását.

##### Egyszerű konvolúció CUDA implementációja

Az alábbi példa bemutatja, hogyan lehet egy egyszerű konvolúciós műveletet implementálni CUDA-ban.

```cpp
#include <stdio.h>

#include <cuda.h>

#define MASK_WIDTH 3

#define MASK_RADIUS (MASK_WIDTH / 2)

__global__ void convolution_2d(unsigned char *input, unsigned char *output, int width, int height, float *mask) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= MASK_RADIUS && x < width - MASK_RADIUS && y >= MASK_RADIUS && y < height - MASK_RADIUS) {
        float sum = 0.0f;
        for (int ky = -MASK_RADIUS; ky <= MASK_RADIUS; ky++) {
            for (int kx = -MASK_RADIUS; kx <= MASK_RADIUS; kx++) {
                int ix = x + kx;
                int iy = y + ky;
                sum += input[iy * width + ix] * mask[(ky + MASK_RADIUS) * MASK_WIDTH + (kx + MASK_RADIUS)];
            }
        }
        output[y * width + x] = min(max((int)sum, 0), 255);
    }
}

int main() {
    int width = 1024;
    int height = 768;
    unsigned char *input = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char *output = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char *d_input, *d_output;
    float h_mask[MASK_WIDTH * MASK_WIDTH] = {
        1, 2, 1,
        2, 4, 2,
        1, 2, 1
    };
    float *d_mask;

    // Normalize mask
    float mask_sum = 0.0f;
    for (int i = 0; i < MASK_WIDTH * MASK_WIDTH; i++) {
        mask_sum += h_mask[i];
    }
    for (int i = 0; i < MASK_WIDTH * MASK_WIDTH; i++) {
        h_mask[i] /= mask_sum;
    }

    // Fill input with some data
    for (int i = 0; i < width * height; i++) {
        input[i] = rand() % 256;
    }

    cudaMalloc(&d_input, width * height * sizeof(unsigned char));
    cudaMalloc(&d_output, width * height * sizeof(unsigned char));
    cudaMalloc(&d_mask, MASK_WIDTH * MASK_WIDTH * sizeof(float));
    cudaMemcpy(d_input, input, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, MASK_WIDTH * MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    convolution_2d<<<gridSize, blockSize>>>(d_input, d_output, width, height, d_mask);

    cudaMemcpy(output, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
    free(input);
    free(output);

    return 0;
}
```

Ebben a kódban a `convolution_2d` kernel egy 3x3-as Gaussian kernel segítségével végzi el a képen a konvolúciós műveletet. A GPU párhuzamos számítási képességei lehetővé teszik a konvolúciós műveletek gyors és hatékony végrehajtását.

#### Képfeldolgozás neurális hálózatokkal

A konvolúciós neurális hálózatok (CNN-ek) a gépi látásban és képfeldolgozásban elért legjelentősebb előrelépések közé tartoznak. A CNN-ek több konvolúciós és pooling réteg segítségével képesek a képek jellemzőit kiemelni és osztályozni. A GPU-k különösen jól teljesítenek ezen a területen, mivel a CNN-ek jelentős mennyiségű mátrixszorzást igényelnek, ami párhuzamosan végezhető.

##### CNN-ek gyorsítása GPU-n

Az alábbi példa bemutatja, hogyan lehet a TensorFlow-t és CUDA-t használni egy egyszerű CNN implementálására és gyorsítására GPU-n.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the CNN model

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# Compile the model

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model

with tf.device('/GPU:0'):
    model.fit(train_images, train_labels, epochs=10, 
              validation_data=(test_images, test_labels))
```

Ebben a példában egy egyszerű CNN modellt definiálunk és tanítunk a CIFAR-10 adatbázison. A `tf.device('/GPU:0')` parancs biztosítja, hogy a számítások a GPU-n fussanak, ami jelentősen felgyorsítja a tanítási folyamatot.

#### Összegzés

A GPU-k hatalmas potenciált kínálnak a képfeldolgozás területén, lehetővé téve az összetett és nagy adatintenzitású műveletek gyors végrehajtását. A széldetektálás, a konvolúciós műveletek és a neurális hálózatok GPU-ra optimalizálása mind jelentős teljesítménybeli előnyöket nyújtanak. A bemutatott példák és kódok szemléltetik, hogyan lehet hatékonyan alkalmazni a GPU-kat a különféle képfeldolgozási feladatokhoz, kihasználva a párhuzamos feldolgozás előnyeit.

### 12.3 Gépi tanulás

A gépi tanulás a mesterséges intelligencia egy olyan területe, amely algoritmusokat és statisztikai modelleket használ, hogy a számítógépek adatokból tanuljanak és döntéseket hozzanak. Az utóbbi években a gépi tanulás területén óriási előrelépések történtek, részben a GPU-k párhuzamos feldolgozási képességeinek köszönhetően. A GPU-k lehetővé teszik a nagy számítási igényű algoritmusok, például a neurális hálózatok gyors és hatékony végrehajtását. Ebben a fejezetben bemutatjuk, hogyan használhatók a GPU-k a gépi tanulási feladatokhoz, különös tekintettel a neurális hálózatok gyorsítására.

#### Neurális hálózatok GPU gyorsítása

A neurális hálózatok a gépi tanulás egyik legfontosabb eszközei, amelyek különböző rétegekből állnak, és képesek komplex mintázatok felismerésére és predikciókra. A GPU-k párhuzamos feldolgozási képességei révén jelentősen felgyorsíthatók a neurális hálózatok betanítása és alkalmazása.

##### Egyszerű példák: TensorFlow és Keras

A TensorFlow és a Keras két népszerű gépi tanulási keretrendszer, amelyek könnyen használhatók és támogatják a GPU gyorsítást. Az alábbiakban bemutatjuk, hogyan lehet egy egyszerű neurális hálózatot betanítani a TensorFlow és a Keras használatával.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Define the neural network model

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model using GPU

with tf.device('/GPU:0'):
    model.fit(train_images, train_labels, epochs=5, 
              validation_data=(test_images, test_labels))
```

Ebben a példában egy egyszerű konvolúciós neurális hálózatot (CNN) definiálunk és tanítunk az MNIST adatbázison. A `tf.device('/GPU:0')` parancs biztosítja, hogy a számítások a GPU-n fussanak, ami jelentősen felgyorsítja a tanítási folyamatot.

#### Haladó példák: PyTorch

A PyTorch egy másik népszerű gépi tanulási keretrendszer, amelyet sok kutató és fejlesztő használ. Az alábbi példa bemutatja, hogyan lehet egy neurális hálózatot betanítani a PyTorch használatával.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Load and preprocess data

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
testset = datasets.MNIST('.', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Define the neural network model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# Initialize the model, loss function, and optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model

epochs = 5
for epoch in range(epochs):
    model.train()
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Test the model

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy: {100 * correct / total}%')
```

Ebben a példában egy egyszerű CNN modellt definiálunk és tanítunk az MNIST adatbázison a PyTorch használatával. A `torch.device("cuda" if torch.cuda.is_available() else "cpu")` parancs biztosítja, hogy a számítások a GPU-n fussanak, ha az elérhető.

#### Mély neurális hálózatok és GPU gyorsítás

A mély neurális hálózatok (DNN-ek) több rétegből álló neurális hálózatok, amelyek képesek komplex mintázatok felismerésére. A DNN-ek betanítása gyakran rendkívül számításigényes, különösen nagy adatbázisok esetében. A GPU-k párhuzamos számítási képességei jelentősen felgyorsítják a betanítási folyamatot.

##### Transfer Learning

A transfer learning egy olyan technika, amely során egy előre betanított neurális hálózatot használunk új feladatok megoldására. Ez a megközelítés különösen hasznos, mivel lehetővé teszi, hogy kevesebb adat és számítási idő mellett is jó eredményeket érjünk el.

Az alábbi példa bemutatja, hogyan lehet a transfer learninget alkalmazni a ResNet50 hálózattal a TensorFlow használatával.

```python
import tensorflow as tf
from tensorflow.keras import layers, models, applications

# Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Load the pre-trained ResNet50 model

base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the base model

base_model.trainable = False

# Add custom layers on top

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model using GPU

with tf.device('/GPU:0'):
    model.fit(train_images, train_labels, epochs=5, 
              validation_data=(test_images, test_labels))
```

Ebben a példában a ResNet50 előre betanított modellt használjuk kiindulási pontként, és hozzáadunk néhány új réteget az új feladat megoldására. A `tf.device('/GPU:0')` parancs biztosítja, hogy a számítások a GPU-n fussanak.

#### Összegzés

A GPU-k hatalmas előnyt kínálnak a gépi tanulás területén, különösen a neurális hálózatok betanításában és alkalmazásában. A bemutatott példák és kódok szemléltetik, hogyan lehet hatékonyan kihasználni a GPU-k párhuzamos feldolgozási képességeit a TensorFlow, Keras és PyTorch használatával. Ezek a technikák jelentősen felgyorsítják a gépi tanulási feladatokat, lehetővé téve a nagyobb modellek és nagyobb adatkészletek kezelését. A transfer learning technikák alkalmazása tovább növeli a hatékonyságot, mivel lehetővé teszik az előre betanított modellek új feladatokra történő gyors alkalmazását.

### 12.4 Valós idejű renderelés

A valós idejű renderelés a számítógépes grafikában a valós idejű képalkotást jelenti, amelyet leggyakrabban a videojátékokban, szimulációkban és interaktív alkalmazásokban használnak. A valós idejű renderelés lehetővé teszi a felhasználó számára, hogy valós időben interakcióba lépjen a virtuális környezettel, miközben a képernyőn látható kép folyamatosan frissül és reagál a felhasználói bemenetekre. A GPU-k alapvető fontosságúak ebben a folyamatban, mivel párhuzamos feldolgozási képességeik révén képesek nagy mennyiségű adatot gyorsan és hatékonyan feldolgozni. Ebben a fejezetben a ray tracing technikákra összpontosítunk, bemutatva, hogyan használhatók a GPU-k a valós idejű renderelés felgyorsítására.

#### Ray Tracing

A ray tracing egy olyan technika, amely a fény sugarainak követésével hoz létre fotorealisztikus képeket. A sugárkövetés kiszámítja, hogy a fény hogyan halad a jeleneten keresztül, hogyan verődik vissza, törik meg, vagy nyelődik el a különböző felületeken. Ez a módszer rendkívül számításigényes, de a modern GPU-k lehetővé teszik a valós idejű ray tracing alkalmazását.

##### Alapvető Ray Tracing CUDA-ban

Az alábbi példa bemutatja, hogyan lehet egy egyszerű ray tracing algoritmust implementálni CUDA használatával.

```cpp
#include <stdio.h>

#include <curand_kernel.h>

struct Vec3 {
    float x, y, z;

    __device__ Vec3 operator+(const Vec3 &b) const {
        return Vec3{x + b.x, y + b.y, z + b.z};
    }

    __device__ Vec3 operator-(const Vec3 &b) const {
        return Vec3{x - b.x, y - b.y, z - b.z};
    }

    __device__ Vec3 operator*(float b) const {
        return Vec3{x * b, y * b, z * b};
    }

    __device__ Vec3& operator+=(const Vec3 &b) {
        x += b.x;
        y += b.y;
        z += b.z;
        return *this;
    }

    __device__ Vec3 normalize() const {
        float length = sqrtf(x * x + y * y + z * z);
        return Vec3{x / length, y / length, z / length};
    }
};

__device__ Vec3 ray_color(const Vec3& ray_dir) {
    Vec3 unit_direction = ray_dir.normalize();
    float t = 0.5f * (unit_direction.y + 1.0f);
    return Vec3{1.0f, 1.0f, 1.0f} * (1.0f - t) + Vec3{0.5f, 0.7f, 1.0f} * t;
}

__global__ void render(Vec3 *framebuffer, int width, int height, float aspect_ratio) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float u = float(x) / (width - 1);
    float v = float(y) / (height - 1);

    Vec3 ray_origin = {0.0f, 0.0f, 0.0f};
    Vec3 ray_direction = {2.0f * u - 1.0f, 2.0f * v - 1.0f / aspect_ratio, -1.0f};

    Vec3 color = ray_color(ray_direction);
    framebuffer[y * width + x] = color;
}

int main() {
    const int width = 800;
    const int height = 600;
    const float aspect_ratio = float(width) / float(height);
    Vec3 *framebuffer;

    cudaMallocManaged(&framebuffer, width * height * sizeof(Vec3));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    render<<<gridSize, blockSize>>>(framebuffer, width, height, aspect_ratio);
    cudaDeviceSynchronize();

    // Save framebuffer to image (omitted for brevity)

    cudaFree(framebuffer);

    return 0;
}
```

Ebben a kódban egy egyszerű ray tracing algoritmust valósítunk meg, amely egy kétdimenziós rácson keresztül iterál, és kiszámítja a sugár irányát minden pixelhez. Az `ray_color` függvény kiszámítja a sugár színét az égbolt egyszerű színátmenetének szimulálásával.

#### Haladó Ray Tracing

A ray tracing továbbfejleszthető azáltal, hogy további funkciókat, például objektumok közötti ütközéseket, árnyékolást, tükröződést és fénytörést adunk hozzá. Az alábbi példa bemutatja, hogyan lehet egy egyszerű gömbbel való ütközést hozzáadni a ray tracing algoritmushoz.

```cpp
__device__ bool hit_sphere(const Vec3& center, float radius, const Vec3& ray_origin, const Vec3& ray_dir, float &t) {
    Vec3 oc = ray_origin - center;
    float a = dot(ray_dir, ray_dir);
    float b = 2.0f * dot(oc, ray_dir);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) {
        return false;
    } else {
        t = (-b - sqrtf(discriminant)) / (2.0f * a);
        return true;
    }
}

__device__ Vec3 ray_color(const Vec3& ray_origin, const Vec3& ray_dir) {
    float t;
    if (hit_sphere(Vec3{0, 0, -1}, 0.5, ray_origin, ray_dir, t)) {
        Vec3 N = (ray_origin + ray_dir * t - Vec3{0, 0, -1}).normalize();
        return 0.5f * Vec3{N.x + 1, N.y + 1, N.z + 1};
    }
    Vec3 unit_direction = ray_dir.normalize();
    t = 0.5f * (unit_direction.y + 1.0f);
    return Vec3{1.0f, 1.0f, 1.0f} * (1.0f - t) + Vec3{0.5f, 0.7f, 1.0f} * t;
}

__global__ void render(Vec3 *framebuffer, int width, int height, float aspect_ratio) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float u = float(x) / (width - 1);
    float v = float(y) / (height - 1);

    Vec3 ray_origin = {0.0f, 0.0f, 0.0f};
    Vec3 ray_direction = {2.0f * u - 1.0f, 2.0f * v - 1.0f / aspect_ratio, -1.0f};

    Vec3 color = ray_color(ray_origin, ray_direction);
    framebuffer[y * width + x] = color;
}

int main() {
    const int width = 800;
    const int height = 600;
    const float aspect_ratio = float(width) / float(height);
    Vec3 *framebuffer;

    cudaMallocManaged(&framebuffer, width * height * sizeof(Vec3));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    render<<<gridSize, blockSize>>>(framebuffer, width, height, aspect_ratio);
    cudaDeviceSynchronize();

    // Save framebuffer to image (omitted for brevity)

    cudaFree(framebuffer);

    return 0;
}
```

Ebben a példában egy gömböt adunk a jelenethez, és kiszámítjuk a gömbbel való ütközést. Ha egy sugár eltalálja a gömböt, a gömb normálvektora alapján kiszámítjuk a színét.

#### Valós idejű Ray Tracing a Vulkan API-val

A modern grafikus API-k, mint a Vulkan, támogatják a valós idejű ray tracing-et hardveres gyorsítással. Az alábbi példa bemutatja, hogyan lehet a Vulkan-t használni a valós idejű ray tracing megvalósítására.

```cpp
#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include <vector>

// Vulkan setup omitted for brevity

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
};

std::vector<Vertex> vertices = {
    {{1.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},
    {{-1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},
    {{-1.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
    {{1.0f, -1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}}
};

std::vector<uint16_t> indices = {
    0, 1, 2, 2, 3, 0
};

void createVertexBuffer() {
    // Vulkan buffer creation code omitted for brevity
}

void createIndexBuffer() {
    // Vulkan buffer creation code omitted for brevity
}

void createRayTracingPipeline() {
    // Ray tracing pipeline creation code omitted for brevity
}

void mainLoop() {
    // Main rendering loop omitted for brevity
}

int main() {
    initVulkan();
    createVertexBuffer();
    createIndexBuffer();
    createRayTracingPipeline();
    mainLoop();
    cleanup();
    return 0;
}
```

A fenti kódrészlet egy alapvető vázlatot ad a Vulkan API-val történő ray tracing implementálásához. A Vulkan Ray Tracing kiterjesztéseket használva lehetővé válik a ray tracing pipeline létrehozása és a ray tracing shader-ek futtatása.

#### Összegzés

A valós idejű renderelés a számítógépes grafika egyik legfontosabb területe, ahol a GPU-k párhuzamos feldolgozási képességei kulcsszerepet játszanak. A ray tracing technikák lehetővé teszik a fotorealisztikus képek valós idejű létrehozását, amelyeket a modern GPU-k teljesítményének köszönhetően valós időben is élvezhetünk. Az egyszerű CUDA implementációktól a fejlett Vulkan API-val történő ray tracing-ig számos technika áll rendelkezésre a valós idejű renderelés hatékony megvalósítására. A bemutatott példák és kódok segítenek megérteni, hogyan lehet ezeket a technikákat alkalmazni a gyakorlatban, és hogyan lehet a GPU-k teljesítményét maximálisan kihasználni a valós idejű renderelés során.

