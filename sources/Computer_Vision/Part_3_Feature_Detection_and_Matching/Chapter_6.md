
\newpage

# Part III: Feature Detection and Matching

\newpage
## Chapter 6: Edge Detection

Edge detection is a fundamental technique in image processing and computer vision, used to identify significant transitions in intensity within an image. These transitions, or edges, represent object boundaries, surface markings, and other critical features. Detecting edges accurately is essential for tasks such as image segmentation, object recognition, and scene understanding. This chapter explores various methods for edge detection, ranging from basic gradient-based techniques to advanced algorithms.

**Subchapters:**
- **Gradient-Based Methods (Sobel, Prewitt):** Techniques that use gradient approximations to identify edges.
- **Canny Edge Detector:** A multi-stage edge detection algorithm known for its accuracy and reliability.
- **Advanced Edge Detection Techniques:** Modern methods that improve upon traditional techniques for more robust edge detection.

### 6.1. Gradient-Based Methods (Sobel, Prewitt)

Gradient-based methods are fundamental techniques in edge detection that utilize the concept of gradients to identify edges within an image. These methods detect edges by looking for significant changes in intensity values. Two of the most commonly used gradient-based edge detection methods are the Sobel and Prewitt operators. This subchapter delves into the mathematical background of these operators and demonstrates their practical implementation using OpenCV in C++.

#### 6.1.1. Mathematical Background

**Gradient and Edge Detection**

The gradient of an image is a vector that points in the direction of the greatest rate of increase in intensity. It is computed by taking the partial derivatives of the image intensity function with respect to the spatial coordinates $x$ and $y$. The gradient magnitude and direction are given by:

$$
G = \sqrt{G_x^2 + G_y^2}
$$
$$
\theta = \tan^{-1}\left(\frac{G_y}{G_x}\right)
$$

where $G_x$ and $G_y$ are the gradients in the $x$ and $y$ directions, respectively.

**Sobel Operator**

The Sobel operator is a discrete differentiation operator that computes an approximation of the gradient of the image intensity function. It uses convolution with two 3x3 kernels to calculate $G_x$ and $G_y$:

$$
K_x = \begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix}
$$

$$
K_y = \begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1
\end{bmatrix}
$$

**Prewitt Operator**

The Prewitt operator is similar to the Sobel operator but uses slightly different kernels. It is another gradient-based edge detection method that emphasizes changes in intensity. The Prewitt kernels are:

$$
K_x = \begin{bmatrix}
-1 & 0 & 1 \\
-1 & 0 & 1 \\
-1 & 0 & 1
\end{bmatrix}
$$

$$
K_y = \begin{bmatrix}
-1 & -1 & -1 \\
0 & 0 & 0 \\
1 & 1 & 1
\end{bmatrix}
$$

#### 6.1.2. Practical Implementation in C++ Using OpenCV

OpenCV provides convenient functions to apply the Sobel and Prewitt operators for edge detection.

**Sobel Edge Detection**

```cpp
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main() {
    // Load the image
    Mat image = imread("image.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Could not open or find the image!" << endl;
        return -1;
    }

    // Apply Sobel operator
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Mat sobel_edge;

    // Compute gradients in x and y directions
    Sobel(image, grad_x, CV_16S, 1, 0, 3);
    Sobel(image, grad_y, CV_16S, 0, 1, 3);

    // Convert gradients to absolute values
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    // Combine gradients
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobel_edge);

    // Display the result
    imshow("Original Image", image);
    imshow("Sobel Edge Detection", sobel_edge);
    waitKey(0);

    return 0;
}
```

**Prewitt Edge Detection**

OpenCV does not have a direct function for Prewitt operators, but we can implement it using convolution.

```cpp
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main() {
    // Load the image
    Mat image = imread("image.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Could not open or find the image!" << endl;
        return -1;
    }

    // Define Prewitt kernels
    Mat kernel_x = (Mat_<float>(3,3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
    Mat kernel_y = (Mat_<float>(3,3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);

    // Apply Prewitt operator
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Mat prewitt_edge;

    // Convolve with Prewitt kernels
    filter2D(image, grad_x, CV_16S, kernel_x);
    filter2D(image, grad_y, CV_16S, kernel_y);

    // Convert gradients to absolute values
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    // Combine gradients
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, prewitt_edge);

    // Display the result
    imshow("Original Image", image);
    imshow("Prewitt Edge Detection", prewitt_edge);
    waitKey(0);

    return 0;
}
```

#### 6.1.3. Detailed Explanation

1. **Gradient Computation**:
    - The Sobel and Prewitt operators calculate the gradients $G_x$ and $G_y$ by convolving the image with the respective kernels. The gradient in the $x$ direction $G_x$ highlights vertical edges, while the gradient in the $y$ direction $G_y$ highlights horizontal edges.

2. **Combining Gradients**:
    - The gradients $G_x$ and $G_y$ are combined to obtain the gradient magnitude, which represents the edge strength at each pixel. This combination is typically done using the `addWeighted` function in OpenCV, which allows for a balanced combination of the two gradients.

3. **Absolute Value Conversion**:
    - Since gradients can have negative values, converting them to their absolute values ensures that all edges are represented with positive intensity values, making it easier to visualize the edges.

#### 6.1.4. Conclusion

Gradient-based methods such as the Sobel and Prewitt operators are foundational techniques for edge detection in image processing. By computing the gradients of an image and identifying significant changes in intensity, these methods effectively highlight edges within the image. OpenCV provides convenient functions to apply these operators, making it accessible for various applications in computer vision. Understanding the mathematical background and implementation of these operators enables the development of robust edge detection algorithms for diverse image processing tasks.

### 6.2. Canny Edge Detector

The Canny Edge Detector is one of the most widely used and robust edge detection algorithms in image processing and computer vision. Developed by John F. Canny in 1986, this algorithm is known for its ability to detect a wide range of edges in images, while maintaining good noise suppression. The Canny Edge Detector achieves this by using a multi-stage process that includes noise reduction, gradient calculation, non-maximum suppression, and edge tracking by hysteresis. This subchapter will cover the mathematical background of the Canny Edge Detector and demonstrate its implementation using OpenCV in C++.

#### 6.2.1. Mathematical Background

The Canny Edge Detector consists of the following steps:

1. **Noise Reduction**:
    - Gaussian filtering is applied to smooth the image and reduce noise. The Gaussian filter is defined as:
      $$
      G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}
      $$
      where $\sigma$ is the standard deviation of the Gaussian filter.

2. **Gradient Calculation**:
    - The gradient magnitude and direction are computed using the Sobel operator:
      $$
      G_x = \begin{bmatrix}
      -1 & 0 & 1 \\
      -2 & 0 & 2 \\
      -1 & 0 & 1
      \end{bmatrix} * I
      \quad \text{and} \quad
      G_y = \begin{bmatrix}
      -1 & -2 & -1 \\
      0 & 0 & 0 \\
      1 & 2 & 1
      \end{bmatrix} * I
      $$
      $$
      G = \sqrt{G_x^2 + G_y^2}
      \quad \text{and} \quad
      \theta = \tan^{-1}\left(\frac{G_y}{G_x}\right)
      $$

3. **Non-Maximum Suppression**:
    - Thin the edges by suppressing non-maximum gradient magnitudes. Only local maxima are retained as edges.

4. **Edge Tracking by Hysteresis**:
    - Use two thresholds (high and low) to track edges. Strong edges (above the high threshold) are retained, and weak edges (between the high and low thresholds) are retained if they are connected to strong edges.

#### 6.2.2. Practical Implementation in C++ Using OpenCV

OpenCV provides a convenient function `Canny` to implement the Canny Edge Detector. Here is the implementation:

```cpp
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main() {
    // Load the image
    Mat image = imread("image.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Could not open or find the image!" << endl;
        return -1;
    }

    // Apply Gaussian blur to reduce noise
    Mat blurred;
    GaussianBlur(image, blurred, Size(5, 5), 1.4);

    // Apply Canny edge detector
    Mat edges;
    double lowThreshold = 50;
    double highThreshold = 150;
    Canny(blurred, edges, lowThreshold, highThreshold);

    // Display the result
    imshow("Original Image", image);
    imshow("Canny Edge Detection", edges);
    waitKey(0);

    return 0;
}
```

#### 6.2.3. Detailed Explanation

1. **Noise Reduction**:
    - The first step is to reduce noise in the image to prevent false edge detection. This is achieved by applying a Gaussian blur, which smooths the image. The `GaussianBlur` function in OpenCV is used for this purpose, where the kernel size and standard deviation are specified.

2. **Gradient Calculation**:
    - The gradient of the image is calculated using the Sobel operator. The gradients in the $x$ and $y$ directions, $G_x$ and $G_y$, are computed, and then the gradient magnitude $G$ and direction $\theta$ are obtained.

3. **Non-Maximum Suppression**:
    - To thin the edges, non-maximum suppression is applied. This step involves checking each pixel to see if it is a local maximum in the direction of the gradient. If a pixel is not a local maximum, its value is set to zero.

4. **Edge Tracking by Hysteresis**:
    - Finally, edge tracking by hysteresis is performed. Two thresholds are used: a high threshold to identify strong edges and a low threshold to identify weak edges. Weak edges are only retained if they are connected to strong edges. This ensures that the detected edges are continuous and reduces the chances of detecting false edges.

#### 6.2.4. Conclusion

The Canny Edge Detector is a powerful and widely used edge detection algorithm that provides robust and accurate edge detection. Its multi-stage process, including noise reduction, gradient calculation, non-maximum suppression, and edge tracking by hysteresis, ensures that it effectively detects edges while minimizing noise and false positives. OpenCV's `Canny` function makes it straightforward to implement this algorithm in C++, allowing for efficient and effective edge detection in various image processing and computer vision applications. By understanding the mathematical background and implementation of the Canny Edge Detector, one can develop advanced edge detection solutions for a wide range of applications.

### 6.3. Advanced Edge Detection Techniques

While traditional methods like the Sobel, Prewitt, and Canny edge detectors are effective for many applications, advanced edge detection techniques offer improved accuracy and robustness for more complex tasks. These advanced techniques leverage more sophisticated algorithms and can better handle noise, varying illumination, and complex textures. This subchapter explores several advanced edge detection methods, including Laplacian of Gaussian (LoG), Scharr filter, and edge detection using machine learning. Practical implementations using OpenCV in C++ are provided to illustrate these techniques.

#### 6.3.1. Laplacian of Gaussian (LoG)

The Laplacian of Gaussian (LoG) method combines Gaussian smoothing with the Laplacian operator. This technique is effective in detecting edges and locating them accurately while reducing noise.

**Mathematical Background**

1. **Gaussian Smoothing**:
    - Apply a Gaussian filter to smooth the image and reduce noise:
      $$
      G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}
      $$

2. **Laplacian Operator**:
    - The Laplacian operator is used to find areas of rapid intensity change:
      $$
      \nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}
      $$

3. **LoG**:
    - Combine the Gaussian and Laplacian operations:
      $$
      LoG(x, y) = \nabla^2(G(x, y) * I(x, y))
      $$

**Practical Implementation in C++ Using OpenCV**

```cpp
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main() {
    // Load the image
    Mat image = imread("image.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Could not open or find the image!" << endl;
        return -1;
    }

    // Apply Gaussian blur to reduce noise
    Mat blurred;
    GaussianBlur(image, blurred, Size(5, 5), 1.4);

    // Apply Laplacian operator
    Mat laplacian;
    Laplacian(blurred, laplacian, CV_16S, 3);

    // Convert result to 8-bit image
    Mat abs_laplacian;
    convertScaleAbs(laplacian, abs_laplacian);

    // Display the result
    imshow("Original Image", image);
    imshow("Laplacian of Gaussian Edge Detection", abs_laplacian);
    waitKey(0);

    return 0;
}
```

#### 6.3.2. Scharr Filter

The Scharr filter is an improved version of the Sobel operator, providing a more accurate approximation of the gradient, particularly for diagonal edges. It is particularly useful in applications requiring high precision.

**Mathematical Background**

The Scharr operator uses different convolution kernels for $G_x$ and $G_y$:

$$
K_x = \begin{bmatrix}
3 & 0 & -3 \\
10 & 0 & -10 \\
3 & 0 & -3
\end{bmatrix}
$$

$$
K_y = \begin{bmatrix}
3 & 10 & 3 \\
0 & 0 & 0 \\
-3 & -10 & -3
\end{bmatrix}
$$

**Practical Implementation in C++ Using OpenCV**

```cpp
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main() {
    // Load the image
    Mat image = imread("image.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Could not open or find the image!" << endl;
        return -1;
    }

    // Apply Scharr operator
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Mat scharr_edge;

    // Compute gradients in x and y directions
    Scharr(image, grad_x, CV_16S, 1, 0);
    Scharr(image, grad_y, CV_16S, 0, 1);

    // Convert gradients to absolute values
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    // Combine gradients
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, scharr_edge);

    // Display the result
    imshow("Original Image", image);
    imshow("Scharr Edge Detection", scharr_edge);
    waitKey(0);

    return 0;
}
```

#### 6.3.3. Edge Detection using Machine Learning

Recent advances in machine learning have led to the development of edge detection algorithms that can learn from data. These algorithms can achieve superior performance by leveraging large datasets and deep learning techniques.

**Mathematical Background**

Machine learning-based edge detection typically involves training a convolutional neural network (CNN) to detect edges. The network learns to identify edges by being trained on labeled datasets containing images and their corresponding edge maps.

**Practical Implementation using OpenCV and a Pre-trained Model**

Using a pre-trained model like Holistically-Nested Edge Detection (HED), we can perform edge detection with high accuracy.

First, download the HED model and the prototxt file from the official sources.

```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
using namespace cv;
using namespace dnn;
using namespace std;

int main() {
    // Load the image
    Mat image = imread("image.jpg");
    if (image.empty()) {
        cerr << "Could not open or find the image!" << endl;
        return -1;
    }

    // Load the pre-trained HED model
    Net net = readNetFromCaffe("deploy.prototxt", "hed_pretrained_bsds.caffemodel");

    // Prepare the image for the network
    Mat blob = blobFromImage(image, 1.0, Size(image.cols, image.rows), Scalar(104.00698793, 116.66876762, 122.67891434), false, false);

    // Set the input to the network
    net.setInput(blob);

    // Run forward pass to get the edge map
    Mat edgeMap = net.forward();

    // Convert the edge map to 8-bit image
    Mat edges;
    edgeMap = edgeMap.reshape(1, image.rows);
    normalize(edgeMap, edges, 0, 255, NORM_MINMAX);
    edges.convertTo(edges, CV_8U);

    // Display the result
    imshow("Original Image", image);
    imshow("HED Edge Detection", edges);
    waitKey(0);

    return 0;
}
```

#### 6.3.4. Detailed Explanation

1. **Laplacian of Gaussian (LoG)**:
    - LoG combines Gaussian smoothing with the Laplacian operator to detect edges. Gaussian smoothing reduces noise, while the Laplacian operator detects regions of rapid intensity change. The result is a method that effectively highlights edges while reducing noise.

2. **Scharr Filter**:
    - The Scharr filter is an improvement over the Sobel operator, providing better precision in edge detection, especially for diagonal edges. It uses optimized convolution kernels to compute the gradients, resulting in more accurate edge maps.

3. **Machine Learning-based Edge Detection**:
    - Machine learning techniques, particularly convolutional neural networks, have revolutionized edge detection. Pre-trained models like HED can achieve high accuracy by learning from large datasets. These models can generalize well to new images, making them robust against noise and varying lighting conditions.

#### 6.3.5. Conclusion

Advanced edge detection techniques offer significant improvements over traditional methods, providing greater accuracy and robustness. Methods like the Laplacian of Gaussian and the Scharr filter enhance the precision of edge detection, while machine learning-based approaches leverage the power of deep learning to achieve state-of-the-art results. OpenCV facilitates the implementation of these advanced techniques, making them accessible for various image processing and computer vision applications. By exploring and utilizing these advanced methods, we can develop more effective and reliable edge detection solutions.

