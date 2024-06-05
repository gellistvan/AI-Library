
\newpage
# Part IV: Image Segmentation

## Chapter 9: Thresholding Techniques

Thresholding is a fundamental technique in image processing used to simplify images by converting grayscale images into binary images. This process is essential for various applications, such as object detection and segmentation. In this chapter, we will explore key thresholding methods, starting with global and adaptive thresholding techniques that apply fixed and varying thresholds across an image. We will then delve into Otsu's method, an advanced approach that automatically determines an optimal threshold to minimize intra-class variance, enhancing the accuracy of image segmentation.

### 9.1. Global and Adaptive Thresholding

Thresholding is a technique that converts a grayscale image into a binary image, where the pixels are assigned one of two values based on a threshold. This process helps in segmenting the foreground from the background. In this subchapter, we will delve into two main types of thresholding: Global Thresholding and Adaptive Thresholding. We will also explore their mathematical background and provide C++ code examples using the OpenCV library.

**Global Thresholding**

Global thresholding involves selecting a single threshold value for the entire image. Every pixel value is compared to this threshold, and based on this comparison, the pixel is either assigned to the foreground or background.

**Mathematical Background**

Let $T$ be the threshold value, and let $I(x, y)$ be the intensity value of a pixel at position $(x, y)$. The binary image $B(x, y)$ is obtained as follows:

$$ B(x, y) =
\begin{cases}
0 & \text{if } I(x, y) < T \\
255 & \text{if } I(x, y) \geq T
\end{cases} $$

**Implementation in C++ using OpenCV**

```cpp
#include <opencv2/opencv.hpp>

#include <iostream>

int main() {
    // Load the image in grayscale
    cv::Mat image = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }

    cv::Mat binaryImage;
    double thresholdValue = 128.0; // Example threshold value
    double maxValue = 255.0; // Value to assign for pixels >= threshold

    // Apply global thresholding
    cv::threshold(image, binaryImage, thresholdValue, maxValue, cv::THRESH_BINARY);

    // Display results
    cv::imshow("Original Image", image);
    cv::imshow("Binary Image", binaryImage);
    cv::waitKey(0);

    return 0;
}
```

In this example, we use OpenCV's `threshold` function, which applies the thresholding operation to the input image.

**Adaptive Thresholding**

Unlike global thresholding, adaptive thresholding calculates the threshold for smaller regions of the image. This method is useful when the image has varying lighting conditions, which can cause global thresholding to perform poorly.

**Mathematical Background**

Adaptive thresholding involves calculating the threshold value for each pixel based on the pixel values in its local neighborhood. Two common methods are mean and Gaussian adaptive thresholding.

**Mean Adaptive Thresholding:**

$$ T(x, y) = \frac{1}{N} \sum_{(x', y') \in N(x, y)} I(x', y') - C $$

**Gaussian Adaptive Thresholding:**

$$ T(x, y) = \sum_{(x', y') \in N(x, y)} G(x', y') \cdot I(x', y') - C $$

where:
- $N(x, y)$ is the neighborhood of the pixel $(x, y)$.
- $G(x', y')$ is the Gaussian weight.
- $C$ is a constant subtracted from the mean or weighted mean.

**Implementation in C++ using OpenCV**

```cpp
#include <opencv2/opencv.hpp>

#include <iostream>

int main() {
    // Load the image in grayscale
    cv::Mat image = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }

    cv::Mat meanBinaryImage, gaussianBinaryImage;
    int blockSize = 11; // Size of the neighborhood
    double C = 2.0; // Constant to subtract from the mean or weighted mean

    // Apply mean adaptive thresholding
    cv::adaptiveThreshold(image, meanBinaryImage, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, blockSize, C);

    // Apply Gaussian adaptive thresholding
    cv::adaptiveThreshold(image, gaussianBinaryImage, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blockSize, C);

    // Display results
    cv::imshow("Original Image", image);
    cv::imshow("Mean Adaptive Thresholding", meanBinaryImage);
    cv::imshow("Gaussian Adaptive Thresholding", gaussianBinaryImage);
    cv::waitKey(0);

    return 0;
}
```

In this example, we use OpenCV's `adaptiveThreshold` function, which performs adaptive thresholding using either the mean or Gaussian method. The `blockSize` parameter defines the size of the local region, and `C` is a constant subtracted from the calculated threshold value.

**Summary**

Global thresholding is straightforward but can be ineffective for images with varying illumination. Adaptive thresholding addresses this limitation by computing local thresholds, providing better results for images with non-uniform lighting conditions. The provided C++ code examples demonstrate how to implement these techniques using the OpenCV library, showcasing the practicality and effectiveness of both methods.

### 9.2. Otsu's Method

Otsu's method is an advanced thresholding technique used to automatically determine the optimal threshold value for converting a grayscale image into a binary image. Unlike global thresholding, which uses a fixed threshold, Otsu's method calculates the threshold based on the image's histogram to minimize intra-class variance.

**Mathematical Background**

Otsu's method aims to find the threshold $T$ that minimizes the weighted sum of intra-class variances of the foreground and background pixels. The key steps involved in Otsu's method are:

1. **Histogram Calculation**: Compute the histogram of the grayscale image.
2. **Class Probabilities**: Calculate the probabilities of the two classes separated by the threshold $T$.
3. **Class Means**: Calculate the means of the two classes.
4. **Intra-class Variance**: Compute the intra-class variance for each possible threshold and find the threshold that minimizes this variance.

The steps can be summarized mathematically as follows:

1. **Histogram Calculation**:
   $$
   \text{Let } p_i \text{ be the probability of intensity level } i \text{ in the image.}
   $$

2. **Class Probabilities and Means**:
   $$
   \omega_1(T) = \sum_{i=0}^{T} p_i \quad \text{(Probability of class 1)}
   $$
   $$
   \omega_2(T) = \sum_{i=T+1}^{L-1} p_i \quad \text{(Probability of class 2)}
   $$
   $$
   \mu_1(T) = \frac{\sum_{i=0}^{T} i \cdot p_i}{\omega_1(T)} \quad \text{(Mean of class 1)}
   $$
   $$
   \mu_2(T) = \frac{\sum_{i=T+1}^{L-1} i \cdot p_i}{\omega_2(T)} \quad \text{(Mean of class 2)}
   $$

3. **Intra-class Variance**:
   $$
   \sigma_w^2(T) = \omega_1(T) \sigma_1^2(T) + \omega_2(T) \sigma_2^2(T)
   $$
   where $\sigma_1^2(T)$ and $\sigma_2^2(T)$ are the variances of the two classes.

4. **Optimal Threshold**:
   $$
   T^* = \arg\min_T \sigma_w^2(T)
   $$

**Implementation in C++ using OpenCV**

OpenCV provides a convenient function to implement Otsu's method. Below is a detailed implementation in C++:

```cpp
#include <opencv2/opencv.hpp>

#include <iostream>

int main() {
    // Load the image in grayscale
    cv::Mat image = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }

    cv::Mat otsuBinaryImage;
    double maxValue = 255.0; // Value to assign for pixels >= threshold
    double otsuThreshold;

    // Apply Otsu's method
    otsuThreshold = cv::threshold(image, otsuBinaryImage, 0, maxValue, cv::THRESH_BINARY | cv::THRESH_OTSU);

    std::cout << "Otsu's threshold: " << otsuThreshold << std::endl;

    // Display results
    cv::imshow("Original Image", image);
    cv::imshow("Otsu's Binary Image", otsuBinaryImage);
    cv::waitKey(0);

    return 0;
}
```

In this example, we use OpenCV's `threshold` function with the `THRESH_OTSU` flag, which automatically computes the optimal threshold using Otsu's method. The computed threshold is also printed.

**Detailed Explanation**

1. **Load the Image**: The image is loaded in grayscale mode using `cv::imread`.
2. **Apply Otsu's Method**: The `threshold` function is used with the `THRESH_BINARY | THRESH_OTSU` flag. This function calculates the optimal threshold value and applies binary thresholding.
3. **Display Results**: The original and thresholded binary images are displayed using `cv::imshow`.

**Custom Implementation of Otsu's Method**

For educational purposes, let's implement Otsu's method from scratch in C++ without relying on OpenCV's built-in function.

```cpp
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>

#include <numeric>

double calculateOtsuThreshold(const cv::Mat& image) {
    // Calculate histogram
    int histSize = 256;
    std::vector<int> histogram(histSize, 0);
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            histogram[image.at<uchar>(i, j)]++;
        }
    }

    // Calculate total number of pixels
    int totalPixels = image.rows * image.cols;

    // Calculate probabilities
    std::vector<double> probabilities(histSize, 0.0);
    for (int i = 0; i < histSize; ++i) {
        probabilities[i] = static_cast<double>(histogram[i]) / totalPixels;
    }

    // Calculate class probabilities and means
    std::vector<double> omega1(histSize, 0.0), mu1(histSize, 0.0);
    omega1[0] = probabilities[0];
    for (int i = 1; i < histSize; ++i) {
        omega1[i] = omega1[i - 1] + probabilities[i];
        mu1[i] = mu1[i - 1] + i * probabilities[i];
    }

    double totalMean = mu1[histSize - 1];

    // Calculate between-class variance for each threshold
    std::vector<double> sigmaB2(histSize, 0.0);
    for (int i = 0; i < histSize; ++i) {
        double omega2 = 1.0 - omega1[i];
        if (omega1[i] > 0 && omega2 > 0) {
            double mu2 = (totalMean - mu1[i]) / omega2;
            sigmaB2[i] = omega1[i] * omega2 * (mu1[i] / omega1[i] - mu2) * (mu1[i] / omega1[i] - mu2);
        }
    }

    // Find the threshold that maximizes between-class variance
    double maxSigmaB2 = 0.0;
    int optimalThreshold = 0;
    for (int i = 0; i < histSize; ++i) {
        if (sigmaB2[i] > maxSigmaB2) {
            maxSigmaB2 = sigmaB2[i];
            optimalThreshold = i;
        }
    }

    return optimalThreshold;
}

int main() {
    // Load the image in grayscale
    cv::Mat image = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }

    // Calculate Otsu's threshold
    double otsuThreshold = calculateOtsuThreshold(image);
    std::cout << "Otsu's threshold (custom implementation): " << otsuThreshold << std::endl;

    // Apply the threshold to create a binary image
    cv::Mat otsuBinaryImage;
    cv::threshold(image, otsuBinaryImage, otsuThreshold, 255, cv::THRESH_BINARY);

    // Display results
    cv::imshow("Original Image", image);
    cv::imshow("Otsu's Binary Image (Custom)", otsuBinaryImage);
    cv::waitKey(0);

    return 0;
}
```

**Explanation of Custom Implementation**

1. **Histogram Calculation**: A histogram of the image is calculated to count the frequency of each intensity level.
2. **Probability Calculation**: The probability of each intensity level is computed by dividing the histogram values by the total number of pixels.
3. **Class Probabilities and Means**: The class probabilities $\omega_1(T)$ and $\omega_2(T)$, and the class means $\mu_1(T)$ and $\mu_2(T)$, are calculated for each possible threshold.
4. **Between-class Variance Calculation**: The between-class variance $\sigma_B^2(T)$ is computed for each threshold.
5. **Optimal Threshold Selection**: The threshold that maximizes the between-class variance is selected as the optimal threshold.

**Summary**

Otsu's method is a powerful technique for automatic thresholding, especially useful when the image histogram has a bimodal distribution. It calculates the optimal threshold by minimizing the intra-class variance, ensuring effective segmentation of the image into foreground and background. The provided C++ code examples demonstrate both the usage of OpenCV's built-in function and a custom implementation, highlighting the practicality and robustness of Otsu's method in image processing tasks.

