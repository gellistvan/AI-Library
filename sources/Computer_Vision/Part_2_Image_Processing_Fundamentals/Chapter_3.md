
\newpage
# Part II: Image Processing Fundamentals

## Chapter 3: Image Preprocessing

Image preprocessing is a crucial step in computer vision, setting the foundation for successful image analysis and interpretation. It involves preparing and enhancing raw image data to improve the performance of subsequent tasks, such as object detection, classification, and segmentation. This chapter delves into key techniques for image preprocessing, ensuring images are in an optimal state for further processing. We will explore methods to reduce noise, apply various filters, and equalize histograms, enhancing the quality and interpretability of images.

**Subchapters:**
- **Noise Reduction:** Techniques to minimize random variations in image brightness or color, preserving important details while removing unwanted artifacts.
- **Image Filtering (Convolution, Smoothing):** Methods for manipulating images using convolution and smoothing operations to emphasize or de-emphasize specific features.
- **Histogram Equalization:** Techniques to adjust the contrast of images, enhancing the visibility of details across the entire intensity range.
### 3.1. Noise Reduction

Noise reduction is a fundamental step in image preprocessing that involves removing unwanted variations in image intensity caused by various factors, such as sensor imperfections, environmental conditions, or transmission errors. Noise can significantly degrade the quality of an image, making it challenging to extract meaningful information. This subchapter explores different techniques for noise reduction, with a focus on mathematical background and practical implementation using OpenCV in C++.

#### 3.1.1. Types of Noise

Before diving into the techniques, it is essential to understand the common types of noise encountered in images:

- **Gaussian Noise:** This type of noise follows a normal distribution and is characterized by random variations in pixel values.
- **Salt-and-Pepper Noise:** This noise appears as random black and white pixels scattered throughout the image.
- **Speckle Noise:** Common in ultrasound and radar images, this noise appears as granular interference.

#### 3.1.2. Mathematical Background

Noise reduction techniques often rely on filtering, which involves modifying the pixel values based on their neighbors. Some commonly used filters include:

1. **Mean Filter (Averaging Filter):**
   The mean filter smooths the image by averaging the pixel values within a neighborhood.
   $$
   I'(x, y) = \frac{1}{mn} \sum_{i=-a}^{a} \sum_{j=-b}^{b} I(x+i, y+j)
   $$
   where $I'$ is the filtered image, $I$ is the original image, and $m \times n$ is the size of the filter kernel.

2. **Gaussian Filter:**
   The Gaussian filter reduces noise by weighting the average based on a Gaussian function.
   $$
   G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}
   $$
   The filtered image is obtained by convolving the original image with the Gaussian kernel.

3. **Median Filter:**
   The median filter replaces each pixel value with the median value of the neighboring pixels.
   $$
   I'(x, y) = \text{median} \{ I(x+i, y+j) \}
   $$
   where $i$ and $j$ range over the neighborhood.

#### 3.1.3. Practical Implementation in C++ Using OpenCV

OpenCV is a powerful library for image processing. Below are examples of implementing the discussed noise reduction techniques using OpenCV in C++.

**Mean Filter (Averaging Filter)**

```cpp
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    // Load the image
    Mat image = imread("image.jpg", IMREAD_COLOR);
    if(image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Apply the mean filter
    Mat meanFiltered;
    blur(image, meanFiltered, Size(5, 5));

    // Display the result
    imshow("Original Image", image);
    imshow("Mean Filtered Image", meanFiltered);
    waitKey(0);
    return 0;
}
```

**Gaussian Filter**

```cpp
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    // Load the image
    Mat image = imread("image.jpg", IMREAD_COLOR);
    if(image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Apply the Gaussian filter
    Mat gaussianFiltered;
    GaussianBlur(image, gaussianFiltered, Size(5, 5), 1.5);

    // Display the result
    imshow("Original Image", image);
    imshow("Gaussian Filtered Image", gaussianFiltered);
    waitKey(0);
    return 0;
}
```

**Median Filter**

```cpp
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    // Load the image
    Mat image = imread("image.jpg", IMREAD_COLOR);
    if(image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Apply the median filter
    Mat medianFiltered;
    medianBlur(image, medianFiltered, 5);

    // Display the result
    imshow("Original Image", image);
    imshow("Median Filtered Image", medianFiltered);
    waitKey(0);
    return 0;
}
```

#### 3.1.4. Conclusion

Noise reduction is a critical preprocessing step that enhances the quality of images by removing unwanted noise. By understanding the types of noise and applying appropriate filtering techniques such as mean, Gaussian, and median filters, we can significantly improve the performance of subsequent image processing tasks. The practical examples using OpenCV in C++ demonstrate the ease and effectiveness of implementing these techniques, providing a solid foundation for further exploration in computer vision.


### 3.2. Image Filtering (Convolution, Smoothing)

Image filtering is a fundamental process in computer vision, used to modify or enhance an image by emphasizing or de-emphasizing specific features. Filtering can be performed in both the spatial and frequency domains, but this subchapter focuses on spatial domain techniques, particularly convolution and smoothing. We will delve into the mathematical background of these operations and provide practical implementations using OpenCV in C++.

#### 3.2.1. Mathematical Background

**Convolution**

Convolution is a mathematical operation that involves sliding a filter (kernel) over an image to produce a new image. The filter is applied to each pixel and its neighbors, and the resulting values are combined to generate the output pixel. The mathematical definition of convolution for a 2D image is:

$$
I'(x, y) = (I * K)(x, y) = \sum_{i=-m}^{m} \sum_{j=-n}^{n} I(x-i, y-j) \cdot K(i, j)
$$

where:
- $I$ is the original image.
- $K$ is the convolution kernel (filter).
- $I'$ is the convolved image.
- $m$ and $n$ define the size of the kernel.

**Smoothing**

Smoothing is a type of filtering that reduces image noise and detail. The most common smoothing techniques are the mean (average) filter and the Gaussian filter.

1. **Mean Filter:**
   The mean filter smooths an image by averaging the pixel values within a neighborhood.

2. **Gaussian Filter:**
   The Gaussian filter applies a Gaussian function to the neighborhood, giving more weight to the central pixels.

#### 3.2.2. Practical Implementation in C++ Using OpenCV

OpenCV provides built-in functions for convolution and smoothing operations. Below are examples demonstrating these techniques.

**Convolution**

```cpp
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    // Load the image
    Mat image = imread("image.jpg", IMREAD_GRAYSCALE);
    if(image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Define the kernel
    float kernelData[] = {0, -1, 0, -1, 5, -1, 0, -1, 0}; // Example of a sharpening filter
    Mat kernel(3, 3, CV_32F, kernelData);

    // Apply convolution
    Mat convolvedImage;
    filter2D(image, convolvedImage, -1, kernel);

    // Display the result
    imshow("Original Image", image);
    imshow("Convolved Image", convolvedImage);
    waitKey(0);
    return 0;
}
```

**Mean Filter**

```cpp
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    // Load the image
    Mat image = imread("image.jpg", IMREAD_GRAYSCALE);
    if(image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Apply the mean filter
    Mat meanFiltered;
    blur(image, meanFiltered, Size(5, 5));

    // Display the result
    imshow("Original Image", image);
    imshow("Mean Filtered Image", meanFiltered);
    waitKey(0);
    return 0;
}
```

**Gaussian Filter**

```cpp
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    // Load the image
    Mat image = imread("image.jpg", IMREAD_GRAYSCALE);
    if(image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Apply the Gaussian filter
    Mat gaussianFiltered;
    GaussianBlur(image, gaussianFiltered, Size(5, 5), 1.5);

    // Display the result
    imshow("Original Image", image);
    imshow("Gaussian Filtered Image", gaussianFiltered);
    waitKey(0);
    return 0;
}
```

#### 3.2.3. Custom Convolution Implementation

For a deeper understanding, let's implement convolution from scratch in C++ without relying on OpenCV's `filter2D` function.

```cpp
#include <opencv2/opencv.hpp>

using namespace cv;

void customConvolution(const Mat& input, Mat& output, const Mat& kernel) {
    int kRows = kernel.rows;
    int kCols = kernel.cols;
    int kCenterX = kCols / 2;
    int kCenterY = kRows / 2;

    output = Mat::zeros(input.size(), input.type());

    for(int i = kCenterY; i < input.rows - kCenterY; ++i) {
        for(int j = kCenterX; j < input.cols - kCenterX; ++j) {
            float sum = 0.0;
            for(int m = 0; m < kRows; ++m) {
                for(int n = 0; n < kCols; ++n) {
                    int x = i + m - kCenterY;
                    int y = j + n - kCenterX;
                    sum += input.at<uchar>(x, y) * kernel.at<float>(m, n);
                }
            }
            output.at<uchar>(i, j) = saturate_cast<uchar>(sum);
        }
    }
}

int main() {
    // Load the image
    Mat image = imread("image.jpg", IMREAD_GRAYSCALE);
    if(image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Define the kernel
    float kernelData[] = {0, -1, 0, -1, 5, -1, 0, -1, 0}; // Example of a sharpening filter
    Mat kernel(3, 3, CV_32F, kernelData);

    // Apply custom convolution
    Mat convolvedImage;
    customConvolution(image, convolvedImage, kernel);

    // Display the result
    imshow("Original Image", image);
    imshow("Custom Convolved Image", convolvedImage);
    waitKey(0);
    return 0;
}
```

#### 3.2.4. Conclusion

Image filtering through convolution and smoothing techniques is essential for enhancing and modifying images in computer vision tasks. Convolution involves applying a kernel to an image to extract or emphasize features, while smoothing reduces noise and detail. Utilizing OpenCV's built-in functions simplifies these tasks, but understanding the underlying mathematics and implementing custom solutions provides a deeper comprehension and flexibility. By mastering these techniques, you can significantly improve the quality and usability of image data for subsequent processing stages.

### 3.3. Histogram Equalization

Histogram equalization is a powerful technique in image processing that enhances the contrast of an image. By redistributing the pixel intensity values, this method makes features in the image more distinct, thereby improving the visibility of details in both dark and bright regions. This subchapter explores the mathematical background of histogram equalization and provides practical implementations using OpenCV in C++.

#### 3.3.1. Mathematical Background

The histogram of an image is a graphical representation of the distribution of pixel intensity values. For an 8-bit grayscale image, the histogram consists of 256 bins, each corresponding to one intensity level ranging from 0 to 255. Histogram equalization aims to transform the intensity values so that the histogram of the output image is approximately uniform.

The steps involved in histogram equalization are as follows:

1. **Compute the Histogram:**
   Calculate the frequency of each intensity level in the image.

2. **Compute the Cumulative Distribution Function (CDF):**
   The CDF of the histogram provides a mapping between the input intensity values and the output values.
   $$
   \text{CDF}(i) = \sum_{j=0}^{i} \text{histogram}(j)
   $$

3. **Normalize the CDF:**
   Scale the CDF to cover the full range of intensity values.
   $$
   \text{CDF}_{\text{min}} = \min(\text{CDF})
   $$
   $$
   \text{CDF}_{\text{norm}}(i) = \frac{\text{CDF}(i) - \text{CDF}_{\text{min}}}{\text{total number of pixels} - \text{CDF}_{\text{min}}} \times 255
   $$

4. **Map the Intensity Values:**
   Use the normalized CDF to map the original intensity values to the new values.

#### 3.3.2. Practical Implementation in C++ Using OpenCV

OpenCV provides a straightforward function for histogram equalization called `equalizeHist`. Below is an example of its usage:

```cpp
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    // Load the image
    Mat image = imread("image.jpg", IMREAD_GRAYSCALE);
    if(image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Apply histogram equalization
    Mat equalizedImage;
    equalizeHist(image, equalizedImage);

    // Display the result
    imshow("Original Image", image);
    imshow("Equalized Image", equalizedImage);
    waitKey(0);
    return 0;
}
```

#### 3.3.3. Custom Histogram Equalization Implementation

For a deeper understanding, let's implement histogram equalization from scratch in C++.

```cpp
#include <opencv2/opencv.hpp>

#include <vector>
using namespace cv;
using namespace std;

void customHistogramEqualization(const Mat& input, Mat& output) {
    // Step 1: Compute the histogram
    vector<int> histogram(256, 0);
    for(int i = 0; i < input.rows; ++i) {
        for(int j = 0; j < input.cols; ++j) {
            histogram[input.at<uchar>(i, j)]++;
        }
    }

    // Step 2: Compute the CDF
    vector<int> cdf(256, 0);
    cdf[0] = histogram[0];
    for(int i = 1; i < 256; ++i) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    // Step 3: Normalize the CDF
    int cdf_min = *min_element(cdf.begin(), cdf.end());
    int totalPixels = input.rows * input.cols;
    vector<int> cdf_normalized(256, 0);
    for(int i = 0; i < 256; ++i) {
        cdf_normalized[i] = round((float)(cdf[i] - cdf_min) / (totalPixels - cdf_min) * 255);
    }

    // Step 4: Map the intensity values
    output = input.clone();
    for(int i = 0; i < input.rows; ++i) {
        for(int j = 0; j < input.cols; ++j) {
            output.at<uchar>(i, j) = cdf_normalized[input.at<uchar>(i, j)];
        }
    }
}

int main() {
    // Load the image
    Mat image = imread("image.jpg", IMREAD_GRAYSCALE);
    if(image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Apply custom histogram equalization
    Mat equalizedImage;
    customHistogramEqualization(image, equalizedImage);

    // Display the result
    imshow("Original Image", image);
    imshow("Custom Equalized Image", equalizedImage);
    waitKey(0);
    return 0;
}
```

#### 3.3.4. Conclusion

Histogram equalization is a powerful technique for enhancing the contrast of images, making features more distinguishable. By redistributing the intensity values using the cumulative distribution function, it ensures a more uniform histogram, improving the visual quality of the image. While OpenCV provides an easy-to-use function for this purpose, understanding and implementing the underlying algorithm from scratch deepens comprehension and provides greater flexibility in customizing image processing workflows.

