
\newpage
## Chapter 7: Corner and Interest Point Detection

In the realm of computer vision, the detection of corners and interest points plays a pivotal role in understanding and interpreting image structures. These key points are essential for tasks such as image matching, object recognition, and motion tracking. By identifying significant points in an image, algorithms can effectively capture important features that remain invariant to transformations like rotation and scaling. This chapter delves into the fundamental techniques used for detecting corners and interest points, providing a comprehensive overview of their underlying principles and applications.

**Subchapters**

1. **Harris Corner Detector**  
   The Harris Corner Detector is a cornerstone in feature detection, renowned for its robustness and accuracy. This subchapter explores the mathematical foundation of the Harris Corner Detector, illustrating how it identifies corners by analyzing the gradient of the image intensity.

2. **Shi-Tomasi Corner Detection**  
   Building on the concepts of the Harris Detector, the Shi-Tomasi method enhances the selection of corners by introducing an eigenvalue-based approach. This section examines how Shi-Tomasi improves upon its predecessor to select the most reliable corners for further processing.

3. **FAST, BRIEF, and ORB**  
   As real-time applications demand speed and efficiency, techniques like FAST (Features from Accelerated Segment Test), BRIEF (Binary Robust Independent Elementary Features), and ORB (Oriented FAST and Rotated BRIEF) have gained prominence. This subchapter provides an in-depth look at these modern, high-performance algorithms, detailing how they achieve rapid and effective corner and interest point detection in various contexts.

### 7.1. Harris Corner Detector

The Harris Corner Detector is one of the most widely used algorithms for detecting corners in images. It was introduced by Chris Harris and Mike Stephens in 1988 and is known for its robustness and accuracy. In this subchapter, we will delve into the mathematical background of the Harris Corner Detector, explain the key concepts, and provide detailed C++ code examples using OpenCV.

**Mathematical Background**

Corners in an image are points where the intensity changes significantly in multiple directions. To detect these points, the Harris Corner Detector uses the following mathematical approach:

1. **Image Gradients**:
   Compute the gradient of the image in both the x and y directions. These gradients represent changes in intensity.

   Let $I(x, y)$ be the image intensity at point $(x, y)$. The gradients $I_x$ and $I_y$ are computed as:
   $$
   $I_x = \frac{\partial I}{\partial x}, \quad I_y = \frac{\partial I}{\partial y}
   $$

2. **Structure Tensor (Second Moment Matrix)**:
   Construct a matrix $M$ using the gradients, which encapsulates the local intensity changes around a point:
   $$
   M = \begin{pmatrix}
   \sum I_x^2 & \sum I_x I_y \\
   \sum I_x I_y & \sum I_y^2
   \end{pmatrix}
   $$
   Here, the sums are computed over a window centered at the point $(x, y)$.

3. **Corner Response Function**:
   The Harris Corner Detector uses the eigenvalues of the matrix $M$ to determine the presence of a corner. Instead of computing the eigenvalues directly, it uses a corner response function $R$:
   $$
   R = \text{det}(M) - k \cdot (\text{trace}(M))^2
   $$
   where $\text{det}(M) = \lambda_1 \lambda_2$ and $\text{trace}(M) = \lambda_1 + \lambda_2$ are the determinant and trace of the matrix $M$, respectively, and $k$ is a sensitivity parameter (typically $k = 0.04$).

4. **Thresholding and Non-Maximum Suppression**:
   Identify points where $R$ is above a certain threshold and apply non-maximum suppression to ensure that only the most prominent corners are detected.

**Implementation in C++**

The following C++ code demonstrates the Harris Corner Detector using OpenCV.

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void detectHarrisCorners(const Mat& src, Mat& dst, int blockSize, int apertureSize, double k, double threshold) {
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    Mat dst_norm, dst_norm_scaled;
    dst = Mat::zeros(src.size(), CV_32FC1);

    // Detecting corners
    cornerHarris(gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

    // Normalizing
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);

    // Drawing a circle around corners
    for (int j = 0; j < dst_norm.rows; j++) {
        for (int i = 0; i < dst_norm.cols; i++) {
            if ((int)dst_norm.at<float>(j, i) > threshold) {
                circle(src, Point(i, j), 5, Scalar(0, 0, 255), 2, 8, 0);
            }
        }
    }

    // Showing the result
    namedWindow("Harris Corners", WINDOW_AUTOSIZE);
    imshow("Harris Corners", src);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: ./HarrisCornerDetector <image_path>" << endl;
        return -1;
    }

    // Load image
    Mat src = imread(argv[1], IMREAD_COLOR);
    if (src.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    Mat dst;
    int blockSize = 2;  // Size of neighborhood considered for corner detection
    int apertureSize = 3;  // Aperture parameter for the Sobel operator
    double k = 0.04;  // Harris detector free parameter
    double threshold = 200;  // Threshold for detecting corners

    detectHarrisCorners(src, dst, blockSize, apertureSize, k, threshold);

    waitKey(0);
    return 0;
}
```

**Explanation of the Code**

1. **Image Preprocessing**:
   The image is converted to grayscale since the Harris Corner Detector operates on single-channel images.

2. **Corner Detection**:
   The `cornerHarris` function computes the Harris response matrix for each pixel in the image. The parameters are:
    - `blockSize`: Size of the neighborhood considered for corner detection.
    - `apertureSize`: Aperture parameter for the Sobel operator used to compute image gradients.
    - `k`: Harris detector free parameter, typically set to 0.04.

3. **Normalization**:
   The result is normalized to the range [0, 255] to make it easier to visualize and threshold.

4. **Thresholding and Visualization**:
   A threshold is applied to the normalized response to identify strong corners. Circles are drawn around these points for visualization.

5. **Running the Code**:
   The main function loads an image, calls the `detectHarrisCorners` function, and displays the result.

By understanding and implementing the Harris Corner Detector, you can effectively identify corners in images, providing a foundation for more advanced computer vision tasks. This algorithm's robustness makes it a reliable choice for various applications, from object recognition to image stitching.

### 7.2. Shi-Tomasi Corner Detection

The Shi-Tomasi Corner Detection algorithm, also known as the Good Features to Track method, is an enhancement over the Harris Corner Detector. Introduced by Jianbo Shi and Carlo Tomasi in 1994, this method improves the selection of corners by focusing on the minimum eigenvalue of the structure tensor. This results in more reliable and stable corner detection, particularly useful in applications like optical flow and motion tracking. In this subchapter, we will explore the mathematical background of the Shi-Tomasi Corner Detector, explain its key concepts, and provide detailed C++ code examples using OpenCV.

**Mathematical Background**

The Shi-Tomasi method builds upon the Harris Corner Detector by considering the eigenvalues of the structure tensor but with a slightly different criterion for detecting corners.

1. **Structure Tensor (Second Moment Matrix)**:
   Similar to the Harris Corner Detector, the Shi-Tomasi method uses the gradients of the image to construct a structure tensor $M$ at each pixel $(x, y)$:
   $$
   M = \begin{pmatrix}
   \sum I_x^2 & \sum I_x I_y \\
   \sum I_x I_y & \sum I_y^2
   \end{pmatrix}
   $$
   Here, the sums are computed over a window centered at the pixel.

2. **Eigenvalues of the Structure Tensor**:
   The eigenvalues $\lambda_1$ and $\lambda_2$ of the matrix $M$ indicate the intensity change in different directions around the pixel.

3. **Corner Response Function**:
   Instead of using the determinant and trace of $M$ like the Harris method, Shi-Tomasi uses the minimum eigenvalue $\lambda_{\text{min}} = \min(\lambda_1, \lambda_2)$ as the corner response function:
   $$
   R = \lambda_{\text{min}}
   $$
   A pixel is considered a corner if $R$ is above a certain threshold. This ensures that only the most prominent corners, which have significant intensity changes in multiple directions, are selected.

**Implementation in C++**

The following C++ code demonstrates the Shi-Tomasi Corner Detection using OpenCV.

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void detectShiTomasiCorners(const Mat& src, Mat& dst, int maxCorners, double qualityLevel, double minDistance) {
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    vector<Point2f> corners;

    // Parameters for Shi-Tomasi corner detection
    int blockSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;

    // Detecting corners
    goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);

    // Draw corners on the image
    dst = src.clone();
    for (size_t i = 0; i < corners.size(); i++) {
        circle(dst, corners[i], 5, Scalar(0, 255, 0), 2, 8, 0);
    }

    // Showing the result
    namedWindow("Shi-Tomasi Corners", WINDOW_AUTOSIZE);
    imshow("Shi-Tomasi Corners", dst);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: ./ShiTomasiCornerDetector <image_path>" << endl;
        return -1;
    }

    // Load image
    Mat src = imread(argv[1], IMREAD_COLOR);
    if (src.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    Mat dst;
    int maxCorners = 100;  // Maximum number of corners to return
    double qualityLevel = 0.01;  // Quality level parameter
    double minDistance = 10;  // Minimum possible Euclidean distance between the returned corners

    detectShiTomasiCorners(src, dst, maxCorners, qualityLevel, minDistance);

    waitKey(0);
    return 0;
}
```

**Explanation of the Code**

1. **Image Preprocessing**:
   The image is converted to grayscale since the Shi-Tomasi Corner Detector operates on single-channel images.

2. **Corner Detection**:
   The `goodFeaturesToTrack` function is used to detect corners based on the Shi-Tomasi method. The parameters are:
    - `maxCorners`: Maximum number of corners to return.
    - `qualityLevel`: Multiplier for the minimum eigenvalue; only corners with a corner response greater than `qualityLevel` times the maximum eigenvalue will be considered.
    - `minDistance`: Minimum possible Euclidean distance between the returned corners.
    - `blockSize`: Size of the neighborhood considered for corner detection.
    - `useHarrisDetector`: Boolean flag indicating whether to use the Harris detector (false for Shi-Tomasi).
    - `k`: Free parameter of the Harris detector (ignored for Shi-Tomasi).

3. **Drawing Corners**:
   Detected corners are drawn on the image using green circles for visualization.

4. **Running the Code**:
   The main function loads an image, calls the `detectShiTomasiCorners` function, and displays the result.

The Shi-Tomasi Corner Detector offers a reliable and efficient method for detecting corners in images. Its use of the minimum eigenvalue ensures that only the most prominent and stable corners are selected, making it suitable for a wide range of computer vision applications. By understanding and implementing this method, you can enhance your ability to analyze and interpret image structures.

### 7.3. FAST, BRIEF, and ORB

In real-time computer vision applications, speed and efficiency are paramount. Traditional corner detection methods, while accurate, often fall short in performance. To address this, modern algorithms like FAST (Features from Accelerated Segment Test), BRIEF (Binary Robust Independent Elementary Features), and ORB (Oriented FAST and Rotated BRIEF) have been developed. These techniques provide rapid and effective corner and feature detection, making them ideal for applications requiring real-time processing. This subchapter delves into the mathematical background of these algorithms, explaining their key concepts and providing detailed C++ code examples using OpenCV.

#### 7.3.1. FAST (Features from Accelerated Segment Test)

**Mathematical Background**

FAST is a corner detection method that identifies corners by examining the intensity of a circular ring of pixels around a candidate pixel. The algorithm is based on the following steps:

1. **Pixel Intensity Comparison**:
   Consider a pixel $p$ and its surrounding pixels on a circle of radius 3. The pixel $p$ is considered a corner if there exists a set of contiguous pixels in the circle that are either all brighter or all darker than the intensity of $p$ by a threshold $t$.

2. **High-Speed Test**:
   To speed up the process, a high-speed test is performed by comparing pixels at positions 1, 5, 9, and 13 on the circle. If at least three of these pixels are all brighter or all darker than $p$ by the threshold $t$, then $p$ is considered a candidate corner.

3. **Non-Maximum Suppression**:
   After identifying candidate corners, non-maximum suppression is applied to retain only the strongest corners.

**Implementation in C++ using OpenCV**

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void detectFASTCorners(const Mat& src, Mat& dst, int threshold, bool nonmaxSuppression) {
    vector<KeyPoint> keypoints;
    FAST(src, keypoints, threshold, nonmaxSuppression);

    // Draw corners on the image
    dst = src.clone();
    drawKeypoints(src, keypoints, dst, Scalar::all(-1), DrawMatchesFlags::DRAW_OVER_OUTIMG);

    // Showing the result
    namedWindow("FAST Corners", WINDOW_AUTOSIZE);
    imshow("FAST Corners", dst);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: ./FASTCornerDetector <image_path>" << endl;
        return -1;
    }

    // Load image
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if (src.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    Mat dst;
    int threshold = 50;  // Threshold for the FAST detector
    bool nonmaxSuppression = true;  // Apply non-maximum suppression

    detectFASTCorners(src, dst, threshold, nonmaxSuppression);

    waitKey(0);
    return 0;
}
```

#### 7.3.2. BRIEF (Binary Robust Independent Elementary Features)

**Mathematical Background**

BRIEF is a feature descriptor that describes an image patch using binary strings. It provides a compact and efficient representation of keypoints.

1. **Intensity Pair Comparisons**:
   BRIEF generates a binary string by comparing the intensities of pairs of pixels within a predefined patch around a keypoint. Each bit in the descriptor is set based on whether the intensity of the first pixel in the pair is greater than the second.

2. **Descriptor Construction**:
   Given a keypoint $p$, BRIEF constructs the descriptor by sampling $n$ pairs of pixels within a patch centered at $p$. The result is a binary string of length $n$.

**Implementation in C++ using OpenCV**

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void detectBRIEFDescriptors(const Mat& src, vector<KeyPoint>& keypoints, Mat& descriptors) {
    Ptr<xfeatures2d::BriefDescriptorExtractor> brief = xfeatures2d::BriefDescriptorExtractor::create();
    brief->compute(src, keypoints, descriptors);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: ./BRIEFDescriptorExtractor <image_path>" << endl;
        return -1;
    }

    // Load image
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if (src.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    // Detect FAST keypoints
    vector<KeyPoint> keypoints;
    int threshold = 50;
    bool nonmaxSuppression = true;
    FAST(src, keypoints, threshold, nonmaxSuppression);

    // Compute BRIEF descriptors
    Mat descriptors;
    detectBRIEFDescriptors(src, keypoints, descriptors);

    cout << "Number of keypoints: " << keypoints.size() << endl;
    cout << "Descriptor size: " << descriptors.size() << endl;

    return 0;
}
```

#### 7.3.3. ORB (Oriented FAST and Rotated BRIEF)

**Mathematical Background**

ORB is a fusion of FAST keypoint detector and BRIEF descriptor with enhancements to improve performance and robustness.

1. **Oriented FAST**:
   ORB uses FAST to detect keypoints but adds orientation information to each keypoint to make the descriptor rotation invariant. The orientation is computed using the intensity centroid method.

2. **Rotated BRIEF**:
   ORB modifies the BRIEF descriptor to account for the keypoint orientation. This is done by rotating the BRIEF sampling pattern according to the orientation of the keypoint.

3. **Scale Invariance**:
   ORB ensures scale invariance by constructing a scale pyramid of the image and detecting keypoints at multiple scales.

**Implementation in C++ using OpenCV**

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void detectORBFeatures(const Mat& src, Mat& dst) {
    Ptr<ORB> orb = ORB::create();
    vector<KeyPoint> keypoints;
    Mat descriptors;

    // Detect ORB keypoints and descriptors
    orb->detectAndCompute(src, Mat(), keypoints, descriptors);

    // Draw keypoints on the image
    dst = src.clone();
    drawKeypoints(src, keypoints, dst, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Showing the result
    namedWindow("ORB Features", WINDOW_AUTOSIZE);
    imshow("ORB Features", dst);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: ./ORBFeatureDetector <image_path>" << endl;
        return -1;
    }

    // Load image
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if (src.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    Mat dst;
    detectORBFeatures(src, dst);

    waitKey(0);
    return 0;
}
```

**Explanation of the Code**

1. **FAST**:
    - `FAST`: This function detects keypoints using the FAST algorithm. The parameters include the threshold for detecting corners and a boolean indicating whether to apply non-maximum suppression.
    - `drawKeypoints`: This function visualizes the detected keypoints on the image.

2. **BRIEF**:
    - `xfeatures2d::BriefDescriptorExtractor`: This class computes BRIEF descriptors for the detected keypoints. The descriptors are binary strings representing the intensity comparisons within a patch around each keypoint.

3. **ORB**:
    - `ORB::create`: This function initializes the ORB detector.
    - `orb->detectAndCompute`: This method detects keypoints and computes descriptors using the ORB algorithm, which combines FAST keypoint detection with BRIEF descriptors, enhanced with orientation and scale invariance.

By understanding and implementing FAST, BRIEF, and ORB, you can leverage the speed and efficiency of these modern algorithms for real-time computer vision applications. These methods provide a robust framework for detecting and describing features in images, making them indispensable tools in the field of computer vision.

