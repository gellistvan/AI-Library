
\newpage
# Part VI: 3D Vision and Geometry

\newpage
## Chapter 16: Stereo Vision and Depth Estimation

Stereo vision and depth estimation are fundamental components of computer vision that enable machines to perceive the world in three dimensions. This chapter delves into the principles and techniques that allow computers to reconstruct depth information from two or more images. By understanding the geometric relationships between different viewpoints, we can estimate the distance and structure of objects within a scene. The following subchapters will explore the foundational concepts of epipolar geometry, the methodologies behind stereo matching algorithms, and the practical applications of extracting depth from stereo images.

### 16.1. Epipolar Geometry

Epipolar geometry is a crucial concept in stereo vision, providing the geometric relationship between two views of the same scene. This relationship simplifies the process of finding corresponding points in stereo images, which is essential for depth estimation. In this subchapter, we will delve into the mathematical background of epipolar geometry, and demonstrate its implementation using C++ and the OpenCV library.

**Mathematical Background**

When capturing two images of the same scene from different viewpoints, there exists a geometric relationship between these images. This relationship can be described using the concepts of epipolar planes, epipolar lines, and the fundamental matrix.

- **Epipolar Plane**: Formed by a 3D point and the optical centers of two cameras.
- **Epipolar Line**: The intersection of the epipolar plane with the image planes of the cameras.
- **Fundamental Matrix (F)**: Encodes the epipolar geometry of the two views.

If $\mathbf{x}$ and $\mathbf{x'}$ are corresponding points in the two images, their relationship can be expressed as:

$$ \mathbf{x'}^T \mathbf{F} \mathbf{x} = 0 $$

Where $\mathbf{F}$ is the fundamental matrix, and $\mathbf{x}$ and $\mathbf{x'}$ are the homogeneous coordinates of the points in the first and second images respectively.

**Derivation of the Fundamental Matrix**

To derive the fundamental matrix, consider two camera matrices $P$ and $P'$ for the two views. These can be represented as:

$$ P = [K|0] $$
$$ P' = [K'R|K't] $$

Where $K$ and $K'$ are the intrinsic matrices, $R$ is the rotation matrix, and $t$ is the translation vector between the cameras. The fundamental matrix $F$ can be derived from the essential matrix $E$:

$$ E = [t]_x R $$

Where $[t]_x$ is the skew-symmetric matrix of the translation vector $t$. The fundamental matrix is related to the essential matrix by:

$$ F = K'^{-T} E K^{-1} $$

**Implementation in C++ using OpenCV**

OpenCV provides a rich set of functions to handle epipolar geometry. Below, we will demonstrate how to compute the fundamental matrix and visualize epipolar lines using OpenCV.

**Step 1: Include Necessary Headers**

```cpp
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
```

**Step 2: Load the Images**

```cpp
int main() {
    cv::Mat img1 = cv::imread("left_image.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("right_image.jpg", cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Could not open or find the images!" << std::endl;
        return -1;
    }
```

**Step 3: Detect Keypoints and Compute Descriptors**

```cpp
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    orb->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);
```

**Step 4: Match Descriptors**

```cpp
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher->match(descriptors1, descriptors2, matches);
```

**Step 5: Select Good Matches**

```cpp
    std::sort(matches.begin(), matches.end());
    const int numGoodMatches = matches.size() * 0.15f;
    matches.erase(matches.begin() + numGoodMatches, matches.end());
```

**Step 6: Extract Point Correspondences**

```cpp
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }
```

**Step 7: Compute the Fundamental Matrix**

```cpp
    cv::Mat fundamentalMatrix = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC);
    std::cout << "Fundamental Matrix:\n" << fundamentalMatrix << std::endl;
```

**Step 8: Draw Epipolar Lines**

```cpp
    std::vector<cv::Vec3f> lines1, lines2;
    cv::computeCorrespondEpilines(points1, 1, fundamentalMatrix, lines1);
    cv::computeCorrespondEpilines(points2, 2, fundamentalMatrix, lines2);

    cv::Mat img1Color, img2Color;
    cv::cvtColor(img1, img1Color, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, img2Color, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < lines1.size(); i++) {
        cv::line(img1Color, cv::Point(0, -lines1[i][2] / lines1[i][1]),
                             cv::Point(img1.cols, -(lines1[i][2] + lines1[i][0] * img1.cols) / lines1[i][1]),
                             cv::Scalar(0, 255, 0));
        cv::circle(img1Color, points1[i], 5, cv::Scalar(0, 0, 255), -1);

        cv::line(img2Color, cv::Point(0, -lines2[i][2] / lines2[i][1]),
                             cv::Point(img2.cols, -(lines2[i][2] + lines2[i][0] * img2.cols) / lines2[i][1]),
                             cv::Scalar(0, 255, 0));
        cv::circle(img2Color, points2[i], 5, cv::Scalar(0, 0, 255), -1);
    }

    cv::imshow("Epipolar Lines in Image 1", img1Color);
    cv::imshow("Epipolar Lines in Image 2", img2Color);
    cv::waitKey(0);

    return 0;
}
```

This code snippet captures the essential steps to compute and visualize epipolar geometry using OpenCV in C++. First, we detect keypoints and compute descriptors in both images. Next, we match these descriptors and select the good matches. From the matched points, we compute the fundamental matrix and visualize the epipolar lines on the images.

By understanding and implementing epipolar geometry, we set the stage for accurate depth estimation, forming the foundation for more advanced stereo vision techniques.

### 16.2. Stereo Matching Algorithms

Stereo matching is the process of finding corresponding points in two or more images of the same scene taken from different viewpoints. This is a critical step in stereo vision, enabling the reconstruction of 3D depth information. In this subchapter, we will explore the mathematical background of stereo matching algorithms and demonstrate their implementation using C++ and OpenCV.

**Mathematical Background**

Stereo matching algorithms can be broadly categorized into two types: local methods and global methods.

- **Local Methods**: These methods compute disparity (the difference in the coordinates of corresponding points) by comparing small patches of pixels around each point. They are usually faster but may suffer in regions with low texture or repetitive patterns.
- **Global Methods**: These methods consider the entire image and formulate the stereo matching as an optimization problem, often yielding more accurate results but at a higher computational cost.

**Disparity Map**

The goal of stereo matching is to compute a disparity map, where each pixel value represents the disparity between the corresponding points in the left and right images. The disparity $d$ for a point $(x, y)$ can be defined as:

$$ d = x_{\text{left}} - x_{\text{right}} $$

Using the disparity map, the depth $Z$ of a point can be computed as:

$$ Z = \frac{f \cdot B}{d} $$

Where $f$ is the focal length of the camera and $B$ is the baseline distance between the two camera centers.

**Implementation in C++ using OpenCV**

OpenCV provides several functions for stereo matching, including the block matching algorithm (StereoBM) and the semi-global block matching algorithm (StereoSGBM). Below, we will demonstrate the implementation of these algorithms using OpenCV.

**Step 1: Include Necessary Headers**

```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
```

**Step 2: Load the Stereo Images**

```cpp
int main() {
    cv::Mat imgLeft = cv::imread("left_image.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat imgRight = cv::imread("right_image.jpg", cv::IMREAD_GRAYSCALE);

    if (imgLeft.empty() || imgRight.empty()) {
        std::cerr << "Could not open or find the images!" << std::endl;
        return -1;
    }
```

**Step 3: Preprocess the Images (Optional)**

Preprocessing steps like histogram equalization can enhance the matching accuracy.

```cpp
    cv::equalizeHist(imgLeft, imgLeft);
    cv::equalizeHist(imgRight, imgRight);
```

**Step 4: Compute Disparity Map using StereoBM**

```cpp
    int numDisparities = 16 * 5;  // Maximum disparity minus minimum disparity
    int blockSize = 21;  // Matched block size. It must be an odd number >=1 

    cv::Ptr<cv::StereoBM> stereoBM = cv::StereoBM::create(numDisparities, blockSize);

    cv::Mat disparityBM;
    stereoBM->compute(imgLeft, imgRight, disparityBM);

    // Normalize the disparity map for visualization
    cv::Mat disparityBM8U;
    disparityBM.convertTo(disparityBM8U, CV_8U, 255 / (numDisparities * 16.0));

    cv::imshow("Disparity Map - BM", disparityBM8U);
    cv::waitKey(0);
```

**Step 5: Compute Disparity Map using StereoSGBM**

```cpp
    int minDisparity = 0;
    int numDisparitiesSGBM = 16 * 5;
    int blockSizeSGBM = 3;

    cv::Ptr<cv::StereoSGBM> stereoSGBM = cv::StereoSGBM::create(minDisparity, numDisparitiesSGBM, blockSizeSGBM);
    stereoSGBM->setP1(8 * imgLeft.channels() * blockSizeSGBM * blockSizeSGBM);
    stereoSGBM->setP2(32 * imgLeft.channels() * blockSizeSGBM * blockSizeSGBM);
    stereoSGBM->setDisp12MaxDiff(1);
    stereoSGBM->setUniquenessRatio(15);
    stereoSGBM->setSpeckleWindowSize(100);
    stereoSGBM->setSpeckleRange(32);
    stereoSGBM->setPreFilterCap(63);
    stereoSGBM->setMode(cv::StereoSGBM::MODE_SGBM);

    cv::Mat disparitySGBM;
    stereoSGBM->compute(imgLeft, imgRight, disparitySGBM);

    // Normalize the disparity map for visualization
    cv::Mat disparitySGBM8U;
    disparitySGBM.convertTo(disparitySGBM8U, CV_8U, 255 / (numDisparitiesSGBM * 16.0));

    cv::imshow("Disparity Map - SGBM", disparitySGBM8U);
    cv::waitKey(0);

    return 0;
}
```

**Explanation of Code**

1. **Header Inclusions**: The necessary OpenCV headers for image processing and stereo vision are included.
2. **Loading Images**: The left and right stereo images are loaded in grayscale.
3. **Preprocessing**: Histogram equalization is applied to enhance image contrast.
4. **StereoBM**:
    - `numDisparities`: The range of disparity values.
    - `blockSize`: The size of the block to match. A larger block size can increase robustness but decrease accuracy.
    - `stereoBM->compute()`: Computes the disparity map.
5. **StereoSGBM**:
    - `minDisparity`: The minimum possible disparity value.
    - `numDisparitiesSGBM`: The number of disparities to search.
    - `blockSizeSGBM`: The size of the block to match.
    - `stereoSGBM->compute()`: Computes the disparity map using a semi-global matching algorithm.

**Choosing Parameters**

The choice of parameters such as `numDisparities`, `blockSize`, `P1`, `P2`, etc., significantly affects the quality of the disparity map. Tuning these parameters according to the specific characteristics of the stereo images is crucial.

**Conclusion**

Stereo matching algorithms are fundamental for extracting depth information from stereo images. By implementing these algorithms using OpenCV, we can effectively compute disparity maps and subsequently derive 3D depth information. The local methods (e.g., StereoBM) provide a good balance between speed and accuracy, while global methods (e.g., StereoSGBM) offer higher accuracy at the cost of computational complexity. Understanding and tuning these algorithms is essential for developing robust stereo vision systems.
