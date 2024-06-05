
\newpage
## Chapter 17: Structure from Motion (SfM)

In the realm of computer vision, Structure from Motion (SfM) is a crucial technique used to reconstruct three-dimensional structures from a series of two-dimensional images. This chapter delves into the principles and methodologies behind SfM, emphasizing its importance in applications ranging from robotics to augmented reality. We will explore two main components: Camera Motion Estimation and 3D Reconstruction from Multiple Views. By understanding these concepts, we gain insights into how dynamic scenes can be interpreted and rendered in three dimensions, enabling machines to perceive and navigate the world in a more sophisticated manner.

### 17.1. Camera Motion Estimation

Camera Motion Estimation is a fundamental step in the Structure from Motion (SfM) pipeline. It involves determining the position and orientation of the camera at each point in time as it captures a series of images. This process is crucial for reconstructing the 3D structure of the scene. In this section, we will discuss the mathematical foundations of camera motion estimation and demonstrate how to implement it using C++ with the OpenCV library.

**Mathematical Background**

The basic idea of camera motion estimation is to find the transformation between different camera poses. This transformation can be represented as a rotation matrix $R$ and a translation vector $t$. The relationship between corresponding points in two views can be described by the essential matrix $E$ or the fundamental matrix $F$.

1. **Essential Matrix**: The essential matrix encapsulates the rotation and translation between two views of a calibrated camera (i.e., the intrinsic parameters are known).
   $$
   \mathbf{x_2}^T \mathbf{E} \mathbf{x_1} = 0
   $$
   where $\mathbf{x_1}$ and $\mathbf{x_2}$ are corresponding points in the first and second image, respectively.

2. **Fundamental Matrix**: The fundamental matrix is used for uncalibrated cameras and relates corresponding points in two images.
   $$
   \mathbf{x_2}^T \mathbf{F} \mathbf{x_1} = 0
   $$

To estimate the camera motion, we generally use the essential matrix, which can be decomposed to obtain the rotation $R$ and translation $t$.

**Implementation Using OpenCV**

Here, we will demonstrate how to estimate camera motion using the OpenCV library in C++. We will follow these steps:

1. Detect and match keypoints between two images.
2. Compute the essential matrix.
3. Decompose the essential matrix to obtain rotation and translation.
4. Verify the motion estimation.

**Step 1: Detect and Match Keypoints**

We will use the ORB (Oriented FAST and Rotated BRIEF) detector to find keypoints and descriptors, and then match them using the BFMatcher (Brute Force Matcher).

```cpp
#include <opencv2/opencv.hpp>

#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Load images
    Mat img1 = imread("image1.jpg", IMREAD_GRAYSCALE);
    Mat img2 = imread("image2.jpg", IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        cerr << "Could not open or find the images!" << endl;
        return -1;
    }

    // Detect ORB keypoints and descriptors
    Ptr<ORB> orb = ORB::create();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    orb->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    // Match descriptors
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Draw matches
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
    imshow("Matches", img_matches);
    waitKey(0);

    return 0;
}
```

**Step 2: Compute the Essential Matrix**

Using the matched keypoints, we compute the essential matrix. This requires the intrinsic parameters of the camera.

```cpp
// Camera intrinsic parameters
Mat K = (Mat_<double>(3, 3) << 718.8560, 0, 607.1928,
                               0, 718.8560, 185.2157,
                               0, 0, 1);

// Convert keypoints to Point2f
vector<Point2f> points1, points2;
for (DMatch m : matches) {
    points1.push_back(keypoints1[m.queryIdx].pt);
    points2.push_back(keypoints2[m.trainIdx].pt);
}

// Compute essential matrix
Mat essential_matrix = findEssentialMat(points1, points2, K, RANSAC, 0.999, 1.0);
```

**Step 3: Decompose the Essential Matrix**

Next, we decompose the essential matrix to obtain the rotation and translation between the two camera views.

```cpp
// Recover pose from essential matrix
Mat R, t;
recoverPose(essential_matrix, points1, points2, K, R, t);

cout << "Rotation Matrix: " << R << endl;
cout << "Translation Vector: " << t << endl;
```

**Step 4: Verify the Motion Estimation**

To verify the correctness of our motion estimation, we can reproject the 3D points back onto the image plane and check for consistency with the original 2D points.

```cpp
// Triangulate points to verify the motion estimation
Mat points4D;
triangulatePoints(R, t, points1, points2, points4D);

// Convert homogeneous coordinates to 3D points
vector<Point3f> points3D;
for (int i = 0; i < points4D.cols; ++i) {
    Mat x = points4D.col(i);
    x /= x.at<float>(3);
    Point3f pt(x.at<float>(0), x.at<float>(1), x.at<float>(2));
    points3D.push_back(pt);
}

// Reproject points and compute error
vector<Point2f> reprojected_points;
projectPoints(points3D, R, t, K, noArray(), reprojected_points);

double error = 0.0;
for (size_t i = 0; i < points1.size(); ++i) {
    error += norm(points1[i] - reprojected_points[i]);
}
error /= points1.size();

cout << "Reprojection error: " << error << endl;
```

**Conclusion**

In this section, we covered the fundamentals of camera motion estimation, including the mathematical background and practical implementation using OpenCV in C++. By following these steps, you can estimate the rotation and translation of a camera as it moves through a scene, which is a crucial component of Structure from Motion. This knowledge lays the foundation for the next step: 3D Reconstruction from Multiple Views.

### 17.2. 3D Reconstruction from Multiple Views

3D reconstruction from multiple views is the process of creating a three-dimensional model of a scene using multiple images taken from different viewpoints. This technique is essential for various applications in computer vision, including robotics, augmented reality, and 3D mapping. In this section, we will delve into the mathematical principles behind 3D reconstruction and demonstrate how to implement it using C++ with the OpenCV library.

**Mathematical Background**

The process of 3D reconstruction from multiple views involves several key steps:

1. **Feature Detection and Matching**: Detecting and matching keypoints across multiple images.
2. **Triangulation**: Using the matched keypoints and the camera poses to compute the 3D coordinates of points in the scene.
3. **Bundle Adjustment**: Refining the 3D points and camera parameters to minimize reprojection error.

**Triangulation**

Triangulation is the process of determining the 3D position of a point given its projections in two or more images. Given the camera matrices $P_1$ and $P_2$ for two views and the corresponding points $\mathbf{x}_1$ and $\mathbf{x}_2$ in these views, we can set up the following system of linear equations:

$$
\mathbf{x}_1 = P_1 \mathbf{X}
$$
$$
\mathbf{x}_2 = P_2 \mathbf{X}
$$

where $\mathbf{X}$ is the homogeneous coordinate of the 3D point. By solving this system, we can find the 3D coordinates of the point $\mathbf{X}$.

**Implementation Using OpenCV**

We will demonstrate the 3D reconstruction process using a series of images. The steps include detecting and matching keypoints, estimating camera motion, triangulating points, and refining the reconstruction with bundle adjustment.

**Step 1: Feature Detection and Matching**

We use ORB to detect and match keypoints, as demonstrated in the previous subchapter.

```cpp
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    // Load images
    Mat img1 = imread("image1.jpg", IMREAD_GRAYSCALE);
    Mat img2 = imread("image2.jpg", IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        cerr << "Could not open or find the images!" << endl;
        return -1;
    }

    // Detect ORB keypoints and descriptors
    Ptr<ORB> orb = ORB::create();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    orb->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    // Match descriptors
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Draw matches
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
    imshow("Matches", img_matches);
    waitKey(0);

    return 0;
}
```

**Step 2: Camera Motion Estimation**

Estimate the essential matrix and decompose it to get the rotation and translation, as previously demonstrated.

```cpp
// Camera intrinsic parameters
Mat K = (Mat_<double>(3, 3) << 718.8560, 0, 607.1928,
                               0, 718.8560, 185.2157,
                               0, 0, 1);

// Convert keypoints to Point2f
vector<Point2f> points1, points2;
for (DMatch m : matches) {
    points1.push_back(keypoints1[m.queryIdx].pt);
    points2.push_back(keypoints2[m.trainIdx].pt);
}

// Compute essential matrix
Mat essential_matrix = findEssentialMat(points1, points2, K, RANSAC, 0.999, 1.0);

// Recover pose from essential matrix
Mat R, t;
recoverPose(essential_matrix, points1, points2, K, R, t);

cout << "Rotation Matrix: " << R << endl;
cout << "Translation Vector: " << t << endl;
```

**Step 3: Triangulation**

Use the recovered pose to triangulate the points.

```cpp
// Triangulate points
Mat points4D;
triangulatePoints(K * Mat::eye(3, 4, CV_64F), K * (Mat_<double>(3, 4) << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0),
                                                                      R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
                                                                      R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2)),
                   points1, points2, points4D);

// Convert homogeneous coordinates to 3D points
vector<Point3f> points3D;
for (int i = 0; i < points4D.cols; ++i) {
    Mat x = points4D.col(i);
    x /= x.at<float>(3);
    Point3f pt(x.at<float>(0), x.at<float>(1), x.at<float>(2));
    points3D.push_back(pt);
}

// Output 3D points
for (const auto& pt : points3D) {
    cout << "Point: " << pt << endl;
}
```

**Step 4: Bundle Adjustment**

Bundle adjustment refines the 3D points and camera parameters to minimize the reprojection error. This is typically done using a library like g2o or Ceres Solver, as OpenCV does not provide a built-in bundle adjustment function.

```cpp
// Bundle Adjustment (using g2o or Ceres Solver)

// This step involves setting up the optimization problem, adding the 3D points and camera parameters,
// defining the cost function, and running the solver to minimize the reprojection error.

#include <ceres/ceres.h>

#include <ceres/rotation.h>

// Define the cost function for bundle adjustment
struct ReprojectionError {
    Point2f observed_point;
    Mat K;

    ReprojectionError(const Point2f& observed_point, const Mat& K) : observed_point(observed_point), K(K) {}

    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const {
        // Camera parameters: [rotation (3), translation (3)]
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Project the point
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        T fx = T(K.at<double>(0, 0));
        T fy = T(K.at<double>(1, 1));
        T cx = T(K.at<double>(0, 2));
        T cy = T(K.at<double>(1, 2));

        T predicted_x = fx * xp + cx;
        T predicted_y = fy * yp + cy;

        // Compute residuals
        residuals[0] = predicted_x - T(observed_point.x);
        residuals[1] = predicted_y - T(observed_point.y);

        return true;
    }
};

int main() {
    // Prepare data for bundle adjustment
    double camera_params[6] = {0, 0, 0, t.at<double>(0), t.at<double>(1), t.at<double>(2)};
    vector<double> points3D_flat;
    for (const auto& pt : points3D) {
        points3D_flat.push_back(pt.x);
        points3D_flat.push_back(pt.y);
        points3D_flat.push_back(pt.z);
    }

    ceres::Problem problem;
    for (size_t i = 0; i < points1.size(); ++i) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
                new ReprojectionError(points1[i], K)),
            nullptr, camera_params, &points3D_flat[3 * i]);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << summary.FullReport() << endl;

    // Output refined 3D points
    for (size_t i = 0; i < points1.size(); ++i) {
        cout << "Refined Point: (" << points3D_flat[3 * i] << ", " << points3D_flat[3 * i + 1] << ", " << points3D_flat[3 * i + 2] << ")" << endl;
    }

    return 0;
}
```

**Conclusion**

In this subchapter, we explored the detailed process of 3D reconstruction from multiple views. We covered the mathematical foundation, including triangulation and bundle adjustment, and provided a comprehensive implementation using C++ and OpenCV. This implementation enables the reconstruction of a 3D model from a series of 2D images, forming a critical part of the Structure from Motion pipeline. This knowledge not only enhances our understanding of 3D scene reconstruction but also equips us with practical tools to apply in real-world applications.
