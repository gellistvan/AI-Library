
\newpage
# Part VII: Motion Analysis

## Chapter 19: Optical Flow

Optical Flow is a fundamental concept in computer vision that involves estimating the motion of objects in a sequence of images or video frames. By understanding and calculating the apparent motion of pixel intensities, we can gain insights into the movement within a scene, enabling various applications such as object tracking, video stabilization, and motion detection. This chapter delves into key methodologies for computing optical flow, focusing on the Lucas-Kanade method, the Horn-Schunck method, and Dense Optical Flow techniques, providing a comprehensive overview of their principles, implementations, and applications in real-world scenarios.

### 19.1. Lucas-Kanade Method

The Lucas-Kanade method is a widely used technique for optical flow estimation, particularly effective for tracking the motion of features between consecutive frames in a video sequence. This method, introduced by Bruce D. Lucas and Takeo Kanade in 1981, assumes that the flow is essentially constant within a small neighborhood of each pixel. This assumption allows us to formulate the optical flow problem as a set of linear equations, making it computationally efficient and robust for real-time applications.

**Mathematical Background**

The Lucas-Kanade method is based on the following assumptions:

1. **Brightness Constancy Assumption**: The intensity of a particular point in the image does not change between frames. Mathematically, this is expressed as:
   $$
   I(x, y, t) = I(x + u, y + v, t + 1)
   $$
   where $I(x, y, t)$ is the intensity of the pixel at $(x, y)$ in the frame at time $t$, and $(u, v)$ is the displacement vector (optical flow) we want to find.

2. **Small Motion Assumption**: The movement between frames is small enough to approximate the changes linearly:
   $$
   I(x + u, y + v, t + 1) \approx I(x, y, t) + \frac{\partial I}{\partial x}u + \frac{\partial I}{\partial y}v + \frac{\partial I}{\partial t}
   $$

From the brightness constancy assumption, we get:
$$
\frac{\partial I}{\partial x}u + \frac{\partial I}{\partial y}v + \frac{\partial I}{\partial t} = 0
$$
This is known as the **optical flow equation**.

To solve for $u$ and $v$, Lucas and Kanade proposed using a local neighborhood of each pixel. If we consider a window of $N$ pixels around the point $(x, y)$, we can write:
$$
A \mathbf{v} = \mathbf{b}
$$
where
$$
A = \begin{bmatrix}
I_{x1} & I_{y1} \\
I_{x2} & I_{y2} \\
\vdots & \vdots \\
I_{xN} & I_{yN}
\end{bmatrix}, \quad
\mathbf{v} = \begin{bmatrix}
u \\
v
\end{bmatrix}, \quad
\mathbf{b} = \begin{bmatrix}
-I_{t1} \\
-I_{t2} \\
\vdots \\
-I_{tN}
\end{bmatrix}
$$

This can be solved using the least squares method:
$$
\mathbf{v} = (A^T A)^{-1} A^T \mathbf{b}
$$

**Implementation in C++ using OpenCV**

OpenCV provides a robust implementation of the Lucas-Kanade method for optical flow, which we can utilize directly. Here's a detailed example demonstrating its usage:

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Read the video file
    VideoCapture cap("video.mp4");
    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    Mat prevFrame, nextFrame, prevGray, nextGray;
    vector<Point2f> prevPts, nextPts;
    vector<uchar> status;
    vector<float> err;

    // Read the first frame
    cap >> prevFrame;
    cvtColor(prevFrame, prevGray, COLOR_BGR2GRAY);

    // Detect corners in the first frame
    goodFeaturesToTrack(prevGray, prevPts, 100, 0.3, 7, Mat(), 7, false, 0.04);

    while (true) {
        // Capture the next frame
        cap >> nextFrame;
        if (nextFrame.empty())
            break;
        cvtColor(nextFrame, nextGray, COLOR_BGR2GRAY);

        // Calculate optical flow using Lucas-Kanade method
        calcOpticalFlowPyrLK(prevGray, nextGray, prevPts, nextPts, status, err);

        // Draw the optical flow vectors
        for (size_t i = 0; i < prevPts.size(); i++) {
            if (status[i]) {
                line(nextFrame, prevPts[i], nextPts[i], Scalar(0, 255, 0), 2);
                circle(nextFrame, nextPts[i], 5, Scalar(0, 255, 0), -1);
            }
        }

        // Display the result
        imshow("Optical Flow", nextFrame);

        // Break the loop on 'q' key press
        if (waitKey(30) == 'q')
            break;

        // Update previous frame and points
        prevGray = nextGray.clone();
        prevPts = nextPts;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
```

**Explanation of the Code**

1. **Reading the Video**: The `VideoCapture` class is used to read the video file. Ensure the path to the video file is correct.
2. **Initialization**: Initialize `Mat` objects for the frames and grayscale images. Vectors for points, status, and error are also initialized.
3. **First Frame Processing**: Read the first frame and convert it to grayscale. Detect good features to track using `goodFeaturesToTrack`.
4. **Optical Flow Calculation**: In a loop, read each frame, convert it to grayscale, and calculate the optical flow using `calcOpticalFlowPyrLK`. This function computes the optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids.
5. **Drawing the Flow Vectors**: For each point, if the status is positive, draw the flow vector and the point.
6. **Display the Result**: Use `imshow` to display the frame with the optical flow vectors.
7. **Updating for Next Iteration**: Update the previous frame and points for the next iteration of the loop.

**Detailed Explanation of `calcOpticalFlowPyrLK`**

The `calcOpticalFlowPyrLK` function in OpenCV uses pyramidal Lucas-Kanade to track the motion of points from one frame to the next. Here are the parameters used in the function call:

- `prevGray`: The previous frame in grayscale.
- `nextGray`: The current frame in grayscale.
- `prevPts`: The points in the previous frame to track.
- `nextPts`: The calculated positions of these points in the current frame.
- `status`: Output status vector (of unsigned chars); each element of the vector is set to 1 if the flow for the corresponding features has been found, otherwise, it is set to 0.
- `err`: Output vector of errors; each element of the vector is set to the error for the corresponding feature.

The pyramidal approach improves the robustness of the Lucas-Kanade method by building a pyramid of images with reduced resolutions and calculating the optical flow at each level of the pyramid.

By using this method, we can effectively track the motion of features between frames, enabling applications such as object tracking, motion detection, and more.

### 19.2. Horn-Schunck Method

The Horn-Schunck method is another influential technique for optical flow estimation, introduced by Berthold K. P. Horn and Brian G. Schunck in 1981. Unlike the Lucas-Kanade method, which assumes constant flow within a neighborhood, the Horn-Schunck method assumes that the optical flow field is smooth over the entire image. This global approach results in a dense optical flow field, where motion is estimated for every pixel in the image.

**Mathematical Background**

The Horn-Schunck method is based on the following assumptions:

1. **Brightness Constancy Assumption**: Similar to the Lucas-Kanade method, it assumes that the intensity of a pixel remains constant over time:
   $$
   I(x, y, t) = I(x + u, y + v, t + 1)
   $$

2. **Smoothness Assumption**: The optical flow varies smoothly over the image. This is enforced by minimizing the gradient of the flow field.

The method formulates the optical flow estimation as an optimization problem. The objective function to be minimized is:
$$
E = \iint \left( \left( \frac{\partial I}{\partial x}u + \frac{\partial I}{\partial y}v + \frac{\partial I}{\partial t} \right)^2 + \alpha^2 \left( |\nabla u|^2 + |\nabla v|^2 \right) \right) dx dy
$$
where $\alpha$ is a regularization parameter that controls the trade-off between the data fidelity term and the smoothness term.

This leads to the following set of equations for the flow components $u$ and $v$:
$$
\frac{\partial I}{\partial x} \left( \frac{\partial I}{\partial x}u + \frac{\partial I}{\partial y}v + \frac{\partial I}{\partial t} \right) - \alpha^2 \nabla^2 u = 0
$$
$$
\frac{\partial I}{\partial y} \left( \frac{\partial I}{\partial x}u + \frac{\partial I}{\partial y}v + \frac{\partial I}{\partial t} \right) - \alpha^2 \nabla^2 v = 0
$$
where $\nabla^2$ denotes the Laplacian operator, which approximates the smoothness term.

**Implementation in C++ using OpenCV**

While OpenCV does not provide a direct implementation of the Horn-Schunck method, it can be implemented using C++. Here's a detailed example of how to implement the Horn-Schunck method:

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

void computeGradients(const Mat& img, Mat& Ix, Mat& Iy, Mat& It, const Mat& prevImg) {
    // Compute the image gradients Ix, Iy, and It
    Sobel(img, Ix, CV_32F, 1, 0, 3);
    Sobel(img, Iy, CV_32F, 0, 1, 3);
    It = img - prevImg;
}

void hornSchunck(const Mat& prevImg, const Mat& nextImg, Mat& u, Mat& v, float alpha, int iterations) {
    Mat Ix, Iy, It;
    computeGradients(nextImg, Ix, Iy, It, prevImg);

    u = Mat::zeros(prevImg.size(), CV_32F);
    v = Mat::zeros(prevImg.size(), CV_32F);

    Mat u_avg, v_avg;
    for (int k = 0; k < iterations; ++k) {
        blur(u, u_avg, Size(3, 3));
        blur(v, v_avg, Size(3, 3));

        for (int y = 0; y < prevImg.rows; ++y) {
            for (int x = 0; x < prevImg.cols; ++x) {
                float Ix_val = Ix.at<float>(y, x);
                float Iy_val = Iy.at<float>(y, x);
                float It_val = It.at<float>(y, x);

                float denominator = alpha * alpha + Ix_val * Ix_val + Iy_val * Iy_val;
                float numerator = Ix_val * u_avg.at<float>(y, x) + Iy_val * v_avg.at<float>(y, x) + It_val;

                u.at<float>(y, x) = u_avg.at<float>(y, x) - Ix_val * numerator / denominator;
                v.at<float>(y, x) = v_avg.at<float>(y, x) - Iy_val * numerator / denominator;
            }
        }
    }
}

int main() {
    // Read the video file
    VideoCapture cap("video.mp4");
    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    Mat prevFrame, nextFrame, prevGray, nextGray, u, v;

    // Read the first frame
    cap >> prevFrame;
    cvtColor(prevFrame, prevGray, COLOR_BGR2GRAY);

    while (true) {
        // Capture the next frame
        cap >> nextFrame;
        if (nextFrame.empty())
            break;
        cvtColor(nextFrame, nextGray, COLOR_BGR2GRAY);

        // Calculate optical flow using Horn-Schunck method
        hornSchunck(prevGray, nextGray, u, v, 0.001, 100);

        // Visualize the optical flow vectors
        Mat flow(nextGray.size(), CV_8UC3, Scalar(0, 0, 0));
        for (int y = 0; y < nextGray.rows; y += 10) {
            for (int x = 0; x < nextGray.cols; x += 10) {
                Point2f flow_at_point(u.at<float>(y, x), v.at<float>(y, x));
                line(flow, Point(x, y), Point(cvRound(x + flow_at_point.x), cvRound(y + flow_at_point.y)), Scalar(0, 255, 0));
                circle(flow, Point(x, y), 1, Scalar(0, 255, 0), -1);
            }
        }

        // Display the result
        imshow("Optical Flow - Horn-Schunck", flow);

        // Break the loop on 'q' key press
        if (waitKey(30) == 'q')
            break;

        // Update previous frame
        prevGray = nextGray.clone();
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
```

**Explanation of the Code**

1. **Reading the Video**: Similar to the Lucas-Kanade method, we use `VideoCapture` to read the video file.
2. **Initialization**: Initialize `Mat` objects for the frames, grayscale images, and the optical flow components $u$ and $v$.
3. **Gradient Computation**: The `computeGradients` function computes the spatial gradients ($I_x$, $I_y$) using the Sobel operator and the temporal gradient ($I_t$).
4. **Horn-Schunck Implementation**: The `hornSchunck` function implements the iterative Horn-Schunck method. It starts by initializing the flow fields $u$ and $v$ to zero. For a specified number of iterations, it computes the local averages of $u$ and $v$ using a box filter (`blur`). The flow updates are then calculated based on the Horn-Schunck equations.
5. **Flow Visualization**: The flow vectors are drawn on the image for visualization. The optical flow is displayed using `imshow`.
6. **Updating for Next Iteration**: Update the previous frame for the next iteration of the loop.

**Detailed Explanation of Key Steps**

1. **Computing Gradients**:
   The `computeGradients` function calculates the spatial and temporal gradients required for the Horn-Schunck method:
   ```cpp
   void computeGradients(const Mat& img, Mat& Ix, Mat& Iy, Mat& It, const Mat& prevImg) {
       Sobel(img, Ix, CV_32F, 1, 0, 3);
       Sobel(img, Iy, CV_32F, 0, 1, 3);
       It = img - prevImg;
   }
   ```

2. **Horn-Schunck Iterative Updates**:
   The main loop in the `hornSchunck` function performs the iterative updates:
   ```cpp
   void hornSchunck(const Mat& prevImg, const Mat& nextImg, Mat& u, Mat& v, float alpha, int iterations) {
       Mat Ix, Iy, It;
       computeGradients(nextImg, Ix, Iy, It, prevImg);

       u = Mat::zeros(prevImg.size(), CV_32F);
       v = Mat::zeros(prevImg.size(), CV_32F);

       Mat u_avg, v_avg;
       for (int k = 0; k < iterations; ++k) {
           blur(u, u_avg, Size(3, 3));
           blur(v, v_avg, Size(3, 3));

           for (int y = 0; y < prevImg.rows; ++y) {
               for (int x = 0; x < prevImg.cols; ++x) {
                   float Ix_val = Ix.at<float>(y, x);
                   float Iy_val = Iy.at<float>(y, x);
                   float It_val = It.at<float>(y, x);

                   float denominator = alpha * alpha + Ix_val * Ix_val + Iy_val * Iy_val;
                   float numerator = Ix_val * u_avg.at<float>(y, x) + Iy_val * v_avg.at<float>(y, x) + It_val;

                   u.at<float>(y, x) = u_avg.at<float>(y, x) - Ix_val * numerator / denominator;
                   v.at<float>(y, x) = v_avg.at<float>(y, x) - Iy_val * numerator / denominator;
               }
           }
       }
   }
   ```

By implementing the Horn-Schunck method, we achieve a dense optical flow estimation, which provides the motion vector for each pixel in the image. This method is particularly useful for applications requiring a comprehensive motion analysis over the entire frame.

### 19.3. Dense Optical Flow Techniques

Dense optical flow techniques estimate the motion vector for every pixel in a sequence of images, providing a detailed motion field that is crucial for applications such as video stabilization, object tracking, and motion segmentation. Unlike sparse optical flow methods that track a few key points, dense optical flow techniques offer a comprehensive motion representation of the entire scene.

Several dense optical flow algorithms have been developed over the years, including variations of the Lucas-Kanade and Horn-Schunck methods. In this subchapter, we will focus on two popular dense optical flow methods available in OpenCV: Farneback's method and the Dual TV-L1 method.

**Mathematical Background**

Dense optical flow algorithms generally build upon the principles of brightness constancy and smoothness constraints, as discussed in the Lucas-Kanade and Horn-Schunck methods. However, they often incorporate additional techniques to handle complex motion and improve robustness against noise and illumination changes.

**Farneback's Method:**
Introduced by Gunnar Farneback in 2003, this method estimates the optical flow using polynomial expansion. The idea is to approximate the neighborhood of each pixel with a polynomial function, allowing the calculation of motion between frames.

**Dual TV-L1 Method:**
This method combines Total Variation (TV) regularization with an L1 norm data term, providing robustness against outliers and preserving sharp motion boundaries. It solves the optical flow problem using a primal-dual approach.

**Implementation in C++ using OpenCV**

OpenCV provides efficient implementations of both Farneback's and Dual TV-L1 methods. Here, we will demonstrate their usage with detailed examples.

**Farneback's Method**

Farneback's method approximates the local neighborhood of each pixel using polynomial expansions. The algorithm estimates the flow by finding the displacement that minimizes the difference between the polynomial approximations of two consecutive frames.

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Read the video file
    VideoCapture cap("video.mp4");
    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    Mat prevFrame, nextFrame, prevGray, nextGray, flow;

    // Read the first frame
    cap >> prevFrame;
    cvtColor(prevFrame, prevGray, COLOR_BGR2GRAY);

    while (true) {
        // Capture the next frame
        cap >> nextFrame;
        if (nextFrame.empty())
            break;
        cvtColor(nextFrame, nextGray, COLOR_BGR2GRAY);

        // Calculate dense optical flow using Farneback's method
        calcOpticalFlowFarneback(prevGray, nextGray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        // Visualize the optical flow vectors
        Mat flowImg(prevGray.size(), CV_8UC3);
        for (int y = 0; y < flow.rows; y += 10) {
            for (int x = 0; x < flow.cols; x += 10) {
                const Point2f flow_at_point = flow.at<Point2f>(y, x);
                line(flowImg, Point(x, y), Point(cvRound(x + flow_at_point.x), cvRound(y + flow_at_point.y)), Scalar(0, 255, 0));
                circle(flowImg, Point(x, y), 1, Scalar(0, 255, 0), -1);
            }
        }

        // Display the result
        imshow("Optical Flow - Farneback", flowImg);

        // Break the loop on 'q' key press
        if (waitKey(30) == 'q')
            break;

        // Update previous frame
        prevGray = nextGray.clone();
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
```

**Explanation of the Code**

1. **Reading the Video**: The `VideoCapture` class reads the video file.
2. **Initialization**: Initialize `Mat` objects for the frames, grayscale images, and the flow field.
3. **First Frame Processing**: Read the first frame and convert it to grayscale.
4. **Dense Optical Flow Calculation**: In a loop, read each frame, convert it to grayscale, and calculate the dense optical flow using `calcOpticalFlowFarneback`. This function computes the flow field using polynomial expansion.
5. **Flow Visualization**: For each point, draw the flow vector and point on the image.
6. **Display the Result**: Use `imshow` to display the frame with the optical flow vectors.
7. **Updating for Next Iteration**: Update the previous frame for the next iteration of the loop.

**Dual TV-L1 Method**

The Dual TV-L1 method combines the advantages of Total Variation regularization and L1 norm data fidelity, resulting in a robust optical flow estimation that preserves sharp edges and handles illumination changes effectively.

```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>
#include <iostream>

using namespace cv;
using namespace cv::optflow;
using namespace std;

int main() {
    // Read the video file
    VideoCapture cap("video.mp4");
    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    Mat prevFrame, nextFrame, prevGray, nextGray, flow;

    // Read the first frame
    cap >> prevFrame;
    cvtColor(prevFrame, prevGray, COLOR_BGR2GRAY);

    Ptr<DenseOpticalFlow> tvl1 = createOptFlow_DualTVL1();

    while (true) {
        // Capture the next frame
        cap >> nextFrame;
        if (nextFrame.empty())
            break;
        cvtColor(nextFrame, nextGray, COLOR_BGR2GRAY);

        // Calculate dense optical flow using Dual TV-L1 method
        tvl1->calc(prevGray, nextGray, flow);

        // Visualize the optical flow vectors
        Mat flowImg(prevGray.size(), CV_8UC3);
        for (int y = 0; y < flow.rows; y += 10) {
            for (int x = 0; x < flow.cols; x += 10) {
                const Point2f flow_at_point = flow.at<Point2f>(y, x);
                line(flowImg, Point(x, y), Point(cvRound(x + flow_at_point.x), cvRound(y + flow_at_point.y)), Scalar(0, 255, 0));
                circle(flowImg, Point(x, y), 1, Scalar(0, 255, 0), -1);
            }
        }

        // Display the result
        imshow("Optical Flow - Dual TV-L1", flowImg);

        // Break the loop on 'q' key press
        if (waitKey(30) == 'q')
            break;

        // Update previous frame
        prevGray = nextGray.clone();
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
```

**Explanation of the Code**

1. **Reading the Video**: The `VideoCapture` class reads the video file.
2. **Initialization**: Initialize `Mat` objects for the frames, grayscale images, and the flow field.
3. **First Frame Processing**: Read the first frame and convert it to grayscale.
4. **Dual TV-L1 Optical Flow Calculation**: In a loop, read each frame, convert it to grayscale, and calculate the dense optical flow using the Dual TV-L1 method. This method is created using `createOptFlow_DualTVL1` and called with `calc`.
5. **Flow Visualization**: For each point, draw the flow vector and point on the image.
6. **Display the Result**: Use `imshow` to display the frame with the optical flow vectors.
7. **Updating for Next Iteration**: Update the previous frame for the next iteration of the loop.

**Detailed Explanation of Key Steps**

1. **Farneback's Method**:
   The `calcOpticalFlowFarneback` function computes the dense optical flow using polynomial expansions. Key parameters include:
    - `pyr_scale`: Parameter specifying the image scale (<1) to build pyramids for each image.
    - `levels`: Number of pyramid layers.
    - `winsize`: Averaging window size.
    - `iterations`: Number of iterations at each pyramid level.
    - `poly_n`: Size of the pixel neighborhood used to find polynomial expansion.
    - `poly_sigma`: Standard deviation of the Gaussian used to smooth derivatives.
    - `flags`: Operation flags.

2. **Dual TV-L1 Method**:
   The Dual TV-L1 method is created using `createOptFlow_DualTVL1`, which returns a pointer to a `DenseOpticalFlow` object. This method is robust and handles illumination changes effectively.

By implementing these dense optical flow techniques, we obtain detailed motion fields that provide valuable information for various computer vision applications. These methods enhance the capability to analyze and interpret dynamic scenes in videos.
