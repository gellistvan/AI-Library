
\newpage
# Part V: Object Detection and Recognition

## Chapter 13: Classical Object Detection

In the realm of computer vision, object detection is a critical task that involves identifying and locating objects within an image. Before the advent of deep learning, classical object detection methods laid the groundwork for many of the advanced techniques we see today. This chapter delves into these foundational approaches, focusing on the Sliding Window approach and the use of Histogram of Oriented Gradients (HOG) features combined with Support Vector Machines (SVM). These techniques, though overshadowed by modern deep learning models, remain essential for understanding the evolution of object detection methodologies.

### 13.1. Sliding Window Approach

The Sliding Window approach is one of the foundational techniques in classical object detection. This method involves moving a fixed-size window across an image and applying a classifier to each sub-window to determine whether it contains the object of interest. Despite its simplicity, this method provides a solid introduction to the challenges and methodologies of object detection.

**Mathematical Background**

The Sliding Window approach is conceptually straightforward but computationally intensive. Given an image $I$ of size $W \times H$ and a window of size $w \times h$, the method involves:

1. Scanning the image from the top-left corner to the bottom-right corner.
2. At each step, extracting a sub-image (window) of size $w \times h$.
3. Applying a classifier to determine whether the object is present in the current window.
4. Sliding the window by a certain stride (step size) and repeating the process.

The number of windows $N$ that need to be evaluated is given by:
$$ N = \left( \frac{W - w}{s} + 1 \right) \times \left( \frac{H - h}{s} + 1 \right) $$
where $s$ is the stride length.

**Implementation in C++ Using OpenCV**

OpenCV is a powerful library for computer vision that provides various tools and functions to facilitate the implementation of the Sliding Window approach. Below is a detailed implementation of this method in C++ using OpenCV.

```cpp
#include <opencv2/opencv.hpp>

#include <iostream>

using namespace cv;
using namespace std;

// Function to perform sliding window
void slidingWindow(Mat& image, Size windowSize, int stride) {
    for (int y = 0; y <= image.rows - windowSize.height; y += stride) {
        for (int x = 0; x <= image.cols - windowSize.width; x += stride) {
            // Extract the current window
            Rect windowRect(x, y, windowSize.width, windowSize.height);
            Mat window = image(windowRect);

            // Here, you would apply your classifier to the 'window' Mat
            // For demonstration, we'll just draw a rectangle around each window
            rectangle(image, windowRect, Scalar(0, 255, 0), 2);

            // Example of printing the coordinates of the current window
            cout << "Window: (" << x << ", " << y << "), Size: " << windowSize << endl;
        }
    }
}

int main() {
    // Load the image
    Mat image = imread("path_to_your_image.jpg");

    if (image.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // Define window size and stride
    Size windowSize(64, 128); // Example window size
    int stride = 32; // Example stride

    // Perform sliding window
    slidingWindow(image, windowSize, stride);

    // Display the result
    namedWindow("Sliding Window", WINDOW_NORMAL);
    imshow("Sliding Window", image);
    waitKey(0);

    return 0;
}
```

**Explanation of the Code**

1. **Header Inclusions**: The necessary headers from the OpenCV library are included.
2. **slidingWindow Function**: This function takes an image, a window size, and a stride length as input. It iterates over the image using nested loops to slide the window across the image. For each window position, it extracts the sub-image (window) and currently, it simply draws a rectangle around it. In a practical scenario, this is where you would apply your object classifier to detect the presence of the target object.
3. **Main Function**: The main function loads an image, defines the window size and stride, and then calls the `slidingWindow` function. It finally displays the resulting image with all the windows marked.

**Computational Considerations**

The Sliding Window approach can be computationally expensive, especially for large images and small window sizes. The total number of windows evaluated can be very high, leading to a significant computational load. Optimization strategies include:

1. **Reducing the Number of Windows**: Increase the stride length, though this may reduce detection accuracy.
2. **Multi-Scale Detection**: Apply the Sliding Window approach at multiple scales to detect objects of different sizes.
3. **Integral Images**: Use integral images to quickly compute features over any rectangular region.

Despite these optimizations, the Sliding Window approach has largely been replaced by more efficient methods in modern computer vision, such as Convolutional Neural Networks (CNNs), but it remains a valuable learning tool for understanding the basics of object detection.

### 13.2. HOG Features and SVM

Histogram of Oriented Gradients (HOG) is a feature descriptor used in computer vision and image processing for object detection. When combined with a Support Vector Machine (SVM), a powerful classifier, it forms a robust method for detecting objects in images. This approach has been particularly successful in pedestrian detection and other similar tasks.

**Mathematical Background**

**HOG Features**

The HOG feature descriptor works by dividing the image into small connected regions called cells, and for each cell, computing a histogram of gradient directions or edge orientations. The steps are as follows:

1. **Gradient Calculation**:
    - Compute the gradient of the image using finite difference filters.
    - For an image $I$, the gradients along the x and y axes ($G_x$ and $G_y$) are calculated as:
      $$
      G_x = I * \begin{bmatrix} -1 & 0 & 1 \end{bmatrix}, \quad G_y = I * \begin{bmatrix} -1 \\ 0 \\ 1 \end{bmatrix}
      $$

2. **Orientation Binning**:
    - For each pixel, the gradient magnitude $m$ and orientation $\theta$ are calculated as:
      $$
      m = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan{\frac{G_y}{G_x}}
      $$
    - The image is divided into cells, and a histogram of gradient directions is computed for each cell.

3. **Block Normalization**:
    - Cells are grouped into larger blocks, and the histograms within each block are normalized to reduce the effect of illumination changes.

**Support Vector Machine (SVM)**

SVM is a supervised learning model used for classification and regression. For object detection, a linear SVM is typically used. The main idea is to find a hyperplane that best separates the data into different classes. Given a set of training examples $(x_i, y_i)$, where $x_i$ is the feature vector and $y_i$ is the label, the SVM aims to solve:
$$
\min \left\{ \frac{1}{2} \| w \|^2 \right\} \text{ subject to } y_i(w \cdot x_i + b) \geq 1, \forall i
$$
where $w$ is the weight vector and $b$ is the bias term.

**Implementation in C++ Using OpenCV**

OpenCV provides built-in functions for computing HOG features and training an SVM classifier. Below is a detailed implementation of this approach in C++ using OpenCV.

```cpp
#include <opencv2/opencv.hpp>

#include <opencv2/ml.hpp>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

// Function to compute HOG descriptors
void computeHOG(Mat& image, vector<float>& descriptors) {
    HOGDescriptor hog(
        Size(64, 128), // winSize
        Size(16, 16),  // blockSize
        Size(8, 8),    // blockStride
        Size(8, 8),    // cellSize
        9              // nbins
    );

    // Compute HOG descriptors for the image
    hog.compute(image, descriptors);
}

int main() {
    // Load the training images and their labels
    vector<Mat> positiveImages; // Images containing the object
    vector<Mat> negativeImages; // Images not containing the object

    // Load images into the vectors (you would add your own images here)
    // Example:
    // positiveImages.push_back(imread("path_to_positive_image.jpg", IMREAD_GRAYSCALE));
    // negativeImages.push_back(imread("path_to_negative_image.jpg", IMREAD_GRAYSCALE));

    vector<Mat> trainingImages;
    vector<int> labels;

    // Assign labels: 1 for positive, -1 for negative
    for (const auto& img : positiveImages) {
        trainingImages.push_back(img);
        labels.push_back(1);
    }
    for (const auto& img : negativeImages) {
        trainingImages.push_back(img);
        labels.push_back(-1);
    }

    // Compute HOG descriptors for all training images
    vector<vector<float>> hogDescriptors;
    for (const auto& img : trainingImages) {
        vector<float> descriptors;
        computeHOG(img, descriptors);
        hogDescriptors.push_back(descriptors);
    }

    // Convert HOG descriptors to a format suitable for SVM training
    int descriptorSize = hogDescriptors[0].size();
    Mat trainingData(static_cast<int>(hogDescriptors.size()), descriptorSize, CV_32F);
    for (size_t i = 0; i < hogDescriptors.size(); ++i) {
        for (int j = 0; j < descriptorSize; ++j) {
            trainingData.at<float>(static_cast<int>(i), j) = hogDescriptors[i][j];
        }
    }

    Mat labelsMat(labels.size(), 1, CV_32SC1, labels.data());

    // Train the SVM
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainingData, ROW_SAMPLE, labelsMat);

    // Save the trained SVM model
    svm->save("hog_svm_model.xml");

    // Load an image to test the SVM
    Mat testImage = imread("path_to_test_image.jpg", IMREAD_GRAYSCALE);

    // Compute HOG descriptors for the test image
    vector<float> testDescriptors;
    computeHOG(testImage, testDescriptors);

    // Convert test descriptors to Mat
    Mat testDescriptorMat(1, descriptorSize, CV_32F);
    for (int i = 0; i < descriptorSize; ++i) {
        testDescriptorMat.at<float>(0, i) = testDescriptors[i];
    }

    // Predict using the trained SVM
    float response = svm->predict(testDescriptorMat);

    // Output the prediction result
    if (response == 1) {
        cout << "Object detected!" << endl;
    } else {
        cout << "Object not detected." << endl;
    }

    return 0;
}
```

**Explanation of the Code**

1. **Header Inclusions**: The necessary headers from the OpenCV library are included.

2. **computeHOG Function**: This function computes the HOG descriptors for a given image using the `HOGDescriptor` class provided by OpenCV. The parameters of the HOG descriptor (window size, block size, block stride, cell size, and number of bins) are set according to the commonly used values for pedestrian detection.

3. **Main Function**:
    - **Load Training Images and Labels**: Positive and negative training images are loaded, and their labels are assigned. In practice, you would load your own dataset here.
    - **Compute HOG Descriptors**: HOG descriptors are computed for each training image using the `computeHOG` function.
    - **Prepare Training Data**: The computed HOG descriptors are converted into a `Mat` format suitable for SVM training.
    - **Train the SVM**: A linear SVM is created and trained using the HOG descriptors and corresponding labels.
    - **Save the Trained Model**: The trained SVM model is saved to a file for later use.
    - **Load Test Image and Predict**: A test image is loaded, and its HOG descriptors are computed. These descriptors are then used to predict the presence of the object using the trained SVM model.

**Computational Considerations**

While HOG + SVM is more efficient than the Sliding Window approach alone, it still requires careful parameter tuning and a substantial amount of training data to achieve high accuracy. The combination of HOG features with a linear SVM remains a robust and interpretable approach for object detection in various scenarios.

This method forms a bridge between classical techniques and modern deep learning methods, offering a valuable perspective on the evolution of object detection algorithms in computer vision.
