
\newpage
## Chapter 11: Clustering-Based Segmentation

In this chapter, we delve into clustering-based segmentation, a fundamental technique in computer vision that groups similar pixels or regions in an image to simplify its analysis. By leveraging clustering algorithms, we can partition an image into meaningful segments, which are crucial for various applications such as object detection, image recognition, and medical imaging. We will explore three primary methods of clustering-based segmentation: K-means Clustering, Mean Shift, and Graph-Based Segmentation. Each technique offers unique advantages and challenges, providing a comprehensive toolkit for effective image segmentation.

**Subchapters:**
- **K-means Clustering**: An introduction to the popular iterative method that partitions an image into K clusters by minimizing the variance within each cluster.
- **Mean Shift**: A mode-seeking algorithm that does not require the number of clusters to be specified a priori and is adept at identifying arbitrarily shaped clusters.
- **Graph-Based Segmentation**: A method that models an image as a graph and uses graph-cut techniques to segment the image into distinct regions based on their relationships.

### 11.1. K-means Clustering

K-means clustering is one of the most widely used algorithms for image segmentation. This method aims to partition an image into K clusters by minimizing the sum of squared distances between the pixels and the corresponding cluster centroids. It’s an iterative algorithm that alternates between assigning pixels to clusters and updating the cluster centroids until convergence.

**Mathematical Background**

The K-means clustering algorithm involves the following steps:

1. **Initialization**: Select K initial cluster centroids.
2. **Assignment Step**: Assign each pixel to the nearest centroid based on the Euclidean distance.
3. **Update Step**: Recalculate the centroids as the mean of all pixels assigned to each cluster.
4. **Convergence Check**: Repeat the assignment and update steps until the centroids no longer change significantly or a maximum number of iterations is reached.

Mathematically, given a set of data points $\mathbf{X} = \{x_1, x_2, \ldots, x_n\}$, the goal is to partition these into K clusters $\mathbf{C} = \{C_1, C_2, \ldots, C_K\}$ such that the sum of squared distances from each point to the centroid of its assigned cluster is minimized:

$$ \arg\min_{\mathbf{C}} \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2 $$

where $\mu_k$ is the centroid of cluster $C_k$.

**Implementation in C++ using OpenCV**

OpenCV provides a convenient implementation of the K-means algorithm, but we will also look into a custom implementation for better understanding.

**Using OpenCV**

First, let's look at how to perform K-means clustering using OpenCV:

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Load the image
    cv::Mat img = cv::imread("path_to_image.jpg");
    if (img.empty()) {
        std::cerr << "Error: Image cannot be loaded!" << std::endl;
        return -1;
    }

    // Convert the image from BGR to Lab color space
    cv::Mat imgLab;
    cv::cvtColor(img, imgLab, cv::COLOR_BGR2Lab);

    // Reshape the image into a 2D array of Lab pixels
    cv::Mat imgLabReshaped = imgLab.reshape(1, imgLab.rows * imgLab.cols);

    // Convert to float for k-means
    cv::Mat imgLabReshapedFloat;
    imgLabReshaped.convertTo(imgLabReshapedFloat, CV_32F);

    // Perform k-means clustering
    int K = 3;
    cv::Mat labels;
    cv::Mat centers;
    cv::kmeans(imgLabReshapedFloat, K, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);

    // Replace pixel values with their center values
    cv::Mat segmentedImg(imgLab.size(), imgLab.type());
    for (int i = 0; i < imgLabReshapedFloat.rows; ++i) {
        int clusterIdx = labels.at<int>(i);
        segmentedImg.at<cv::Vec3b>(i / imgLab.cols, i % imgLab.cols) = centers.at<cv::Vec3f>(clusterIdx);
    }

    // Convert back to BGR color space
    cv::Mat segmentedImgBGR;
    cv::cvtColor(segmentedImg, segmentedImgBGR, cv::COLOR_Lab2BGR);

    // Show the result
    cv::imshow("Segmented Image", segmentedImgBGR);
    cv::waitKey(0);

    return 0;
}
```

**Custom Implementation in C++**

For a deeper understanding, here’s how you can implement K-means clustering from scratch in C++:

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

using namespace cv;
using namespace std;

class KMeans {
public:
    KMeans(int K, int maxIterations) : K(K), maxIterations(maxIterations) {}

    void fit(const Mat& data) {
        int nSamples = data.rows;
        int nFeatures = data.cols;

        // Initialize cluster centers randomly
        centers = Mat::zeros(K, nFeatures, CV_32F);
        for (int i = 0; i < K; ++i) {
            data.row(rand() % nSamples).copyTo(centers.row(i));
        }

        labels = Mat::zeros(nSamples, 1, CV_32S);
        Mat newCenters = Mat::zeros(K, nFeatures, CV_32F);
        vector<int> counts(K, 0);

        for (int iter = 0; iter < maxIterations; ++iter) {
            // Assignment step
            for (int i = 0; i < nSamples; ++i) {
                float minDist = numeric_limits<float>::max();
                int bestCluster = 0;
                for (int j = 0; j < K; ++j) {
                    float dist = norm(data.row(i) - centers.row(j));
                    if (dist < minDist) {
                        minDist = dist;
                        bestCluster = j;
                    }
                }
                labels.at<int>(i) = bestCluster;
                newCenters.row(bestCluster) += data.row(i);
                counts[bestCluster]++;
            }

            // Update step
            for (int j = 0; j < K; ++j) {
                if (counts[j] != 0) {
                    newCenters.row(j) /= counts[j];
                }
                else {
                    data.row(rand() % nSamples).copyTo(newCenters.row(j));
                }
            }

            // Check for convergence
            if (norm(newCenters - centers) < 1e-4) {
                break;
            }

            newCenters.copyTo(centers);
            newCenters = Mat::zeros(K, nFeatures, CV_32F);
            fill(counts.begin(), counts.end(), 0);
        }
    }

    Mat predict(const Mat& data) {
        int nSamples = data.rows;
        Mat resultLabels = Mat::zeros(nSamples, 1, CV_32S);
        for (int i = 0; i < nSamples; ++i) {
            float minDist = numeric_limits<float>::max();
            int bestCluster = 0;
            for (int j = 0; j < K; ++j) {
                float dist = norm(data.row(i) - centers.row(j));
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = j;
                }
            }
            resultLabels.at<int>(i) = bestCluster;
        }
        return resultLabels;
    }

    Mat getCenters() {
        return centers;
    }

private:
    int K;
    int maxIterations;
    Mat centers;
    Mat labels;
};

int main() {
    // Load the image
    Mat img = imread("path_to_image.jpg");
    if (img.empty()) {
        cerr << "Error: Image cannot be loaded!" << endl;
        return -1;
    }

    // Convert the image from BGR to Lab color space
    Mat imgLab;
    cvtColor(img, imgLab, COLOR_BGR2Lab);

    // Reshape the image into a 2D array of Lab pixels
    Mat imgLabReshaped = imgLab.reshape(1, imgLab.rows * imgLab.cols);

    // Convert to float for k-means
    Mat imgLabReshapedFloat;
    imgLabReshaped.convertTo(imgLabReshapedFloat, CV_32F);

    // Perform custom k-means clustering
    int K = 3;
    int maxIterations = 100;
    KMeans kmeans(K, maxIterations);
    kmeans.fit(imgLabReshapedFloat);
    Mat labels = kmeans.predict(imgLabReshapedFloat);
    Mat centers = kmeans.getCenters();

    // Replace pixel values with their center values
    Mat segmentedImg(imgLab.size(), imgLab.type());
    for (int i = 0; i < imgLabReshapedFloat.rows; ++i) {
        int clusterIdx = labels.at<int>(i);
        segmentedImg.at<Vec3b>(i / imgLab.cols, i % imgLab.cols) = centers.at<Vec3f>(clusterIdx);
    }

    // Convert back to BGR color space
    Mat segmentedImgBGR;
    cvtColor(segmentedImg, segmentedImgBGR, COLOR_Lab2BGR);

    // Show the result
    imshow("Segmented Image", segmentedImgBGR);
    waitKey(0);

    return 0;
}
```

In the custom implementation:
- **Initialization**: Cluster centers are initialized randomly from the data points.
- **Assignment Step**: Each pixel is assigned to the nearest cluster center.
- **Update Step**: New cluster centers are computed as the mean of the assigned points.
- **Convergence Check**: The algorithm checks if the change in cluster centers is below a threshold to stop the iterations.

This detailed explanation and implementation should give you a robust understanding of how K-means clustering works and how it can be applied to image segmentation in C++.

### 11.2. Mean Shift

Mean shift is a non-parametric clustering technique that does not require prior knowledge of the number of clusters. It is a mode-seeking algorithm that iteratively shifts data points towards regions of higher density until convergence. This property makes it particularly useful for image segmentation, where clusters correspond to dense regions of pixels in the feature space.

**Mathematical Background**

The mean shift algorithm aims to find the modes of a density function given a set of data points. The key idea is to iteratively move each point towards the average of points in its neighborhood, defined by a kernel function. The process involves the following steps:

1. **Kernel Density Estimation**: For each data point $x_i$, compute the weighted average of all data points within a window (kernel) centered at $x_i$.
2. **Mean Shift Vector**: The mean shift vector $m(x)$ is the difference between the weighted mean of the neighborhood and the data point $x_i$:
   $$
   m(x) = \frac{\sum_{x_j \in N(x)} K(x_j - x) x_j}{\sum_{x_j \in N(x)} K(x_j - x)} - x
   $$
   where $K(x_j - x)$ is the kernel function and $N(x)$ is the neighborhood of $x$.
3. **Update Step**: Update each data point by moving it in the direction of the mean shift vector:
   $$
   x \leftarrow x + m(x)
   $$
4. **Convergence Check**: Repeat the process until the points converge to the modes of the density function.

The Gaussian kernel is commonly used for the kernel function:
$$
K(x) = \exp \left( -\frac{\|x\|^2}{2h^2} \right)
$$
where $h$ is the bandwidth parameter controlling the size of the window.

**Implementation in C++ using OpenCV**

OpenCV provides an implementation of the mean shift algorithm, but we will also explore a custom implementation to gain a deeper understanding.

**Using OpenCV**

First, let's see how to perform mean shift segmentation using OpenCV:

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Load the image
    cv::Mat img = cv::imread("path_to_image.jpg");
    if (img.empty()) {
        std::cerr << "Error: Image cannot be loaded!" << std::endl;
        return -1;
    }

    // Convert the image to the CIELab color space
    cv::Mat imgLab;
    cv::cvtColor(img, imgLab, cv::COLOR_BGR2Lab);

    // Perform mean shift filtering
    cv::Mat imgLabFiltered;
    cv::pyrMeanShiftFiltering(imgLab, imgLabFiltered, 21, 51);

    // Convert the filtered image back to BGR color space
    cv::Mat segmentedImg;
    cv::cvtColor(imgLabFiltered, segmentedImg, cv::COLOR_Lab2BGR);

    // Show the result
    cv::imshow("Segmented Image", segmentedImg);
    cv::waitKey(0);

    return 0;
}
```

**Custom Implementation in C++**

Now, let's look at a custom implementation of the mean shift algorithm:

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// Gaussian kernel function
float gaussianKernel(const Vec3f& x, const Vec3f& y, float bandwidth) {
    return exp(-norm(x - y) / (2 * bandwidth * bandwidth));
}

void meanShift(Mat& img, Mat& result, float bandwidth, int maxIter) {
    int rows = img.rows;
    int cols = img.cols;

    result = img.clone();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            Vec3f point = img.at<Vec3f>(i, j);
            Vec3f shift;
            int iter = 0;

            do {
                Vec3f numerator = Vec3f(0, 0, 0);
                float denominator = 0;
                shift = Vec3f(0, 0, 0);

                for (int m = 0; m < rows; ++m) {
                    for (int n = 0; n < cols; ++n) {
                        Vec3f neighbor = img.at<Vec3f>(m, n);
                        float weight = gaussianKernel(point, neighbor, bandwidth);

                        numerator += weight * neighbor;
                        denominator += weight;
                    }
                }

                Vec3f newPoint = numerator / denominator;
                shift = newPoint - point;
                point = newPoint;
                iter++;
            } while (norm(shift) > 1e-3 && iter < maxIter);

            result.at<Vec3f>(i, j) = point;
        }
    }
}

int main() {
    // Load the image
    Mat img = imread("path_to_image.jpg");
    if (img.empty()) {
        cerr << "Error: Image cannot be loaded!" << endl;
        return -1;
    }

    // Convert the image from BGR to Lab color space
    Mat imgLab;
    cvtColor(img, imgLab, COLOR_BGR2Lab);

    // Convert to float
    imgLab.convertTo(imgLab, CV_32F);

    // Perform custom mean shift clustering
    Mat result;
    float bandwidth = 20.0;
    int maxIter = 100;
    meanShift(imgLab, result, bandwidth, maxIter);

    // Convert back to BGR color space
    result.convertTo(result, CV_8U);
    Mat segmentedImg;
    cvtColor(result, segmentedImg, COLOR_Lab2BGR);

    // Show the result
    imshow("Segmented Image", segmentedImg);
    waitKey(0);

    return 0;
}
```

In the custom implementation:
- **Kernel Density Estimation**: For each pixel, we calculate the weighted average of all pixels within the neighborhood using the Gaussian kernel.
- **Mean Shift Vector**: The mean shift vector is computed as the difference between the weighted mean and the current pixel value.
- **Update Step**: Each pixel is iteratively updated by moving it in the direction of the mean shift vector until convergence.
- **Convergence Check**: The algorithm stops iterating when the shift is below a threshold or the maximum number of iterations is reached.

This detailed explanation and implementation should give you a solid understanding of the mean shift algorithm and how it can be applied to image segmentation in C++.

### 11.3. Graph-Based Segmentation

Graph-based segmentation is a powerful technique that models an image as a graph, where pixels or regions of pixels are represented as nodes, and edges between nodes represent the similarity (or dissimilarity) between them. This method uses graph-cut techniques to partition the graph into distinct segments that correspond to meaningful regions in the image.

**Mathematical Background**

In graph-based segmentation, an image is represented as an undirected graph $G = (V, E)$, where:
- $V$ is the set of vertices (nodes) representing the pixels or superpixels.
- $E$ is the set of edges representing the connections (similarities) between nodes.

Each edge $(u, v)$ has a weight $w(u, v)$ that quantifies the similarity between nodes $u$ and $v$. A common choice for the weight function is the Gaussian similarity function:
$$
w(u, v) = \exp \left( -\frac{\|I(u) - I(v)\|^2}{2\sigma^2} \right)
$$
where $I(u)$ and $I(v)$ are the intensities (or feature vectors) of pixels $u$ and $v$, and $\sigma$ is a scaling parameter.

**Normalized Cut**

One popular method for graph-based segmentation is the normalized cut, which partitions the graph into disjoint subsets such that the cut cost is minimized while the similarity within subsets is maximized. The normalized cut value for a partition $(A, B)$ is defined as:
$$
\text{Ncut}(A, B) = \frac{\text{cut}(A, B)}{\text{assoc}(A, V)} + \frac{\text{cut}(A, B)}{\text{assoc}(B, V)}
$$
where:
- $\text{cut}(A, B) = \sum_{u \in A, v \in B} w(u, v)$ is the total weight of edges between sets $A$ and $B$.
- $\text{assoc}(A, V) = \sum_{u \in A, t \in V} w(u, t)$ is the total weight of edges from set $A$ to all nodes in the graph.

**Implementation in C++ using OpenCV**

OpenCV provides an implementation of graph-based segmentation via the `cv::segmentation::createGraphSegmentation` function. Here is how you can use it:

```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/segmentation.hpp>
#include <iostream>

using namespace cv;
using namespace cv::ximgproc::segmentation;
using namespace std;

int main() {
    // Load the image
    Mat img = imread("path_to_image.jpg");
    if (img.empty()) {
        cerr << "Error: Image cannot be loaded!" << endl;
        return -1;
    }

    // Create a graph-based segmenter
    Ptr<GraphSegmentation> segmenter = createGraphSegmentation();

    // Perform graph-based segmentation
    Mat segmented;
    segmenter->processImage(img, segmented);

    // Normalize the segmented image to display
    double minVal, maxVal;
    minMaxLoc(segmented, &minVal, &maxVal);
    segmented.convertTo(segmented, CV_8U, 255 / maxVal);

    // Display the result
    imshow("Segmented Image", segmented);
    waitKey(0);

    return 0;
}
```

**Custom Implementation in C++**

Here is a simplified custom implementation of graph-based segmentation using the minimum cut method:

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

using namespace cv;
using namespace std;

struct Edge {
    int u, v;
    float weight;
};

class Graph {
public:
    Graph(int numVertices) : numVertices(numVertices) {}

    void addEdge(int u, int v, float weight) {
        edges.push_back({u, v, weight});
    }

    void segmentGraph(float threshold, Mat& segmentedImg) {
        // Sort edges by weight
        sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
            return a.weight < b.weight;
        });

        // Initialize each vertex to be its own component
        vector<int> parent(numVertices);
        for (int i = 0; i < numVertices; ++i) {
            parent[i] = i;
        }

        // Helper function to find the root of a component
        function<int(int)> findRoot = [&](int u) {
            while (u != parent[u]) {
                parent[u] = parent[parent[u]];
                u = parent[u];
            }
            return u;
        };

        // Merge components based on the edges
        for (const Edge& edge : edges) {
            int rootU = findRoot(edge.u);
            int rootV = findRoot(edge.v);
            if (rootU != rootV && edge.weight < threshold) {
                parent[rootU] = rootV;
            }
        }

        // Label each pixel based on its component
        segmentedImg.create(img.rows, img.cols, CV_32S);
        for (int y = 0; y < img.rows; ++y) {
            for (int x = 0; x < img.cols; ++x) {
                int idx = y * img.cols + x;
                segmentedImg.at<int>(y, x) = findRoot(idx);
            }
        }
    }

private:
    int numVertices;
    vector<Edge> edges;
};

int main() {
    // Load the image
    Mat img = imread("path_to_image.jpg");
    if (img.empty()) {
        cerr << "Error: Image cannot be loaded!" << endl;
        return -1;
    }

    // Convert image to Lab color space
    Mat imgLab;
    cvtColor(img, imgLab, COLOR_BGR2Lab);

    // Initialize graph with the number of vertices equal to the number of pixels
    int numVertices = img.rows * img.cols;
    Graph graph(numVertices);

    // Add edges to the graph
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            int idx = y * img.cols + x;
            Vec3f color = imgLab.at<Vec3f>(y, x);

            if (x < img.cols - 1) {
                Vec3f colorRight = imgLab.at<Vec3f>(y, x + 1);
                float weight = norm(color - colorRight);
                graph.addEdge(idx, idx + 1, weight);
            }

            if (y < img.rows - 1) {
                Vec3f colorDown = imgLab.at<Vec3f>(y + 1, x);
                float weight = norm(color - colorDown);
                graph.addEdge(idx, idx + img.cols, weight);
            }
        }
    }

    // Perform graph-based segmentation
    Mat segmentedImg;
    float threshold = 10.0;  // Adjust the threshold as needed
    graph.segmentGraph(threshold, segmentedImg);

    // Normalize segmented image to display
    double minVal, maxVal;
    minMaxLoc(segmentedImg, &minVal, &maxVal);
    segmentedImg.convertTo(segmentedImg, CV_8U, 255 / (maxVal - minVal));

    // Display the result
    imshow("Segmented Image", segmentedImg);
    waitKey(0);

    return 0;
}
```

In this custom implementation:
- **Graph Construction**: Each pixel is a node, and edges are created between adjacent pixels with weights based on color similarity.
- **Component Initialization**: Initially, each pixel is its own component.
- **Edge Sorting**: Edges are sorted by weight.
- **Union-Find**: The union-find algorithm is used to merge components based on edge weights.
- **Labeling**: Each pixel is labeled based on its component, and the segmentation result is displayed.

This detailed explanation and implementation should provide a comprehensive understanding of graph-based segmentation and how it can be applied to image segmentation using C++.
