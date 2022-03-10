# Traffic-Density-Estimator
Computer Vision implementation in Traffic Density Estimation




SUMMARY:

Traffic arising from automobiles is a prominent issue with increasing vehicle size and a surge in
urban city population, the problem of traffic congestion has been growing exponentially. As the
global population continues to urbanize, without a well-planned road and traffic management
infrastructure this problem will worsen.

This projects attempts to build a seamless tool using Computer Vision and Image Processing to accurately
estimate the traffic density by augmenting the existing solutions at minimum added cost.
 
 
 

PIPELINE:

![Pipeline](https://user-images.githubusercontent.com/92863991/157686744-31ff30b6-d279-4210-ad54-c57485f2cb28.png)

This pipeline consists of six distinct stages. We begin with image acquisition, where two images from
wide angle cameras located on opposite street lights are captured synchronously within set time
intervals. Then these images will then be feature detected and matched using appropriate computer
vision algorithms such as SIFT (Scale Invariant Feature Transform), SURF (Speeded Up Robust
Feature), etc. The two feature matched images will then be stitched together to form a single image
that contains information about the road and vehicles present on it when the images were captured.
The newly stitched image is fed to an object detection algorithm that differentiates the vehicles from
their surroundings and returns their total count. Based on the total number of vehicles present in
the images pertaining to a specific duration of time interval, simple arithmetic calculation will
provide us with an approximate density of traffic on the selected road. Since the density will be
calculated in relative terms, the road (under observation) in any navigation application can now be
pseudo-colored (from Blue to Red) with respect to the corresponding traffic density.




IMPACT:

The primary impact of this topic would be on the millions of existing customers, by providing
real-time accurate traffic data which enables our customers to plan their commute efficiently and
economically. Furthermore, with growing concerns over user-data privacy practices, it is imperative
that a shift to alternate solutions that do not require the continuous harvest of user-data for
location tracking is considered. 

On a larger scale, projects like these would be able to provide critical insights for urban
infrastructure planning. Such data can help governments allocate resources towards the
development of highways, parking lots, metro lines and more to address issues arising from traffic
congestion that contribute to lower productivity and it’s associated economic implications.
