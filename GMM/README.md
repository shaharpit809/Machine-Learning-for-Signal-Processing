# GMM for Parallax

Parallax is an effect in which the position or direction of an object appears to differ when viewed from different positions.
For this problem, I was given data about x and y coordinates of stars in the month of June and December. The problem statement came with an assumption that the displacements of stars only occurred on the x-axis. Considering this assumption, I was able to calculate the disparity (amount of oscillation) of all the 2700 stars.

After plotting a histogram, there were only 2 clusters that were visible, so I first calculated the 2 cluster means values using K-means and from the output of the K-means algorithm, I implemented Expectation-Maximization(EM) algorithm from scratch to identify which stars belonged to our galaxy and also compare why EM for Gaussian Mixture Models(GMM) works better as compared to K-means.