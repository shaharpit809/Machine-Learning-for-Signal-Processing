# Removing white noise

In this problem, I was given a file 'x.wav' which was contaminated by white noise. Once the file was loaded, I applied Hann's window on the input file.
For Hann's window, I create a window of size N and carry out element wise multiplication with N samples of the input file. This helps in creating the X matrix for calculation of Discrete Fourier Transform (DFT) matrix. After multiplication, we move by N/2 samples again with the size of N samples.

Once all the samples were covered, I applied DFT on the data matrix, ie Y = FX. From this step I got a spectogram with complex values which in turn have noise at the start and end of the file.