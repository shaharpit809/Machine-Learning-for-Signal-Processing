# Instantaneous Source Separation

In this problem, I was given a total of 20 recordings of Jazz music. Each recording has N time domain samples. In this music there are
K unknown number of musical sources played at the same time. The goal of the problem was to find the K different mixing sources and thus separate them from the input files to get a clear audio of the Jazz music.

After hearing all the recording, I had a feeling that the number of sources were much less than 20, so I first performed PCA with whitening to reduce the number of dimensions. PCA was performed on a matrix of 20 X N dimension. After PCA, I take a look at the eigenvalues to decide on the number of K sources. 

On my whitened/dimension reduced data matrix Z(K X N), I applied Independent Compnent Analysis(ICA). Update rules for ICA used were : \\

$\Delta W \leftarrow (NI - g(Y)f(Y)')W$ \\
$W \leftarrow W + \rho \Delta W$ \\
$ Y \leftarrow WZ$

where

$ W $: The ICA unmixing matrix you're estimating \\
$ Y $: The K X N source matrix you're estimating \\
$ Z $: Whitened/dim reduced version of your input (using PCA) \\
$ g(x): tanh(x) $ \\
$ f(x): x^3 $ \\
$\rho$: learning rate \\
$ N $: number of samples \\


Once ICA converges, the audio files are separated and clean Jazz music is retained.
