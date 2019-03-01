# Instantaneous Source Separation

In this problem, I was given a total of 20 recordings of Jazz music. Each recording has N time domain samples. In this music there are
K unknown number of musical sources played at the same time. The goal of the problem was to find the K different mixing sources and thus separate them from the input files to get a clear audio of the Jazz music.

After hearing all the recording, I had a feeling that the number of sources were much less than 20, so I first performed PCA with whitening to reduce the number of dimensions. PCA was performed on a matrix of 20 X N dimension. After PCA, I take a look at the eigenvalues to decide on the number of K sources. 

On my whitened/dimension reduced data matrix Z(K X N), I applied Independent Compnent Analysis(ICA). Update rules for ICA used were :

![Eqn1](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/eqn1.png)

where

![Eqn2](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/eqn2.png): The ICA unmixing matrix you're estimating \
![Eqn3](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/eqn3.png): The K X N source matrix you're estimating \
![Eqn4](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/eqn4.png): Whitened/dim reduced version of your input (using PCA) \
![Eqn5](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/eqn5.png) \
![Eqn6](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/eqn6.png) \
![Eqn7](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/eqn7.png): learning rate \
![Eqn8](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/eqn8.png): number of samples


Once ICA converges, the audio files are separated and clean Jazz music is retained.
