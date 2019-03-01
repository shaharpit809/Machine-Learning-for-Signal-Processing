# Single-channel Source Separation using NMF

In this problem, I was given an audio file named 'trs.wav' which is a speech signal of a speaker. First I perform STFT to convert it into time-frequency domain. \\

For STFT, I take a frame of 1024 samples with 50% overlap and then apply Hann's window which leaves me with a complexed-value matrix. This matrix has complex conjugacy which means that bottom half of the spectrogram is a mirrored version of the upper half. This is because of the fact that $cos(\theta) = cos(2 \pi - \theta)$ (the real part) and $sin(\theta) = -sin(2 \pi - \theta)$ (the imaginary part). So I only took 513 samples because the other half could be easy recovered during inverse-DFT.

After taking magnitude of the input file, I apply Non-negative matrix factorization (NMF) on S (magnitude of input file) such that \\
$ S \approx W_{S}H_{S}$ \\
where $ W_{S} $ is a set of basis vectors and $ W_{S} \epsilon R_{+} ^{513 \times 30} $ where $ R_+ $ is a set of nonnegative real numbers.

For updating $W_{S} $ and $ H_{S} $, I use the below given update rules. \\

![NMF Update Rules](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/NMF_Update_Rules.PNG)

The error function for NMF is given by : 

![NMF Error Function](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/NMF_Error_function.PNG)

Similarly, I take another audio file 'trn.wav' which only consists of noise and perform the above steps to get $ W_{N} $

Once training on both clean and noisy signal was completed, I took a test audio file 'x_nmf.wav' to check how the model performed in separating the source. Since basis vectors were already learnt in the training step, let $ W = [W_{S}W_{N}] $. Now since the activation matrix is ready, only activation of these basis vectors needs to be learnt. 

Once both $ W $ and $ H $ were ready, a masking matrix could be created from them using the below given formula :

![NMF Masking Matrix](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/NMF_Masking_Matrix.PNG)

The above matrix $M$ can then be used to recover the speech source.
