# Single-channel Source Separation using NMF

In this problem, I was given an audio file named 'trs.wav' which is a speech signal of a speaker. First I perform STFT to convert it into time-frequency domain. \

For STFT, I take a frame of 1024 samples with 50% overlap and then apply Hann's window which leaves me with a complexed-value matrix. This matrix has complex conjugacy which means that bottom half of the spectrogram is a mirrored version of the upper half. This is because of the fact that 
![](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/NMF_eqn1.png) (the real part) and 
![](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/NMF_eqn2.png) (the imaginary part). 
So I only took 513 samples because the other half could be easy recovered during inverse-DFT.

After taking magnitude of the input file, I apply Non-negative matrix factorization (NMF) on S (magnitude of input file) such that \
![](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/NMF_eqn3.png) \
where 
![](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/NMF_eqn4.png) 
is a set of basis vectors and 
![](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/NMF_eqn5.png) 
where 
![](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/NMF_eqn6.png) 
is a set of nonnegative real numbers.

For updating 
![](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/NMF_eqn4.png) 
and 
![](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/NMF_eqn7.png), 
I use the below given update rules.

![NMF Update Rules](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/NMF_Update_Rules.PNG)

The error function for NMF is given by : 

![NMF Error Function](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/NMF_Error_function.PNG)

Similarly, I take another audio file 'trn.wav' which only consists of noise and perform the above steps to get
![](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/NMF_eqn8.png)

Once training on both clean and noisy signal was completed, I took a test audio file 'x_nmf.wav' to check how the model performed in separating the source. Since basis vectors were already learnt in the training step, let 
![](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/NMF_eqn9.png). 
Now since the activation matrix is ready, only activation of these basis vectors needs to be learnt. 

Once both 
![](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/eqn2.png)
and 
![](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/NMF_eqn10.png) 
were ready, a masking matrix could be created from them using the below given formula :

![NMF Masking Matrix](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/NMF_Masking_Matrix.PNG)

The above matrix 
![](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/NMF_eqn11.png)
can then be used to recover the speech source.
