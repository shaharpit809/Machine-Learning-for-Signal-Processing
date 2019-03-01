# KNN Source Separation

In this problem, I was given 3 audio files such as: \
* trs.wav - Clean signal of the speaker
* trn.wav - Noise signal
* x_nmf.wav - Test signal consisting of speaker signal with noise

After converting all the files in time-frequency, I created a masking matrix called Ideal Binary Masks (IBM) such that :
![](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/knn_ss_eqn1.png)

By using the nearest neighbor indices, I collected the IBM vectors from 
![](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/knn_ss_eqn2.png)

If the t-th frame belonged to sound signal then there should have been more number of sound neighbours as compared to noise. So I created anothe matrix 
![](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/knn_ss_eqn3.png)
which 1 if 1 was the majority and vice-versa.

![](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/knn_ss_eqn4.png)

I then use the above matrix 
![](https://github.com/shaharpit809/Machine-Learning-for-Signal-Processing/blob/master/img/knn_ss_eqn3.png)
to suppress the noise and recover a clear audio file consisting of only speaker's signal.
