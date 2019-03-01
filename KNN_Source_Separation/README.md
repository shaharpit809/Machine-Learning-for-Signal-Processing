# KNN Source Separation

In this problem, I was given 3 audio files such as: \\
* trs.wav - Clean signal of the speaker
* trn.wav - Noise signal
* x_nmf.wav - Test signal consisting of speaker signal with noise

After converting all the files in time-frequency, I created a masking matrix called Ideal Binary Masks (IBM) such that :
\begin{equation}
  B_{f,t}=\begin{cases}
    1, & \text{if $S_{f,t} \geq N_{f,t}$}\\
    0, & \text{otherwise}.
  \end{cases}
\end{equation}

By using the nearest neighbor indices, I collected the IBM vectors from $ B $ i.e. $ \{ B_{:,i_1} , B_{:,i_2}, ..., B_{:,i_k} \} $ \\ 

If the t-th frame belonged to sound signal then there should have been more number of sound neighbours as compared to noise. So I created anothe matrix $ D $ which 1 if 1 was the majority and vice-versa.

$ D_{:,t} = median(\{ B_{:,i_1} , B_{:,i_2}, ..., B_{:,i_k} \}) $

I then use the above matrix $D$ to suppress the noise and recover a clear audio file consisting of only speaker's signal.