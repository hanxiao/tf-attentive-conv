# tf-attentive-conv: Attentive Convolution

Han Xiao <artex.xh@gmail.com>

## What is it?
This is a Tensorflow implementation of Yin Wenpeng's paper "Attentive Convolution" at TACL in 2018. [Wenpeng's original code](https://github.com/yinwenpeng/Attentive_Convolution) is written in Theano. 

I only implement the light attentive convolution described in Sect. 3.1 of the paper. The following figure (from the paper) illustrates this idea: 

![](.github/e4ff1f17.png)


Authors argue that `AttConv` outperforms some of pioneering attentive RNNs in both intra-context (`context=query`, i.e. self-attention) and extra-context (`context!=query`) settings.

## Run 

Run `app.py` for a simple test on toy data.