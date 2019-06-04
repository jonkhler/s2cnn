> Why there is a very little difference when evaluating on the non-rotational input and the rotational input?

Our formula are equivariant, but in our implementation we discretized the formula which causes this little difference.

> The Spherical CNN takes lots of memory. Now we plan to build a larger model and train on multiple gpus. Is this possible?

This is due to the 3 dimensional internal representations.
Last attempt to support multi-gpu was done here [#8](https://github.com/jonas-koehler/s2cnn/issues/8)
You can also look at other architecture that consume less memory like [1902.04615](https://arxiv.org/abs/1902.04615)
