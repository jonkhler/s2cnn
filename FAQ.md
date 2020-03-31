> Why there is a very little difference when evaluating on the non-rotational input and the rotational input?

Our formula are equivariant, but in our implementation we discretized the formula which causes this little difference.

> The Spherical CNN takes lots of memory. Now we plan to build a larger model and train on multiple gpus. Is this possible?

This is due to the 3 dimensional internal representations.
Last attempt to support multi-gpu was done here [#8](https://github.com/jonas-koehler/s2cnn/issues/8)
You can also look at other architecture that consume less memory like [1902.04615](https://arxiv.org/abs/1902.04615)

> 1. You provided two options: identity and equators. Why is that ‘identity’ is localised while ‘equator’ is non-local support?

These are two choices of shape of filters. In 2d images the filters (also called kernel) are usually squared 3x3, but it can also be rectangular. In theory all shapes are allowed. In our framework we convolve over the sphere s2 and the group SO3, in these spaces as well we can chose the shape of the kernel as we wish. We took a localized one (similar to the 3x3 square of the 2d images) that we places wlog at the north pole. The second shape we tried is a ring around the equator, non-local because it turns around the entire space.

> 2. Take the identity one for example, the sampling grid is chose to be close to the north pole, does this mean it only sees the input data around north pole?

No, the kernel is roatated around in all possible orientations. Like the 3x3 kernel on 2d images who is translated on the entire image.

> 3. For the choice of max_alpha, by default, it is 2*pi. Is it still meaning if using smaller value for max_alpha, e.g. pi or pi/2. 

For the identity shape you will get a shape of pie. For the equatorial kernel you will get a portion of ring. Again all the shapes are allowed and it is not intuitive to me which one is better than another.

