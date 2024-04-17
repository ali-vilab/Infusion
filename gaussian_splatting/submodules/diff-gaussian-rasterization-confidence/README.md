# Confidence based Differential Gaussian Rasterization

**Feature**:
1. Set a confidence score for each Gaussian and scale gradient with the confidence scores.
2. Implement depth and alpha rendering.

**Acknowledge**: Thanks to the [3D Gaussian Splatting](https://github.com/graphdeco-inria/diff-gaussian-rasterization) and [DreamGaussian](https://github.com/ashawkey/diff-gaussian-rasterization). 

```python
raster_settings = GaussianRasterizationSettings(
    image_height=int(viewpoint_camera.image_height),
    ...
    confidence=confidence
)

rasterizer = GaussianRasterizer(raster_settings=raster_settings)

rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
    means3D=means3D,
    ...
    cov3D_precomp=cov3D_precomp,
)
```


Used as the rasterization engine for the paper "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields". If you can make use of it in your own research, please be so kind to cite us.

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>
