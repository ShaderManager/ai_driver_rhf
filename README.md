Arnold RHF driver
===========================================================

Raw output driver for Solid Angle's Arnold with filter based on publication "Boosting Monte Carlo Rendering by Ray Histogram Fusion" http://dev.ipol.im/~mdelbra/rhf/ 

Library dependencies: C++11-compatible compiler, Arnold SDK (version >= 4.1), OpenImageIO (version >= 1.5) and TBB

===========================================================
Current limitations:
1. Only RGBA Beauty is supported at this moment (i.e. 4 float channels).
2. Image is stored in memory wholly. It means driver can allocate significant amount of memory.
3. Plugin saves picture in 32 bit float format. Subject to change in future.

===========================================================
Node parameters:
+ STRING filename (default = output.tif) - Output filename. Image writing is handled by OpenImageIO
+ STRING filter (default = gaussian) - Pixel samples filter (before RHF filter). Plugin uses filters from OpenImageIO library
+ FLOAT filter_width (default = 2) - Size of pixel samples filter
+ FLOAT threshold (default = 1) - Distance threshold for RHF filter (k parameter in paper). This parameter affects blurriness of filtered image.
+ INT patch_size (default = 1) - Radius of comparison pixel patch (w parameter). Real size is window of (2 * patch_size + 1)x(2 * patch_size + 1) pixels.
+ INT search_window_size (default = 6) - Radius of search window (b parameter). Real size is window of (2 * search_window_size + 1)x(2 * search_window_size + 1) pixels.
+ INT scales (default = 2) - Number of scales n_s. Multiple scales of input picture is used for filtering low-frequency noises.
+ INN knn (default=2) - Minimum number of nearest neighbor patches with similiar colors (i.e. with minimal chi^2 distance).