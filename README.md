# TAD-Laplacian-Identification
Python algorithm for Hi-C TAD identification based on graph Laplacian method (also known as Fiedler vector method)

The original [paper](https://www.ncbi.nlm.nih.gov/pubmed/27153657) and corresponding [matlab implementation](https://github.com/laseaman/4D_Nucleome_Analysis_Toolbox)

# Using
First of all, I recommend that you familiarize yourself with the attached notebook. Don't hesitate to ask questions and contribute feedback for future improvements.

Use the command below to import the class and all required dependencies 

`from TAD_Laplacian import *`

Create a new class instance with:

`tad = TAD_Laplacian()`

Use a method **fit** to find TADs of input matrix **H**. 

`tad.fit(H, #parameters )`

Let's consider the method parameters:
* **precision**::int - serve as Hi-C matrix data limiter. If the pixel value more than **precision** then it changes with value of precision else doesn't change. In case of **precision** value equals 0 data limiter is off.
* **connected**::bool - if True, changes all the **k,k+1** and **k,k-1** subdiagonal elements to 1, else remains them unchanged.
* **norm_laplacian**::bool - if True, use [normalized laplacian](https://en.wikipedia.org/wiki/Laplacian_matrix) matrix, else unnormalized one.
* **toeplitz**::bool - if True, applying [Toeplitz normalization](https://en.wikipedia.org/wiki/Toeplitz_matrix) to the input matrix, else not.
* **recursive**::bool - if True, recursively apply the algorithm to subTADs of found TADs until the stop conditions will be achieved, else return only first iteration of Fiedler vector algorithm.
* **minlimit**::int - set minimum number of bins in a found TAD to continue of its decomposing recursively (it's the one of the stop conditions).
* **threshold**::float - maximum Fiedler value of a found TAD to stop decompose it recursively (it's the second stop condition)

After the previous method finished the work, you may visualize obtained result with next method:

`tad.visuzalize()`

It may be very useful to investigate results with some data transformation of the original matrix **H**. For such purpose there is a parameter **transform_func** which takes function (or lambda function). Note that after functional transformation of input matrix **H**, the out matrix must have the same shape.



