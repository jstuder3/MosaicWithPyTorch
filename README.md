# MosaicWithPyTorch
Recreate your images using other images.

Usage:

In main.py, set the "images_folder" to the folder where all of your images are located in. 
Set "target_image" to the name of the image you want to recreate using the other images.

Select into how many patches the large image should be split ("target_subdivions").
Each image will cover "search_suvisivions"**2 patches. This must be chosen >=1 (because there's a bug that I don't have the energy to find).
When the small images are inserted into the final image, you can select the height of the inserted patches in pixels using "insertion_subdivisions".

To avoid repeating the same image over and over again, you can set "num_samples" to randomly select from the top k matches per patch. (sampling distribution weighted according to their respective similarities)

Finally, set "search_preprocess" and "insertion_preprocess" to True the first time you run this. 
"search_preprocess" will determine the average color per image. Rerun this every time you change "search_subdivisions".
"insertion_preprocess" will create subsampled/downsized versions of all images which will be inserted into the final image. Rerun this every time you change "insertion_subdivisions".
Additionally, you need to rerun the preprocessing steps every time you change (e.g. add or remove) images.

The preprocessing step can take quite some time. To reduce this, you can use a GPU (needs to be CUDA-capable). 
If you have a version of PyTorch installed that can use GPUs, this will be done automatically. Otherwise, the CPU will be used.

Library requirements:

-- pytorch
-- torchvision
-- PIL (Python Image Library)
