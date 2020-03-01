# Philips Challenge Docker Image

This repository contains a Dockerfile and scripts to run a neural networks for a specific use case to try to detect 1 of 4 different Philips products from a given image.

**Instructions on how to get predictions:**

1. After you have docker running - for example by following these instructions for Debian: https://docs.docker.com/install/linux/docker-ce/debian/

2. `git clone` or download and extract this repository.

3. Put all the images you want to predict inside the *validation-images* folder (**Note:** the images should be in JPG format)

4. **Execute in terminal while in directory where all files were cloned/extracted:**
 *  `docker build .`
 *  `docker images` (verify that image ID appears)
 *  `docker tag <IMAGE_ID> philips_object_detection:ozzy` (<IMAGE_ID> is the ID for the most recently created image which is outputed from the previous command and the tag is to have a more friendly name and tag)
 *  `docker run philips_object_detection:ozzy`

5. Hopefully get some good predictions.

6. When done with docker image can remove it with  `docker image rm -f <IMAGE_ID>`

