# How to Run
+ Use 'pytest models/demos/resnet/demo/demo.py::test_demo_sample[8-models/demos/resnet/demo/images/]' to run the demo, where 8 is the batch size, and 'models/demos/resnet/demo/images/' is where the images are located. Our model supports batch size of 2 and 1 as well, however the demo focuses on batch size 8 which has the highest throughput among the three options. This demo includes preprocessing, postprocessing and inference time for batch size 8. The demo will run the images through the inference twice. First, to capture the compile time, and cache all the ops, Second, to capture the best inference time on TT hardware.
+ Our second demo is designed to run ImageNet dataset, run this with 'pytest models/demos/resnet/demo/demo.py::test_demo_imagenet[8, 400]', again 8 refer to batch size here and 400 is number of iterations(batches), hence the model will process 400 batch of size 8, total of 3200 images.

# Inputs
+ Inputs by defaults are provided from 'models/demos/resnet/demo/images/' which includes 8 images from ImageNet dataset. If you wish to modify the input images, modify the abovementioned command by replacing the path with the path to your images. Example: 'pytest models/demos/resnet/demo/demo.py::test_demo_sample[8-path/to/your/images]'.
+ You must put at least 8 images in your directory, and if more images located in your directory, 8 of them will randomly be picked. In this demo we assume images come from ImageNet dataset, if your images are from a different source you might have to modify the preprocessing part of the demo.


# Details
+ The entry point to metal resnet model is 'ResNet' in 'metalResNetBlock50.py'. The model picks up certain configs and weights from TorchVision pretrained model. We have used 'torchvision.models.ResNet50_Weights.IMAGENET1K_V1' version from TorchVision as our reference.
Our ImageProcessor on other hand is based on 'microsoft/resnet-50' from hugginface.
+ For the second demo (ImageNet), the demo will load the images from ImageNet batch by batch. When executed, the first iteration (batch) will always be slower since the iteration includes the compilation as well. Afterwards, each iterations take only miliseconds. For exact performance measurements please check out the first demo.
