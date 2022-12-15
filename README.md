# FingeripsPositionEstimationofaRobotHand-
This is my final project for my machine learning class


Fingertips Position Estimation of a Robot Hand 

Professor: Lerrel Pinto
Student: Yinuo Xu

Method
1.1 MyDataSet Class
The init function in this code is used to load either train data or test data, depending on the value of the train parameter passed to the function. Additionally, a sequence of transformations can be specified as the second parameter to this function. This allows for flexible and customizable data loading and processing.

The getitem function is responsible for reading the .png images using cv2.imread(), as well as the depth data and ground truth using np.load(). The ground truth is then multiplied by 1000 in order to convert the scale from millimeters to meters. The depth value is not modified after being loaded, as attempts to divide it by 1000 resulted in worse model performance. The return format of this function is a tuple containing the images, depth data, and ground truth, in the format (image1, image2, image3, depth), Y.

The len function simply returns the length of the data in this DataSet, allowing for easy determination of the size of the dataset.

Overall, these functions provide a convenient and effective way to load, process, and analyze data for machine learning purposes.

1.2 Data Preprocess
During preprocessing of the data, a sequence of transformations is put in a list and applied to the images. The transformations are designed to improve the contrast and visibility of the hand in the images, and to help regularize the model during training.

The first step in the sequence is to convert the Tensor data to a PIL image using the ToPILImage() function. This allows the data to be manipulated using standard image processing techniques.

Next, the Grayscale(3) function is used to convert the image to black and white, with the parameter 3 chosen to preserve the original dimensions of the image. This helps to make the hand in the image more distinct and easier to segment.

The RandomHorizontalFlip() function is then used to randomly flip the image horizontally with a default probability of 0.5. This acts as a regularizer and helps reduce the likelihood of overfitting during model training.

Finally, the ToTensor() function is used to convert the image back to a Tensor, and the mean and standard deviation of the images are calculated. The Normalize() function with parameters ([111.0820, 117.8825, 123.7023], [60.2689, 56.3253, 56.8279]) is then used to normalize the images.

Overall, the sequence of transformations is designed to improve the quality of the input data and make it easier for the model to learn from it.
MyImageTransformations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([111.0820, 117.8825, 123.7023], 
                           [60.2689, 56.3253, 56.8279])
])

1.3 Train Function
MSELoss is applied as a loss function to the customized train function.

1.4 Data Loading and Model Setup
A customized dataset was created and the data was loaded to it. Then, a dataloader was created based on the data. The torchvision.models.resnet50 model, a pretrained convolutional neural network with 50 layers, was imported and customized to fit twelve-channel data by changing the layers:
model.fc = nn.Linear(2048, 12)
model.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)

1.5 Training Model Process
The actual training process for my model involves using stochastic gradient descent as the chosen optimizer. In order to ensure the stability of the training process and avoid potential issues with getting stuck, the learning rate was carefully set to 0.01 and the momentum was set to 0.9. The learning rate was chosen to be neither too large, which could cause an unstable process, nor too small, which could result in a long and potentially stuck training process. The model was trained over a total of 70 epochs, using the torch.save() function to save the trained parameters of the model at the end of the training process.

Overall, this training process allowed me to effectively optimize the model using SGD and save the resulting parameters for future use.

1.6 File Generation
All of the data used in this process is being loaded from the lazy-data source, with the exception of the id numbers, which are being loaded from a specific file located at './csci-ua-473-intro-to-machine-learning-fall22/test/test/testX.pt'. In order to calculate the id numbers individually and efficiently, the batch size was carefully set to 1. This allows for the calculation of each id number separately, without the need for additional processing or data manipulation. The final output of this process was divided by 1000 in order to fit the original data that was used for Data Preprocessing. This ensures that the calculated id numbers are accurate and consistent with the original data, allowing for effective and efficient analysis.                        

Discussion

3.1 Different Epoch Values
Based on the data given, it appears that the loss and error values decrease as the epoch value increases. This indicates that the model is improving as it trains for more epochs.
In general, the number of epochs (or the number of times the model sees the entire training dataset) is a hyperparameter that can be adjusted to influence the performance of a deep learning model. Increasing the number of epochs can allow the model to continue improving, but it can also lead to overfitting if the model begins to memorize the training data rather than learning generalizable patterns.
In this particular case, it appears that the model's performance improves as the number of epochs increases up to around 30, after which the error plateaus. This suggests that the model may have reached a good balance between underfitting and overfitting, and that additional training may not significantly improve the model's performance. As we could also notice on the graph, the error tends to be flat afterwards.

3.2 Different Model Selection
Based on the data given, it appears that the error values for the different ResNet models vary. Specifically, the ResNet50 model has the lowest error, while the ResNet34 has highest error value.
In general, the choice of model architecture can have a significant impact on the performance of a deep learning model. Different model architectures have different numbers of layers, different parameter sizes, and different capacities to learn complex patterns in the data. As a result, the performance of a given model can vary depending on the architecture chosen.
In this particular case, it appears that the ResNet50 model has the lowest error, indicating that it is the best performing model among the ones listed. This suggests that the ResNet50 model may be well-suited for the problem and the data at hand.

3.3 Different Optimizers for Training
Based on the data and graph given, it appears that the choice of optimizer can have a significant impact on the performance of a deep learning model. In particular, the Adam optimizer appears to have the lowest loss and error values among the three optimizers listed.
In general, the optimizer is an important component of the training process of a deep learning model. It determines how the model's parameters are updated based on the computed gradients, and can influence the convergence speed and the final performance of the model. Different optimizers have different properties and can be better suited for different types of problems and data.
In this particular case, the Adam optimizer appears to be the best performing optimizer among the ones listed, achieving the lowest loss and error values. And it also has the smoothest decreasing trend. This suggests that it may be well-suited for the problem and the data at hand.


Future Work

4.1 Model
Consider switching to a different model architecture such as VGG, Inception, or DenseNet instead of the prebuilt ResNet model. These models have been trained on large datasets of images and can be used for tasks such as image classification, detection, and segmentation.

4.2 Data Preprocessing
Enhance the MyImageTransformations sequence by adding more data preprocessing techniques. For example, calculate the mean and std values separately for each of the three input images (image1, image2, and image3) rather than calculating them together. Additionally, consider normalizing the depth data to potentially improve the performance of the model.

4.3 Optimizer
Experiment with using different optimizers and their corresponding hyperparameters, such as the learning rate, to optimize the model. Also, extend the experimental results to include a larger range of epoch values, rather than just showing the results for epoch = 10. This will provide a more comprehensive view of the model's performance and allow for better comparison with other models and optimizers.




