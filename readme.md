# Agricultural Plant Disease and Pest Detection System Based on Yolov11+Django+DeepSeek with a Login Interface in Python Source Code + ONNX Model + Beautiful Web Interface

Effect Display: Login interface main interface based on YOLO11 target detection system, supports image, video and real-time camera detection, integrated DeepSee.

## Effect Display

Login interface

![](01-f767d8d0.jpg)

Main Interface

![](02-2805a3f6.jpg)

A target detection system based on YOLO11, supporting image, video, and real-time camera detection, integrated with DeepSeek AI analysis functions.

### Functional Characteristics

-

Image target detection

-

Video file detection

-

Real-time camera detection

-

DeepSeek AI Intelligent Analysis

-

Adjustable detection parameters (confidence, IoU)

-

Beautiful user interface

### [Test Environment]

windows10 anaconda3+python3.10 torch 2.3.1 ultralytics 8.4.7 Django==5.2.11

### Model can detect 41 categories

['appleblackrot', 'apple healthy', 'apple scab', 'bell pepper bacterial spot', 'bell pepper healthy', 'cassava Brown Streak Disease', 'cassava bacterial blight', 'cassava green mottle', 'cedar applerust', 'cherry healthy', 'cherry powdery mildew', 'corn cerespora leaf spot', 'corn common rust', 'corn healthy', 'grape black rot', 'grape esca', 'grape healthy', 'grape leaf blight', 'healthy cassava', 'mosaic  cassava', 'northern leaf blight', 'orange citrus greening', 'peach bacterial spot', 'peach healthy', 'potato early blight', 'potato healthy', 'potato late blight', 'rice brown spot', 'rice healthy', 'rice hispa', 'rice leaf blast', 'spider mites two-spotted spider mite', 'squash powdery mildew', 'strawberry healthy', 'strawberry leaf scorch', 'tomato bacterial spot', 'tomato early blight', 'tomato late blight', 'tomato leaf healthy', 'tomato leaf mould', 'tomato septoria leaf spot']

[Training Dataset Introduction]

Dataset format: YOLO format (txt file without split path, only contains jpg images and corresponding yolo format txt files)

Number of images (number of jpg files): 33085

Number of annotations (number of txt files): 33085

Number of training set: 30105

Number of validation set: 1988

Test set size: 992

Label category count: 41

Label category name(NoteyoloThe format category order does not correspond to this, but instead...labelsFolderclasses.txtPrecision):['apple black rot','apple healthy','apple scab','bell pepper bacterial spot','bell pepper healthy','cassava Brown Streak Disease','cassava bacterial blight','cassava green mottle','cedar apple rust','cherry healthy','cherry powdery mildew','corn cerespora leaf spot','corn common rust','corn healthy','grape black rot','grape esca','grape healthy','grape leaf blight','healthy cassava','mosaic  cassava','northern leaf blight','orange citrus greening','peach bacterial spot','peach healthy','potato early blight','potato healthy','potato late blight','rice brown spot','rice healthy','rice hispa','rice leaf blast','spider mites two-spotted spider mite','squash powdery mildew','strawberry healthy','strawberry leaf scorch','tomato bacterial spot','tomato early blight','tomato late blight','tomato leaf healthy','tomato leaf mould','tomato septoria leaf spot']

Number of boxes labeled in each category:

apple black rot (apple black rot) frame count = 807

apple healthy (healthy apple) frame count = 847

apple scab (apple scab) frame count = 842

Bell pepper bacterial spot (sweet pepper bacterial leaf spot) frame count = 837

Bell pepper healthy (Healthy bell pepper) frame count = 838

cassava Brown Streak Disease (Mutton Chop Disease) frame count = 2472

Cassava bacterial blight (cassava bacterial wilt) frame count = 1330

Cassava green mottle (Mutant cassava) frame count = 2643

Cedar Apple Rust (Apple Rust) Frame Count = 646

Cherry Healthy (Healthy Cherry) Frames = 830

Cherry powdery mildew (cherry white mold) frame count = 844

corn cerespora leaf spot (corn stalk rot) frame count = 848

Corn common rust (corn common rust) frame count = 849

Corn Healthy (Healthy Corn) Frames = 849

Grape black rot (Vitis vinifera) frame count = 836

Grape esca (Vitis vinifera) frame count = 822

Grape Healthy (Healthy Grapes) Frames = 833

Grape leaf blight (Vitis vine, Vitis spp.) frame count = 835

Healthy Cassava (Healthy Yam) Frames = 1303

Mosaic Cassava (Potato Leaf Blight) Frame Count = 7156

Northern leaf blight (corn stripe disease) frame count = 869

Orange Citrus Greening (Citrus Huanglongbing) frame count = 824

Peach bacterial spot (Peach bacterial leaf spot) frame count = 835

Peach Healthy (Healthy Peach) Frames = 836

Potato early blight (potato early blight) frame count = 840

potato healthy (healthy potato) frame count = 363

Potato late blight (potato late blight) frame count = 834

Rice brown spot (rice blight) frame count = 851

rice healthy (healthy rice) frame count = 799

Rice hispa (rice iron-clawed beetle) frame count = 789

Rice leaf blast (rice blast) frame count = 830

Spider mites Two-spotted spider mite (Two-spotted mealybug) frame count = 652

Squash powdery mildew (pumpkin powdery mildew) frame count = 937

Strawberry Healthy (Healthy Strawberries) Frames = 825

Strawberry leaf scorch (Strawberry Leaf Blight) frame count = 920

Tomato bacterial spot (bacterial leaf spot of tomatoes) frame count = 821

Tomato early blight (Early Blight of Tomato) frame count = 863

Tomato late blight (late blight of tomatoes) frame count = 941

Tomato leaf healthy (Healthy Tomato Leaf) frame count = 826

Tomato leaf mold (Phytophthora infestans) frame count = 842

Tomato septoria leaf spot (Tomato Septoria Leaf Spot) frame count = 848

Total frame count = 44412

Using labelImg:

Annotation rules: Draw a rectangle around the category.

Important Notice: No

Special : This dataset does not guarantee the accuracy of the trained models or weight files.

Image Preview:

![](03-8b6f3884.jpg)

![](04-82a3255d.jpg)

Annotation example:

![](05-de4b15fb.jpg)

[Training Information]

Number of training set images = 30223

Number of validation set images = 2001

Training map = 92.1%

Training Accuracy = 91.1%

Training recall rate (Recall) = 90.4%

[Verification set accuracy]

Category = Map50 (%)

MAP50

all=92

apple black rot=100

apple healthy=99

apple scab=100

bell pepper bacterial spot=100

bell pepper healthy=100

cassava Brown Streak Disease=53

cassava bacterial blight=45

cassava green mottle=58

cedar apple rust=100

cherry healthy=100

cherry powdery mildew=100

corn cerespora leaf spot=97

corn common rust=100

corn healthy=100

grape black rot=100

grape esca=100

grape healthy=100

grape leaf blight=99

healthy cassava=43

mosaic cassava=57

northern leaf blight=96

orange citrus greening=100

peach bacterial spot=100

peach healthy=99

potato early blight=100

potato healthy=100

potato late blight=100

rice brown spot=98

rice healthy=76

rice hispa=85

rice leaf blast=93

spider mites two-spotted spider mite=99

squash powdery mildew=98

strawberry healthy=100

strawberry leaf scorch=97

tomato bacterial spot=98

tomato early blight=99

tomato late blight=96

tomato leaf healthy=100

tomato leaf mould=100

tomato septoria leaf spot=100

### Install Dependencies

```
pip install -r requirements.txt
```

### Database Migration

```
python manage.py makemigrations
python manage.py migrate
```

### Running Projects

```
python manage.py runserver
```

Visit http://127.0.0.1:8000

### Default login information

-

Username: admin

-

Password: admin

### Main File Structure of the Project

```
yolo11-django-deepseek-det-system/
├── manage.py               # Django management script
├── yolo_detection/ # Project configuration directory
│   ├── __init__.py
├── settings.py # Project settings
├── urls.py # Main URL configuration
└── wsgi.py # WSGI configuration
├── detection/  # Detection Application
│   ├── __init__.py
├── apps.py # Application configuration
├── forms.py
├── models.py
├── urls.py # Application URL configuration
└── views.py # View functions
├── templates/ # Template directory
├── test_img/ # Test image directory
├── index.html # Main page
└── login.html # Login Page
├── static/  # Static file directory
├── media/  # Media file directory
└── uploads/ # Upload Files Directory
├── requirements.txt  # Dependency list
├── weights/  # Template directory
├── yolov11n.onnx # Model file
└── results.png # Training log image
├── class_names # Model category file
└── model_description.txt # Brief introduction to the training and process map0.5 parameters
```

### Technology Stack

-

Django 4.2+

-

YOLO11 (Ultralytics)

-

OpenCV

-

DeepSeek API

-

Chart.js

-

HTML5/CSS3/JavaScript

### Notes

-

Ensure that the `yolov11n.onnx` model file is in the project's weights directory.

-

DeepSeek API key needs to be configured in `settings.py`

-

Recommend using Python 3.8+

### Common Evaluation Parameters Introduction

In the task of object detection, evaluating the performance of a model is crucial. The terms you mentioned are commonly used metrics for assessing the performance of a model. Here are detailed explanations of these terms:

Class: This usually refers to the category of objects that the model is designed to detect. For example, a model may be trained to detect different categories of objects such as vehicles, pedestrians, or animals.

Images: Represents the number of images in the validation set. The validation set is a dataset used to evaluate model performance, separate from the training set to ensure fairness in evaluation results.

Instances: The total number of target objects in all images. This includes the sum of all category objects, for example, if the validation set contains 100 images, each image has an average of 5 target objects, then Instances is 500.

P (Precision): Precision is the ratio of true positives among the instances predicted as positive by the model. The formula is: Precision = TP / (TP + FP), where TP represents True Positives and FP represents False Positives.

R (Recall): Recall is the proportion of all true positives correctly predicted by the model. The formula is: Recall = TP / (TP + FN), where FN represents false negatives.

mAP50: Mean Average Precision at IoU (Intersection over Union) threshold of 0.5, which represents the average precision when the IoU (Intersection over Union) is set to 0.5. The IoU metric measures the degree of overlap between predicted bounding boxes and ground truth bounding boxes. mAP is a comprehensive metric that considers both precision and recall, used to evaluate the performance of models across different recall levels. At an IoU of 0.5, if the overlap between the predicted bounding box and the ground truth bounding box reaches or exceeds 50%, it is considered correct.

mAP50-95: Refers to the average precision of a model within the range of IoU from 0.5 to 0.95 (with a step of 0.05). This is a more stringent evaluation criterion, demanding higher overlap between predicted bounding boxes and true ones. In object detection tasks, a higher IoU threshold implies that the model needs to be more accurate in target object localization. The computation of mAP50-95 takes into account multiple IoU thresholds ranging from lenient to strict, thus providing a comprehensive assessment of the model's performance. These metrics collectively form an important framework for evaluating the performance of object detection models. By comparing the performance of different models on these metrics, one can determine which model may be more effective in practical applications.

### Common Issues

In target detection training, a low Mean Average Precision (MAP) may be due to one of the following reasons:

Reason two: Small Objects: If the dataset contains a large number of small objects, it is likely to result in a low map value. This is because the features of small objects are not distinct, making it difficult for the model to learn them.

Reason 3: Mismatched model tuning parameters: For instance, if the learning rate is adjusted too high, it may lead to an overly rapid learning capability, resulting in disordered adjustments of model parameters.

Reason 4: Overfitting (This is now a rare occurrence in models): If the model performs exceptionally well on training data but poorly on validation or test data, it may be due to overfitting. This is often caused by an excessive number of model parameters relative to the amount of training data available, leading the model to learn noise or specific patterns from the training data and fail to generalize to new data. Nowadays, target detection models are well-equipped to handle this situation, with very few instances of overfitting occurring.

Reason 5: Different scenes: The validation set has high accuracy, but the test set does not, which may be due to inconsistencies between the training model's scene images and those in the test map. To address these issues, the following measures can be taken to improve MAP:

(1) Optimize model structure: Select the appropriate model based on the characteristics of the task and dataset, and try using different network architectures and components to improve model performance.

(2) Enhance Data Preprocessing: Proper preprocessing and enhancement of data, such as data normalization, missing value imputation, data expansion, etc., to enhance the generalization capability of the model.

(3) Adjusting the Loss Function: Try using different loss functions or combining multiple loss functions to optimize model performance.

(4) Optimize training strategies: Adjust hyperparameters such as learning rate, batch size, and number of training iterations, and employ optimization algorithms like learning rate decay and momentum to enhance model training effectiveness.

(5) Use pre-trained models: Transfer learning using pre-trained models on large datasets can accelerate model convergence and improve performance. (6) Increase the number of datasets

### Provided documents

Python source code

yolo11n.onnx model (without providing pytorch model)

Training map, P, R curve graph (in weights\results.png)

Test images (under test_img folder)

Please note that the dataset is provided in the dataset_address.xlsx file.


## Payment

Here is a pay link on Stripe ( https://buy.stripe.com/3cs8yP7sY87d0vu9AB ). Please contact me lonlonago@foxmail.com after funding $89, and I will send you a complete data files , thank you!

![111.png](111.png)
