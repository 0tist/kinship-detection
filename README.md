# Kinship-Detection
# What is it ?
It is a model built using state of the art VGG Face model to predict if two people are related based solely on images of their faces
# Why ?
VISUAL kinship recognition has an abundance of practical uses, such as issues of human trafficking and in missing children detection, problems really valuable in today’s refugee crises around the world, and social media platforms. 

Use cases exist for the academic world as well, whether for machine vision (e.g., reducing the search space in large-scale face retrieval) or a different field entirely (e.g. historical & genealogical lineage studies)
# How does it work ?
My model is based on the principles of Siamese networks.
![alt text](https://miro.medium.com/max/1400/1*8Nsq1BYQCuj9giAwltDubQ.png)

* It's an implementation of CNN’S VGG16 called VGGface, that extracts facial features in the form of a vector.

* I get the feature vector for all images and then I concatenate the feature vector of designated images of a pair to form a combined feature vector for the pair,I do this for each pair
![alt text](https://i0.wp.com/sefiks.com/wp-content/uploads/2019/04/vgg-face-architecture.jpg?w=1805&ssl=1)

* I then use different similarity measures and define a threshold line for our class segregation 

The idea is input images according to labels generated from our generated labelled dataset(related and unrelated) to extract facial feature predictors using the VGG-Face model.
These features are then concatenated to produce a single feature vector of the same length which is then fed into a CNN Network to classify this combined data using the labels of our dataset.

# Dataset
The data is a subset of the  [Families In the Wild](https://web.northeastern.edu/smilelab/fiw/), the largest and most comprehensive image database for automatic kinship recognition , which had been uploaded along with my program

 ### The data set contained the following files: ###
* train-faces.zip - the training set is divided in Families (FXXXX), then individuals (MIDX). Images in the same MIDX folder belong to the same person. Images in the same FXXXX folder belong to the same family.
* train.csv - training labels. Remember, not every individual in a family shares a kinship relationship. For example, a mother and father are kin to their children, but not to each other.
* test-faces.zip - the test set contains face images of unknown individuals
* this dataset has images of 471 families, with 2320 distinct people having a total of 12348 pictures.
* the goal is to predict if each pair of images in test-faces are related (one) or not (zero).


# How to implement/run it ?
#### My code is basically the model architecture for the implementations of kinship detection

* you can run this code by downloading the dataset provided in this repo in the train folder and then training the model provided in the kinship-detection-with-vgg16 ipynb file on that dataset by just executing the kinship-detection-with-vgg16 ipynb file 

* make sure that the train folder and the kinship-detection-with-vgg16 ipynb file are saved in the same folder for proper implmentation

#### you can get the output/predicitions on unseen images by using the following code(already included in the ipynb):
```python
test_path = "../to_be_predicted/"

def chunker(seq, size=32):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


from tqdm import tqdm

submission = pd.DataFrame()

predictions = []

for batch in tqdm(chunker(submission.img_pair.values)):
    X1 = [x.split("-")[0] for x in batch]
    X1 = np.array([read_img(test_path + x) for x in X1])

    X2 = [x.split("-")[1] for x in batch]
    X2 = np.array([read_img(test_path + x) for x in X2])

    pred = model.predict([X1, X2]).ravel().tolist()
    predictions += pred

submission['is_related'] = predictions

submission.to_csv("vgg_face.csv", index=False)
```
#### in order to get predicitions on a set of photos->
* you store them together in a folder and initialize test_path with the path of that folder 
* the predicitons corresponding to your photos will be stored in the form of a csv file named as "vgg_face.csv"

# Refernces 
VGG Face: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/

One Shot Learning and Siamese Network: https://machinelearningmastery.com/one-shot-learning-with-siamese-networks-contrastive-and-triplet-loss-for-face-recognition/
