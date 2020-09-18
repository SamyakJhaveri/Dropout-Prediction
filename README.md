# Dropout Prediction on XGBoost
As part of Machine Learning Internship at SilverTouch Technologies

## Introduction
MOOC(Massive Open Online Courses) are free online courses for anyone to enroll in. It has been observed that such MOOC platforms, being designed for a wide audience 
of varied grasping potential, lack a certain degree of personalization that may be required in some groups of students. 

MOOC courses, especially in these testing times of lockdowns and social distancing, have come to the forefront of imparting quality education thus establishing 
themselves as an integral part of the educational systems all around the world. An addition to the customizability and personalization of the courses/platforms 
would help to engage students better and therefore reduce the chances of them dropping out of the courses should they find the courses difficult to complete or 
redundantly lengthy. This project aims to develop a machine learning solution by using dropout prediction and collaborative filtering recommendation engine to 
better analyse the data about a student’s activities on the MOOC platform and provide recommendations of course material accordingly.

##  The Dataset and Exploratory Data Analysis
The Knowledge Engineering Laboratory, Tsinghua University launched a large-scale Chinese education-oriented knowledge map, including high-quality terminology concepts 
in different areas of the various disciplines. The data in this site is based on the National Science and Technology Terminology Committee collated term public website 
terms online classification system, the use of Chinese Wikipedia, Baidu Encyclopedia and various Internet resources acquired. 

XuetangX, a Chinese MOOC learning platform initiated by Tsinghua University, was officially launched online on Oct 10th, 2013. In April 2014, XuetangX signed a contract 
with edX, one of the biggest global MOOC learning platforms co-founded by Harvard University and MIT, to acquire the exclusive authorization of edX's high-quality 
international courses. In December 2014, XuetangX signed the Memorandum of Cooperation with FUN, the national MOOC platform in France, to make bilateral effort in course
construction, platform development and other aspects. So far, there are more than 100 Chinese courses and over 260 international courses available on XuetangX. The dataset 
was used by researchers at the Thirty-Third AAAI Conference on Artificial Intelligence (AAAI-19)

The datasets provided by the Tsinghua University consists of student metadata and their web tracking logs on the XuetangX website.

It is  available at http://moocdata.cn/data/user-activity


## Dropout Prediction Model
Students' high dropout rate on MOOC platforms has been heavily criticized, and predicting their likelihood of dropout would be useful for maintaining and encouraging 
students' learning activities.

The dropout prediction dataset given by the website was in the form of large csv files compressed in a tar.gz format. After parsing this compressed file and extracting 
it to its original form, four files were obtained, namely - train_logs, train_truth, test_logs and test_truth

The train log and truth files were merged together to form a larger csv file(4.21 GB) and the same was done for the test files(1.77 GB).

Here, ‘truth; indicates whether a student dropped out of particular course - 1 means he/she dropped out, 0 means he/she continued the course.
The ‘action’ column consists of categorical data about the activity a user does, corresponding to the timestamp value in the ‘time’ column. These
track logs have 17 unique actions that it tracks such as ‘click_about’, ‘load_video’, ‘pause_video’, etc.

A preliminary EDA revealed that the dataset has information of roughly 155000 unique students studying a total of 247 courses from the MOOC platform.



