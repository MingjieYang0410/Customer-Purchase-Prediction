# Short-Video-Recommendation
### I. Guidance
**Step 1**: Build required repository by running:
```bash
sh ./scr/prepare_dir.sh
```


**Step 2**: Download the dataset and unzip the data files onto ./ddata/

https://pan.baidu.com/s/14uH_aHuNJ0hS6REr7jYuqQ   
code: u3vv


**Step 3**: Build the training set by running:
(Time Warning: 50 minutes)
```bash
sh ./scr/pre_processing.sh
```

**Step  4**: Training model by running:
```bash
python main_func.py
```
### II. Introduction
This project explores the effectiveness of ESSM 
architecture on the customer purchase prediction task. 

Customer purchase prediction aims at mining high potential
buyers for items to guide targeted advertising and business strategy 
generation. Unlike online CVR estimation, this task focuses on explainability 
and accuracy instead of serendipity and diversity. The reason is that the 
targeted advertisement like pushing messages and email advertisements 
hurt user's satisfaction for poor recommendations, and the formulation of business 
strategy requires explainable and trustworthy 
results. Due to these reasons, customer purchase prediction 
can be interpreted as a data mining task which digs out buyers 
for items and categories from users historical interactions.

ESSM is designed to solve the data sparsity
and the sample selection bias, which are the two main
issues for online CVR estimation. However, we observe the similar issues on 
customer purchase prediction task. First, the positive sample for user purchases are 
extremely sparse, which only takes account for 0.9% of the training data. Second, 
we don't have the label that effectively reflect the buying inclination 
of users. To be specific, the only label we have is joint probability of exposure, interaction and 
purchase, however, it is unreasonable to label an instance as negative just because corresponding item 
is not expose to user by the online recommendation system.

To mitigate these issues, we bring in another label indicating weather or not the item is exposed by the 
system and interacted by the user in the future. The number of positive samples takes account for more than
15% over the training data, which effectively eliminates the data sparsity problem and provides 
the embedding layer with more positive feedbacks. More importantly, the output of CVR network can be interpreted as the probability that 
the user will purchase the item given the item is already exposed to the user, which serves our purpose perfectly.

We observed 3% AUC gain on the main task compared to the single model, and we believe the improvement will be more
tremendous for online evaluation.


THE BEST HYPER-PARAMETERS OBSERVED:

DIEN for the subnetwork, GRU + DNN for CVR network

Learning Rate: 0.008

Embedding Size: 16

Label Smoothing: 0.1 for auxiliary loss

Behavior Sequence Length: 100

Best AUC: 0.92 ± 0.015

SUPPORTED MODELS:
DIEN, DIN, DNN, DeepFM, GRU + DNN, GRU + FM, ESSM

### III. Environment
Python 3.7

Tensorflow 2.3

Gensim 3.8.3 (Note: The latest version raises errors)

Scikit-learn 1.0.1

Pandas 1.3.5

### IV. References
Deep Interest Evolution Network for Click-Through Rate Prediction https://arxiv.org/pdf/1809.03672v1.pdf

Deep Interest Network for Click-Through Rate Prediction https://arxiv.org/pdf/1706.06978.pdf

Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate https://arxiv.org/pdf/1804.07931.pdf

DeepFM: A Factorization-Machine based Neural Network for CTR Prediction https://www.ijcai.org/Proceedings/2017/0239.pdf

https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0 Accessed 2021

https://github.com/mouna99/dien  Accessed 2021

https://github.com/bubblezhong/DIN_DIEN_tf2.0  Accessed 2021

https://github.com/StephenBo-China/DIEN-DIN  Accessed 2021

https://github.com/anzhizh/2019-taida-jdata-top3  Accessed 2021

