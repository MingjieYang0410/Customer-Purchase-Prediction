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


### II. Environment
Python 3.7

Tensorflow 2.3

Gensim 3.8.3 (Note: The latest version raises errors)

Scikit-learn 1.0.1

Pandas 1.3.5

### III. References
https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0 Accessed 2021

https://github.com/mouna99/dien  Accessed 2021

https://github.com/bubblezhong/DIN_DIEN_tf2.0  Accessed 2021

https://github.com/StephenBo-China/DIEN-DIN  Accessed 2021

https://github.com/anzhizh/2019-taida-jdata-top3  Accessed 2021

