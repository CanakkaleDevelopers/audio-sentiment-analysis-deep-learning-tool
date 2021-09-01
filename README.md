# DepSemo
## Audio sentiment analysis deep learning tool
A research tool for anybody can build, train, test and analysis deep learning models on audio data for the purpose of emotion classification.


**TR  Bu çalışmanın ileri aşamadaki sonuçları [Veri Bilimi](https://dergipark.org.tr/en/pub/veri) dergi'sinde (ISSN:2667-582X) bilimsel makale olarak yayımlanmak üzere hakem incelemesindedir.**


**EN The advanced results of this study are under peer review to be published as a scientific article in the [Veri Bilimi (Data Science)](https://dergipark.org.tr/en/pub/veri) journal (ISSN:2667-582X)**

## Features

- Download and unarchive auido emotion related datasets.
- Auto-Decompose dataset labels -meta data creation-.
- Create and save Audio Features like MFCC.
- Create and save Deep Learning models, for  audio sentiment analysis, audio emotion classification.
- Live monitor for training with TensorBoard.
- Test models before use.


## Installation

```
git clone https://github.com/COMUProjectTeam/audio-sentiment-analysis-deep-learning-tool
```

then go to directory

```
cd "DepSemo root directory path"
```

then install necessary packages using pip

```
pip install -r requirements.txt
```

for run program run app.py

```
python3 app.py or python app.py
```
---


# How to Use 

<img align="left" width="150" height="250" src="https://i.im.ge/2021/09/01/Q1Nter.png">

After running app.py from your terminal, go to http://127.0.0.1:5000/ with your browser. If everythings went right, you have to see DepSemo main page. You can navigate between modules via toolbar on left.


> Remember, you have to download dataset for create metadata, and have to train a model before test a model, so go one by one for create a classifier model!

<br />
<br />
<br />
<br />
<br />

---

## Known Datasets

Hosting public datasets in a AWS S3 bucket for fast download and a link that we're sure works.

| Dataset | Official Page |
| ------ | ------ |
| RAVDESS [1] | [Link](https://www.google.com/search?q=ravdess&rlz=1C1FKPE_trTR967TR967&oq=ravdess&aqs=chrome.0.35i39j69i59j0i512j0i20i263i512l2j0i512l5.2190j0j7&sourceid=chrome&ie=UTF-8)|
| SAVEE [2] | [Link](http://kahlan.eps.surrey.ac.uk/savee/)|
| Emo-DB [3] | [Link](http://emodb.bilderbar.info/start.html)|
| CREMA-D [4]  | [Link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4313618/)|


## Backends

Diffrent tools and frameworks has been used for accomplish this task.

| Used For | Framework & Tool  |
| ------ | ------ |
| FLASK | Web backend |
| KERAS | Deep learning API with TensorFlow backend|
| Librosa | Audio feature extractions and data augmentation|
| TensorBoard | Training live feedback GUI|
| SQLITE | Local database |
| AWS S3 | Cloud database |
| Pandas & Numpy | Generic usage |

## System Architecture
<img  src="https://i.im.ge/2021/09/02/Q1CFz0.png">

## Contributors

- [Emir Kivrak](https://github.com/emirkivrak)
- [Assoc. Prof. Dr. Bahadır Karasulu](https://scholar.google.com.tr/citations?user=NEhs3ttTIzkC&hl=tr)
- [Can Sözbir](https://github.com/cansozbir)
- [Atakan Türkay](https://github.com/atakanhr)

## License and Citation 

* Under MIT license.

## Screenshots
In Feature extraction page, we can extract desired audio features from per auido.
<img  width="500" height="500" src="https://i.im.ge/2021/09/01/Q1PurP.md.png">

<br/>
In training page we can set validation test split, batch size, epoch count etc.
<img  width="500" height="500" src="https://i.im.ge/2021/09/01/Q1PF51.png">


## Citation for datasets.

[1]     Livingstone SR, Russo FA. “The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)  A dynamic, multimodal set of facial and vocal expressions in North American English”. PIoS one, 13(5), e0196391, 2018.
[2]     Haq S, Jackson PJB. “Speaker-Dependent Audio-Visual Emotion Recognition (SAVEE)”. AVSP, 53-58, 2009.
[3]     Burkhardt F, Paescheke A, Rolfes M, Sendlmeier F, Weiss B.  “A database of German emotional speech”.  9th European Conference on Speech Communication and Technology, 2005.
[4]     Cao H, Copper DG, Keutmann MK, Gur RC, Nenkova A, Verma R. “CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset”. IEEE Transactions on Affective Computing, 5(4), 377-390, 2014.
