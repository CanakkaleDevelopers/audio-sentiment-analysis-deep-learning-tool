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

<img align="left" width="150" height="300" src="https://i.im.ge/2021/09/01/Q1Nter.png">

After running app.py from your terminal, go to http://127.0.0.1:5000/ with your browser. If everythings went right, you have to see DepSemo main page. You can navigate between modueles via toolbar on left.


This text you see here is *actually- written in Markdown! To get a feel
for Markdown's syntax, type some text into the left window and
watch the results in the right.

## Tech

Dillinger uses a number of open source projects to work properly:



## Installation

Dillinger requires [Node.js](https://nodejs.org/) v10+ to run.

Install the dependencies and devDependencies and start the server.

```sh
cd dillinger
npm i
node app
```

For production environments...

```sh
npm install --production
NODE_ENV=production node app
```

## Plugins

Dillinger is currently extended with the following plugins.
Instructions on how to use them in your own application are linked below.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Development

Want to contribute? Great!

Dillinger uses Gulp + Webpack for fast developing.
Make a change in your file and instantaneously see your updates!

Open your favorite Terminal and run these commands.

First Tab:

```sh
node app
```

Second Tab:

```sh
gulp watch
```

(optional) Third:

```sh
karma test
```

#### Building for source

For production release:

```sh
gulp build --prod
```

Generating pre-built zip archives for distribution:

```sh
gulp build dist --prod
```

## Docker

Dillinger is very easy to install and deploy in a Docker container.

By default, the Docker will expose port 8080, so change this within the
Dockerfile if necessary. When ready, simply use the Dockerfile to
build the image.

```sh
cd dillinger
docker build -t <youruser>/dillinger:${package.json.version} .
```

This will create the dillinger image and pull in the necessary dependencies.
Be sure to swap out `${package.json.version}` with the actual
version of Dillinger.

Once done, run the Docker image and map the port to whatever you wish on
your host. In this example, we simply map port 8000 of the host to
port 8080 of the Docker (or whatever port was exposed in the Dockerfile):

```sh
docker run -d -p 8000:8080 --restart=always --cap-add=SYS_ADMIN --name=dillinger <youruser>/dillinger:${package.json.version}
```

> Note: `--capt-add=SYS-ADMIN` is required for PDF rendering.

Verify the deployment by navigating to your server address in
your preferred browser.

```sh
127.0.0.1:8000
```

## License

MIT
