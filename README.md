# Dunlin Classifier
## About the Project

 This program focuses on automating the annotation process for studying the natural behavior of dunlins
 (Calidris alpina) in the Dutch Wadden Sea mudflats.

Filming dunlins in their natural habitat provides valuable data on behavior, diet, prey abundance, and individual
differences. However, manual annotation of videos is time-consuming and prone to observer bias. The need to play back
videos at slower speeds for accurate annotation further adds to the time investment.

Automating the annotation process using machine learning can significantly reduce the time required, allowing
researchers to focus on other tasks. It also enables the analysis of larger datasets with more consistent annotations,
minimizing human error. Existing tracking programs like TRex and Tracktor lack a built-in classification system, and
pretrained models are often designed for static backgrounds with mammals, insects, or fish.

This project aims to develop a machine learning model tailored for dunlins, capable of generating behavior time series
from wild dunlin videos.


### Prerequisites

conda 22.11+

python 3.7+

### Install Miniconda ###

Anaconda is a Python environment manager that makes it easy to install SLEAP and its necessary dependencies without affecting other Python software on your computer.

Miniconda is a lightweight version of Anaconda. To install it:

Go to: https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links

Download the latest version for your OS.

Follow the installer instructions.
### Installation

make a new conda environment using the requirements.txt:

```bash
conda create --name dunlin_classifier --file requirements.txt
```


