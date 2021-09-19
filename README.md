# NLP-wordcloud

## About the project
This project extract theme and word frequency from texts to generate wordclouds of each document and the total text. This was one of my earlier project using VS code; therefore, there are some inmature usage of import and `.py` files in general.


## Getting Started
### Prerequisites
- **Word Separation (WS)**: Chinese/Japanese text struggle with the word separation problem due to the lack of spaces between each word. This project support `jieba`, `ckiptagger`, and `Articut` packages to perform word separation. (For `ckiptagger` please download the model files from [this link](https://github.com/ckiplab/ckiptagger) and add in the `data` directory)
- **Data structure**: `numpy` and `pandas` packages as the structure for processing data.
- **Wordcloud**: For the plotting of wordcloud, I used the `wordcloud` and `matplotlib.pyplot` packages to generate the plot. Please also install a `.oft` file for the text style of the wordcloud.


### Clone the repo
```sh
git clone https://github.com/enjuichang/NLP-wordcloud.git
```

### Data input

- **Corpus**: This needs to be in the format of `.txt`. If the text is in `.doc` format, please transform the document into `.txt` and put into the `text` directory.

- **Theme**: If theme classification is needed, please create a `.json` file inlcuding all the category types into `data` directory.

- **stopwords**: If `ckiptagger` and `jieba` packages were used, please use `stopwords-tw.txt` to exclude function words to increase word separation efficiency.

## Roadmap
- `main.py`: This python file includes the data download process, data preprocessing, and the generation of bag-of-words model (BoW) for word frequency. 
- `wordcloud.py`: This python file generates the word clouds for each document and all documents through the word frequencies.
- `piechart.py`: This python file generates the pie charts for each document and all documents through the word frequencies.

### Data output
#### CSV
- *TXTNAME*_df.csv: Records the word frequencies for each document and all documents
- *TXTNAME*_theme.csv: Records the theme frequencies for each document and all documents
#### PNG
- *TXTNAME*.png: Wordcloud figures for each document and all documents
- *TXTNAME*_pie.png: Pie chart of the theme of each document and all documents
 
