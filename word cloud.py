#for plotting images & adjusting colors
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from main import path_list, wordfreq_total, wordfreq_dict

def word_cloud_gen(wordfreq_dict,path_list):
    for i, dict in enumerate(wordfreq_dict):
        # create the WordCloud object
        wordcloud = WordCloud(min_word_length =3,
                              background_color='white',font_path = 'SourceHanSansTW-Regular.otf')

        # generate the word cloud
        wordcloud.generate_from_frequencies(dict)

        # Plot
        fig = plt.figure(figsize=(6, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        fig.savefig(f"png/{path_list[i][5:]}.png")


if __name__ == "__main__":
    word_cloud_gen(wordfreq_dict,path_list)
    path_total = ["csv/total_df.csv"]

    # Total Word Cloud
    wordcloud = WordCloud(min_word_length =3,
                                background_color='white',font_path = 'SourceHanSansTW-Regular.otf')
    # generate the word cloud
    wordcloud.generate_from_frequencies(wordfreq_total)

    # Plot
    fig = plt.figure(figsize=(6, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    fig.savefig(f"png/{path_list[0][5:]}.png")