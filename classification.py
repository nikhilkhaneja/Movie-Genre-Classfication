
import pandas as pd
import numpy as np
import nltk
import re
import sys
import csv
from PIL import Image
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge
from wordcloud import WordCloud, STOPWORDS
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

np.set_printoptions(threshold=sys.maxsize)
pd.options.mode.chained_assignment = None

def clean_text(text):
    # remove backslash-apostrophe
    text = re.sub("\'", "", text)
    # remove everything except alphabets
    text = re.sub("[^a-zA-Z]"," ",text)
    # remove whitespaces
    text = ' '.join(text.split())
    # convert text to lowercase
    text = text.lower()

    return text
def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()
  fdist = nltk.FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms)

  # visualize words and frequencies
  plt.figure(figsize=(12,15))
  ax = sns.barplot(data=d, x= "count", y = "word", palette = "BrBG")
  ax.set(ylabel = 'Word')
  plt.show()

def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

pd.set_option('display.max_colwidth', 300)
meta = pd.read_csv("D:\web\Data\es_data.csv")
meta['clean_plot'] = meta['Description'].apply(lambda x: clean_text(x))
print(meta.head(10))
genre = [0] * 22
print(meta["Genre"][0])
genres = []
gen = meta.Genre2.unique().tolist()
gen.remove('None')
print(gen)
for i in range (0,1000):
    if (meta.Drama[i] == 1):
        genre[0]+=1
    if (meta.Crime[i] == 1):
        genre[1]+=1
    if (meta.Adventure[i] == 1):
        genre[2]+=1
    if (meta.Romance[i] == 1):
        genre[3]+=1
    if (meta.SciFi[i] == 1):
        genre[4]+=1
    if (meta.War[i] == 1):
        genre[5]+=1
    if (meta.Family[i] == 1):
        genre[6]+=1
    if (meta.Music[i] == 1):
        genre[7]+=1
    if (meta.Comedy[i] == 1):
        genre[8]+=1
    if (meta.Mystery[i] == 1):
        genre[9]+=1
    if (meta.Musical[i] == 1):
        genre[10]+=1
    if (meta.Biography[i] == 1):
        genre[11]+=1
    if (meta.Action[i] == 1):
        genre[12]+=1
    if (meta.Western[i] == 1):
        genre[13]+=1
    if (meta.Thriller[i] == 1):
        genre[14]+=1
    if (meta.Horror[i] == 1):
        genre[15]+=1
    if (meta.FilmNoir[i] == 1):
        genre[16]+=1
    if (meta.Fantasy[i] == 1):
        genre[17]+=1
    if (meta.Sport[i] == 1):
        genre[18]+=1
    if (meta.Mystery[i] == 1):
        genre[19] += 1
    if (meta.History[i] == 1):
        genre[20] += 1

print(str(genre))
dict = {}

for key in gen:
    for value in genre:
        dict[key] = value
        genre.remove(value)
        break

all_genres_df = pd.DataFrame({'Genre': list(dict.keys()),
                              'Count': list(dict.values())})
g = all_genres_df.nlargest(columns="Count", n = 50)
plt.figure(figsize=(12,15))
ax = sns.barplot(data=g, x= "Count", y = "Genre",palette = "rocket")
ax.set(ylabel = 'Count')
plt.show()
freq_words(meta['clean_plot'], 100)
stop_words = set(stopwords.words('english'))
meta['clean_plot'] = meta['clean_plot'].apply(lambda x: remove_stopwords(x))
freq_words(meta['clean_plot'], 100)
#mask = np.array(Image.open('D:\img.jpg'))

stopworddd = ['a', "a's", 'able', 'about', 'above', 'according',"film",'accordingly', 'across', 'actually', 'after', 'afterwards', 'again', 'against', "ain't", 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear', 'appreciate', 'appropriate', 'are', "aren't", 'around', 'as', 'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away', 'awfully', 'b', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'both', 'brief', 'but', 'by', 'c', "c'mon", "c's", 'came', 'can', "can't", 'cannot', 'cant', 'cause', 'causes', 'certain', 'certainly', 'changes', 'clearly', 'co', 'com', 'come', 'comes', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 'contains', 'corresponding', 'could', "couldn't", 'course', 'currently', 'd', 'definitely', 'described', 'despite', 'did', "didn't", 'different', 'do', 'does', "doesn't", 'doing', "don't", 'done', 'down', 'downwards', 'during', 'e', 'each', 'edu', 'eg', 'eight', 'either', 'else', 'elsewhere', 'enough', 'entirely', 'especially', 'et', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'far', 'few', 'fifth', 'first', 'five', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 'four', 'from', 'further', 'furthermore', 'g', 'get', 'gets', 'getting', 'given', 'gives', 'go', 'goes', 'going',"things","electronic work","electronic","work",'gone', 'got', 'gotten', 'greetings', 'h', 'had', "hadn't", 'happens', 'hardly', 'has', "hasn't", 'have', "haven't", 'having', 'he', "he's", 'hello', 'help', 'hence', 'her', 'here', "here's", 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him', 'himself', 'his', 'hither', 'hopefully', 'how', 'howbeit', 'however', 'i', "i'd", "i'll", "i'm", "i've", 'ie', 'if', 'ignored', 'immediate', 'in', 'inasmuch', 'inc', 'indeed', 'indicate', 'indicated', 'indicates', 'inner', 'insofar', 'instead', 'into', 'inward', 'is', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'know', 'knows', 'known', 'l', 'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', "let's", 'like', 'liked', 'likely', 'little', 'look', 'looking', 'looks', 'ltd', 'm', 'mainly', 'many', 'may', 'maybe', 'me', 'mean', 'meanwhile', 'merely', 'might', 'more', 'moreover', 'most', 'mostly', 'much', 'must', 'my', 'myself', 'n', 'name', 'namely', 'nd', 'near', 'nearly', 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nine', 'no', 'nobody', 'non', 'none', 'noone', 'nor', 'normally', 'not', 'nothing', 'novel', 'now', 'nowhere', 'o', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own', 'p', 'particular', 'particularly', 'per', 'perhaps', 'placed', 'please', 'plus', 'possible', 'presumably', 'probably', 'provides', 'q', 'que', 'quite', 'qv', 'r', 'rather', 'rd', 're', 'really', 'reasonably', 'regarding', 'regardless', 'regards', 'relatively', 'respectively', 'right', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'she', 'should', "shouldn't", 'since', 'six', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying', 'still', 'sub', 'such', 'sup', 'sure', 't', "t's", 'take', 'taken', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', "that's", 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', "there's", 'thereafter', 'thereby',"place","moment", 'therefore', 'therein', 'theres', 'thereupon', 'these', 'they', "they'd", "they'll", "they're", "they've", 'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless', 'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used', 'useful', 'uses', 'using', 'usually', 'uucp', 'v', 'value', 'various', 'very', 'via', 'viz', 'vs', 'w', 'want', 'wants', 'was', "wasn't", 'way', 'we', "we'd", "we'll", "we're", "we've", 'welcome', 'well', 'went', 'were', "weren't", 'what', "what's", 'whatever', 'when', 'whence', 'whenever', 'where', "where's", 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', "who's", 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'willing', 'wish', 'with', 'within', 'without', "won't", 'wonder', 'would', 'would', "wouldn't", 'x', 'y', 'yes', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', 'z', 'zero',"made","time","thing","man","hand","men","word","thought","day","electronic work","found"]
adjectives = ['able', 'bad', 'best',"back", 'better', 'big', 'black', 'certain', 'clear', 'early', 'easy', 'free', 'full', 'good', 'great', 'hard', 'high', 'important', 'large', 'late', 'little', 'local', 'long', 'low', 'major', 'new', 'old', 'only', 'other', 'political', 'possible', 'public', 'real', 'recent', 'right', 'small', 'social', 'special', 'strong', 'sure', 'true', 'white', 'whole', 'young']
common_verbs = ['say', 'make', 'go', 'take', 'come', 'see', 'know', 'get', 'give', 'find', 'think', 'tell', 'become', 'show', 'leave', 'feel', 'put', 'bring', 'begin', 'keep', 'hold', 'write', 'stand', 'hear', 'let', 'mean', 'set', 'meet', 'run', 'pay', 'sit', 'speak', 'lie', 'lead', 'read', 'grow', 'lose', 'fall', 'send', 'build', 'draw', 'break', 'spend', 'cut', 'rise', 'drive', 'buy', 'wear', 'choose', 'said', 'made', 'went', 'took', 'came', 'saw', 'knew', 'got', 'gavefound', 'thought', 'told', 'became', 'showed', 'left', 'felt', 'put', 'brought', 'began', 'kept', 'held', 'wrote', 'stood', 'heard', 'let', 'meant', 'set', 'met', 'ran', 'paid', 'sat', 'spoke', 'lay', 'led', 'read', 'grew', 'lost', 'fell', 'sent', 'built', 'drew', 'broke', 'spent', 'cut', 'rose', 'drove', 'bought', 'wore', 'chose', 'gone', 'taken', 'seen', 'known', 'given', 'shown', 'written', 'spoken', 'lain', 'grown', 'fallen', 'drawn', 'broken', 'risen', 'driven', 'worn', 'chosen', 'wanted', 'used', 'worked', 'called', 'tried', 'asked', 'needed', 'seemed', 'helped', 'played', 'moved', 'lived', 'believed', 'happened', 'included', 'continued', 'changed', 'watched', 'followed', 'stopped', 'created', 'opened', 'walked', 'offered', 'remembered', 'appeared', 'served', 'died', 'stayed', 'reached', 'killed', 'raised', 'passed', 'decided', 'returned', 'explained', 'hoped', 'carried', 'received', 'agreed', 'covered', 'caused', 'closed', 'increased', 'reduced', 'noted', 'entered', 'shared', 'saved', 'protected', 'occurred', 'accepted', 'determined', 'prepared', 'argued', 'recognized', 'indicated', 'arrived', 'answered', 'compared', 'acted', 'studied', 'removed', 'sounded', 'formed', 'visited', 'avoided', 'imagined', 'finished', 'responded', 'maintained', 'contained', 'applied', 'treated', 'affected', 'worried', 'mentioned', 'improved', 'signed', 'existed', 'noticed', 'travelled', 'prevented', 'admitted', 'suffered', 'published', 'counted', 'achieved', 'announced', 'touched', 'attended', 'defined', 'introduced']

stopwords = stopworddd + adjectives + common_verbs
#wordclouds
for ge in gen:
    movies_for_wordcloud = meta[["ID","clean_plot",ge]]
    movies_for_wordcloud = movies_for_wordcloud[movies_for_wordcloud[ge] != 0]
    values = " ".join(map(str, (movies_for_wordcloud["clean_plot"].tolist())))
    wc = WordCloud(stopwords=stopwords,width=800,height=800,background_color = 'black').generate(values)
    plt.figure(figsize = (15,15),facecolor= None)
    plt.imshow(wc)
    plt.title(ge)
    plt.axis("off")
    plt.show()

def Convert(string):
    li = list(string.split(","))
    return li

meta['genre_new'] = meta['Genre'].apply(lambda x: Convert(x))
print(meta.head(10))
multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(meta['genre_new'])
y = multilabel_binarizer.transform(meta['genre_new'])


tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=1000)
xtrain, xval, ytrain, yval = train_test_split(meta['clean_plot'], y, test_size=0.1,random_state = 10)
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)

lr = Ridge(alpha = 0.15)
clf = OneVsRestClassifier(lr)
clf.fit(xtrain_tfidf, ytrain)

print(lr)
print(clf)

y_pred = clf.predict(xval_tfidf)
print(y_pred)
print(multilabel_binarizer.inverse_transform(y_pred)[52])


def infer_tags(q):
    q = clean_text(q)
    q = remove_stopwords(q)
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = clf.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)

for i in range(5):
    k = xval.sample(1).index[0]
    print("Movie: ", meta['Title'][k], "\nPredicted genre: ", infer_tags(xval[k])), print("Actual genre: ",meta['genre_new'][k], "\n")

scr = f1_score(yval, y_pred, average="micro")
print(scr)
