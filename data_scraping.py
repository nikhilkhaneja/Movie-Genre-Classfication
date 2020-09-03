
import bs4
import pandas as pd
import requests

url = 'https://www.imdb.com/search/title/?count=250&groups=top_1000&sort=user_rating'
url2  = 'https://www.imdb.com/search/title/?groups=top_1000&sort=user_rating,desc&count=250&start=251&ref_=adv_nxt'
url3 = 'https://www.imdb.com/search/title/?groups=top_1000&sort=user_rating,desc&count=250&start=501&ref_=adv_nxt'
url4 = 'https://www.imdb.com/search/title/?groups=top_1000&sort=user_rating,desc&count=250&start=751&ref_=adv_nxt'
def get_page_contents(url4):
    page = requests.get(url4, headers={"Accept-Language": "en-US"})
    return bs4.BeautifulSoup(page.text, "html.parser")

soup = get_page_contents(url4)

def numeric_value(movie, tag, class_=None, order=None):
    if order:
        if len(movie.findAll(tag, class_)) > 1:
            to_extract = movie.findAll(tag, class_)[order]['data-value']
        else:
            to_extract = None
    else:
        to_extract = movie.find(tag, class_)['data-value']

    return to_extract


def text_value(movie, tag, class_=None):
    if movie.find(tag, class_):
        return movie.find(tag, class_).text
    else:
        return


def nested_text_value(movie, tag_1, class_1, tag_2, class_2, order=None):
    if not order:
        return movie.find(tag_1, class_1).find(tag_2, class_2).text
    else:
        return [val.text for val in movie.find(tag_1, class_1).findAll(tag_2, class_2)[order]]


def extract_attribute(soup, tag_1, class_1='', tag_2='', class_2='',
                      text_attribute=True, order=None, nested=False):
    movies = soup.findAll('div', class_='lister-item-content')
    data_list = []
    for movie in movies:
        if text_attribute:
            if nested:
                data_list.append(nested_text_value(movie, tag_1, class_1, tag_2, class_2, order))
            else:
                data_list.append(text_value(movie, tag_1, class_1))
        else:
            data_list.append(numeric_value(movie, tag_1, class_1, order))

    return data_list


titles = extract_attribute(soup, 'a')
release = extract_attribute(soup, 'span', 'lister-item-year text-muted unbold')
audience_rating = extract_attribute(soup, 'span', 'certificate')
runtime = extract_attribute(soup, 'span', 'runtime')
genre = extract_attribute(soup, 'span', 'genre')
imdb_rating = extract_attribute(soup, 'div', 'inline-block ratings-imdb-rating', False)
votes = extract_attribute(soup, 'span' , {'name' : 'nv'}, False, 0)
earnings = extract_attribute(soup, 'span' , {'name' : 'nv'}, False, 1)
directors = extract_attribute(soup, 'p', '', 'a', '', True, 0, True)
actors = extract_attribute(soup, 'p', '', 'a', '', True, slice(1, 5, None), True)

desc= extract_attribute(soup, 'p', 'text-muted')

df_dict = {'Title': titles, 'Relase': release, 'Audience Rating': audience_rating,
           'Runtime': runtime, 'Genre': genre, 'IMDB Rating': imdb_rating,
           'Votes': votes, 'Box Office Earnings': earnings, 'Director': directors,
           'Actors': actors}
df = pd.DataFrame.from_dict(df_dict)
df['Genre'] = df['Genre'].str.replace('\n', '')
df['IMDB Rating'] = df['IMDB Rating'].str.replace('\n', '')
df['Relase'] = df['Relase'].str.replace('-', '')

df.to_csv('D:\web\data4.csv', mode='a', header=True)
pd.set_option("display.max_rows", None, "display.max_columns", None)
print("done")