
import lxml
import re
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from requests import get
url = 'https://www.imdb.com/search/title/?count=250&groups=top_1000&sort=user_rating'
url2  = 'https://www.imdb.com/search/title/?groups=top_1000&sort=user_rating,desc&count=250&start=251&ref_=adv_nxt'
url3 = 'https://www.imdb.com/search/title/?groups=top_1000&sort=user_rating,desc&count=250&start=501&ref_=adv_nxt'
url4 = 'https://www.imdb.com/search/title/?groups=top_1000&sort=user_rating,desc&count=250&start=751&ref_=adv_nxt'

class plot(object):
	"""docstring for IMDB"""
	def __init__(self, url4):
		super(IMDB, self).__init__()
		page = get(url4)

		self.soup = BeautifulSoup(page.content, 'lxml')


	def bodyContent(self):
		content = self.soup.find(id="main")
		return content.find_all("div", class_="lister-item mode-advanced")

	def movieData(self):
		movieFrame = self.bodyContent()
		movieTitle = []
		movieDescription = []

		for movie in movieFrame:
			movieFirstLine = movie.find("h3", class_="lister-item-header")
			movieTitle.append(movieFirstLine.find("a").text)
			movieDescription.append(movie.find_all("p", class_="text-muted")[-1].text.lstrip())
		movieData = [movieTitle,movieDescription]
		df_dict = {'Title': movieTitle, 'Description': movieDescription}
		df = pd.DataFrame.from_dict(df_dict)
		df.to_csv('D:\web\plotdata4.csv', mode='a', header=True)
id1 = plot(url4)
id1.movieData()
