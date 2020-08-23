from flask import Flask, jsonify, request, render_template
from buffalo.algo.als import ALS
from buffalo.misc import aux, log
from buffalo.algo.options import ALSOption
import buffalo.data
from buffalo.data.mm import MatrixMarketOptions

import numpy as np
import pandas as pd
import helper as hp
from scipy.io import mmwrite
from scipy.io import mmread
from scipy.sparse import csr_matrix
import json
import random
import requests
from xml.etree import ElementTree
from xml.etree.ElementTree import parse

app = Flask(__name__)

users = pd.read_json('./model/users.json', typ='frame')
books = pd.read_json('./model/books.json', typ='frame')
#books = pd.read_excel('./model/books.xlsx')
books['ISBN'] = books['ISBN'].astype(str)
books['권'] = books['권'].fillna('')

userbook_map = hp.get_userbook_map(users)
user_items, uid_to_idx, idx_to_uid, mid_to_idx, idx_to_mid = hp.df_to_matrix(userbook_map, 'user_id', 'book_id')
iid = list(idx_to_mid.values())
uid = list(idx_to_uid.values())

#model = ALS(opt, data_opt=data_opt)
model = ALS()
model.load('./model/als.optimize.bin')

headers = {
	    'X-Naver-Client-Id': '',
	    'X-Naver-Client-Secret': '',
	}

def get_bookimage(params):
	response = requests.get('https://openapi.naver.com/v1/search/book_adv.xml', headers=headers, params=params)
	root = ElementTree.fromstring(response.text)
	
	img_url = ''
	if root.tag == 'rss':
		for node in root:
			item = node.find("item")
			image = item.find("image")
			img_url = image.text

	return img_url

@app.route("/")
def list_user_history():
	
	idx = random.randint(0, len(users) - 1)
	user = users.iloc[idx]

	if user['user_id'][0] == 'm':
		user['gender'] = '남성'
		user['color'] = '#007bff'
	else:
		user['gender'] = '여성'
		user['color'] = '#e83e8c'

	arr = user['user_id'].split('_')
	if len(arr) == 3:
		user['theme'] = '실용'
	else:
		user['theme'] = '교양'

	user['age'] = user['user_id'][1:3]

	history = pd.DataFrame(user['books'], columns=['ISBN'])
	history['ISBN'] = history['ISBN'].astype(str)
	#print(history.head())
	
	history = pd.merge(history, books, on='ISBN')

	img_list = []
	for val in history['ISBN']:	
		#book = history.iloc[i]
		#val = book['ISBN']
		params = (
		    ('d_isbn', val),
		    ('start', '1')
		)

		img_url = get_bookimage(params)		
		img_list.append(img_url)

	history['image'] = pd.Series(img_list)
	temp_dict = history.to_dict(orient='records')
	#print(history.head())

	return render_template("user_history.html", user=user, history=temp_dict)

@app.route("/recomm", methods=['GET'])
def recommend_book():

	user_id = request.args.get('user_id', '')
	idx = uid_to_idx[user_id]
	user = users.iloc[idx]

	if user['user_id'][0] == 'm':
		user['gender'] = '남성'
		user['color'] = '#007bff'
	else:
		user['gender'] = '여성'
		user['color'] = '#e83e8c'

	arr = user['user_id'].split('_')
	if len(arr) == 3:
		user['theme'] = '실용'
	else:
		user['theme'] = '교양'

	user['age'] = user['user_id'][1:3]

	seen = user['books']
	#print(seen)

	pool = iid.copy()
	for val in seen:
		pool.remove(val)
	
	recomm_books = model.topk_recommendation(user_id, topk=10, pool=pool)

	pred = pd.DataFrame(recomm_books, columns=['ISBN'])
	pred['ISBN'] = pred['ISBN'].astype(str)
	
	pred = pd.merge(pred, books, on='ISBN')

	img_list = []
	for val in pred['ISBN']:	
		params = (
		    ('d_isbn', val),
		    ('start', '1')
		)

		img_url = get_bookimage(params)		
		img_list.append(img_url)

	pred['image'] = pd.Series(img_list)
	temp_dict = pred.to_dict(orient='records')
	
	return render_template("recomm.html", user=user, recomm=temp_dict)