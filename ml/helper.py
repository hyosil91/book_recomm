# -*- coding: utf-8 -*-
import io
import os
import json
import distutils.dir_util
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp


def threshold_interactions_df(df, row_name, col_name, row_min, col_min):
    """Limit interactions df to minimum row and column interactions.

    Parameters
    ----------
    df : DataFrame
        DataFrame which contains a single row for each interaction between
        two entities. Typically, the two entities are a user and an item.
    row_name : str
        Name of column in df which corresponds to the eventual row in the
        interactions matrix.
    col_name : str
        Name of column in df which corresponds to the eventual column in the
        interactions matrix.
    row_min : int
        Minimum number of interactions that the row entity has had with
        distinct column entities.
    col_min : int
        Minimum number of interactions that the column entity has had with
        distinct row entities.
    Returns
    -------
    df : DataFrame
        Thresholded version of the input df. Order of rows is not preserved.

    Examples
    --------

    df looks like:

    user_id | item_id
    =================
      1001  |  2002
      1001  |  2004
      1002  |  2002

    thus, row_name = 'user_id', and col_name = 'item_id'

    If we were to set row_min = 2 and col_min = 1, then the returned df would
    look like

    user_id | item_id
    =================
      1001  |  2002
      1001  |  2004

    """

    n_rows = df[row_name].unique().shape[0]
    n_cols = df[col_name].unique().shape[0]
    sparsity = float(df.shape[0]) / float(n_rows*n_cols) * 100
    print('Starting interactions info')
    print('Number of rows: {}'.format(n_rows))
    print('Number of cols: {}'.format(n_cols))
    print('Sparsity: {:4.3f}%'.format(sparsity))

    done = False
    while not done:
        starting_shape = df.shape[0]
        col_counts = df.groupby(row_name)[col_name].count()
        df = df[~df[row_name].isin(col_counts[col_counts < col_min].index.tolist())]
        row_counts = df.groupby(col_name)[row_name].count()
        df = df[~df[col_name].isin(row_counts[row_counts < row_min].index.tolist())]
        ending_shape = df.shape[0]
        if starting_shape == ending_shape:
            done = True

    n_rows = df[row_name].unique().shape[0]
    n_cols = df[col_name].unique().shape[0]
    sparsity = float(df.shape[0]) / float(n_rows*n_cols) * 100
    print('Ending interactions info')
    print('Number of rows: {}'.format(n_rows))
    print('Number of columns: {}'.format(n_cols))
    print('Sparsity: {:4.3f}%'.format(sparsity))
    return df

def get_df_matrix_mappings(df, row_name, col_name):
    """Map entities in interactions df to row and column indices

    Parameters
    ----------
    df : DataFrame
        Interactions DataFrame.
    row_name : str
        Name of column in df which contains row entities.
    col_name : str
        Name of column in df which contains column entities.

    Returns
    -------
    rid_to_idx : dict
        Maps row ID's to the row index in the eventual interactions matrix.
    idx_to_rid : dict
        Reverse of rid_to_idx. Maps row index to row ID.
    cid_to_idx : dict
        Same as rid_to_idx but for column ID's
    idx_to_cid : dict
    """


    # Create mappings
    rid_to_idx = {}
    idx_to_rid = {}
    for (idx, rid) in enumerate(df[row_name].unique().tolist()):
        rid_to_idx[rid] = idx
        idx_to_rid[idx] = rid

    cid_to_idx = {}
    idx_to_cid = {}
    for (idx, cid) in enumerate(df[col_name].unique().tolist()):
        cid_to_idx[cid] = idx
        idx_to_cid[idx] = cid

    return rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid

def df_to_matrix(df, row_name, col_name):
    """Take interactions dataframe and convert to a sparse matrix

    Parameters
    ----------
    df : DataFrame
    row_name : str
    col_name : str

    Returns
    -------
    interactions : sparse csr matrix
    rid_to_idx : dict
    idx_to_rid : dict
    cid_to_idx : dict
    idx_to_cid : dict

    """

    rid_to_idx, idx_to_rid,\
        cid_to_idx, idx_to_cid = get_df_matrix_mappings(df,
                                                        row_name,
                                                        col_name)

    def map_ids(row, mapper):
        return mapper[row]

    I = df[row_name].apply(map_ids, args=[rid_to_idx]).to_numpy()
    J = df[col_name].apply(map_ids, args=[cid_to_idx]).to_numpy()
    V = np.ones(I.shape[0])
    interactions = sp.coo_matrix((V, (I, J)), dtype=np.float64)
    interactions = interactions.tocsr()
    return interactions, rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid

def train_test_split(interactions, split_count, fraction=None):
    """
    Split recommendation data into train and test sets

    Params
    ------
    interactions : scipy.sparse matrix
        Interactions between users and items.
    split_count : int
        Number of user-item-interactions per user to move
        from training to test set.
    fractions : float
        Fraction of users to split off some of their
        interactions into test set. If None, then all
        users are considered.
    """
    # Note: likely not the fastest way to do things below.
    train = interactions.copy().tocoo()
    test = sp.lil_matrix(train.shape)

    if fraction:
        try:
            user_index = np.random.choice(
                np.where(np.bincount(train.row) >= split_count * 2)[0],
                replace=False,
                size=np.int64(np.floor(fraction * train.shape[0]))
            ).tolist()
        except:
            print(('Not enough users with > {} '
                  'interactions for fraction of {}')\
                  .format(2*split_count, fraction))
            raise
    else:
        user_index = range(train.shape[0])

    train = train.tolil()

    for user in user_index:
        test_interactions = np.random.choice(interactions.getrow(user).indices,
                                        size=split_count,
                                        replace=False)
        train[user, test_interactions] = 0.
        # These are just 1.0 right now
        test[user, test_interactions] = interactions[user, test_interactions]


    # Test and training are truly disjoint
    assert(train.multiply(test).nnz == 0)
    return train.tocsr(), test.tocsr(), user_index

def write_json(data, fname):
    def _conv(o):
        if isinstance(o, np.int64):
            return int(o)
        raise TypeError

    parent = os.path.dirname(fname)
    distutils.dir_util.mkpath("./arena_data/" + parent)
    with io.open("./arena_data/" + fname, "w", encoding="utf8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)

def load_json(fname):
    with open(fname) as f:
        json_obj = json.load(f)

    return json_obj

def debug_json(r):
    print(json.dumps(r, ensure_ascii=False, indent=4))

def remove_seen(seen, l):
    seen = set(seen)
    return [x for x in l if not (x in seen)]

def most_popular(playlists, col, topk_count):
    c = Counter()

    for doc in playlists:
        c.update(doc[col])

    topk = c.most_common(topk_count)
    return c, [k for k, v in topk]

def load_data(train_path, qst_path):
    SEED = 123

    books = pd.read_csv('/content/drive/My Drive/coc_contest/books_clean.csv')
    books.rename(columns={'제어번호': 'book_id', '저자': 'author', 'ISBN번호': 'isbn', '분류기호': 'class_no', '발행처': 'publisher', '발행년도': 'pub_year', '제목': 'title'}, inplace=True)
    #books.rename(columns={'제어번호': 'book_id'}, inplace=True)
    books['book_id'] = books['book_id'].astype(str)
    
    users = pd.read_json(train_path, typ='frame')
    users['user_id'] = users['user_id'].astype(str)
    
    qst_list = pd.read_json(qst_path, typ='frame')
    qst_list['user_id'] = qst_list['user_id'].astype(str)
    
    qst_sample = qst_list.sample(n=150, random_state=SEED)
    qst_sample = qst_sample.reset_index()
    
    qst_sample2 = qst_list[qst_list['user_id'].isin(qst_sample['user_id']) == False]
    qst_sample2 = qst_sample2.reset_index()
    
    all_users = users.append(qst_sample)
    all_users = all_users.append(qst_sample2)
    all_users = all_users.reset_index()
    
    #users = users.drop(['index'], axis=1)
    #qst_sample = qst_sample.drop(['index'], axis=1)
    #qst_sample2 = qst_sample2.drop(['index'], axis=1)
    #all_users = all_users.drop(['level_0', 'index'], axis=1)
    
    del qst_list
    
    return books, users, qst_sample, qst_sample2, all_users

def load_data2(train_path, qst_path, ans_path):
    
    users = pd.read_json(train_path, typ='frame')
    users['user_id'] = users['user_id'].astype(str)
    
    qst_list = pd.read_json(qst_path, typ='frame')
    qst_list['user_id'] = qst_list['user_id'].astype(str)

    ans_list = pd.read_json(ans_path, typ='frame')
    ans_list['user_id'] = ans_list['user_id'].astype(str)
    
    all_users = users.append(qst_list)
    all_users = all_users.reset_index()
    
    all_users = all_users.drop(['index'], axis=1)
    
    return users, qst_list, ans_list, all_users

def get_userbook_map(userlist):
	userlist_df = userlist[['user_id', 'books']]

	# unnest songs
	userlst_song_map_unnest = np.dstack((
	    np.repeat(userlist_df.user_id.values, list(map(len, userlist_df.books))), 
	    np.concatenate(userlist_df.books.values))
	)

	# unnested 데이터프레임 생성
	userlist_df = pd.DataFrame(data=userlst_song_map_unnest[0], columns=userlist_df.columns)
	userlist_df.rename(columns={'books': 'book_id'}, inplace=True)
	  
	# unnest 객체 제거
	del userlst_song_map_unnest  
	  
	return userlist_df      

class Evaluator:
    def load_json(self, fname):
      with open(fname) as f:
          json_obj = json.load(f)

      return json_obj

    def _idcg(self, l):
        return sum((1.0 / np.log(i + 2) for i in range(l)))

    def __init__(self):
        self._idcgs = [self._idcg(i) for i in range(101)]
        self.hit_cnt = 0

    def _ndcg(self, gt, rec):
        dcg = 0.0
        for i, r in enumerate(rec):
            if r in gt:
                dcg += 1.0 / np.log(i + 2)
                self.hit_cnt += 1

        return dcg / self._idcgs[len(gt)]

    def _eval(self, gt_fname, rec_fname):
        gt_playlists = self.load_json(gt_fname)
        gt_dict = {g["id"]: g for g in gt_playlists}
        #rec_playlists = self.load_json(rec_fname)        
        # dataframe
        rec_playlists = rec_fname

        gt_ids = set([g["id"] for g in gt_playlists])
        #rec_ids = set([r["id"] for r in rec_playlists])
        rec_ids = set([val for val in rec_playlists['id']])

        #if gt_ids != rec_ids:
        #    raise Exception("결과의 플레이리스트 수가 올바르지 않습니다.")

        #rec_song_counts = [len(p["songs"]) for p in rec_playlists]
        #rec_tag_counts = [len(p["tags"]) for p in rec_playlists]

        rec_song_counts = [len(songs) for songs in rec_playlists['songs']]
        rec_tag_counts = [len(tags) for tags in rec_playlists['tags']]

        #if set(rec_song_counts) != set([100]):
        #    raise Exception("추천 곡 결과의 개수가 맞지 않습니다.")

        if set(rec_tag_counts) != set([10]):
            raise Exception("추천 태그 결과의 개수가 맞지 않습니다.")

        rec_unique_song_counts = [len(set(songs)) for songs in rec_playlists['songs']]
        rec_unique_tag_counts = [len(set(tags)) for tags in rec_playlists['tags']]

        if set(rec_unique_song_counts) != set([100]):
            raise Exception("한 플레이리스트에 중복된 곡 추천은 허용되지 않습니다.")

        if set(rec_unique_tag_counts) != set([10]):
            raise Exception("한 플레이리스트에 중복된 태그 추천은 허용되지 않습니다.")

        music_ndcg = 0.0
        tag_ndcg = 0.0

        #for rec in rec_playlists:
        for idx, rec in rec_playlists.iterrows():
            gt = gt_dict[rec["id"]]
            music_ndcg += self._ndcg(gt["songs"], rec["songs"][:100])
            tag_ndcg += self._ndcg(gt["tags"], rec["tags"][:10])

        music_ndcg = music_ndcg / len(rec_playlists)
        tag_ndcg = tag_ndcg / len(rec_playlists)
        score = music_ndcg * 0.85 + tag_ndcg * 0.15

        return music_ndcg, tag_ndcg, score

    def evaluate(self, gt_fname, rec_fname):
        try:
            music_ndcg, tag_ndcg, score = self._eval(gt_fname, rec_fname)
            print(f"Hit Count: {self.hit_cnt}")
            print(f"Music nDCG: {music_ndcg:.6}")
            print(f"Tag nDCG: {tag_ndcg:.6}")
            print(f"Score: {score:.6}")
        except Exception as e:
            print(e)

def remake_songs_df(listsong_map, all_songs_df):  
  uniq_songs = listsong_map.song_id.drop_duplicates()
  uniq_songs_df = pd.DataFrame(uniq_songs, columns=['song_id'])
  songs_df = pd.merge(uniq_songs_df, all_songs_df, on='song_id')

  del uniq_songs, uniq_songs_df
  
  return songs_df            