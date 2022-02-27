from util import *
import pandas as pd
import datetime


def map_to_id(df, col_name):
    """
    Function map categorical features from 0 to the number of unique values.
    :param df: Original date set.
    :param col_name: The name of selected columns to map.
    :return: Processed data, the dict for each features.
    """
    for col in col_name:
        key = sorted(df[col].unique().tolist())
        dict_ = dict(zip(key, range(1, len(key) + 1)))  # 为了给mask留位置 否则0号会被严重影响
        df.loc[:, col] = df[col].map(lambda x: dict_[x])

    num_items = len(sorted(df["item_sku_id"].drop_duplicates(keep='first'))) + 1
    num_cats = len(sorted(df["item_third_cate_cd"].drop_duplicates(keep='first'))) + 1
    num_sex = len(sorted(df["sex"].drop_duplicates(keep='first'))) + 1
    num_ulevel = len(sorted(df["user_level"].drop_duplicates(keep='first'))) + 1
    num_atype = len(sorted(df["action_type"].drop_duplicates(keep='first'))) + 1
    num_city = len(sorted(df["city"].drop_duplicates(keep='first'))) + 1

    num_province = len(sorted(df["province"].drop_duplicates(keep='first'))) + 1
    num_county = len(sorted(df["county"].drop_duplicates(keep='first'))) + 1
    num_brand_code = len(sorted(df["brand_code"].drop_duplicates(keep='first'))) + 1
    num_shope = len(sorted(df["shop_id"].drop_duplicates(keep='first'))) + 1
    num_vender = len(sorted(df["vender_id"].drop_duplicates(keep='first'))) + 1

    temp = df[["item_sku_id", "item_third_cate_cd"]].sort_values("item_sku_id").drop_duplicates(subset='item_sku_id',
                                                                                                keep='first')
    cat_list = temp["item_third_cate_cd"].tolist()
    return num_items, num_cats, num_sex, num_ulevel, num_atype, num_city, \
           num_province, num_county, num_brand_code, num_shope, num_vender, cat_list


def _label_trans(x, dic_):
    if x in dic_:
        return 1
    else:
        return 0


def sliding_window_2_basic_form(df, label_start, label_end, inter_start, inter_end, fea_end):
    """
    Sliding window sampling:
    Feature Range: Defining the range to obtain statistical features.
    Interaction Range: Defining the range to sample user-item pairs a.k.a. instances
    Label Range: Defining how the sampled user-item pairs will be labeled.
    :param df: Original data set.
    :param label_start: Start date of the label range.
.    :param label_end: End date of the label range.
    :param inter_start: Start date of the interaction range.
    :param inter_end:  End date of the interaction range.
    :param fea_end: End date of the feature range.
    :return: Data set with the basic form (raw instances with corresponding labels).
    """
    fea_list = []
    all_data = []
    for i in range(len(label_start)):
        # get times
        lb_st = df.loc[(df['month'] == label_start[i][0]) & (df['day'] == label_start[i][1]), 'month_day'].values[0]
        lb_en = df.loc[(df['month'] == label_end[i][0]) & (df['day'] == label_end[i][1]), 'month_day'].values[0]
        cand_st = df.loc[(df['month'] == inter_start[i][0]) & (df['day'] == inter_start[i][1]), 'month_day'].values[0]
        cand_en = df.loc[(df['month'] == inter_end[i][0]) & (df['day'] == inter_end[i][1]), 'month_day'].values[0]
        fea_position = df.loc[(df['month'] == fea_end[i][0]) & (df['day'] == fea_end[i][1]), 'month_day'].values[0]

        cand_bool = (df['month_day'] >= cand_st) & (df['month_day'] <= cand_en)
        label_bool = (df['month_day'] >= lb_st) & (df['month_day'] <= lb_en) & (df['action_type'] == 2)
        label_bool_click = (df['month_day'] >= lb_st) & (df['month_day'] <= lb_en)

        df_inter = df.loc[cand_bool].copy()  # get potential interactions
        df_inter = df_inter[['user_log_acct', 'item_sku_id', 'month_day']].copy()
        df_inter = df_inter.drop_duplicates(subset=['user_log_acct', 'item_sku_id'])
        df_inter = df_inter.loc[(df_inter.item_sku_id.isnull() == False)]  # process

        df_label = df.loc[label_bool].copy()  # get interactions of buying
        df_label_click = df.loc[label_bool_click].copy()  # get interactions of clicking
        label = df_label[['user_log_acct', 'item_sku_id', 'day']].copy()  # process
        label_click = df_label_click[['user_log_acct', 'item_sku_id', 'day']].copy()  # process

        # add new columns
        df_inter['label'] = 0
        df_inter['label_click'] = 0

        df_inter['user_item'] = df_inter['user_log_acct'].astype(str) + '_' + df_inter['item_sku_id'].astype(str)
        label['user_item'] = label['user_log_acct'].astype(str) + '_' + label['item_sku_id'].astype(str)
        label_click['user_item'] = \
            label_click['user_log_acct'].astype(str) + '_' + label_click['item_sku_id'].astype(str)

        dic_cnt = label['user_item'].value_counts().to_dict()
        dic_cnt_click = label_click['user_item'].value_counts().to_dict()

        df_inter['label'] = df_inter['user_item'].apply(lambda x: _label_trans(x, dic_cnt)).values
        df_inter['label_click'] = \
            df_inter['user_item'].apply(lambda x: _label_trans(x, dic_cnt_click)).values

        all_data.append(df_inter)
        fea_list.append(fea_position)

    return all_data, fea_list


def get_feature(df, df_basic_list, feature_columns_user, feature_columns_item):
    """
    Get basic features for use and item
    :param df: Original data
    :param df_basic_list: Data set basic form
    :param feature_columns_user: Selected columns for users
    :param feature_columns_item:  Selected columns for items
    :return: Data set with features
    """
    data_with_feature = []
    for df_basic in df_basic_list:
        jd_user = df[feature_columns_user].drop_duplicates(['user_log_acct'], keep='first')
        jd_item = df[feature_columns_item].drop_duplicates(['item_sku_id'], keep='first')

        u_fea_cols = [col for col in jd_user.columns if col not in ['user_log_acct']]
        i_fea_cols = [col for col in jd_item.columns if col not in ['item_sku_id']]

        df_with_feature = df_basic.merge(jd_user, on='user_log_acct', how='left')  # Merge with basic
        df_with_feature = df_with_feature.merge(jd_item, on='item_sku_id', how='left')

        #        neg_df_train = df_with_feature[df_with_feature.label_click == 0].reset_index(drop=True)
        #        pos_df_train = df_with_feature[df_with_feature.label_click != 0].reset_index(drop=True)
        #        neg_df_train = neg_df_train.sample(n=int(len(pos_df_train) * 5))  # Down Sampling
        #        df_with_feature = pd.concat([neg_df_train, pos_df_train], axis=0, ignore_index=True)

        data_with_feature.append(df_with_feature)
    return data_with_feature


def get_history_convert_type(df, df_withfea_mapped, fea_range_list):
    """
    Get users' behavior sequences
    :param df: Original data
    :param df_withfea_mapped: basic form
    :param fea_range_list: feature ranges for each window
    :return: data set with behavior sequences
    """
    df_final = []
    for i, df_sub in enumerate(df_withfea_mapped):
        ind_fea = (df['month_day'] <= fea_range_list[i])
        data_fea = df.loc[ind_fea].copy()

        df_sub.sort_values(['user_log_acct', 'month_day'], inplace=True)
        data_fea.sort_values(['user_log_acct', 'month_day'], inplace=True)

        item_seq = data_fea.groupby(['user_log_acct'])['item_sku_id'].agg(list).reset_index()
        item_seq.columns = ['user_log_acct', 'item_seq']
        df_sub = df_sub.merge(item_seq, on='user_log_acct', how='left')

        cate_seq = data_fea.groupby(['user_log_acct'])['item_third_cate_cd'].agg(list).reset_index()
        cate_seq.columns = ['user_log_acct', 'cate_seq']
        df_sub = df_sub.merge(cate_seq, on='user_log_acct', how='left')

        type_seq = data_fea.groupby(['user_log_acct'])['action_type'].agg(list).reset_index()
        type_seq.columns = ['user_log_acct', 'type_seq']
        df_sub = df_sub.merge(type_seq, on='user_log_acct', how='left')

        df_sub = df_sub.loc[(df_sub.item_seq.isnull() == False)]  # process
        df_final.append(df_sub)

    return df_final


def map_user_to_id(df_final):
    df_all = pd.concat(df_final, axis=0, ignore_index=True)
    key = sorted(df_all["user_log_acct"].unique().tolist())
    num_users = len(key)
    dict_ = dict(zip(key, range(len(key))))
    for i in range(len(df_final)):
        df_final[i].loc[:, "user_log_acct"] = df_final[i]["user_log_acct"].map(lambda x: dict_[x])

    return num_users


def gen_item_feats(df_item, df_final):
    df_item_fea = df_item.copy()

    for col in ['item_third_cate_cd', 'vender_id']:
        dic_ = df_item[col].value_counts().to_dict()
        df_item_fea['{}_cnt'.format(col)] = df_item_fea[col].map(dic_).values

    for col in ['shop_score']:
        dic_ = df_item.groupby('item_third_cate_cd')[col].mean().to_dict()
        df_item_fea['cate_{}_mean'.format(col)] = df_item_fea['item_third_cate_cd'].map(dic_).values

    for col in ['item_sku_id', 'brand_code']:
        dic_ = df_item.groupby('shop_id')[col].nunique()
        df_item_fea['shop_id_{}_nunique'.format(col)] = df_item_fea['shop_id'].map(dic_).values

    for col in ['item_sku_id', 'brand_code']:
        dic_ = df_item.groupby('item_third_cate_cd')[col].nunique()
        df_item_fea['item_third_cate_cd_{}_nunique'.format(col)] = df_item_fea['item_third_cate_cd'].map(dic_).values

    del df_item_fea['item_third_cate_cd']
    del df_item_fea['shop_id']
    del df_item_fea['brand_code']
    del df_item_fea['vender_id']
    df_with_item_fea = []
    for df in df_final:
        temp = df.merge(df_item_fea, on='item_sku_id', how='left')
        df_with_item_fea.append(temp)

    return df_with_item_fea


df = load_data("./ddata/merged_DataFrame_fillna.pkl")

col = ["item_sku_id", "item_third_cate_cd", "sex",
       "action_type", "city", "user_level", 'province', 'county','brand_code', 'shop_id', 'vender_id']

num_items, num_cats, num_sex, num_ulevel, num_atype, num_city, num_province, \
    num_county, num_brand_code, num_shope,num_vender, cat_list = map_to_id(df, col)

label_start = [(4, 11), (4 ,6),(4, 1), (3,27), (3, 22), (3,17)]
label_end = [(4, 15),(4, 10),(4, 5), (4, 1), (3, 26), (3, 21)]
inter_start = [(4, 6), (4, 1),(3,27), (3, 22), (3, 17), (3,12)]
inter_end = [(4, 10), (4, 5),(3, 31), (3, 26), (3, 21), (3, 16)]
fea_end = [(4, 10), (4, 5),(3, 31), (3, 26), (3, 21), (3, 16)]

all_data, fea_list = sliding_window_2_basic_form(df, label_start, label_end, inter_start, inter_end, fea_end)

user_side_columns = ["user_log_acct", "sex", "city", "user_level", 'province', 'county']
item_side_columns = ["item_sku_id", "item_third_cate_cd",'brand_code', 'shop_id', 'vender_id']
df_with_feature = get_feature(df, all_data, user_side_columns, item_side_columns)

df_final = get_history_convert_type(df, df_with_feature, fea_list)

jd_item = df[['item_sku_id','brand_code','shop_id','item_third_cate_cd','vender_id','shop_score']].\
    drop_duplicates(['item_sku_id'], keep='first')

df_final = gen_item_feats(jd_item, df_final)

store_data(df_final,"./ddata/start_data/df_final.pkl")

num_users = map_user_to_id(df_final)
store_data(num_users, "./ddata/start_data/num_users.pkl")
store_data(num_items, "./ddata/start_data/num_items.pkl")
store_data(num_cats, "./ddata/start_data/num_cats.pkl")
store_data(num_sex, "./ddata/start_data/num_sex.pkl")
store_data(num_ulevel, "./ddata/start_data/num_ulevel.pkl")
store_data(num_atype, "./ddata/start_data/num_atype.pkl")
store_data(num_city, "./ddata/start_data/num_city.pkl")

store_data(num_province, "./ddata/start_data/num_province.pkl")
store_data(num_county, "./ddata/start_data/num_county.pkl")
store_data(num_brand_code, "./ddata/start_data/num_brand_code.pkl")
store_data(num_shope, "./ddata/start_data/num_shope.pkl")
store_data(num_vender, "./ddata/start_data/num_vender.pkl")

store_data(cat_list, "./ddata/start_data/cat_list.pkl")
