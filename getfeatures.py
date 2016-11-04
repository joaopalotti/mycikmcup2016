import numpy as np
import pandas as pd
import datetime
import pickle

from sklearn import preprocessing

def jaccard(a,b):
    return 1.0 * len(a.intersection(b)) / len(a.union(b))

def minmaxscale(X):
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(X)

def check_time(start_time):
    end_time = datetime.datetime.now()
    print "# Now it's ", end_time.isoformat()
    print "# Since the begin: ", (end_time - start_time).seconds, " seconds"

def groundtruth(df):
    outfile = open("groundtruth.txt", "w")
    for row in df[["queryId","itemId","clicked","viewed","purchased"]].itertuples():
        relevance = 0
        #if row[3] or row[4]:
        if row[3]:
            relevance = 1
        elif row[5]:
            relevance = 2

        itemid = int(row[2])
        qid = int(row[1])

        outfile.write("%d %d %d" % (qid, itemid, relevance))
        outfile.write("\n")
    outfile.close()

def create_output(result, train_queries, test_queries):

        tset = set(train_queries["queryId"].values)
        result_train = result.loc[result["queryId"].isin(tset)]
        to_ranklib(result_train, "official_train.txt")

        # Print the test results
        tset = set(test_queries["queryId"].values)
        result_test = result.loc[result["queryId"].isin(tset)]
        to_ranklib(result_test, "official_test.txt")
        result_test[["queryId","itemId"]].to_csv("my_test_ids.txt", index=False)


def to_ranklib(df, filename):

    outfile = open(filename, "w")
    # Using itertuples because itermrows would take much longer
    # Indices:      1          2        3         4        5          || 6         7              8          9
    for row in df[["queryId","itemId","clicked","viewed","purchased","rank","rank_size","itemView","itemClick",\
        # 10            11         12        13        14            15           16
        "itemPurchase","itemMeanRank","itemMaxRank","itemMinRank","itemMedianRank", "pricelog2", "jaccard",\
        #17     18     19        20      21       22            23        24             25                26
        "maxp","minp","medianp","meanp","rangep","howexpensive","catperc","count_clicks", "uniqueUserView","uUserView",
        #27              28          29              30          31                 32           33
        "uUserPurchase", "avg_ndcg", "avg_duration", "mmr_view", "productNameSize", "querySize", "queriesPerSession",
        #34               35
        "ndcgPerSession", "durationPerSession"]].itertuples():

        relevance = 0
        if row[3] or row[4]:
            relevance = 1
        elif row[5]:
            relevance = 2

        qid = row[1]

        outfile.write("%d qid:%d" % (relevance, qid))
        for i, v in enumerate(row[6:], start=1):
            outfile.write(" %d:%.4f" % (i,v))
        outfile.write("\n")
    outfile.close()

start_time = datetime.datetime.now()
print "# Running baseline. Now it's", start_time.isoformat()

##################################################################
#----------------------------------------------------------------#
#---------------------- LOADS DATASETS --------------------------#
#----------------------------------------------------------------#
##################################################################

# Loading queries (assuming data placed in <dataset-train/>
queries = pd.read_csv('dataset-train/train-queries.csv', sep=';')
queries["eventdate"] = pd.to_datetime(queries["eventdate"])

# Loading only the queries that have keyword search.
queries = queries[~queries["searchstring.tokens"].isnull()]

# Extract a mapping of each query and which items appeared
query_item = []
for query, items in queries[["queryId", "items"]].values:
    items = map(np.int64,items.split(','))
    for i in items:
        query_item.append( (query, i) )
query_item = pd.DataFrame().from_records(query_item, columns=["queryId","itemId"])

# Loading item views
item_views = pd.read_csv('dataset-train/train-item-views.csv', sep=';')
item_views.sort_values(["sessionId", "userId", "eventdate", "timeframe", "itemId"], inplace=True)
print('Item views', len(item_views))

# Loading clicks
clicks = pd.read_csv('dataset-train/train-clicks.csv', sep=';')
clicks.sort_values(["queryId", "timeframe", "itemId"], inplace=True)
print('Clicks', len(clicks))

# Loading purchases
purchases = pd.read_csv('dataset-train/train-purchases.csv', sep=';')
print('Purchases', len(purchases))
purchases.sort_values(["sessionId", "userId", "eventdate", "timeframe", "itemId", "ordernumber"], inplace=True)

# Loading products
products = pd.read_csv('dataset-train/products.csv', sep=';')
print('Products', len(products))
products.sort_values(["itemId"], inplace=True)

# Loading product category
products_category = pd.read_csv('dataset-train/product-categories.csv', sep=';')
print('Products Categories', len(products))
products_category.sort_values(["itemId"], inplace=True)

# Add info regarding sessionid
query_item = pd.merge(query_item, queries[["queryId", "sessionId"]], how="left")
query_item = pd.merge(query_item,  clicks, how="left")
query_item.rename(columns={"timeframe":"clickTime"}, inplace=True)
query_item = pd.merge(query_item,  item_views, how="left")

query_item.rename(columns={"eventdate":"eventdateView", "timeframe":"viewTime", "userId": "userView"}, inplace=True)
query_item = pd.merge(query_item, purchases, how="left")
query_item.rename(columns={"eventdate":"eventdatePurchase", "timeframe":"purchaseTime", "userId": "userPurchase"}, inplace=True)

"""
    'rank' is a value between 0 and 1, with 1 if the item is at the top of a list and 0 if this is the last value of a result list.
    Later we will calculate the division of this value by the number of items for each query and finally do 1 minus this value,
    such as the first position item will have rank value of 1, the second will have value of 1 - (1/N), where N is the number of
    items in a given result list.
"""
query_item["rank"] = 1
query_item["rank"] = query_item[["queryId","rank"]].groupby("queryId")["rank"].cumsum()
query_item["rank"] = query_item["rank"] - 1

items_per_query = query_item[["queryId","rank"]].groupby("queryId")["rank"].max()
items_per_query.name = "rank_size"

query_item = pd.merge(query_item, items_per_query.reset_index(), how="left")
query_item["rank"] = 1.0 - (query_item["rank"] / query_item["rank_size"])

# labels:
query_item["clicked"] = ~query_item["clickTime"].isnull()
query_item["viewed"] = ~query_item["viewTime"].isnull()
query_item["purchased"] = ~query_item["purchaseTime"].isnull()

# products info
products_info = pd.merge(query_item[["queryId", "itemId"]].drop_duplicates(), queries[["queryId", "searchstring.tokens"]], on="queryId", how="left").merge(products, on="itemId", how="left")

# NDCGs is a file with
ndcgs = pd.read_csv("ndcg_test.csv")
print('NDCGs', len(ndcgs))

print "Default datasets loaded in memory."
check_time(start_time)

##################################################################
#----------------------------------------------------------------#
#-------------------FEATURE CALCULATION--------------------------#
#----------------------------------------------------------------#
##################################################################

# Separate train and test queries
train_queries = queries[queries['is.test'] == False]
test_queries = queries[queries['is.test'] == True]

train_queries.reset_index(inplace=True, drop=True)
test_queries.reset_index(inplace=True, drop=True)
#

# Calculates mean ndcg and duration for queries based on their keywords
queries = pd.merge(queries, ndcgs, how="left")
qstr_ndcg = queries[["searchstring.tokens", "qndcg"]].groupby("searchstring.tokens")["qndcg"].mean().reset_index().rename(columns={"qndcg": "avg_ndcg"})
queries = pd.merge(queries, qstr_ndcg, how="left")

qstr_duration = queries[["searchstring.tokens", "duration"]].groupby("searchstring.tokens")["duration"].mean().reset_index().rename(columns={"duration":"avg_duration"})
queries = pd.merge(queries, qstr_duration, how="left")

queries["avg_duration"].fillna(queries["avg_duration"].mean()/2., inplace=True)
queries["avg_ndcg"].fillna(queries["avg_ndcg"].mean()/2., inplace=True)
#

# Item Features
print "Creating item features"
check_time(start_time)

# Counts views, clicks and purchases of each item:
itemViewPopularity = item_views["itemId"].value_counts()
itemClickPopularity = clicks["itemId"].value_counts()
itemPurchasePopularity = purchases["itemId"].value_counts()

testset = set(test_queries["queryId"].values)

# Rank is now set to be the rank_size if the query is a test one. --> worsen LB
# query_item["rank"] = query_item["rank"].where(~query_item["queryId"].isin(testset), query_item["rank_size"])
itemMeanRank = query_item[~query_item["queryId"].isin(testset)][["rank","itemId"]].groupby("itemId")["rank"].mean()
itemMeanRank.name = "itemMeanRank"
itemMaxRank = query_item[~query_item["queryId"].isin(testset)][["rank","itemId"]].groupby("itemId")["rank"].max()
itemMaxRank.name = "itemMaxRank"
itemMinRank = query_item[~query_item["queryId"].isin(testset)][["rank","itemId"]].groupby("itemId")["rank"].min()
itemMinRank.name = "itemMinRank"
itemMedianRank = query_item[~query_item["queryId"].isin(testset)][["rank","itemId"]].groupby("itemId")["rank"].median()
itemMedianRank.name = "itemMedianRank"

itemFeatures = pd.merge(itemMedianRank.reset_index(), itemMaxRank.reset_index()).merge(itemMeanRank.reset_index()).merge(itemMinRank.reset_index())

# Users that view and purchased an item
userView = query_item[["itemId","userView"]].groupby("itemId")["userView"].unique().apply(lambda x:len(x))
userView.name = "uUserView"
userPurchase = query_item[["itemId","userPurchase"]].groupby("itemId")["userPurchase"].unique().apply(lambda x:len(x))
userPurchase.name = "uUserPurchase"

query_item = pd.merge(query_item, userView.reset_index(), how="left")
query_item = pd.merge(query_item, userPurchase.reset_index(), how="left")

## add info regarding query string to group values per querystr
queries_str = queries[["queryId", "searchstring.tokens"]]
query_item = pd.merge(query_item, queries_str, how="left")


#####
# Failures:
#####

# Added average time of clicks per query str: adding this features the LB score descreased from 0.4824 to 0.4709
#avgClickTime = query_item[["searchstring.tokens", "clickTime"]].groupby("searchstring.tokens")["clickTime"].mean().reset_index().rename(columns={"clickTime":"avgClickTime"})
#avgViewTime = query_item[["searchstring.tokens", "viewTime"]].groupby("searchstring.tokens")["viewTime"].mean().reset_index().rename(columns={"viewTime":"avgViewTime"})
#avgPurchaseTime = query_item[["searchstring.tokens", "purchaseTime"]].groupby("searchstring.tokens")["purchaseTime"].mean().reset_index().rename(columns={"purchaseTime":"avgPurchaseTime"})

# From 0.4825 to 0.4813
#avgClickTime = query_item[["searchstring.tokens", "itemId", "clickTime"]].groupby(["searchstring.tokens","itemId"])["clickTime"].mean().reset_index().rename(columns={"clickTime":"avgClickTimeItem"})
#avgClickTime["avgClickTimeItem"].fillna(avgClickTime["avgClickTimeItem"].mean(), inplace=True)
#query_item = pd.merge(query_item, avgClickTime)

#avgClickTime["avgClickTime"].fillna(0.0, inplace=True) # Resulted in a small loss: from 0.4824 to 0.4795
#avgViewTime["avgViewTime"].fillna(0.0, inplace=True)
#avgPurchaseTime["avgPurchaseTime"].fillna(0.0, inplace=True)
#query_item = pd.merge(query_item, avgClickTime).merge(avgViewTime).merge(avgPurchaseTime)

# Trying now to use a bin(0-1) click, view, purchase
#hasPastClick = query_item[["searchstring.tokens", "clickTime"]].groupby("searchstring.tokens")["clickTime"].mean().reset_index().rename(columns={"clickTime":"hasPastClick"})
#hasPastClick["hasPastClick"] = ~hasPastClick["hasPastClick"].isnull()

#hasPastView = query_item[["searchstring.tokens", "viewTime"]].groupby("searchstring.tokens")["viewTime"].mean().reset_index().rename(columns={"viewTime":"hasPastView"})
#hasPastView["hasPastView"] = ~hasPastView["hasPastView"].isnull()

#hasPastPurchase = query_item[["searchstring.tokens", "purchaseTime"]].groupby("searchstring.tokens")["purchaseTime"].mean().reset_index().rename(columns={"purchaseTime":"hasPastPurchase"})
#hasPastPurchase["hasPastPurchase"] = ~hasPastPurchase["hasPastPurchase"].isnull()
#query_item = pd.merge(query_item, hasPastClick).merge(hasPastView).merge(hasPastPurchase)

#qstr_item_clicked = query_item[["searchstring.tokens", "itemId", "clicked"]].groupby(["searchstring.tokens", "itemId"]).sum().reset_index().rename(columns={"clicked":"iclicked"})
#qstr_item_viewed = query_item[["searchstring.tokens", "itemId", "viewed"]].groupby(["searchstring.tokens", "itemId"]).sum().reset_index().rename(columns={"viewed":"iviewed"})
#qstr_item_purchased = query_item[["searchstring.tokens", "itemId", "purchased"]].groupby(["searchstring.tokens", "itemId"]).sum().reset_index().rename(columns={"purchased":"ipurchased"})
#query_item = pd.merge(query_item, qstr_item_clicked) # .merge(qstr_item_viewed).merge(qstr_item_purchased)

# Exploring different ways to average the history of clicks
#qstr_clicks_per_query = query_item[["queryId","searchstring.tokens","clicked"]].groupby(["searchstring.tokens","queryId"])["clicked"].sum().reset_index().rename(columns={"clicked":"cperq"})
#query_item = pd.merge(query_item, qstr_clicks_per_query)

print "Merging Item datasets"
check_time(start_time)
#

itemViewPopularity.name = "itemView"
itemClickPopularity.name = "itemClick"
itemPurchasePopularity.name = "itemPurchase"

itemFeatures = pd.merge(itemFeatures, itemViewPopularity.reset_index(), left_on="itemId", right_on="index", how="right")
itemFeatures = pd.merge(itemFeatures, itemClickPopularity.reset_index())
itemFeatures = pd.merge(itemFeatures, itemPurchasePopularity.reset_index())

del itemFeatures["index"]

# Unique users view, click, purchase:
uniqueUserView = item_views[["itemId","userId"]].groupby("itemId")["userId"].unique().apply(lambda x: len(x))
uniqueUserView.name = "uniqueUserView"
itemFeatures = pd.merge(itemFeatures, uniqueUserView.reset_index())

print "Collecting info about products"
check_time(start_time)

# Check the amount of intersection between the search string and the product name
products_info["searchstring.tokens"].fillna("", inplace=True)
products_info["jaccard"] = products_info[["searchstring.tokens","product.name.tokens"]].apply(lambda x: jaccard(set(x[0].split(",")), set(x[1].split(","))), axis=1)

# Count number of keywords to describe product
productNameSize = products_info["product.name.tokens"].apply(lambda x : len(x.split(",")))
productNameSize.name = "productNameSize"  # not using, as it decreased the LB score
products_info = pd.concat((products_info, productNameSize), axis=1)

# Add basic information regaring price
max_price = products_info[["queryId","pricelog2"]].groupby("queryId")["pricelog2"].max()
min_price = products_info[["queryId","pricelog2"]].groupby("queryId")["pricelog2"].min()
median_price = products_info[["queryId","pricelog2"]].groupby("queryId")["pricelog2"].median()
mean_price = products_info[["queryId","pricelog2"]].groupby("queryId")["pricelog2"].mean()
range_price = max_price - min_price
prices = pd.concat((max_price, min_price, mean_price, median_price, range_price), axis=1)
prices.columns = [["maxp", "minp", "meanp", "medianp", "rangep"]]
products_info = pd.merge(prices.reset_index(), products_info, on="queryId")

products_info["howexpensive"] = 1.0 - (products_info["pricelog2"] / products_info["maxp"])
# TODO: check if price changes over time.

# Adding information regarding distribution of categories per query:
query_category = pd.merge(query_item[["queryId","itemId"]], products_category)
query_category.sort_values(["queryId","categoryId"], inplace=True)

category_counts = query_category.groupby(["queryId","categoryId"]).agg('count').reset_index().rename(columns = {"itemId":"counts"})
category_counts["catperc"] = category_counts.groupby("queryId")["counts"].apply(lambda x: x / float(x.sum()))
query_category = pd.merge(query_category, category_counts[["queryId","categoryId","catperc"]])

# calculate mmr for items
mmr_view = query_item[["rank","itemId"]].where(query_item["viewed"], np.nan, axis=0)
mmr_view = mmr_view.groupby("itemId")["rank"].mean().reset_index().rename(columns={"rank":"mmr_view"})

# TODO: feaetures related to sessionids
queriesPerSession = queries[["sessionId","queryId"]].groupby("sessionId")["queryId"].count()
queriesPerSession.name = "queriesPerSession"
queries = pd.merge(queries, queriesPerSession.reset_index())

ndcgPerSession = queries[["sessionId","qndcg"]].groupby("sessionId")["qndcg"].mean()
ndcgPerSession.name = "ndcgPerSession"
ndcgPerSession.fillna(ndcgPerSession.mean()/2., inplace=True)
queries = pd.merge(queries, ndcgPerSession.reset_index())

durationPerSession = queries[["sessionId","duration"]].groupby("sessionId")["duration"].mean()
durationPerSession.name = "durationPerSession"
queries = pd.merge(queries, durationPerSession.reset_index())

# QuerySize
querySize = queries["searchstring.tokens"].apply(lambda x: 0 if type(x) == np.float else len(x.split(",")))
# .apply(lambda x:len(x.split(",")))
querySize.name = "querySize"
queries = pd.concat((querySize, queries), axis=1)


##################################################################
#----------------------------------------------------------------#
#-------------------SCALING AND MERGING FEATURES-----------------#
#----------------------------------------------------------------#
##################################################################

print "Scaling datasets"
check_time(start_time)

# scale datasets
itemFeatures[["itemView","itemClick","itemPurchase","uniqueUserView"]] = minmaxscale(itemFeatures[["itemView","itemClick","itemPurchase","uniqueUserView"]])
query_item[["rank_size","uUserView","uUserPurchase"]] = minmaxscale(query_item[["rank_size","uUserView","uUserPurchase"]])
products_info[["pricelog2","maxp","minp","meanp","medianp","rangep","productNameSize"]] = minmaxscale(products_info[["pricelog2","maxp","minp","meanp","medianp","rangep","productNameSize"]])

print "Creating supermerge dataset."
check_time(start_time)

# Merge intermediary results. One of the slowest operations. This will take a loooong time. Go home, comeback tomorrow.
result = pd.merge(query_item[["queryId","itemId","clicked","viewed","purchased","rank", "rank_size","uUserView","uUserPurchase"]], itemFeatures, how="left")

result = pd.merge(result, products_info[["queryId","itemId","pricelog2","jaccard","maxp","minp","meanp","medianp","rangep","howexpensive","productNameSize"]], on=["queryId", "itemId"], how="left")

result = pd.merge(result, query_category[["queryId","itemId","catperc"]], how="left")

# Add information regarding the number of clicked items per searchstring
### Only for queryfull queries
terms_item_events = pd.merge(queries[["queryId","searchstring.tokens"]], query_item[["queryId","itemId","clickTime","viewTime","purchaseTime"]], on="queryId")
#terms_item_click = pd.merge(queries[["queryId","searchstring.tokens"]], query_item[["queryId","itemId","clickTime"]], on="queryId")
tie = terms_item_events.groupby(["searchstring.tokens","itemId"])[["clickTime","viewTime","purchaseTime"]].count()
tie.columns = ["count_clicks","count_views","count_purchases"]

event_counts = pd.merge( tie.reset_index(), terms_item_events[["itemId","queryId","searchstring.tokens"]])
event_counts[["count_clicks", "count_views", "count_purchases"]] = minmaxscale(event_counts[["count_clicks","count_views","count_purchases"]])

#
result = pd.merge( result, event_counts[["itemId","queryId","count_clicks","count_views","count_purchases"]])
#

queries[["querySize", "avg_duration", "queriesPerSession", "ndcgPerSession", "durationPerSession"]] = minmaxscale(queries[["querySize", "avg_duration", "queriesPerSession", "ndcgPerSession", "durationPerSession"]])
result = pd.merge(result, queries[["queryId", "avg_ndcg", "avg_duration", "querySize", "queriesPerSession", "ndcgPerSession", "durationPerSession"]])

# Info regaring how common/popular a query is:
# queries[["qstr_pop"]] = minmaxscale(queries[["qstr_pop"]])
# result = pd.merge( result, queries[["queryId","qstr_pop"]])
result = pd.merge(result, mmr_view, how="left")

result["mmr_view"] = result["mmr_view"].fillna(result["mmr_view"].mean()/.2)
result["itemMedianRank"] = result["itemMedianRank"].fillna(result["itemMedianRank"].mean()/2.)
result["itemMaxRank"] = result["itemMaxRank"].fillna(result["itemMaxRank"].mean()/2.)
result["itemMinRank"] = result["itemMinRank"].fillna(result["itemMinRank"].mean()/2.)
result["itemMeanRank"] = result["itemMeanRank"].fillna(result["itemMeanRank"].mean()/2.)

print "Done. Filling na's and sorting to generate final output."
check_time(start_time)
# Print results to ranklib
result.fillna(0.0, inplace=True)
result.sort_values(["queryId","itemId"], inplace=True)

print "Generating output..."
check_time(start_time)
#create_output(result, train_queries, test_queries)
create_output(result, train_queries, test_queries)

