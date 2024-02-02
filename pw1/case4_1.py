import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from datetime import datetime

# 在美國由於聖誕節，12月是購物的旺季。請探勘分析比較 12 月與 1 ~ 11月的顧客購物行為。 有哪些相似的地方，有哪些差異的地方?
# 讀取Customer-Lookup資料、Foodmart交易資料
customer_data = pd.read_csv('P1_Foodmart\Customer-Lookup.csv')
transactions_data = pd.read_csv('P1_Foodmart\FoodMart-Transactions-1998.csv')

# 讀取產品資料
product_data = pd.read_csv('P1_Foodmart\Product-Lookup.csv')
columns_to_keep = ['product_id', 'product_brand', 'product_name', 'product_retail_price']
product_data = product_data[columns_to_keep]

# 合併相同日期、客戶和商店的記錄以形成唯一的交易識別
transactions_data = transactions_data.groupby(['transaction_date', 'customer_id', 'store_id'])['product_id'].apply(list).reset_index()

# 合併顧客/交易資料
merged_data = transactions_data.merge(customer_data, on='customer_id')

# 選取我想保留的欄位
# columns_to_keep = ['customer_country', 'birthdate', 'marital_status', 'yearly_income', 'gender', 'total_children', 'education', 'member_card', 'occupation', 'product_id']
columns_to_keep = ['product_id', 'transaction_date']
merged_data = merged_data[columns_to_keep]

# 將product_id換成product_name
product_lookup = pd.read_csv('P1_Foodmart\Product-Lookup.csv')
product_info = product_lookup[['product_id', 'product_brand']]
id_to_name_mapping = dict(zip(product_info['product_id'], product_info['product_brand']))
merged_data['product_brand'] = merged_data['product_id'].apply(lambda x: [id_to_name_mapping[i] for i in x])
merged_data = merged_data.drop(columns=['product_id'])

# 將transaction_date轉換成日期時間格式
merged_data['transaction_date'] = pd.to_datetime(merged_data['transaction_date'])

# 分成12月跟 1~11月
december_data = merged_data[merged_data['transaction_date'].dt.month == 12]
jan_to_nov_data = merged_data[merged_data['transaction_date'].dt.month.isin(range(1, 12))]


# 轉換成可接受的格式
dec_data = []
for _, row in december_data.iterrows():
    transaction = [str(row[column]) for column in december_data]
    dec_data.append(transaction)

nov_data = []
for _, row in jan_to_nov_data.iterrows():
    transaction = [str(row[column]) for column in jan_to_nov_data]
    nov_data.append(transaction)

dec_te = TransactionEncoder()
dec_te_ary = dec_te.fit(dec_data).transform(dec_data)
dec_df = pd.DataFrame(dec_te_ary, columns=dec_te.columns_)

nov_te = TransactionEncoder()
nov_te_ary = nov_te.fit(nov_data).transform(nov_data)
nov_df = pd.DataFrame(nov_te_ary, columns=nov_te.columns_)

# 12月關聯
frequent_itemsets_december = fpgrowth(dec_df, min_support=0.0005, use_colnames=True)
rules_december = association_rules(frequent_itemsets_december, metric="confidence", min_threshold=0.7)

# 1~12月關聯
frequent_itemsets_nov = fpgrowth(nov_df, min_support=0.0005, use_colnames=True)
rules_nov = association_rules(frequent_itemsets_nov, metric="confidence", min_threshold=0.7)

print("December Frequent Rules:")
print(rules_december)

print("January to November Frequent Rules:")
print(rules_nov)

# 將 print 的內容轉換成 DataFrame
rules_december_df = pd.DataFrame(rules_december)
rules_nov_df = pd.DataFrame(rules_nov)

# 將 DataFrame 寫入 csv 檔案
rules_december_df.to_csv('C:/Users/rick/Desktop/DM資料採掘/pw1/rules_december_case4_1.csv', index=False)
rules_nov_df.to_csv('C:/Users/rick/Desktop/DM資料採掘/pw1/rules_nov_case4_1.csv', index=False)

print("已將 December Frequent Rules 寫入 CSV 檔案:", 'C:/Users/rick/Desktop/DM資料採掘/pw1/rules_december_case4_1.csv')
print("已將 January to November Frequent Rules 寫入 CSV 檔案:", 'C:/Users/rick/Desktop/DM資料採掘/pw1/rules_nov_case4_1.csv')