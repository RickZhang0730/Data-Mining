import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 讀取顧客背景資料
customer_data = pd.read_csv('P1_Foodmart\Customer-Lookup.csv')
# 讀取交易資料
transaction_data = pd.read_csv('P1_Foodmart\FoodMart-Transactions-1998.csv')
# 讀取產品資料
product_data = pd.read_csv('P1_Foodmart\Product-Lookup.csv')

# 合併顧客背景資料與交易資料，再合併商品資料
merged_data = pd.merge(customer_data, transaction_data, on='customer_id', how='inner')
merged_data = pd.merge(merged_data, product_data, on='product_id', how='inner')


# 選擇要探勘的顧客背景資料欄位和交易商品資料欄位與產品資料欄位
selected_columns = ['gender','education','product_brand']  #選取有興趣的欄位
itemset = merged_data[selected_columns]

transactions = []
for _, row in itemset.iterrows():
    transaction = [str(row[column]) for column in selected_columns]
    transactions.append(transaction)

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 設定最小支持度和最小置信度
min_support = 0.000025  # 自行設定
min_confidence = 0.5  # 自行設定

frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# 對關聯度進行排序输出
top_10_association_rules = rules.sort_values(by="confidence", ascending=False).head(10)
print("Top 10 Association Rules:")
print(top_10_association_rules)

# 將 print 的內容轉換成 DataFrame
top_10_association_df = pd.DataFrame(top_10_association_rules)

# 將 DataFrame 寫入 csv 檔案
top_10_association_df.to_csv('C:/Users/rick/Desktop/DM資料採掘/pw1/top_10_association_rules_case3_2.csv', index=False)

print("已將 Top 10 Association Rules 寫入 CSV 檔案:", 'C:/Users/rick/Desktop/DM資料採掘/pw1/top_10_association_rules_case3_2.csv')
