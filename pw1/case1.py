import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 讀取Foodmart交易資料
df_transactions = pd.read_csv('P1_Foodmart/FoodMart-Transactions-1998.csv')

# 相同的 transaction_date, customer_id, store_id 視為一筆 transaction
# 合併相同日期、客戶和商店的記錄以形成唯一的交易識別
df_transactions = df_transactions.groupby(['transaction_date', 'customer_id', 'store_id'])['product_id'].apply(list).reset_index()

product_id_list = df_transactions['product_id'].tolist()

te = TransactionEncoder()
te_ary = te.fit(product_id_list).transform(product_id_list)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = fpgrowth(df, min_support=0.00015, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)

# 對置信度進行排序输出
top_10_confidence_rules = rules.sort_values(by="confidence", ascending=False).head(10)
print("Top 10 Confidence Rules (Sorted by Confidence):")
print(top_10_confidence_rules)

# 對提升度進行排序输出
top_10_lift_rules = rules.sort_values(by="lift", ascending=False).head(10)
print("Top 10 Lift Rules (Sorted by Lift):")
print(top_10_lift_rules)

# 將 print 的內容轉換成 DataFrame
top_10_confidence_df = pd.DataFrame(top_10_confidence_rules)
top_10_lift_df = pd.DataFrame(top_10_lift_rules)

# 將 DataFrame 寫入 csv 檔案
top_10_confidence_df.to_csv('C:/Users/rick/Desktop/DM資料採掘/pw1/top_10_confidence_rules_case1.csv', index=False)
top_10_lift_df.to_csv('C:/Users/rick/Desktop/DM資料採掘/pw1/top_10_lift_rules_case1.csv', index=False)

print("已將 Top 10 Confidence Rules 寫入 CSV 檔案:", 'C:/Users/rick/Desktop/DM資料採掘/pw1/top_10_confidence_rules_case1.csv')
print("已將 Top 10 Lift Rules 寫入 CSV 檔案:", 'C:/Users/rick/Desktop/DM資料採掘/pw1/top_10_lift_rules_case1.csv')
