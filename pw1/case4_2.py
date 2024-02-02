import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from datetime import datetime

#在美國由於聖誕節，12月是購物的旺季。請探勘分析比較 12 月與 1 ~ 11月的顧客購物行為。 有哪些相似的地方，有哪些差異的地方?

#讀取顧客資料
customer_data = pd.read_csv('P1_Foodmart\Customer-Lookup.csv')
#保留顧客資料所需要的欄位
columns_to_keep1 = ['customer_id']
customer_data = customer_data[columns_to_keep1]

#讀取產品資料
product_data = pd.read_csv('P1_Foodmart\Product-Lookup.csv')
#保留產品資料所需要的欄位
columns_to_keep2 = ['product_id', 'product_brand', 'product_name']
product_data = product_data[columns_to_keep2]

#讀取交易資料
transactions_data = pd.read_csv('P1_Foodmart\FoodMart-Transactions-1998.csv', parse_dates=['transaction_date'])
#將日期時間列轉換為日期時間格式，並提取月份
transactions_data['transaction_date'] = pd.to_datetime(transactions_data['transaction_date'])
transactions_data['month'] = transactions_data['transaction_date'].dt.month
#僅保留含有12月的行
december_data = transactions_data[transactions_data['month'] == 12]
#其餘1月到11月的行
others_data = transactions_data[transactions_data['month'].isin(range(1, 12))]
#保留交易資料所需要的欄位
columns_to_keep3 = ['transaction_date', 'product_id','customer_id']
december_data = december_data[columns_to_keep3]
others_data = others_data[columns_to_keep3]

#合併顧客/交易資料
merged_data1 = december_data.merge(customer_data, on='customer_id')
merged_data2 = others_data.merge(customer_data, on='customer_id')
#再合併產品資料
merged_data1 = merged_data1.merge(product_data, on='product_id')
merged_data2 = merged_data2.merge(product_data, on='product_id')

#再來把要去做分析的留下
columns_to_keep4 = ['transaction_date','product_id','customer_id','product_brand', 'product_name']
merged_data1 = merged_data1[columns_to_keep4]
merged_data2 = merged_data2[columns_to_keep4]

#對12月的資料進行處理
#使用TransactionEncoder將資料轉換成適合進行頻繁項集採掘的形式
te_dec = TransactionEncoder()
te_ary_dec = te_dec.fit(merged_data1.groupby('customer_id')['product_id'].apply(list)).transform(merged_data1.groupby('customer_id')['product_id'].apply(list))
df_dec = pd.DataFrame(te_ary_dec, columns=te_dec.columns_)
#使用fpgrowth找出12月的頻繁項集
frequent_itemsets_dec = fpgrowth(df_dec, min_support=0.0000005, use_colnames=True)
#根據頻繁項集生成關聯規則
rules_dec = association_rules(frequent_itemsets_dec, metric="confidence", min_threshold=0.5)

#對其他月份的資料進行關聯規則採掘
#使用TransactionEncoder將資料轉換成適合進行頻繁項集採掘的形式
te_others = TransactionEncoder()
te_ary_others = te_others.fit(merged_data2.groupby('customer_id')['product_id'].apply(list)).transform(merged_data2.groupby('customer_id')['product_id'].apply(list))
df_others = pd.DataFrame(te_ary_others, columns=te_others.columns_)
#使用fpgrowth找出其他月份的頻繁項集
frequent_itemsets_others = fpgrowth(df_others, min_support=0.0005, use_colnames=True)
#根據頻繁項集生成關聯規則
rules_others = association_rules(frequent_itemsets_others, metric="confidence", min_threshold=0.8)

#輸出12月的關聯規則
print("December Association Rules:")
print(rules_dec)

#輸出其它月份的關聯規則
print("Others Association Rules:")
print(rules_others)


# 將 print 的內容轉換成 DataFrame
rules_dec_df = pd.DataFrame(rules_dec)
rules_others_df = pd.DataFrame(rules_others)

# 將 DataFrame 寫入 csv 檔案
rules_dec_df.to_csv('C:/Users/rick/Desktop/DM資料採掘/pw1/rules_dec_case4_2.csv', index=False)
rules_others_df.to_csv('C:/Users/rick/Desktop/DM資料採掘/pw1/rules_others_case4_2.csv', index=False)

print("已將 December Frequent Rules 寫入 CSV 檔案:", 'C:/Users/rick/Desktop/DM資料採掘/pw1/rules_dec_case4_2.csv')
print("已將 January to November Frequent Rules 寫入 CSV 檔案:", 'C:/Users/rick/Desktop/DM資料採掘/pw1/rules_others_case4_2.csv')
