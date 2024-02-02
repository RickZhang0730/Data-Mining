import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from IPython.display import display

# 讀取Customer-Lookup資料
customer_data = pd.read_csv('P1_Foodmart\Customer-Lookup.csv')

# 選題目要的顧客 attribute
selected_columns = ['customer_state_province', 'yearly_income', 'gender', 'total_children', 'num_children_at_home', 'education', 'occupation', 'homeowner']
itemset = customer_data[selected_columns]

transactions = []
for _, row in itemset.iterrows():
    transaction = [str(row[column]) for column in selected_columns]
    transactions.append(transaction)

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = fpgrowth(df, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.9)

# 對關聯度進行排序输出
top_10_association_rules = rules.sort_values(by="confidence", ascending=False).head(10)
print("Top 10 Association Rules:")
print(top_10_association_rules)

# 將 print 的內容轉換成 DataFrame
top_10_association_df = pd.DataFrame(top_10_association_rules)

# 將 DataFrame 寫入 csv 檔案
top_10_association_df.to_csv('C:/Users/rick/Desktop/DM資料採掘/pw1/top_10_association_rules_case2.csv', index=False)

print("已將 Top 10 Association Rules 寫入 CSV 檔案:", 'C:/Users/rick/Desktop/DM資料採掘/pw1/top_10_association_rules_case2.csv')