import pandas as pd

llm_mails = pd.read_csv("llm_gen_emails.csv")
llm_mails.label.value_counts()
llm_mails = llm_mails[['label','email']]

d1 = pd.read_csv("s1.csv")
print(d1.label.value_counts())
sp = d1[d1.label=='spam']
hm = d1[d1['label'] == 'ham'].sample(n=1000, random_state=42)
d1 = pd.concat([sp, hm]).reset_index(drop=True)
d1 = d1.sample(frac = 1, random_state = 42).reset_index(drop = True)
d1.label = d1.label.replace({'spam':1, 'ham':0})

d2 = pd.read_csv('s2.csv')
sp = d2[d2.label== 1].sample(n = 500, random_state=42)
hm = d2[d2.label == 0].sample(n = 250, random_state=42)
d2 = pd.concat([sp, hm]).reset_index(drop=True)
d2 = d2.sample(frac = 1, random_state = 42).reset_index(drop = True)

d3 = pd.read_csv('s3.csv')
d3 = d3[['label','email']]
d3.label = d3.label.replace({'spam':1, 'ham':0, 'not spam':0})

d4 = pd.read_csv('s4.csv')
d4 = d4.iloc[:,2:-1]
d4 = d4.rename(columns={'Message': 'email', 'Spam/Ham': 'label'})
d4 = d4[['label', 'email']]
sp = d4[d4.label== 'spam'].sample(n = 1000, random_state=42)
hm = d4[d4.label == 'ham'].sample(n = 1000, random_state=42)
d4 = pd.concat([sp, hm]).reset_index(drop=True)
d4 = d4.sample(frac = 1, random_state = 42).reset_index(drop = True)
d4.label = d4.label.replace({'spam':1, 'ham':0})

final_data = pd.concat([llm_mails, d1, d2, d3, d4]).reset_index(drop=True)
print("\n",final_data.label.value_counts())

final_data = final_data.sample(frac = 1, random_state = 42).reset_index(drop = True)
print("\n", final_data.head(20))

final_data.to_csv('final_data.csv', index=False)