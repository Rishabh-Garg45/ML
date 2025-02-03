import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("final_data.csv")
data.fillna('', inplace = True)

data['mail_len'] = data['email'].apply(len)
data['num_words'] = data['email'].apply(lambda x: len(x.split()))
data['num_sent'] = data['email'].apply(lambda x: len(x.split('.')))

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
plt.subplots_adjust(hspace=0.4)

avg_len = data.groupby('label')['mail_len'].mean().reset_index()
print("\nAverage Length of Mails:")
print(avg_len)
sns.barplot(x='label', y='mail_len', data=avg_len, palette='Set2', ax=axs[0, 0])
axs[0, 0].set_title('Average Length of Emails for Spam and Ham')
axs[0, 0].set_xlabel('Email Type (0: Ham, 1: Spam)')
axs[0, 0].set_ylabel('Average Length')

avg_words = data.groupby('label')['num_words'].mean().reset_index()
print("\nAverage Number of Words in Emails:")
print(avg_words)
sns.barplot(x='label', y='num_words', data=avg_words, palette='Set2', ax=axs[0, 1])
axs[0, 1].set_title('Average Number of Words in Emails for Spam and Ham')
axs[0, 1].set_xlabel('Email Type (0: Ham, 1: Spam)')
axs[0, 1].set_ylabel('Average Number of Words')

avg_sent = data.groupby('label')['num_sent'].mean().reset_index()
print("\nAverage Number of Sentences in Emails:")
print(avg_sent)
sns.barplot(x='label', y='num_sent', data=avg_sent, palette='Set2', ax=axs[1, 0])
axs[1, 0].set_title('Average Number of Sentences in Emails for Spam and Ham')
axs[1, 0].set_xlabel('Email Type (0: Ham, 1: Spam)')
axs[1, 0].set_ylabel('Average Number of Sentences')

correlation_matrix = data[['mail_len', 'num_words', 'num_sent']].corr()
print("\nThe Relationship between Features:")
print(correlation_matrix)

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axs[1, 1])
axs[1, 1].set_title('Correlation Matrix of Features')
axs[1, 1].set_xlabel('Features')
axs[1, 1].set_ylabel('Features')

plt.show()