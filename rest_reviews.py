import mysql.connector as ms 
import pandas as pd

mydb = ms.connect(host='localhost',user='root',passwd='12345',database='project_db')
print("Conntection esablished !! ")
cursor=mydb.cursor()
cursor.execute("SELECT review,sentiment FROM sentiment_analysis")

data=cursor.fetchall()
columns=[i[0] for i in cursor.description]

df = pd.DataFrame(data,columns=columns)


import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
SIA = SentimentIntensityAnalyzer()

df['compound'] = [SIA.polarity_scores(x)['compound'] for x in df['review']]
df['positive'] = [SIA.polarity_scores(x)['pos'] for x in df['review']]
df['neutral'] = [SIA.polarity_scores(x)['neu'] for x in df['review']]
df['negative'] = [SIA.polarity_scores(x)['neg'] for x in df['review']]

df['sentiment']='neutral'
df.loc[df.compound>0.05,'sentiment']='positive'
df.loc[df.compound<-0.05,'sentiment']='negative'
print(df.head())

update_query = "UPDATE sentiment_analysis SET sentiment = %s WHERE review = %s"
for index,rows in df.iterrows():
    cursor.execute(update_query,(rows['sentiment'],rows['review']))

mydb.commit()
cursor.close()
mydb.close()