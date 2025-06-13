import mysql.connector as ms
import pandas as pd

mydb=ms.connect(host='localhost',user='root',passwd='12345',database='Prices')
print("Connection has been successfully established !")
cursor=mydb.cursor()
query="SELECT price,year FROM housing WHERE location = %s "
location_filter='hyderabad'
cursor.execute(query,(location_filter,))

data=cursor.fetchall()
columns=[i[0] for i in cursor.description]

df= pd.DataFrame(data,columns=columns)
mydb.close()

X= df[['year']]
Y= df['price']

from sklearn.model_selection import train_test_split  
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=7)

from sklearn.linear_model import LinearRegression  
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

import matplotlib.pyplot as plt
plt.scatter(X,Y,color='red',label='actual data')
plt.plot(X,model.predict(X),color='blue',label='regression line')
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Year vs Price with Regression Line')
plt.show()






