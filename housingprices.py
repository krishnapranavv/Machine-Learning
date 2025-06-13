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



