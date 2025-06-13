import mysql.connector as ms
mydb=ms.connect(host='localhost',user='root',passwd='12345',database='Prices')
print("Connection has been successfully established !")
mycursor=mydb.cursor()
mycursor.execute('select * from housing')
mydata=mycursor.fetchall()
print(mydata)
mydb.close()