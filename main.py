from tkinter.dialog import DIALOG_ICON
# from flask import Flask
from flask import jsonify
from datetime import date

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from flask import Flask, redirect, url_for, render_template, request, flash


from sklearn.neighbors._classification import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import firestore


app = Flask(__name__)

# class Data(object):
#     def __init__(self, diagnosa, nama, tanggal):
#         self.diagnosa = diagnosa
#         self.nama = nama
#         self.tanggal = tanggal


def connectDB():
    cred = credentials.Certificate("eova-11011-firebase-adminsdk-f1qdv-0374a1b140.json")
    firebase_admin.initialize_app(cred)

    db = firestore.client()
    
    # firebase_admin.initialize_app(cred, {
    #     "databaseURL": "https://eova-11011.australia-southeast1.firebasedatabase.app/riwayat" 
    #     })
    # dbconn = db.reference("eova-11011")
    return db

def knn(input_user):
    data = pd.read_csv('dataset_kanker_60_fix.csv')
    scaling = StandardScaler()
    for col in ["PERUT TERASA MEMBESAR", "PERUT KEMBUNG", "NYERI PERUT", "MUAL/MUNTAH", "NAFSU MAKAN MENURUN", "CEPAT KENYANG", "GANGGUAN BAK", "GANGGUAN BAB", "GANGGUAN MENSTRUASI", "PENURUNAN BB"]:
        data[col] = scaling.fit_transform(data[col].values.reshape(-1,1))
        data.head()

    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=0)

    k_range = range(1,len(X_test)+1)
    scores = {}
    scores_list = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        y_pred = knn.predict(X_test)
        scores[k] = accuracy_score(y_test, y_pred)
        scores_list.append(accuracy_score(y_test,y_pred))

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train,y_train)
    print(X_test)
    y_prediksi=knn.predict(input_user)
    
    # clf = SVC(kernel="linear")
    # clf.fit(X_train,y_train)

    # y_prediksi=clf.predict(input_user)

    return y_prediksi

def split(word):
	return [char for char in word]    # untuk split string ke char --> 10101 --> 1,0,1,0,1


@app.route('/prediksi/<input1>', methods=['GET', 'POST'])
def welcome(input1):
    print (input1)

    # os.exec("python prediksi.py" input1 input2 input3 input4 input5 input6 input7 input8 input9 input10)
    # os.system('python prediksi.py "%s"' % input1)

    input_user=split(input1)
    print(input_user)
    df = pd.DataFrame(input_user)
    print (df)
    scaling = StandardScaler()

    for col in df.columns: 
        df[col] = scaling.fit_transform(df[col].values.reshape(-1,1))
    df_transpose = df.transpose()

    print(df_transpose)
    hasil = knn(df_transpose)

    print(hasil)

    diagnosa = ''
    detail = ''

    if hasil == 2:
        diagnosa = 'TIDAK BERESIKO KANKER'
        detail = 'Tetap jaga organ reproduksi Anda.'

    elif hasil == 3:
        diagnosa = 'BERESIKO KANKER'
        detail = 'Anda beresiko terkena kanker ovarium.'

    today = date.today()
    Dict = dict({"diagnosa": diagnosa, "nama": "Zuhairi", "tanggal": str(today)})

    doc_ref = dbconn.collection(u'eova-11011').add(Dict)


    # data = {"info":[{"status" : diagnosa}]
    # }

    # return jsonify(data)

    return render_template('index.html', diagnosa=diagnosa, detail=detail)

@app.route('/riwayat', methods=['GET', 'POST'])
def riwayat():
    doc_ref = dbconn.collection(u'eova-11011').stream()
    # doc = doc_ref.get()
    my_dict = []
    for doc in doc_ref:
        my_dict.append(doc.to_dict())
        # print(f'{doc.id} => {doc.to_dict()}')
    print (my_dict)

    return render_template('table.html', data=my_dict)

@app.route('/testprediksi/<input1>', methods=['GET', 'POST'])
def testing(input1):
    print (input1)

    # os.exec("python prediksi.py" input1 input2 input3 input4 input5 input6 input7 input8 input9 input10)
    # os.system('python prediksi.py "%s"' % input1)

    input_user=split(input1)
    print(input_user)
    df = pd.DataFrame(input_user)
    print (df)
    scaling = StandardScaler()

    for col in df.columns: 
        df[col] = scaling.fit_transform(df[col].values.reshape(-1,1))
    df_transpose = df.transpose()

    print(df_transpose)
    hasil = knn(df_transpose)

    print(hasil)

    diagnosa = ''
    detail = ''

    if hasil == 2:
        diagnosa = 'TIDAK BERESIKO KANKER'
        detail = 'Tetap jaga organ reproduksi Anda.'

    elif hasil == 3:
        diagnosa = 'BERESIKO KANKER'
        detail = 'Anda beresiko terkena kanker ovarium.\nLakukan pemeriksaan kesehatan reproduksi ke rumah sakit terdekat.'

    data = {"info":[{"status" : diagnosa}]
    }

    return jsonify(data)
    
if __name__ == "__main__":
    dbconn = connectDB()
    app.run(host="134.209.109.247")