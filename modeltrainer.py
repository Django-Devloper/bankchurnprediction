import pandas as pd
from sklearn.preprocessing import OneHotEncoder , LabelEncoder , StandardScaler
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input , Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf 

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
y_test = pd.read_csv('sample_submission.csv')

def delete_column(column_name , data):
    data = data.drop(column_name,axis=1)
    return data

train_data =delete_column(['id','CustomerId','Surname'] , train_data)
test_data =delete_column(['id','CustomerId','Surname'] , test_data)

def pickler_saver(column_name , encoder):
    with open(column_name+'encoder.pkl' , 'wb') as file:
        pickle.dump(encoder ,file)

def encoder(column_name , data , one_hot=False):
    if one_hot:
        column_encoder = OneHotEncoder()
        column_name_encode = column_encoder.fit_transform(data[[column_name]]).toarray()
        column_name_df = pd.DataFrame(column_name_encode ,columns= column_encoder.get_feature_names_out([column_name]))
        data = pd.concat([data.drop(column_name,axis=1),column_name_df],axis=1)
    else:
        column_encoder = LabelEncoder()
        data[column_name] = column_encoder.fit_transform(data[column_name])
    pickler_saver(column_name ,column_encoder)
    return data
x_train = encoder('Gender' , train_data)
x_train = encoder('Geography' , x_train , one_hot=True)
x_test = encoder('Gender' , test_data)
x_test = encoder('Geography' , x_test , one_hot=True)
y_train = x_train['Exited']
x_train = x_train.drop('Exited',axis=1)
y_test = y_test.drop('id',axis=1)

scaller = StandardScaler()
x_train = scaller.fit_transform(x_train)
x_test = scaller.transform(x_test)
pickler_saver('scaller' ,scaller)

model = Sequential([
    Input(shape =[x_train.shape[1],]),
    Dense(16 ,activation='relu'),
    Dense(8 ,activation='relu'),
    Dense(1 , activation='sigmoid')
])
opt = tf.keras.optimizers.Adam(learning_rate = 0.01)
model.compile(optimizer = opt , loss = 'binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor = 'val_loss',patience=5 , restore_best_weights=True)
history = model.fit(x_train ,y_train , validation_data = (x_test, y_test),epochs=100 , callbacks = [early_stop])
model.save('bankchurnprediction.keras')