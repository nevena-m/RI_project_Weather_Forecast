import pandas as pd
from pandas import DataFrame
import numpy as np
import os
from datetime import datetime
import random
import math

g_data_folder = 'data'
g_y_colname = 'Basel Precipitation Total'
test_size = 0.3

dropout_columns = [
    "timestamp",
    "Basel Vapor Pressure Deficit [2 m]",
    "Basel Vapor Pressure Deficit [2 m].1",
    "Basel Vapor Pressure Deficit [2 m].2",
    "Basel Temperature",
    "Basel Temperature.1",
    "Basel Temperature.2",
    "Basel Wind Direction Dominant [10 m]"]

df = None
min_scale = None
max_scale = None

x_test = None
y_test = None
x_train = None
y_train = None
n_x_test = 0
n_x_train = 0

random.seed(a=None, version=2)

# Ulazne vrednosti i parametri
maxiters = 100000; eta = 0.5; alpha = 0.9
N = 6        # koliko dana koristimo za predikciju
m = None

log_file = open("logger.txt", "w") # fajl koji ce sluziti za log programa
log_batch = []
log_batch_size = 0
log_batch_max = 200

log_output = True

def log(msg: str):
    if log_output:
        global log_file, log_batch, log_batch_max, log_batch_size
        log_batch += ["<{}> - {}\n".format(datetime.now().strftime("%H:%M:%S"), msg)]
        log_batch_size += 1
        if log_batch_max < log_batch_size:
            log_file.writelines(log_batch)
            log_batch_size = 0
            log_batch.clear()

def pd_load_file(file_path: str, num_rows_to_skip: int) -> DataFrame:
    try:
        if not os.path.isfile(file_path):
            log("Proveriti putanju do fajla")
            raise Exception('Proverite putanju do fajla!')
        if file_path.endswith('.csv'):
            log("ucitan fajl: "+ file_path)
            return pd.read_csv(file_path, low_memory=False, encoding='utf-8', skiprows=num_rows_to_skip)
        
    except pd.errors.ParserError:
        log("Nema dovoljno memorije")
        raise Exception('Doslo je do greske. Nema dovoljno memorije!')


def load_files(num_rows_to_skip: int):
    global df
    try:
        df_list = []
        shape = 0
        i = 0
        log("Ucitavanje fajlova...")
        for file_name in os.listdir(g_data_folder):
            df_list += [pd_load_file(os.path.join(g_data_folder, file_name), num_rows_to_skip)]
            shape += df_list[-1].shape[0]
            df_list[i]
            i += 1
        df = pd.concat(df_list, ignore_index=True)
        df.drop(columns=dropout_columns, inplace=True) # ne treba nam posto je sve poredjano
        log('Ukupno redova: '+ str(df.shape[0]))
        
    except:
        log("doslo je do greske prilikom ucitavanja fajlova")
        raise Exception('Doslo je do greske')


# Izdvajamo narednih N dana i lepimo njihove atribute zaredom. Od funkcije koja poziva ovu 
# se ocekuje da sama brine da li je indeks < (sizeSkupa - N - 1). 
# isTrainSet je flag koji odredjuje da li uzmimamo iz skupa za trening ili za testiranje
def nextNDays(dayIndex: int, prevOutput, isTrainSet: bool = True) -> pd.Series:
    if dayIndex > ((n_x_train if isTrainSet else n_x_test) - N - 1):
        raise Exception("Indeks je veci od sizeSkupa-N-1. Proveri unos")
    if isTrainSet:
        # bajes, output prosle iteracije, sledeci ulazni svorovi
        return [1.0, prevOutput] + list(pd.concat([x_train.iloc[i,:] for i in range(dayIndex,dayIndex+N)], ignore_index=True))
    else:
        return [1.0, prevOutput] + list(pd.concat([x_test.iloc[i,:] for i in range(dayIndex,dayIndex+N)], ignore_index=True))

def activation_f(u):
    return 1.0 / (1.0 + math.exp(-u))

# Ucenje
def train():
    global maxiters, N, m
    # broj dana koje koristimo za predikciju * broj atributa u danu + bajes + output
    n = len(nextNDays(0, 0)) - 1
    m = (int)(n*0.75)
    # broj output cvorova    
    p = 1                    # neka ostane 1 
    log("broj dana za predikciju= {}; broj ulaznih cvorova= {}; broj cvorova skriveni sloj= {}".format(N, n+1, m))
    
    first_iter = True
    
    h = [0.0 for i in range(m+1)]
    dh = [0.0 for i in range(m+1)]
    # Inicijalizacija W'(ij)    
    w_ = [[random.random()  - 0.5 for i in range(m+1)] for j in range(n+1)]  
    dw_ = [[0.0 for i in range(m+1)] for j in range(n+1)] 
    # Inicijalizacija W''(jk)
    w__ = [[random.random()  - 0.5 for i in range(p+1)] for j in range(m+1)] 
    dw__ = [[0.0 for i in range(p+1)] for j in range(m+1)]
    o = [0.0 for i in range(n_x_train+1)]

    day = 0 # zero based
    while maxiters > 0: 
        maxiters -= 1
        # U svakoj iteraciji se biraju uzastopni dani
        if day == (n_x_train-N-1):
            day = 0
        
        if not first_iter: # nije prva iteracija
            x = nextNDays(day, o[day-1]) # lista ulaznih cvorova
            first_iter = False
        else:
            x = nextNDays(day, 0) # stavljamo nulu umesto prethodnog izlaza
        day += 1 
        
        # Izracunavanje h1,..., hm cvorova skrivenog sloja
        h[0] = 1.0
        for j in range(1,m+1):
            u = 0
            for i in range(0, n+1):
                u += x[i] * w_[i][j]
            h[j] = activation_f(u)
        
        # Izracunavanje o1,...,op
        for k in range(1, p+1):    
            u = 0
            for j in range(0,m+1):
                u += h[j] * w__[j][k]
            o[day] = activation_f(u) 

        # Izracunavanje deltaH(j), ne uzimamo tezinu bajesa
        for j in range(1,m+1):   
            dh[j] = 0.0
            for k in range(1,p+1):
                dh[j] += w__[j][k] * (y_train[day] - o[day]) * o[day] * (1.0 - o[day])
    
        #Azuriranje W'(ij) i deltaW'(ij)
        for j in range(1,m+1):
            for i in range(0,n+1): 
                dw_[i][j] = eta * x[i] * h[j] * (1 - h[j]) * dh[j] + alpha * dw_[i][j] 
                w_[i][j] += dw_[i][j]

        #Azuriranje W''(jk) i deltaW''(jk)
        for k in range(1,p):   
            for j in range(0,m):
                dw__[j][k] = eta * h[j] * (y_train[day] - o[day]) * o[day] * (1.0 - o[day]) + alpha * dw__[j][k]        
                w__[j][k] += dw__[j][k]

        log("maxiter={} => day_index={}; o = {}; y = {}".format(maxiters, day, o[day], y_train[day]))
    with open("model.txt", "w") as model_output:
        model_output.write("broj dana za predikciju= {}; broj ulaznih cvorova= {}; broj cvorova skriveni sloj= {}".format(N, n+1, m))
        model_output.write("w'")
        model_output.write(str(w_))
        model_output.write("w''")
        model_output.write(str(w__))
        model_output.write("h")
        model_output.write(str(h))

def test():
    print("TODO")
    # TODO: ucitavanje sacuvanog modela ako ga ima, ili ponovno treniranje i prolazak kroz skup za test

def test_train_split():
    log("Podela na test i trening skup")
    global df, x_train, y_train, x_test, y_test, test_size, g_y_colname, n_x_test, n_x_train
    n = df.shape[0]
    
    n_x_test = (int)(n * test_size)
    n_x_train = n - n_x_test

    # ovo je shallow
    y_train = df[g_y_colname][:n_x_train]
    y_test = df[g_y_colname][n_x_train:]
    x_train = df.drop(columns=[g_y_colname])[:n_x_train]
    x_test = df.drop(columns=[g_y_colname])[n_x_train:]
    log("train={}; test={}".format( x_train.shape, x_test.shape))
    
def scale_func(x):
    return (x-min_scale)/(max_scale-min_scale)

def scale_attributes():
    global df, min_scale, max_scale
    
    log("Skaliranje atributa")
    min_scale = min(df.min(axis=0))
    max_scale = max(df.max(axis=0))
    df = df.apply(scale_func)

if __name__ == "__main__":
    log("======== START ==========")
    load_files(9)
    scale_attributes()
    test_train_split()
    train()
    log_batch_size=201
    log("======== END ==========")
    log_file.close()
