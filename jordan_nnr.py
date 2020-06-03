from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas import DataFrame, Series, read_csv, errors, concat
from numpy import zeros, array, append, concatenate, multiply, vstack, matrix, around
from numpy.random import rand, seed
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from os import path, listdir
from datetime import datetime
from math import sqrt, exp
import re

g_model_folder = 'models'
g_data_folder = 'data'
g_y_colname = 'precipitation'
g_plot_folder = 'plots'

test_size = 0.3
colnames = {
    "Basel Temperature [2 m elevation corrected]": "tempMin",
    "Basel Temperature [2 m elevation corrected].1": "tempMax",
    "Basel Temperature [2 m elevation corrected].2": "tempMean",
    # "Basel Relative Humidity [2 m]": "rHumidMin",
    # "Basel Relative Humidity [2 m].1": "rHumidMax",
    "Basel Relative Humidity [2 m].2": "rHumidMean",
    "Basel Precipitation Total": "precipitation",
    "Basel Cloud Cover Total": "cloudCoverage",
    "Basel Evapotranspiration": "evapor",
    # "Basel Wind Speed [10 m]": "windSpeedMin",
    # "Basel Wind Speed [10 m].1": "windSpeedMax",
    "Basel Wind Speed [10 m].2": "windSpeedMean",
    # "Basel Soil Temperature [0-10 cm down]": "soilTempMin",
    # "Basel Soil Temperature [0-10 cm down].1": "soilTempMax",
    "Basel Soil Temperature [0-10 cm down].2": "soilTempMean",
    # "Basel Soil Moisture [0-10 cm down]": "soilMoistMin",
    # "Basel Soil Moisture [0-10 cm down].1": "soilMoistMax"
    "Basel Soil Moisture [0-10 cm down].2": "soilMoistMean"
}
n_attrs = len(colnames) - 1  # broj atributa koji se koriste za predvidjanje
df = None
x_test = None
y_test = None
x_train = None
y_train = None

seed(7)


def activation_f(u):
    return 1.0 / (1.0 + exp(-u))


eta = 0.3
alpha = 0.2
N = 6                           # broj dana za predikciju
m = (int)((N * n_attrs) * 4)    # broj skrivenih cvorova
epoch = 0
max_epochs = 20
eps_decimals = 4

w_ = None
w__ = None
best_w_ = None
best_w__ = None
best_mse = None
h = None
bias = array([1.], dtype=float)

file_time = datetime.now().strftime("%H_%M_%S")
model_file = "model"+file_time+".txt"

log_output = True
log_file = open(path.join("logger", "logger_"+file_time+".txt"),
                "w") if log_output else None
log_batch = []
log_batch_size = 0
log_batch_max = 200


def log(msg: str):
    if log_output:
        global log_file, log_batch, log_batch_max, log_batch_size
        log_batch += ["<{}> - {}\n".format(
            datetime.now().strftime("%H:%M:%S"), msg)]
        log_batch_size += 1
        if log_batch_max < log_batch_size:
            log_file.writelines(log_batch)
            log_batch_size = 0
            log_batch.clear()


def f_scale_y(x):
    # novi range [0, 100]->[0.1, 0.9]
    # pogodnihi kada imamo puno nula za vrednosti
    return x / 100 * 0.8 + 0.1


def f_unscale_y(x):
    return (x - 0.1) * 100 / 0.8


def pd_load_file(file_path: str, num_rows_to_skip: int) -> DataFrame:
    try:
        if not path.isfile(file_path):
            log("Proveriti putanju do fajla")
            raise Exception('Proverite putanju do fajla!')
        if file_path.endswith('.csv'):
            log("ucitan fajl: " + file_path)
            return read_csv(file_path, low_memory=False, encoding='utf-8', skiprows=num_rows_to_skip, usecols=[colname for colname in colnames.keys()])

    except errors.ParserError:
        log("Nema dovoljno memorije")
        raise Exception('Doslo je do greske. Nema dovoljno memorije!')


def load_files(num_rows_to_skip: int):
    global df
    df_list = []
    shape = 0
    i = 0
    log("Ucitavanje fajlova...")
    for file_name in listdir(g_data_folder):
        df_list += [pd_load_file(path.join(g_data_folder,
                                           file_name), num_rows_to_skip)]
        shape += df_list[-1].shape[0]
        df_list[i]
        i += 1
    df = concat(df_list, ignore_index=True)
    df.rename(columns=colnames, inplace=True)
    log("Preimenovanje kolona: "+str([c for c in df.columns]))
    log('Ukupno redova: ' + str(df.shape[0]))


def test_train_split_mine():
    global df, x_train, y_train, x_test, y_test, test_size, g_y_colname, n_x_test, n_x_train
    Y = df[g_y_colname]
    X = df.drop(columns=[g_y_colname])
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, shuffle=False, test_size=test_size)
    n_x_train = x_train.shape[0]
    n_x_test = x_test.shape[0]
    log("Podela na test i trening skup\ntrain: x = {:>20}, y = {:>20}; test: x = {:>20}, y = {:>20};".format(
        str(x_train.shape), str(y_train.shape), str(x_test.shape), str(y_test.shape)))

    x_test.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)


def scale_attributes():
    global x_train, x_test, y_train, y_test
    scaler = MinMaxScaler(feature_range=(0.1, 0.9))
    scaler.fit(x_train, y_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    y_train = f_scale_y(y_train)
    y_test = f_scale_y(y_test)


# Cuvanje modela u tekstualnom fajlu. Bitno nam je da sacuvamo sve tezinske matrice.
# Cuva se model koji je imao najmanju mse prilikom treniranja.
# Format fajla:
# n m -- dimenzije w'
# -- linije koje odgovaraju redovima matrice w'
# m -- velicina w''
# -- linija koja odgovara nizu w''
def save_model():
    global best_w__, best_w_, model_file, g_model_folder
    tmp = "Cuvanje modela: {}".format(model_file)
    log(tmp)
    print(tmp)
    model_path = path.join(g_model_folder, model_file)
    with open(model_path, "a") as model:
        model.write((str)(best_w_.shape[0]) +
                    " " + (str)(best_w_.shape[1]) + '\n')
        for i in best_w_:
            model.write(' '.join([str(k) for k in i]) + '\n')
        model.write((str)(best_w__.shape[0]) + '\n')
        model.write(' '.join([str(k) for k in best_w__]) + '\n')
    tmp = "Uspesno cuvanje modela..."
    log(tmp)
    print(tmp)


def load_model():
    global best_w_, best_w__, g_model_folder, model_file, m
    # Moze da bira hoce li da se ucita postojeci ili ovaj koji je sada izgenerisan
    ans = (input("Ucitati model? (Y/N)\n")).lower().strip()
    while ans not in ['y', 'n']:
        ans = input(
            "Format odgovora y/n ili Y/N! Ponovite unos:\n").lower().strip()
    if best_w_ is None or best_w__ is None:
        print("Zapoceto treniranje...")
        train()
        return model_file
    if ans == 'y':
        print("Lista dostupnih modela:")
        i = 1
        model_list = []
        for file_name in listdir(g_model_folder):
            if file_name.startswith("model"):
                model_list += [file_name]
                print("\t", i, file_name)
            i += 1
        model_no = (int)(
            input("Unesite broj modela koji zelite da ucitate:\n"))
        model_path = path.join(g_model_folder, model_list[model_no-1])
        log("Zapoceto ucitavanje modela: {}".format(model_list[model_no-1]))
        with open(model_path, "r") as model:
            w_shape = [int(a) for a in model.readline().split(' ')]
            tmp = []
            for i in range(w_shape[0]):
                tmp += [float(a) for a in model.readline().split(' ')]
            best_w_ = matrix(tmp).reshape(w_shape[0], w_shape[1])
            m = int(model.readline().split(' ')[0])
            best_w__ = array([float(a) for a in model.readline().split(' ')])
        print("Ucitan model: {}".format(model_list[model_no-1]))
        model_file = model_list[model_no-1]
        return model_file


def test():
    global N, m, best_w__, best_w_, h, epoch, y_test, bias, n_attrs
    log("Zapoceta evaluacija...")
    init_iter = True
    num_of_patterns = len(x_test) - N
    o = zeros(num_of_patterns)
    output_arr = array([])
    day = 0
    while day < (num_of_patterns - 1):
        input_nodes = concatenate(
            (bias, (x_test[day:(day+N)]).reshape(-1), output_arr))
        day += 1
        # Izracunavanje h1,..., hm cvorova skrivenog sloja
        # U prvoj iteraciji nemamo output cvor tako da ne koristimo celu w' matricu
        u = []
        if init_iter:
            for i in best_w_[:-1].T:
                u += [activation_f(multiply(i, input_nodes).sum())]
            init_iter = False
        else:
            for i in best_w_.T:
                u += [activation_f(multiply(i, input_nodes).sum())]
        h = array(u)
        h[0] = 1  # bias ostaje nepromenjen

        # Izracunavanje o1,...,op
        o[day] = activation_f(sum(multiply(h, best_w__)))
        output_arr = array([o[day]])
    mse = mean_squared_error(y_test[N:], o)
    mae = mean_absolute_error(y_test[N:], o)
    msg = "MSE - test: {:<25} MAE - test: {:<25}".format(mse, mae)
    unsc_y = f_unscale_y(y_train[N:])
    unsc_o = f_unscale_y(o)
    for i in range(len(unsc_o)):
        if unsc_o[i] < 0:
            unsc_o[i] = 0
    print(msg)
    log(msg)
    plt.figure()
    plt.plot(range(365), unsc_y[-365:], 'r', range(365), unsc_o[-365:], 'b')
    plt.title('Last 365 days of test data')
    plt.xlabel('day No.')
    plt.ylabel('precipitation amount')
    save_path = path.join(g_plot_folder, 'plot_test_model-' +
                          model_file[:-4]+'-time_'+file_time+'.png')
    plt.savefig(save_path, format='png')
    plt.show()
    return mse


def train():
    # na kraju iteracije dodajemo output koji smo dobili.
    # vodimo racuna o epohama velicinama matrica
    global N, m, w__, w_, h, epoch, y_train, max_epochs, bias, n_attrs, eps_decimals, best_mse, best_w_, best_w__
    init_iter = True
    n = N * n_attrs + 1     # broj ulaznih cvorova + 1 za bias
    n_x = len(x_train)
    num_of_patterns = n_x - N
    m += 1                  # broj cvorova skrivenog sloja + bias
    w_ = rand(n, m) - 0.5
    dw_ = zeros((n, m))
    w__ = rand(m) - 0.5     # posto imamo samo jedan output ovo je obican niz
    dw__ = zeros(m)
    o = zeros(num_of_patterns)
    output_arr = array([])
    day = 0
    end_of_epoch = n_x - N - 1
    while epoch <= max_epochs:
        if day == end_of_epoch:
            # mse = test() # evaluacija nad nepoznatim podacima
            mse = mean_squared_error(y_train[N:], o)
            if best_mse is None or mse < best_mse:
                best_mse = mse
                best_w_ = w_
                best_w__ = w__

            mae = mean_absolute_error(y_train[N:], o)
            rmse = sqrt(mse)
            unsc_y = f_unscale_y(y_train[N:])
            unsc_o = f_unscale_y(o)
            msg = "epoch:{:>4}/{:<4} mse: {:<25} mae: {:<25} rmse: {:<25}".format(
                epoch, max_epochs, mse, mae, rmse)
            day = 0
            epoch += 1
            log(msg)
            print(msg)
            # Ukoliko je negativna vrednost, smatracemo da je nula.
            for i in range(len(o)):
                if unsc_o[i] < 0:
                    unsc_o[i] = 0
            # Plot za poslednjih godinu dana prve, sredisnje i poslednje epohe koji ide na istu figuru
            if epoch in [1, max_epochs//2, max_epochs]:
                plt.figure()
                plt.plot(range(365), unsc_y[-365:], 'r', range(365), unsc_o[-365:], 'b')
                plt.title('Last 365 days of Epoch: ' + str(epoch))
                plt.xlabel('day No.')
                plt.ylabel('precipitation amount')
                save_path = path.join(g_plot_folder, 'plot_epoch_'+str(epoch)+'_time_'+file_time+'.png')
                plt.savefig(save_path, format='png')

        input_nodes = concatenate(
            (bias, (x_train[day:(day+N)]).reshape(-1), output_arr))
        n = len(input_nodes)
        day += 1

        # Izracunavanje h1,..., hm cvorova skrivenog sloja
        u = []
        for i in w_.T:
            u += [activation_f(sum(multiply(i, input_nodes)))]
        h = array(u)
        h[0] = 1  # bias ostaje nepromenjen

        # Izracunavanje o1,...,op
        o[day] = activation_f(sum(multiply(h, w__)))
        output_arr = array([o[day]])
        o_unsc = round(f_unscale_y(o[day]), ndigits=eps_decimals)
        log("true: {:<20} pred: {:<30} true_unsc: {:<10} pred_unsc: {:<10} <0 : {}".format(
            y_train[day+N], o[day], round(f_unscale_y(y_train[day+N]), ndigits=eps_decimals), 0 if o_unsc < 0 else o_unsc, (o_unsc < 0)))

        error = (y_train[day+N] - o[day]) * o[day] * (1.0 - o[day])
        # Izracunavanje deltaH(j)
        dh = w__ * error

        # Azuriranje W'(ij) i deltaW'(ij)
        H = h * (1-h) * dh * eta
        dw_ *= alpha
        for j in range(0, m):
            dw_.T[j] += (input_nodes * H[j])
        w_ += dw_

        # Azuriranje W''(jk) i deltaW''(jk)
        dw__ *= alpha
        dw__ += (eta * h * error)
        w__ += dw__

        # Ako smo prosli inicijalnu iteraciju postavljamo output vrednost na ulaz
        # i prosirujemo matricu
        if init_iter:
            w_ = vstack((w_, rand(m)-0.5))
            dw_ = vstack((dw_, zeros(m)))
            init_iter = False
    save_model()
    plt.show()


if __name__ == "__main__":
    log("======== START ==========")
    tmp = input("parametar alpha:\n[default = " +
                str(alpha)+", enter za skip]\n")
    if tmp != '':
        alpha = float(tmp)
    tmp = input("parametar eta:\n[default = "+str(eta)+", enter za skip]\n")
    if tmp != '':
        eta = float(tmp)
    msg = "broj dana za predikciju= {}; broj ulaznih cvorova= {}; broj cvorova skriveni sloj= {}\nalpha = {} eta = {}".format(
        N, N*n_attrs, m, alpha, eta)
    log(msg)
    print(msg)
    load_files(9)
    test_train_split_mine()
    scale_attributes()
    model_name = load_model()
    test()

    log_batch_size = 201
    log("======== END ==========")
    log_file.close()

