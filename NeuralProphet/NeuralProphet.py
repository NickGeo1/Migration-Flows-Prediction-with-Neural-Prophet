from neuralprophet import NeuralProphet, save, load
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_excel('balkan_route.xlsx',skiprows=15, usecols="A:J")  # Use nrows=n for n first rows
df = df.rename(columns={"Date": "ds"})

final_df = []
original_names = []
for i, c in enumerate(df.columns[1:]):
    original_names.append(c)
    df1 = df[["ds", c]]
    df1 = df1.rename(columns={c: "y"})
    df1["ID"] = f"series{i+1}"
    final_df.append(df1)

single_df_len = len(final_df[0]["ds"].values)  # Length of a single time series
final_df = pd.concat(tuple(final_df))  # contains a 3 column df: ds, y and ID. each column has all series data
final_df = final_df.reset_index(drop=True) # make indexing global instead of 1...365 1..365 ...
print(final_df)

def train_on_one_80_test_on_one_20(load_model: bool, test_idx: int = 0):

    global final_df
    # Make dataframe to store R^2 and NMAE in exel
    '''fcst_metrics = pd.DataFrame({"Data" : [f"Train {original_names[test_idx]}", f"Test {original_names[test_idx]}"], "R^2" : [None, None], "NMAE" : [None, None] })'''

    if load_model:
        m = load(f"trained_model_80_test20_{original_names[test_idx]}.np")
    else:
        m = NeuralProphet(n_lags=4, n_forecasts=1, normalize="standardize", n_changepoints=4)
        m.add_events("EU-Turkey Cutoff")

    # Create event vector which lasts from EU-Turkey Cutoff date to end
    df_event = pd.DataFrame({"event": "EU-Turkey Cutoff", "ds": pd.date_range(start='2016-03-20', end='2016-09-21', freq='D')})
    final_df = m.create_df_with_events(final_df, df_event)

    #Split to train and test the chosen time series.
    series = final_df[test_idx * single_df_len:(test_idx + 1) * single_df_len]
    df_train, df_test = (series[:int(len(series) * 0.8)], series[int(len(series) * 0.8):])

    print(df_train)
    print(df_test)

    if not load_model:
        metrics = m.fit(df_train)
        test_metrics = m.test(df_test)

    train = m.predict(df_train)
    test = m.predict(df_test)

    m.set_plotting_backend("matplotlib")

    fig, axs = plt.subplots(1, 2)

    X = train["ds"].values
    Y = train["y"].values # observed
    Yhat = train["yhat1"].values  # predicted

    r2 = r2_score(Y[m.n_lags:], Yhat[m.n_lags:])
    mae = mean_absolute_error(Y[m.n_lags:], Yhat[m.n_lags:])
    nmae = mae/(max(Y) - min(Y))

    axs[0].plot(X, Yhat, label="predicted")
    axs[0].scatter(X, Y, s=10,color='k',label="observed")
    axs[0].set_xlabel("Dates")
    axs[0].set_ylabel("Values")
    axs[0].set_title(f"Train {original_names[test_idx]} R^2 = {round(r2, 2)}  NMAE = {round(nmae, 2)}")
    axs[0].legend()
    plt.sca(axs[0])
    plt.xticks(rotation=90, fontsize=10)

    X = test["ds"].values
    Y =  test["y"].values # observed
    Yhat = test["yhat1"].values # predicted

    r2_test = r2_score(Y[m.n_lags:], Yhat[m.n_lags:])
    mae_test = mean_absolute_error(Y[m.n_lags:], Yhat[m.n_lags:])
    nmae_test = mae_test / (max(Y) - min(Y))

    axs[1].plot(X, Yhat, label="predicted")
    axs[1].scatter(X,  Y, s=10, color='k',label="observed")
    axs[1].set_xlabel("Dates")
    axs[1].set_ylabel("Values")
    axs[1].set_title(f"Test {original_names[test_idx]} R^2 = {round(r2_test, 2)} NMAE = {round(nmae_test, 2)}")
    axs[1].legend()
    plt.sca(axs[1])
    plt.xticks(rotation=90, fontsize=10)

    m.plot_parameters(df_name=f"series{test_idx+1}")
    m.plot_components(train)
    m.plot_components(test)

    plt.show()

    # Store metrics in exel
    '''fcst_metrics["R^2"] = [round(r2, 2), round(r2_test, 2)]
    fcst_metrics["NMAE"] = [round(nmae, 2), round(nmae_test, 2)]
    existing_data = pd.read_excel("80_20_metrics.xlsx")
    updated_data = pd.concat([existing_data, fcst_metrics], ignore_index=True)
    updated_data.to_excel("80_20_metrics.xlsx", index=False)'''

    if not load_model:
        save(m,f"trained_model_80_test20_{original_names[test_idx]}.np")



def train_on_all_100_test_on_one_100(load_model: bool, test_idx: int = 0):

    global final_df

    #Create fcst metrics df
    '''df_dic = {"Metric/using test" : [f"R^2/{original_names[test_idx]}", f"NMAE/{original_names[test_idx]}"]}
    for i in range(9):
        df_dic[f"{original_names[i]}"] = [None]*2
    fcst_metrics = pd.DataFrame(df_dic)'''

    if load_model:
        m = load(f"trained_model_100_test_{original_names[test_idx]}.np")

    else:
        m = NeuralProphet(unknown_data_normalization=True, n_lags=4, n_forecasts=1, normalize="standardize", n_changepoints=4)
        m.add_events("EU-Turkey Cutoff")

    # Create event vector which lasts from EU-Turkey Cutoff date to end
    df_event = pd.DataFrame({"event": "EU-Turkey Cutoff", "ds": pd.date_range(start='2016-03-20', end='2016-09-21', freq='D')})
    final_df = m.create_df_with_events(final_df, df_event)

    df_test = final_df[test_idx * single_df_len:(test_idx + 1) * single_df_len]  # Store the desired test country from train data
    df_train = final_df.drop(df_test.index)  # Remove the test country from train data

    if not load_model:
        metrics = m.fit(df_train)  # Train on all but one series

    train = m.predict(df_train)  # Train data
    test = m.predict(df_test)  # Test data

    m.set_plotting_backend("matplotlib")

    for i in range(8):

        X = train["ds"].values[i*single_df_len:(i+1)*single_df_len]
        Y = train["y"].values[i * single_df_len:(i + 1) * single_df_len]  # observed
        Yhat = train["yhat1"].values[i*single_df_len:(i+1)*single_df_len]  # predicted

        r2 = r2_score(Y[m.n_lags:],Yhat[m.n_lags:])
        mae = mean_absolute_error(Y[m.n_lags:],Yhat[m.n_lags:])
        nmae = mae / (max(Y) - min(Y))

        name_idx = int(train['ID'][i*single_df_len][-1]) - 1

        plt.figure()
        plt.plot(X,Yhat,label="predicted")
        #plt.vlines(x=X[284], ymin=0, ymax=max(Y), colors='green', label='Train end')
        plt.scatter(X,Y,s = 10,color='k',label="observed")
        plt.xlabel("Dates")
        plt.xticks(rotation=90, fontsize=10)
        plt.ylabel("Values")
        #plt.title(f"{original_names[name_idx]}")
        plt.title(f"Train {original_names[name_idx]} R^2 = {round(r2, 2)} NMAE = {round(nmae, 2)}")
        plt.legend()

        # Update fcst metrics with the specific train timeseries metrics
        '''fcst_metrics[f"{original_names[name_idx]}"] = [round(r2, 2), round(nmae, 2)]'''

    X = test["ds"].values
    Y = test["y"].values  # observed
    Yhat = test["yhat1"].values  # predicted

    r2 = r2_score(Y[m.n_lags:], Yhat[m.n_lags:])
    mae = mean_absolute_error(Y[m.n_lags:], Yhat[m.n_lags:])
    nmae = mae / (max(Y) - min(Y))

    plt.figure()
    plt.plot(X,Yhat,label="predicted")
    #plt.vlines(x=X[284], ymin=0, ymax=max(Y), colors='green', label='Train end')
    plt.scatter(X,Y,s=10,color='k',label="observed")
    plt.xlabel("Dates")
    plt.xticks(rotation=90, fontsize=10)
    plt.ylabel("Values")
    #plt.title(f"{original_names[test_idx]}")
    plt.title(f"Test {original_names[test_idx]} R^2 = {round(r2, 2)} NMAE = {round(nmae, 2)}")
    plt.legend()

    m.plot_components(test)
    m.plot_parameters(df_name= f"series{test_idx+1}")  # same for all timeseries(global params)
    plt.show()

    # Update fcst metrics with the test timeseries metrics
    '''fcst_metrics[f"{original_names[test_idx]}"] = [round(r2, 2), round(nmae, 2)]

    # Store metrics in exel
    existing_data = pd.read_excel("100_all_100_one_metrics.xlsx")
    updated_data = pd.concat([existing_data, fcst_metrics], ignore_index=True)
    updated_data.to_excel("100_all_100_one_metrics.xlsx", index=False)'''

    if not load_model:
        save(m,f"trained_model_100_test_{original_names[test_idx]}.np")



for i in range(9):

    train_on_all_100_test_on_one_100(True, i)
    #train_on_one_80_test_on_one_20(True, i)


