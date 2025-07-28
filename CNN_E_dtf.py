import pandas as pd
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import shapiro
import warnings
warnings.filterwarnings('ignore')
import os
import tensorflow as tf
from sklearn.model_selection import KFold
import optuna 
tf.random.set_seed(47)
np.random.seed(47)
# import keras_tuner as kt

X = pd.read_csv("C:/Users/Administrator/Desktop/PFE/data/new/miscplants_Xp.csv", sep=';')
Y = pd.read_csv("C:/Users/Administrator/Desktop/PFE/data/new/miscplants_Y.csv", sep=';')
M = pd.read_csv("C:/Users/Administrator/Desktop/PFE/data/new/miscplants_M.csv", sep=';', na_values ='missing')

def split_data(Var):
    if Var not in Y.columns or Var not in M.columns:
        raise ValueError(f"Errer Erreur Erreur ! ! !")
    mask = M[Var]
    # Split X and Y based on M.csv values
    X_cal = X[mask == 'cal']
    Y_cal = Y.loc[X_cal.index, Var]
    X_val = X[mask == 'val']
    Y_val = Y.loc[X_val.index, Var]
    X_test = X[mask == 'test']
    Y_test = Y.loc[X_test.index, Var]
    return (X_cal, Y_cal), (X_val, Y_val), (X_test, Y_test)

# Var = "adl"
Modd = "CNN-R_v1E"
for Var in Y.columns:
    # Data Splitting
    (X_cal, Y_cal), (X_val, Y_val), (X_test, Y_test) = split_data(Var)
    Y_train = pd.concat([Y_cal, Y_val])
    X_train = pd.concat([X_cal, X_val])

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    num_features = 700  # Spectral features

    mean_train, std_train = X_train.mean(axis=0), X_train.std(axis=0)
    X_train_N = (X_train - mean_train) / std_train
    X_test_N = (X_test - mean_train) / std_train

    X_train_f = X_train_N[..., np.newaxis]
    X_test_f = X_test_N[..., np.newaxis]

    # print(f"Y: {Y_train.shape}, {Y_test.shape}")
    # print(f"X: {X_train.shape}, {X_test.shape}")

    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPUs Available:", gpus)
    else:
        print("No GPU detected; please configure CUDA.")

    ###############################################  build the model
    def build_model(trial):
        filter_size = 700
        padding_type = 'valid'
        num_filters = trial.suggest_int("num_filters", 1, 32)
        # Dense layers
        num_dense_layers = trial.suggest_int("num_dense_layers", 1, 3)
        dense_units = [trial.suggest_int(f"dense_{i}_units", 8, 128, step=4) for i in range(num_dense_layers)]
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.6, step=0.005) 
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.03, log=True) # log=True for logarithmic scale
        l2_reg = trial.suggest_float("l2_reg", 0.0, 0.1, step=5e-4) 
        batch_size = trial.suggest_int("batch_size", 32, 256, step=16)
        initializer = tf.keras.initializers.he_normal(seed=123)
        inputs = tf.keras.Input(shape=(700,))
        x = tf.keras.layers.Reshape((700, 1))(inputs)
        x = tf.keras.layers.Conv1D(filters = num_filters, kernel_size=filter_size,
                                   padding = padding_type,
                                   kernel_initializer = initializer,
                                   kernel_regularizer = tf.keras.regularizers.l2(l2_reg))(x)
        # print("Conv1D output shape:", x.shape) 
        x = tf.keras.layers.Activation('elu')(x)
        x = tf.keras.layers.Flatten()(x)
        for i, units in enumerate(dense_units):
            x = tf.keras.layers.Dense(units, kernel_initializer=initializer,
                                      kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
            x = tf.keras.layers.Activation('elu')(x)
            if num_dense_layers > 1:
                x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
        output = tf.keras.layers.Dense(1, activation="linear", kernel_initializer=initializer)(x)
        model = tf.keras.Model(inputs=inputs, outputs=output)
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss="mse",
                      metrics=[tf.keras.metrics.RootMeanSquaredError()]) 

        return model, batch_size

    def objective(trial):
        epochs_list = []
        es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=5e-2, patience=20, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=10)
        kf = KFold(n_splits=5, shuffle=True, random_state=47)  
        rmse_scores = []
        history_dicts = []  

        for train_idx, val_idx in kf.split(X_train_f):  
            X_train_cv, X_val_cv = X_train_f[train_idx], X_train_f[val_idx]
            Y_train_cv, Y_val_cv = Y_train.iloc[train_idx], Y_train.iloc[val_idx]

            model, batch_size = build_model(trial)  # rebuild the model for each fold
            history = model.fit(X_train_cv, Y_train_cv,
                                validation_data=(X_val_cv, Y_val_cv),
                                epochs=300,
                                batch_size=batch_size,
                                callbacks=[es_callback, reduce_lr],
                                verbose=0)
            epochs_list.append(len(history.history['loss']))
            history_dicts.append(history.history) # store history for each fold


            val_loss, val_rmse = model.evaluate(X_val_cv, Y_val_cv, verbose=0)
            rmse_scores.append(val_rmse)
            avg_epochs = int(np.mean(epochs_list))
            trial.set_user_attr("avg_epochs", avg_epochs)
            trial.set_user_attr("epochs_list", epochs_list)
            trial.set_user_attr("fold_histories", history_dicts)

        return np.mean(rmse_scores)

    print("\n","we innnnnnnnnnnnnnnnnnnnnnn babyyyy!!!!!!")
    ################################################# Start the Optuna study
    ii = datetime.now()
    iii = str(ii.strftime("%Y-%m-%d_%Hh%M"))
    print(" **********************************    Start time:", iii,"   ***********************************","\n")
    # Set up the Optuna study 
    study = optuna.create_study(
        direction="minimize",
        study_name="/optuna/cnn_hpo",
        # storage=f"sqlite:///optuna/cnn_study__{iii}.db",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=100)

    # Print the best trial
    print("Best trial :")
    print(f"  RMSE on Val: {study.best_value:.4f}")
    print("  Best params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    # for i in range(5): winsound.Beep(500, 500)

    ########################################################  Get best params
    best_params = study.best_trial.params
    #  Build model function that accepts params directly (not trial object)
    def build_best_model(params):
        filter_size = 700
        padding_type = 'valid'
        num_filters = params['num_filters']  # or keep as params['num_filters'] if you want

        initializer = tf.keras.initializers.he_normal(seed=123)

        inputs = tf.keras.Input(shape=(700,))
        x = tf.keras.layers.Reshape((700, 1))(inputs)
        x = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=filter_size,
                                   padding=padding_type,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg']))(x)
        x = tf.keras.layers.Activation('elu')(x)
        print("Conv1D output shape:", x.shape)
        x = tf.keras.layers.Flatten()(x)

        for i in range(params['num_dense_layers']):
            units = params[f'dense_{i}_units']
            x = tf.keras.layers.Dense(units, kernel_initializer=initializer,
                                  kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg']))(x)
            x = tf.keras.layers.Activation('elu')(x)
            if params['num_dense_layers'] > 1:
                x = tf.keras.layers.Dropout(rate=params['dropout_rate'])(x)

        output = tf.keras.layers.Dense(1, activation="linear", kernel_initializer=initializer)(x)
        model = tf.keras.Model(inputs=inputs, outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                      loss="mse",
                      metrics=[tf.keras.metrics.RootMeanSquaredError()])
        return model


    ############################################### Train final model on full training set with estimated epochs
    final_model = build_best_model(best_params)
    final_model.fit(X_train_f, Y_train,
                    epochs=study.best_trial.user_attrs["avg_epochs"],
                    batch_size=best_params['batch_size'],
                    verbose=0)
    # Evaluate on test set
    test_loss, test_rmse = final_model.evaluate(X_test_f, Y_test, verbose=1)
    print(f"RMSE on test set: {test_rmse:.4f}")


    ################################################ Calcul des métriques
    # Chemin vers ton fichier CSV
    csv_path = "C:/Users/Administrator/Desktop/PFE/otha/results/resultats_CNN.csv"


    rmse = test_rmse
    rpd = np.std(Y_test) / rmse
    relative_error = rmse / np.mean(Y_test)
    # Préparation de la nouvelle ligne
    new_row = pd.DataFrame({
        "Modèle": [Modd],
        "Variable": [Var],
        "RMSE": [rmse],
        "RE": [relative_error],
        "RPD": [rpd]
    })
    # Si le fichier existe déjà, on l'ouvre et on ajoute la nouvelle ligne
    if os.path.exists(csv_path):
        existing_results = pd.read_csv(csv_path)
        updated_results = pd.concat([existing_results, new_row], ignore_index=True)
    else:
        # Si le fichier n'existe pas encore, on crée un nouveau fichier avec juste cette ligne
        updated_results = new_row
    # Sauvegarde
    updated_results.to_csv(csv_path, index=False, sep=',')