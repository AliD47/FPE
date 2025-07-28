import pandas as pd
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
import tensorflow as tf
tf.random.set_seed(47)
np.random.seed(47)

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
def build_best_model(params):
    filter_size = params['filter_size']
    padding_type = 'same'  # This is what they used in v1D
    num_filters = params['num_filters']

    initializer = tf.keras.initializers.he_normal(seed=123)

    inputs = tf.keras.Input(shape=(700,))
    x = tf.keras.layers.Reshape((700, 1))(inputs)
    x = tf.keras.layers.Conv1D(
        filters=num_filters,
        kernel_size=filter_size,
        padding=padding_type,
        kernel_initializer=initializer,
        kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg'])
    )(x)
    x = tf.keras.layers.Activation('elu')(x)
    x = tf.keras.layers.Flatten()(x)

    # Only one dense layer, no dropout
    units = params['dense_0_units']
    x = tf.keras.layers.Dense(
        units,
        kernel_initializer=initializer,
        kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg'])
    )(x)
    x = tf.keras.layers.Activation('elu')(x)

    output = tf.keras.layers.Dense(1, activation="linear", kernel_initializer=initializer)(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    return model


for Var in Y.columns:
    print(f"------------------ Processing variable: {Var} ------------------")
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
    # Reshape for CNN input
    X_train_f = X_train_N[..., np.newaxis]
    X_test_f = X_test_N[..., np.newaxis]

    print(f"Y_train: {Y_train.shape}, Y_test: {Y_test.shape}")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    # CNN-R_v1E
    Modd = "CNN-R_v1D"
    best_params = {
        "num_filters": 30,
        "num_dense_layers": 3,
        "filter_size": 10,
        "dense_0_units": 32,
        "dense_1_units": 64,
        "dense_2_units": 124,
        "dropout_rate": 0.2,
        "learning_rate": 0.005,
        "l2_reg": 0.01,
        "batch_size": 32
    }

    # Train final model on full training set with estimated epochs
    final_model = build_best_model(best_params)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta=5e-2, patience=15, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.4, patience=10)
    final_model.fit(X_train_f, Y_train,
                epochs=100,
                batch_size=best_params['batch_size'],
                callbacks=[es_callback, reduce_lr],
                verbose=0)

    # Evaluate on test set
    test_loss, test_rmse = final_model.evaluate(X_test_f, Y_test, verbose=1)
    print(f"RMSE on test set: {test_rmse:.4f}")
    
    ############### Calcul des métriques ###############
    # Calcul des métriques
    rmse = test_rmse
    rpd = np.std(Y_test) / rmse
    relative_error = rmse / np.mean(Y_test)

    # Préparation de la nouvelle ligne pour l'ajouter au CSV
    new_row = pd.DataFrame({
        "Modèle": [Modd],
        "Variable": [Var],
        "RMSE": [rmse],
        "RE": [relative_error],
        "RPD": [rpd]
    })

    # Chemin vers la fichier CSV des résultats
    csv_path = "C:/Users/Administrator/Desktop/PFE/otha/results/resultats_CNN_defaults.csv"

    # Si le fichier existe déjà, on l'ouvre et on ajoute la nouvelle ligne
    if os.path.exists(csv_path):
        existing_results = pd.read_csv(csv_path)
        updated_results = pd.concat([existing_results, new_row], ignore_index=True)
    else:
        # Si le fichier n'existe pas encore, on crée un nouveau fichier avec juste cette ligne
        updated_results = new_row

    # Sauvegarde
    updated_results.to_csv(csv_path, index=False, sep=',')
    print(updated_results)
