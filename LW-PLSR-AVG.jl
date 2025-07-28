using CSV, DataFrames, Plots, LinearAlgebra, Random, Jchemo, GLMakie, CairoMakie

X = CSV.read("C:/Users/Administrator/Desktop/PFE/data/new/miscplants_Xp.csv", DataFrame)
Y = CSV.read("C:/Users/Administrator/Desktop/PFE/data/new/miscplants_Y.csv", DataFrame)
M = CSV.read("C:/Users/Administrator/Desktop/PFE/data/new/miscplants_M.csv", DataFrame)

function split_data(x, y, m, var)
    # Load CSV files as DataFrames
    
    # Ensure the variable exists in Y and M
    if !(var in names(y)) || !(var in names(m))
        throw(ArgumentError("Variable '$var' not found in Y or M."))
    end
    
    # Extract the partition column for the target variable
    mask = m[:, var]
    
    # Split X and Y based on M values
    X_cal = x[mask .== "cal", :]
    Y_cal = y[mask .== "cal", var]
    
    X_val = x[mask .== "val", :]
    Y_val = y[mask .== "val", var]
    
    X_test = x[mask .== "test", :]
    Y_test = y[mask .== "test", var]
    
    return (X_cal, Y_cal), (X_val, Y_val), (X_test, Y_test)
end
function save_results!(var_name::String, RMSE, R2, RE, RPD)
    global results_df  

    # Ensure the values are scalars
    RMSE = only(RMSE)
    R2 = only(R2)
    RE = only(RE)
    RPD = only(RPD)  # Ensure RPD is a scalar

    # Create a new row as a DataFrame
    new_row = DataFrame(Variable=[var_name], RMSE=[RMSE],  
                        R2=[R2], RE=[RE], RPD=[RPD])

    # Append the new row to results_df
    append!(results_df, new_row)
end
results_df = DataFrame(Variable=String[], RMSE=Float64[],  R2=Float64[], RE=Float64[], RPD=Float64[])


# # # # # # #   Variable to use to split the data  # # # # # # # # 
variables = ["adf", "adl", "cf", "cp", "dmdcell", "ndf"]
for Var in variables
    (X_cal, Y_cal), (X_val, Y_val), (X_test, Y_test) = split_data(X, Y, M, Var)
    # Combine calibration and validation sets
    X_train = vcat(X_cal, X_val)
    Y_train = vcat(Y_cal, Y_val)

    nlvdis = [5; 15; 25] ; metric = [:mah]
    h = [1; 1.8; 2.5; 3.5; 5] ; k = [150; 300; 500; 600; 750; 1000] 
    nlv = [0:10, 0:15, 0:20, 5:10, 5:15, 5:20]
    pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k, nlv = nlv)

    model = lwplsravg()
    zres = gridscore(model, X_cal, Y_cal, X_val, Y_val; score = rmsep, pars, verbose = false)
    u = findall(zres.y1 .== minimum(zres.y1))[1]
    zres[u, :]   
    model = lwplsravg(nlvdis = zres.nlvdis[u], metric = zres.metric[u], 
        h = zres.h[u], k = zres.k[u], nlv = zres.nlv[u], verbose = false)
    fit!(model, X_train, Y_train)
    pred = Jchemo.predict(model, X_test).pred
    # @show rmsep(pred, Y_test)

    RMSE = only(rmsep(pred, Y_test))  
    optimal_nlv = zres.nlv[u]  
    R2 = only(r2(pred, Y_test))
    RE = RMSE / (only(meanv(pred))) * 100
    RPD = only(rpd(pred, Y_test))
    save_results!(Var, RMSE, R2, RE, RPD)

end

results_df

CSV.write("results_lwplsr-avg.csv", results_df)






