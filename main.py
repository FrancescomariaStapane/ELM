
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from algorithms import *
from result_data import *
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from scipy.io import arff
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings(
    "ignore",
    message="__array_wrap__ must accept context and return_scalar arguments",
    category=DeprecationWarning,
)
datasets_ids = [
    159,  # Gamma telescope: binary classification, 10 real features, 19020 instances,
    59,  # Letter Recognition: 26-class classification,  16 integer features, 20000 instances
    23,  # Chess (King-Rook vs. King): 18-class classification, 6 integer and categorical features, 28056 instances
    2,  # Adult: binary classification, 14 integer and categorical features, 48842 instances
    148,  # Statlog (Shuttle): 7-class classification, 9 integer features, 58000 instances
    158  # Poker hands: 10-class classification, 10 integer features, 1025010 instances
]


def initialize_heatmap_data(n_neuron_len, lmda_len):
    return np.zeros((len(n_neuron_len), len(lmda_len)))


def update_heatmap_data(heatmap_data, mean_loss, n_neuron_idx, lmda_idx):
    heatmap_data[n_neuron_idx, lmda_idx] = mean_loss


def plot_heatmap(heatmap_data, n_neuron_len, lmda_len, name, type):
    n_neuron_len = n_neuron_len.tolist()
    lmda_len = lmda_len.tolist()

    lmda_len_str = [f"{x:.1e}" for x in lmda_len]
    if type == 'c':
        max_val = np.max(heatmap_data)
        max_pos = np.unravel_index(np.argmax(heatmap_data), heatmap_data.shape)
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(heatmap_data, annot=False, fmt=".2f", xticklabels=lmda_len_str, yticklabels=n_neuron_len,
                         cmap="viridis")

        ax.add_patch(plt.Rectangle((max_pos[1], max_pos[0]), 1, 1, fill=False, edgecolor='blue', lw=3))
        ax.text(max_pos[1] + 0.5, max_pos[0] + 0.5, f"Max: {max_val:.2f}", color='blue', ha="center", va="center",
                fontsize=12, fontweight='bold')
        plt.xlabel("Learning Rate")
        plt.ylabel("Number of Neurons")
        plt.title(str("Accuracy Heatmap " + name))
    else:
        min_val = np.min(heatmap_data)
        min_pos = np.unravel_index(np.argmin(heatmap_data), heatmap_data.shape)
        plt.figure(figsize=(10, 8))

        ax = sns.heatmap(heatmap_data, annot=False, fmt=".2f", xticklabels=lmda_len_str, yticklabels=n_neuron_len,
                         cmap="viridis")

        ax.add_patch(plt.Rectangle((min_pos[1], min_pos[0]), 1, 1, fill=False, edgecolor='red', lw=3))
        ax.text(min_pos[1] + 0.5, min_pos[0] + 0.5, f"Min: {min_val:.2f}", color='red', ha="center", va="center",
                fontsize=12, fontweight='bold')
        plt.xlabel("Lambda")
        plt.ylabel("Number of Neurons")
        plt.title(str("Loss Heatmap " + name))

    plt.show()
def one_hot_encode_targets(targets):
    classes_lookup_table, y = np.unique(targets, return_inverse=True)
    onehot_encoded_targets = np.zeros((y.size, y.max() + 1), dtype=int)
    for i in range(len(onehot_encoded_targets)):
        for j in range(len(classes_lookup_table)):
            if y[i] == j:
                onehot_encoded_targets[i][j] = 1
    return onehot_encoded_targets

def one_hot_encode_features(dataset, features):
    dataset_one_hot = dataset.copy()
    for feature in features:
        dummies = pd.get_dummies(dataset[feature], dtype=int)
        dataset_one_hot = pd.concat([dataset_one_hot, dummies], axis=1)
    for feature in features:
        dataset_one_hot = dataset_one_hot.drop([feature], axis=1)
    return dataset_one_hot

def slice_dataset_balanced(x, y, test_frac):
    if not (0 <= test_frac <= 1):
        raise ValueError("val_frac and test_frac must be between 0 and 1.")
    labels = np.argmax(y, axis=1)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=42)
    trainval_idx, test_idx = next(sss_test.split(x, labels))

    x_trainval, x_test = x[trainval_idx], x[test_idx]
    y_trainval, y_test = y[trainval_idx], y[test_idx]
    x_train, y_train = x_trainval, y_trainval
    return x_test, y_test, x_train, y_train

def slice_dataset_regression(x, y, test_frac):
    if not (0 <= test_frac <= 1):
        raise ValueError("test_frac must be between 0 and 1.")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_frac, random_state=42, shuffle=True
    )

    return x_train, y_train, x_test, y_test
def chess_map(letter):
    return ord(letter)-96

def print_results(results: list[Experiment]):
    file_name = "results.txt"
    with open(file_name, "a", encoding="utf-8") as f:
        for experiment in results:
            f.write("\n\n--------------------------------------------------------------------------\n")
            f.write("results for " + experiment.name + "\n")
            f.write("\n")
            for info in experiment.experiment_infos:
                f.write(str(info.name + ": " + str(info.value) + "\n"))
            if len(experiment.repeated_measurements) > 0:
                f.write(str("number of measurements: " + str(experiment.repeated_measurements[0].get_n_of_measurements()) + "\n"))
            for measurement in experiment.repeated_measurements:
                f.write("\n")
                f.write("mean " + measurement.name + ": " + str(measurement.get_mean()) + "\n")
                f.write("std " + measurement.name + ": " + str(measurement.get_std()) + "\n")
                # print("median ",measurement.name, ": ",measurement.get_median())
            print("printed results for ", experiment.name)



def test_grid_search(model: MlAlgorithm, iterations: int, test_name: str):
    if isinstance(model,CrossEntropyElm):
        # Model parameters configuration
        n_neuron_len = torch.tensor([10, 70, 300, 1000, 2500])
        # n_neuron_len = torch.tensor([10,20,30,50,70,80,100,200,300,400,500,600,700,800,900,1000,1500,2000,2500])
        learning_rate_len = torch.tensor([1.0e-1, 1.0e-2, 1.0e-3, 2.0e-4])
        # learning_rate_len = torch.tensor([1.0e-1,2.0e-1,1.0e-2,2.0e-2,1.0e-3,2.0e-3,1.0e-4,2.0e-4])

        heatmap_data = initialize_heatmap_data(n_neuron_len, learning_rate_len)
        best_result = None
        best_accuracy = 0.0
        try:
            for n_neuron in n_neuron_len:
                for learning_rate in learning_rate_len:
                    # print("\n\n--------------------")
                    # print("n_neurons: ", n_neuron.item())
                    # print("learning_rate: ", learning_rate.item())

                    model.n_neurons = n_neuron.item()
                    model.learning_rate = learning_rate.item()
                    model.refresh()
                    if not model.valid:
                        return best_result
                    results = test_model_classification(model, iterations, test_name)
                    # ----------- HEATMAP -----------
                    n_neuron_idx = n_neuron_len.tolist().index(n_neuron)
                    lmda_idx = learning_rate_len.tolist().index(learning_rate)
                    mean_accuracy = results.get_repeated_measurement("accuracy").get_mean()
                    if mean_accuracy > best_accuracy:
                        best_accuracy = mean_accuracy
                        best_result = results
                    update_heatmap_data(heatmap_data, mean_accuracy, n_neuron_idx, lmda_idx)
                    # ----------- FINISH HEATMAP -----------

            print("Test ended!")
            # plot_heatmap(heatmap_data, n_neuron_len, learning_rate_len, test_name,type = 'c')
        except KeyboardInterrupt:
            return best_result
        return best_result
    elif isinstance(model, RegularizedElm):
        best_result = None
        try:
            n_neurons = torch.tensor([100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000])
            lmbdas = torch.tensor([10e-1, 10e-2, 10e-3, 10e-4])
            heatmap_data = initialize_heatmap_data(n_neurons, lmbdas)
            best_rmse = np.inf
            exit= False
            if not model.valid:
                exit = True
            for n_neuron in n_neurons:
                if exit:
                    break
                for lmbda in lmbdas:
                    # print("\n\n--------------------")
                    # print("n_neurons: ", n_neuron.item())
                    # print("lambda: ", lmbda.item())

                    model.n_neurons = n_neuron.item()
                    model.learning_rate = lmbda.item()
                    model.refresh()
                    if not model.valid:
                        exit = True
                        break
                    results = test_model_regression(model, iterations, test_name)
                    if not model.valid:
                        exit = True
                        break
                    mean_mase = results.get_repeated_measurement("MASE").get_mean()
                    mean_rmse = results.get_repeated_measurement("RMSE").get_mean()
                    print()
                    print("mean mase over ", iterations, " iterations:", mean_mase)

                    # ----------- HEATMAP -----------
                    n_neuron_idx = n_neurons.tolist().index(n_neuron)
                    lmda_idx = lmbdas.tolist().index(lmbda)
                    if mean_rmse < best_rmse:
                        best_rmse = mean_rmse
                        best_result = results
                    update_heatmap_data(heatmap_data, mean_rmse, n_neuron_idx, lmda_idx)
                    # ----------- FINISH HEATMAP -----------

            print("Test ended!")
            # plot_heatmap(heatmap_data, n_neurons, lmbdas, test_name, type = 'r')
        except KeyboardInterrupt:
            return best_result
        return best_result

def test_model_classification(model: CrossEntropyElm, iterations: int, test_name: str):
    train_times = []
    test_times = []
    accuracies = []
    f1s = []
    train_energies = []
    test_energies = []
    try:
        for i in range(iterations):
            print("\n\n--------------------")
            print("Experiment: ", test_name)
            print("n_neurons: ", model.get_n_neurons())
            print("lambda: ",model.get_learning_rate())
            print("iteration: ",i+1,"/",iterations)
            model.refresh()
            train_time, train_energy = model.learn()
            test_time, test_energy, returns = model.test()
            accuracy = returns[0]
            f1 = returns[1]
            train_times.append(train_time)
            test_times.append(test_time)
            accuracies.append(accuracy)
            f1s.append(f1)
            train_energies.append(train_energy)
            test_energies.append(test_energy)
            # print("accuracy: ", accuracy.item())

        repeated_measurements = [
            RepeatedMeasurement("train time (ms)",train_times),
            RepeatedMeasurement( "test time (ms)",test_times),
            RepeatedMeasurement("train energy (kWh)",train_energies),
            RepeatedMeasurement("test energy (kWh)",test_energies),
            RepeatedMeasurement("accuracy",accuracies),
            RepeatedMeasurement("f1",f1s),
        ]

        single_measurements = [
            ExperimentInfo("default accuracy",model.get_default_accuracy()),
            ExperimentInfo("neurons", model.get_n_neurons()),
            ExperimentInfo("learning rate", model.get_learning_rate()),
            ExperimentInfo("final n. features", model.get_final_m_features()),
            ExperimentInfo("original n. features", model.get_original_n_features()),
            ExperimentInfo("training instances", model.xtr.shape[0]),
            ExperimentInfo("classes", model.get_n_classes()),

        ]
        experiment = Experiment(test_name, single_measurements, repeated_measurements)
        return experiment
    except MemoryError:
        print("\n\n--------------------")
        print("Experiment: ", test_name)
        print("n_neurons: ", model.get_n_neurons())
        print("learning rate: ", model.get_learning_rate())
        print("Aborted, memory constraint violation")
        experiment = Experiment(test_name,[],[])
        return experiment

def test_model_regression(model: RegularizedElm, iterations: int, test_name: str):
    train_times = []
    test_times = []
    normalized_losses = []
    train_energies = []
    test_energies = []
    percentage_errors = []
    mases = []
    try:
        for i in range(iterations):
            print("\n\n--------------------")
            print("Experiment: ", test_name)
            print("n_neurons: ", model.get_n_neurons())
            print("lambda: ",model.get_lambda())
            print("iteration: ",i+1,"/",iterations)
            if not model.valid:
                break
            model.refresh()
            train_time, train_energy = model.learn()
            test_time, test_energy, returns = model.test()
            rmse = np.sqrt(returns[0])
            mase = returns[1]
            train_times.append(train_time)
            test_times.append(test_time)
            normalized_losses.append(rmse)
            train_energies.append(train_energy)
            test_energies.append(test_energy)
            percentage_errors.append(rmse / model.interval * 100)
            mases.append(mase)
            print("mase: ", float(mase))
        repeated_measurements = [
            RepeatedMeasurement("train time (ms)", train_times),
            RepeatedMeasurement("test time (ms)", test_times),
            RepeatedMeasurement("train energy (kWh)", train_energies),
            RepeatedMeasurement("test energy (kWh)", test_energies),
            RepeatedMeasurement("RMSE", normalized_losses),
            RepeatedMeasurement("Error percentage over target interval (%)", percentage_errors),
            RepeatedMeasurement("MASE", mases),
        ]

        single_measurements = [ExperimentInfo("neurons", model.get_n_neurons()),
                               ExperimentInfo("learning rate", model.get_lambda()),
                               ExperimentInfo("final n. features", model.get_final_m_features()),
                               ExperimentInfo("original n. features", model.get_original_n_features()),
                               ExperimentInfo("training instances", model.xtr.shape[0]),
                               ExperimentInfo("max target - min target", float(model.interval))]
        experiment = Experiment(test_name, single_measurements, repeated_measurements)
        return experiment
    except MemoryError:
        print("\n\n--------------------")
        print("Experiment: ", test_name)
        print("n_neurons: ", model.get_n_neurons())
        print("lambda: ", model.get_lambda())
        print("Aborted, memory constraint violation")
        experiment = Experiment(test_name,[],[])
        return experiment
def test_letters(grid=True, iterations=5, n_neurons=2500, lr=0.001):
    print("testing dataset letters (ucml 59)")
    dataset = fetch_ucirepo(id=59)
    x = dataset.data.features
    y = dataset.data.targets
    x = pd.DataFrame(x).to_numpy()
    y = pd.DataFrame(y).to_numpy()
    y = one_hot_encode_targets(y)
    test_frac = 0.1
    x_ts, y_ts, x_tr, y_tr = slice_dataset_balanced(x, y, test_frac)
    y_tr = torch.from_numpy(y_tr.argmax(axis=1))
    y_ts = torch.from_numpy(y_ts.argmax(axis=1))
    x_tr = numpy_to_float_tensor(x_tr)
    x_ts = numpy_to_float_tensor(x_ts)
    # rf_model = RandomForest(x_tr, y_tr, x_ts, y_ts, 1000, 15)
    elm_model = CrossEntropyElm(x_tr, y_tr, x_ts, y_ts, n_neurons, lr)
    experiments = []
    if grid:
        experiments.append(test_grid_search(elm_model, iterations, "CE ELM for classification, letter recognition, grid search"))
    else:
        experiments.append(test_model_classification(elm_model, iterations, "CE ELM for classification, letter recognition, custom hyperparameters"))
    print_results(experiments)

def test_chess(grid=True, iterations=5, n_neurons=2500, lr=0.01):
    dataset = fetch_ucirepo(id=23)
    x = dataset.data.features
    y = dataset.data.targets
    x = pd.DataFrame(x).to_numpy()
    y = pd.DataFrame(y).to_numpy()
    for i in [0, 2, 4]:
        subst = np.array(list(map(chess_map, x[:, i])))  # mappo le lettere ai numeri
        x[:, i] = subst
    x = x.astype('float32')
    y = one_hot_encode_targets(y)
    test_frac = 0.1
    x_ts, y_ts, x_tr, y_tr = slice_dataset_balanced(x, y, test_frac)
    y_tr = torch.from_numpy(y_tr.argmax(axis=1))
    y_ts = torch.from_numpy(y_ts.argmax(axis=1))
    x_tr = numpy_to_float_tensor(x_tr)
    x_ts = numpy_to_float_tensor(x_ts)
    # rf_model = RandomForest(x_tr, y_tr, x_ts, y_ts, 1000, 30)
    elm_model = CrossEntropyElm(x_tr, y_tr, x_ts, y_ts, n_neurons, lr)
    experiments = []
    if grid:
        experiments.append(test_grid_search(elm_model, iterations, "CE ELM for classification, chess: king + rook, grid search"))
    else:
        experiments.append(test_model_classification(elm_model, iterations, "CE ELM for classification, chess: king + rook, custom hyperparameters"))
    print_results(experiments)

def test_adult(grid=True, iterations=5, n_neurons=1000, lr=0.001):
    dataset = fetch_ucirepo(id=2)
    x = dataset.data.features
    y = dataset.data.targets.iloc[:, 0]
    map = {'<=50K': 0, '<=50K.': 0, '>50K': 1, '>50K.': 1}
    y = y.map(map)

    original_features = x.shape[1]
    categorical_features = []
    for i in [1, 3, 5, 6, 7, 8, 9, 13]:
        categorical_features.append(dataset.data.features.columns[i])
    x = one_hot_encode_features(x, categorical_features)
    x = pd.DataFrame(x).to_numpy()
    y = pd.DataFrame(y).to_numpy()

    x = x.astype('float32')
    y = one_hot_encode_targets(y)
    test_frac = 0.1
    x_ts, y_ts, x_tr, y_tr = slice_dataset_balanced(x, y, test_frac)
    y_tr = torch.from_numpy(y_tr.argmax(axis=1))
    y_ts = torch.from_numpy(y_ts.argmax(axis=1))
    x_tr = numpy_to_float_tensor(x_tr)
    x_ts = numpy_to_float_tensor(x_ts)
    # rf_model = RandomForest(x_tr, y_tr, x_ts, y_ts, 300, 30)
    elm_model = CrossEntropyElm(x_tr, y_tr, x_ts, y_ts, n_neurons, lr, original_features)
    experiments = []
    if grid:
        experiments.append(test_grid_search(elm_model, iterations, "CE ELM for classification, adult, grid search"))
    else:
        experiments.append(test_model_classification(elm_model, iterations, "CE ELM for classification, adult, custom hyperparameters"))
    print_results(experiments)

def test_shuttle(grid=True, iterations=5, n_neurons=2500, lr=0.001):
    dataset = fetch_ucirepo(id=148)
    x = dataset.data.features
    y = dataset.data.targets
    x = pd.DataFrame(x).to_numpy()
    y = pd.DataFrame(y).to_numpy()

    y = one_hot_encode_targets(y)
    test_frac = 0.1
    x_ts, y_ts, x_tr, y_tr = slice_dataset_balanced(x, y, test_frac)
    y_tr = torch.from_numpy(y_tr.argmax(axis=1))
    y_ts = torch.from_numpy(y_ts.argmax(axis=1))
    x_tr = numpy_to_float_tensor(x_tr)
    x_ts = numpy_to_float_tensor(x_ts)
    elm_model = CrossEntropyElm(x_tr, y_tr, x_ts, y_ts, n_neurons, lr)
    experiments = []
    if grid:
        experiments.append(test_grid_search(elm_model, iterations, "CE ELM for classification, Statlog (Shuttle), grid search"))
    else:
        experiments.append(test_model_classification(elm_model, iterations, "CE ELM for classification, Statlog (Shuttle), custom hyperparameters"))
    print_results(experiments)


def test_gamma(grid=True, iterations=5, n_neurons=300, lr=0.001):
    dataset = fetch_ucirepo(id=159)
    x = dataset.data.features
    y = dataset.data.targets
    x = pd.DataFrame(x).to_numpy()
    y = pd.DataFrame(y).to_numpy()

    y = one_hot_encode_targets(y)
    test_frac = 0.1
    x_ts, y_ts, x_tr, y_tr = slice_dataset_balanced(x, y, test_frac)
    y_tr = torch.from_numpy(y_tr.argmax(axis=1))
    y_ts = torch.from_numpy(y_ts.argmax(axis=1))
    x_tr = numpy_to_float_tensor(x_tr)
    x_ts = numpy_to_float_tensor(x_ts)
    elm_model = CrossEntropyElm(x_tr, y_tr, x_ts, y_ts, n_neurons, lr)
    experiments = []
    if grid:
        experiments.append(test_grid_search(elm_model, iterations, "CE ELM for classification, MAGIC Gamma Telescope, grid search"))
    else:
        experiments.append(test_model_classification(elm_model, iterations, "CE ELM for classification, MAGIC Gamma Telescope, custom hyperparameters"))

    print_results(experiments)


def test_mnist(grid=True, iterations=5, n_neurons=2500, lr=0.01):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    test_dataset = datasets.MNIST('../data', train=False,
                              transform=transform)
    x_tr = train_dataset.data.unsqueeze(1).float() / 255.0  # shape: [60000, 1, 28, 28]
    x_tr = x_tr.view(x_tr.size(0), -1)  # shape: [60000, 784]
    y_tr = train_dataset.targets  # shape: [60000]

    x_ts = test_dataset.data.unsqueeze(1).float() / 255.0  # shape: [10000, 1, 28, 28]
    x_ts = x_ts.view(x_ts.size(0), -1)  # shape: [10000, 784]

    y_ts = test_dataset.targets  # shape: [10000]
    elm_model = CrossEntropyElm(x_tr, y_tr, x_ts, y_ts, n_neurons, lr)
    experiments = []
    if grid:
        experiments.append(test_grid_search(elm_model, iterations, "CE ELM for classification, MNIST, grid search"))
    else:
        experiments.append(test_model_classification(elm_model, iterations, "CE ELM for classification, MNIST, custom hyperparameters"))
    print_results(experiments)


def test_housing(grid=True, iterations=5, n_neurons=500, lmbda=0.01):
    df = pd.read_csv("datasets/housing.csv")
    df = df.dropna()
    y = torch.tensor(df['median_house_value'].values, dtype=torch.float32)
    x = df.drop(columns=['median_house_value'])
    original_features = x.shape[1]
    x = one_hot_encode_features(x, ["ocean_proximity"]).to_numpy(dtype=float)
    x = torch.tensor(x, dtype=torch.float32)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x = torch.tensor(x, dtype=torch.float32)
    x_tr, y_tr, x_ts, y_ts  = slice_dataset_regression(x, y, 0.1)
    elm = RegularizedElm(x_tr, y_tr, x_ts, y_ts, n_neurons, lmbda, original_features)
    experiments = []
    if grid:
        experiments.append(test_grid_search(elm, iterations, "Regularized ELM for regression, California Housing Prices, grid search"))
    else:
        experiments.append(test_model_regression(elm, iterations, "Regularized ELM for regression, California Housing Prices, custom hyperparameters"))
    print_results(experiments)

def test_superconductivity(grid=True, iterations=5, n_neurons=500, lmbda=0.01):
    df = pd.read_csv("datasets/superconductivity.csv")
    df = df.dropna()
    target_column = 'critical_temp'
    y = torch.tensor(df[target_column].values, dtype=torch.float32)
    x = df.drop(columns=[target_column])
    # x = pd.DataFrame.to_numpy(x.values, dtype=torch.float32)

    x = torch.tensor(x.values, dtype=torch.float32)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x = torch.tensor(x, dtype=torch.float32)
    x_tr, y_tr, x_ts, y_ts = slice_dataset_regression(x, y, 0.1)
    elm = RegularizedElm(x_tr, y_tr, x_ts, y_ts, n_neurons, lmbda)
    # test_grid_search(elm, 1, "elm superconductivity")
    experiments = []
    if grid:
        experiments.append(test_grid_search(elm, iterations, "Regularized ELM for regression, Superconductivity Data, grid search"))
    else:
        experiments.append(test_model_regression(elm, iterations, "Regularized ELM for regression, Superconductivity Data, custom hyperparameters"))

    print_results(experiments)



def test_nyctaxi(grid=True, iterations=5, n_neurons=100, lmbda=0.01):
    data = arff.loadarff('datasets/nyctaxi.arff')
    df = pd.DataFrame(data[0])
    # df = df.head(100000)
    df = df.head(30000)
    df = df.dropna()
    target_column = 'tipamount'
    y = torch.tensor(df[target_column].values, dtype=torch.float32)
    x = df.drop(columns=[target_column])
    # x = pd.DataFrame.to_numpy(x.values, dtype=torch.float32)

    x = torch.tensor(x.values, dtype=torch.float32)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x = torch.tensor(x, dtype=torch.float32)

    x_tr, y_tr, x_ts, y_ts = slice_dataset_regression(x, y, 0.1)
    elm = RegularizedElm(x_tr, y_tr, x_ts, y_ts, n_neurons, lmbda)
    # hpelm = HpElmRegression(x_tr, y_tr, x_ts, y_ts, 500, 'sigm')
    experiments = []
    if grid:
        experiments.append(test_grid_search(elm, iterations, "Regularized ELM for regression, NYC Taxi. grid search"))
    else:
        experiments.append(test_model_regression(elm, iterations, "Regularized ELM for regression, NYC Taxi, custom hyperparameters"))

    print_results(experiments)

def test_medical(grid=True, iterations=5, n_neurons=100, lmbda=0.01):
    data = arff.loadarff('datasets/medical.arff')
    df = pd.DataFrame(data[0])
    # df = df.head(100000)
    df = df.dropna()
    target_column = 'AverageTotalPayments'
    y = torch.tensor(df[target_column].values, dtype=torch.float32)
    x = df.drop(columns=[target_column])
    # x = pd.DataFrame.to_numpy(x.values, dtype=torch.float32)

    x = torch.tensor(x.values, dtype=torch.float32)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x = torch.tensor(x, dtype=torch.float32)

    x_tr, y_tr, x_ts, y_ts = slice_dataset_regression(x, y, 0.1)
    elm = RegularizedElm(x_tr, y_tr, x_ts, y_ts, n_neurons, lmbda)
    # hpelm = HpElmRegression(x_tr, y_tr, x_ts, y_ts, 500, 'sigm')
    experiments = []
    if grid:
        experiments.append(test_grid_search(elm, iterations, "Regularized ELM for regression, Medical Charges, grid search"))
    else:
        experiments.append(test_model_regression(elm, iterations, "Regularized ELM for regression, Medical Charges, custom hyperparameters"))
    print_results(experiments)



def test_blog(grid=True, iterations=5, n_neurons=1000, lmbda=0.01):
    df = pd.read_csv("datasets/blogData_train.csv")
    df.columns = [[("C"+str(i+1)) for i in range(df.shape[1])]]
    df = df.dropna()
    df = df.head(10000)
    target_column = 'C281'
    y = torch.tensor(df[target_column].values, dtype=torch.float32)
    x = df.drop(columns=[("C"+str(i)) for i in range(1,51)])
    x = torch.tensor(x.values, dtype=torch.float32)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x = torch.tensor(x, dtype=torch.float32)
    x_tr, y_tr, x_ts, y_ts  = slice_dataset_regression(x, y, 0.1)
    elm = RegularizedElm(x_tr, y_tr, x_ts, y_ts, n_neurons, lmbda)
    experiments = []
    if grid:
        experiments.append(test_grid_search(elm, iterations, "Regularized ELM for regression, Blog Feedback, grid search"))
    else:
        experiments.append(test_model_regression(elm, iterations, "Regularized ELM for regression, Blog Feedback, custom hyperparameters"))
    # experiments.append(test_model_regression(elm, 1, "elm housing"))
    print_results(experiments)
def main():
    # For each of the tests, grid=True runs exhaustive hyperparameter search, grid=False runs the test with default / custom values
    # Hyperparameters are n_neurons and lr for classification tasks and n_neurons and lmbda for regression tasks.
    # Default parameters for each function are grid=False and the hyperparameters that yield best performance.
    # final results are written on file results.txt

    #example of running task with custom hyperparameters
    #test_housing(grid=False, n_neurons= 125, lmbda =  0.15, iterations = 10)

    #example of running task with default hyperparameters
    #test_housing(grid=False)

    test_letters()
    test_chess()
    test_adult()
    test_shuttle()
    test_gamma()
    test_mnist()
    test_superconductivity()
    test_nyctaxi()
    test_medical()
    test_blog()


if __name__ == '__main__':
    main()
