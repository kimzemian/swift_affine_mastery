import toml
import pickle, dill


def load_config(path):
    with open(path) as f:
        config = toml.load(f)
    return config

def save_model(model, path):
    serialized_model = dill.dumps(model)
    with open(path, "wb") as file:
        pickle.dump(serialized_model, file)


def load_model(path):
    with open(path, "rb") as file:
        serialized_model = pickle.load(file)
    return dill.loads(serialized_model)