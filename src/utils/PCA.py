
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def find_critical_indices_map(crash_dict):
    X = crash_dict["AAPL"]["all_X_train"]
    X_flat = X.reshape(X.shape[0], -1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    final_vector = X_scaled[-1, :]
    X_centered = X_scaled - final_vector

    n_components = 32
    pca = PCA(n_components=n_components)
    pca.fit(X_centered)
    # pca.fit(X_scaled)

    return scaler, pca
