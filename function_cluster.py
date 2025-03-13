import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

class KMeansClustering:
    def __init__(self, max_clusters=10, random_state=42, n_clusters=None):
        """
        Clase para realizar clustering usando KMeans con selección automática
        del número óptimo de clusters utilizando KElbowVisualizer.

        Parámetros:
        max_clusters: int
            Número máximo de clusters a evaluar.
        random_state: int
            Semilla para la generación de números aleatorios.
        n_clusters: int or None
            Número de clusters para el modelo KMeans. Si es None, se seleccionará automáticamente.
        """
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.n_clusters = n_clusters
        self.optimal_clusters = None
        self.kmeans = None

    def fit(self, X):
        """
        Ajusta el modelo KMeans a los datos. Si n_clusters es None, selecciona automáticamente
        el número óptimo de clusters utilizando KElbowVisualizer.

        Parámetros:
        X: ndarray
            Datos de entrada.
        """
        if self.n_clusters is None:
            # Usar KElbowVisualizer para determinar el número óptimo de clusters
            model = KMeans(random_state=self.random_state,
                           init='k-means++', 
                           n_init='auto',
                           max_iter=300,
                           tol=0.0001)
            visualizer = KElbowVisualizer(model, k=(2, self.max_clusters))
            visualizer.fit(X)
            plt.close()
            self.optimal_clusters = visualizer.elbow_value_
        else:
            self.optimal_clusters = self.n_clusters
        
        # Ajustar el modelo KMeans con el número óptimo de clusters o el especificado
        self.kmeans = KMeans(n_clusters=self.optimal_clusters,
                             random_state=self.random_state,                         
                             init='k-means++', 
                             n_init='auto', 
                             max_iter=300, 
                             tol=0.0001)
        self.kmeans.fit(X)

    def predict(self, X):
        """
        Predice las etiquetas de los clusters para los datos de entrada.

        Parámetros:
        X: ndarray
            Datos de entrada.

        Retorna:
        ndarray
            Etiquetas de los clusters.
        """
        if self.kmeans is None:
            raise ValueError("El modelo debe ser ajustado antes de hacer predicciones. Llama a fit(X) primero.")
        return self.kmeans.predict(X)

    def fit_predict(self, X):
        """
        Ajusta el modelo y predice las etiquetas de los clusters para los datos de entrada.

        Parámetros:
        X: ndarray
            Datos de entrada.

        Retorna:
        ndarray
            Etiquetas de los clusters.
        """
        self.fit(X)
        return self.predict(X)

    def get_optimal_clusters(self):
        """
        Retorna el número óptimo de clusters seleccionado.

        Retorna:
        int
            Número óptimo de clusters.
        """
        if self.optimal_clusters is None:
            raise ValueError("El modelo debe ser ajustado antes de obtener el número óptimo de clusters.")
        return self.optimal_clusters


class GMMClustering:
    def __init__(self, max_components=10, criterion='bic', n_components=None, random_state=42):
        """
        Clase para realizar clustering usando Gaussian Mixture Models con selección automática
        del número óptimo de componentes.

        Parámetros:
        max_components: int
            Número máximo de componentes a evaluar.
        criterion: str
            Criterio a utilizar para seleccionar el modelo ('aic' o 'bic').
        random_state: int
            Semilla para la generación de números aleatorios.
        """
        self.max_components = max_components
        self.criterion = criterion
        self.n_components = n_components
        self.random_state = random_state
        self.optimal_components = None
        self.gmm = None

    def fit(self, X):
        """
        Ajusta el modelo GMM a los datos con el número óptimo de componentes.

        Parámetros:
        X: ndarray
            Datos de entrada.
        """
        if self.n_components is None:
            criterions = []
            
            # Evaluar GMM para diferentes números de componentes
            for n_components in range(1, self.max_components + 1):
                gmm = GaussianMixture(n_components=n_components, random_state=self.random_state)
                gmm.fit(X)
                
                if self.criterion == 'aic':
                    criterions.append(gmm.aic(X))
                elif self.criterion == 'bic':
                    criterions.append(gmm.bic(X))
                else:
                    raise ValueError("El criterio debe ser 'aic' o 'bic'")
            
            # Seleccionar el número óptimo de componentes
            self.optimal_components = np.argmin(criterions) + 1
        else:
            self.optimal_components = self.n_components

        # Ajustar el modelo GMM con el número óptimo de componentes
        self.gmm = GaussianMixture(n_components=self.optimal_components, random_state=self.random_state)
        self.gmm.fit(X)

    def predict(self, X):
        """
        Predice las etiquetas de los clusters para los datos de entrada.

        Parámetros:
        X: ndarray
            Datos de entrada.

        Retorna:
        ndarray
            Etiquetas de los clusters.
        """
        if self.gmm is None:
            raise ValueError("El modelo debe ser ajustado antes de hacer predicciones. Llama a fit(X) primero.")
        return self.gmm.predict(X)

    def fit_predict(self, X):
        """
        Ajusta el modelo y predice las etiquetas de los clusters para los datos de entrada.

        Parámetros:
        X: ndarray
            Datos de entrada.

        Retorna:
        ndarray
            Etiquetas de los clusters.
        """
        self.fit(X)
        return self.predict(X)

    def get_optimal_components(self):
        """
        Retorna el número óptimo de componentes seleccionado.

        Retorna:
        int
            Número óptimo de componentes.
        """
        if self.optimal_components is None:
            raise ValueError("El modelo debe ser ajustado antes de obtener el número óptimo de componentes.")
        return self.optimal_components


class AgglomerativeClusteringOptimal:
    def __init__(self, max_clusters=10, n_clusters=None):
        """
        Clase para realizar clustering jerárquico aglomerativo con selección automática
        del número óptimo de clusters usando el método del codo.

        Parámetros:
        max_clusters: int
            Número máximo de clusters a evaluar.
        n_clusters: int or None
            Número de clusters. Si es None, se seleccionará automáticamente.
        """
        self.max_clusters = max_clusters
        self.n_clusters = n_clusters
        self.optimal_clusters = None
        self.model = None

    def calculate_wcss(self, data):
        """
        Calcula la suma de cuadrados dentro del cluster (WCSS) para diferentes números de clusters.

        Parámetros:
        data: ndarray
            Datos de entrada.

        Retorna:
        list: WCSS para cada número de clusters.
        """
        wcss = []
        for n_clusters in range(1, self.max_clusters + 1):
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clustering.fit(data).labels_
            centroids = [
                data[cluster_labels == i].mean(axis=0) for i in range(n_clusters)
            ]
            wcss.append(
                sum(
                    np.linalg.norm(data[cluster_labels == i] - centroids[i]) ** 2
                    for i in range(n_clusters)
                )
            )
        return wcss

    def find_optimal_clusters(self, data):
        """
        Encuentra el número óptimo de clusters usando el método del codo.

        Parámetros:
        data: ndarray
            Datos de entrada.

        Retorna:
        int: Número óptimo de clusters.
        """
        wcss = self.calculate_wcss(data)
        first_derivative = np.diff(wcss)
        second_derivative = np.diff(first_derivative)
        elbow_point = np.argmin(second_derivative) + 2
        return elbow_point

    def fit(self, X):
        """
        Ajusta el modelo a los datos.

        Parámetros:
        X: ndarray
            Datos de entrada.
        """
        if self.n_clusters is None:
            self.optimal_clusters = self.find_optimal_clusters(X)
        else:
            self.optimal_clusters = self.n_clusters
        
        self.model = AgglomerativeClustering(n_clusters=self.optimal_clusters)
        self.model.fit(X)

    def predict(self, X):
        """
        Predice las etiquetas de los clusters para los datos de entrada.

        Parámetros:
        X: ndarray
            Datos de entrada.

        Retorna:
        ndarray: Etiquetas de los clusters.
        """
        if self.model is None:
            raise ValueError("El modelo debe ser ajustado antes de hacer predicciones. Llama a fit(X) primero.")
        return self.model.fit(X)

    def fit_predict(self, X):
        """
        Ajusta el modelo y predice las etiquetas de los clusters para los datos de entrada.

        Parámetros:
        X: ndarray
            Datos de entrada.

        Retorna:
        ndarray: Etiquetas de los clusters.
        """
        self.fit(X)
        return self.predict(X).labels_

    def get_optimal_clusters(self):
        """
        Retorna el número óptimo de clusters seleccionado.

        Retorna:
        int: Número óptimo de clusters.
        """
        if self.optimal_clusters is None:
            raise ValueError("El modelo debe ser ajustado antes de obtener el número óptimo de clusters.")
        return self.optimal_clusters