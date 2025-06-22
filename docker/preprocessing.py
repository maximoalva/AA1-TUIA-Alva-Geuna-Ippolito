import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

class DateTransformer(BaseEstimator, TransformerMixin):
    """Transforma la fecha en Year, Month y Season"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X["Date"] = pd.to_datetime(X["Date"])
        X["Year"] = X["Date"].dt.year
        X["Month"] = X["Date"].dt.month
        
        # Crear Season
        def asignar_estacion(mes):
            if mes in [12, 1, 2]:
                return 'Summer'
            elif mes in [3, 4, 5]:
                return 'Autumn'
            elif mes in [6, 7, 8]:
                return 'Winter'
            else:
                return 'Spring'
        
        X['Season'] = X['Month'].apply(asignar_estacion)
        X = X.drop(['Date', 'Month'], axis=1)
        return X

class RegionMapper(BaseEstimator, TransformerMixin):
    """Mapea Location a Region"""
    
    def __init__(self):
        self.location_to_region = {
            "Albury": 0, "BadgerysCreek": 9, "Cobar": 0, "CoffsHarbour": 7, "Moree": 7, "Newcastle": 9,
            "NorahHead": 9, "NorfolkIsland": 6, "Penrith": 9, "Richmond": 3, "Sydney": 9, "SydneyAirport": 9,
            "WaggaWagga": 0, "Williamtown": 9, "Wollongong": 9, "Canberra": 0, "Tuggeranong": 0, "MountGinini": 0,
            "Ballarat": 3, "Bendigo": 3, "Sale": 3, "MelbourneAirport": 3, "Melbourne": 3, "Mildura": 8,
            "Nhil": 11, "Portland": 11, "Watsonia": 3, "Dartmoor": 11, "Brisbane": 7, "Cairns": 4,
            "GoldCoast": 7, "Townsville": 4, "Adelaide": 8, "MountGambier": 11, "Nuriootpa": 8, "Woomera": 8,
            "Albany": 1, "Witchcliffe": 1, "PearceRAAF": 1, "PerthAirport": 1, "Perth": 1, "SalmonGums": 1,
            "Walpole": 1, "Hobart": 10, "Launceston": 10, "AliceSprings": 2, "Darwin": 5, "Katherine": 5, "Uluru": 2
        }
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['Region'] = X['Location'].map(self.location_to_region)
        X = X.drop('Location', axis=1)
        return X

class OutlierClipper(BaseEstimator, TransformerMixin):
    """Aplica clipping a outliers"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['Rainfall'] = np.where(X['Rainfall'] > 3.2, 3.2, X['Rainfall'])
        X['Evaporation'] = np.where(X['Evaporation'] > 21.8, 21.8, X['Evaporation'])
        X['WindSpeed9am'] = np.where(X['WindSpeed9am'] > 55, 55, X['WindSpeed9am'])
        X['WindSpeed3pm'] = np.where(X['WindSpeed3pm'] > 57, 57, X['WindSpeed3pm'])
        return X

class RainTodayEncoder(BaseEstimator, TransformerMixin):
    """Codifica RainToday de texto a booleano"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if 'RainToday' in X.columns and X['RainToday'].dtype == 'object':
            X["RainToday"] = X["RainToday"].map({"No": False, "Yes": True})
        return X

class RegionalImputer(BaseEstimator, TransformerMixin):
    """Imputa valores faltantes usando estadísticas por región precomputadas"""
    
    def __init__(self, median_values=None, mode_values=None):
        # Valores precomputados durante el entrenamiento
        self.median_values = median_values or {}
        self.mode_values = mode_values or {}
        self.knn_imputers = {}
        
        # Variables a imputar por mediana
        self.median_vars = ['MinTemp', 'MaxTemp', 'Rainfall', 'Sunshine', 'WindGustSpeed', 
                           'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 
                           'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']
        
        # Variables a imputar por moda
        self.mode_vars = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
        
        # Variables a imputar con KNN
        self.knn_vars = ['Evaporation', 'Cloud9am', 'Cloud3pm']
    
    def fit(self, X, y=None):
        # Si no se proporcionaron valores precomputados, calcularlos
        if not self.median_values:
            for var in self.median_vars:
                if var in X.columns:
                    self.median_values[var] = X.groupby('Region')[var].median().to_dict()
        
        if not self.mode_values:
            for var in self.mode_vars:
                if var in X.columns:
                    self.mode_values[var] = X.groupby('Region')[var].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]).to_dict()
        
        # Entrenar KNN imputers por región (esto siempre se hace durante fit)
        for region in X['Region'].unique():
            if pd.notna(region):
                region_data = X[X['Region'] == region]
                if len(region_data) > 0:
                    imputer = KNNImputer(n_neighbors=5)
                    available_knn_vars = [var for var in self.knn_vars if var in region_data.columns]
                    if available_knn_vars:
                        imputer.fit(region_data[available_knn_vars])
                        self.knn_imputers[region] = imputer
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Imputar por mediana
        for var in self.median_vars:
            if var in X.columns and var in self.median_values:
                for region, median_val in self.median_values[var].items():
                    mask = (X['Region'] == region) & (X[var].isna())
                    X.loc[mask, var] = median_val
        
        # Imputar por moda
        for var in self.mode_vars:
            if var in X.columns and var in self.mode_values:
                for region, mode_val in self.mode_values[var].items():
                    mask = (X['Region'] == region) & (X[var].isna())
                    X.loc[mask, var] = mode_val
        
        # Imputar con KNN por región
        for region in X['Region'].unique():
            if pd.notna(region) and region in self.knn_imputers:
                region_mask = X['Region'] == region
                region_data = X[region_mask]
                available_knn_vars = [var for var in self.knn_vars if var in X.columns]
                
                if len(region_data) > 0 and available_knn_vars:
                    imputed_data = self.knn_imputers[region].transform(region_data[available_knn_vars])
                    X.loc[region_mask, available_knn_vars] = imputed_data
        
        return X

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Crea variables dummy para variables categóricas"""
    
    def __init__(self):
        self.categorical_vars = ['Region', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'Season']
        self.dummy_columns = {}
    
    def fit(self, X, y=None):
        # Guardar las columnas dummy que se crearán
        for var in self.categorical_vars:
            if var in X.columns:
                dummies = pd.get_dummies(X[var], prefix=var, drop_first=True)
                self.dummy_columns[var] = dummies.columns.tolist()
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for var in self.categorical_vars:
            if var in X.columns:
                # Crear dummies
                dummies = pd.get_dummies(X[var], prefix=var, drop_first=True)
                
                # Asegurar que todas las columnas esperadas estén presentes
                for col in self.dummy_columns[var]:
                    if col not in dummies.columns:
                        dummies[col] = 0
                
                # Mantener solo las columnas esperadas
                dummies = dummies[self.dummy_columns[var]]
                
                # Concatenar y eliminar variable original
                X = pd.concat([X.drop(columns=var), dummies], axis=1)
        
        return X

class ColumnAligner(BaseEstimator, TransformerMixin):
    """Asegura que las columnas finales coincidan con las del entrenamiento"""
    
    def __init__(self):
        self.feature_names = None
    
    def fit(self, X, y=None):
        self.feature_names = X.columns.tolist()
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Agregar columnas faltantes con valor 0
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        
        # Mantener solo las columnas esperadas en el orden correcto
        X = X[self.feature_names]
        
        return X

class OptimalThresholdClassifier(BaseEstimator, TransformerMixin):
    """Wrapper que calcula y aplica el umbral óptimo usando Youden's J"""
    
    def __init__(self, estimator=None):
        self.estimator = estimator or LogisticRegression(
            solver='newton-cg', penalty=None, max_iter=408, 
            class_weight='balanced', random_state=42
        )
        self.optimal_threshold = 0.5  # Default threshold
        self.roc_auc = None
    
    def fit(self, X, y):
        from sklearn.metrics import roc_curve, auc
        
        # Entrenar el modelo base
        self.estimator.fit(X, y)
        
        # Calcular probabilidades
        y_proba = self.estimator.predict_proba(X)[:, 1]
        
        # Calcular ROC curve
        fpr, tpr, thresholds = roc_curve(y, y_proba)
        self.roc_auc = auc(fpr, tpr)
        
        # Eliminar el primer valor de thresholds (inf) y sus correspondientes tpr, fpr
        if len(thresholds) > 1:
            thresholds_clean = thresholds[1:]
            tpr_clean = tpr[1:]
            fpr_clean = fpr[1:]
            
            # Calcular Youden's J
            J = tpr_clean - fpr_clean
            ix = np.argmax(J)
            self.optimal_threshold = thresholds_clean[ix]
        
        print(f"Mejor umbral según Youden's J: {self.optimal_threshold:.6f}")
        print(f"ROC AUC: {self.roc_auc:.4f}")
        
        return self
    
    def predict(self, X, use_optimal_threshold=True):
        """Predice usando el umbral óptimo por defecto"""
        proba = self.predict_proba(X)[:, 1]
        threshold = self.optimal_threshold if use_optimal_threshold else 0.5
        return (proba >= threshold).astype(int)
    
    def predict_proba(self, X):
        """Retorna las probabilidades del modelo base"""
        return self.estimator.predict_proba(X)
    
    def get_params(self, deep=True):
        """Get parameters for GridSearch compatibility"""
        params = {'estimator': self.estimator, 'optimal_threshold': self.optimal_threshold}
        if deep and hasattr(self.estimator, 'get_params'):
            params.update({'estimator__' + k: v for k, v in self.estimator.get_params().items()})
        return params
    
    def set_params(self, **params):
        """Set parameters for GridSearch compatibility"""
        estimator_params = {}
        for key, value in params.items():
            if key.startswith('estimator__'):
                estimator_params[key[11:]] = value
            elif key == 'estimator':
                self.estimator = value
            elif key == 'optimal_threshold':
                self.optimal_threshold = value
        
        if estimator_params and hasattr(self.estimator, 'set_params'):
            self.estimator.set_params(**estimator_params)
        
        return self
