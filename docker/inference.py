import joblib
import pandas as pd
import sklearn
from preprocessing import DateTransformer, RegionMapper, OutlierClipper, RainTodayEncoder, RegionalImputer, CategoricalEncoder, ColumnAligner, OptimalThresholdClassifier
import logging
from sys import stdout
import warnings
warnings.simplefilter('ignore')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s %(levelname)s %(filename)s: %(message)s")
consoleHandler = logging.StreamHandler(stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

# Cargar pipeline
pipeline = joblib.load('pipeline.pkl')
logger.info('loaded pipeline')

# Cargar input
df_input = pd.read_csv('/files/input.csv')
logger.info('loaded input')

# Hacer predicciones
output = pipeline.predict(df_input)
logger.info('made predictions')

# Guardar predicciones en output.csv
pd.DataFrame(output, columns=['MEDV_predicted']).to_csv('/files/output.csv', index=False)
logger.info('saved output')