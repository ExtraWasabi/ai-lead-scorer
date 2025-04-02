import pandas as pd
import numpy as np

# Beispielhafte Funktion für die Datenaufbereitung
def preprocess_data(data):
    # Dummy-Daten (kann später durch echte Daten ersetzt werden)
    # Feature Scaling, Normalisierung oder Transformationen nach Bedarf hinzufügen.
    data = data.copy()
    data['website_visits'] = np.log1p(data['website_visits'])  # Beispiel für Transformation
    return data
