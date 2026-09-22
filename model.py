"""
South African Spousal Ethnicity Prediction Model
------------------------------------------------
Predicts the probability distribution of which language group 
an individual is most likely to marry into based on Census data.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_and_pair_spouses(file_path):
    print("Loading census data (this may take a few minutes)...")
    df = pd.read_csv(file_path, low_memory=False, encoding='latin1')
    df.columns = df.columns.str.strip()
    
    print("Filtering data...")
    df['P04_AGE'] = pd.to_numeric(df['P04_AGE'], errors='coerce')
    df = df[df['P04_AGE'] >= 18]
    
    df['P05_RELATION'] = df['P05_RELATION'].astype(str).str.strip()
    
    heads = df[df['P05_RELATION'] == 'Head/acting head'].set_index('QID')
    spouses = df[df['P05_RELATION'] == 'Husband/wife/partner'].set_index('QID')
    
    couples = heads.join(spouses, lsuffix='_head', rsuffix='_spouse', how='inner')
    print(f"Successfully paired {len(couples)} couples.")

    if len(couples) == 0:
        raise ValueError("No couples found!")

    # Reshape for Individual-Level Prediction (Target is now the PARTNER'S language)
    features = ['P02_SEX', 'P04_AGE', 'P08_LANGUAGE', 'DERP_USUALRESPROV']
    
    # Process Heads (Target is Spouse's Language)
    df_heads = couples[[f"{f}_head" for f in features]].copy()
    df_heads.columns = ['Sex', 'Age', 'Language', 'Province']
    df_heads['Spouse_Language'] = couples['P08_LANGUAGE_spouse'].values
    
    # Process Spouses (Target is Head's Language)
    df_spouses = couples[[f"{f}_spouse" for f in features]].copy()
    df_spouses.columns = ['Sex', 'Age', 'Language', 'Province']
    df_spouses['Spouse_Language'] = couples['P08_LANGUAGE_head'].values
    
    model_data = pd.concat([df_heads, df_spouses], ignore_index=True)
    model_data = model_data.dropna()
    print(f"Final ML dataset size: {len(model_data)} individuals.")
    
    return model_data

def train_prediction_model(model_data):
    print("\nPreparing multiclass features and target...")
    X = model_data[['Sex', 'Age', 'Language', 'Province']]
    y = model_data['Spouse_Language'] # The target is now the exact language

    X_encoded = pd.get_dummies(X, columns=['Sex', 'Language', 'Province'], drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    print("Training Random Forest Multiclass model...")
    # Removed class_weight='balanced' so probabilities reflect true statistical likelihoods in reality
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    print("\nModel Evaluation:")
    y_pred = rf_model.predict(X_test)
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    
    return rf_model, X_encoded.columns

def predict_spouse_ethnicity(sex, age, language, province, trained_model, columns):
    person = pd.DataFrame({'Sex': [sex], 'Age': [age], 'Language': [language], 'Province': [province]})
    person_encoded = pd.get_dummies(person)
    person_encoded = person_encoded.reindex(columns=columns, fill_value=0)
    
    # Get probabilities for all languages
    probabilities = trained_model.predict_proba(person_encoded)[0]
    classes = trained_model.classes_
    
    # Zip them together, sort by highest probability, and print
    results = pd.DataFrame({
        'Spouse_Ethnicity': classes,
        'Probability': probabilities * 100
    }).sort_values(by='Probability', ascending=False)
    
    print(f"\n--- MATCH PROBABILITIES FOR: {sex}, {age} yrs old, {language} speaker in {province} ---")
    for index, row in results.head(5).iterrows(): # Show top 5 matches
        print(f"{row['Spouse_Ethnicity']}: {row['Probability']:.2f}%")

if __name__ == "__main__":
    FILE_NAME = "census.csv" 
    
    try:
        processed_data = load_and_pair_spouses(FILE_NAME)
        model, model_columns = train_prediction_model(processed_data)
        
        # Test 1: A 30-year-old male isiZulu speaker in Gauteng
        predict_spouse_ethnicity('Male', 30, 'isiZulu', 'Gauteng', model, model_columns)
        
        # Test 2: A 25-year-old female Afrikaans speaker in Western Cape
        predict_spouse_ethnicity('Female', 25, 'Afrikaans', 'Western Cape', model, model_columns)
        
    except FileNotFoundError:
        print(f"Error: Could not find '{FILE_NAME}'. Make sure it is in the same folder as this script.")