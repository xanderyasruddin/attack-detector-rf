import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# converting csv data into a dataframe
df = pd.read_csv("cybersecurity_attacks.csv")

# handling null values
df['Alerts/Warnings'] = df["Alerts/Warnings"].apply(lambda x: "Yes" if x == "Alert Triggered" else "No")
df['Malware Indicators'] = df["Malware Indicators"].apply(lambda x: "Yes" if x == "IoC Detected" else "No")
df['IDS/IPS Alerts'] = df['IDS/IPS Alerts'].apply(lambda x: "No Data" if pd.isna(x) else x)
df['Firewall Logs'] = df['Firewall Logs'].apply(lambda x: "No Data" if pd.isna(x) else x)
df['Proxy Information'] = df['Proxy Information'].apply(lambda x: "No Data" if pd.isna(x) else x)

# creating more columns for timestamping
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

# categorising the numerical categories
df['Destination Port'] = df['Destination Port'].astype(str)
df['Source Port'] = df['Source Port'].astype(str)

# reduce unique values for get_dummies
def categorise_port(port):
    try:
        port = int(port)
    except:
        return "Unknown"
    if port < 1024:
        return "Well-Known"
    elif port < 49152:
        return "Registered"
    else:
        return "Dynamic"
    
df['Source Port Category'] = df['Source Port'].apply(categorise_port)
df['Destination Port Category'] = df['Destination Port'].apply(categorise_port)
df['Proxy Information'] = df['Proxy Information'].apply(lambda x: 'Present' if x != "No Data" else "Missing")

# normalising the numerical data
cols_normalise = ['Anomaly Scores', 'Packet Length']
scaler = MinMaxScaler()
df[cols_normalise] = scaler.fit_transform(df[cols_normalise])

# one-hot encoding categorical columns
categorical_cols = ['Protocol', 'Traffic Type', 'Packet Type',
                    'Severity Level', 'Action Taken']

numerical_cols = ['Anomaly Scores', 'Packet Length']
feature_cols = numerical_cols + categorical_cols
feature_cols = ['Anomaly Scores', 'Packet Length'] + categorical_cols
df_features = df[feature_cols]

# initialising input data and target
X = pd.get_dummies(df_features, columns=categorical_cols) 
y = df['Attack Type']

# training and testing model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=47,
    n_jobs=-1,
    class_weight='balanced',
    )
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy:.4f}")

print("----Classification Report----")
print(classification_report(
    y_test, 
    y_pred,
    zero_division=0))