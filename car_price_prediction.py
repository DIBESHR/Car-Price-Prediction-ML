
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("/content/car_prediction_data.csv")


print("FIRST 5 ROWS:\n")

print(df.head())

print("\nDATASET INFORMATION:\n")

print(df.info())


print("\nMISSING VALUES:\n")

print(df.isnull().sum())


df = df.dropna()


original_df = df.copy()


car_encoder = LabelEncoder()

fuel_encoder = LabelEncoder()

seller_encoder = LabelEncoder()

transmission_encoder = LabelEncoder()


df['Car_Name'] = car_encoder.fit_transform(df['Car_Name'])

df['Fuel_Type'] = fuel_encoder.fit_transform(df['Fuel_Type'])

df['Seller_Type'] = seller_encoder.fit_transform(df['Seller_Type'])

df['Transmission'] = transmission_encoder.fit_transform(df['Transmission'])

print("\nAFTER ENCODING:\n")

print(df.head())


X = df.drop('Selling_Price', axis=1)

y = df['Selling_Price']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


model = LinearRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


result = pd.DataFrame({
    'Actual Price': y_test.values,
    'Predicted Price': y_pred
})

print("\nACTUAL vs PREDICTED:\n")

print(result.head())


mae = mean_absolute_error(y_test, y_pred)

print("\nMAE VALUE :", mae)


plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")

plt.ylabel("Predicted Price")

plt.title("Car Price Prediction")

plt.show()


print("\nAVAILABLE CAR NAMES:\n")

print(original_df['Car_Name'].unique())


search_car = input("\nEnter Car Name : ")


car_data = original_df[
    original_df['Car_Name'].str.lower() == search_car.lower()
]



if len(car_data) == 0:

    print("\nCar not found in dataset!")

else:


    car = car_data.iloc[0]

    print("\n================================")
    print("CAR DETAILS")
    print("================================")

    print("Car Name :", car['Car_Name'])

    print("Year :", car['Year'])

    print("Present Price :", car['Present_Price'])

    print("KM Driven :", car['Kms_Driven'])

    print("Fuel Type :", car['Fuel_Type'])

    print("Seller Type :", car['Seller_Type'])

    print("Transmission :", car['Transmission'])

    print("Owner :", car['Owner'])

    print("Actual Selling Price :",
          car['Selling_Price'],
          "Lakhs")


    car_name_encoded = car_encoder.transform(
        [car['Car_Name']]
    )[0]

    fuel_encoded = fuel_encoder.transform(
        [car['Fuel_Type']]
    )[0]

    seller_encoded = seller_encoder.transform(
        [car['Seller_Type']]
    )[0]

    transmission_encoded = transmission_encoder.transform(
        [car['Transmission']]
    )[0]


    new_car = pd.DataFrame([{
        'Car_Name': car_name_encoded,
        'Year': car['Year'],
        'Present_Price': car['Present_Price'],
        'Kms_Driven': car['Kms_Driven'],
        'Fuel_Type': fuel_encoded,
        'Seller_Type': seller_encoded,
        'Transmission': transmission_encoded,
        'Owner': car['Owner']
    }])


    new_car_scaled = scaler.transform(new_car)


    predicted_price = model.predict(new_car_scaled)


    print("\n================================")
    print("PREDICTION RESULT")
    print("================================")

    print("Predicted Selling Price :",
          round(predicted_price[0], 2),
          "Lakhs")