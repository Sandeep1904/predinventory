import streamlit as st
import pandas as pd
from transformer import Transformer
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import numpy as np



st.title("Let's predict what food is ordered and how much")
giturl = "https://github.com/Sandeep1904/predinventory"
st.write("#### ‼️ Naive implementation of a transformer doesn't perform really well, hence currently working on some training enhancements. You can checkout the code on [github]({giturl}) till then.")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_data
def createdfd(df):
    dfd = df.drop_duplicates(subset=['Item Name'])
    dfd.drop(['Order Number', 'Order Date', 'Quantity', 'Total products'], axis=1, inplace=True)
    dfd.reset_index(drop=True, inplace=True)
    return dfd

df = load_data('restaurant-1-orders.csv')
st.write(df.head(5))

st.write("We also create subset of this df to use later as our dictionary")
dfd = createdfd(df)
st.write(dfd.head(5))

modelurl = "https://github.com/Sandeep1904/predinventory/blob/main/model.ipynb"
st.write(f"For other data processing steps, please refer to the [jupyter notebook]({modelurl}).")
st.write("As the model is already trained, we are going to use it directly in this app for predictions.")


START_TOKEN = ' '
END_TOKEN = '.'
PADDING_TOKEN = '|'

numbers = ['1','2','3','4','5','6','7','8','9','0','-']
index_to_time = {}

for i, n in enumerate(numbers):
    index_to_time[i] = n

index_to_obj = dfd.loc[:, "Item Name"].to_dict()




lt = len(index_to_time.keys())
lo = len(index_to_obj.keys())

index_to_obj[lo] = START_TOKEN
index_to_obj[lo + 1] = END_TOKEN
index_to_obj[lo + 2] = PADDING_TOKEN
index_to_obj[lo + 3] = '0'

index_to_time[lt] = START_TOKEN
index_to_time[lt + 1] = END_TOKEN
index_to_time[lt + 2] = PADDING_TOKEN


time_to_index = {v: k for k, v in index_to_time.items()}
obj_to_index = {v: k for k, v in index_to_obj.items()}


d_model = 64
ffn_hidden = 256
num_heads = 8
drop_prob = 0.1
num_layers = 12
max_sequence_length = 30
obj_size = len(obj_to_index)

@st.cache_resource
def load_transformer():
    transformer = Transformer(d_model,
                          ffn_hidden,
                          num_heads,
                          drop_prob,
                          num_layers,
                          max_sequence_length,
                          obj_size,
                          time_to_index,
                          obj_to_index,
                          START_TOKEN,
                          END_TOKEN,
                          PADDING_TOKEN)
    transformer.load_state_dict(torch.load('trained_transformer_model.pth'))

    return transformer

transformer = load_transformer()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


NEG_INFTY = -1e9

def create_masks(time_batch, item_batch):
    num_sentences = len(time_batch)
    # max_sequence_length = max(len(items) for items in item_batch) + 1
    
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length], True, device=device) # device should be same
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)

    encoder_padding_mask = torch.zeros([num_sentences, max_sequence_length, max_sequence_length], dtype=torch.bool, device=device) # device and dtype should be same
    decoder_padding_mask_self_attention = torch.zeros([num_sentences, max_sequence_length, max_sequence_length], dtype=torch.bool, device=device)
    decoder_padding_mask_cross_attention = torch.zeros([num_sentences, max_sequence_length, max_sequence_length], dtype=torch.bool, device=device)

    for idx in range(num_sentences):
        time_len = len(time_batch[idx])
        item_len = len(item_batch[idx])
        print(f"{max_sequence_length} {item_len} {time_len}")
        time_to_padding_mask = torch.arange(time_len + 1, max_sequence_length, device=device) # device should be same
        item_to_padding_mask = torch.arange(item_len + 1, max_sequence_length, device=device)

        encoder_padding_mask[idx, :, time_to_padding_mask] = True
        encoder_padding_mask[idx, time_to_padding_mask, :] = True

        decoder_padding_mask_self_attention[idx, :, item_to_padding_mask] = True
        decoder_padding_mask_self_attention[idx, item_to_padding_mask, :] = True

        decoder_padding_mask_cross_attention[idx, :, time_to_padding_mask] = True
        decoder_padding_mask_cross_attention[idx, time_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0.0)
    decoder_self_attention_mask = torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0.0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0.0)

    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask

##### -----------------------------------------------------------------------

regurl = "https://foodregression.streamlit.app/"
st.subheader("First we will implement XGBoost to predict the volumes in that week")
st.write("For a detailed analysis on regression, please refer to this [app]({regurl})")


st.header("Data Preprocessing and Feature Engineering")
st.write("Before we can build a model, we need to clean and prepare our data. This involves removing irrelevant columns, converting data types, and handling missing values.")

@st.cache_data
def preprocess(df):
    df.drop(columns=['Order Number', 'Total products', 'Item Name', 'Product Price'], inplace=True, errors='ignore') # added errors='ignore'
    df["Order Date"] = pd.to_datetime(df["Order Date"], format='%d/%m/%Y %H:%M', errors='coerce') # added errors='coerce'
    df.sort_values(by='Order Date', inplace=True)
    df.dropna(subset=['Order Date'], inplace=True) # drop rows with NaT from incorrect date format.
    df.drop(df.head(10).index, inplace=True, errors='ignore') # added errors='ignore'
    df.reset_index(drop=True, inplace=True)
    df = df.resample('D', on='Order Date').sum().reset_index()
    return df

df = preprocess(df)

st.write("We've grouped the data by day, summing the quantities ordered.  Here's the transformed data:")
st.dataframe(df.head(7))

st.write("### Let's visualize the order quantities over time to understand the patterns and identify any trends or seasonality.")

fig, ax = plt.subplots(figsize=(20, 5))  # Create figure and axes objects
sns.lineplot(df, x='Order Date', y='Quantity', ax=ax)  # Plot on the axes
plt.title("Daily Order Quantities") # Add title to the plot
st.pyplot(fig)  # Display the plot using st.pyplot

st.write("#### As we can see, there's significant variance in the data, with an apparent upward trend. Let's examine the distribution of order quantities and identify potential outliers.")

fig, ax = plt.subplots(figsize=(20, 8))
sns.boxplot(df['Quantity'], ax=ax)
plt.title("Box Plot of Order Quantities")
st.pyplot(fig)

st.write("#### 👆 The box plot reveals several outliers.  Let's remove these to improve the performance of our models.")

@st.cache_data
def remove_outliers(df):
    Q1 = df['Quantity'].quantile(0.25)
    Q3 = df['Quantity'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['Quantity'] >= lower_bound) & (df['Quantity'] <= upper_bound)]
    return df

df = remove_outliers(df)

st.write("#### 👇 After removing outliers, the distribution of order quantities becomes clearer:")

fig, ax = plt.subplots(figsize=(20, 8))
sns.kdeplot(data=df, x='Quantity', ax=ax)
plt.title("Distribution of Order Quantities (Outliers Removed)")
st.pyplot(fig)

st.write("First, we'll create new features from the date and time information, as well as lagged and rolling mean features.")

@st.cache_data
def prepare_data_for_xgboost(df):
    df['year'] = df['Order Date'].dt.year
    df['month'] = df['Order Date'].dt.month
    df['day'] = df['Order Date'].dt.day
    df['day_of_week'] = df['Order Date'].dt.dayofweek
    df['is_weekend'] = (df['Order Date'].dt.dayofweek >= 5).astype(int)
    df['hour'] = df['Order Date'].dt.hour  # Hour of the day. It's 0 because the data is aggregated daily.

    df['Quantity_lag_1'] = df['Quantity'].shift(1)
    df['Quantity_rolling_mean_7'] = df['Quantity'].rolling(window=7).mean()
    df = df.fillna(0)  # Handle missing values

    features = ['year', 'month', 'day', 'day_of_week', 'is_weekend', 'hour', 'Quantity_lag_1', 'Quantity_rolling_mean_7']
    X = df[features]
    y = df['Quantity']
    return X, y

X, y = prepare_data_for_xgboost(df)

st.write("Here's a sample of our engineered features:")
st.dataframe(X.head())

st.subheader("Training the XGBoost Model")
st.write("We'll split the data into training and testing sets and train our XGBoost model. We'll then evaluate its performance on the test set.")

@st.cache_data  # Cache the trained model
def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # Important: shuffle=False for time series
    params = {
        "n_estimators": 500, # increased estimators
        "max_depth": 4,
        "min_samples_split": 5,
        "learning_rate": 0.01,
        "loss": "squared_error",
        "random_state": 42 # added for reproducibility
    }
    model = ensemble.GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    st.write(f"Training MAE: {mae_train}")
    st.write(f"Training MSE: {mse_train}")

    return model, X_test, y_test

model, X_test, y_test = train_xgboost(X, y)



y_pred_test = model.predict(X_test)

mae_test = mean_absolute_error(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
st.write(f"Test MAE: {mae_test}")
st.write(f"Test MSE: {mse_test}")

st.subheader("XGBoost Model Predictions")
st.write("Let's visualize the actual vs. predicted order quantities from our XGBoost model.")

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual')  # Use y_test.index for x-axis
plt.plot(X_test.index, y_pred_test, label='XGBoost Predictions', linestyle='--') # Use X_test.index for x-axis
plt.title('XGBoost Model - Actual vs Predicted (Test Data)')
plt.legend()
st.pyplot(fig)

##### -----------------------------------------------------------------------

# Take user input for data and assign it to time_sentence
# User input in Streamlit

_, inputdf = train_test_split(df, shuffle=False, test_size=0.2)
inputdf = pd.DataFrame(inputdf)
time_sentence_input = st.selectbox("Select a date:", inputdf['Order Date'].unique())
time_for_transformer = str(time_sentence_input.strftime('%Y-%m-%d'))


transformer.eval()
transformer.to(device)
st.button("Get Predictions")
item_sentence = [[START_TOKEN]]
time_sentence = (time_for_transformer,)
encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(time_sentence, item_sentence)

# input_ids = transformer.decoder.sentence_embedding.batch_tokenize(item_sentence, start_token=False, end_token=False)
# input_ids = input_ids.to(device)

predictions = transformer(time_sentence,
                            item_sentence,
                            encoder_self_attention_mask.to(device),
                            decoder_self_attention_mask.to(device),
                            decoder_cross_attention_mask.to(device),
                            enc_start_token=False,
                            enc_end_token=False,
                            dec_start_token=True,  # Important: Start token is needed
                            dec_end_token=False)

probability_distribution = torch.softmax(predictions[0][-1], dim=-1) # Get the probabilities for the LAST token

# Create a dictionary mapping items to probabilities
item_probabilities = {}
for item, index in obj_to_index.items():
    if index < probability_distribution.shape[0]: # Check if index is within distribution range
        item_probabilities[item] = probability_distribution[index].item()
    else:
        print(f"Warning: Index {index} is out of range of probability distribution.")

# If you want to sort the items by probability:
sorted_item_probabilities = pd.DataFrame(sorted(item_probabilities.items(), key=lambda item: item[1], reverse=True), columns=['Dishes', 'Probabilites'])
st.write(f"For the given time period: {time_sentence_input}, following are the most likely dishes to be ordered:", sorted_item_probabilities)


@st.cache_data
def create_date_features(date_str):
    """Creates features from the date string, matching training data."""
    try:
        order_date = pd.to_datetime(date_str)  # Use pandas to convert to datetime
        df_date = pd.DataFrame({'Order Date': [order_date]}) # Create a dataframe with the date
        df_date['year'] = df_date['Order Date'].dt.year
        df_date['month'] = df_date['Order Date'].dt.month
        df_date['day'] = df_date['Order Date'].dt.day
        df_date['day_of_week'] = df_date['Order Date'].dt.dayofweek
        df_date['is_weekend'] = (df_date['Order Date'].dt.dayofweek >= 5).astype(int)
        df_date['hour'] = df_date['Order Date'].dt.hour


        # Lagged and rolling features are NOT possible for a single prediction
        # as they require previous data.  
        # Options:
        # 1. Use 0 for lagged/rolling features (simplest, but might not be accurate)
        # 2. Use a fixed value (e.g., the mean from your training data)
        # 3. If you have access to historical data, calculate the features.
        df_date['Quantity_lag_1'] = X_test.Quantity_lag_1.mean()  # Or a fixed value
        df_date['Quantity_rolling_mean_7'] = X_test.Quantity_rolling_mean_7.mean()  # Or a fixed value

        features = ['year', 'month', 'day', 'day_of_week', 'is_weekend', 'hour', 'Quantity_lag_1', 'Quantity_rolling_mean_7']
        X_date = df_date[features]

        return X_date

    except ValueError as e:
        st.error(f"Error creating date features: {e}")
        return None

time_for_xgb = create_date_features(time_sentence_input)
st.write(time_for_xgb)
xgbpred = int(model.predict(time_for_xgb))
st.write(f"Predicted Total Volume =  {xgbpred}")

sorted_item_probabilities['Volumes'] = sorted_item_probabilities['Probabilites'] * xgbpred
# Softmax Normalization:
volumes_np = sorted_item_probabilities['Volumes'].values  # Convert to NumPy array
volumes_softmax = np.exp(volumes_np - np.max(volumes_np))  # Subtract max for numerical stability
volumes_softmax = volumes_softmax / np.sum(volumes_softmax)

# Scale by Predicted Volume:
sorted_item_probabilities['Volumes_Softmax'] = volumes_softmax * xgbpred

# If you need integer volumes for display or some other reason:
sorted_item_probabilities['Volumes_Softmax_Rounded'] = sorted_item_probabilities['Volumes_Softmax'].round().astype(int)

sum_volumes_softmax = sorted_item_probabilities['Volumes_Softmax'].sum()
sum_volumes_softmax_rounded = sorted_item_probabilities['Volumes_Softmax_Rounded'].sum()

st.write(f"Sum of Volumes (softmax): {sum_volumes_softmax}")
st.write(f"Sum of Volumes (softmax rounded): {sum_volumes_softmax_rounded}")
st.write("#### Well this discrepancy is a problem! Will fix this soon.")
st.write(sorted_item_probabilities)
