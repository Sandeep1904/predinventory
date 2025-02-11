import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformer import Transformer
import torch


st.title("Let's predict what food is ordered and how much")

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


transformer.eval()
transformer.to(device)
item_sentence = [[START_TOKEN]]
time_sentence = ('2019-07-07',)
for word_counter in range(10):
    encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(time_sentence, item_sentence)
    predictions = transformer(time_sentence,
                                item_sentence,
                                encoder_self_attention_mask.to(device),
                                decoder_self_attention_mask.to(device),
                                decoder_cross_attention_mask.to(device),
                                enc_start_token=False,
                                enc_end_token=False,
                                dec_start_token=True,
                                dec_end_token=False)
    next_token_prob_distribution = predictions[0][word_counter]
    next_token_index = torch.argmax(next_token_prob_distribution).item()
    next_token = index_to_obj[next_token_index]
    print(item_sentence)
    print(item_sentence[0])
    item_sentence[0].append(next_token)
    print(item_sentence)
    if next_token == END_TOKEN:
        break

st.write(f"Testing (2017-10-07): {item_sentence[0]}")
st.write("--------------------------------------------------------------------------------")
