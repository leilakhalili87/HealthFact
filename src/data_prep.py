import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

class HealthDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.tokenizer = tokenizer
        self.data = dataframe
        self.picked_sen = dataframe.picked_sen
        self.targets = self.data.label
        self.max_len = max_len

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        picked_sen = str(self.picked_sen[index])
        picked_sen = " ".join(picked_sen.split())

        inputs = self.tokenizer.encode_plus(
            picked_sen,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
#         token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }


def read_file(df0):
    df = df0.drop(df0[df0['label']== -1].index).reset_index()
    # enc_data = pd.DataFrame(enc.fit_transform(df[['label']]).toarray())
    # df = df.join(enc_data)
    # df['one_hot'] = df[[0,1,2,3]].values.tolist()
    return df[['claim', 'main_text', 'label', 'explanation']]


def Pick_K_Sen(corpus, K):
    for i in range(len(corpus)):
    # for i in range(1):
        if i%100==0:
            print(i)
        sentence_transformer_model = SentenceTransformer('bert-base-uncased')
        claim = corpus['claim'].iloc[i]
        main = corpus['main_text'].iloc[i]
        sent = [sentence for sentence in sent_tokenize(main)]
        sentence_embeddings = sentence_transformer_model.encode(sent)
        claim_embedding = sentence_transformer_model.encode(claim)
        cos_scores = util.pytorch_cos_sim(claim_embedding, sentence_embeddings)[0]
        # Consider main text which are smaller than K sentences
        if len(sent) < K:
            var = len(sent)
        else:
            var = K
        top_results = torch.topk(cos_scores, k=var)[1].tolist()
        corpus.at[i, 'top_k'] = ' '.join(sent[key] for key in top_results)
    corpus['picked_sen'] = corpus['claim'].str.cat(corpus['top_k'],sep=" ")
    return corpus[['claim', 'label', 'picked_sen']]


def save_df(df, name):
    df.to_pickle(name)


def Preprocess(dataset, name, K=5):
    df0 = dataset[str(name)].to_pandas()
    df = read_file(df0)
    df_new = Pick_K_Sen(df, K)
    save_df(df_new, './output/'+ str(name) + '.pkl')
