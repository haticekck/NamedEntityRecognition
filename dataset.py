import torch
from torch.utils.data import Dataset
import pandas as pd

class NERDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label2id, max_length=64): #token length açıklamasını raporda yap!
        self.data = dataframe.reset_index(drop=True) #satır numaraları tekrar sıralanır, aşağı() 
        #önemli çünkü çıkarttığımız veriler böylelikle maskelenmiş olur drop(true) yeni index satırı oluşturur
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx): #aligment işlemi burada yapılır çok önemli çünkü tokenization sonrası label'ların
        #subword'lere göre hizalanması gerekir
        tokens = self.data.loc[idx, 'tokens']
        labels = self.data.loc[idx, 'labels']
        
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True, #burada önemli çünkü tokenler zaten listelenmiş diye söylüyoruz
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Label alignment için word_ids
        word_ids = encoding.word_ids(batch_index=0)
        
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special token veya padding
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # Yeni bir word'ün ilk subword'ü
                aligned_labels.append(self.label2id[labels[word_idx]])
            else:
                # Aynı word'ün devam eden subword'leri
                # İlk subword'e label ver, diğerlerine -100 (ignore)
                aligned_labels.append(-100)
            
            previous_word_idx = word_idx
        #-100'ün amacı: special tokenlar ve padding için loss hesaplamasında ignore edilmesi
        
        # Encoding'e label'ları ekle
        encoding['labels'] = torch.tensor(aligned_labels, dtype=torch.long)
        
        # Batch dimension'ı kaldır önemli bir adım!!!
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        return item


def create_label_mappings(dataframe):
    all_labels = set()
    for labels in dataframe['labels']:
        all_labels.update(labels)
    
    # Alfabetik sırala
    label_list = sorted(list(all_labels))
    
    # Mapping'leri oluştur
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    return label2id, id2label


def get_dataloaders(train_df, val_df, test_df, tokenizer, label2id, 
                   batch_size=16, max_length=64, num_workers=4):
    from torch.utils.data import DataLoader
    
    # Dataset'leri oluştur
    train_dataset = NERDataset(train_df, tokenizer, label2id, max_length)
    val_dataset = NERDataset(val_df, tokenizer, label2id, max_length)
    test_dataset = NERDataset(test_df, tokenizer, label2id, max_length)
    
    # DataLoader'ları oluştur
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test kodu
    from transformers import AutoTokenizer
    
    print("Dataset sınıfı test ediliyor...\n")
    
    # Örnek veri yükle
    train_df = pd.read_pickle('data/splits/train.pkl')
    
    # Tokenizer yükle
    tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
    
    # Label mapping oluştur
    label2id, id2label = create_label_mappings(train_df)
    
    print(f"Toplam {len(label2id)} farklı label var:")
    print(f"Label2id: {label2id}")
    print(f"\nDataset boyutu: {len(train_df)}")
    
    # Dataset oluştur
    dataset = NERDataset(train_df, tokenizer, label2id)
    
    # İlk örnek üzerinden anlama
    sample = dataset[0]
    
    print("\nİlk örnek:")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    
    # Token'ları decode et
    tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'])
    labels = [id2label.get(l.item(), 'PAD') if l.item() != -100 else 'IGN' 
              for l in sample['labels']]
    
    print("\nİlk 20 token ve label:")
    for i, (token, label) in enumerate(zip(tokens[:20], labels[:20])):
        print(f"{i:2d}. {token:15s} -> {label}")
    
    print("\nDataset sınıfı başarıyla test edildi!")