import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

class SimpleDataSplitter:
    def __init__(self, data_path='data/twnertc_data.pkl'):
        self.df = pd.read_pickle(data_path)
        self.domains = self.df['domain'].unique().tolist()
        
    def create_random_splits(self, test_size=0.15, val_size=0.15, random_state=42):
        print("\n" + "="*60)
        print("RANDOM TRAIN/VAL/TEST SPLIT")
        print("="*60)
        
        # Önce train+val ve test'i ayır
        train_val_df, test_df = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            stratify=self.df['domain']  # Domain dağılımını koru
        )
        
        # Sonra train ve val'i ayır
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size / (1 - test_size),
            random_state=random_state,
            stratify=train_val_df['domain']
        )
        
        print(f"\nToplam: {len(self.df)} örnek")
        print(f"Train: {len(train_df)} ({len(train_df)/len(self.df)*100:.1f}%)")
        print(f"Val:   {len(val_df)} ({len(val_df)/len(self.df)*100:.1f}%)")
        print(f"Test:  {len(test_df)} ({len(test_df)/len(self.df)*100:.1f}%)")
        
        # Her split'teki domain dağılımını göster
        print("\nDomain Dağılımı:")
        print(f"\n{'Domain':<15} {'Train':>8} {'Val':>8} {'Test':>8}")
        print("-" * 45)
        
        for domain in self.domains:
            train_count = len(train_df[train_df['domain'] == domain])
            val_count = len(val_df[val_df['domain'] == domain])
            test_count = len(test_df[test_df['domain'] == domain])
            print(f"{domain:<15} {train_count:>8} {val_count:>8} {test_count:>8}")
        
        return train_df, val_df, test_df
    
    def save_splits(self, train_df, val_df, test_df, output_dir='data/splits'):
        os.makedirs(output_dir, exist_ok=True)
        
        train_df.to_pickle(f'{output_dir}/train.pkl')
        val_df.to_pickle(f'{output_dir}/val.pkl')
        test_df.to_pickle(f'{output_dir}/test.pkl')
        
        print(f"\nSplit'ler kaydedildi: {output_dir}/")
        print(f"  - train.pkl ({len(train_df)} örnek)")
        print(f"  - val.pkl ({len(val_df)} örnek)")
        print(f"  - test.pkl ({len(test_df)} örnek)")


if __name__ == "__main__":
    # Veriyi yükle ve split'le
    splitter = SimpleDataSplitter('data/twnertc_data.pkl')
    
    # Random split oluştur
    train_df, val_df, test_df = splitter.create_random_splits(
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )
    
    # Kaydet
    splitter.save_splits(train_df, val_df, test_df)
    
    print("SPLIT OLUŞTURMA TAMAMLANDI!")