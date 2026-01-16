import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class TWNERTCLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.domains = None
        self.entity_types = None
        
    def load_data(self):
        data = []
        
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split('\t')
                if len(parts) != 3:
                    continue
                    
                domain = parts[0]
                labels = parts[1].split()
                tokens = parts[2].split()
                
                # Token ve label sayısı eşleşmeli !!!
                if len(tokens) != len(labels):
                    print(f"Uyarı: Token-Label mismatch, atlanıyor: {line[:50]}...")
                    continue
                
                data.append({
                    'domain': domain,
                    'tokens': tokens,
                    'labels': labels
                })
        
        self.data = pd.DataFrame(data)
        self.domains = self.data['domain'].unique().tolist()
        
        # Entity tiplerini çıkar
        all_labels = [label for labels in self.data['labels'] for label in labels]
        self.entity_types = sorted(list(set([
            label.split('-')[1] for label in all_labels if label != 'O'
        ])))
        
        print(f"{len(self.data)} örnek yüklendi")
        print(f"{len(self.domains)} domain: {self.domains}")
        print(f"{len(self.entity_types)} entity tipi: {self.entity_types}")
        
        return self.data
    
    def analyze_data(self):
        if self.data is None:
            raise ValueError("Önce load_data() çağırın")
        
        print("\n" + "="*60)
        print("VERİ ANALİZİ")
        print("="*60)
        
        # Domain dağılımı
        print("\n1. DOMAIN DAĞILIMI:")
        print(self.data['domain'].value_counts())
        
        # Token istatistikleri
        print("\n2. TOKEN İSTATİSTİKLERİ:")
        token_lengths = self.data['tokens'].apply(len)
        print(f"Ortalama cümle uzunluğu: {token_lengths.mean():.2f}")
        print(f"Min: {token_lengths.min()}, Max: {token_lengths.max()}")
        print(f"Median: {token_lengths.median():.2f}")
        
        # Label dağılımı
        print("\n3. LABEL DAĞILIMI:")
        all_labels = [label for labels in self.data['labels'] for label in labels]
        label_counts = Counter(all_labels)
        
        for label, count in label_counts.most_common(10):
            percentage = (count / len(all_labels)) * 100
            print(f"{label:15s}: {count:6d} ({percentage:5.2f}%)")
        
        # Domain bazlı entity sayıları
        print("\n4. DOMAIN BAZLI ENTITY SAYILARI:")
        for domain in self.domains:
            domain_data = self.data[self.data['domain'] == domain]
            domain_labels = [label for labels in domain_data['labels'] for label in labels]
            entity_count = sum(1 for l in domain_labels if l.startswith('B-'))
            print(f"{domain:15s}: {len(domain_data):5d} cümle, {entity_count:5d} entity")
        
        # Entity tipi bazlı sayılar
        print("\n5. ENTITY TİPİ DAĞILIMI:")
        entity_type_counts = Counter([
            label.split('-')[1] for label in all_labels if label.startswith('B-')
        ])
        
        for entity_type, count in entity_type_counts.most_common():
            print(f"{entity_type:15s}: {count:5d}")
        
        return {
            'domain_counts': self.data['domain'].value_counts().to_dict(),
            'token_stats': {
                'mean': token_lengths.mean(),
                'min': token_lengths.min(),
                'max': token_lengths.max(),
                'median': token_lengths.median()
            },
            'label_counts': label_counts,
            'entity_type_counts': entity_type_counts
        }
    
    def visualize_data(self, save_path='data_analysis.png'):
        if self.data is None:
            raise ValueError("Önce load_data() çağırın")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Domain dağılımı
        self.data['domain'].value_counts().plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Domain Distribution')
        axes[0, 0].set_xlabel('Domain')
        axes[0, 0].set_ylabel('Sample Count')
        
        # 2. Cümle uzunluğu dağılımı
        token_lengths = self.data['tokens'].apply(len)
        axes[0, 1].hist(token_lengths, bins=50, edgecolor='black')
        axes[0, 1].set_title('Sentence Length Distribution')
        axes[0, 1].set_xlabel('Token Count')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(token_lengths.mean(), color='r', linestyle='--', label='Mean')
        axes[0, 1].legend()
        
        # 3. Entity tipi dağılımı
        all_labels = [label for labels in self.data['labels'] for label in labels]
        entity_type_counts = Counter([
            label.split('-')[1] for label in all_labels if label.startswith('B-')
        ])
        
        entities_df = pd.DataFrame.from_dict(entity_type_counts, orient='index', columns=['count'])
        entities_df.sort_values('count', ascending=True).plot(kind='barh', ax=axes[1, 0], legend=False)
        axes[1, 0].set_title('Entity Type Distribution')
        axes[1, 0].set_xlabel('Count')
        
        # 4. Domain bazlı entity yoğunluğu
        domain_entity_density = {}
        for domain in self.domains:
            domain_data = self.data[self.data['domain'] == domain]
            domain_labels = [label for labels in domain_data['labels'] for label in labels]
            entity_count = sum(1 for l in domain_labels if l.startswith('B-'))
            total_tokens = sum(len(tokens) for tokens in domain_data['tokens'])
            domain_entity_density[domain] = (entity_count / total_tokens) * 100
        
        pd.Series(domain_entity_density).plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Domain-Based Entity Ratio (%)')
        axes[1, 1].set_xlabel('Domain')
        axes[1, 1].set_ylabel('Entity/Token Ratio (%)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nGörselleştirme kaydedildi: {save_path}")
        plt.close()


if __name__ == "__main__":
    # Veri yükle
    loader = TWNERTCLoader('data/raw/twnertc_coarse_dd.DUMP')
    df = loader.load_data()
    
    # Analiz yap
    stats = loader.analyze_data()
    
    # Görselleştir
    loader.visualize_data()
    
    # Veriyi kaydet (sonraki scriptler için)
    df.to_pickle('data/twnertc_data.pkl')
    print("\nVeri kaydedildi: data/twnertc_data.pkl")