import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# ==================== 1. Dataset Class ====================
class NERDataset(Dataset):
    def __init__(self, texts, tags, tokenizer, label2id, max_length=128):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        words = self.texts[idx]
        labels = self.tags[idx]
        
        # Tokenize with word-level alignment
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Align labels with tokens
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                aligned_labels.append(self.label2id[labels[word_idx]])
            else:
                # For subword tokens, use -100 or same label
                aligned_labels.append(-100)
            previous_word_idx = word_idx
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

# ==================== 2. Data Loading ====================
def load_twnertc_data(file_path):
    """
    TWNERTC formatını yükle
    Format: Her satır bir token ve etiketi içerir, cümleler boş satırla ayrılır
    Örnek:
    Ahmet B-PER
    Istanbul B-LOC
    'e O
    
    gitti O
    . O
    """
    sentences = []
    sentence_tags = []
    current_tokens = []
    current_labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                if current_tokens:
                    sentences.append(current_tokens)
                    sentence_tags.append(current_labels)
                    current_tokens = []
                    current_labels = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    token = parts[0]
                    label = parts[-1]
                    current_tokens.append(token)
                    current_labels.append(label)
        
        # Son cümleyi ekle
        if current_tokens:
            sentences.append(current_tokens)
            sentence_tags.append(current_labels)
    
    return sentences, sentence_tags

# ==================== 3. Model Configuration ====================
MODEL_CONFIGS = {
    'mbert': {
        'name': 'bert-base-multilingual-cased',
        'description': 'Multilingual BERT (Baseline)'
    },
    'berturk': {
        'name': 'dbmdz/bert-base-turkish-cased',
        'description': 'BERTurk (Turkish-specific)'
    },
    'electra': {
        'name': 'dbmdz/electra-base-turkish-cased-discriminator',
        'description': 'ELECTRA Turkish'
    },
    'xlm-roberta': {
        'name': 'xlm-roberta-base',
        'description': 'XLM-RoBERTa Base'
    }
}

# ==================== 4. Training Function ====================
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

# ==================== 5. Evaluation Function ====================
def evaluate(model, dataloader, id2label, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=2)
            
            # Decode predictions and labels
            for pred, label, mask in zip(preds, labels, attention_mask):
                pred_labels = []
                true_labs = []
                
                for p, l, m in zip(pred, label, mask):
                    if m.item() == 1 and l.item() != -100:
                        pred_labels.append(id2label[p.item()])
                        true_labs.append(id2label[l.item()])
                
                if pred_labels:
                    predictions.append(pred_labels)
                    true_labels.append(true_labs)
    
    # Calculate metrics
    f1 = f1_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    report = classification_report(true_labels, predictions)
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'report': report
    }

# ==================== 6. Main Training Pipeline ====================
def train_model(model_key, train_texts, train_tags, val_texts, val_tags, 
                label2id, id2label, config, device):
    """
    Train a single model configuration
    """
    print(f"\n{'='*80}")
    print(f"Training: {MODEL_CONFIGS[model_key]['description']}")
    print(f"{'='*80}")
    
    # Load tokenizer and model
    model_name = MODEL_CONFIGS[model_key]['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    ).to(device)
    
    # Create datasets
    train_dataset = NERDataset(train_texts, train_tags, tokenizer, label2id, config['max_length'])
    val_dataset = NERDataset(val_texts, val_tags, tokenizer, label2id, config['max_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_f1 = 0
    results = []
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        metrics = evaluate(model, val_loader, id2label, device)
        print(f"Validation F1: {metrics['f1']:.4f}")
        print(f"Validation Precision: {metrics['precision']:.4f}")
        print(f"Validation Recall: {metrics['recall']:.4f}")
        
        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_f1': metrics['f1'],
            'val_precision': metrics['precision'],
            'val_recall': metrics['recall']
        })
        
        # Save best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), f"best_model_{model_key}.pt")
            print(f"✓ New best model saved! F1: {best_f1:.4f}")
    
    print(f"\n{MODEL_CONFIGS[model_key]['description']} - Classification Report:")
    print(metrics['report'])
    
    return {
        'model_key': model_key,
        'best_f1': best_f1,
        'final_metrics': metrics,
        'training_history': results
    }

# ==================== 7. Main Experiment Runner ====================
def run_experiments(domain_dep_data, domain_indep_data, config):
    """
    Run all experiments: 4 models × 2 settings = 8 experiments
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    all_results = {}
    
    # Domain-Dependent Experiments
    print("\n" + "="*80)
    print("DOMAIN-DEPENDENT EXPERIMENTS")
    print("="*80)
    
    dd_train_texts, dd_train_tags = domain_dep_data['train']
    dd_val_texts, dd_val_tags = domain_dep_data['val']
    dd_test_texts, dd_test_tags = domain_dep_data['test']
    
    # Get label mappings
    all_labels = set()
    for tags in dd_train_tags + dd_val_tags + dd_test_tags:
        all_labels.update(tags)
    
    label2id = {label: i for i, label in enumerate(sorted(all_labels))}
    id2label = {i: label for label, i in label2id.items()}
    
    for model_key in MODEL_CONFIGS.keys():
        result = train_model(
            model_key, 
            dd_train_texts, dd_train_tags,
            dd_val_texts, dd_val_tags,
            label2id, id2label, config, device
        )
        all_results[f'{model_key}_domain_dependent'] = result
    
    # Domain-Independent Experiments
    print("\n" + "="*80)
    print("DOMAIN-INDEPENDENT EXPERIMENTS (Cross-Domain)")
    print("="*80)
    
    di_train_texts, di_train_tags = domain_indep_data['train']
    di_val_texts, di_val_tags = domain_indep_data['val']
    di_test_texts, di_test_tags = domain_indep_data['test']
    
    for model_key in MODEL_CONFIGS.keys():
        result = train_model(
            model_key,
            di_train_texts, di_train_tags,
            di_val_texts, di_val_tags,
            label2id, id2label, config, device
        )
        all_results[f'{model_key}_domain_independent'] = result
    
    return all_results

# ==================== 8. Results Analysis ====================
def analyze_results(results):
    """
    Create comprehensive analysis of all experiments
    """
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    # Create results table
    df_data = []
    for exp_name, result in results.items():
        model_key = result['model_key']
        setting = 'Domain-Dependent' if 'dependent' in exp_name else 'Domain-Independent'
        
        df_data.append({
            'Model': MODEL_CONFIGS[model_key]['description'],
            'Setting': setting,
            'F1': result['best_f1'],
            'Precision': result['final_metrics']['precision'],
            'Recall': result['final_metrics']['recall']
        })
    
    df = pd.DataFrame(df_data)
    print("\n" + df.to_string(index=False))
    
    # Domain transfer analysis
    print("\n" + "="*80)
    print("DOMAIN TRANSFER ANALYSIS (Performance Drop)")
    print("="*80)
    
    for model_key in MODEL_CONFIGS.keys():
        dd_f1 = results[f'{model_key}_domain_dependent']['best_f1']
        di_f1 = results[f'{model_key}_domain_independent']['best_f1']
        drop = dd_f1 - di_f1
        drop_pct = (drop / dd_f1) * 100
        
        print(f"{MODEL_CONFIGS[model_key]['description']}:")
        print(f"  In-Domain F1: {dd_f1:.4f}")
        print(f"  Cross-Domain F1: {di_f1:.4f}")
        print(f"  Performance Drop: {drop:.4f} ({drop_pct:.2f}%)")
        print()
    
    # Save results to JSON
    with open('experiment_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("Results saved to 'experiment_results.json'")

# ==================== 9. Main Entry Point ====================
if __name__ == "__main__":
    # Configuration
    config = {
        'batch_size': 16,
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'epochs': 5,
        'max_length': 128,
        'warmup_ratio': 0.1
    }
    
    # Example: Load your data
    # Dosya yollarınızı buraya yazın
    print("Loading TWNERTC dataset...")
    
    # Domain-Dependent setup (örneğin: news domain)
    # train_texts_dd, train_tags_dd = load_twnertc_data('data/news_train.txt')
    # val_texts_dd, val_tags_dd = load_twnertc_data('data/news_val.txt')
    # test_texts_dd, test_tags_dd = load_twnertc_data('data/news_test.txt')
    
    # Domain-Independent setup (örneğin: news'te train, social'da test)
    # train_texts_di, train_tags_di = load_twnertc_data('data/news_train.txt')
    # val_texts_di, val_tags_di = load_twnertc_data('data/social_val.txt')
    # test_texts_di, test_tags_di = load_twnertc_data('data/social_test.txt')
    
    # Dummy data for demonstration
    print("Creating dummy data for demonstration...")
    dummy_texts = [['Ahmet', 'İstanbul', "'a", 'gitti'], ['Mehmet', 'Ankara', "'da", 'yaşıyor']]
    dummy_tags = [['B-PER', 'B-LOC', 'O', 'O'], ['B-PER', 'B-LOC', 'O', 'O']]
    
    domain_dep_data = {
        'train': (dummy_texts * 50, dummy_tags * 50),
        'val': (dummy_texts * 10, dummy_tags * 10),
        'test': (dummy_texts * 10, dummy_tags * 10)
    }
    
    domain_indep_data = {
        'train': (dummy_texts * 50, dummy_tags * 50),
        'val': (dummy_texts * 10, dummy_tags * 10),
        'test': (dummy_texts * 10, dummy_tags * 10)
    }
    
    print("\nStarting experiments...")
    print(f"Configuration: {config}")
    
    # Run all experiments
    results = run_experiments(domain_dep_data, domain_indep_data, config)
    
    # Analyze and display results
    analyze_results(results)
    
    print("\n✓ All experiments completed!")
    print("Check 'experiment_results.json' for detailed results.")
    print("Best models saved as 'best_model_*.pt'")