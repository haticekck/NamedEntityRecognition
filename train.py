import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
import os
import json
from dataset import NERDataset, create_label_mappings

# Model konfigürasyonları
MODELS = {
    'mbert': 'bert-base-multilingual-cased',
    'berturk': 'dbmdz/bert-base-turkish-cased',
    'electra': 'dbmdz/electra-base-turkish-cased-discriminator',
    'xlm-r': 'xlm-roberta-base'
}


def compute_metrics(eval_pred, id2label):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Label listelerini oluştur
    true_labels = []
    pred_labels = []
    
    # Accuracy için
    correct = 0
    total = 0
    
    for prediction, label in zip(predictions, labels):
        true_list = []
        pred_list = []
        
        for pred_id, label_id in zip(prediction, label):
            if label_id != -100:  # Padding ve subword'leri ignore et
                true_list.append(id2label[label_id])
                pred_list.append(id2label[pred_id])
                
                # Token-level accuracy
                if pred_id == label_id:
                    correct += 1
                total += 1
        
        if true_list:  # Boş liste kontrolü
            true_labels.append(true_list)
            pred_labels.append(pred_list)
    
    # Token-level accuracy
    token_accuracy = correct / total if total > 0 else 0
    
    # Seqeval metrikleri
    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
        "accuracy": token_accuracy
    }


def train_model(model_name, model_path, train_df, val_df, label2id, id2label, 
                output_dir, max_length=64, batch_size=32, epochs=3, 
                learning_rate=2e-5):
    print("\n" + "="*70)
    print(f"MODEL EĞİTİMİ: {model_name} ({model_path})")
    print("="*70)
    
    # Tokenizer yükle
    print("\n1. Tokenizer yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Dataset'leri oluştur
    print("2. Dataset'ler oluşturuluyor...")
    train_dataset = NERDataset(train_df, tokenizer, label2id, max_length)
    val_dataset = NERDataset(val_df, tokenizer, label2id, max_length)
    
    print(f"   Train: {len(train_dataset)} örnek")
    print(f"   Val:   {len(val_dataset)} örnek")
    
    # Model yükle
    print("3. Model yükleniyor...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=42,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    # Data collator, aslında amacı dinamik bir şekilde padding yapmak yani 
    # max token length 128 olsa bile dinamik olarak tokenlar arasında en yüksek olana eşitleyecekti.
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Trainer
    print("4. Trainer oluşturuluyor...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Eğitim
    print("5. Eğitim başlıyor...\n")
    trainer.train()
    
    print("\nEğitim tamamlandı!")
    
    # En iyi modeli kaydet
    best_model_path = f'{output_dir}/best_model'
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    
    print(f"En iyi model kaydedildi: {best_model_path}")
    
    return trainer


def evaluate_model(trainer, test_df, label2id, id2label):
    """Test seti üzerinde değerlendir"""
    print("\n" + "-"*70)
    print("TEST SETİ DEĞERLENDİRMESİ")
    print("-"*70)
    
    # Test dataset oluştur
    test_dataset = NERDataset(
        test_df, 
        trainer.tokenizer, 
        label2id,
        trainer.args.per_device_eval_batch_size
    )
    
    # Predict
    predictions = trainer.predict(test_dataset)
    
    # Loss'u al
    test_loss = predictions.metrics.get('test_loss', None)
    
    # Metrikleri hesapla
    pred_logits = predictions.predictions
    pred_labels = np.argmax(pred_logits, axis=2)
    true_labels_ids = predictions.label_ids
    
    # Label listelerine çevir
    true_labels = []
    pred_labels_converted = []
    
    # Accuracy için
    correct = 0
    total = 0
    
    for pred_seq, true_seq in zip(pred_labels, true_labels_ids):
        true_list = []
        pred_list = []
        
        for pred_id, true_id in zip(pred_seq, true_seq):
            if true_id != -100:
                true_list.append(id2label[true_id])
                pred_list.append(id2label[pred_id])
                
                if pred_id == true_id:
                    correct += 1
                total += 1
        
        if true_list:
            true_labels.append(true_list)
            pred_labels_converted.append(pred_list)
    
    # Token-level accuracy
    token_accuracy = correct / total if total > 0 else 0
    
    # Overall metrics
    results = {
        'loss': test_loss,
        'accuracy': token_accuracy,
        'precision': precision_score(true_labels, pred_labels_converted),
        'recall': recall_score(true_labels, pred_labels_converted),
        'f1': f1_score(true_labels, pred_labels_converted)
    }
    
    # Detailed classification report
    classification_rep = classification_report(
        true_labels, 
        pred_labels_converted,
        digits=4
    )
    
    print("\nTest Sonuçları:")
    if test_loss is not None:
        print(f"  Loss:      {test_loss:.4f}")
    print(f"  Accuracy:  {token_accuracy:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")
    
    print("\nDetaylı Rapor:")
    print(classification_rep)
    
    return results, classification_rep


def main(model_name):
    # Model path'i al
    if model_name not in MODELS:
        raise ValueError(f"Geçersiz model: {model_name}. Seçenekler: {list(MODELS.keys())}")
    
    model_path = MODELS[model_name]
    output_dir = f'results/{model_name}'
    
    # Veriyi yükle
    print(f"\nVeri yükleniyor...")
    train_df = pd.read_pickle('data/splits/train.pkl')
    val_df = pd.read_pickle('data/splits/val.pkl')
    test_df = pd.read_pickle('data/splits/test.pkl')
    
    # Label mappings oluştur
    label2id, id2label = create_label_mappings(train_df)
    
    # Eğit
    trainer = train_model(
        model_name, 
        model_path,
        train_df, 
        val_df,
        label2id,
        id2label,
        output_dir
    )
    
    # Test et
    results, classification_rep = evaluate_model(
        trainer,
        test_df,
        label2id,
        id2label
    )
    
    # Sonuçları kaydet
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(f'{output_dir}/classification_report.txt', 'w') as f:
        f.write(classification_rep)
    
    print(f"\nSonuçlar kaydedildi: {output_dir}/")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NER Model Eğitimi')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['mbert', 'berturk', 'electra', 'xlm-r'],
                       help='Model adı')
    
    args = parser.parse_args()
    
    results = main(args.model)
    
    print("\n" + "="*70)
    print("EĞİTİM TAMAMLANDI!")
    print("="*70)
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")