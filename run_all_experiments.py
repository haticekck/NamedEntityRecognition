import pandas as pd
import json
import os
from train import main, MODELS

def run_all_models():
    all_results = {}
    
    print("\n" + "#"*80)
    print("# TÜM MODELLER EĞİTİLİYOR")
    print("#"*80)
    
    for model_name in MODELS.keys():
        print(f"\n{'='*80}")
        print(f"Model: {model_name}")
        print(f"{'='*80}")
        
        try:
            results = main(model_name)
            all_results[model_name] = results
            
            print(f"\n{model_name} tamamlandı:")
            print(f"  F1:       {results['f1']:.4f}")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            
        except Exception as e:
            print(f"\n{model_name} HATA: {str(e)}")
            all_results[model_name] = {'error': str(e)}
    
    # Tüm sonuçları kaydet
    os.makedirs('results', exist_ok=True)
    with open('results/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "#"*80)
    print("# TÜM MODELLER TAMAMLANDI!")
    print("#"*80)
    print(f"\nSonuçlar kaydedildi: results/all_results.json")
    
    return all_results


def create_summary_table(results_path='results/all_results.json'):
    with open(results_path, 'r') as f:
        all_results = json.load(f)
    
    # DataFrame oluştur
    rows = []
    for model_name, results in all_results.items():
        if 'error' not in results:
            rows.append({
                'model': model_name,
                'loss': results.get('loss', None),
                'accuracy': results.get('accuracy', None),
                'precision': results['precision'],
                'recall': results['recall'],
                'f1': results['f1']
            })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('f1', ascending=False)
    
    print("\n" + "="*80)
    print("MODEL KARŞILAŞTIRMASI")
    print("="*80)
    print(f"\n{'Model':<15} {'Loss':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)
    for _, row in df.iterrows():
        loss_str = f"{row['loss']:.4f}" if pd.notna(row['loss']) else "N/A"
        acc_str = f"{row['accuracy']:.4f}" if pd.notna(row['accuracy']) else "N/A"
        print(f"{row['model']:<15} {loss_str:>10} {acc_str:>10} {row['precision']:>10.4f} {row['recall']:>10.4f} {row['f1']:>10.4f}")
    
    best_model = df.iloc[0]
    print(f"\nEN İYİ MODEL: {best_model['model']}")
    print(f"  F1:        {best_model['f1']:.4f}")
    print(f"  Accuracy:  {best_model['accuracy']:.4f}")
    print(f"  Precision: {best_model['precision']:.4f}")
    print(f"  Recall:    {best_model['recall']:.4f}")
    if pd.notna(best_model['loss']):
        print(f"  Loss:      {best_model['loss']:.4f}")
    
    # Kaydet
    df.to_csv('results/summary_table.csv', index=False)
    print(f"\nÖzet tablo kaydedildi: results/summary_table.csv")
    
    return df


def create_final_report():
    df = pd.read_csv('results/summary_table.csv')
    
    report = []
    report.append("="*80)
    report.append("TÜRKÇE NER PROJESİ - SONUÇ RAPORU")
    report.append("="*80)
    report.append("")
    report.append("Dataset: TWNERTC")
    report.append(f"Toplam Model: {len(df)}")
    report.append("")
    
    # Sonuçlar
    report.append("MODEL PERFORMANSLARI")
    report.append("-"*80)
    report.append(f"{'Model':<15} {'Loss':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    report.append("-"*70)
    
    for _, row in df.iterrows():
        loss_str = f"{row['loss']:.4f}" if pd.notna(row['loss']) else "N/A"
        acc_str = f"{row['accuracy']:.4f}" if pd.notna(row['accuracy']) else "N/A"
        report.append(f"{row['model']:<15} {loss_str:>10} {acc_str:>10} {row['precision']:>10.4f} {row['recall']:>10.4f} {row['f1']:>10.4f}")
    
    report.append("")
    
    # En iyi model
    best = df.iloc[0]
    report.append("EN İYİ MODEL")
    report.append("-"*80)
    report.append(f"Model:     {best['model']}")
    report.append(f"F1 Score:  {best['f1']:.4f}")
    report.append(f"Accuracy:  {best['accuracy']:.4f}")
    report.append(f"Precision: {best['precision']:.4f}")
    report.append(f"Recall:    {best['recall']:.4f}")
    if pd.notna(best['loss']):
        report.append(f"Loss:      {best['loss']:.4f}")
    report.append("")
    
    # İstatistikler
    report.append("İSTATİSTİKLER")
    report.append("-"*80)
    report.append(f"Ortalama F1:        {df['f1'].mean():.4f}")
    report.append(f"Ortalama Accuracy:  {df['accuracy'].mean():.4f}")
    report.append(f"Ortalama Precision: {df['precision'].mean():.4f}")
    report.append(f"Ortalama Recall:    {df['recall'].mean():.4f}")
    report.append("")
    report.append(f"En yüksek F1:       {df['f1'].max():.4f} ({df.loc[df['f1'].idxmax(), 'model']})")
    report.append(f"En düşük F1:        {df['f1'].min():.4f} ({df.loc[df['f1'].idxmin(), 'model']})")
    report.append("")
    
    # Öneriler
    report.append("ÖNERİLER")
    report.append("-"*80)
    
    # Türkçe modeller vs multilingual
    turkish_models = df[df['model'].isin(['berturk', 'electra'])]['f1'].mean()
    multi_models = df[df['model'].isin(['mbert', 'xlm-r'])]['f1'].mean()
    
    if turkish_models > multi_models:
        report.append("• Türkçe-specific modeller (BERTurk, Electra) daha iyi performans gösterdi")
        report.append(f"  Türkçe modeller avg: {turkish_models:.4f}")
        report.append(f"  Multilingual avg:    {multi_models:.4f}")
    else:
        report.append("• Multilingual modeller (mBERT, XLM-R) beklenmedik şekilde daha iyi performans gösterdi")
        report.append(f"  Multilingual avg:    {multi_models:.4f}")
        report.append(f"  Türkçe modeller avg: {turkish_models:.4f}")
    
    report.append("")
    report.append(f"• Production için önerilen model: {best['model']}")
    report.append(f"• Bu model {best['f1']:.2%} F1 score ile en iyi performansı gösterdi")
    report.append("")
    report.append("="*80)
    
    report_text = "\n".join(report)
    
    # Kaydet
    with open('results/final_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print("\nRapor kaydedildi: results/final_report.txt")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Tüm NER Modellerini Çalıştır')
    parser.add_argument('--mode', type=str, default='run',
                       choices=['run', 'summary', 'report'],
                       help='Mode: run (modelleri çalıştır), summary (özet tablo), report (final rapor)')
    
    args = parser.parse_args()
    
    if args.mode == 'run':
        # Tüm modelleri çalıştır
        results = run_all_models()
        
        # Otomatik olarak özet ve rapor oluştur
        create_summary_table()
        create_final_report()
        
    elif args.mode == 'summary':
        create_summary_table()
        
    elif args.mode == 'report':
        create_final_report()