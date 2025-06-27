#!/usr/bin/env python3
"""
Script di esempio per testare la pipeline Manga to Anime
"""

import os
import sys

def test_pipeline():
    """
    Test della pipeline con dati di esempio
    """
    
    # Path di esempio - modifica questi secondo la tua configurazione
    MANGA_DATASET_PATH = r"c:\Users\franc\OneDrive\Desktop\uni\Magistrale\2 semestre\Computer Vision\ProgettoCV\Manga109s_released_2023_12_07"
    TOONCRAFTER_PATH = r"c:\Users\franc\OneDrive\Desktop\uni\Magistrale\2 semestre\Computer Vision\ProgettoCV\ToonCrafter"
    
    # Verifica che i path esistano
    if not os.path.exists(MANGA_DATASET_PATH):
        print(f"❌ Dataset manga non trovato: {MANGA_DATASET_PATH}")
        print("Modifica il path MANGA_DATASET_PATH nello script")
        return False
    
    if not os.path.exists(TOONCRAFTER_PATH):
        print(f"❌ ToonCrafter non trovato: {TOONCRAFTER_PATH}")
        print("Modifica il path TOONCRAFTER_PATH nello script")
        return False
    
    # Importa i moduli locali
    sys.path.append(os.path.dirname(__file__))
    
    try:
        from Manga109Dataset import Manga109Dataset
        from manga_animation_pipeline import MangaAnimationPipeline
    except ImportError as e:
        print(f"❌ Errore nell'importazione: {e}")
        return False
    
    # Test 1: Carica dataset e mostra statistiche
    print("🧪 Test 1: Caricamento dataset Manga109")
    try:
        dataset = Manga109Dataset(MANGA_DATASET_PATH)
        print(f"   ✅ Dataset caricato: {len(dataset)} panel totali")
        
        # Mostra alcuni manga disponibili
        manga_names = list(set(panel['manga'] for panel in dataset.panels))[:5]
        print(f"   📚 Primi 5 manga disponibili: {manga_names}")
        
    except Exception as e:
        print(f"   ❌ Errore nel caricamento dataset: {e}")
        return False
    
    # Test 2: Inizializza pipeline
    print("\n🧪 Test 2: Inizializzazione pipeline")
    try:
        pipeline = MangaAnimationPipeline(
            MANGA_DATASET_PATH,
            TOONCRAFTER_PATH,
            "test_output"
        )
        print("   ✅ Pipeline inizializzata")
    except Exception as e:
        print(f"   ❌ Errore nell'inizializzazione pipeline: {e}")
        return False
    
    # Test 3: Analisi di un manga (solo analisi, nessuna generazione)
    print("\n🧪 Test 3: Analisi panel manga")
    if manga_names:
        test_manga = manga_names[0]
        print(f"   Analizzando manga: {test_manga}")
        
        try:
            analysis = pipeline.analyze_manga_panels(test_manga)
            
            if 'error' in analysis:
                print(f"   ⚠️  {analysis['error']}")
            else:
                print(f"   ✅ Analisi completata:")
                print(f"      - Panel totali: {analysis['total_panels']}")
                print(f"      - Pagine: {analysis['pages']}")
                print(f"      - Sequenze raccomandate: {len(analysis['recommended_sequences'])}")
                
                # Mostra dettagli delle prime sequenze
                for i, seq in enumerate(analysis['recommended_sequences'][:3]):
                    print(f"      - Sequenza {i+1}: {seq['name']} ({seq['recommended_config']})")
        
        except Exception as e:
            print(f"   ❌ Errore nell'analisi: {e}")
            return False
    
    # Test 4: Preparazione input per ToonCrafter (senza esecuzione)
    print("\n🧪 Test 4: Preparazione input ToonCrafter")
    try:
        if analysis['recommended_sequences']:
            seq = analysis['recommended_sequences'][0]
            
            # Carica le immagini dei panel
            panel1_idx = dataset.panels.index(seq['panel1'])
            panel2_idx = dataset.panels.index(seq['panel2'])
            
            panel1_img, _ = dataset[panel1_idx]
            panel2_img, _ = dataset[panel2_idx]
            
            print(f"   Panel 1: {panel1_img.size} pixels")
            print(f"   Panel 2: {panel2_img.size} pixels")
            
            # Prepara input (questo crea i file necessari per ToonCrafter)
            prompt_dir = pipeline.converter.prepare_tooncrafter_input(
                panel1_img, panel2_img,
                seq['panel1'], seq['panel2'],
                "test_output", f"test_{seq['name']}"
            )
            
            print(f"   ✅ Input preparato in: {prompt_dir}")
            
            # Verifica che i file siano stati creati
            frame1_path = os.path.join(prompt_dir, f"test_{seq['name']}_frame1.png")
            frame3_path = os.path.join(prompt_dir, f"test_{seq['name']}_frame3.png")
            prompt_path = os.path.join(prompt_dir, "prompts.txt")
            
            if all(os.path.exists(p) for p in [frame1_path, frame3_path, prompt_path]):
                print("   ✅ Tutti i file di input creati correttamente")
                
                # Mostra il prompt generato
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt_text = f.read().strip()
                print(f"   📝 Prompt generato: \"{prompt_text}\"")
            else:
                print("   ⚠️  Alcuni file di input mancanti")
        
    except Exception as e:
        print(f"   ❌ Errore nella preparazione input: {e}")
        return False
    
    print("\n🎉 Tutti i test completati con successo!")
    print("\n📋 Prossimi passi:")
    print("1. Verifica che ToonCrafter sia configurato correttamente")
    print("2. Esegui la pipeline completa con:")
    print(f"   python manga_animation_pipeline.py \\")
    print(f"     --manga_dataset \"{MANGA_DATASET_PATH}\" \\")
    print(f"     --tooncrafter_path \"{TOONCRAFTER_PATH}\" \\")
    print(f"     --manga_name \"{test_manga}\" \\")
    print(f"     --max_sequences 2")
    print("3. Per esecuzione automatica, aggiungi --auto_run")
    
    return True


if __name__ == "__main__":
    print("🧪 Test Pipeline Manga to Anime\n")
    
    success = test_pipeline()
    
    if success:
        print("\n✅ Test completato con successo!")
    else:
        print("\n❌ Test fallito - controlla gli errori sopra")
        sys.exit(1)
