#!/usr/bin/env python3
"""
Esempio di utilizzo del modulo preprocessing manga per ToonCrafter
Dimostra le funzionalità principali del sistema di preprocessing
"""

import os
import sys
import tempfile
from pathlib import Path

# Aggiungi il path per import
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def esempio_base():
    """Esempio di utilizzo base con preprocessing automatico"""
    print("🎌 === ESEMPIO BASE: Preprocessing Automatico ===")
    
    try:
        from colab_tooncrafter_runner import run_with_manga_preprocessing
        
        # Configurazione base
        tooncrafter_path = "path/to/ToonCrafter"  # Cambia con il tuo path
        prompt_dir = "input/manga_panels"         # Directory con i tuoi panel manga
        output_dir = "output/videos"              # Directory output
        
        print(f"📁 Input: {prompt_dir}")
        print(f"📁 Output: {output_dir}")
        print(f"🎨 Preprocessing: Abilitato (preset 'default')")
        print(f"⚙️ Configurazione: 'dramatic_change'")
        
        # Nota: Questo esempio richiede ToonCrafter installato e configurato
        # Decommentare per uso reale:
        
        # success = run_with_manga_preprocessing(
        #     tooncrafter_path=tooncrafter_path,
        #     prompt_dir=prompt_dir,
        #     output_dir=output_dir,
        #     config_type="dramatic_change",
        #     preprocessing_preset="default",
        #     enable_quality_analysis=True
        # )
        # 
        # if success:
        #     print("✅ Conversione completata con successo!")
        # else:
        #     print("❌ Errore durante la conversione")
        
        print("ℹ️ Per uso reale, decommentare il codice sopra e fornire path corretti")
        
    except ImportError as e:
        print(f"❌ Errore import: {e}")
        print("💡 Assicurarsi che i moduli preprocessing siano disponibili")


def esempio_analisi_qualita():
    """Esempio di analisi qualità di un'immagine"""
    print("\n🔍 === ESEMPIO: Analisi Qualità Immagine ===")
    
    try:
        from manga_quality_analyzer import MangaQualityAnalyzer
        from test_manga_preprocessing import create_test_image
        
        # Crea analyzer
        analyzer = MangaQualityAnalyzer()
        
        # Crea immagine test per demo
        test_image_path = create_test_image(640, 480)
        print(f"📸 Usando immagine test: {os.path.basename(test_image_path)}")
        
        # Analisi qualità completa
        print("\n📊 Analizzando qualità...")
        metrics = analyzer.calculate_overall_quality_metrics(test_image_path)
        
        print(f"   📏 Nitidezza: {metrics.sharpness_score:.2f}/1.00")
        print(f"   🔳 Contrasto: {metrics.contrast_score:.2f}/1.00")
        print(f"   🔇 Rumore: {metrics.noise_level:.2f}/1.00")
        print(f"   ✏️ Line art: {metrics.line_art_quality:.2f}/1.00")
        print(f"   🎯 Score totale: {metrics.overall_score:.2f}/1.00")
        print(f"   📈 Probabilità successo: {metrics.success_probability:.1%}")
        
        # Suggerimenti ottimizzazione
        print("\n🔧 Analizzando suggerimenti...")
        suggestions = analyzer.suggest_optimizations(test_image_path)
        
        grade = suggestions['quality_assessment']['overall_grade']
        print(f"   🏆 Grado qualità: {grade}")
        
        if suggestions['preprocessing_recommendations']:
            print("   💡 Preprocessing consigliato:")
            for rec_name, rec_data in suggestions['preprocessing_recommendations'].items():
                if rec_data['recommended']:
                    print(f"      • {rec_name}: {rec_data['reason']}")
        
        if suggestions['parameter_adjustments']:
            print("   ⚙️ Parametri ToonCrafter suggeriti:")
            for param_name, param_data in suggestions['parameter_adjustments'].items():
                print(f"      • {param_name}: {param_data['recommended_value']} - {param_data['reason']}")
        
        # Cleanup
        os.remove(test_image_path)
        
    except ImportError as e:
        print(f"❌ Errore import: {e}")
    except Exception as e:
        print(f"❌ Errore: {e}")


def esempio_preprocessing_solo():
    """Esempio di solo preprocessing senza ToonCrafter"""
    print("\n🎨 === ESEMPIO: Solo Preprocessing ===")
    
    try:
        from manga_preprocessor import MangaPreprocessor
        from test_manga_preprocessing import create_test_image
        
        # Crea preprocessor
        preprocessor = MangaPreprocessor()
        
        # Crea immagine test
        input_path = create_test_image(512, 320)
        output_path = tempfile.mktemp(suffix='_preprocessed_example.png')
        
        print(f"📸 Input: {os.path.basename(input_path)}")
        print(f"💾 Output: {os.path.basename(output_path)}")
        
        # Opzioni preprocessing personalizzate
        enhancement_options = {
            'contrast_enhancement': True,
            'line_art_sharpening': True,
            'noise_reduction': True,
            'tone_normalization': True,
            'edge_reinforcement': True,
            'preserve_screentones': True
        }
        
        print("\n🔄 Applicando preprocessing...")
        results = preprocessor.preprocess_manga_panel(
            input_path,
            output_path,
            enhancement_options
        )
        
        if results['success']:
            print("✅ Preprocessing completato!")
            print(f"   📋 Steps applicati: {', '.join(results['processing_steps'])}")
            
            # Mostra risultati analisi
            analysis = results['analysis']
            print(f"   📏 Complessità line art: {analysis['line_art_complexity']:.2f}")
            
            content = analysis['content_classification']
            main_content = max(content, key=content.get)
            print(f"   🎭 Contenuto principale: {main_content} ({content[main_content]:.2f})")
            
            print(f"   📦 Panel rilevati: {len(analysis['panel_borders'])}")
            print(f"   💬 Balloon di testo: {len(analysis['text_balloons'])}")
            
            # Mostra info file
            input_size = os.path.getsize(input_path)
            output_size = os.path.getsize(output_path)
            print(f"   📊 Dimensione file: {input_size} → {output_size} bytes")
            
        else:
            print("❌ Preprocessing fallito")
        
        # Cleanup
        os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
            
    except ImportError as e:
        print(f"❌ Errore import: {e}")
    except Exception as e:
        print(f"❌ Errore: {e}")


def esempio_preset_configurazioni():
    """Esempio dei preset di configurazione disponibili"""
    print("\n⚙️ === ESEMPIO: Preset Configurazioni ===")
    
    try:
        from colab_tooncrafter_runner import create_preprocessing_presets
        
        presets = create_preprocessing_presets()
        
        print("📋 Preset di preprocessing disponibili:")
        
        for preset_name, preset_config in presets.items():
            print(f"\n🔧 {preset_name.upper()}:")
            
            enabled_features = [feature for feature, enabled in preset_config.items() if enabled]
            disabled_features = [feature for feature, enabled in preset_config.items() if not enabled]
            
            if enabled_features:
                print(f"   ✅ Abilitato: {', '.join(enabled_features)}")
            if disabled_features:
                print(f"   ❌ Disabilitato: {', '.join(disabled_features)}")
            
            # Descrizione uso
            descriptions = {
                'default': "Configurazione bilanciata per la maggior parte dei manga",
                'high_quality': "Per immagini già di alta qualità (riduce processing aggressivo)",
                'low_quality_scan': "Per scansioni di bassa qualità (massimo miglioramento)",
                'digital_manga': "Per manga digitali (processing minimale)",
                'action_sequence': "Ottimizzato per scene d'azione dinamiche"
            }
            
            if preset_name in descriptions:
                print(f"   💡 Uso: {descriptions[preset_name]}")
        
        print(f"\n🎯 Utilizzo preset:")
        print(f"run_with_manga_preprocessing(")
        print(f"    preprocessing_preset='default',  # Scegli preset")
        print(f"    ...)")
        
    except ImportError as e:
        print(f"❌ Errore import: {e}")
    except Exception as e:
        print(f"❌ Errore: {e}")


def esempio_runner_avanzato():
    """Esempio di utilizzo avanzato del runner con controllo completo"""
    print("\n🎛️ === ESEMPIO: Runner Avanzato ===")
    
    try:
        from colab_tooncrafter_runner import ColabMangaToonCrafterRunner
        
        # Inizializza runner
        runner = ColabMangaToonCrafterRunner("path/to/ToonCrafter", enable_preprocessing=True)
        
        print("✅ Runner inizializzato con preprocessing abilitato")
        
        # Parametri ToonCrafter personalizzati
        custom_params = {
            'unconditional_guidance_scale': 10.0,  # Guidance più alto per qualità
            'ddim_steps': 50,                      # Steps bilanciati
            'video_length': 16,                    # 16 frame output
            'frame_stride': 6,                     # Movimento moderato
            'guidance_rescale': 0.7                # Rescale standard
        }
        
        # Opzioni preprocessing personalizzate
        preprocessing_options = {
            'contrast_enhancement': True,
            'line_art_sharpening': True,    # Importante per line art
            'noise_reduction': True,        # Rimuovi artefatti scansione
            'tone_normalization': True,     # Uniforma toni
            'edge_reinforcement': True,     # Preserva stile manga
            'preserve_screentones': True    # Mantieni retini
        }
        
        print("\n⚙️ Configurazione personalizzata:")
        print(f"   🎛️ Parametri ToonCrafter: {custom_params}")
        print(f"   🎨 Preprocessing: {preprocessing_options}")
        
        print("\n💡 Per esecuzione reale:")
        print("success = runner.run_custom_parameters_conversion(")
        print("    base_name='manga_scene_01',")
        print("    prompt='dramatic manga to anime transformation',")
        print("    custom_params=custom_params,")
        print("    output_dir='output',")
        print("    input_dir='input',")
        print("    enable_manga_preprocessing=True,")
        print("    preprocessing_options=preprocessing_options")
        print(")")
        
        # Nota: Decommentare per uso reale con ToonCrafter configurato
        
    except ImportError as e:
        print(f"❌ Errore import: {e}")
    except Exception as e:
        print(f"❌ Errore: {e}")


def main():
    """Esegue tutti gli esempi"""
    print("🎌 ESEMPI DI UTILIZZO PREPROCESSING MANGA per ToonCrafter")
    print("=" * 70)
    
    esempi = [
        ("Utilizzo Base", esempio_base),
        ("Analisi Qualità", esempio_analisi_qualita),
        ("Solo Preprocessing", esempio_preprocessing_solo),
        ("Preset Configurazioni", esempio_preset_configurazioni),
        ("Runner Avanzato", esempio_runner_avanzato)
    ]
    
    for nome, funzione in esempi:
        try:
            funzione()
        except Exception as e:
            print(f"\n❌ Errore in esempio '{nome}': {e}")
    
    print("\n" + "=" * 70)
    print("🎯 RIEPILOGO:")
    print("• Usa run_with_manga_preprocessing() per facilità d'uso")
    print("• Abilita sempre il preprocessing per migliori risultati")
    print("• Scegli preset appropriati per il tipo di manga")
    print("• Controlla i suggerimenti dell'analisi qualità")
    print("\n📖 Per documentazione completa: README_MANGA_PREPROCESSING.md")


if __name__ == "__main__":
    main()