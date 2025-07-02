#!/usr/bin/env python3
"""
Test script per il modulo di preprocessing manga
Testa le funzionalità principali senza richiedere ToonCrafter completo
"""

import os
import sys
import tempfile
import numpy as np
from PIL import Image
import cv2

# Aggiungi il path del modulo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from manga_preprocessor import MangaPreprocessor
    from manga_quality_analyzer import MangaQualityAnalyzer
    from colab_tooncrafter_runner import ColabMangaToonCrafterRunner, create_preprocessing_presets
    print("✅ Import moduli riuscito")
except ImportError as e:
    print(f"❌ Errore import: {e}")
    sys.exit(1)


def create_test_image(width: int = 512, height: int = 320) -> str:
    """
    Crea un'immagine test che simula un panel manga
    
    Returns:
        Percorso dell'immagine test creata
    """
    # Crea immagine bianca
    img = Image.new('RGB', (width, height), color='white')
    img_array = np.array(img)
    
    # Aggiungi alcuni elementi tipici del manga
    # 1. Bordi neri (frame del panel)
    cv2.rectangle(img_array, (10, 10), (width-10, height-10), (0, 0, 0), 3)
    
    # 2. Personaggio stilizzato (cerchio per testa + linee per corpo)
    center_x, center_y = width//3, height//3
    cv2.circle(img_array, (center_x, center_y), 30, (0, 0, 0), 2)  # Testa
    cv2.line(img_array, (center_x, center_y+30), (center_x, center_y+100), (0, 0, 0), 2)  # Corpo
    
    # 3. Balloon di testo
    balloon_center = (width*2//3, height//4)
    cv2.ellipse(img_array, balloon_center, (60, 30), 0, 0, 360, (0, 0, 0), 2)
    
    # 4. Linee di azione
    cv2.line(img_array, (center_x+50, center_y), (center_x+120, center_y-20), (0, 0, 0), 1)
    cv2.line(img_array, (center_x+50, center_y+10), (center_x+110, center_y+30), (0, 0, 0), 1)
    
    # 5. Aggiungi un po' di rumore per testare noise reduction
    noise = np.random.randint(0, 20, img_array.shape, dtype=np.uint8)
    img_array = np.clip(img_array.astype(int) + noise.astype(int), 0, 255).astype(np.uint8)
    
    # Salva immagine temporanea
    test_img = Image.fromarray(img_array)
    temp_path = tempfile.mktemp(suffix='_test_manga.png')
    test_img.save(temp_path, 'PNG')
    
    return temp_path


def test_manga_preprocessor():
    """Test del preprocessor manga"""
    print("\n🎨 Testing MangaPreprocessor...")
    
    try:
        # Crea immagine test
        test_image_path = create_test_image()
        print(f"   📸 Immagine test creata: {os.path.basename(test_image_path)}")
        
        # Inizializza preprocessor
        preprocessor = MangaPreprocessor()
        
        # Test preprocessing completo
        output_path = tempfile.mktemp(suffix='_preprocessed.png')
        
        results = preprocessor.preprocess_manga_panel(
            test_image_path, 
            output_path,
            enhancement_options={
                'contrast_enhancement': True,
                'line_art_sharpening': True,
                'noise_reduction': True,
                'tone_normalization': True,
                'edge_reinforcement': True,
                'preserve_screentones': True
            }
        )
        
        if results['success']:
            print("   ✅ Preprocessing completato con successo")
            print(f"   📋 Steps applicati: {', '.join(results['processing_steps'])}")
            
            # Verifica analisi
            analysis = results['analysis']
            print(f"   📊 Line art complexity: {analysis['line_art_complexity']:.2f}")
            print(f"   🎭 Content classification: {analysis['content_classification']}")
            print(f"   📦 Panel borders rilevati: {len(analysis['panel_borders'])}")
            print(f"   💬 Text balloons rilevati: {len(analysis['text_balloons'])}")
            
            # Verifica file output
            if os.path.exists(output_path):
                output_size = os.path.getsize(output_path)
                print(f"   💾 File output: {output_size} bytes")
                os.remove(output_path)
            
        else:
            print("   ❌ Preprocessing fallito")
            return False
        
        # Cleanup
        os.remove(test_image_path)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Errore test preprocessor: {e}")
        return False


def test_quality_analyzer():
    """Test dell'analizzatore di qualità"""
    print("\n📊 Testing MangaQualityAnalyzer...")
    
    try:
        # Crea immagine test
        test_image_path = create_test_image()
        print(f"   📸 Immagine test creata: {os.path.basename(test_image_path)}")
        
        # Inizializza analyzer
        analyzer = MangaQualityAnalyzer()
        
        # Test calcolo metriche
        metrics = analyzer.calculate_overall_quality_metrics(test_image_path)
        
        print(f"   📏 Sharpness score: {metrics.sharpness_score:.2f}")
        print(f"   🔳 Contrast score: {metrics.contrast_score:.2f}")
        print(f"   🔇 Noise level: {metrics.noise_level:.2f}")
        print(f"   ✏️ Line art quality: {metrics.line_art_quality:.2f}")
        print(f"   🎯 Overall score: {metrics.overall_score:.2f}")
        print(f"   📈 Success probability: {metrics.success_probability:.1%}")
        
        # Test suggerimenti
        suggestions = analyzer.suggest_optimizations(test_image_path)
        
        grade = suggestions['quality_assessment']['overall_grade']
        print(f"   🏆 Quality grade: {grade}")
        
        if suggestions['preprocessing_recommendations']:
            recs = list(suggestions['preprocessing_recommendations'].keys())
            print(f"   🔧 Preprocessing consigliato: {', '.join(recs)}")
        
        # Test report generation
        report = analyzer.generate_quality_report(test_image_path)
        print(f"   📄 Report generato: {len(report)} caratteri")
        
        # Cleanup
        os.remove(test_image_path)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Errore test quality analyzer: {e}")
        return False


def test_runner_integration():
    """Test integrazione con ColabMangaToonCrafterRunner"""
    print("\n🎛️ Testing ColabMangaToonCrafterRunner integration...")
    
    try:
        # Crea runner (senza path ToonCrafter reale per test)
        runner = ColabMangaToonCrafterRunner("dummy_path", enable_preprocessing=True)
        
        if runner.manga_preprocessor is not None:
            print("   ✅ Preprocessor integrato correttamente")
        else:
            print("   ❌ Preprocessor non disponibile")
            return False
            
        if runner.quality_analyzer is not None:
            print("   ✅ Quality analyzer integrato correttamente")
        else:
            print("   ❌ Quality analyzer non disponibile")
            return False
        
        # Test analisi qualità
        test_image_path = create_test_image()
        
        quality_analysis = runner.analyze_image_quality(test_image_path, verbose=False)
        
        if quality_analysis:
            print("   ✅ Analisi qualità funzionante")
            metrics = quality_analysis['metrics']
            print(f"   📊 Score: {metrics.overall_score:.2f}")
        else:
            print("   ❌ Analisi qualità fallita")
        
        # Test preprocessing 
        output_path = tempfile.mktemp(suffix='_runner_test.png')
        success = runner.preprocess_manga_image(test_image_path, output_path)
        
        if success and os.path.exists(output_path):
            print("   ✅ Preprocessing via runner funzionante")
            os.remove(output_path)
        else:
            print("   ❌ Preprocessing via runner fallito")
        
        # Cleanup
        os.remove(test_image_path)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Errore test runner integration: {e}")
        return False


def test_preprocessing_presets():
    """Test dei preset di preprocessing"""
    print("\n⚙️ Testing preprocessing presets...")
    
    try:
        presets = create_preprocessing_presets()
        
        expected_presets = ['default', 'high_quality', 'low_quality_scan', 'digital_manga', 'action_sequence']
        
        for preset_name in expected_presets:
            if preset_name in presets:
                preset = presets[preset_name]
                print(f"   ✅ Preset '{preset_name}': {len(preset)} opzioni")
            else:
                print(f"   ❌ Preset '{preset_name}' mancante")
                return False
        
        # Verifica struttura preset
        sample_preset = presets['default']
        expected_options = [
            'contrast_enhancement', 'line_art_sharpening', 'noise_reduction',
            'tone_normalization', 'edge_reinforcement', 'preserve_screentones'
        ]
        
        for option in expected_options:
            if option not in sample_preset:
                print(f"   ❌ Opzione '{option}' mancante nel preset default")
                return False
        
        print("   ✅ Tutti i preset configurati correttamente")
        return True
        
    except Exception as e:
        print(f"   ❌ Errore test presets: {e}")
        return False


def main():
    """Esegue tutti i test"""
    print("🧪 AVVIANDO TEST MODULI PREPROCESSING MANGA")
    print("=" * 60)
    
    tests = [
        ("Manga Preprocessor", test_manga_preprocessor),
        ("Quality Analyzer", test_quality_analyzer),
        ("Runner Integration", test_runner_integration),
        ("Preprocessing Presets", test_preprocessing_presets)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 Test: {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"🎯 RISULTATI TEST: {passed}/{total} passati")
    
    if passed == total:
        print("🎉 Tutti i test sono passati! Moduli pronti per l'uso.")
        return True
    else:
        print("⚠️ Alcuni test sono falliti. Controllare l'implementazione.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)