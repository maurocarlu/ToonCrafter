#!/usr/bin/env python3
"""
Script ottimizzato per eseguire ToonCrafter su panel manga con parametri avanzati
VERSIONE GOOGLE COLAB - ottimizzato per l'ambiente Colab
Include modulo di preprocessing manga-specifico
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path
import time
import tempfile
import shutil
from PIL import Image  # ✅ AGGIUNTO PER RESCALING
from typing import Dict, Optional, Tuple

# Import moduli preprocessing manga
try:
    from .manga_preprocessor import MangaPreprocessor
    from .manga_quality_analyzer import MangaQualityAnalyzer
except ImportError:
    # Fallback per import relativi
    try:
        from manga_preprocessor import MangaPreprocessor
        from manga_quality_analyzer import MangaQualityAnalyzer
    except ImportError:
        print("⚠️ Warning: Moduli preprocessing manga non disponibili. Usando solo rescaling base.")
        MangaPreprocessor = None
        MangaQualityAnalyzer = None

class ColabMangaToonCrafterRunner:
    """
    Runner ottimizzato per ToonCrafter specifico per panel manga su Google Colab
    Include funzionalità di preprocessing manga-specifico
    """
    
    def __init__(self, tooncrafter_path: str, enable_preprocessing: bool = True):
        self.tooncrafter_path = Path(tooncrafter_path)
        self.enable_preprocessing = enable_preprocessing
        
        # Inizializza moduli preprocessing se disponibili
        if enable_preprocessing and MangaPreprocessor is not None:
            self.manga_preprocessor = MangaPreprocessor()
            self.quality_analyzer = MangaQualityAnalyzer()
            print("✅ Preprocessing manga abilitato")
        else:
            self.manga_preprocessor = None
            self.quality_analyzer = None
            if enable_preprocessing:
                print("⚠️ Preprocessing manga non disponibile - usando solo rescaling")
    
    def analyze_image_quality(self, image_path: str, verbose: bool = True) -> Optional[Dict]:
        """
        Analizza la qualità dell'immagine e suggerisce ottimizzazioni
        
        Args:
            image_path: Percorso dell'immagine da analizzare
            verbose: Se stampare informazioni dettagliate
            
        Returns:
            Dizionario con analisi qualità o None se non disponibile
        """
        if self.quality_analyzer is None:
            return None
        
        try:
            metrics = self.quality_analyzer.calculate_overall_quality_metrics(image_path)
            suggestions = self.quality_analyzer.suggest_optimizations(image_path)
            
            if verbose:
                print(f"📊 Analisi qualità: {os.path.basename(image_path)}")
                print(f"   🎯 Score complessivo: {metrics.overall_score:.2f}/1.00")
                print(f"   📈 Probabilità successo: {metrics.success_probability:.1%}")
                print(f"   🏆 Grado: {suggestions['quality_assessment']['overall_grade']}")
                
                if suggestions['quality_assessment']['main_issues']:
                    print(f"   ⚠️ Problemi: {', '.join(suggestions['quality_assessment']['main_issues'])}")
            
            return {
                'metrics': metrics,
                'suggestions': suggestions
            }
            
        except Exception as e:
            if verbose:
                print(f"⚠️ Errore analisi qualità: {e}")
            return None
    
    def preprocess_manga_image(self, 
                              image_path: str, 
                              output_path: str,
                              preprocessing_options: Optional[Dict] = None,
                              quality_analysis: Optional[Dict] = None) -> bool:
        """
        Applica preprocessing manga-specifico all'immagine
        
        Args:
            image_path: Percorso immagine input
            output_path: Percorso immagine output
            preprocessing_options: Opzioni di preprocessing personalizzate
            quality_analysis: Analisi qualità pre-calcolata
            
        Returns:
            True se successo, False altrimenti
        """
        if self.manga_preprocessor is None:
            # Fallback al resize normale
            return self.resize_image_to_tooncrafter_format(image_path, output_path)
        
        try:
            # Usa opzioni di default intelligenti basate su analisi qualità
            if preprocessing_options is None:
                preprocessing_options = self._get_intelligent_preprocessing_options(quality_analysis)
            
            print(f"🎨 Preprocessing manga: {os.path.basename(image_path)}")
            
            # Applica preprocessing completo
            results = self.manga_preprocessor.preprocess_manga_panel(
                image_path, 
                output_path, 
                preprocessing_options
            )
            
            if results['success']:
                print(f"   ✅ Preprocessing completato: {', '.join(results['processing_steps'])}")
                
                # Mostra info analisi se disponibile
                analysis = results.get('analysis', {})
                if 'line_art_complexity' in analysis:
                    print(f"   📏 Complessità line art: {analysis['line_art_complexity']:.2f}")
                if 'content_classification' in analysis:
                    content = analysis['content_classification']
                    main_content = max(content, key=content.get)
                    print(f"   🎭 Contenuto principale: {main_content} ({content[main_content]:.2f})")
                
                return True
            else:
                print(f"   ❌ Preprocessing fallito")
                return False
                
        except Exception as e:
            print(f"   ❌ Errore preprocessing: {e}")
            # Fallback al resize normale
            return self.resize_image_to_tooncrafter_format(image_path, output_path)
    
    def _get_intelligent_preprocessing_options(self, quality_analysis: Optional[Dict] = None) -> Dict:
        """
        Genera opzioni di preprocessing intelligenti basate sull'analisi qualità
        
        Args:
            quality_analysis: Risultati analisi qualità
            
        Returns:
            Dizionario con opzioni di preprocessing ottimizzate
        """
        # Opzioni di default
        options = {
            'contrast_enhancement': True,
            'line_art_sharpening': True,
            'noise_reduction': True,
            'tone_normalization': True,
            'edge_reinforcement': True,
            'preserve_screentones': True
        }
        
        # Ottimizza basato su analisi qualità
        if quality_analysis and 'suggestions' in quality_analysis:
            suggestions = quality_analysis['suggestions']
            
            # Disabilita sharpening se già abbastanza nitido
            if quality_analysis['metrics'].sharpness_score > 0.8:
                options['line_art_sharpening'] = False
            
            # Aumenta noise reduction se necessario
            if quality_analysis['metrics'].noise_level > 0.4:
                options['noise_reduction'] = True
            
            # Applica suggerimenti specifici
            if 'preprocessing_recommendations' in suggestions:
                recs = suggestions['preprocessing_recommendations']
                
                if 'noise_reduction' in recs and recs['noise_reduction']['recommended']:
                    options['noise_reduction'] = True
                
                if 'contrast_enhancement' in recs and recs['contrast_enhancement']['recommended']:
                    options['contrast_enhancement'] = True
        
        return options
    
    def resize_image_to_tooncrafter_format(self, image_path, output_path, target_width=512, target_height=320):
        """
        📐 Ridimensiona immagine al formato richiesto da ToonCrafter (512x320)
        """
        try:
            with Image.open(image_path) as img:
                # ✅ DEBUG: Mostra info PRIMA del rescaling
                original_size = img.size
                original_mode = img.mode
                print(f"   📸 PRIMA - File: {os.path.basename(image_path)}")
                print(f"   📐 PRIMA - Dimensioni: {original_size[0]}x{original_size[1]} ({original_mode})")
                
                # Converti in RGB se necessario
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    print(f"   🔄 Convertito da {original_mode} a RGB")
                
                # Ridimensiona 
                img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                # ✅ DEBUG: Verifica dimensioni DOPO rescaling
                new_size = img_resized.size
                new_mode = img_resized.mode
                print(f"   📐 DOPO - Dimensioni: {new_size[0]}x{new_size[1]} ({new_mode})")
                
                # Salva immagine ridimensionata
                img_resized.save(output_path, 'PNG', quality=95)
                
                # ✅ DEBUG: Verifica file salvato
                saved_size = os.path.getsize(output_path) / 1024  # KB
                print(f"   💾 Salvato: {os.path.basename(output_path)} ({saved_size:.1f} KB)")
                
                # ✅ VERIFICA FINALE: Rileggi il file salvato per conferma
                with Image.open(output_path) as saved_img:
                    final_size = saved_img.size
                    final_mode = saved_img.mode
                    print(f"   ✅ VERIFICATO - Dimensioni finali: {final_size[0]}x{final_size[1]} ({final_mode})")
                    
                    if final_size == (target_width, target_height):
                        print(f"   ✅ Rescaling completato con successo!")
                    else:
                        print(f"   ⚠️ Warning: Dimensioni non corrispondono al target!")
            
            return True
                
        except Exception as e:
            print(f"   ❌ Errore ridimensionamento: {e}")
            return False
    
    def run_custom_parameters_conversion(self, 
                                        base_name, 
                                        prompt, 
                                        custom_params, 
                                        output_dir, 
                                        input_dir,
                                        enable_manga_preprocessing: bool = True,
                                        preprocessing_options: Optional[Dict] = None):
        """
        🎛️ Esecuzione con parametri completamente personalizzati + preprocessing manga + rescaling automatico
        
        Args:
            base_name: Nome base per i file
            prompt: Prompt di testo per ToonCrafter
            custom_params: Parametri personalizzati ToonCrafter
            output_dir: Directory di output
            input_dir: Directory di input
            enable_manga_preprocessing: Se abilitare preprocessing manga-specifico
            preprocessing_options: Opzioni personalizzate per preprocessing
        """
        print(f"\n🎬 === INIZIANDO CONVERSIONE: {base_name} ===")
        
        # Trova frame1 e frame3
        frame1_path = None
        frame3_path = None
        
        for file in os.listdir(input_dir):
            if file.startswith(f"{base_name}_frame1"):
                frame1_path = os.path.join(input_dir, file)
            elif file.startswith(f"{base_name}_frame3"):
                frame3_path = os.path.join(input_dir, file)
        
        if not frame1_path or not frame3_path:
            print(f"❌ Frame non trovati per {base_name}")
            return False
        
        print(f"📸 Input: {os.path.basename(frame1_path)} → {os.path.basename(frame3_path)}")
        
        # ✅ ANALISI QUALITÀ (se preprocessing abilitato)
        quality_analysis_frame1 = None
        quality_analysis_frame3 = None
        
        if enable_manga_preprocessing and self.quality_analyzer is not None:
            print(f"🔍 Analizzando qualità immagini...")
            quality_analysis_frame1 = self.analyze_image_quality(frame1_path, verbose=True)
            quality_analysis_frame3 = self.analyze_image_quality(frame3_path, verbose=True)
            
            # Suggerisci parametri ToonCrafter ottimizzati se la qualità è bassa
            if quality_analysis_frame1 and quality_analysis_frame1['metrics'].success_probability < 0.6:
                self._suggest_tooncrafter_parameters(quality_analysis_frame1, custom_params)
        
        # ✅ CREA DIRECTORY TEMPORANEA CON PROCESSING AVANZATO
        temp_input_dir = tempfile.mkdtemp(prefix=f"tooncrafter_{base_name}_")
        
        try:
            # ✅ PREPROCESSING + RESCALING
            temp_frame1 = os.path.join(temp_input_dir, os.path.basename(frame1_path))
            temp_frame3 = os.path.join(temp_input_dir, os.path.basename(frame3_path))
            
            print(f"📁 Directory temporanea: {temp_input_dir}")
            
            if enable_manga_preprocessing and self.manga_preprocessor is not None:
                print(f"🎨 Applicando preprocessing manga + ridimensionamento a 512x320...")
                
                # Preprocessing + resize per frame1
                success1 = self._preprocess_and_resize(
                    frame1_path, temp_frame1, preprocessing_options, quality_analysis_frame1
                )
                
                # Preprocessing + resize per frame3  
                success3 = self._preprocess_and_resize(
                    frame3_path, temp_frame3, preprocessing_options, quality_analysis_frame3
                )
                
                if not success1 or not success3:
                    print(f"❌ Errore durante preprocessing")
                    return False
                    
            else:
                print(f"📐 Ridimensionando immagini a 512x320...")
                
                # Solo ridimensionamento standard
                if not self.resize_image_to_tooncrafter_format(frame1_path, temp_frame1):
                    print(f"❌ Errore ridimensionamento {frame1_path}")
                    return False
                
                if not self.resize_image_to_tooncrafter_format(frame3_path, temp_frame3):
                    print(f"❌ Errore ridimensionamento {frame3_path}")
                    return False
            
            print(f"✅ Immagini processate e pronte per ToonCrafter")
            
            # ✅ COMANDO TOONCRAFTER CON DIRECTORY TEMPORANEA
            inference_script = self.tooncrafter_path / "scripts" / "evaluation" / "inference.py"
            base_config = self.tooncrafter_path / "configs" / "inference_512_v1.0.yaml"
            checkpoint = self.tooncrafter_path / "checkpoints" / "tooncrafter_512_interp_v1" / "model.ckpt"
            
            # Crea nome output con seed
            seed = 123
            name = f"tooncrafter_{base_name}_seed{seed}"
            final_output_dir = os.path.join(output_dir, name)
            
            cmd = [
                "python3", str(inference_script),
                "--seed", str(seed),
                "--ckpt_path", str(checkpoint),
                "--config", str(base_config),
                "--savedir", str(final_output_dir),
                "--n_samples", "1",
                "--bs", "1", 
                "--height", "320", 
                "--width", "512",
                "--unconditional_guidance_scale", str(custom_params['unconditional_guidance_scale']),
                "--ddim_steps", str(custom_params['ddim_steps']),
                "--ddim_eta", "1.0",
                "--prompt_dir", temp_input_dir,  # ✅ USA DIRECTORY TEMPORANEA
                "--text_input",
                "--video_length", str(custom_params['video_length']),
                "--frame_stride", str(custom_params['frame_stride']),
                "--timestep_spacing", "uniform_trailing",
                "--guidance_rescale", str(custom_params['guidance_rescale']),
                "--perframe_ae",
                "--interp"
            ]
            
            # OUTPUT DETTAGLIATO
            print(f"📝 PROMPT: '{prompt}'")
            print(f"🎛️ PARAMETRI:")
            print(f"   • frame_stride: {custom_params['frame_stride']}")
            print(f"   • ddim_steps: {custom_params['ddim_steps']}")
            print(f"   • guidance_scale: {custom_params['unconditional_guidance_scale']}")
            print(f"   • guidance_rescale: {custom_params['guidance_rescale']}")
            print(f"   • video_length: {custom_params['video_length']}")
            preprocessing_status = "Abilitato" if enable_manga_preprocessing and self.manga_preprocessor else "Disabilitato"
            print(f"   • preprocessing_manga: {preprocessing_status}")
            print(f"📁 Input: {temp_input_dir} (512x320)")
            print(f"📁 Output: {final_output_dir}")
            print(f"🚀 Avviando ToonCrafter...")
            
            # DEBUG: Mostra comando completo
            print(f"🔧 Comando: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.tooncrafter_path)
            
            if result.returncode == 0:
                print(f"✅ Conversione completata per '{base_name}'!")
                
                # Verifica file generati
                generated_files = []
                if os.path.exists(final_output_dir):
                    for root, dirs, files in os.walk(final_output_dir):
                        for file in files:
                            if file.endswith(('.mp4', '.avi', '.mov')):
                                video_path = os.path.join(root, file)
                                size_mb = os.path.getsize(video_path) / (1024*1024)
                                generated_files.append(video_path)
                                print(f"📹 Video generato: {os.path.basename(video_path)} ({size_mb:.1f} MB)")
                
                if not generated_files:
                    print("⚠️ Nessun video trovato nella directory di output!")
                
                return True
            else:
                print(f"❌ ToonCrafter fallito!")
                print(f"   Return code: {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr}")
                if result.stdout:
                    print(f"   Output: {result.stdout}")
                return False
                
        except Exception as e:
            print(f"❌ Errore durante esecuzione: {e}")
            return False
        
        finally:
            # ✅ PULISCI DIRECTORY TEMPORANEA
            try:
                shutil.rmtree(temp_input_dir)
                print(f"🗑️ Directory temporanea pulita")
            except:
                pass
    
    def _preprocess_and_resize(self, 
                              input_path: str, 
                              output_path: str, 
                              preprocessing_options: Optional[Dict] = None,
                              quality_analysis: Optional[Dict] = None) -> bool:
        """
        Applica preprocessing manga e ridimensiona a formato ToonCrafter
        
        Args:
            input_path: Percorso immagine input
            output_path: Percorso immagine output
            preprocessing_options: Opzioni preprocessing
            quality_analysis: Analisi qualità
            
        Returns:
            True se successo, False altrimenti
        """
        try:
            # Step 1: Preprocessing manga
            temp_preprocessed = tempfile.mktemp(suffix='.png')
            
            success = self.preprocess_manga_image(
                input_path, 
                temp_preprocessed, 
                preprocessing_options, 
                quality_analysis
            )
            
            if not success:
                return False
            
            # Step 2: Ridimensiona a formato ToonCrafter
            success = self.resize_image_to_tooncrafter_format(temp_preprocessed, output_path)
            
            # Pulisci file temporaneo
            if os.path.exists(temp_preprocessed):
                os.remove(temp_preprocessed)
            
            return success
            
        except Exception as e:
            print(f"   ❌ Errore _preprocess_and_resize: {e}")
            return False
    
    def _suggest_tooncrafter_parameters(self, quality_analysis: Dict, custom_params: Dict):
        """
        Suggerisci modifiche ai parametri ToonCrafter basate sull'analisi qualità
        
        Args:
            quality_analysis: Risultati analisi qualità
            custom_params: Parametri attuali (modificati in-place)
        """
        if 'suggestions' not in quality_analysis:
            return
        
        suggestions = quality_analysis['suggestions']
        
        if 'parameter_adjustments' in suggestions:
            adjustments = suggestions['parameter_adjustments']
            
            modified = []
            
            if 'guidance_scale' in adjustments:
                old_value = custom_params['unconditional_guidance_scale']
                custom_params['unconditional_guidance_scale'] = adjustments['guidance_scale']['recommended_value']
                modified.append(f"guidance_scale: {old_value} → {adjustments['guidance_scale']['recommended_value']}")
            
            if 'ddim_steps' in adjustments:
                old_value = custom_params['ddim_steps']
                custom_params['ddim_steps'] = adjustments['ddim_steps']['recommended_value']
                modified.append(f"ddim_steps: {old_value} → {adjustments['ddim_steps']['recommended_value']}")
            
            if 'frame_stride' in adjustments:
                old_value = custom_params['frame_stride']
                custom_params['frame_stride'] = adjustments['frame_stride']['recommended_value']
                modified.append(f"frame_stride: {old_value} → {adjustments['frame_stride']['recommended_value']}")
            
            if modified:
                print(f"🎛️ Parametri ottimizzati automaticamente:")
                for mod in modified:
                    print(f"   • {mod}")
                print(f"   Motivo: Qualità input bassa - ottimizzazione automatica")

# Alias per compatibilità con il notebook
MangaToonCrafterRunner = ColabMangaToonCrafterRunner


# ✅ FUNZIONI DI UTILITÀ PER PREPROCESSING MANGA

def create_preprocessing_presets() -> Dict[str, Dict]:
    """
    Crea preset di preprocessing ottimizzati per diversi tipi di manga
    
    Returns:
        Dizionario con preset predefiniti
    """
    return {
        'default': {
            'contrast_enhancement': True,
            'line_art_sharpening': True,
            'noise_reduction': True,
            'tone_normalization': True,
            'edge_reinforcement': True,
            'preserve_screentones': True
        },
        'high_quality': {
            'contrast_enhancement': True,
            'line_art_sharpening': False,  # Già di alta qualità
            'noise_reduction': False,      # Non necessario
            'tone_normalization': True,
            'edge_reinforcement': True,
            'preserve_screentones': True
        },
        'low_quality_scan': {
            'contrast_enhancement': True,
            'line_art_sharpening': True,
            'noise_reduction': True,       # Importante per scan di bassa qualità
            'tone_normalization': True,
            'edge_reinforcement': True,
            'preserve_screentones': True
        },
        'digital_manga': {
            'contrast_enhancement': False, # Già ottimizzato
            'line_art_sharpening': False,  # Già nitido
            'noise_reduction': False,      # Nessun rumore di scansione
            'tone_normalization': True,
            'edge_reinforcement': True,
            'preserve_screentones': True
        },
        'action_sequence': {
            'contrast_enhancement': True,
            'line_art_sharpening': True,   # Importante per linee dinamiche
            'noise_reduction': True,
            'tone_normalization': True,
            'edge_reinforcement': True,    # Cruciale per scene d'azione
            'preserve_screentones': True
        }
    }


def run_with_manga_preprocessing(tooncrafter_path: str,
                                prompt_dir: str,
                                output_dir: str,
                                config_type: str = "dramatic_change",
                                preprocessing_preset: str = "default",
                                custom_preprocessing: Optional[Dict] = None,
                                enable_quality_analysis: bool = True) -> bool:
    """
    Funzione di convenienza per eseguire ToonCrafter con preprocessing manga
    
    Args:
        tooncrafter_path: Percorso installazione ToonCrafter
        prompt_dir: Directory con immagini e prompt
        output_dir: Directory output
        config_type: Tipo configurazione ToonCrafter
        preprocessing_preset: Preset preprocessing ('default', 'high_quality', etc.)
        custom_preprocessing: Opzioni preprocessing personalizzate
        enable_quality_analysis: Se abilitare analisi qualità
        
    Returns:
        True se successo, False altrimenti
    """
    # Configurazioni ToonCrafter predefinite
    configs = {
        "colab_fast": {
            'unconditional_guidance_scale': 7.5,
            'ddim_steps': 25,
            'video_length': 16,
            'frame_stride': 10,
            'guidance_rescale': 0.7
        },
        "smooth_transition": {
            'unconditional_guidance_scale': 7.5,
            'ddim_steps': 50,
            'video_length': 16,
            'frame_stride': 8,
            'guidance_rescale': 0.7
        },
        "dramatic_change": {
            'unconditional_guidance_scale': 10.0,
            'ddim_steps': 50,
            'video_length': 16,
            'frame_stride': 6,
            'guidance_rescale': 0.7
        },
        "action_sequence": {
            'unconditional_guidance_scale': 12.0,
            'ddim_steps': 75,
            'video_length': 16,
            'frame_stride': 5,
            'guidance_rescale': 0.8
        }
    }
    
    if config_type not in configs:
        print(f"❌ Configurazione '{config_type}' non riconosciuta")
        return False
    
    # Ottieni preset preprocessing
    presets = create_preprocessing_presets()
    if preprocessing_preset not in presets:
        print(f"❌ Preset preprocessing '{preprocessing_preset}' non riconosciuto")
        return False
    
    preprocessing_options = presets[preprocessing_preset]
    if custom_preprocessing:
        preprocessing_options.update(custom_preprocessing)
    
    print(f"🎌 Avviando ToonCrafter con preprocessing manga")
    print(f"   📁 Input: {prompt_dir}")
    print(f"   📁 Output: {output_dir}")
    print(f"   ⚙️ Config: {config_type}")
    print(f"   🎨 Preprocessing: {preprocessing_preset}")
    
    # Inizializza runner
    runner = ColabMangaToonCrafterRunner(tooncrafter_path, enable_preprocessing=True)
    
    # Cerca file di input
    success_count = 0
    total_count = 0
    
    try:
        for file in os.listdir(prompt_dir):
            if file.endswith("_frame1.png") or file.endswith("_frame1.jpg"):
                base_name = file.replace("_frame1.png", "").replace("_frame1.jpg", "")
                
                # Cerca file prompt corrispondente
                prompt_file = os.path.join(prompt_dir, f"{base_name}.txt")
                prompt = "manga to anime style transformation"
                
                if os.path.exists(prompt_file):
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        prompt = f.read().strip()
                
                print(f"\n🎬 Processando: {base_name}")
                total_count += 1
                
                # Esegui conversione con preprocessing
                success = runner.run_custom_parameters_conversion(
                    base_name=base_name,
                    prompt=prompt,
                    custom_params=configs[config_type],
                    output_dir=output_dir,
                    input_dir=prompt_dir,
                    enable_manga_preprocessing=True,
                    preprocessing_options=preprocessing_options
                )
                
                if success:
                    success_count += 1
                    print(f"✅ {base_name} completato con successo")
                else:
                    print(f"❌ {base_name} fallito")
    
    except Exception as e:
        print(f"❌ Errore durante elaborazione: {e}")
        return False
    
    print(f"\n🎉 RIEPILOGO:")
    print(f"   ✅ Successi: {success_count}/{total_count}")
    print(f"   📁 Output salvato in: {output_dir}")
    
    return success_count > 0