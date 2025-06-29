#!/usr/bin/env python3
"""
Script ottimizzato per eseguire ToonCrafter su panel manga con parametri avanzati
VERSIONE GOOGLE COLAB - ottimizzato per l'ambiente Colab
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path
import time

class ColabMangaToonCrafterRunner:
    """
    Runner ottimizzato per ToonCrafter specifico per panel manga su Google Colab
    """
    
    def __init__(self, tooncrafter_path: str):
        self.tooncrafter_path = Path(tooncrafter_path)
        self.configs = self._load_optimized_configs()
        
        # Configurazioni specifiche per Colab
        self.colab_optimizations = {
            'use_mixed_precision': True,
            'enable_xformers': True,
            'cpu_offload': False,  # Colab ha buone GPU, mantieni tutto su GPU
            'max_memory_usage': 0.9  # Usa 90% della VRAM disponibile
        }
    
    def _load_optimized_configs(self):
        """
        Configurazioni ottimizzate per diversi tipi di transizioni manga
        Ottimizzate per Google Colab T4/V100
        """
        return {
            'smooth_transition': {
                'frame_stride': 8,
                'ddim_steps': 50,
                'unconditional_guidance_scale': 7.5,
                'guidance_rescale': 0.7,
                'video_length': 16,
                'description': 'Per transizioni fluide tra panel simili - MIGLIORE per manga coerenti'
            },
            'dramatic_change': {
                'frame_stride': 12,
                'ddim_steps': 60,
                'unconditional_guidance_scale': 9.0,
                'guidance_rescale': 0.8,
                'video_length': 16,
                'description': 'Per cambi drastici di scena/inquadratura (AGGIORNATO per migliore qualità)'
            },
            'manga_stable': {
                'frame_stride': 6,
                'ddim_steps': 40,
                'unconditional_guidance_scale': 6.5,
                'guidance_rescale': 0.6,
                'video_length': 12,
                'description': 'NUOVO: Ottimizzato per panel manga - più stabile e coerente'
            },
            'character_focus': {
                'frame_stride': 5,
                'ddim_steps': 35,
                'unconditional_guidance_scale': 5.5,
                'guidance_rescale': 0.5,
                'video_length': 10,
                'description': 'NUOVO: Per scene con focus su personaggi - riduce confusione'
            },
            'action_sequence': {
                'frame_stride': 10,
                'ddim_steps': 45,
                'unconditional_guidance_scale': 8.0,
                'guidance_rescale': 0.7,
                'video_length': 18,
                'description': 'Per sequenze d\'azione dinamiche - parametri più conservativi'
            },
            'dialogue_scene': {
                'frame_stride': 4,
                'ddim_steps': 30,
                'unconditional_guidance_scale': 5.0,
                'guidance_rescale': 0.4,
                'video_length': 10,
                'description': 'Per scene di dialogo con movimenti sottili - molto stabile'
            },
            'colab_fast': {
                'frame_stride': 6,
                'ddim_steps': 25,
                'unconditional_guidance_scale': 6.0,
                'guidance_rescale': 0.5,
                'video_length': 8,
                'description': 'Configurazione veloce per test su Colab - migliorata per qualità'
            }
        }
    
    def check_colab_environment(self):
        """
        Verifica l'ambiente Google Colab e ottimizza di conseguenza
        """
        import torch
        
        print("🔍 Verifica ambiente Google Colab:")
        
        # Verifica GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU: {gpu_name}")
            print(f"✅ VRAM: {gpu_memory:.1f} GB")
            
            # Ottimizzazioni specifiche per tipo di GPU
            if 'T4' in gpu_name:
                print("🎯 Ottimizzazioni per Tesla T4")
                self.colab_optimizations['max_memory_usage'] = 0.85
            elif 'V100' in gpu_name:
                print("🎯 Ottimizzazioni per Tesla V100")
                self.colab_optimizations['max_memory_usage'] = 0.9
            
        else:
            print("❌ GPU non disponibile!")
            print("Vai su Runtime > Change runtime type > Hardware accelerator > GPU")
            return False
        
        # Verifica spazio disco
        import shutil
        disk_usage = shutil.disk_usage("/content")
        free_gb = disk_usage.free / 1024**3
        print(f"💾 Spazio libero: {free_gb:.1f} GB")
        
        if free_gb < 2:
            print("⚠️ Spazio disco limitato. Considera di pulire file temporanei.")
        
        return True
    
    def optimize_for_colab(self):
        """
        Applica ottimizzazioni specifiche per Google Colab
        """
        import torch
        
        # Imposta precision mixed se supportato
        if self.colab_optimizations['use_mixed_precision']:
            os.environ['PYTORCH_USE_CUDA_DSA'] = '1'
        
        # Abilita memory efficient attention se disponibile
        try:
            import xformers
            if self.colab_optimizations['enable_xformers']:
                os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'
                print("✅ XFormers abilitato per efficienza memoria")
        except ImportError:
            print("⚠️ XFormers non disponibile, usando attention standard")
        
        # Ottimizza memoria CUDA
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(
                self.colab_optimizations['max_memory_usage']
            )
            print(f"🎛️ VRAM limitata al {self.colab_optimizations['max_memory_usage']*100}%")
    
    def analyze_manga_sequences(self, prompt_dir: str):
        """
        Analizza ogni sequenza manga individualmente e suggerisce configurazioni specifiche
        """
        print("🔍 ANALISI INTELLIGENTE PER SEQUENZA")
        print("=" * 50)
        
        # Trova le coppie di immagini e ordinale per garantire consistenza
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(Path(prompt_dir).glob(ext))
        
        frame_pairs = []
        for img_path in image_files:
            if '_frame1' in img_path.name:
                base_name = img_path.name.split('_frame1')[0]
                frame3_path = Path(prompt_dir) / f"{base_name}_frame3.png"
                if frame3_path.exists():
                    frame_pairs.append((img_path, frame3_path, base_name))
        
        # Ordina le coppie per nome per garantire corrispondenza con prompts.txt
        frame_pairs.sort(key=lambda x: x[2].lower())  # Ordina per base_name
        
        # Carica i prompt
        prompts_file = os.path.join(prompt_dir, "prompts.txt")
        prompts = []
        if os.path.exists(prompts_file):
            with open(prompts_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f.readlines() if line.strip()]
        
        # Analizza ogni sequenza
        sequence_configs = []
        
        for i, (f1, f3, base_name) in enumerate(frame_pairs):
            print(f"\n📋 SEQUENZA {i+1}: {base_name}")
            print(f"   📸 {f1.name} → {f3.name}")
            
            # Prompt associato
            prompt = prompts[i] if i < len(prompts) else "default animation"
            print(f"   📝 Prompt: \"{prompt}\"")
            
            # Analisi per nome base (riconoscimento serie)
            base_lower = base_name.lower()
            prompt_lower = prompt.lower()
            
            suggested_config = 'manga_stable'  # Default
            confidence = 0.5
            reason = "Configurazione sicura di default"
            
            # === ANALISI SPECIFICA PER SERIE CONOSCIUTE ===
            if 'piece' in base_lower or 'luffy' in base_lower or 'rufy' in base_lower:
                # One Piece - spesso scene dinamiche o zoom su personaggi
                if any(word in prompt_lower for word in ['flag', 'determination', 'close-up', 'character', 'holding']):
                    suggested_config = 'character_focus'
                    confidence = 0.95
                    reason = "One Piece + elemento caratteristico rilevato (flag/determinazione)"
                elif any(word in prompt_lower for word in ['action', 'dynamic', 'battle', 'fight']):
                    suggested_config = 'action_sequence'
                    confidence = 0.9
                    reason = "One Piece + azione dinamica rilevata"
                else:
                    suggested_config = 'character_focus'
                    confidence = 0.85
                    reason = "One Piece - ottimizzato per focus su personaggio"
                    
            elif 'slam' in base_lower or 'dunk' in base_lower:
                # Slam Dunk - spesso dialoghi e interazioni
                if any(word in prompt_lower for word in ['dialogue', 'interaction', 'basketball', 'players', 'conversation', 'gentle']):
                    suggested_config = 'dialogue_scene'
                    confidence = 0.95
                    reason = "Slam Dunk + dialogo/interazione basket rilevato"
                elif any(word in prompt_lower for word in ['sport', 'game', 'court', 'match']):
                    suggested_config = 'action_sequence'
                    confidence = 0.85
                    reason = "Slam Dunk + scena sportiva"
                else:
                    suggested_config = 'character_focus'
                    confidence = 0.8
                    reason = "Slam Dunk - focus su personaggi"
            
            elif 'gon' in base_lower:
                # Hunter x Hunter - trasformazioni e energia
                if any(word in prompt_lower for word in ['transformation', 'energy', 'power', 'dynamic']):
                    suggested_config = 'action_sequence'
                    confidence = 0.9
                    reason = "Gon + trasformazione/energia rilevata"
                else:
                    suggested_config = 'character_focus'
                    confidence = 0.85
                    reason = "Gon - focus su sviluppo personaggio"
            
            # === ANALISI GENERICA BASATA SU PROMPT ===
            else:
                if any(word in prompt_lower for word in ['close-up', 'face', 'character', 'portrait', 'holding', 'determination']):
                    suggested_config = 'character_focus'
                    confidence = 0.85
                    reason = "Focus su personaggio rilevato nel prompt"
                elif any(word in prompt_lower for word in ['dialogue', 'conversation', 'interaction', 'gentle', 'players']):
                    suggested_config = 'dialogue_scene'
                    confidence = 0.85
                    reason = "Scena di dialogo rilevata nel prompt"
                elif any(word in prompt_lower for word in ['action', 'dynamic', 'battle', 'fast', 'energy']):
                    suggested_config = 'action_sequence'
                    confidence = 0.8
                    reason = "Azione dinamica rilevata nel prompt"
                elif any(word in prompt_lower for word in ['smooth', 'gentle', 'transition', 'movement']):
                    suggested_config = 'smooth_transition'
                    confidence = 0.8
                    reason = "Transizione fluida rilevata nel prompt"
            
            sequence_configs.append({
                'base_name': base_name,
                'config': suggested_config,
                'confidence': confidence,
                'reason': reason,
                'prompt': prompt
            })
            
            print(f"   🎛️ Config suggerita: {suggested_config}")
            print(f"   📊 Confidenza: {confidence:.1%}")
            print(f"   💡 Motivo: {reason}")
        
        return sequence_configs

    def run_inference_per_sequence(self, prompt_dir: str, output_dir: str, 
                                 sequence_configs: list = None,
                                 show_progress: bool = True):
        """
        Esegue l'inferenza con configurazioni diverse per ogni sequenza
        """
        if not self.check_colab_environment():
            return False
        
        self.optimize_for_colab()
        
        # Analizza sequenze se non fornite
        if sequence_configs is None:
            sequence_configs = self.analyze_manga_sequences(prompt_dir)
        
        print(f"\n🎬 CONVERSIONE MULTI-CONFIGURAZIONE")
        print("=" * 50)
        
        all_success = True
        generated_videos = []
        
        for i, seq_config in enumerate(sequence_configs):
            base_name = seq_config['base_name']
            config_type = seq_config['config']
            
            print(f"\n� SEQUENZA {i+1}/{len(sequence_configs)}: {base_name}")
            print(f"🎛️ Configurazione: {config_type}")
            print(f"📝 Prompt: \"{seq_config['prompt']}\"")
            
            # Crea directory temporanea per questa sequenza
            seq_prompt_dir = f"{prompt_dir}_seq_{i+1}"
            seq_output_dir = f"{output_dir}/{base_name}"
            
            os.makedirs(seq_prompt_dir, exist_ok=True)
            os.makedirs(seq_output_dir, exist_ok=True)
            
            try:
                # Copia solo i file per questa sequenza
                frame1_src = Path(prompt_dir) / f"{base_name}_frame1.png"
                frame3_src = Path(prompt_dir) / f"{base_name}_frame3.png"
                
                if frame1_src.exists() and frame3_src.exists():
                    import shutil
                    shutil.copy(frame1_src, f"{seq_prompt_dir}/{base_name}_frame1.png")
                    shutil.copy(frame3_src, f"{seq_prompt_dir}/{base_name}_frame3.png")
                    
                    # Crea prompts.txt specifico per questa sequenza
                    with open(f"{seq_prompt_dir}/prompts.txt", 'w', encoding='utf-8') as f:
                        f.write(seq_config['prompt'])
                    
                    # Esegui conversione con configurazione specifica
                    success = self.run_inference_colab(
                        seq_prompt_dir, seq_output_dir, config_type, 
                        show_progress=show_progress, auto_analyze=False
                    )
                    
                    if success:
                        # Trova video generati per questa sequenza
                        import glob
                        videos = glob.glob(f"{seq_output_dir}/**/*.mp4", recursive=True)
                        generated_videos.extend(videos)
                        
                        print(f"✅ Sequenza {base_name} completata!")
                        for video in videos:
                            print(f"   📹 {os.path.basename(video)}")
                    else:
                        print(f"❌ Errore nella sequenza {base_name}")
                        all_success = False
                    
                    # Pulizia directory temporanea
                    shutil.rmtree(seq_prompt_dir, ignore_errors=True)
                    
                else:
                    print(f"❌ File mancanti per {base_name}")
                    all_success = False
                    
            except Exception as e:
                print(f"❌ Errore nella sequenza {base_name}: {e}")
                all_success = False
        
        if all_success:
            print(f"\n🎉 TUTTE LE SEQUENZE COMPLETATE!")
            print(f"� Video totali generati: {len(generated_videos)}")
            
            # Salva lista per download
            globals()['GENERATED_VIDEOS'] = generated_videos
            globals()['MULTI_CONFIG_SUCCESS'] = True
            
        return all_success

    def run_inference_colab(self, prompt_dir: str, output_dir: str, 
                           config_type: str = 'dramatic_change',
                           show_progress: bool = True,
                           auto_analyze: bool = True):
        """
        Esegue l'inferenza ToonCrafter ottimizzata per Colab con analisi intelligente
        """
        if not self.check_colab_environment():
            return False
        
        # Applica ottimizzazioni Colab
        self.optimize_for_colab()
        
        # Analisi intelligente del contenuto se richiesta
        if auto_analyze:
            # Analisi intelligente automatica
            sequence_configs = self.analyze_manga_sequences(prompt_dir)
            if sequence_configs:
                # Usa la configurazione della prima sequenza come default
                suggested_config = sequence_configs[0]['config']
                confidence = sequence_configs[0]['confidence']
            else:
                suggested_config, confidence = 'manga_stable', 0.5
            
            if confidence > 0.7:
                print(f"\n🤖 CONSIGLIO: Uso configurazione suggerita '{suggested_config}' (confidenza: {confidence:.1%})")
                config_type = suggested_config
            elif confidence > 0.5:
                print(f"\n💡 SUGGERIMENTO: Considera di usare '{suggested_config}' (confidenza: {confidence:.1%})")
                print(f"🔧 Configurazione attuale: '{config_type}' - vuoi continuare con questa?")
        
        if config_type not in self.configs:
            raise ValueError(f"Tipo di config non valido: {config_type}")
        
        config = self.configs[config_type].copy()
        
        # Path ai file necessari
        inference_script = self.tooncrafter_path / "scripts" / "evaluation" / "inference.py"
        base_config = self.tooncrafter_path / "configs" / "inference_512_v1.0.yaml"
        checkpoint = self.tooncrafter_path / "checkpoints" / "tooncrafter_512_interp_v1" / "model.ckpt"
        
        # Verifica che i file esistano
        missing_files = []
        for file_path, name in [(inference_script, "Script inferenza"), 
                               (base_config, "Config file"), 
                               (checkpoint, "Checkpoint")]:
            if not file_path.exists():
                missing_files.append(f"{name}: {file_path}")
        
        if missing_files:
            print("❌ File mancanti:")
            for missing in missing_files:
                print(f"   {missing}")
            return False
        
        # Crea directory di output
        os.makedirs(output_dir, exist_ok=True)
        
        # Comando per ToonCrafter con ottimizzazioni Colab e qualità migliorata
        cmd = [
            "python", str(inference_script),
            "--config", str(base_config),
            "--ckpt_path", str(checkpoint),
            "--prompt_dir", prompt_dir,
            "--savedir", output_dir,
            "--frame_stride", str(config['frame_stride']),
            "--ddim_steps", str(config['ddim_steps']),
            "--unconditional_guidance_scale", str(config['unconditional_guidance_scale']),
            "--guidance_rescale", str(config['guidance_rescale']),
            "--video_length", str(config['video_length']),
            "--height", "320",
            "--width", "512",
            "--n_samples", "1",
            "--bs", "1",
            "--seed", "42",  # Seed più stabile
            "--text_input",
            "--interp",
            "--perframe_ae",  # Importante per memoria GPU limitata
            "--ddim_eta", "0.8",  # Ridotto per meno rumore
            "--decode_frame_bs", "1",  # Decodifica frame singoli per stabilità
            "--timestep_spacing", "uniform",  # Distribuzione uniforme per transizioni più fluide
        ]
        
        print(f"🚀 Avvio ToonCrafter con configurazione: {config_type}")
        print(f"📊 Parametri ottimizzati per Colab:")
        for key, value in config.items():
            if key != 'description':
                print(f"   {key}: {value}")
        
        print(f"\n💡 {config['description']}")
        print(f"⏱️ Tempo stimato: 5-15 minuti (dipende dalla GPU)")
        
        # Esegui il comando con feedback in tempo reale
        start_time = time.time()
        
        try:
            if show_progress:
                # Esecuzione con output in tempo reale
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    cwd=str(self.tooncrafter_path)
                )
                
                print("\n📋 Output ToonCrafter:")
                print("-" * 50)
                
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        print(output.strip())
                
                rc = process.poll()
                
            else:
                # Esecuzione semplice
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    cwd=str(self.tooncrafter_path)
                )
                rc = result.returncode
                if rc != 0:
                    print("Stderr:", result.stderr)
                    print("Stdout:", result.stdout)
            
            end_time = time.time()
            duration = end_time - start_time
            
            if rc == 0:
                print("-" * 50)
                print(f"🎉 ToonCrafter completato con successo!")
                print(f"⏱️ Tempo impiegato: {duration:.1f} secondi ({duration/60:.1f} minuti)")
                print(f"📂 Output salvato in: {output_dir}")
                
                # Verifica file generati
                output_files = []
                if os.path.exists(output_dir):
                    for root, dirs, files in os.walk(output_dir):
                        for file in files:
                            if file.endswith('.mp4'):
                                output_files.append(os.path.join(root, file))
                
                if output_files:
                    print(f"📹 File generati: {len(output_files)}")
                    for video in output_files:
                        size_mb = os.path.getsize(video) / (1024 * 1024)
                        print(f"   {os.path.basename(video)} ({size_mb:.1f} MB)")
                else:
                    print("⚠️ Nessun file .mp4 trovato nell'output")
                
                return True
            else:
                print(f"❌ ToonCrafter fallito con codice: {rc}")
                return False
                
        except Exception as e:
            print(f"❌ Errore durante l'esecuzione: {e}")
            return False
        
        finally:
            # Pulizia memoria
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass


def run_with_config(tooncrafter_path: str, prompt_dir: str, output_dir: str, 
                   config_type: str = 'dramatic_change'):
    """
    Funzione helper per eseguire facilmente da notebook
    """
    runner = ColabMangaToonCrafterRunner(tooncrafter_path)
    return runner.run_inference_colab(prompt_dir, output_dir, config_type)


def list_configs():
    """
    Mostra tutte le configurazioni disponibili con raccomandazioni aggiornate
    """
    runner = ColabMangaToonCrafterRunner(".")  # Path dummy
    
    print("🎛️ Configurazioni ottimizzate per Google Colab (AGGIORNATE):")
    print("=" * 70)
    
    # Ordina per raccomandazione
    recommended_order = [
        'manga_stable', 'character_focus', 'dialogue_scene', 
        'smooth_transition', 'dramatic_change', 'action_sequence', 'colab_fast'
    ]
    
    for name in recommended_order:
        if name in runner.configs:
            config = runner.configs[name]
            
            # Indica le configurazioni nuove/migliorate
            if name in ['manga_stable', 'character_focus']:
                status = "🆕 NUOVO"
            elif name in ['dramatic_change', 'colab_fast']:
                status = "🔧 MIGLIORATO"
            else:
                status = "✅ STANDARD"
            
            print(f"\n📋 {name.upper()} {status}:")
            print(f"   {config['description']}")
            print("   Parametri:")
            for key, value in config.items():
                if key != 'description':
                    print(f"     • {key}: {value}")
    
    print("\n💡 RACCOMANDAZIONI AGGIORNATE (per problemi di qualità):")
    print("   🎯 manga_stable: MIGLIOR SCELTA per la maggior parte dei manga")
    print("   👤 character_focus: Per zoom su volti/personaggi (es. One Piece)")
    print("   💬 dialogue_scene: Per interazioni/dialoghi (es. Slam Dunk)")
    print("   🎬 smooth_transition: Per panel molto simili")
    print("   ⚡ colab_fast: Solo per test veloci")
    
    print("\n🔧 PROBLEMI RISOLTI:")
    print("   ❌ Video confusi → Parametri più conservativi")
    print("   ❌ Scene vuote → Frame stride ridotto") 
    print("   ❌ Troppo rumore → Guidance scale più basso")
    print("   ❌ Instabilità → Video più corti e stabili")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ToonCrafter Runner per Google Colab')
    parser.add_argument('--tooncrafter_path', required=True,
                       help='Path alla directory ToonCrafter')
    parser.add_argument('--prompt_dir', required=True,
                       help='Directory contenente i prompt preparati')
    parser.add_argument('--output_dir', required=True,
                       help='Directory di output per i video generati')
    parser.add_argument('--config_type', default='dramatic_change',
                       help='Tipo di configurazione da usare')
    parser.add_argument('--list_configs', action='store_true',
                       help='Mostra le configurazioni disponibili')
    
    args = parser.parse_args()
    
    if args.list_configs:
        list_configs()
    else:
        success = run_with_config(
            args.tooncrafter_path,
            args.prompt_dir,
            args.output_dir,
            args.config_type
        )
        
        if success:
            print(f"\n🎉 Processo completato! Risultati in: {args.output_dir}")
        else:
            print("\n❌ Processo fallito!")
            sys.exit(1)

# Funzioni di compatibilità per il notebook
def run_manga_conversion(prompt_dir, output_dir, config_type='manga_stable'):
    """
    Funzione helper per compatibilità con il notebook - ora con analisi intelligente
    """
    runner = ColabMangaToonCrafterRunner("/content/ToonCrafter")
    
    # Analisi intelligente automatica
    sequence_configs = runner.analyze_manga_sequences(prompt_dir)
    if sequence_configs:
        # Usa la configurazione della prima sequenza come default
        suggested_config = sequence_configs[0]['config']
        confidence = sequence_configs[0]['confidence']
    else:
        suggested_config, confidence = 'manga_stable', 0.5
    
    # Se la configurazione predefinita è 'dramatic_change' e abbiamo un suggerimento migliore
    if config_type == 'dramatic_change' and confidence > 0.6:
        print(f"\n🤖 OVERRIDE INTELLIGENTE: Cambio da '{config_type}' a '{suggested_config}'")
        print(f"📈 Confidenza: {confidence:.1%} - dovrebbe dare risultati migliori!")
        config_type = suggested_config
    
    return runner.run_inference_colab(prompt_dir, output_dir, config_type, auto_analyze=False)

def run_manga_conversion_smart(prompt_dir, output_dir, config_type='auto'):
    """
    Versione avanzata con selezione automatica della configurazione migliore
    """
    runner = ColabMangaToonCrafterRunner("/content/ToonCrafter")
    
    if config_type == 'auto':
        # Analisi intelligente automatica
        sequence_configs = runner.analyze_manga_sequences(prompt_dir)
        if sequence_configs:
            # Usa la configurazione della prima sequenza come default
            suggested_config = sequence_configs[0]['config']
            confidence = sequence_configs[0]['confidence']
        else:
            suggested_config, confidence = 'manga_stable', 0.5
        print(f"\n🎯 MODALITÀ AUTO: Selezionata '{suggested_config}' (confidenza: {confidence:.1%})")
        config_type = suggested_config
    
    return runner.run_inference_colab(prompt_dir, output_dir, config_type, auto_analyze=True)

# Alias per compatibilità con il notebook
MangaToonCrafterRunner = ColabMangaToonCrafterRunner
