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
                'frame_stride': 12,
                'ddim_steps': 60,
                'unconditional_guidance_scale': 8.5,
                'guidance_rescale': 0.7,
                'video_length': 16,
                'description': 'Per transizioni fluide tra panel simili'
            },
            'dramatic_change': {
                'frame_stride': 18,
                'ddim_steps': 80,
                'unconditional_guidance_scale': 12.0,
                'guidance_rescale': 0.9,
                'video_length': 20,
                'description': 'Per cambi drastici di scena/inquadratura (RACCOMANDATO per manga)'
            },
            'action_sequence': {
                'frame_stride': 15,
                'ddim_steps': 70,
                'unconditional_guidance_scale': 10.0,
                'guidance_rescale': 0.8,
                'video_length': 24,
                'description': 'Per sequenze d\'azione dinamiche'
            },
            'dialogue_scene': {
                'frame_stride': 8,
                'ddim_steps': 50,
                'unconditional_guidance_scale': 7.0,
                'guidance_rescale': 0.6,
                'video_length': 12,
                'description': 'Per scene di dialogo con movimenti sottili'
            },
            'colab_fast': {
                'frame_stride': 10,
                'ddim_steps': 40,
                'unconditional_guidance_scale': 8.0,
                'guidance_rescale': 0.6,
                'video_length': 12,
                'description': 'Configurazione veloce per test su Colab'
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
    
    def run_inference_colab(self, prompt_dir: str, output_dir: str, 
                           config_type: str = 'dramatic_change',
                           show_progress: bool = True):
        """
        Esegue l'inferenza ToonCrafter ottimizzata per Colab
        """
        if not self.check_colab_environment():
            return False
        
        # Applica ottimizzazioni Colab
        self.optimize_for_colab()
        
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
        
        # Comando per ToonCrafter con ottimizzazioni Colab
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
            "--seed", "123",
            "--text_input",
            "--interp",
            "--perframe_ae",  # Importante per memoria GPU limitata
            "--ddim_eta", "1.0"
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
    Mostra tutte le configurazioni disponibili
    """
    runner = ColabMangaToonCrafterRunner(".")  # Path dummy
    
    print("🎛️ Configurazioni ottimizzate per Google Colab:")
    print("=" * 60)
    
    for name, config in runner.configs.items():
        print(f"\n📋 {name.upper()}:")
        print(f"   {config['description']}")
        print("   Parametri:")
        for key, value in config.items():
            if key != 'description':
                print(f"     • {key}: {value}")
    
    print("\n💡 Raccomandazioni:")
    print("   • dramatic_change: Migliore per panel manga con grandi differenze")
    print("   • colab_fast: Per test veloci su Colab")
    print("   • smooth_transition: Per panel molto simili")


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
def run_manga_conversion(prompt_dir, output_dir, config_type='dramatic_change'):
    """Funzione helper per compatibilità con il notebook"""
    runner = ColabMangaToonCrafterRunner("/content/ToonCrafter")
    return runner.run_inference(prompt_dir, output_dir, config_type)

# Alias per compatibilità con il notebook
MangaToonCrafterRunner = ColabMangaToonCrafterRunner
