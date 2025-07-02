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
    
    def run_custom_parameters_conversion(self, base_name, prompt, custom_params, output_dir, input_dir):
        """
        🎛️ Esecuzione con parametri completamente personalizzati
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
        
        # Comando ToonCrafter con parametri custom
        inference_script = self.tooncrafter_path / "scripts" / "evaluation" / "inference.py"
        base_config = self.tooncrafter_path / "configs" / "inference_t2v_512_v1.0.yaml"
        checkpoint = self.tooncrafter_path / "checkpoints" / "tooncrafter_512_interp_v1" / "model.ckpt"
        
        cmd = [
            "python", str(inference_script),
            "--config", str(base_config),
            "--ckpt_path", str(checkpoint),
            "--prompt", prompt,
            "--image_path", frame1_path,
            "--image_path_2", frame3_path,
            "--n_samples", "1",
            "--batch_size", "1",
            "--seed", "42",
            "--save_dir", str(output_dir),
            # Parametri personalizzati
            "--frame_stride", str(custom_params['frame_stride']),
            "--ddim_steps", str(custom_params['ddim_steps']),
            "--unconditional_guidance_scale", str(custom_params['unconditional_guidance_scale']),
            "--guidance_rescale", str(custom_params['guidance_rescale']),
            "--video_length", str(custom_params['video_length'])
        ]
        
        # 🆕 OUTPUT DETTAGLIATO
        print(f"📝 PROMPT: '{prompt}'")
        print(f"🎛️ PARAMETRI:")
        print(f"   • frame_stride: {custom_params['frame_stride']}")
        print(f"   • ddim_steps: {custom_params['ddim_steps']}")
        print(f"   • guidance_scale: {custom_params['unconditional_guidance_scale']}")
        print(f"   • guidance_rescale: {custom_params['guidance_rescale']}")
        print(f"   • video_length: {custom_params['video_length']}")
        print(f"📁 Output: {output_dir}")
        print(f"🚀 Avviando ToonCrafter...")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.tooncrafter_path)
            
            if result.returncode == 0:
                print(f"✅ Conversione completata per '{base_name}'!")
                
                # 🆕 VERIFICA FILE GENERATI
                generated_files = []
                if os.path.exists(output_dir):
                    for root, dirs, files in os.walk(output_dir):
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
                    print(f"   Error: {result.stderr[:200]}...")
                return False
                
        except Exception as e:
            print(f"❌ Errore durante esecuzione: {e}")
            return False

    
# Alias per compatibilità con il notebook
MangaToonCrafterRunner = ColabMangaToonCrafterRunner
