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
        
        # ✅ COMANDO CORRETTO BASATO SU run.sh
        inference_script = self.tooncrafter_path / "scripts" / "evaluation" / "inference.py"
        base_config = self.tooncrafter_path / "configs" / "inference_512_v1.0.yaml"  # ✅ Nome corretto
        checkpoint = self.tooncrafter_path / "checkpoints" / "tooncrafter_512_interp_v1" / "model.ckpt"
        
        # ✅ Crea nome output con seed
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
            "--prompt_dir", input_dir,
            "--text_input",
            "--video_length", str(custom_params['video_length']),
            "--frame_stride", str(custom_params['frame_stride']),
            "--timestep_spacing", "uniform_trailing",
            "--guidance_rescale", str(custom_params['guidance_rescale']),
            "--perframe_ae",
            "--interp"
        ]
        
        # 🆕 OUTPUT DETTAGLIATO
        print(f"📝 PROMPT: '{prompt}'")
        print(f"🎛️ PARAMETRI:")
        print(f"   • frame_stride: {custom_params['frame_stride']}")
        print(f"   • ddim_steps: {custom_params['ddim_steps']}")
        print(f"   • guidance_scale: {custom_params['unconditional_guidance_scale']}")
        print(f"   • guidance_rescale: {custom_params['guidance_rescale']}")
        print(f"   • video_length: {custom_params['video_length']}")
        print(f"📁 Input: {input_dir}")
        print(f"📁 Output: {final_output_dir}")
        print(f"🚀 Avviando ToonCrafter...")
        
        # 🆕 DEBUG: Mostra comando completo
        print(f"🔧 Comando: {' '.join(cmd)}")
        
        try:
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

# Alias per compatibilità con il notebook
MangaToonCrafterRunner = ColabMangaToonCrafterRunner