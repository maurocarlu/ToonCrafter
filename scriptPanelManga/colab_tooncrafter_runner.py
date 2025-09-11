#!/usr/bin/env python3
"""
Script ottimizzato per eseguire ToonCrafter su panel manga con parametri avanzati
VERSIONE GOOGLE COLAB - ottimizzato per l'ambiente Colab
"""

import os
import sys
import shlex
import subprocess
import yaml
from pathlib import Path
import time
import tempfile
import shutil
from PIL import Image  
from panelPreProcessing import PanelPreProcessor, create_manga_preprocessing_config

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
    print("ℹ️ Matplotlib non disponibile (probabile conflitto NumPy). Visualizzazioni disabilitate.")



class ColabMangaToonCrafterRunner:
    """
    Runner ottimizzato per ToonCrafter specifico per panel manga su Google Colab
    """
    
    def __init__(self, tooncrafter_path: str):
        self.tooncrafter_path = Path(tooncrafter_path)
        # ✅ AGGIUNGI PREPROCESSOR
        self.preprocessor = PanelPreProcessor(debug_mode=True)
    
    def resize_image_to_tooncrafter_format(self, image_path, output_path, target_width=512, target_height=320, 
                                     show_images=True, apply_preprocessing=True, preprocessing_config=None):
        """
        📐 Ridimensiona immagine al formato richiesto da ToonCrafter (512x320)
        🎨 Con preprocessing opzionale e visualizzazioni separate
        """
        try:
            original_img = Image.open(image_path)
            original_size = original_img.size
            original_mode = original_img.mode
            print(f"   📸 File: {os.path.basename(image_path)}")
            print(f"   📐 ORIGINALE - Dimensioni: {original_size[0]}x{original_size[1]} ({original_mode})")

            if original_img.mode != 'RGB':
                original_img = original_img.convert('RGB')
                print(f"   🔄 Convertito da {original_mode} a RGB")

            # === LETTERBOX (preserva aspect ratio) ===
            # === SCALE + CENTER-CROP (no padding, no stretch) ===
            ow, oh = original_img.size
            scale = max(target_width / ow, target_height / oh)  # copri interamente il canvas
            rw, rh = int(round(ow * scale)), int(round(oh * scale))
            img_scaled = original_img.resize((rw, rh), Image.Resampling.LANCZOS)

            left = max(0, (rw - target_width) // 2)
            top = max(0, (rh - target_height) // 2)
            right = left + target_width
            bottom = top + target_height
            img_rescaled = img_scaled.crop((left, top, right, bottom))
            rescaled_size = img_rescaled.size
            print(f"   📐 CENTER-CROP - {rescaled_size[0]}x{rescaled_size[1]} (scale={scale:.3f}, crop=({left},{top}))")

            if show_images and plt is not None:
                print(f"   🖼️ VISUALIZZAZIONE 1: Pre vs Post Letterbox")
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1); plt.imshow(original_img); plt.title(f'ORIGINALE\n{original_size[0]}x{original_size[1]}'); plt.axis('off')
                plt.subplot(1, 2, 2); plt.imshow(img_rescaled); plt.title(f'LETTERBOX\n{rescaled_size[0]}x{rescaled_size[1]}'); plt.axis('off')
                plt.tight_layout(); plt.show()
            elif show_images:
                print("   (visualizzazione disabilitata: matplotlib non disponibile)")

            # Preprocessing opzionale (usa img_rescaled come input)
            final_image = img_rescaled
            
            # ✅ PREPROCESSING CON CONFIGURAZIONE UNICA
            if apply_preprocessing:
                print(f"   🎨 Applicando preprocessing...")
                
                # Salva temporaneamente l'immagine rescaled per il preprocessing
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_rescaled_path = temp_file.name
                    img_rescaled.save(temp_rescaled_path, 'PNG')
                
                try:
                    # USA SEMPRE LA STESSA CONFIGURAZIONE
                    config = preprocessing_config or create_manga_preprocessing_config()
                    processed_image = self.preprocessor.apply_preprocessing_pipeline(
                        temp_rescaled_path, 
                        pipeline_config=config
                    )
                    
                    if processed_image:
                        print(f"   ✅ Preprocessing completato")
                        final_image = processed_image
                        
                        # ✅ VISUALIZZAZIONE 2: RESCALED vs PREPROCESSED
                        if show_images:
                            print(f"   🖼️ VISUALIZZAZIONE 2: Post-Rescaling vs Post-Preprocessing")
                            plt.figure(figsize=(10, 5))
                            
                            # Post-rescaling (input del preprocessing)
                            plt.subplot(1, 2, 1)
                            plt.imshow(img_rescaled)
                            plt.title(f'POST-RESCALING\n(Input Preprocessing)\n{rescaled_size[0]}x{rescaled_size[1]}')
                            plt.axis('off')
                            
                            # Post-preprocessing (finale)
                            plt.subplot(1, 2, 2)
                            plt.imshow(processed_image)
                            plt.title(f'POST-RESCALING-PREPROCESSING\n(Finale)\n{processed_image.size[0]}x{processed_image.size[1]}')
                            plt.axis('off')
                            
                            plt.tight_layout()
                            plt.show()
                            
                    else:
                        print(f"   ⚠️ Preprocessing fallito, uso immagine rescaled")
                        
                finally:
                    # Cleanup file temporaneo
                    try:
                        os.unlink(temp_rescaled_path)
                    except:
                        pass
            else:
                print(f"   ⚪ Preprocessing disabilitato, uso immagine rescaled")
            
            # ✅ SALVA IMMAGINE FINALE
            final_image.save(output_path, 'PNG', quality=95)
            
            # ✅ DEBUG: Verifica file salvato
            saved_size = os.path.getsize(output_path) / 1024  # KB
            print(f"   💾 Salvato: {os.path.basename(output_path)} ({saved_size:.1f} KB)")
            print(f"   ✅ Processo completato con successo!")
        
            return True
                
        except Exception as e:
            print(f"   ❌ Errore durante processo: {e}")
            return False
    
    def run_custom_parameters_conversion(self, base_name, prompt, custom_params, output_dir, input_dir, 
                                         show_resize=True, enable_preprocessing=True, preprocessing_config=None,
                                         lora_path=None, lora_scale=1.0):
        """
        🎛️ Esecuzione con parametri completamente personalizzati + rescaling automatico
        🎨 NUOVO: Con preprocessing opzionale
        """
        print(f"\n🎬 === INIZIANDO CONVERSIONE: {base_name} ===")
        
        # Mostra configurazione preprocessing
        if enable_preprocessing:
            config = preprocessing_config or create_manga_preprocessing_config()
            print(f"🎨 Preprocessing abilitato:")
            for step, step_config in config.items():
                if step_config.get('enabled', False):
                    print(f"   ✅ {step}: {step_config}")
        else:
            print(f"⚪ Preprocessing disabilitato")
        
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
        
        # ✅ CREA DIRECTORY TEMPORANEA CON RESCALING
        temp_input_dir = tempfile.mkdtemp(prefix=f"tooncrafter_{base_name}_")
        
        try:
            # ✅ RESCALING + COPIA FILE SPECIFICI
            temp_frame1 = os.path.join(temp_input_dir, os.path.basename(frame1_path))
            temp_frame3 = os.path.join(temp_input_dir, os.path.basename(frame3_path))
            
            print(f"📁 Directory temporanea: {temp_input_dir}")
            print(f"📐 Ridimensionando immagini a 512x320...")
            
            # ✅ VISUALIZZA RESCALING FRAME1
            print(f"\n🖼️ RESCALING FRAME1:")
            if not self.resize_image_to_tooncrafter_format(frame1_path, temp_frame1, show_images=show_resize, apply_preprocessing=enable_preprocessing, preprocessing_config=preprocessing_config):
                print(f"❌ Errore ridimensionamento {frame1_path}")
                return False
            
            # ✅ VISUALIZZA RESCALING FRAME3
            print(f"\n🖼️ RESCALING FRAME3:")
            if not self.resize_image_to_tooncrafter_format(frame3_path, temp_frame3, show_images=show_resize, apply_preprocessing=enable_preprocessing, preprocessing_config=preprocessing_config):
                print(f"❌ Errore ridimensionamento {frame3_path}")
                return False
            
            # ✅ CREA FILE PROMPT SPECIFICO NELLA DIRECTORY TEMPORANEA
            prompt_file_path = os.path.join(temp_input_dir, f"{base_name}.txt")
            with open(prompt_file_path, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            print(f"📝 File prompt creato: {os.path.basename(prompt_file_path)}")
            print(f"   Contenuto: '{prompt}'")
            
            # ✅ DEBUG: Mostra contenuto directory temporanea
            temp_files = os.listdir(temp_input_dir)
            print(f"📂 File nella directory temporanea: {temp_files}")
            
            print(f"\n✅ Immagini ridimensionate e file prompt creato nella directory temporanea")
            
            # ✅ COMANDO TOONCRAFTER CON DIRECTORY TEMPORANEA
            inference_script = self.tooncrafter_path / "scripts" / "evaluation" / "inference.py"
            base_config = self.tooncrafter_path / "configs" / "inference_512_v1.0.yaml"
            checkpoint = self.tooncrafter_path / "checkpoints" / "tooncrafter_512_interp_v1" / "model.ckpt"
            
            # Crea nome output con seed
            seed = 123
            name = f"tooncrafter_{base_name}_seed{seed}"
            final_output_dir = os.path.join(output_dir, name)
            
            # ✅ Interprete portabile
            python_exec = sys.executable or ("python" if os.name == "nt" else "python3")
            print(f"🐍 Interpreter: {python_exec}")

            cmd = [
                python_exec, "-u", str(inference_script),
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
                "--ddim_eta", "0.0",
                "--prompt_dir", temp_input_dir,
                "--text_input",
                "--video_length", str(custom_params['video_length']),
                "--frame_stride", str(custom_params['frame_stride']),
                "--timestep_spacing", "uniform",
                "--guidance_rescale", str(custom_params['guidance_rescale']),
                "--perframe_ae",
                "--interp"
            ]
            if lora_path:
                cmd += ["--lora_path", str(lora_path), "--lora_scale", str(lora_scale)]

            print("🔧 Comando:", " ".join(shlex.quote(c) for c in cmd))

            # --- esecuzione con streaming dello stdout ---
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in proc.stdout:
                    if not line:
                        break
                    print(line, end="")
            finally:
                proc.stdout.close()
                rc = proc.wait()

            if rc != 0:
                print(f"❌ Inference fallita (rc={rc})")
                return False
            # Successo: continua
            if True:
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

# Alias per compatibilità con il notebook
MangaToonCrafterRunner = ColabMangaToonCrafterRunner