#!/usr/bin/env python3
"""
Script ottimizzato per eseguire ToonCrafter su panel manga con parametri avanzati
per gestire transizioni drastiche tra panel
"""

import os
import sys
import argparse
import subprocess
import yaml
from pathlib import Path
import shutil

class MangaToonCrafterRunner:
    """
    Runner ottimizzato per ToonCrafter specifico per panel manga
    """
    
    def __init__(self, tooncrafter_path: str):
        self.tooncrafter_path = Path(tooncrafter_path)
        self.configs = self._load_optimized_configs()
    
    def _load_optimized_configs(self):
        """
        Configurazioni ottimizzate per diversi tipi di transizioni manga
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
                'description': 'Per cambi drastici di scena/inquadratura'
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
            }
        }
    
    def create_custom_config(self, base_config_path: str, 
                           output_config_path: str, 
                           modifications: dict):
        """
        Crea una configurazione personalizzata modificando quella base
        """
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Applica modifiche per migliorare le transizioni drastiche
        if 'model' in config and 'params' in config['model']:
            # Modifiche al modello per migliore handling di grandi cambiamenti
            model_params = config['model']['params']
            
            # Aumenta la capacità del modello di gestire variazioni
            if 'unet_config' in model_params and 'params' in model_params['unet_config']:
                unet_params = model_params['unet_config']['params']
                
                # Aumenta i canali di attenzione per migliore comprensione spaziale
                if 'num_head_channels' in unet_params:
                    unet_params['num_head_channels'] = min(128, unet_params['num_head_channels'] * 2)
                
                # Abilita controlli temporali più sofisticati
                unet_params['temporal_attention'] = True
                unet_params['temporal_conv'] = True
                
        # Applica modifiche personalizzate
        for key, value in modifications.items():
            if '.' in key:
                # Supporta chiavi nested come 'model.params.scale_factor'
                keys = key.split('.')
                current = config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                config[key] = value
        
        # Salva la configurazione modificata
        with open(output_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return output_config_path
    
    def run_inference(self, prompt_dir: str, output_dir: str, 
                     config_type: str = 'dramatic_change',
                     custom_params: dict = None):
        """
        Esegue l'inferenza ToonCrafter con parametri ottimizzati
        """
        if config_type not in self.configs:
            raise ValueError(f"Tipo di config non valido: {config_type}")
        
        config = self.configs[config_type].copy()
        
        # Sovrascrivi con parametri personalizzati se forniti
        if custom_params:
            config.update(custom_params)
        
        # Path ai file necessari
        inference_script = self.tooncrafter_path / "scripts" / "evaluation" / "inference.py"
        base_config = self.tooncrafter_path / "configs" / "inference_512_v1.0.yaml"
        checkpoint = self.tooncrafter_path / "checkpoints" / "tooncrafter_512_interp_v1" / "model.ckpt"
        
        # Verifica che i file esistano
        if not inference_script.exists():
            raise FileNotFoundError(f"Script di inferenza non trovato: {inference_script}")
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint}")
        
        # Crea directory di output
        os.makedirs(output_dir, exist_ok=True)
        
        # Comando per ToonCrafter
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
            "--perframe_ae",
            "--ddim_eta", "1.0"
        ]
        
        print(f"Eseguendo ToonCrafter con config: {config_type}")
        print(f"Parametri: {config}")
        print(f"Comando: {' '.join(cmd)}")
        
        # Esegui il comando
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.tooncrafter_path))
            
            if result.returncode == 0:
                print("ToonCrafter completato con successo!")
                print("Output:", result.stdout)
            else:
                print("Errore durante l'esecuzione di ToonCrafter:")
                print("Stderr:", result.stderr)
                print("Stdout:", result.stdout)
                
            return result.returncode == 0
            
        except Exception as e:
            print(f"Errore nell'esecuzione: {e}")
            return False
    
    def batch_process(self, sequences_info: list, base_output_dir: str):
        """
        Processa un batch di sequenze con configurazioni ottimizzate
        """
        results = []
        
        for i, seq_info in enumerate(sequences_info):
            print(f"\n--- Processando sequenza {i+1}/{len(sequences_info)}: {seq_info['name']} ---")
            
            # Determina il tipo di configurazione basato sui metadati
            config_type = self._determine_config_type(seq_info)
            
            output_dir = os.path.join(base_output_dir, seq_info['name'])
            
            success = self.run_inference(
                prompt_dir=seq_info['prompt_dir'],
                output_dir=output_dir,
                config_type=config_type
            )
            
            results.append({
                'sequence': seq_info['name'],
                'success': success,
                'config_type': config_type,
                'output_dir': output_dir
            })
        
        return results
    
    def _determine_config_type(self, seq_info: dict) -> str:
        """
        Determina automaticamente il tipo di configurazione basato sui metadati
        """
        # Logica semplice per determinare il tipo di transizione
        # Puoi espandere questa logica basandoti sui metadati dei panel
        
        sequence_name = seq_info.get('name', '').lower()
        
        if 'action' in sequence_name or 'fight' in sequence_name:
            return 'action_sequence'
        elif 'dialogue' in sequence_name or 'talk' in sequence_name:
            return 'dialogue_scene'
        elif 'smooth' in sequence_name:
            return 'smooth_transition'
        else:
            # Default per transizioni drastiche
            return 'dramatic_change'


def main():
    parser = argparse.ArgumentParser(description='Runner ottimizzato ToonCrafter per manga')
    parser.add_argument('--tooncrafter_path', required=True,
                       help='Path alla directory ToonCrafter')
    parser.add_argument('--prompt_dir', required=True,
                       help='Directory contenente i prompt preparati')
    parser.add_argument('--output_dir', required=True,
                       help='Directory di output per i video generati')
    parser.add_argument('--config_type', default='dramatic_change',
                       choices=['smooth_transition', 'dramatic_change', 'action_sequence', 'dialogue_scene'],
                       help='Tipo di configurazione da usare')
    parser.add_argument('--list_configs', action='store_true',
                       help='Mostra le configurazioni disponibili')
    
    args = parser.parse_args()
    
    runner = MangaToonCrafterRunner(args.tooncrafter_path)
    
    if args.list_configs:
        print("Configurazioni disponibili:")
        for name, config in runner.configs.items():
            print(f"\n{name}:")
            print(f"  Descrizione: {config['description']}")
            for key, value in config.items():
                if key != 'description':
                    print(f"  {key}: {value}")
        return
    
    # Esegui l'inferenza
    success = runner.run_inference(
        prompt_dir=args.prompt_dir,
        output_dir=args.output_dir,
        config_type=args.config_type
    )
    
    if success:
        print(f"\nInferenza completata! Risultati salvati in: {args.output_dir}")
    else:
        print("\nErrore durante l'inferenza!")
        sys.exit(1)


if __name__ == "__main__":
    main()
