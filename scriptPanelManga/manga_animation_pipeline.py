#!/usr/bin/env python3
"""
Pipeline completa per convertire panel manga in scene anime utilizzando ToonCrafter ottimizzato
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
import time

# Importa i nostri moduli
from manga_to_anime import MangaToAnimeConverter
from advanced_tooncrafter_runner import MangaToonCrafterRunner

class MangaAnimationPipeline:
    """
    Pipeline completa per la conversione da manga ad anime
    """
    
    def __init__(self, manga_dataset_path: str, tooncrafter_path: str, output_base_dir: str):
        self.manga_dataset_path = manga_dataset_path
        self.tooncrafter_path = tooncrafter_path
        self.output_base_dir = output_base_dir
        
        # Inizializza i componenti
        self.converter = MangaToAnimeConverter(manga_dataset_path, tooncrafter_path)
        self.runner = MangaToonCrafterRunner(tooncrafter_path)
        
        # Crea directory di output
        os.makedirs(output_base_dir, exist_ok=True)
    
    def analyze_manga_panels(self, manga_name: str) -> Dict:
        """
        Analizza i panel di un manga per determinare le migliori strategie di conversione
        """
        manga_panels = [p for p in self.converter.manga_dataset.panels if p['manga'] == manga_name]
        
        if not manga_panels:
            return {'error': f'Nessun panel trovato per {manga_name}'}
        
        # Ordina per pagina
        manga_panels.sort(key=lambda x: (x['page'], x['panel_id']))
        
        analysis = {
            'manga_name': manga_name,
            'total_panels': len(manga_panels),
            'pages': len(set(p['page'] for p in manga_panels)),
            'panels_per_page': {},
            'recommended_sequences': []
        }
        
        # Analizza panel per pagina
        for panel in manga_panels:
            page = panel['page']
            if page not in analysis['panels_per_page']:
                analysis['panels_per_page'][page] = []
            analysis['panels_per_page'][page].append(panel)
        
        # Suggerisci sequenze basate sulla densità di panel
        for page, page_panels in analysis['panels_per_page'].items():
            if len(page_panels) >= 2:
                for i in range(len(page_panels) - 1):
                    sequence = {
                        'name': f"{manga_name}_page{page}_seq{i}",
                        'panel1': page_panels[i],
                        'panel2': page_panels[i + 1],
                        'recommended_config': self._suggest_config_type(page_panels[i], page_panels[i + 1])
                    }
                    analysis['recommended_sequences'].append(sequence)
        
        return analysis
    
    def _suggest_config_type(self, panel1: Dict, panel2: Dict) -> str:
        """
        Suggerisce il tipo di configurazione basato sui metadati dei panel
        """
        # Calcola la differenza nelle dimensioni delle bounding box
        bbox1 = panel1['bbox']
        bbox2 = panel2['bbox']
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        area_ratio = max(area1, area2) / min(area1, area2) if min(area1, area2) > 0 else 1
        
        # Determina il tipo basato sulla differenza di area (proxy per il cambiamento di scena)
        if area_ratio > 3.0:
            return 'dramatic_change'
        elif area_ratio > 1.5:
            return 'action_sequence'
        else:
            return 'smooth_transition'
    
    def run_full_pipeline(self, manga_name: str, max_sequences: int = 5, 
                         auto_run: bool = False) -> Dict:
        """
        Esegue la pipeline completa per un manga
        """
        print(f"🎌 Avvio pipeline per manga: {manga_name}")
        
        # 1. Analizza il manga
        print("📊 Analisi panel manga...")
        analysis = self.analyze_manga_panels(manga_name)
        
        if 'error' in analysis:
            return analysis
        
        print(f"   Trovati {analysis['total_panels']} panel in {analysis['pages']} pagine")
        print(f"   Sequenze raccomandate: {len(analysis['recommended_sequences'])}")
        
        # 2. Processa le sequenze (limita il numero per test)
        sequences_to_process = analysis['recommended_sequences'][:max_sequences]
        
        print(f"\n🎬 Processamento {len(sequences_to_process)} sequenze...")
        
        processed_sequences = []
        
        for i, seq_info in enumerate(sequences_to_process):
            print(f"   Sequenza {i+1}/{len(sequences_to_process)}: {seq_info['name']}")
            
            # Prepara input per ToonCrafter
            panel1_idx = self.converter.manga_dataset.panels.index(seq_info['panel1'])
            panel2_idx = self.converter.manga_dataset.panels.index(seq_info['panel2'])
            
            panel1_img, _ = self.converter.manga_dataset[panel1_idx]
            panel2_img, _ = self.converter.manga_dataset[panel2_idx]
            
            prompt_dir = self.converter.prepare_tooncrafter_input(
                panel1_img, panel2_img, 
                seq_info['panel1'], seq_info['panel2'],
                self.output_base_dir, seq_info['name']
            )
            
            sequence_data = {
                'name': seq_info['name'],
                'prompt_dir': prompt_dir,
                'config_type': seq_info['recommended_config'],
                'panel1_info': seq_info['panel1'],
                'panel2_info': seq_info['panel2']
            }
            
            processed_sequences.append(sequence_data)
        
        # 3. Genera comandi o esegui ToonCrafter
        if auto_run:
            print(f"\n🚀 Esecuzione automatica ToonCrafter...")
            results = self.run_tooncrafter_batch(processed_sequences)
        else:
            print(f"\n📝 Generazione comandi per esecuzione manuale...")
            results = self.generate_execution_commands(processed_sequences)
        
        # 4. Salva report completo
        report = {
            'manga_name': manga_name,
            'analysis': analysis,
            'processed_sequences': len(processed_sequences),
            'results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        report_path = os.path.join(self.output_base_dir, f"{manga_name}_pipeline_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 Report salvato: {report_path}")
        
        return report
    
    def run_tooncrafter_batch(self, sequences: List[Dict]) -> List[Dict]:
        """
        Esegue ToonCrafter su un batch di sequenze
        """
        results = []
        
        for i, seq in enumerate(sequences):
            print(f"   🎨 ToonCrafter {i+1}/{len(sequences)}: {seq['name']}")
            
            output_dir = os.path.join(self.output_base_dir, "animations", seq['name'])
            
            success = self.runner.run_inference(
                prompt_dir=seq['prompt_dir'],
                output_dir=output_dir,
                config_type=seq['config_type']
            )
            
            result = {
                'sequence_name': seq['name'],
                'success': success,
                'config_type': seq['config_type'],
                'output_dir': output_dir if success else None,
                'panel1_page': seq['panel1_info']['page'],
                'panel2_page': seq['panel2_info']['page']
            }
            
            results.append(result)
            
            if success:
                print(f"     ✅ Completato: {output_dir}")
            else:
                print(f"     ❌ Errore durante la generazione")
        
        return results
    
    def generate_execution_commands(self, sequences: List[Dict]) -> List[Dict]:
        """
        Genera comandi per l'esecuzione manuale
        """
        commands = []
        
        script_path = os.path.join(os.path.dirname(__file__), "advanced_tooncrafter_runner.py")
        
        for seq in sequences:
            output_dir = os.path.join(self.output_base_dir, "animations", seq['name'])
            
            cmd = [
                "python", script_path,
                "--tooncrafter_path", self.tooncrafter_path,
                "--prompt_dir", seq['prompt_dir'],
                "--output_dir", output_dir,
                "--config_type", seq['config_type']
            ]
            
            commands.append({
                'sequence_name': seq['name'],
                'command': ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmd),
                'config_type': seq['config_type']
            })
        
        # Salva script batch per Windows
        batch_file = os.path.join(self.output_base_dir, "run_tooncrafter_batch.bat")
        with open(batch_file, 'w') as f:
            f.write("@echo off\n")
            f.write("echo Esecuzione batch ToonCrafter per manga...\n\n")
            
            for i, cmd_info in enumerate(commands):
                f.write(f"echo Sequenza {i+1}/{len(commands)}: {cmd_info['sequence_name']}\n")
                f.write(f"{cmd_info['command']}\n")
                f.write("if %ERRORLEVEL% neq 0 (\n")
                f.write(f"    echo Errore nella sequenza {cmd_info['sequence_name']}\n")
                f.write("    pause\n")
                f.write(")\n\n")
            
            f.write("echo Batch completato!\n")
            f.write("pause\n")
        
        print(f"📝 Script batch creato: {batch_file}")
        
        return commands


def main():
    parser = argparse.ArgumentParser(description='Pipeline completa Manga to Anime')
    parser.add_argument('--manga_dataset', required=True,
                       help='Path al dataset Manga109')
    parser.add_argument('--tooncrafter_path', required=True,
                       help='Path alla directory ToonCrafter')
    parser.add_argument('--manga_name', required=True,
                       help='Nome del manga da processare')
    parser.add_argument('--output_dir', default='manga_anime_output',
                       help='Directory di output')
    parser.add_argument('--max_sequences', type=int, default=3,
                       help='Numero massimo di sequenze da processare')
    parser.add_argument('--auto_run', action='store_true',
                       help='Esegui automaticamente ToonCrafter (altrimenti genera solo comandi)')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Solo analisi dei panel, senza processamento')
    
    args = parser.parse_args()
    
    # Verifica che i path esistano
    if not os.path.exists(args.manga_dataset):
        print(f"❌ Dataset manga non trovato: {args.manga_dataset}")
        sys.exit(1)
    
    if not os.path.exists(args.tooncrafter_path):
        print(f"❌ ToonCrafter non trovato: {args.tooncrafter_path}")
        sys.exit(1)
    
    # Inizializza pipeline
    pipeline = MangaAnimationPipeline(
        args.manga_dataset, 
        args.tooncrafter_path, 
        args.output_dir
    )
    
    if args.analyze_only:
        # Solo analisi
        analysis = pipeline.analyze_manga_panels(args.manga_name)
        
        if 'error' in analysis:
            print(f"❌ {analysis['error']}")
            sys.exit(1)
        
        print(f"\n📊 Analisi per {args.manga_name}:")
        print(f"   Panel totali: {analysis['total_panels']}")
        print(f"   Pagine: {analysis['pages']}")
        print(f"   Sequenze raccomandate: {len(analysis['recommended_sequences'])}")
        
        for seq in analysis['recommended_sequences'][:5]:  # Mostra solo le prime 5
            print(f"   • {seq['name']} ({seq['recommended_config']})")
    else:
        # Pipeline completa
        report = pipeline.run_full_pipeline(
            args.manga_name, 
            args.max_sequences,
            args.auto_run
        )
        
        if 'error' in report:
            print(f"❌ {report['error']}")
            sys.exit(1)
        
        print(f"\n🎉 Pipeline completata per {args.manga_name}!")
        print(f"   Sequenze processate: {report['processed_sequences']}")
        
        if args.auto_run:
            successful = sum(1 for r in report['results'] if r['success'])
            print(f"   Animazioni generate: {successful}/{len(report['results'])}")
        else:
            print(f"   Comandi generati per esecuzione manuale")
            print(f"   Esegui: {os.path.join(args.output_dir, 'run_tooncrafter_batch.bat')}")


if __name__ == "__main__":
    main()
