import os
import sys
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
from typing import List, Tuple, Optional
import argparse
from pathlib import Path

# Aggiungi il path di ToonCrafter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ToonCrafter'))

from Manga109Dataset import Manga109Dataset


class MangaToAnimeConverter:
    """
    Convertitore da panel manga a scene anime utilizzando ToonCrafter con tecniche avanzate
    per gestire transizioni drastiche tra panel
    """
    
    def __init__(self, manga_dataset_path: str, tooncrafter_path: str):
        self.manga_dataset = Manga109Dataset(manga_dataset_path)
        self.tooncrafter_path = tooncrafter_path
        
    def create_intermediate_frames(self, img1: Image.Image, img2: Image.Image, 
                                 num_intermediate: int = 2) -> List[Image.Image]:
        """
        Crea frame intermedi utilizzando tecniche di morphing e blending
        per ridurre il salto drastico tra due panel manga
        """
        # Converte in array numpy
        arr1 = np.array(img1.resize((512, 320)))
        arr2 = np.array(img2.resize((512, 320)))
        
        intermediate_frames = []
        
        for i in range(1, num_intermediate + 1):
            # Calcola il peso per il blending
            weight = i / (num_intermediate + 1)
            
            # Metodo 1: Simple alpha blending
            simple_blend = cv2.addWeighted(arr1, 1-weight, arr2, weight, 0)
            
            # Metodo 2: Optical flow based morphing (più sofisticato)
            morph_frame = self._optical_flow_morph(arr1, arr2, weight)
            
            # Combina i due metodi
            final_frame = cv2.addWeighted(simple_blend, 0.3, morph_frame, 0.7, 0)
            
            # Applica filtri per rendere più fluido
            final_frame = self._apply_smoothing_filters(final_frame)
            
            intermediate_frames.append(Image.fromarray(final_frame.astype(np.uint8)))
            
        return intermediate_frames
    
    def _optical_flow_morph(self, img1: np.ndarray, img2: np.ndarray, 
                           weight: float) -> np.ndarray:
        """
        Utilizza optical flow per creare un morphing più realistico
        """
        # Converte in grayscale per optical flow
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Calcola optical flow
        flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None)
        
        # Crea una griglia di coordinate
        h, w = gray1.shape
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Applica il flow con il peso specificato
        if flow[0] is not None and len(flow[0]) > 0:
            # Interpola il movimento
            new_x = x + flow[0][:, :, 0] * weight
            new_y = y + flow[0][:, :, 1] * weight
            
            # Rimappa l'immagine
            morphed = cv2.remap(img1, new_x, new_y, cv2.INTER_LINEAR)
        else:
            # Fallback al simple blending
            morphed = cv2.addWeighted(img1, 1-weight, img2, weight, 0)
            
        return morphed
    
    def _apply_smoothing_filters(self, img: np.ndarray) -> np.ndarray:
        """
        Applica filtri per rendere l'immagine più fluida e anime-like
        """
        # Converte in PIL per filtri avanzati
        pil_img = Image.fromarray(img.astype(np.uint8))
        
        # Applica un leggero blur per smoothing
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Aumenta leggermente la saturazione per look più anime
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(1.1)
        
        # Aumenta leggermente il contrasto
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.05)
        
        return np.array(pil_img)
    
    def generate_text_prompt(self, panel1_info: dict, panel2_info: dict, 
                           transition_type: str = "smooth") -> str:
        """
        Genera prompt di testo intelligenti basati sui metadati dei panel
        per guidare meglio ToonCrafter
        """
        manga_name = panel1_info['manga']
        
        # Prompt base specifici per manga
        base_prompts = {
            "smooth": f"smooth anime transition, {manga_name} style, fluid movement",
            "action": f"dynamic action scene, {manga_name} manga style, intense movement",
            "emotional": f"emotional scene transition, {manga_name} anime style, dramatic change",
            "dialogue": f"character conversation, {manga_name} manga style, subtle animation"
        }
        
        # Aggiungi dettagli sulla transizione
        prompt = base_prompts.get(transition_type, base_prompts["smooth"])
        prompt += ", high quality anime animation, consistent art style"
        
        return prompt
    
    def prepare_tooncrafter_input(self, panel1: Image.Image, panel2: Image.Image, 
                                panel1_info: dict, panel2_info: dict,
                                output_dir: str, sequence_name: str) -> str:
        """
        Prepara l'input per ToonCrafter con pre-processing avanzato
        """
        # Ridimensiona i panel alle dimensioni richieste da ToonCrafter
        target_size = (512, 320)
        panel1_resized = panel1.resize(target_size, Image.Resampling.LANCZOS)
        panel2_resized = panel2.resize(target_size, Image.Resampling.LANCZOS)
        
        # Crea frame intermedi per transizioni più fluide
        intermediate_frames = self.create_intermediate_frames(
            panel1_resized, panel2_resized, num_intermediate=2
        )
        
        # Crea directory per i prompt
        prompt_dir = os.path.join(output_dir, "prompts", sequence_name)
        os.makedirs(prompt_dir, exist_ok=True)
        
        # Salva il primo e ultimo frame (ToonCrafter ha bisogno di frame1 e frame3)
        frame1_path = os.path.join(prompt_dir, f"{sequence_name}_frame1.png")
        frame3_path = os.path.join(prompt_dir, f"{sequence_name}_frame3.png")
        
        panel1_resized.save(frame1_path)
        panel2_resized.save(frame3_path)
        
        # Genera prompt di testo intelligente
        prompt_text = self.generate_text_prompt(panel1_info, panel2_info)
        
        # Salva il prompt
        prompt_file = os.path.join(prompt_dir, "prompts.txt")
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt_text)
        
        return prompt_dir
    
    def process_manga_sequence(self, manga_name: str, max_panels: int = 10, 
                             output_dir: str = "output_anime_sequences"):
        """
        Processa una sequenza di panel di un manga specifico
        """
        # Filtra panel dello stesso manga
        manga_panels = [p for p in self.manga_dataset.panels if p['manga'] == manga_name]
        
        if len(manga_panels) < 2:
            print(f"Non ci sono abbastanza panel per {manga_name}")
            return
        
        # Ordina per pagina e pannello
        manga_panels.sort(key=lambda x: (x['page'], x['panel_id']))
        
        # Limita il numero di panel da processare
        manga_panels = manga_panels[:max_panels]
        
        sequences_created = []
        
        # Crea sequenze tra panel consecutivi
        for i in range(len(manga_panels) - 1):
            panel1_info = manga_panels[i]
            panel2_info = manga_panels[i + 1]
            
            # Carica le immagini dei panel
            panel1_img, _ = self.manga_dataset[self.manga_dataset.panels.index(panel1_info)]
            panel2_img, _ = self.manga_dataset[self.manga_dataset.panels.index(panel2_info)]
            
            # Prepara input per ToonCrafter
            sequence_name = f"{manga_name}_seq_{i:03d}"
            prompt_dir = self.prepare_tooncrafter_input(
                panel1_img, panel2_img, panel1_info, panel2_info,
                output_dir, sequence_name
            )
            
            sequences_created.append({
                'sequence_name': sequence_name,
                'prompt_dir': prompt_dir,
                'panel1': panel1_info,
                'panel2': panel2_info
            })
            
            print(f"Creata sequenza: {sequence_name}")
        
        return sequences_created
    
    def run_tooncrafter_batch(self, sequences: List[dict], 
                            tooncrafter_config: dict = None):
        """
        Esegue ToonCrafter su un batch di sequenze con parametri ottimizzati
        """
        if tooncrafter_config is None:
            # Parametri ottimizzati per transizioni drastiche di manga
            tooncrafter_config = {
                'frame_stride': 15,  # Aumentato per movimenti più ampi
                'ddim_steps': 75,    # Più steps per migliore qualità
                'unconditional_guidance_scale': 10.0,  # Più guidance per seguire il prompt
                'video_length': 16,
                'guidance_rescale': 0.8,  # Migliore balance
                'height': 320,
                'width': 512
            }
        
        # Script ToonCrafter
        tooncrafter_script = os.path.join(
            self.tooncrafter_path, 'scripts', 'evaluation', 'inference.py'
        )
        
        for seq in sequences:
            cmd_args = [
                'python', tooncrafter_script,
                '--config', os.path.join(self.tooncrafter_path, 'configs/inference_512_v1.0.yaml'),
                '--ckpt_path', os.path.join(self.tooncrafter_path, 'checkpoints/tooncrafter_512_interp_v1/model.ckpt'),
                '--prompt_dir', seq['prompt_dir'],
                '--savedir', os.path.join(seq['prompt_dir'], '..', '..', 'results', seq['sequence_name']),
                '--frame_stride', str(tooncrafter_config['frame_stride']),
                '--ddim_steps', str(tooncrafter_config['ddim_steps']),
                '--unconditional_guidance_scale', str(tooncrafter_config['unconditional_guidance_scale']),
                '--video_length', str(tooncrafter_config['video_length']),
                '--guidance_rescale', str(tooncrafter_config['guidance_rescale']),
                '--height', str(tooncrafter_config['height']),
                '--width', str(tooncrafter_config['width']),
                '--text_input',
                '--interp',
                '--perframe_ae'
            ]
            
            print(f"Eseguendo ToonCrafter per {seq['sequence_name']}...")
            print(f"Comando: {' '.join(cmd_args)}")
            
            # Esegui il comando (puoi decommentare per esecuzione automatica)
            # os.system(' '.join(cmd_args))


def main():
    parser = argparse.ArgumentParser(description='Converti panel manga in scene anime')
    parser.add_argument('--manga_dataset', required=True, 
                       help='Path al dataset Manga109')
    parser.add_argument('--tooncrafter_path', required=True,
                       help='Path alla directory ToonCrafter')
    parser.add_argument('--manga_name', required=True,
                       help='Nome del manga da processare')
    parser.add_argument('--max_panels', type=int, default=5,
                       help='Numero massimo di panel da processare')
    parser.add_argument('--output_dir', default='output_anime_sequences',
                       help='Directory di output')
    
    args = parser.parse_args()
    
    # Crea il convertitore
    converter = MangaToAnimeConverter(args.manga_dataset, args.tooncrafter_path)
    
    # Processa il manga
    sequences = converter.process_manga_sequence(
        args.manga_name, args.max_panels, args.output_dir
    )
    
    if sequences:
        print(f"Create {len(sequences)} sequenze per {args.manga_name}")
        
        # Genera i comandi per ToonCrafter
        converter.run_tooncrafter_batch(sequences)
        
        print("\nPer eseguire ToonCrafter, usa i comandi generati sopra")
        print("oppure modifica il codice per esecuzione automatica")
    

if __name__ == "__main__":
    main()
