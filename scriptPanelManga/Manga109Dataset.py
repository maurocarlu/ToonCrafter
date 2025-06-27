import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
import logging

class Manga109Dataset(Dataset):
    def __init__(self, root_dir):
        """
        Dataset per Manga109
        
        Args:
            root_dir (string): Directory con tutte le immagini e annotazioni
        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.annotations_dir = os.path.join(root_dir, 'annotations')
        
        # Configura logging
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Leggi i nomi dei manga dalle annotazioni
        self.manga_names = [f.split('.')[0] for f in os.listdir(self.annotations_dir) 
                           if f.endswith('.xml')]
        
        # Lista per memorizzare le informazioni sui pannelli
        self.panels = []
        self._load_panels()
        
        # Verifica immagini
        self._verify_images()
    
    def _load_panels(self):
        """Carica solo le informazioni essenziali dei pannelli"""
        for manga_name in self.manga_names:
            manga_img_dir = os.path.join(self.images_dir, manga_name)
            if not os.path.exists(manga_img_dir):
                self.logger.warning(f"Directory immagini per {manga_name} non trovata: {manga_img_dir}")
                continue
                
            # Carica annotazioni
            annotation_path = os.path.join(self.annotations_dir, f"{manga_name}.xml")
            try:
                tree = ET.parse(annotation_path)
                root = tree.getroot()
                
                # Per ogni pagina nel manga
                for page in root.findall(".//page"):
                    page_number = page.get('index')
                    
                    # Prova diverse estensioni dei file
                    img_path = None
                    for ext in ['.jpg', '.png', '.jpeg']:
                        path = os.path.join(manga_img_dir, f"{page_number}{ext}")
                        if os.path.exists(path):
                            img_path = path
                            break
                    
                    if img_path is None:
                        continue  # Salta se non trova l'immagine
                    
                    # Raccogli informazioni sui pannelli
                    panels = page.findall(".//frame")
                    for panel in panels:
                        try:
                            # Estrai coordinate del pannello dagli ATTRIBUTI
                            xmin = int(panel.get('xmin'))
                            ymin = int(panel.get('ymin'))
                            xmax = int(panel.get('xmax'))
                            ymax = int(panel.get('ymax'))
                            
                            self.panels.append({
                                'manga': manga_name,
                                'page': page_number,
                                'panel_id': panel.get('id'),
                                'img_path': img_path,
                                'bbox': [xmin, ymin, xmax, ymax]
                            })
                        except (TypeError, ValueError):
                            # Salta pannelli con dati incompleti
                            continue
            except Exception as e:
                self.logger.error(f"Errore nel caricamento di {manga_name}: {e}")
    
    def _verify_images(self):
        """Verifica che le immagini esistano e filtra pannelli con immagini mancanti"""
        valid_panels = []
        for panel in self.panels:
            if os.path.exists(panel['img_path']):
                valid_panels.append(panel)
        
        removed = len(self.panels) - len(valid_panels)
        if removed > 0:
            self.logger.info(f"Rimossi {removed} pannelli con immagini mancanti")
        
        self.panels = valid_panels
    
    def __len__(self):
        return len(self.panels)
    
    def __getitem__(self, idx):
        """Restituisce il pannello e i metadati"""
        panel_info = self.panels[idx]
        
        # Carica l'immagine
        img = Image.open(panel_info['img_path'])
        
        # Ritaglia il pannello
        xmin, ymin, xmax, ymax = panel_info['bbox']
        panel = img.crop((xmin, ymin, xmax, ymax))
        
        # Restituisco l'immagine e i metadati completi
        return panel, panel_info['manga']  # Restituisco il nome del manga come etichetta