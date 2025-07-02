#!/usr/bin/env python3
"""
Classe per il preprocessing avanzato di panel manga prima della conversione anime
Ottimizzato per preservare line art e migliorare la qualità della generazione
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt

class PanelPreProcessor:
    """
    🎨 Classe per preprocessing avanzato di panel manga
    Migliora la qualità delle immagini per ottimizzare la generazione anime
    """
    
    def __init__(self, debug_mode=True):
        """
        Inizializza il preprocessor
        
        Args:
            debug_mode (bool): Se True, mostra visualizzazioni step-by-step
        """
        self.debug_mode = debug_mode
    
    def contrast_adaptive_enhancement(self, image_path, output_path=None, 
                                    clahe_clip_limit=3.0, clahe_tile_grid=(8,8),
                                    preserve_lines=True, line_threshold=0.1):
        """
        🎯 Contrast Adaptive Enhancement per preservare line art sottili
        
        Utilizza CLAHE (Contrast Limited Adaptive Histogram Equalization) 
        per migliorare il contrasto mantenendo i dettagli delle linee
        
        Args:
            image_path (str): Path dell'immagine input
            output_path (str): Path output (opzionale)
            clahe_clip_limit (float): Limite di clipping per CLAHE (2.0-4.0)
            clahe_tile_grid (tuple): Griglia per l'adattamento locale
            preserve_lines (bool): Se True, applica protezione speciale per le linee
            line_threshold (float): Soglia per la detection delle linee (0.05-0.2)
        
        Returns:
            PIL.Image: Immagine processata
        """
        
        print(f"🎨 === CONTRAST ADAPTIVE ENHANCEMENT ===")
        print(f"📸 Input: {image_path}")
        print(f"🎛️ CLAHE clip_limit: {clahe_clip_limit}")
        print(f"🎛️ Tile grid: {clahe_tile_grid}")
        print(f"🎛️ Preserve lines: {preserve_lines}")
        
        try:
            # ✅ CARICA IMMAGINE
            with Image.open(image_path) as img:
                img_rgb = img.convert('RGB')
                img_array = np.array(img_rgb)
                
                if self.debug_mode:
                    original_stats = self._get_image_stats(img_array)
                    print(f"📊 Stats originali: {original_stats}")
                
                # ✅ CONVERSIONE IN LAB COLOR SPACE
                # LAB è ideale per miglioramenti di luminosità/contrasto
                img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                l_channel, a_channel, b_channel = cv2.split(img_lab)
                
                # ✅ DETECTION LINEE (se richiesta)
                line_mask = None
                if preserve_lines:
                    line_mask = self._detect_line_art(l_channel, line_threshold)
                    print(f"🖋️ Linee detectate: {np.sum(line_mask > 0)} pixels")
                
                # ✅ APPLICAZIONE CLAHE SUL CANALE L (LUMINOSITÀ)
                clahe = cv2.createCLAHE(
                    clipLimit=clahe_clip_limit, 
                    tileGridSize=clahe_tile_grid
                )
                
                l_enhanced = clahe.apply(l_channel)
                
                # ✅ PROTEZIONE LINEE (se richiesta)
                if preserve_lines and line_mask is not None:
                    # Blend conservativo sulle aree di linee
                    blend_factor = 0.7  # Mantieni 70% originale sulle linee
                    line_areas = line_mask > 0
                    l_enhanced[line_areas] = (
                        blend_factor * l_channel[line_areas] + 
                        (1 - blend_factor) * l_enhanced[line_areas]
                    ).astype(np.uint8)
                    print(f"🛡️ Protezione linee applicata con blend factor: {blend_factor}")
                
                # ✅ RICOSTRUZIONE IMMAGINE LAB
                img_lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
                
                # ✅ CONVERSIONE BACK TO RGB
                img_enhanced = cv2.cvtColor(img_lab_enhanced, cv2.COLOR_LAB2RGB)
                
                # ✅ CONVERSIONE IN PIL IMAGE
                img_pil_enhanced = Image.fromarray(img_enhanced)
                
                # ✅ STATISTICHE FINALI
                if self.debug_mode:
                    enhanced_stats = self._get_image_stats(img_enhanced)
                    print(f"📊 Stats enhanced: {enhanced_stats}")
                    
                    # Calcola miglioramento contrasto
                    contrast_improvement = enhanced_stats['std'] / original_stats['std']
                    print(f"📈 Miglioramento contrasto: {contrast_improvement:.2f}x")
                
                # ✅ VISUALIZZAZIONE COMPARATIVA
                if self.debug_mode:
                    self._show_enhancement_comparison(
                        img_rgb, img_enhanced, line_mask,
                        f"CLAHE Enhancement (clip={clahe_clip_limit})"
                    )
                
                # ✅ SALVATAGGIO (se richiesto)
                if output_path:
                    img_pil_enhanced.save(output_path, 'PNG', quality=95)
                    print(f"💾 Salvato: {output_path}")
                
                print(f"✅ Contrast Adaptive Enhancement completato!")
                return img_pil_enhanced
                
        except Exception as e:
            print(f"❌ Errore durante enhancement: {e}")
            return None
    
    def _detect_line_art(self, gray_channel, threshold=0.1):
        """
        🖋️ Detecta line art utilizzando edge detection avanzato
        """
        # Gaussian blur leggero per ridurre noise
        blurred = cv2.GaussianBlur(gray_channel, (3, 3), 1.0)
        
        # Edge detection con Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilatazione leggera per catturare linee sottili
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Normalizza come maschera
        line_mask = (dilated_edges > 0).astype(np.uint8) * 255
        
        return line_mask
    
    def _get_image_stats(self, img_array):
        """
        📊 Calcola statistiche dell'immagine per analisi qualità
        """
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        
        return {
            'mean': np.mean(gray),
            'std': np.std(gray),
            'min': np.min(gray),
            'max': np.max(gray),
            'contrast_ratio': np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0
        }
    
    def _show_enhancement_comparison(self, original, enhanced, line_mask=None, title="Enhancement"):
        """
        🖼️ Mostra comparazione visiva del preprocessing
        """
        fig_cols = 3 if line_mask is not None else 2
        fig_width = fig_cols * 5
        
        plt.figure(figsize=(fig_width, 6))
        
        # Immagine originale
        plt.subplot(1, fig_cols, 1)
        plt.imshow(original)
        plt.title('ORIGINALE')
        plt.axis('off')
        
        # Immagine enhanced
        plt.subplot(1, fig_cols, 2)
        plt.imshow(enhanced)
        plt.title(f'ENHANCED\n{title}')
        plt.axis('off')
        
        # Maschera linee (se disponibile)
        if line_mask is not None:
            plt.subplot(1, fig_cols, 3)
            plt.imshow(line_mask, cmap='gray')
            plt.title('LINE MASK\n(aree protette)')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def apply_preprocessing_pipeline(self, image_path, output_path=None, pipeline_config=None):
        """
        🔄 Pipeline completo di preprocessing personalizzabile
        
        Args:
            image_path (str): Path immagine input
            output_path (str): Path output
            pipeline_config (dict): Configurazione pipeline
        
        Returns:
            PIL.Image: Immagine processata
        """
        
        # Configurazione default
        default_config = {
            'contrast_enhancement': {
                'enabled': True,
                'clahe_clip_limit': 3.0,
                'clahe_tile_grid': (8, 8),
                'preserve_lines': True,
                'line_threshold': 0.1
            }
            # Placeholder per future funzioni:
            # 'noise_reduction': {...},
            # 'color_normalization': {...},
            # 'sharpening': {...}
        }
        
        config = pipeline_config or default_config
        
        print(f"🔄 === PIPELINE PREPROCESSING ===")
        print(f"📸 Input: {image_path}")
        
        current_image = image_path
        
        # ✅ STEP 1: Contrast Enhancement
        if config.get('contrast_enhancement', {}).get('enabled', False):
            print(f"\n🎨 STEP 1: Contrast Adaptive Enhancement")
            ce_config = config['contrast_enhancement']
            
            enhanced_img = self.contrast_adaptive_enhancement(
                current_image,
                clahe_clip_limit=ce_config.get('clahe_clip_limit', 3.0),
                clahe_tile_grid=tuple(ce_config.get('clahe_tile_grid', [8, 8])),
                preserve_lines=ce_config.get('preserve_lines', True),
                line_threshold=ce_config.get('line_threshold', 0.1)
            )
            
            if enhanced_img:
                current_image = enhanced_img
                print(f"✅ Contrast enhancement applicato")
            else:
                print(f"⚠️ Contrast enhancement fallito, continuo con originale")
                current_image = Image.open(image_path)
        
        # ✅ SALVATAGGIO FINALE
        if output_path and isinstance(current_image, Image.Image):
            current_image.save(output_path, 'PNG', quality=95)
            print(f"💾 Pipeline completata, salvato: {output_path}")
        
        print(f"🏁 Pipeline preprocessing completata!")
        return current_image if isinstance(current_image, Image.Image) else Image.open(current_image)

# ===== UTILITY FUNCTIONS =====

def create_default_preprocessing_config():
    """
    🔧 Crea configurazione default per preprocessing
    """
    return {
        'contrast_enhancement': {
            'enabled': True,
            'clahe_clip_limit': 3.0,
            'clahe_tile_grid': (8, 8),
            'preserve_lines': True,
            'line_threshold': 0.1
        }
    }

def create_aggressive_preprocessing_config():
    """
    ⚡ Configurazione aggressiva per manga con contrasto basso
    """
    return {
        'contrast_enhancement': {
            'enabled': True,
            'clahe_clip_limit': 4.0,  # Più aggressivo
            'clahe_tile_grid': (6, 6),  # Griglia più fine
            'preserve_lines': True,
            'line_threshold': 0.15  # Più sensibile alle linee
        }
    }

def create_conservative_preprocessing_config():
    """
    🛡️ Configurazione conservativa per manga già di buona qualità
    """
    return {
        'contrast_enhancement': {
            'enabled': True,
            'clahe_clip_limit': 2.0,  # Più conservativo
            'clahe_tile_grid': (10, 10),  # Griglia più ampia
            'preserve_lines': True,
            'line_threshold': 0.08  # Meno sensibile
        }
    }