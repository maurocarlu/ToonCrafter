#!/usr/bin/env python3
"""
Modulo di preprocessing specializzato per panel manga per ToonCrafter
Implementa funzionalità avanzate di miglioramento specifiche per l'arte manga
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
from typing import Tuple, Dict, List, Optional
from pathlib import Path


class MangaPanelAnalyzer:
    """
    Analizzatore specializzato per panel manga
    """
    
    def __init__(self):
        self.panel_borders = None
        self.content_classification = None
        self.line_art_complexity = 0.0
    
    def detect_panel_borders(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Rileva automaticamente i bordi dei panel manga
        
        Args:
            image: Immagine in formato numpy array (BGR)
            
        Returns:
            Lista di bordi dei panel come (x, y, w, h)
        """
        # Converti in grayscale per l'analisi
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Applica blur gaussiano per ridurre il rumore
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Rileva edges usando Canny
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Trova contorni
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtra contorni per trovare possibili panel
        panel_borders = []
        min_area = image.shape[0] * image.shape[1] * 0.1  # Almeno 10% dell'immagine
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Controlla rapporto aspetto ragionevole per panel manga
                aspect_ratio = w / h
                if 0.5 <= aspect_ratio <= 3.0:
                    panel_borders.append((x, y, w, h))
        
        self.panel_borders = panel_borders
        return panel_borders
    
    def classify_content(self, image: np.ndarray) -> Dict[str, float]:
        """
        Classifica il contenuto del panel (personaggio, sfondo, azione, dialogo)
        
        Args:
            image: Immagine in formato numpy array (BGR)
            
        Returns:
            Dizionario con probabilità per ogni categoria
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Analisi della densità di edges per determinare il tipo di contenuto
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
        
        # Analisi della varianza dell'intensità
        intensity_variance = np.var(gray)
        
        # Rileva possibili balloon di testo (aree bianche circolari/ellittiche)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_balloon_score = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Area minima per balloon
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:  # Forma abbastanza circolare
                        text_balloon_score += circularity
        
        # Classificazione basata su euristica
        classification = {
            'character': min(1.0, edge_density * 3.0),  # Personaggi hanno molti dettagli
            'background': min(1.0, intensity_variance / 1000.0),  # Sfondi hanno varietà tonale
            'action': min(1.0, edge_density * 2.0 if edge_density > 0.1 else 0.0),  # Azioni hanno linee dinamiche
            'dialogue': min(1.0, text_balloon_score / 2.0)  # Presenza di balloon
        }
        
        self.content_classification = classification
        return classification
    
    def analyze_line_art_complexity(self, image: np.ndarray) -> float:
        """
        Analizza la complessità delle linee (line art complexity)
        
        Args:
            image: Immagine in formato numpy array (BGR)
            
        Returns:
            Score di complessità delle linee (0.0-1.0)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Rileva edges con diversi threshold per catturare linee di varia intensità
        edges_strong = cv2.Canny(gray, 100, 200)
        edges_weak = cv2.Canny(gray, 50, 100)
        
        # Calcola densità di edges
        strong_density = np.sum(edges_strong > 0) / (image.shape[0] * image.shape[1])
        weak_density = np.sum(edges_weak > 0) / (image.shape[0] * image.shape[1])
        
        # Analizza la distribuzione delle linee
        # Usa filtro Sobel per intensità del gradiente
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Calcola complessità come combinazione di densità e intensità
        avg_gradient = np.mean(gradient_magnitude)
        complexity = (strong_density * 0.4 + weak_density * 0.3 + min(avg_gradient / 255.0, 1.0) * 0.3)
        
        self.line_art_complexity = min(complexity, 1.0)
        return self.line_art_complexity


class MangaImageEnhancer:
    """
    Modulo di miglioramento immagini specifico per manga
    """
    
    def __init__(self):
        pass
    
    def contrast_adaptive_enhancement(self, image: Image.Image, preserve_lines: bool = True) -> Image.Image:
        """
        Migliora il contrasto preservando le linee sottili del manga
        
        Args:
            image: Immagine PIL
            preserve_lines: Se preservare le linee sottili
            
        Returns:
            Immagine migliorata
        """
        # Converti in numpy per elaborazione
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            # Immagine a colori - lavora in spazio LAB per preservare i colori
            img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l_channel = img_lab[:, :, 0]
            
            # Applica CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_channel)
            
            img_lab[:, :, 0] = l_enhanced
            img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        else:
            # Immagine in grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_enhanced = clahe.apply(img_array)
        
        result = Image.fromarray(img_enhanced)
        
        # Se richiesto, preserva le linee sottili combinando con l'originale
        if preserve_lines:
            # Crea una maschera delle linee
            gray = np.array(image.convert('L'))
            edges = cv2.Canny(gray, 50, 150)
            line_mask = edges > 0
            
            # Combina originale e migliorato dove ci sono linee
            enhanced_array = np.array(result)
            original_array = np.array(image)
            
            if len(enhanced_array.shape) == 3:
                for c in range(enhanced_array.shape[2]):
                    enhanced_array[:, :, c] = np.where(
                        line_mask, 
                        original_array[:, :, c] * 0.7 + enhanced_array[:, :, c] * 0.3,
                        enhanced_array[:, :, c]
                    )
            else:
                enhanced_array = np.where(
                    line_mask,
                    original_array * 0.7 + enhanced_array * 0.3,
                    enhanced_array
                )
            
            result = Image.fromarray(enhanced_array.astype(np.uint8))
        
        return result
    
    def line_art_sharpening(self, image: Image.Image, strength: float = 1.0) -> Image.Image:
        """
        Rinforza i contorni mantenendo i dettagli
        
        Args:
            image: Immagine PIL
            strength: Intensità del sharpening (0.0-2.0)
            
        Returns:
            Immagine con contorni rinformati
        """
        if strength <= 0:
            return image
        
        # Applica un filtro unsharp mask specifico per line art
        img_array = np.array(image)
        
        # Crea kernel per edge enhancement
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * strength * 0.2
        
        if len(img_array.shape) == 3:
            enhanced = np.zeros_like(img_array)
            for c in range(img_array.shape[2]):
                enhanced[:, :, c] = cv2.filter2D(img_array[:, :, c], -1, kernel)
        else:
            enhanced = cv2.filter2D(img_array, -1, kernel)
        
        # Clamp values
        enhanced = np.clip(enhanced, 0, 255)
        
        return Image.fromarray(enhanced.astype(np.uint8))
    
    def noise_reduction(self, image: Image.Image, method: str = 'bilateral') -> Image.Image:
        """
        Rimuove artefatti di scansione tipici dei manga
        
        Args:
            image: Immagine PIL
            method: Metodo di riduzione rumore ('bilateral', 'gaussian', 'median')
            
        Returns:
            Immagine con rumore ridotto
        """
        img_array = np.array(image)
        
        if method == 'bilateral':
            # Bilateral filter preserva i bordi mentre rimuove il rumore
            if len(img_array.shape) == 3:
                denoised = cv2.bilateralFilter(img_array, 9, 75, 75)
            else:
                denoised = cv2.bilateralFilter(img_array, 9, 75, 75)
        elif method == 'gaussian':
            denoised = cv2.GaussianBlur(img_array, (3, 3), 0)
        elif method == 'median':
            denoised = cv2.medianBlur(img_array, 3)
        else:
            return image
        
        return Image.fromarray(denoised)
    
    def tone_normalization(self, image: Image.Image, target_range: Tuple[int, int] = (10, 245)) -> Image.Image:
        """
        Normalizza i toni per coerenza tra panel
        
        Args:
            image: Immagine PIL
            target_range: Range di toni target (min, max)
            
        Returns:
            Immagine con toni normalizzati
        """
        img_array = np.array(image)
        
        # Calcola range attuale
        min_val = np.min(img_array)
        max_val = np.max(img_array)
        
        if max_val - min_val == 0:
            return image
        
        # Normalizza al range target
        normalized = (img_array - min_val) / (max_val - min_val)
        normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
        
        return Image.fromarray(normalized.astype(np.uint8))


class MangaStylePreserver:
    """
    Modulo per preservare lo stile manga durante il preprocessing
    """
    
    def __init__(self):
        pass
    
    def edge_detection_and_reinforcement(self, image: Image.Image, threshold1: int = 50, threshold2: int = 150) -> Image.Image:
        """
        Preserva e rinforza la struttura del line art
        
        Args:
            image: Immagine PIL
            threshold1: Soglia bassa per Canny
            threshold2: Soglia alta per Canny
            
        Returns:
            Immagine con bordi rinforzati
        """
        img_array = np.array(image.convert('L'))
        
        # Rileva edges
        edges = cv2.Canny(img_array, threshold1, threshold2)
        
        # Dilata leggermente i bordi per rinformarlo
        kernel = np.ones((2, 2), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Combina con l'immagine originale
        if image.mode == 'RGB':
            img_array = np.array(image)
            # Scurisci i pixel dove ci sono bordi
            mask = edges_dilated > 0
            for c in range(3):
                img_array[:, :, c] = np.where(mask, img_array[:, :, c] * 0.7, img_array[:, :, c])
            result = Image.fromarray(img_array)
        else:
            img_array = np.where(edges_dilated > 0, img_array * 0.7, img_array)
            result = Image.fromarray(img_array.astype(np.uint8))
        
        return result
    
    def screentone_processing(self, image: Image.Image) -> Dict[str, any]:
        """
        Gestione intelligente dei retini manga
        
        Args:
            image: Immagine PIL
            
        Returns:
            Dizionario con informazioni sui retini e immagine processata
        """
        img_array = np.array(image.convert('L'))
        
        # Rileva pattern di retini usando FFT
        f_transform = np.fft.fft2(img_array)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Cerca picchi che indicano pattern ripetitivi (retini)
        height, width = magnitude_spectrum.shape
        center_y, center_x = height // 2, width // 2
        
        # Analizza frequenze medie per rilevare retini
        freq_mask = np.zeros_like(magnitude_spectrum)
        cv2.circle(freq_mask, (center_x, center_y), min(height, width) // 4, 1, -1)
        
        avg_freq = np.mean(magnitude_spectrum[freq_mask == 1])
        screentone_detected = avg_freq > np.mean(magnitude_spectrum) * 1.2
        
        processing_info = {
            'screentone_detected': screentone_detected,
            'frequency_strength': avg_freq,
            'processed_image': image  # Per ora restituisce l'originale
        }
        
        return processing_info
    
    def text_balloon_detection(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Identifica e maschera i balloon di testo
        
        Args:
            image: Immagine PIL
            
        Returns:
            Lista di bounding box dei balloon (x, y, w, h)
        """
        img_array = np.array(image.convert('L'))
        
        # Threshold per isolare aree bianche (balloon)
        _, binary = cv2.threshold(img_array, 180, 255, cv2.THRESH_BINARY)
        
        # Trova contorni
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        balloon_boxes = []
        min_area = 1000  # Area minima per considerare un balloon
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Calcola circolarità/ellitticità
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # I balloon sono tipicamente circolari/ellittici
                    if 0.2 <= circularity <= 1.0:
                        x, y, w, h = cv2.boundingRect(contour)
                        # Verifica rapporto aspetto ragionevole
                        aspect_ratio = w / h
                        if 0.3 <= aspect_ratio <= 3.0:
                            balloon_boxes.append((x, y, w, h))
        
        return balloon_boxes


class MangaPreprocessor:
    """
    Classe principale per il preprocessing manga-specifico
    """
    
    def __init__(self):
        self.analyzer = MangaPanelAnalyzer()
        self.enhancer = MangaImageEnhancer()
        self.style_preserver = MangaStylePreserver()
    
    def preprocess_manga_panel(self, 
                             image_path: str, 
                             output_path: str,
                             enhancement_options: Dict = None) -> Dict[str, any]:
        """
        Preprocessing completo di un panel manga
        
        Args:
            image_path: Percorso immagine input
            output_path: Percorso immagine output
            enhancement_options: Opzioni di miglioramento
            
        Returns:
            Dizionario con risultati e metriche
        """
        if enhancement_options is None:
            enhancement_options = {
                'contrast_enhancement': True,
                'line_art_sharpening': True,
                'noise_reduction': True,
                'tone_normalization': True,
                'edge_reinforcement': True,
                'preserve_screentones': True
            }
        
        # Carica immagine
        image = Image.open(image_path)
        original_image = image.copy()
        
        # Converti per analisi
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        
        # Analisi del panel
        panel_borders = self.analyzer.detect_panel_borders(img_bgr)
        content_classification = self.analyzer.classify_content(img_bgr)
        line_complexity = self.analyzer.analyze_line_art_complexity(img_bgr)
        
        # Preprocessing step by step
        processed_image = image
        processing_steps = []
        
        # 1. Riduzione rumore (per prima per migliorare gli step successivi)
        if enhancement_options.get('noise_reduction', True):
            processed_image = self.enhancer.noise_reduction(processed_image, 'bilateral')
            processing_steps.append('noise_reduction')
        
        # 2. Miglioramento contrasto adattivo
        if enhancement_options.get('contrast_enhancement', True):
            processed_image = self.enhancer.contrast_adaptive_enhancement(processed_image, preserve_lines=True)
            processing_steps.append('contrast_enhancement')
        
        # 3. Normalizzazione toni
        if enhancement_options.get('tone_normalization', True):
            processed_image = self.enhancer.tone_normalization(processed_image)
            processing_steps.append('tone_normalization')
        
        # 4. Rinforzo line art
        if enhancement_options.get('line_art_sharpening', True):
            # Intensità basata sulla complessità delle linee
            strength = min(1.5, 0.5 + line_complexity)
            processed_image = self.enhancer.line_art_sharpening(processed_image, strength)
            processing_steps.append('line_art_sharpening')
        
        # 5. Rinforzo bordi per preservare stile manga
        if enhancement_options.get('edge_reinforcement', True):
            processed_image = self.style_preserver.edge_detection_and_reinforcement(processed_image)
            processing_steps.append('edge_reinforcement')
        
        # 6. Analisi retini (informativo)
        screentone_info = None
        if enhancement_options.get('preserve_screentones', True):
            screentone_info = self.style_preserver.screentone_processing(processed_image)
        
        # 7. Rilevamento balloon di testo
        text_balloons = self.style_preserver.text_balloon_detection(processed_image)
        
        # Salva immagine processata
        processed_image.save(output_path, 'PNG', quality=95)
        
        # Risultati
        results = {
            'success': True,
            'input_path': image_path,
            'output_path': output_path,
            'processing_steps': processing_steps,
            'analysis': {
                'panel_borders': panel_borders,
                'content_classification': content_classification,
                'line_art_complexity': line_complexity,
                'text_balloons': text_balloons,
                'screentone_info': screentone_info
            },
            'enhancement_options': enhancement_options,
            'original_size': original_image.size,
            'processed_size': processed_image.size
        }
        
        return results