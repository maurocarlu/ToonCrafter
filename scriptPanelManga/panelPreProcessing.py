#!/usr/bin/env python3
"""
Classe per il preprocessing avanzato di panel manga prima della conversione anime
Ottimizzato per preservare la line art e migliorare la qualità della generazione
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
    
    def contrast_adaptive_enhancement(self, image_input, output_path=None, 
                                clahe_clip_limit=3.0, clahe_tile_grid=(8,8),
                                preserve_lines=True, line_threshold=0.1):
        """
        🎯 Contrast Adaptive Enhancement per preservare line art sottili
        
        Args:
            image_input: Path dell'immagine (str) o oggetto PIL.Image  # ✅ CAMBIATO
            output_path (str): Path output (opzionale)
            clahe_clip_limit (float): Limite di clipping per CLAHE (2.0-4.0)
            clahe_tile_grid (tuple): Griglia per l'adattamento locale
            preserve_lines (bool): Se True, applica protezione speciale per le linee
            line_threshold (float): Soglia per la detection delle linee (0.05-0.2)
        
        Returns:
            PIL.Image: Immagine processata
        """
        
        print(f"🎨 === CONTRAST ADAPTIVE ENHANCEMENT ===")
        print(f"🎛️ CLAHE clip_limit: {clahe_clip_limit}")
        print(f"🎛️ Tile grid: {clahe_tile_grid}")
        print(f"🎛️ Preserve lines: {preserve_lines}")
        
        try:
            # ✅ GESTISCI SIA PATH CHE PIL.IMAGE (AGGIUNTO)
            if isinstance(image_input, str):
                print(f"📸 Input: {image_input}")
                with Image.open(image_input) as img:
                    img_rgb = img.convert('RGB')
            elif hasattr(image_input, 'convert'):
                print(f"📸 Input: PIL.Image {image_input.size}")
                img_rgb = image_input.convert('RGB')
            else:
                raise ValueError("Input deve essere un path (str) o PIL.Image")
            
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
        """🔄 Pipeline completo di preprocessing manga"""
        
        # ✅ CORREGGI LA CHIAMATA ALLA CONFIGURAZIONE
        config = pipeline_config or create_manga_preprocessing_config()
        
        print(f"🔄 === PIPELINE PREPROCESSING MANGA ===")
        print(f"📸 Input: {image_path}")
        
        current_image = image_path
        
        # ✅ NUOVO STEP 0: Character Segmentation (se abilitato)
        if config.get('character_segmentation', {}).get('enabled', False):
            print(f"\n👤 STEP 0: Character Segmentation")
            seg_config = config['character_segmentation']
            segmented_image = self.segment_character(
                current_image,
                model=seg_config.get('model', 'isnet-anime')
            )
            if segmented_image:
                current_image = segmented_image
                print("✅ Segmentazione applicata come primo passo.")
            else:
                print("⚠️ Segmentazione fallita, continuo con l'immagine originale.")
        
        # ✅ STEP 1: Noise Reduction
        if config.get('noise_reduction', {}).get('enabled', False):
            print(f"\n🧹 STEP 1: Noise Reduction")
            nr_config = config['noise_reduction']
            
            denoised_img = self.noise_reduction_manga(
                current_image,
                method=nr_config.get('method', 'bilateral'),
                bilateral_d=nr_config.get('bilateral_d', 5),
                bilateral_sigma_color=nr_config.get('bilateral_sigma_color', 50),
                bilateral_sigma_space=nr_config.get('bilateral_sigma_space', 50),
                preserve_edges=nr_config.get('preserve_edges', True)
            )
            
            if denoised_img:
                current_image = denoised_img
                print(f"✅ Noise reduction applicato")
        
        # ✅ STEP 2: Screentone Normalization
        if config.get('screentone_normalization', {}).get('enabled', False):
            print(f"\n✨ STEP 2: Screentone Normalization")
            st_config = config['screentone_normalization']
            
            normalized_img = self.screentone_normalization(
                current_image,
                fft_threshold=st_config.get('fft_threshold', 1.5),
                median_kernel_size=st_config.get('median_kernel_size', 11),
                preserve_rgb=st_config.get('preserve_rgb', True)
            )
            
            if normalized_img:
                current_image = normalized_img
                print(f"✅ Screentone normalization applicata")
        
        # ✅ STEP 3: Edge Reinforcement
        if config.get('edge_reinforcement', {}).get('enabled', False):
            print(f"\n🖋️ STEP 2: Edge Detection e Reinforcement")
            er_config = config['edge_reinforcement']
            
            reinforced_img = self.edge_detection_reinforcement(  # ✅ QUESTO METODO ESISTE
                current_image,
                edge_detection_method=er_config.get('edge_detection_method', 'combined'),
                canny_low=er_config.get('canny_low', 50),
                canny_high=er_config.get('canny_high', 150),
                sobel_threshold=er_config.get('sobel_threshold', 100),
                reinforcement_strength=er_config.get('reinforcement_strength', 0.15),
                blur_before_detection=er_config.get('blur_before_detection', True),
                dilate_edges=er_config.get('dilate_edges', True),
                dilate_iterations=er_config.get('dilate_iterations', 0)
            )
            
            if reinforced_img:
                current_image = reinforced_img
                print(f"✅ Edge reinforcement applicato")
        
        # ✅ STEP 4: Contrast Enhancement (CLAHE)
        if config.get('contrast_enhancement', {}).get('enabled', False):
            print(f"\n🎨 STEP 3: Contrast Enhancement (CLAHE)")
            ce_config = config['contrast_enhancement']
            
            enhanced_img = self.contrast_adaptive_enhancement(  # ✅ QUESTO METODO ESISTE
                current_image,
                clahe_clip_limit=ce_config.get('clahe_clip_limit', 2.0),
                clahe_tile_grid=tuple(ce_config.get('clahe_tile_grid', [8, 8])),
                preserve_lines=ce_config.get('preserve_lines', True),
                line_threshold=ce_config.get('line_threshold', 0.1)
            )
            
            if enhanced_img:
                current_image = enhanced_img
                print(f"✅ Contrast enhancement applicato")
            else:
                print(f"⚠️ Contrast enhancement fallito")
    
        # ✅ SALVATAGGIO FINALE
        if output_path and isinstance(current_image, Image.Image):
            current_image.save(output_path, 'PNG', quality=95)
            print(f"💾 Pipeline completata, salvato: {output_path}")
        
        print(f"🏁 Pipeline preprocessing completata!")
        return current_image if isinstance(current_image, Image.Image) else Image.open(current_image)

    def edge_detection_reinforcement(self, image_input, output_path=None,
                           edge_detection_method='canny', 
                           canny_low=50, canny_high=150,
                           sobel_threshold=100,
                           reinforcement_strength=0.3,
                           blur_before_detection=True,
                           blur_kernel_size=3,
                           dilate_edges=True,
                           dilate_iterations=1):
        """
        🖋️ Edge Detection e Reinforcement per preservare struttura manga
        
        Detecta e rinforza i contorni per mantenere la caratteristica 
        definizione delle linee tipica dei manga
        
        Args:
            image_path (str): Path dell'immagine input
            output_path (str): Path output (opzionale)
            edge_detection_method (str): 'canny', 'sobel', 'laplacian', 'combined'
            canny_low (int): Soglia bassa per Canny (30-70)
            canny_high (int): Soglia alta per Canny (100-200)
            sobel_threshold (int): Soglia per Sobel (50-150)
            reinforcement_strength (float): Intensità rinforzo (0.1-0.8)
            blur_before_detection (bool): Blur preliminare per ridurre noise
            blur_kernel_size (int): Dimensione kernel blur (3, 5, 7)
            dilate_edges (bool): Dilata edges per linee più definite
            dilate_iterations (int): Iterazioni dilatazione (1-3)
        
        Returns:
            PIL.Image: Immagine con edges rinforzati
        """
        
        print(f"🖋️ === EDGE DETECTION E REINFORCEMENT ===")
        print(f"🎛️ Metodo: {edge_detection_method}")
        print(f"🎛️ Reinforcement strength: {reinforcement_strength}")
        print(f"🎛️ Canny thresholds: {canny_low}-{canny_high}")
        
        try:
            # ✅ GESTISCI SIA PATH CHE PIL.IMAGE
            if isinstance(image_input, str):
                # È un path di file
                print(f"📸 Input: {image_input}")
                with Image.open(image_input) as img:
                    img_rgb = img.convert('RGB')
            elif hasattr(image_input, 'convert'):
                # È un oggetto PIL.Image
                print(f"📸 Input: PIL.Image {image_input.size}")
                img_rgb = image_input.convert('RGB')
            else:
                raise ValueError("Input deve essere un path (str) o PIL.Image")
            
            img_array = np.array(img_rgb)
            
            if self.debug_mode:
                original_stats = self._get_image_stats(img_array)
                print(f"📊 Stats originali: {original_stats}")
            
            # ✅ CONVERSIONE IN GRAYSCALE PER EDGE DETECTION
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # ✅ BLUR PRELIMINARE (se richiesto)
            if blur_before_detection:
                img_gray_processed = cv2.GaussianBlur(img_gray, 
                                                    (blur_kernel_size, blur_kernel_size), 
                                                    1.0)
                print(f"🌀 Blur applicato: kernel {blur_kernel_size}x{blur_kernel_size}")
            else:
                img_gray_processed = img_gray.copy()
            
            # ✅ EDGE DETECTION (MULTIPLI METODI)
            edges_combined = None
            
            if edge_detection_method == 'canny':
                edges = cv2.Canny(img_gray_processed, canny_low, canny_high)
                edges_combined = edges
                print(f"🎯 Canny edges: {np.sum(edges > 0)} pixels")
                
            elif edge_detection_method == 'sobel':
                # Sobel X e Y
                sobel_x = cv2.Sobel(img_gray_processed, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(img_gray_processed, cv2.CV_64F, 0, 1, ksize=3)
                sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                edges = (sobel_magnitude > sobel_threshold).astype(np.uint8) * 255
                edges_combined = edges
                print(f"📐 Sobel edges: {np.sum(edges > 0)} pixels")
                
            elif edge_detection_method == 'laplacian':
                laplacian = cv2.Laplacian(img_gray_processed, cv2.CV_64F)
                edges = (np.abs(laplacian) > sobel_threshold).astype(np.uint8) * 255
                edges_combined = edges
                print(f"🌊 Laplacian edges: {np.sum(edges > 0)} pixels")
                
            elif edge_detection_method == 'combined':
                # COMBINAZIONE MULTI-METODO per massima copertura
                
                # Canny per edges principali
                edges_canny = cv2.Canny(img_gray_processed, canny_low, canny_high)
                
                # Sobel per dettagli
                sobel_x = cv2.Sobel(img_gray_processed, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(img_gray_processed, cv2.CV_64F, 0, 1, ksize=3)
                sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                edges_sobel = (sobel_magnitude > sobel_threshold * 0.7).astype(np.uint8) * 255
                
                # Laplacian per texture fine
                laplacian = cv2.Laplacian(img_gray_processed, cv2.CV_64F)
                edges_laplacian = (np.abs(laplacian) > sobel_threshold * 0.5).astype(np.uint8) * 255
                
                # Combina tutti gli edges
                edges_combined = cv2.bitwise_or(edges_canny, 
                                            cv2.bitwise_or(edges_sobel, edges_laplacian))
                
                print(f"🔥 Combined edges:")
                print(f"   - Canny: {np.sum(edges_canny > 0)} pixels")
                print(f"   - Sobel: {np.sum(edges_sobel > 0)} pixels") 
                print(f"   - Laplacian: {np.sum(edges_laplacian > 0)} pixels")
                print(f"   - Combined: {np.sum(edges_combined > 0)} pixels")
            
            # ✅ POST-PROCESSING EDGES
            if dilate_edges:
                kernel = np.ones((3, 3), np.uint8)
                edges_combined = cv2.dilate(edges_combined, kernel, iterations=dilate_iterations)
                print(f"🔧 Edges dilatati: {dilate_iterations} iterazioni")
            
            # ✅ REINFORCEMENT su luminanza (LAB) — sostituisce il blend RGB precedente
            img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            L, A, B = cv2.split(img_lab)

            edges_f = (edges_combined.astype(np.float32) / 255.0)
            L_f = L.astype(np.float32) / 255.0

            L_reinf = L_f - (edges_f * reinforcement_strength)
            L_reinf = np.clip(L_reinf, 0.0, 1.0)
            L_reinf_u8 = (L_reinf * 255).astype(np.uint8)

            img_lab_reinf = cv2.merge([L_reinf_u8, A, B])
            img_reinforced = cv2.cvtColor(img_lab_reinf, cv2.COLOR_LAB2RGB)
            
            # ✅ CONVERSIONE IN PIL IMAGE
            img_pil_reinforced = Image.fromarray(img_reinforced)
            
            # ✅ STATISTICHE FINALI
            if self.debug_mode:
                reinforced_stats = self._get_image_stats(img_reinforced)
                print(f"📊 Stats reinforced: {reinforced_stats}")
                
                # Calcola numero di edges
                edge_density = np.sum(edges_combined > 0) / (edges_combined.shape[0] * edges_combined.shape[1])
                print(f"📏 Edge density: {edge_density:.3f} ({edge_density*100:.1f}%)")
            
            # ✅ VISUALIZZAZIONE COMPARATIVA
            if self.debug_mode:
                self._show_edge_reinforcement_comparison(
                    img_rgb, img_reinforced, edges_combined,
                    f"Edge Reinforcement ({edge_detection_method})"
                )
            
            # ✅ SALVATAGGIO (se richiesto)
            if output_path:
                img_pil_reinforced.save(output_path, 'PNG', quality=95)
                print(f"💾 Salvato: {output_path}")
            
            print(f"✅ Edge Detection e Reinforcement completato!")
            return img_pil_reinforced
            
        except Exception as e:
            print(f"❌ Errore durante edge reinforcement: {e}")
            return None

    def noise_reduction_manga(self, image_input, output_path=None,
                     method='bilateral', 
                     bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75,
                     median_kernel=5,
                     preserve_edges=True):
        """
        🧹 Noise Reduction ottimizzato per manga
        
        Rimuove noise mantenendo i dettagli delle linee
        
        Args:
            image_path (str): Path dell'immagine input
            output_path (str): Path output (opzionale)
            method (str): 'bilateral', 'median', 'gaussian', 'combined'
            bilateral_d (int): Diametro bilateral filter (5-15)
            bilateral_sigma_color (int): Sigma colore bilateral (50-150)
            bilateral_sigma_space (int): Sigma spazio bilateral (50-150)
            median_kernel (int): Kernel size median filter (3, 5, 7)
            preserve_edges (bool): Preserva edges durante denoising
            
        Returns:
            PIL.Image: Immagine denoised
        """
        
        print(f"🧹 === NOISE REDUCTION MANGA ===")
        print(f"🎛️ Metodo: {method}")
        
        try:
            # ✅ GESTISCI SIA PATH CHE PIL.IMAGE
            if isinstance(image_input, str):
                print(f"📸 Input: {image_input}")
                with Image.open(image_input) as img:
                    img_rgb = img.convert('RGB')
            elif hasattr(image_input, 'convert'):
                print(f"📸 Input: PIL.Image {image_input.size}")
                img_rgb = image_input.convert('RGB')
            else:
                raise ValueError("Input deve essere un path (str) o PIL.Image")
            
            img_array = np.array(img_rgb)
            
            if method == 'bilateral':
                # Bilateral filter: ottimo per preservare edges
                img_denoised = cv2.bilateralFilter(img_array, bilateral_d, 
                                                bilateral_sigma_color, bilateral_sigma_space)
                print(f"🎯 Bilateral filter: d={bilateral_d}, σcolor={bilateral_sigma_color}, σspace={bilateral_sigma_space}")
                
            elif method == 'median':
                # Median filter: rimuove noise impulsivo
                img_denoised = cv2.medianBlur(img_array, median_kernel)
                print(f"📐 Median filter: kernel={median_kernel}")
                
            elif method == 'gaussian':
                # Gaussian filter: smoothing generale
                img_denoised = cv2.GaussianBlur(img_array, (5, 5), 1.0)
                print(f"🌀 Gaussian filter applicato")
                
            elif method == 'combined':
                # Combinazione per denoising ottimale
                # 1. Bilateral per preservare edges
                img_temp = cv2.bilateralFilter(img_array, bilateral_d//2, 
                                            bilateral_sigma_color//2, bilateral_sigma_space//2)
                # 2. Median leggero per noise impulsivo
                img_denoised = cv2.medianBlur(img_temp, 3)
                print(f"🔥 Combined denoising applicato")
            
            # Edge preservation enhancement
            if preserve_edges:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_mask = (edges > 0)
                for channel in range(3):
                    img_denoised[edge_mask, channel] = (
                        0.7 * img_array[edge_mask, channel] +
                        0.3 * img_denoised[edge_mask, channel]
                    )
                print(f"🛡️ Edge preservation applicato")
            
            img_pil_denoised = Image.fromarray(img_denoised)
            
            # Visualizzazione
            if self.debug_mode:
                self._show_denoising_comparison(img_rgb, img_denoised, method)
            
            if output_path:
                img_pil_denoised.save(output_path, 'PNG', quality=95)
                print(f"💾 Salvato: {output_path}")
            
            print(f"✅ Noise reduction completato!")
            return img_pil_denoised
            
        except Exception as e:
            print(f"❌ Errore durante noise reduction: {e}")
            return None

    def _show_edge_reinforcement_comparison(self, original, reinforced, edges, title="Edge Reinforcement"):
        """
        🖼️ Mostra comparazione visiva del reinforcement
        """
        plt.figure(figsize=(15, 5))
        
        # Immagine originale
        plt.subplot(1, 3, 1)
        plt.imshow(original)
        plt.title('ORIGINALE')
        plt.axis('off')
        
        # Edges detectati
        plt.subplot(1, 3, 2)
        plt.imshow(edges, cmap='gray')
        plt.title('EDGES DETECTATI\n(linee da rinforzare)')
        plt.axis('off')
        
        # Immagine reinforced
        plt.subplot(1, 3, 3)
        plt.imshow(reinforced)
        plt.title(f'REINFORCED\n{title}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def _show_denoising_comparison(self, original, denoised, method):
        """🖼️ Mostra comparazione denoising"""
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title('ORIGINALE\n(con noise)')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(denoised)
        plt.title(f'DENOISED\n({method})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def screentone_normalization(self, image_input, output_path=None, 
                             fft_threshold=1.5, median_kernel_size=11,
                             preserve_rgb=True):
        """
        ✨ Normalizza i retini (screentones) usando analisi in frequenza (FFT).
        
        Detecta pattern periodici (retini) e li normalizza con un filtro mediano
        per trasformarli in aree di grigio più uniformi.
        
        Args:
            image_input: Path dell'immagine (str) o oggetto PIL.Image
            output_path (str): Path output (opzionale)
            fft_threshold (float): Soglia per rilevare i picchi FFT (più alto = meno sensibile)
            median_kernel_size (int): Dimensione del kernel per il filtro mediano (deve essere dispari)
            preserve_rgb (bool): Se True, preserva i colori originali
            
        Returns:
            PIL.Image: Immagine con retini normalizzati
        """
        print(f"✨ === SCREENTONE NORMALIZATION ===")
        print(f"🎛️ FFT threshold: {fft_threshold}")
        print(f"🎛️ Median kernel: {median_kernel_size}x{median_kernel_size}")
        
        try:
            # ✅ GESTISCI SIA PATH CHE PIL.IMAGE
            if isinstance(image_input, str):
                print(f"📸 Input: {image_input}")
                img_pil = Image.open(image_input)
            elif hasattr(image_input, 'convert'):
                print(f"📸 Input: PIL.Image {image_input.size}")
                img_pil = image_input
            else:
                raise ValueError("Input deve essere un path (str) o PIL.Image")

            # Preserva l'immagine RGB originale
            img_rgb = img_pil.convert('RGB')
            img_array = np.array(img_rgb)
            
            # Converti in grayscale per l'analisi FFT
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # ✅ CALCOLA FFT
            f = np.fft.fft2(img_gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = np.abs(fshift)
            log_spectrum = np.log1p(magnitude_spectrum)
            
            # ✅ ANALISI SPETTRO
            spectrum_std = np.std(log_spectrum)
            spectrum_max = np.max(log_spectrum)
            spectrum_ratio = spectrum_max / np.mean(log_spectrum)
            
            print(f"📊 Analisi spettro:")
            print(f"   - Deviazione standard: {spectrum_std:.2f}")
            print(f"   - Ratio max/media: {spectrum_ratio:.2f}")

            # ✅ RILEVA PATTERN RETINI
            has_screentones = spectrum_std > fft_threshold
            
            # ✅ MOSTRA ANALISI SPETTRO (se in debug mode)
            if self.debug_mode:
                self._show_fft_analysis(img_gray, log_spectrum, spectrum_std)

            if has_screentones:
                # Median sul grayscale per spegnere il pattern periodico
                if median_kernel_size % 2 == 0:
                    median_kernel_size += 1
                gray_med = cv2.medianBlur(img_gray, median_kernel_size)

                if preserve_rgb:
                    # Sostituisci luminanza (LAB) con il gray normalizzato
                    img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(img_lab)
                    l_new = cv2.normalize(gray_med, None, 0, 255, cv2.NORM_MINMAX).astype(l.dtype)
                    img_lab_new = cv2.merge([l_new, a, b])
                    img_normalized = cv2.cvtColor(img_lab_new, cv2.COLOR_LAB2RGB)
                else:
                    img_normalized = cv2.cvtColor(gray_med, cv2.COLOR_GRAY2RGB)

                img_pil_normalized = Image.fromarray(img_normalized)

                if self.debug_mode and hasattr(self, '_show_screentone_comparison'):
                    self._show_screentone_comparison(img_rgb, img_normalized, 
                        f"Screentone Normalization (kernel={median_kernel_size})")
            else:
                print(f"⚪ Nessun retino significativo rilevato (std={spectrum_std:.2f} < soglia {fft_threshold})")
                img_pil_normalized = img_pil.convert('RGB')

            if output_path:
                img_pil_normalized.save(output_path, 'PNG', quality=95)
                print(f"💾 Salvato: {output_path}")

            print("✅ Screentone normalization completata!")
            return img_pil_normalized

        except Exception as e:
            print(f"❌ Errore durante normalizzazione retini: {e}")
            return None
        
    def _show_screentone_comparison(self, original, normalized, title="Screentone Normalization"):
        """
        🖼️ Mostra comparazione visiva della normalizzazione retini
        """
        plt.figure(figsize=(12, 6))
        
        # Immagine originale
        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title('ORIGINALE\n(con retini)')
        plt.axis('off')
        
        # Immagine normalizzata
        plt.subplot(1, 2, 2)
        plt.imshow(normalized)
        plt.title(f'NORMALIZZATA\n{title}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def _show_fft_analysis(self, img_gray, log_spectrum, spectrum_std):
        """
        📊 Mostra l'analisi FFT per il debug dei retini
        """
        plt.figure(figsize=(15, 5))
            
        # Immagine originale grayscale
        plt.subplot(1, 3, 1)
        plt.imshow(img_gray, cmap='gray')
        plt.title('Grayscale Input')
        plt.axis('off')
            
        # Spettro FFT
        plt.subplot(1, 3, 2)
        plt.imshow(log_spectrum, cmap='viridis')
        plt.colorbar(label='Log Magnitude')
        plt.title(f'Spettro FFT Log\n(std={spectrum_std:.2f})')
        plt.axis('off')
            
        # Visualizzazione 3D dello spettro
        ax = plt.subplot(1, 3, 3, projection='3d')
        y, x = np.mgrid[0:log_spectrum.shape[0], 0:log_spectrum.shape[1]]
        ax.plot_surface(x, y, log_spectrum, cmap='viridis', 
                    linewidth=0, antialiased=False, alpha=0.7)
        ax.set_title('Spettro 3D\n(picchi = pattern periodici)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Magnitude')
            
        plt.tight_layout()
        plt.show()
    
    def segment_character(self, image_input, output_path=None, model='u2net', bg_color=(255, 255, 255, 0)):
        """
        👤 Segmentazione personaggio (rembg) con post-processing della maschera.
        Modelli suggeriti per manga B/W: 'u2net' o 'isnet-general-use'.
        """
        print(f"👤 === CHARACTER SEGMENTATION ===")
        print(f"🎛️ Modello rembg: {model}")
        try:
            from rembg import remove as remove_bg, new_session
            session = new_session(model)
        except Exception as e:
            print(f"❌ rembg non disponibile: {e}")
            print("Suggerimento: pip install 'rembg==2.0.56' 'onnxruntime==1.17.3' e riavvia il runtime.")
            return None

        try:
            if isinstance(image_input, str):
                img_pil = Image.open(image_input).convert("RGBA")
            elif hasattr(image_input, 'convert'):
                img_pil = image_input.convert("RGBA")
            else:
                raise ValueError("Input deve essere un path (str) o PIL.Image")

            # Segmentazione
            seg_rgba = remove_bg(img_pil, session=session, bgcolor=bg_color)

            # Post-processing maschera (migliora bordi su line art)
            import numpy as np, cv2
            seg = np.array(seg_rgba)
            alpha = seg[:, :, 3]

            # Threshold + morfologia (chiudi buchi piccoli, rimuovi puntinato retini)
            _, mask = cv2.threshold(alpha, 200, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            # Feather leggero per evitare contorni duri
            mask_blur = cv2.GaussianBlur(mask, (3, 3), 0)

            # Applica nuova alpha
            seg[:, :, 3] = mask_blur
            img_segmented = Image.fromarray(seg)

            print(f"✅ Segmentazione personaggio completata.")
            if self.debug_mode and hasattr(self, '_show_segmentation_comparison'):
                self._show_segmentation_comparison(img_pil, img_segmented)

            if output_path:
                img_segmented.save(output_path, 'PNG')
                print(f"💾 Salvato personaggio segmentato: {output_path}")

            return img_segmented

        except Exception as e:
            print(f"❌ Errore durante la segmentazione del personaggio: {e}")
            return None
    
    def _show_segmentation_comparison(self, original_rgba, segmented_rgba):
        """🖼️ Confronto visivo segmentazione."""
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("   (visualizzazione disabilitata: matplotlib non disponibile)")
            return

        # Garantisce formato RGBA per la visualizzazione
        try:
            orig = original_rgba.convert("RGBA")
        except Exception:
            orig = Image.fromarray(np.array(original_rgba)).convert("RGBA")
        try:
            segm = segmented_rgba.convert("RGBA")
        except Exception:
            segm = Image.fromarray(np.array(segmented_rgba)).convert("RGBA")

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.imshow(orig); plt.title("Originale"); plt.axis("off")
        plt.subplot(1, 2, 2); plt.imshow(segm); plt.title("Segmentato"); plt.axis("off")
        plt.tight_layout(); plt.show()
    
    


def create_manga_preprocessing_config():
    """
    🎨 Configurazione unica completa per preprocessing manga
    Include tutti i metodi ottimizzati: noise reduction, edge reinforcement e CLAHE
    """
    return {
        'noise_reduction': {
            'enabled': True,
            'method': 'bilateral',
            'bilateral_d': 7,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75,
            'preserve_edges': True
        },
        'edge_reinforcement': {
            'enabled': True,
            'edge_detection_method': 'combined',
            'canny_low': 50,
            'canny_high': 150,
            'sobel_threshold': 100,
            'reinforcement_strength': 0.15,
            'blur_before_detection': True,
            'dilate_edges': True,
            'dilate_iterations': 0
        },
        'screentone_normalization': {     
            'enabled': True,
            'fft_threshold': 1.5,
            'median_kernel_size': 11,
            'preserve_rgb': True
        },
        'character_segmentation': {
            'enabled': False,
            'model': 'u2net'
        },
        'contrast_enhancement': {
            'enabled': True,
            'clahe_clip_limit': 2.0,
            'clahe_tile_grid': (4, 4),
            'preserve_lines': True,
            'line_threshold': 0.1
        }
    }