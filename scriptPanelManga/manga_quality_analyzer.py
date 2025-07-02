#!/usr/bin/env python3
"""
Modulo di analisi qualità per preprocessing manga
Valuta la qualità dell'input e suggerisce ottimizzazioni per ToonCrafter
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import os
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Struttura per le metriche di qualità"""
    sharpness_score: float
    contrast_score: float
    noise_level: float
    line_art_quality: float
    overall_score: float
    success_probability: float


class MangaQualityAnalyzer:
    """
    Analizzatore di qualità specializzato per panel manga
    """
    
    def __init__(self):
        self.quality_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'fair': 0.4,
            'poor': 0.2
        }
    
    def calculate_sharpness_score(self, image: np.ndarray) -> float:
        """
        Calcola lo score di nitidezza dell'immagine
        
        Args:
            image: Immagine in formato numpy array
            
        Returns:
            Score di nitidezza (0.0-1.0)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Usa il filtro Laplaciano per misurare la nitidezza
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalizza il risultato (valori tipici: 0-2000)
        sharpness = min(laplacian_var / 1000.0, 1.0)
        
        return sharpness
    
    def calculate_contrast_score(self, image: np.ndarray) -> float:
        """
        Calcola lo score di contrasto dell'immagine
        
        Args:
            image: Immagine in formato numpy array
            
        Returns:
            Score di contrasto (0.0-1.0)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calcola la deviazione standard come misura del contrasto
        contrast = np.std(gray) / 128.0  # Normalizza per max std possibile
        
        return min(contrast, 1.0)
    
    def calculate_noise_level(self, image: np.ndarray) -> float:
        """
        Stima il livello di rumore nell'immagine
        
        Args:
            image: Immagine in formato numpy array
            
        Returns:
            Livello di rumore (0.0-1.0, dove 0 = nessun rumore)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Usa un filtro mediano per stimare il rumore
        median_filtered = cv2.medianBlur(gray, 5)
        noise = np.mean(np.abs(gray.astype(float) - median_filtered.astype(float)))
        
        # Normalizza (valori tipici: 0-50)
        noise_level = min(noise / 25.0, 1.0)
        
        return noise_level
    
    def calculate_line_art_quality(self, image: np.ndarray) -> float:
        """
        Valuta la qualità del line art per manga
        
        Args:
            image: Immagine in formato numpy array
            
        Returns:
            Score di qualità line art (0.0-1.0)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Rileva bordi con diversi threshold
        edges_strong = cv2.Canny(gray, 100, 200)
        edges_weak = cv2.Canny(gray, 50, 100)
        
        # Calcola densità e coerenza dei bordi
        strong_density = np.sum(edges_strong > 0) / (gray.shape[0] * gray.shape[1])
        weak_density = np.sum(edges_weak > 0) / (gray.shape[0] * gray.shape[1])
        
        # Analizza la continuità delle linee
        # Usa morfologia per valutare la connettività
        kernel = np.ones((3, 3), np.uint8)
        edges_closed = cv2.morphologyEx(edges_strong, cv2.MORPH_CLOSE, kernel)
        connectivity = np.sum(edges_closed > 0) / max(np.sum(edges_strong > 0), 1)
        
        # Calcola score combinato
        # Bilancia densità (presenza di linee) e continuità
        line_quality = (strong_density * 0.4 + weak_density * 0.3 + connectivity * 0.3)
        
        return min(line_quality * 3.0, 1.0)  # Scala per range appropriato
    
    def analyze_resolution_adequacy(self, image: np.ndarray, target_size: Tuple[int, int] = (512, 320)) -> Dict[str, any]:
        """
        Analizza se la risoluzione è adeguata per ToonCrafter
        
        Args:
            image: Immagine in formato numpy array
            target_size: Dimensioni target (width, height)
            
        Returns:
            Dizionario con analisi della risoluzione
        """
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        # Calcola fattori di scala
        scale_x = width / target_width
        scale_y = height / target_height
        
        # Determina se serve upscaling o downscaling
        needs_upscaling = scale_x < 1.0 or scale_y < 1.0
        needs_downscaling = scale_x > 2.0 or scale_y > 2.0
        
        # Calcola la perdita di qualità stimata dal rescaling
        quality_loss = 0.0
        if needs_upscaling:
            quality_loss = (1.0 - min(scale_x, scale_y)) * 0.3  # Max 30% loss per upscaling
        elif needs_downscaling:
            quality_loss = min((max(scale_x, scale_y) - 2.0) / 4.0, 0.4)  # Max 40% loss per heavy downscaling
        
        adequacy_score = 1.0 - quality_loss
        
        return {
            'current_size': (width, height),
            'target_size': target_size,
            'scale_factors': (scale_x, scale_y),
            'needs_upscaling': needs_upscaling,
            'needs_downscaling': needs_downscaling,
            'estimated_quality_loss': quality_loss,
            'adequacy_score': adequacy_score,
            'recommendation': self._get_resolution_recommendation(scale_x, scale_y)
        }
    
    def _get_resolution_recommendation(self, scale_x: float, scale_y: float) -> str:
        """Genera raccomandazione basata sui fattori di scala"""
        if scale_x < 0.5 or scale_y < 0.5:
            return "Risoluzione troppo bassa - considera immagine di qualità superiore"
        elif scale_x < 1.0 or scale_y < 1.0:
            return "Risoluzione bassa - risultato potrebbe essere sfocato"
        elif scale_x > 4.0 or scale_y > 4.0:
            return "Risoluzione molto alta - considera pre-ridimensionamento"
        elif scale_x > 2.0 or scale_y > 2.0:
            return "Risoluzione alta - downsample preserverà la qualità"
        else:
            return "Risoluzione ottimale per ToonCrafter"
    
    def detect_problematic_elements(self, image: np.ndarray) -> Dict[str, any]:
        """
        Rileva elementi che potrebbero causare problemi nella generazione
        
        Args:
            image: Immagine in formato numpy array
            
        Returns:
            Dizionario con elementi problematici rilevati
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        problems = {}
        
        # 1. Aree troppo scure o chiare
        dark_ratio = np.sum(gray < 30) / gray.size
        bright_ratio = np.sum(gray > 225) / gray.size
        
        problems['extreme_values'] = {
            'dark_ratio': dark_ratio,
            'bright_ratio': bright_ratio,
            'problematic': dark_ratio > 0.3 or bright_ratio > 0.3
        }
        
        # 2. Dettagli molto fini che potrebbero perdersi
        # Rileva linee molto sottili
        kernel_thin = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        thin_lines = cv2.filter2D(gray, -1, kernel_thin)
        thin_line_ratio = np.sum(np.abs(thin_lines) > 50) / gray.size
        
        problems['fine_details'] = {
            'thin_line_ratio': thin_line_ratio,
            'problematic': thin_line_ratio > 0.1
        }
        
        # 3. Testo troppo piccolo
        # Rileva possibili caratteri di testo molto piccoli
        contours, _ = cv2.findContours(
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        small_text_count = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 5 <= w <= 20 and 5 <= h <= 20:  # Possibile carattere piccolo
                small_text_count += 1
        
        problems['small_text'] = {
            'count': small_text_count,
            'problematic': small_text_count > 20
        }
        
        # 4. Composizione troppo complessa
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        problems['complexity'] = {
            'edge_density': edge_density,
            'problematic': edge_density > 0.3
        }
        
        return problems
    
    def calculate_overall_quality_metrics(self, image_path: str) -> QualityMetrics:
        """
        Calcola metriche di qualità complete per l'immagine
        
        Args:
            image_path: Percorso dell'immagine da analizzare
            
        Returns:
            Oggetto QualityMetrics con tutti i punteggi
        """
        # Carica immagine
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Impossibile caricare immagine: {image_path}")
        
        # Calcola metriche individuali
        sharpness = self.calculate_sharpness_score(image)
        contrast = self.calculate_contrast_score(image)
        noise_level = self.calculate_noise_level(image)
        line_art_quality = self.calculate_line_art_quality(image)
        
        # Calcola score complessivo (media pesata)
        weights = {
            'sharpness': 0.25,
            'contrast': 0.20,
            'noise': 0.15,  # Inverso - meno rumore è meglio
            'line_art': 0.40
        }
        
        overall_score = (
            sharpness * weights['sharpness'] +
            contrast * weights['contrast'] +
            (1.0 - noise_level) * weights['noise'] +  # Inverti noise_level
            line_art_quality * weights['line_art']
        )
        
        # Calcola probabilità di successo basata su score e problemi
        resolution_analysis = self.analyze_resolution_adequacy(image)
        problems = self.detect_problematic_elements(image)
        
        # Penalizzazioni per problemi
        problem_penalty = 0.0
        if problems['extreme_values']['problematic']:
            problem_penalty += 0.1
        if problems['fine_details']['problematic']:
            problem_penalty += 0.15
        if problems['small_text']['problematic']:
            problem_penalty += 0.05
        if problems['complexity']['problematic']:
            problem_penalty += 0.1
        
        success_probability = max(0.0, overall_score - problem_penalty)
        success_probability *= resolution_analysis['adequacy_score']
        
        return QualityMetrics(
            sharpness_score=sharpness,
            contrast_score=contrast,
            noise_level=noise_level,
            line_art_quality=line_art_quality,
            overall_score=overall_score,
            success_probability=success_probability
        )
    
    def suggest_optimizations(self, image_path: str) -> Dict[str, any]:
        """
        Analizza l'immagine e suggerisce ottimizzazioni automatiche
        
        Args:
            image_path: Percorso dell'immagine da analizzare
            
        Returns:
            Dizionario con suggerimenti di ottimizzazione
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Impossibile caricare immagine: {image_path}")
        
        metrics = self.calculate_overall_quality_metrics(image_path)
        resolution_analysis = self.analyze_resolution_adequacy(image)
        problems = self.detect_problematic_elements(image)
        
        suggestions = {
            'preprocessing_recommendations': {},
            'parameter_adjustments': {},
            'quality_assessment': {
                'overall_grade': self._get_quality_grade(metrics.overall_score),
                'success_probability': metrics.success_probability,
                'main_issues': []
            }
        }
        
        # Suggerimenti preprocessing
        if metrics.sharpness_score < 0.5:
            suggestions['preprocessing_recommendations']['sharpening'] = {
                'recommended': True,
                'strength': min(2.0, 1.0 / max(metrics.sharpness_score, 0.1)),
                'reason': 'Immagine sfocata - necessario sharpening'
            }
            suggestions['quality_assessment']['main_issues'].append('Bassa nitidezza')
        
        if metrics.contrast_score < 0.4:
            suggestions['preprocessing_recommendations']['contrast_enhancement'] = {
                'recommended': True,
                'method': 'adaptive',
                'reason': 'Contrasto basso - necessario miglioramento adattivo'
            }
            suggestions['quality_assessment']['main_issues'].append('Contrasto insufficiente')
        
        if metrics.noise_level > 0.3:
            suggestions['preprocessing_recommendations']['noise_reduction'] = {
                'recommended': True,
                'method': 'bilateral',
                'strength': min(metrics.noise_level * 2.0, 1.0),
                'reason': 'Rumore elevato - necessaria riduzione'
            }
            suggestions['quality_assessment']['main_issues'].append('Presenza di rumore')
        
        if metrics.line_art_quality < 0.5:
            suggestions['preprocessing_recommendations']['line_enhancement'] = {
                'recommended': True,
                'edge_reinforcement': True,
                'reason': 'Line art debole - necessario rinforzo bordi'
            }
            suggestions['quality_assessment']['main_issues'].append('Line art di bassa qualità')
        
        # Suggerimenti parametri ToonCrafter
        if metrics.success_probability < 0.6:
            suggestions['parameter_adjustments']['guidance_scale'] = {
                'recommended_value': 10.0,  # Più alto per immagini problematiche
                'reason': 'Score qualità basso - aumenta guidance scale'
            }
            suggestions['parameter_adjustments']['ddim_steps'] = {
                'recommended_value': 75,  # Più steps per qualità migliore
                'reason': 'Necessari più steps per gestire complessità'
            }
        
        if problems['complexity']['problematic']:
            suggestions['parameter_adjustments']['frame_stride'] = {
                'recommended_value': 5,  # Movimento più sottile
                'reason': 'Immagine complessa - riduci frame stride'
            }
        
        # Raccomandazioni generali
        if not suggestions['quality_assessment']['main_issues']:
            suggestions['quality_assessment']['main_issues'].append('Nessun problema significativo rilevato')
        
        return suggestions
    
    def _get_quality_grade(self, score: float) -> str:
        """Converte score numerico in grado qualitativo"""
        if score >= self.quality_thresholds['excellent']:
            return 'Eccellente'
        elif score >= self.quality_thresholds['good']:
            return 'Buona'
        elif score >= self.quality_thresholds['fair']:
            return 'Discreta'
        else:
            return 'Scarsa'
    
    def generate_quality_report(self, image_path: str, output_path: Optional[str] = None) -> str:
        """
        Genera un report dettagliato della qualità dell'immagine
        
        Args:
            image_path: Percorso dell'immagine da analizzare
            output_path: Percorso opzionale per salvare il report
            
        Returns:
            Report testuale dettagliato
        """
        try:
            metrics = self.calculate_overall_quality_metrics(image_path)
            suggestions = self.suggest_optimizations(image_path)
            
            # Genera report
            report_lines = [
                "=" * 60,
                f"REPORT QUALITÀ MANGA PANEL",
                "=" * 60,
                f"File: {os.path.basename(image_path)}",
                "",
                "📊 METRICHE DI QUALITÀ:",
                f"  • Nitidezza: {metrics.sharpness_score:.2f}/1.00",
                f"  • Contrasto: {metrics.contrast_score:.2f}/1.00", 
                f"  • Livello rumore: {metrics.noise_level:.2f}/1.00",
                f"  • Qualità line art: {metrics.line_art_quality:.2f}/1.00",
                "",
                f"🎯 SCORE COMPLESSIVO: {metrics.overall_score:.2f}/1.00",
                f"📈 PROBABILITÀ SUCCESSO: {metrics.success_probability:.1%}",
                f"🏆 GRADO QUALITÀ: {suggestions['quality_assessment']['overall_grade']}",
                "",
                "⚠️  PROBLEMI RILEVATI:",
            ]
            
            for issue in suggestions['quality_assessment']['main_issues']:
                report_lines.append(f"  • {issue}")
            
            if suggestions['preprocessing_recommendations']:
                report_lines.extend([
                    "",
                    "🔧 PREPROCESSING CONSIGLIATO:",
                ])
                for key, rec in suggestions['preprocessing_recommendations'].items():
                    if rec['recommended']:
                        report_lines.append(f"  • {key.upper()}: {rec['reason']}")
            
            if suggestions['parameter_adjustments']:
                report_lines.extend([
                    "",
                    "⚙️  PARAMETRI TOONCRAFTER SUGGERITI:",
                ])
                for key, adj in suggestions['parameter_adjustments'].items():
                    report_lines.append(f"  • {key}: {adj['recommended_value']} - {adj['reason']}")
            
            report_lines.extend([
                "",
                "=" * 60,
                f"Report generato automaticamente dal MangaQualityAnalyzer",
                "=" * 60
            ])
            
            report_text = "\n".join(report_lines)
            
            # Salva report se richiesto
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
            
            return report_text
            
        except Exception as e:
            error_report = f"Errore nell'analisi qualità: {str(e)}"
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(error_report)
            return error_report