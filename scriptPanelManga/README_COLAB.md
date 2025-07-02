# 🎌 Manga to Anime Pipeline per Google Colab

**Sistema completo per convertire panel manga in scene anime utilizzando ToonCrafter + Preprocessing Manga-Specifico**

## 🆕 NUOVO: Preprocessing Manga Intelligente

Il sistema ora include un **modulo di preprocessing manga-specifico** che migliora significativamente la qualità dell'output:

### ✨ Funzionalità Avanzate:
- **🔍 Analisi Qualità Automatica**: Valuta nitidezza, contrasto, rumore e line art quality
- **🎨 Preprocessing Adattivo**: Miglioramenti personalizzati basati sull'analisi dell'input  
- **🎛️ Ottimizzazione Parametri**: Auto-ottimizzazione parametri ToonCrafter per input problematici
- **🖼️ Preservazione Stile**: Mantiene caratteristiche manga originali (line art, retini, balloon)
- **📊 Metriche Dettagliate**: Report completo con suggerimenti di miglioramento

### 🚀 Utilizzo con Preprocessing:

```python
from scriptPanelManga.colab_tooncrafter_runner import run_with_manga_preprocessing

# Esecuzione con preprocessing automatico (CONSIGLIATO)
success = run_with_manga_preprocessing(
    tooncrafter_path="ToonCrafter",
    prompt_dir="test_prompts/manga_test", 
    output_dir="output_videos/result",
    config_type="dramatic_change",
    preprocessing_preset="default"  # o "high_quality", "low_quality_scan", etc.
)
```

**[📖 Documentazione Completa Preprocessing →](README_MANGA_PREPROCESSING.md)**

---

Questa è la versione ottimizzata per **Google Colab** della pipeline per convertire panel manga in scene anime utilizzando ToonCrafter.

## 🚀 Come Usare (Guida Rapida)

### 1. Apri Google Colab
1. Vai su [Google Colab](https://colab.research.google.com/)
2. Carica il notebook `manga_to_anime_colab.ipynb`
3. Assicurati di avere la GPU attivata: `Runtime` > `Change runtime type` > `Hardware accelerator` > `GPU`

### 2. Prepara i tuoi file
Hai 2 opzioni:

**Opzione A: Google Drive**
- Carica i file del progetto su Google Drive nella cartella `MyDrive/ProgettoCV/`
- Carica alcune immagini manga in `MyDrive/manga_panels/`

**Opzione B: Upload diretto**
- Usa le celle di upload nel notebook per caricare i file direttamente

### 3. Esegui il notebook
- Esegui tutte le celle in ordine
- Il processo completo richiede 15-30 minuti
- I video generati verranno scaricati automaticamente

## 📋 File Necessari

### Script principali:
- `advanced_tooncrafter_runner.py` o `colab_tooncrafter_runner.py`
- `manga_to_anime.py`
- `Manga109Dataset.py`

### ✨ NUOVI Moduli Preprocessing:
- `manga_preprocessor.py` - Preprocessing manga-specifico avanzato
- `manga_quality_analyzer.py` - Analisi qualità e ottimizzazione automatica
- `test_manga_preprocessing.py` - Test suite per validazione funzionalità

### Immagini manga:
- Almeno 2 immagini PNG/JPG
- Risoluzione minima: 512x320 pixel
- Panel che mostrano una progressione logica

### 🎨 Preprocessing Automatico:
Il sistema ora analizza automaticamente la qualità delle immagini input e applica ottimizzazioni appropriate:
- **Analisi qualità real-time** con metriche dettagliate
- **Preprocessing adattivo** basato sul tipo di manga
- **Ottimizzazione automatica** parametri ToonCrafter

## 🎛️ Configurazioni Ottimizzate per Colab

Il sistema include configurazioni specificamente ottimizzate per Google Colab:

### 🌊 `smooth_transition`
- **Uso**: Panel manga molto simili
- **Velocità**: ⭐⭐⭐⭐
- **Qualità**: ⭐⭐⭐
- **VRAM**: 6-8 GB

### 🎭 `dramatic_change` (RACCOMANDATO)
- **Uso**: Cambi drastici tra panel manga
- **Velocità**: ⭐⭐⭐
- **Qualità**: ⭐⭐⭐⭐⭐
- **VRAM**: 8-10 GB

### ⚔️ `action_sequence`
- **Uso**: Scene d'azione dinamiche
- **Velocità**: ⭐⭐
- **Qualità**: ⭐⭐⭐⭐⭐
- **VRAM**: 10-12 GB

### ⚡ `colab_fast`
- **Uso**: Test veloci e prototipi
- **Velocità**: ⭐⭐⭐⭐⭐
- **Qualità**: ⭐⭐⭐
- **VRAM**: 4-6 GB

## 🔧 Ottimizzazioni per Google Colab

### Gestione Memoria GPU
- **Automatic mixed precision** per efficienza memoria
- **XFormers** per attention ottimizzata
- **Progressive memory management** durante inferenza
- **Garbage collection** automatico tra le esecuzioni

### Ottimizzazioni Specifiche GPU

**Tesla T4 (Colab gratuito)**
- Configurazioni ottimizzate per 15GB VRAM
- Parametri bilanciati velocità/qualità
- Automatic cleanup tra processi

**Tesla V100 (Colab Pro)**
- Configurazioni ad alta qualità
- Sfruttamento completo della potenza GPU
- Batch processing ottimizzato

## 📊 Tempi di Esecuzione Stimati

| Configurazione | T4 (Gratuito) | V100 (Pro) | Qualità Output | Con Preprocessing |
|---------------|---------------|-------------|----------------|-------------------|
| `colab_fast` | 3-5 min | 2-3 min | Buona | +1-2 min |
| `smooth_transition` | 5-8 min | 3-5 min | Molto buona | +1-2 min |
| `dramatic_change` | 8-12 min | 5-8 min | Eccellente | +2-3 min |
| `action_sequence` | 10-15 min | 6-10 min | Eccellente | +2-3 min |

*Il preprocessing aggiunge 1-3 minuti ma migliora significativamente la qualità output*

## 🎯 Suggerimenti per Migliori Risultati

### 🆕 Preprocessing Automatico (RACCOMANDATO)
1. **Abilita preprocessing manga**: Usa `run_with_manga_preprocessing()` per risultati ottimali
2. **Preset intelligenti**: Scegli tra `default`, `high_quality`, `low_quality_scan`, `action_sequence`
3. **Analisi qualità automatica**: Il sistema analizza e ottimizza automaticamente l'input
4. **Ottimizzazione parametri**: I parametri ToonCrafter vengono auto-regolati per input problematici

### Preparazione Immagini
1. **Risoluzione**: Ridimensiona a 512x320 o multipli
2. **Formato**: Preferisci PNG per la qualità
3. **Contenuto**: Evita panel troppo dettagliati o confusi
4. **Sequenza**: Scegli panel che mostrano una progressione logica
5. **✨ Qualità**: Il preprocessing gestisce automaticamente scan di bassa qualità e artefatti

### Prompt Efficaci
```
# Per transizioni drammatiche
"dramatic scene transition, manga to anime style, dynamic character movement"

# Per scene d'azione
"action sequence, manga style, intense movement and dynamic effects"

# Per dialoghi
"character dialogue scene, anime style, subtle facial animation"

# Per scene fluide
"smooth anime transition, manga style, fluid character movement"
```

### Ottimizzazione Performance
- **Chiudi altre schede** del browser durante l'esecuzione
- **Riavvia il runtime** se hai problemi di memoria
- **Usa configurazioni più veloci** per i test iniziali
- **Salva risultati intermedi** per evitare perdita di progresso

## 🐛 Risoluzione Problemi

### ❌ "GPU non disponibile"
**Soluzione**: `Runtime` > `Change runtime type` > `Hardware accelerator` > `GPU`

### ❌ "CUDA out of memory"
**Soluzioni**:
1. Riavvia il runtime: `Runtime` > `Restart runtime`
2. Usa configurazione `colab_fast`
3. Riduci `video_length` a 12 frame
4. Usa `perframe_ae` flag (già incluso)

### ❌ "Checkpoint non trovato"
**Soluzioni**:
1. Verifica che il download sia completato
2. Controlla la connessione internet
3. Scarica manualmente da [HuggingFace](https://huggingface.co/Doubiiu/ToonCrafter)

### ❌ "File mancanti"
**Soluzioni**:
1. Ricarica i file usando le celle di upload
2. Verifica i path su Google Drive
3. Esegui la cella di verifica prerequisiti

### ⚠️ "Risultati di bassa qualità"
**Soluzioni**:
1. Usa configurazione `dramatic_change` invece di `colab_fast`
2. Aumenta `ddim_steps` a 80-100
3. Migliora la qualità delle immagini input
4. Prova prompt più dettagliati

## 📈 Monitoraggio Risorse

### Durante l'esecuzione, puoi monitorare:
- **GPU Usage**: `!nvidia-smi` in una cella
- **RAM Usage**: Guarda la barra in alto a destra
- **Disk Space**: Controlla lo spazio disponibile

### Limiti Google Colab
- **Tempo max sessione**: ~12 ore (Gratuito), ~24 ore (Pro)
- **VRAM**: 15GB (T4), 16GB (V100)
- **Storage**: ~100GB temporaneo
- **Network**: Limitazioni su download pesanti

## 🎉 Esempio di Utilizzo Completo

```python
# 1. Setup iniziale (esegui una volta)
from google.colab import drive
drive.mount('/content/drive')

# 2. Carica e configura
!git clone https://github.com/ToonCrafter/ToonCrafter.git
# ... (segui il notebook per setup completo)

# 3. Esecuzione principale
from colab_tooncrafter_runner import run_with_config

success = run_with_config(
    tooncrafter_path="ToonCrafter",
    prompt_dir="test_prompts/manga_test", 
    output_dir="output_videos/result",
    config_type="dramatic_change"
)

# 4. Download risultati
from google.colab import files
files.download("output_videos/result/samples_separate/video.mp4")
```

## 📞 Supporto

Per problemi specifici:
1. **Controlla la sezione troubleshooting** sopra
2. **Esegui le celle di verifica** nel notebook
3. **Verifica i log** per errori specifici
4. **Riavvia il runtime** e riprova

## 🎓 Prossimi Passi

Dopo aver padroneggiato la pipeline base:
1. **Integra il dataset Manga109 completo**
2. **Sperimenta con configurazioni personalizzate**
3. **Combina più sequenze** per episodi più lunghi
4. **Esplora prompt engineering avanzato**

---

**Buona animazione! 🎌➡️🎬**
