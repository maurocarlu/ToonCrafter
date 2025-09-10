import streamlit as st
from pathlib import Path
import os
import subprocess
import tempfile
import shutil
import sys
from PIL import Image
from moviepy.editor import VideoFileClip, concatenate_videoclips

# --- Configurazione Iniziale e Import Custom ---
st.set_page_config(layout="wide", page_title="Manga to Anime Pipeline")

# NEW: calcola la root del progetto (ToonCrafter) a partire da questo file
TOON_ROOT = Path(__file__).resolve().parents[1]

# Aggiungi gli script custom al path di sistema
sys.path.insert(0, str((TOON_ROOT / "scriptPanelManga").resolve()))
try:
    from panelPreProcessing import PanelPreProcessor
except ImportError:
    st.error("ERRORE CRITICO: `panelPreProcessing.py` non trovato. Assicurati che si trovi in `ToonCrafter/scriptPanelManga/`.")
    st.stop()

# Prova a importare il runner (se esiste) per evitare il bug timesteps senza toccare inference.py
try:
    from colab_tooncrafter_runner import ColabMangaToonCrafterRunner
    HAVE_RUNNER = True
except Exception:
    HAVE_RUNNER = False

# --- Funzioni di Setup (con cache per efficienza) ---
@st.cache_resource
def setup_environment():
    """Scarica i modelli necessari solo una volta per sessione."""
    st.info("Inizializzazione dell'ambiente (solo al primo avvio)...")
    
    # 1. Download del modello ToonCrafter
    tooncrafter_ckpt_path = Path("./checkpoints/tooncrafter_512_interp_v1/model.ckpt")
    if not tooncrafter_ckpt_path.exists():
        with st.spinner("Scaricando il checkpoint di ToonCrafter (~5GB)..."):
            os.makedirs(tooncrafter_ckpt_path.parent, exist_ok=True)
            # Usiamo aria2c se disponibile, altrimenti wget
            if shutil.which("aria2c"):
                os.system("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/ToonCrafter/resolve/main/model.ckpt -d ./checkpoints/tooncrafter_512_interp_v1 -o model.ckpt")
            else:
                os.system("wget -q https://huggingface.co/camenduru/ToonCrafter/resolve/main/model.ckpt -P ./checkpoints/tooncrafter_512_interp_v1/")

    # 2. Setup di RIFE
    rife_dir = Path("./ECCV2022-RIFE")
    rife_weights_path = rife_dir / "train_log" / "flownet.pkl"
    if not rife_weights_path.exists():
        with st.spinner("Setup di RIFE in corso..."):
            os.system(f"git clone -q https://github.com/hzwer/ECCV2022-RIFE.git {rife_dir}")
            gdrive_id = "1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_"
            zip_path = rife_dir / "train_log.zip"
            os.system(f"gdown --id {gdrive_id} -O {str(zip_path)}")
            os.system(f"unzip -q -o {str(zip_path)} -d {str(rife_dir)}")
            os.remove(zip_path)

    st.success("Ambiente pronto!")
    return True

# --- Interfaccia Utente ---
st.title("🎌 Manga to Anime: Generative AI Pipeline")
st.markdown("Questa applicazione utilizza **ToonCrafter** e **RIFE** per trasformare pannelli manga statici in brevi animazioni fluide.")

# Esegui il setup
if 'setup_done' not in st.session_state:
    if setup_environment():
        st.session_state.setup_done = True

with st.sidebar:
    st.header("⚙️ Parametri di Generazione")

    st.subheader("ToonCrafter")
    frame_stride = st.slider("Frame Stride (Movimento)", 3, 12, 8, help="Valori bassi = più movimento. Valori alti = movimento più lento.")
    ddim_steps = st.slider("DDIM Steps (Qualità)", 25, 100, 50)
    
    st.subheader("LoRA")
    use_lora = st.checkbox("Usa LoRA", value=True)
    lora_path_input = st.text_input("Path del LoRA (opzionale)", "./ckpts/best_lora.safetensors")
    lora_scale = st.slider("LoRA Scale", 0.0, 1.0, 0.3, 0.05)

    st.subheader("Preprocessing")
    use_preprocessing = st.checkbox("Applica Preprocessing Avanzato", value=True)

    st.subheader("Post-processing (Fluidità)")
    use_rife = st.checkbox("Aumenta Fluidità con RIFE", value=True)
    final_fps = st.select_slider("FPS Finali (con RIFE)", options=[24, 30, 48], value=24)

st.header("1. Carica i Pannelli Manga")
st.markdown("Carica i frame in ordine (es. `frame1.png`, `frame3.png`, `frame5.png`...) per abilitare il chaining automatico.")
uploaded_files = st.file_uploader("Carica due o più frame", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

st.header("2. Inserisci il Prompt")
prompt = st.text_area("Prompt (descrivi la scena o lo stile)", "anime style, cinematic, clean line art, high quality")

# --- Logica di Esecuzione ---
if st.button("🚀 Genera Animazione", use_container_width=True):
    if uploaded_files and len(uploaded_files) >= 2:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sorted_files = sorted(uploaded_files, key=lambda f: f.name)
            
            frame_pairs = [(sorted_files[i], sorted_files[i+1]) for i in range(len(sorted_files) - 1)]
            st.info(f"Rilevate {len(frame_pairs)} coppie da interpolare per {len(frame_pairs)} segmenti video.")
            
            generated_segments = []
            preprocessor = PanelPreProcessor(debug_mode=False)

            for i, (file1, file2) in enumerate(frame_pairs):
                st.markdown(f"--- \n ### Processando Segmento {i+1}/{len(frame_pairs)}: `{file1.name}` -> `{file2.name}`")
                
                segment_input_dir = temp_path / f"s{i}_in"
                segment_output_dir = temp_path / f"s{i}_out"
                segment_input_dir.mkdir()
                
                img1_path = segment_input_dir / "input_frame1.png"
                img2_path = segment_input_dir / "input_frame3.png"
                
                with st.spinner("Applicando preprocessing..."):
                    img1 = Image.open(file1)
                    img2 = Image.open(file2)
                    if use_preprocessing:
                        img1 = preprocessor.apply_preprocessing_pipeline(img1)
                        img2 = preprocessor.apply_preprocessing_pipeline(img2)
                    img1.save(img1_path)
                    img2.save(img2_path)

                (segment_input_dir / "prompt.txt").write_text(prompt)
                # compat: alcune pipeline leggono "prompts.txt"
                (segment_input_dir / "prompts.txt").write_text(prompt)

                with st.spinner(f"ToonCrafter sta generando il segmento {i+1}... (può richiedere 2-3 minuti)"):
                    generated_ok = False
                    segment_output_dir.mkdir(exist_ok=True, parents=True)

                    if HAVE_RUNNER:
                        # Usa il runner Python (gestisce i timesteps correttamente)
                        try:
                            # NEW: usa la root effettiva di ToonCrafter
                            runner = ColabMangaToonCrafterRunner(str(TOON_ROOT))
                            params = {
                                "frame_stride": int(frame_stride),
                                "ddim_steps": int(ddim_steps),
                                "unconditional_guidance_scale": 7.5,
                                "guidance_rescale": 0.0,
                                "video_length": 16,
                            }
                            lora_kwargs = {}
                            if use_lora and Path(lora_path_input).exists():
                                lora_kwargs = {"lora_path": lora_path_input, "lora_scale": float(lora_scale)}

                            base_name = "input"  # perché i file sono input_frame1.png / input_frame3.png
                            ok = runner.run_custom_parameters_conversion(
                                base_name=base_name,
                                prompt=prompt,
                                custom_params=params,
                                output_dir=str(segment_output_dir),
                                input_dir=str(segment_input_dir),
                                # opzionale: evita doppio preprocessing (già fatto sopra)
                                enable_preprocessing=False,
                                **lora_kwargs,
                            )
                            generated_ok = bool(ok)
                        except Exception as e:
                            st.warning(f"Runner non disponibile o errore: uso fallback inference.py. Dettagli: {e}")
                            generated_ok = False  # usiamo il fallback sotto

                    if not generated_ok:
                        # Fallback CLI: chiama direttamente inference.py
                        cmd_tc = [
                            sys.executable, str(TOON_ROOT / "scripts" / "evaluation" / "inference.py"),
                            "--config", str(TOON_ROOT / "configs" / "inference_512_v1.0.yaml"),
                            "--ckpt_path", str(TOON_ROOT / "checkpoints" / "tooncrafter_512_interp_v1" / "model.ckpt"),
                            "--prompt_dir", str(segment_input_dir),
                            "--savedir", str(segment_output_dir),
                            "--ddim_steps", str(ddim_steps),
                            "--frame_stride", str(frame_stride),
                            "--unconditional_guidance_scale", "7.5",
                            "--video_length", "16",
                            "--height", "320", "--width", "512",
                            "--ddim_eta", "0.0",
                            "--perframe_ae",
                            "--interp", "--text_input",
                        ]
                        if use_lora and Path(lora_path_input).exists():
                            cmd_tc.extend(["--lora_path", lora_path_input, "--lora_scale", str(lora_scale)])

                        # NEW: cwd sulla root del progetto per path relativi consistenti
                        result_tc = subprocess.run(cmd_tc, capture_output=True, text=True, cwd=str(TOON_ROOT))
                        if result_tc.returncode != 0:
                            st.error(f"Errore generazione segmento {i+1}.")
                            st.code(result_tc.stderr or result_tc.stdout)
                            break

                    # NEW: cerca MP4 anche nelle sottocartelle (output runner)
                    found = list((Path(segment_output_dir) / "samples_separate").glob("*.mp4"))
                    if not found:
                        found = list(Path(segment_output_dir).rglob("samples_separate/*.mp4"))
                    if not found and HAVE_RUNNER:
                        found = list(Path(segment_output_dir).rglob("*.mp4"))

                    if not found:
                        st.error(f"Errore generazione segmento {i+1}: nessun MP4 trovato.")
                        if HAVE_RUNNER:
                            st.caption("Suggerimento: verifica i log del runner (può aver salvato in una sottocartella).")
                        else:
                            st.caption("Suggerimento: abilita il runner Python in scriptPanelManga per maggiore compatibilità.")
                        break

                    generated_segments.append(str(found[0]))
                    st.success(f"✅ Segmento {i+1} generato!")
            
            if len(generated_segments) == len(frame_pairs):
                final_video_path = Path(generated_segments[0])
                if len(generated_segments) > 1:
                    with st.spinner("Concatenando i segmenti..."):
                        clips = [VideoFileClip(p) for p in generated_segments]
                        final_clip = concatenate_videoclips([clips[0]] + [c.subclip(1.0/c.fps) for c in clips[1:]])
                        final_video_path = temp_path / "chained.mp4"
                        final_clip.write_videofile(str(final_video_path), fps=clips[0].fps, codec="libx264", logger=None)
                        st.success("✅ Video concatenato!")

                if use_rife:
                    with st.spinner("Aumentando la fluidità con RIFE..."):
                        rife_out = temp_path / "final_high_fps.mp4"
                        cmd_rife = [sys.executable, str(TOON_ROOT / "ECCV2022-RIFE" / "inference_video.py"),
                                    "--exp", "1", "--video", str(final_video_path), "--output", str(rife_out),
                                    "--fp16", "--fps", str(final_fps)]
                        result_rife = subprocess.run(cmd_rife, capture_output=True, text=True, cwd=str(TOON_ROOT / "ECCV2022-RIFE"))
                        if result_rife.returncode == 0 and rife_out.exists():
                            final_video_path = rife_out
                            st.success("✅ Fluidità aumentata!")
                        else:
                            st.warning("RIFE ha fallito, mostro il video precedente."); st.code(result_rife.stderr)
                
                st.header("🎉 Risultato Finale")
                video_bytes = final_video_path.read_bytes()
                st.video(video_bytes)
                st.download_button("⬇️ Scarica il Video", video_bytes, "animazione_finale.mp4", "video/mp4", use_container_width=True)
    else:
        st.warning("Per favore, carica almeno due frame per iniziare.")