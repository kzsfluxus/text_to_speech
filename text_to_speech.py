#!/usr/bin/env python3
"""
Text-to-Speech konverter script magyar nyelvhez.
Használat: python text_to_speech.py <input_file> [output_file]
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from TTS.api import TTS

# Logging beállítása
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_argparse():
    """Parancssori argumentumok beállítása."""
    parser = argparse.ArgumentParser(
        description='Szöveget hangfájllá alakít magyar TTS modell segítségével.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Példák:
  python text_to_speech.py szoveg.txt
  python text_to_speech.py szoveg.txt kimenet.wav
  python text_to_speech.py szoveg.txt --chunk-size 500
        """
    )
    
    parser.add_argument('input_file', help='Bemeneti szövegfájl (.txt)')
    parser.add_argument('output_file', nargs='?', default='output.wav', 
                       help='Kimeneti hangfájl (alapértelmezett: output.wav)')
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Szöveg darabolási méret karakterekben (alapértelmezett: 1000)')
    parser.add_argument('--model', default="tts_models/hu/css10/vits",
                       help='TTS modell neve (alapértelmezett: tts_models/hu/css10/vits)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Részletes kimenet')
    
    return parser

def validate_input_file(file_path):
    """Bemeneti fájl validálása."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"A fájl nem található: {file_path}")
    
    if not path.is_file():
        raise ValueError(f"A megadott útvonal nem fájl: {file_path}")
    
    if path.suffix.lower() not in ['.txt', '.md']:
        logger.warning(f"Figyelem: a fájl kiterjesztése nem .txt vagy .md: {file_path}")
    
    return path

def read_text_file(file_path):
    """Szövegfájl biztonságos beolvasása különböző encoding-okkal."""
    encodings = ['utf-8', 'utf-8-sig', 'iso-8859-2', 'cp1250']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                text = file.read().strip()
                if text:
                    logger.info(f"Fájl sikeresen beolvasva ({encoding} encoding)")
                    return text
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Hiba a fájl olvasásakor ({encoding}): {e}")
            continue
    
    raise ValueError("Nem sikerült beolvasni a fájlt egyetlen encoding-gal sem")

def clean_text(text):
    """Szöveg tisztítása TTS-hez."""
    # Többszörös szóközök és sortörések csökkentése
    import re
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '. ', text)
    
    # Speciális karakterek kezelése
    replacements = {
        '&': ' és ',
        '@': ' kukac ',
        '#': ' hashtag ',
        '%': ' százalék ',
        '+': ' plusz ',
        '=': ' egyenlő ',
        '<': ' kisebb mint ',
        '>': ' nagyobb mint ',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text.strip()

def split_text_into_chunks(text, max_chunk_size=1000):
    """Szöveg darabolása mondatok mentén."""
    import re
    
    # Mondatok keresése
    sentences = re.split(r'[.!?]+\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Ha az aktuális mondat hozzáadása túllépné a limitet
        if len(current_chunk) + len(sentence) + 2 > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Ha egyetlen mondat is túl hosszú, darabolja fel
                while len(sentence) > max_chunk_size:
                    chunks.append(sentence[:max_chunk_size])
                    sentence = sentence[max_chunk_size:]
                current_chunk = sentence
        else:
            current_chunk += ". " + sentence if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def initialize_tts(model_name, verbose=False):
    """TTS modell inicializálása hibakezeléssel."""
    try:
        logger.info(f"TTS modell betöltése: {model_name}")
        
        # GPU elérhetőség ellenőrzése
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Eszköz: {device}")
        except ImportError:
            device = "cpu"
            logger.info("PyTorch nem elérhető, CPU használata")
        
        tts = TTS(model_name=model_name, progress_bar=verbose)
        logger.info("TTS modell sikeresen betöltve")
        return tts
        
    except Exception as e:
        logger.error(f"Hiba a TTS modell betöltésekor: {e}")
        logger.info("Elérhető modellek listázása:")
        try:
            available_models = TTS.list_models()
            hungarian_models = [m for m in available_models if 'hu' in m.lower()]
            if hungarian_models:
                logger.info("Magyar modellek:")
                for model in hungarian_models:
                    logger.info(f"  - {model}")
            else:
                logger.info("Nem találhatók magyar modellek")
        except:
            pass
        raise

def convert_text_to_speech(tts, text_chunks, output_path, verbose=False):
    """Szöveg konvertálása hanggá több részletben."""
    import tempfile
    import wave
    
    logger.info(f"Szöveg konvertálása {len(text_chunks)} részletben...")
    
    temp_files = []
    
    try:
        # Minden chunk-ot külön fájlba konvertálunk
        for i, chunk in enumerate(text_chunks, 1):
            if verbose:
                logger.info(f"Feldolgozás: {i}/{len(text_chunks)} részlet")
                logger.debug(f"Szöveg előnézet: {chunk[:100]}...")
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_files.append(temp_file.name)
            temp_file.close()
            
            tts.tts_to_file(text=chunk, file_path=temp_file.name)
        
        # Ha csak egy chunk van, egyszerűen átnevezzük
        if len(temp_files) == 1:
            import shutil
            shutil.move(temp_files[0], output_path)
        else:
            # Több fájl összefűzése
            concatenate_wav_files(temp_files, output_path)
            
        logger.info(f"Hangfájl sikeresen létrehozva: {output_path}")
        
    finally:
        # Ideiglenes fájlok törlése
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass

def concatenate_wav_files(input_files, output_path):
    """WAV fájlok összefűzése."""
    import wave
    
    logger.info("Hangfájlok összefűzése...")
    
    with wave.open(output_path, 'wb') as output_wav:
        for i, input_file in enumerate(input_files):
            with wave.open(input_file, 'rb') as input_wav:
                if i == 0:
                    # Első fájlból vesszük a paramétereket
                    output_wav.setparams(input_wav.getparams())
                
                # Adatok másolása
                output_wav.writeframes(input_wav.readframes(input_wav.getnframes()))

def get_file_info(file_path):
    """Fájl információk lekérése."""
    try:
        import wave
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            duration = frames / sample_rate
            
            logger.info(f"Hangfájl info:")
            logger.info(f"  - Időtartam: {duration:.2f} másodperc")
            logger.info(f"  - Mintavételi frekvencia: {sample_rate} Hz")
            logger.info(f"  - Fájlméret: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB")
    except:
        logger.info(f"Fájlméret: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB")

def main():
    """Főprogram."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Logging szint beállítása
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Bemeneti fájl validálása
        input_path = validate_input_file(args.input_file)
        
        # Szöveg beolvasása
        logger.info(f"Szöveg beolvasása: {input_path}")
        text = read_text_file(input_path)
        
        if not text:
            raise ValueError("A fájl üres vagy nem tartalmaz olvasható szöveget")
        
        logger.info(f"Beolvasott szöveg hossza: {len(text)} karakter")
        
        # Szöveg tisztítása
        text = clean_text(text)
        
        # Szöveg darabolása
        text_chunks = split_text_into_chunks(text, args.chunk_size)
        logger.info(f"Szöveg {len(text_chunks)} részletre darabolva")
        
        # TTS inicializálása
        tts = initialize_tts(args.model, args.verbose)
        
        # Kimenet konvertálása
        output_path = Path(args.output_file)
        
        # Biztonsági másolat készítése ha már létezik
        if output_path.exists():
            backup_path = output_path.with_suffix(f'.backup{output_path.suffix}')
            logger.info(f"Biztonsági másolat: {backup_path}")
            import shutil
            shutil.copy2(output_path, backup_path)
        
        # Konvertálás
        convert_text_to_speech(tts, text_chunks, args.output_file, args.verbose)
        
        # Eredmény információi
        get_file_info(args.output_file)
        
        logger.info("✅ Sikeres konvertálás!")
        
    except KeyboardInterrupt:
        logger.info("Megszakítva a felhasználó által")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Hiba történt: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
