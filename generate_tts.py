import os
import sys
import importlib
import mysql.connector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

spec = importlib.util.spec_from_file_location(
    "tts_utils", os.path.join(BASE_DIR, "utils", "tts_utils.py")
)
tts_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tts_mod)
generate_scenario_wav = tts_mod.generate_scenario_wav

DB_HOST = os.getenv("DB_HOST", "192.168.31.15")
DB_USER = os.getenv("DB_USER", "exohunt")
DB_PASSWORD = os.getenv("DB_PASSWORD", "xHd2009a")
DB_NAME = os.getenv("DB_NAME", "exohunt")
DB_PORT = int(os.getenv("DB_PORT", "3306"))


def get_scenarios():
    conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT,
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT ID, nume, text FROM scenarii ORDER BY ID ASC")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def main():
    print("=== Generator TTS Romanesc (Piper) ===")
    print()

    print("Conectare la baza de date...")
    try:
        scenarios = get_scenarios()
    except Exception as e:
        print(f"EROARE la conectarea la baza de date: {e}")
        sys.exit(1)

    print(f"  S-au gasit {len(scenarios)} scenarii.")
    print()

    generated = 0
    skipped = 0
    errors = 0

    for sc in scenarios:
        sc_id = sc["ID"]
        sc_nume = sc["nume"]
        sc_text = (sc.get("text") or "").strip()

        if not sc_text:
            print(f"  [{sc_id}] {sc_nume} -> SKIP (fara text)")
            skipped += 1
            continue

        try:
            print(f"  [{sc_id}] {sc_nume} -> Generare TTS: \"{sc_text[:60]}...\"")
            if generate_scenario_wav(sc_id, sc_text):
                output_file = os.path.join("assets", f"{sc_id}.wav")
                file_size = os.path.getsize(output_file)
                print(f"    Salvat: {output_file} ({file_size/1024:.1f} KB)")
                generated += 1
            else:
                print(f"    EROARE la generare")
                errors += 1
        except Exception as e:
            print(f"    EROARE: {e}")
            errors += 1

    print()
    print("=== Rezumat ===")
    print(f"  Generate: {generated}")
    print(f"  Omise:    {skipped}")
    print(f"  Erori:    {errors}")
    print(f"  Total:    {len(scenarios)}")


if __name__ == "__main__":
    main()
