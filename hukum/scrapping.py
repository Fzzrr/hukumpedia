import re
import json
from pdfminer.high_level import extract_text

def extract_bab_pasal_from_pdf(pdf_path):
    # Ekstrak seluruh teks dari PDF
    full_text = extract_text(pdf_path)

    # Pola untuk bab dan pasal
    pattern_bab = re.compile(r'(BAB\s+[IVXLCDM]+)', re.IGNORECASE)
    pattern_pasal = re.compile(r'(Pasal\s+\d+[A-Za-z]*)', re.IGNORECASE)

    # Pecah teks berdasarkan BAB
    parts = pattern_bab.split(full_text)
    # Hasil split: [teks sebelum bab, BAB1, teks bab1, BAB2, teks bab2, ...]

    data = []
    for i in range(1, len(parts), 2):
        bab_title = parts[i].strip().upper()
        bab_text = parts[i+1].strip()

        # Ambil judul bab: baris pertama non-kosong SETELAH baris pertama
        lines = bab_text.splitlines()
        judul_bab = ""
        found_judul = False
        for idx, line in enumerate(lines):
            # Lewati baris kosong di awal
            if not line.strip():
                continue
            # Baris pertama non-kosong dianggap judul bab
            judul_bab = line.strip()
            found_judul = True
            pasal_start_idx = idx + 1
            break
        # Jika tidak ditemukan judul, kosongkan
        if not found_judul:
            judul_bab = ""
            pasal_start_idx = 0
        # Jika judul bab sama dengan nama bab, kosongkan
        if judul_bab.upper() == bab_title:
            judul_bab = ""

        # Gabungkan kembali teks pasal mulai dari pasal_start_idx
        pasal_text = "\n".join(lines[pasal_start_idx:]) if found_judul else bab_text

        # Pecah pasal
        pasal_parts = pattern_pasal.split(pasal_text)
        # pola pasal_parts: [teks sebelum pasal, Pasal1, teks pasal1, Pasal2, teks pasal2, ...]

        bab_entry = {
            "bab": bab_title,
            "judul": judul_bab,
            "pasal_list": []
        }

        for j in range(1, len(pasal_parts), 2):
            pasal_nomor = pasal_parts[j].strip()
            pasal_isi = pasal_parts[j+1].strip() if (j+1) < len(pasal_parts) else ""
            pasal_isi = re.sub(r'\s+', ' ', pasal_isi)  # bersihkan spasi berlebih

            bab_entry["pasal_list"].append({
                "pasal": pasal_nomor,
                "teks": pasal_isi
            })

        data.append(bab_entry)

    return data

def save_to_json(data, out_file):
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    pdf_path = "Undang-Undang Dasar Republik Indonesia Tahun 1945.pdf"
    out_file = "out_dataset_bab_pasal.json"

    data = extract_bab_pasal_from_pdf(pdf_path)
    save_to_json(data, out_file)

    print(f"Berhasil mengekstrak {len(data)} bab beserta pasalnya ke file {out_file}")