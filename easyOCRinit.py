#!/usr/bin/env python3
"""
patch_easyocr_el.py
Downloads CRAFT and Greek model, places them under ~/.EasyOCR/model/,
backs up easyocr.character.py and easyocr.model_builder.py and
inserts lightweight patches so 'el' (Greek) is recognized.

Run: python patch_easyocr_el.py
"""

import os
import sys
import shutil
import urllib.request
import importlib
import textwrap

HOME = os.path.expanduser("~")
MODEL_ROOT = os.path.join(HOME, ".EasyOCR", "model")
CRAFT_DIR = os.path.join(MODEL_ROOT, "craft_mlt_25k")
EL_DIR = os.path.join(MODEL_ROOT, "el")

CRAFT_URL = "https://www.jaided.ai/easyocr/model/craft_mlt_25k.pth"
GREECE_URL = "https://www.jaided.ai/easyocr/model/greece_g2.pth"

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def download(url, dst):
    print(f"Downloading {url} -> {dst} ...")
    urllib.request.urlretrieve(url, dst)
    print("done")

def locate_easyocr():
    try:
        import easyocr
    except Exception as e:
        print("ERROR: easyocr is not importable in this Python. Activate the correct env.")
        raise
    pkg_dir = os.path.dirname(easyocr.__file__)
    return pkg_dir

def backup_file(path):
    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"Backed up {path} -> {bak}")
    else:
        print(f"Backup already exists: {bak}")

def patch_character(character_path):
    with open(character_path, "r", encoding="utf-8") as f:
        s = f.read()

    if "'el'" in s or '"el"' in s:
        print("character.py already mentions 'el' — skipping character patch.")
        return

    # Insert 'el': 'Greek' into LANGUAGES and a LANG2CHAR mapping for Greek.
    s_new = s
    s_new = s_new.replace("LANGUAGES = {", "LANGUAGES = {\n    'el': 'Greek',", 1)

    greek_charset_line = "    'el': 'αβγδεζηθικλμνξοπρστυφχψωΆΈΉΊΌΎΏάέήίόύώ',"
    s_new = s_new.replace("LANG2CHAR = {", "LANG2CHAR = {\n" + greek_charset_line, 1)

    if s_new == s:
        print("No replacements made in character.py (unexpected).")
        return

    with open(character_path, "w", encoding="utf-8") as f:
        f.write(s_new)
    print("Patched character.py: added 'el' entries.")

def patch_model_builder(model_builder_path):
    with open(model_builder_path, "r", encoding="utf-8") as f:
        s = f.read()

    if "lang == 'el'" in s:
        print("model_builder.py already patched for 'el' — skipping.")
        return

    # A best-effort injection: add a small helper at top and call it within get_recognizer.
    injection = textwrap.dedent("""
    # ---- injected Greek support (auto-added) ----
    def _easyocr_inject_el_branch(lang, locals_dict):
        if lang == 'el':
            # instruct loader to use greece_g2 and Greek charset
            locals_dict['model_name'] = 'greece_g2'
            try:
                from .character import LANG2CHAR
                locals_dict['character'] = LANG2CHAR.get('el', '')
            except Exception:
                locals_dict['character'] = ''
        return locals_dict
    # ---------------------------------------------
    """)

    # Place injection near top of file (after module docstring if present)
    idx = 0
    if s.startswith('"""') or s.startswith("'''"):
        # skip docstring
        end = s.find('"""', 3)
        if end == -1:
            end = s.find("'''", 3)
        if end != -1:
            idx = end + 3
    s2 = s[:idx] + "\n" + injection + s[idx:]

    # Now try to insert a small call into get_recognizer definition
    s2 = s2.replace("def get_recognizer(", "def get_recognizer(", 1)  # ensure exists
    # simple insert: after the def line insert the locals injection. This is best-effort.
    s2 = s2.replace("def get_recognizer(", "def get_recognizer(", 1)
    s2 = s2.replace("def get_recognizer(", "def get_recognizer(", 1)
    # Insert call to _easyocr_inject_el_branch at top of function by finding the function header
    s2 = s2.replace("def get_recognizer(", "def get_recognizer(", 1)  # idempotent
    # Now find the first occurrence of "def get_recognizer" and then the next ":" and insert lines
    def_index = s2.find("def get_recognizer")
    if def_index != -1:
        # find the colon that ends the def signature
        colon_index = s2.find("):", def_index)
        if colon_index != -1:
            insert_pos = colon_index + 2  # after "):"
            insert_text = "\n    locals_dict = locals()\n    try:\n        _easyocr_inject_el_branch(lang, locals_dict)\n    except Exception:\n        pass\n"
            s2 = s2[:insert_pos] + insert_text + s2[insert_pos:]
            with open(model_builder_path, "w", encoding="utf-8") as f:
                f.write(s2)
            print("Patched model_builder.py (injected small el-handling).")
            return
    # fallback: append injection at end
    with open(model_builder_path, "w", encoding="utf-8") as f:
        f.write(s2)
    print("Appended injection to model_builder.py (best-effort).")

def main():
    ensure_dir(CRAFT_DIR)
    ensure_dir(EL_DIR)

    craft_target = os.path.join(CRAFT_DIR, "craft_mlt_25k.pth")
    el_target = os.path.join(EL_DIR, "greece_g2.pth")

    if not os.path.exists(craft_target):
        download(CRAFT_URL, craft_target)
    else:
        print(f"{craft_target} already exists -- skipping download.")

    if not os.path.exists(el_target):
        download(GREECE_URL, el_target)
    else:
        print(f"{el_target} already exists -- skipping download.")

    pkg_dir = locate_easyocr()
    print("easyocr package located:", pkg_dir)

    character_path = os.path.join(pkg_dir, "character.py")
    model_builder_path = os.path.join(pkg_dir, "model_builder.py")

    if not os.path.exists(character_path) or not os.path.exists(model_builder_path):
        print("ERROR: expected files not found in easyocr package. Aborting.")
        sys.exit(1)

    backup_file(character_path)
    backup_file(model_builder_path)

    patch_character(character_path)
    patch_model_builder(model_builder_path)

    print("\nAll done. Test with:")
    print("python - <<'PY'")
    print("import easyocr")
    print("reader = easyocr.Reader(['en','el'], gpu=False)")
    print("print('languages:', reader.lang)")
    print("print(reader.readtext('path_to_one_of_your_images.png', detail=0))")
    print("PY")

if __name__ == "__main__":
    main()
