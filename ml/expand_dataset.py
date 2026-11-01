
"""Расширение датасета: +600 имён с явным гендером.

Формат: (латинское_имя, gender). Транслитератор → кириллица.
Идемпотентен: повторный запуск не создаёт дубликатов.

Запуск:
    python -m ml.expand_dataset
"""
import json
from pathlib import Path

from ml.preprocess import normalize
from ml.transliterate import transliterate

NEW_NAMES = {
    "elven": [
        # male (60)
        ("Aegnor", "male"), ("Aelion", "male"), ("Aerandir", "male"),
        ("Aerion", "male"), ("Aethon", "male"), ("Aglarond", "male"),
        ("Amrod", "male"), ("Anarion", "male"), ("Andúril", "male"),
        ("Angrod", "male"), ("Aranrúth", "male"), ("Argonui", "male"),
        ("Arminas", "male"), ("Artaher", "male"), ("Beleg", "male"),
        ("Bregolas", "male"), ("Calion", "male"), ("Caranthir", "male"),
        ("Celeborn", "male"), ("Celebrimbor", "male"), ("Celegorm", "male"),
        ("Curufin", "male"), ("Daeron", "male"), ("Denethor", "male"),
        ("Dior", "male"), ("Ecthelion", "male"), ("Edrahil", "male"),
        ("Egalmoth", "male"), ("Eldarion", "male"), ("Elendil", "male"),
        ("Elros", "male"), ("Elrond", "male"), ("Eöl", "male"),
        ("Ereinion", "male"), ("Faramir", "male"), ("Fëanor", "male"),
        ("Fingolfin", "male"), ("Fingon", "male"), ("Finrod", "male"),
        ("Galdor", "male"), ("Gildor", "male"), ("Gil-galad", "male"),
        ("Glorfindel", "male"), ("Guilin", "male"), ("Haldir", "male"),
        ("Halmir", "male"), ("Húrin", "male"), ("Isildur", "male"),
        ("Lindir", "male"), ("Maedhros", "male"), ("Maeglin", "male"),
        ("Maglor", "male"), ("Mithrellas", "male"), ("Orodreth", "male"),
        ("Orophin", "male"), ("Rúmil", "male"), ("Thranduil", "male"),
        ("Turgon", "male"), ("Tuor", "male"), ("Turin", "male"),

        # female (60)
        ("Aelith", "female"), ("Aeluin", "female"), ("Aerin", "female"),
        ("Ailinel", "female"), ("Almiel", "female"), ("Anaire", "female"),
        ("Andreth", "female"), ("Aredhel", "female"), ("Arwen", "female"),
        ("Celebrían", "female"), ("Culúrien", "female"), ("Elbereth", "female"),
        ("Elwing", "female"), ("Eärwen", "female"), ("Finduilas", "female"),
        ("Galadriel", "female"), ("Gilraen", "female"), ("Idril", "female"),
        ("Indis", "female"), ("Ioreth", "female"), ("Ivorwen", "female"),
        ("Lalwen", "female"), ("Lúthien", "female"), ("Melian", "female"),
        ("Míriel", "female"), ("Morwen", "female"), ("Nellas", "female"),
        ("Nienor", "female"), ("Nimloth", "female"), ("Nimrodel", "female"),
        ("Rían", "female"), ("Silmarwen", "female"), ("Tauriel", "female"),
        ("Yávien", "female"), ("Aeriel", "female"), ("Aelithiel", "female"),
        ("Aelinor", "female"), ("Aeriel", "female"), ("Aeslin", "female"),
        ("Ailinel", "female"), ("Almarien", "female"), ("Amariel", "female"),
        ("Anariel", "female"), ("Andrielle", "female"), ("Aranel", "female"),
        ("Arienh", "female"), ("Arweniel", "female"), ("Aurelith", "female"),
        ("Calael", "female"), ("Calathiel", "female"), ("Celinn", "female"),
        ("Celebrindal", "female"), ("Elaria", "female"), ("Eledhwen", "female"),
        ("Eleniel", "female"), ("Elith", "female"), ("Elnara", "female"),
        ("Emelith", "female"), ("Eruanna", "female"), ("Faenor", "female"),
        ("Faelwen", "female"), ("Finiel", "female"),
    ],
    "human": [
        # male (60)
        ("Abraham", "male"), ("Adrian", "male"), ("Albert", "male"),
        ("Alexander", "male"), ("Alfred", "male"), ("Alistair", "male"),
        ("Ambrose", "male"), ("Andrew", "male"), ("Angus", "male"),
        ("Anselm", "male"), ("Archibald", "male"), ("Arnold", "male"),
        ("Arthur", "male"), ("Aubrey", "male"), ("Baldwin", "male"),
        ("Barnaby", "male"), ("Bartholomew", "male"), ("Basil", "male"),
        ("Benedict", "male"), ("Bernard", "male"), ("Bertram", "male"),
        ("Bertrand", "male"), ("Brian", "male"), ("Caleb", "male"),
        ("Casimir", "male"), ("Cedric", "male"), ("Charles", "male"),
        ("Christopher", "male"), ("Clarence", "male"), ("Claude", "male"),
        ("Clifford", "male"), ("Conrad", "male"), ("Cornelius", "male"),
        ("Crispin", "male"), ("Cuthbert", "male"), ("Cyril", "male"),
        ("Damian", "male"), ("Daniel", "male"), ("David", "male"),
        ("Desmond", "male"), ("Dominic", "male"), ("Donald", "male"),
        ("Douglas", "male"), ("Duncan", "male"), ("Edgar", "male"),
        ("Edmund", "male"), ("Edward", "male"), ("Edwin", "male"),
        ("Elliot", "male"), ("Eric", "male"), ("Ernest", "male"),
        ("Ethan", "male"), ("Eustace", "male"), ("Fabian", "male"),
        ("Felix", "male"), ("Ferdinand", "male"), ("Francis", "male"),
        ("Frederick", "male"), ("Gabriel", "male"), ("Gareth", "male"),

        # female (60)
        ("Adelaide", "female"), ("Agatha", "female"), ("Agnes", "female"),
        ("Alice", "female"), ("Amelia", "female"), ("Angelica", "female"),
        ("Anna", "female"), ("Arabella", "female"), ("Audrey", "female"),
        ("Aurora", "female"), ("Beatrice", "female"), ("Bella", "female"),
        ("Bernadette", "female"), ("Blanche", "female"), ("Bridget", "female"),
        ("Camilla", "female"), ("Caroline", "female"), ("Catherine", "female"),
        ("Cecilia", "female"), ("Celestine", "female"), ("Charlotte", "female"),
        ("Clara", "female"), ("Clarissa", "female"), ("Clementine", "female"),
        ("Constance", "female"), ("Cora", "female"), ("Cordelia", "female"),
        ("Cynthia", "female"), ("Daphne", "female"), ("Deborah", "female"),
        ("Delia", "female"), ("Diana", "female"), ("Dorothy", "female"),
        ("Edith", "female"), ("Edwina", "female"), ("Elaine", "female"),
        ("Eleanor", "female"), ("Elizabeth", "female"), ("Ella", "female"),
        ("Emily", "female"), ("Emma", "female"), ("Esme", "female"),
        ("Estella", "female"), ("Esther", "female"), ("Evelyn", "female"),
        ("Felicity", "female"), ("Fiona", "female"), ("Flora", "female"),
        ("Florence", "female"), ("Genevieve", "female"), ("Georgia", "female"),
        ("Gertrude", "female"), ("Grace", "female"), ("Gwendolyn", "female"),
        ("Harriet", "female"), ("Hazel", "female"), ("Helen", "female"),
        ("Henrietta", "female"), ("Hildegard", "female"), ("Imogen", "female"),
    ],
    "orcish": [
        # male (60)
        ("Azog", "male"), ("Bolg", "male"), ("Gorbag", "male"),
        ("Gothmog", "male"), ("Grishnákh", "male"), ("Grubash", "male"),
        ("Grukhal", "male"), ("Kragga", "male"), ("Kragnak", "male"),
        ("Kruglak", "male"), ("Lugbag", "male"), ("Lugdush", "male"),
        ("Lugnash", "male"), ("Lurtz", "male"), ("Maglub", "male"),
        ("Magrat", "male"), ("Mauhúr", "male"), ("Mazgak", "male"),
        ("Mogdush", "male"), ("Mograt", "male"), ("Mogrul", "male"),
        ("Muzgash", "male"), ("Nazgrat", "male"), ("Nekros", "male"),
        ("Ogdush", "male"), ("Ogruk", "male"), ("Olog", "male"),
        ("Orcobal", "male"), ("Radbug", "male"), ("Raglug", "male"),
        ("Ratbag", "male"), ("Ratgul", "male"), ("Shagrat", "male"),
        ("Shagruk", "male"), ("Sharku", "male"), ("Sharlug", "male"),
        ("Shrag", "male"), ("Snaga", "male"), ("Thrag", "male"),
        ("Thrak", "male"), ("Thraklug", "male"), ("Thrarat", "male"),
        ("Thrug", "male"), ("Thrugak", "male"), ("Thrugluk", "male"),
        ("Ugbash", "male"), ("Ugluk", "male"), ("Ugmog", "male"),
        ("Ugnash", "male"), ("Ugrat", "male"), ("Ugrul", "male"),
        ("Urmak", "male"), ("Urukdug", "male"), ("Urukgul", "male"),
        ("Urukmog", "male"), ("Uruknar", "male"), ("Urukrat", "male"),
        ("Urukrug", "male"), ("Urukshak", "male"), ("Urukthak", "male"),

        # female (60) — орочьи женские, добавим -а/-га для женственности
        ("Azga", "female"), ("Bolga", "female"), ("Gorbaga", "female"),
        ("Gothmaga", "female"), ("Grisha", "female"), ("Grubasha", "female"),
        ("Grukhala", "female"), ("Kragga", "female"), ("Kragna", "female"),
        ("Krugla", "female"), ("Lugba", "female"), ("Lugdusha", "female"),
        ("Lugna", "female"), ("Lurza", "female"), ("Magluba", "female"),
        ("Magra", "female"), ("Mauha", "female"), ("Mazga", "female"),
        ("Mogdusha", "female"), ("Mogra", "female"), ("Mogru", "female"),
        ("Muzga", "female"), ("Nazga", "female"), ("Nekra", "female"),
        ("Ogdu", "female"), ("Ogru", "female"), ("Ola", "female"),
        ("Orcoba", "female"), ("Radba", "female"), ("Ragla", "female"),
        ("Ratba", "female"), ("Ratgu", "female"), ("Shagra", "female"),
        ("Shagru", "female"), ("Sharkua", "female"), ("Sharlu", "female"),
        ("Shra", "female"), ("Snagua", "female"), ("Thraga", "female"),
        ("Thraka", "female"), ("Thraklu", "female"), ("Thrara", "female"),
        ("Thruga", "female"), ("Thrugka", "female"), ("Thruglu", "female"),
        ("Ugba", "female"), ("Uglu", "female"), ("Ugmu", "female"),
        ("Ugna", "female"), ("Ugra", "female"), ("Ugru", "female"),
        ("Urma", "female"), ("Urukdu", "female"), ("Urukgu", "female"),
        ("Urukmo", "female"), ("Uruknа", "female"), ("Urukra", "female"),
        ("Urukru", "female"), ("Uruksha", "female"), ("Uruktha", "female"),
    ],
    "northern": [
        # male (60)
        ("Agnar", "male"), ("Alvar", "male"), ("Andor", "male"),
        ("Anders", "male"), ("Arne", "male"), ("Arvid", "male"),
        ("Asbjorn", "male"), ("Asgeir", "male"), ("Aslak", "male"),
        ("Asmund", "male"), ("Balder", "male"), ("Bard", "male"),
        ("Birger", "male"), ("Bjarni", "male"), ("Bjorn", "male"),
        ("Bodvar", "male"), ("Bolli", "male"), ("Brand", "male"),
        ("Broddi", "male"), ("Brynjolf", "male"), ("Dag", "male"),
        ("Dagfinn", "male"), ("Egil", "male"), ("Eilif", "male"),
        ("Einar", "male"), ("Eirik", "male"), ("Eldar", "male"),
        ("Elias", "male"), ("Endre", "male"), ("Erland", "male"),
        ("Erlend", "male"), ("Eskil", "male"), ("Espen", "male"),
        ("Eystein", "male"), ("Eyvind", "male"), ("Finn", "male"),
        ("Finnbogi", "male"), ("Flemming", "male"), ("Folke", "male"),
        ("Fredrik", "male"), ("Frode", "male"), ("Galti", "male"),
        ("Gamli", "male"), ("Gardar", "male"), ("Gauti", "male"),
        ("Geir", "male"), ("Gest", "male"), ("Gisle", "male"),
        ("Gorm", "male"), ("Grim", "male"), ("Gudbrand", "male"),
        ("Gudmund", "male"), ("Gunnar", "male"), ("Gustav", "male"),
        ("Guttorm", "male"), ("Håkon", "male"), ("Halfdan", "male"),
        ("Hallbjorn", "male"), ("Halldor", "male"), ("Hallvard", "male"),

        # female (60)
        ("Astrid", "female"), ("Aud", "female"), ("Bergljot", "female"),
        ("Bjorg", "female"), ("Borghild", "female"), ("Dagny", "female"),
        ("Elsa", "female"), ("Freya", "female"), ("Freydis", "female"),
        ("Frida", "female"), ("Gerda", "female"), ("Groa", "female"),
        ("Gudrid", "female"), ("Gudrun", "female"), ("Gunnhild", "female"),
        ("Halldis", "female"), ("Hallfrid", "female"), ("Halldora", "female"),
        ("Hedda", "female"), ("Helga", "female"), ("Hervor", "female"),
        ("Hilda", "female"), ("Hildigunn", "female"), ("Hjordis", "female"),
        ("Inga", "female"), ("Inger", "female"), ("Ingeborg", "female"),
        ("Ingibjorg", "female"), ("Ingrid", "female"), ("Ingunn", "female"),
        ("Jorunn", "female"), ("Kari", "female"), ("Karin", "female"),
        ("Katla", "female"), ("Kirsten", "female"), ("Kjersti", "female"),
        ("Liv", "female"), ("Magnhild", "female"), ("Maren", "female"),
        ("Margit", "female"), ("Marit", "female"), ("Mette", "female"),
        ("Oda", "female"), ("Olga", "female"), ("Ragnhild", "female"),
        ("Randi", "female"), ("Rigmor", "female"), ("Runa", "female"),
        ("Signe", "female"), ("Signy", "female"), ("Sigrid", "female"),
        ("Sigrun", "female"), ("Solveig", "female"), ("Steinunn", "female"),
        ("Svanhild", "female"), ("Thea", "female"), ("Thora", "female"),
        ("Thordis", "female"), ("Tora", "female"), ("Unn", "female"),
    ],
    "southern": [
        # male (60)
        ("Alessandro", "male"), ("Alessio", "male"), ("Alfonso", "male"),
        ("Alfredo", "male"), ("Amadeo", "male"), ("Andrea", "male"),
        ("Angelo", "male"), ("Anselmo", "male"), ("Antonio", "male"),
        ("Arcangelo", "male"), ("Arnaldo", "male"), ("Arturo", "male"),
        ("Augusto", "male"), ("Aurelio", "male"), ("Baldassare", "male"),
        ("Bartolomeo", "male"), ("Battista", "male"), ("Benedetto", "male"),
        ("Bernardo", "male"), ("Bruno", "male"), ("Calogero", "male"),
        ("Camillo", "male"), ("Carlo", "male"), ("Carmelo", "male"),
        ("Casimiro", "male"), ("Cesare", "male"), ("Ciro", "male"),
        ("Claudio", "male"), ("Clemente", "male"), ("Corrado", "male"),
        ("Cosimo", "male"), ("Cristoforo", "male"), ("Damiano", "male"),
        ("Dante", "male"), ("Dario", "male"), ("Dino", "male"),
        ("Domenico", "male"), ("Donato", "male"), ("Durante", "male"),
        ("Edoardo", "male"), ("Egidio", "male"), ("Eligio", "male"),
        ("Emanuele", "male"), ("Emilio", "male"), ("Enrico", "male"),
        ("Enzo", "male"), ("Ercole", "male"), ("Ermanno", "male"),
        ("Ernesto", "male"), ("Ettore", "male"), ("Eugenio", "male"),
        ("Ezechiele", "male"), ("Fabio", "male"), ("Fabrizio", "male"),
        ("Fausto", "male"), ("Federico", "male"), ("Felice", "male"),
        ("Ferdinando", "male"), ("Filippo", "male"), ("Fiorello", "male"),

        # female (60)
        ("Alessandra", "female"), ("Alessia", "female"), ("Alfonsina", "female"),
        ("Amedea", "female"), ("Angelica", "female"), ("Angelina", "female"),
        ("Antonella", "female"), ("Antonia", "female"), ("Arianna", "female"),
        ("Aurora", "female"), ("Beatrice", "female"), ("Bella", "female"),
        ("Benedetta", "female"), ("Bianca", "female"), ("Brunella", "female"),
        ("Camilla", "female"), ("Carina", "female"), ("Carlotta", "female"),
        ("Carmela", "female"), ("Caterina", "female"), ("Cecilia", "female"),
        ("Celestina", "female"), ("Chiara", "female"), ("Cinzia", "female"),
        ("Clara", "female"), ("Clarissa", "female"), ("Concetta", "female"),
        ("Consolata", "female"), ("Cristina", "female"), ("Diana", "female"),
        ("Donatella", "female"), ("Elena", "female"), ("Elisa", "female"),
        ("Elisabetta", "female"), ("Emilia", "female"), ("Emma", "female"),
        ("Emanuela", "female"), ("Ernestina", "female"), ("Eva", "female"),
        ("Fabiana", "female"), ("Federica", "female"), ("Filomena", "female"),
        ("Fiorella", "female"), ("Flaminia", "female"), ("Francesca", "female"),
        ("Gabriella", "female"), ("Gemma", "female"), ("Gianna", "female"),
        ("Ginevra", "female"), ("Giorgia", "female"), ("Giovanna", "female"),
        ("Giulia", "female"), ("Grazia", "female"), ("Ilaria", "female"),
        ("Imelda", "female"), ("Isabella", "female"), ("Livia", "female"),
        ("Luisa", "female"), ("Maddalena", "female"), ("Marcella", "female"),
    ],
}


def expand(root: Path | None = None) -> None:
    if root is None:
        root = Path(__file__).resolve().parent.parent
    path = root / "data" / "seed_names.json"
    backup = root / "data" / "seed_names_backup.json"

    with path.open("r", encoding="utf-8") as f:
        existing = json.load(f)

    with backup.open("w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    seen = {(e["name"].lower(), e["style"]) for e in existing}
    result = list(existing)
    added, skipped = 0, 0

    for style, items in NEW_NAMES.items():
        for latin, gender in items:
            try:
                cyr = transliterate(latin)
                clean = normalize(cyr)
            except ValueError:
                skipped += 1
                continue
            key = (clean.lower(), style)
            if key in seen:
                skipped += 1
                continue
            seen.add(key)
            result.append({
                "name": clean, "style": style,
                "gender": gender, "source": "expanded_v3",
            })
            added += 1

    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Добавлено: {added}, пропущено: {skipped}, всего: {len(result)}")


if __name__ == "__main__":
    expand()