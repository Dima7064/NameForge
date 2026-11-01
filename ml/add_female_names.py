
"""Массовая разметка и добавление женских имён.

Логика:
- Имя есть в датасете, но gender=male → ПЕРЕЗАПИСЬ на female.
- Имени нет → ДОБАВИТЬ с female.
- Имя уже female → пропуск (idempotent).

Запуск:
    python -m ml.add_female_names
"""
import json
from collections import Counter
from pathlib import Path

from ml.preprocess import normalize
from ml.transliterate import transliterate

FEMALE_NAMES = {
    "elven": [
        "Arwen", "Galadriel", "Lúthien", "Idril", "Elwing", "Melian",
        "Yavanna", "Varda", "Nienna", "Estë", "Vairë", "Nessa",
        "Nerdanel", "Finduilas", "Aredhel", "Anairë", "Eärwen",
        "Celebrían", "Gilraen", "Ioreth", "Morwen", "Nienor", "Aerin",
        "Andreth", "Emeldir", "Rían", "Silmarien", "Tar-Míriel",
        "Ancalimë", "Almarian", "Erendis", "Emerië", "Írimë", "Lalwen",
        "Findis", "Indis", "Míriel", "Elenwë", "Amarië", "Ailinel",
        "Almiel", "Fíriel", "Ivriniel", "Lalaith", "Laurelin", "Nimloth",
        "Nimrodel", "Tauriel", "Yávien", "Artanis", "Aelithiel",
        "Aelwen", "Aeriel", "Aelindil", "Aeluin", "Amariel", "Anariel",
        "Andrielle", "Aranel", "Arien", "Arienh", "Arweniel", "Aurelith",
        "Calael", "Calathiel", "Celinn", "Celebrindal", "Culúrien",
        "Elaria", "Eledhwen", "Eleniel", "Elith", "Elnara", "Emelith",
        "Eruanna", "Faelwen", "Finiel", "Galathil", "Lindiriel",
        "Lórindol", "Mithrellas", "Nellas", "Aelith", "Aelinor",
        "Aeslin", "Almarien", "Elenwë", "Elbereth", "Aerlinn",
        "Aelindis", "Aelgwen", "Aerithiel", "Anarwen", "Andrethiel",
        "Araneth", "Ardiel", "Arwenith", "Celebrieth", "Celuinen",
        "Edlothiel", "Elaeriel", "Elarinya", "Eldalótë", "Elemírë",
        "Elenmil", "Elensar", "Elerondiel", "Elriniel", "Emelwen",
        "Eryniel", "Faelindis", "Faeriel", "Finduilaseth", "Galadhon",
        "Gilraeniel", "Idrieth", "Ilmariel", "Irimiel", "Ivrineth",
        "Laureliniel", "Lendiel", "Lindethiel", "Lúthieniel", "Maethiel",
        "Meldiel", "Melianthe", "Mírieliel", "Morwennel", "Nimmirel",
        "Nimrodeliel", "Níniel", "Ornethiel", "Silmarwen", "Taurieliel",
        "Tinúviel", "Uilosiel", "Vardiel", "Yavanniel", "Aerithil",
        "Aethelind", "Ailineliel", "Alderiel", "Almarion", "Amarieth",
        "Anarieth", "Andriethil", "Anoriel", "Aranethiel", "Ariethiel",
        "Arweneth", "Aurelien", "Calathwen", "Celebriwen", "Celebrieth",
        "Celuinen", "Elariael", "Eldalindë", "Elemíriel", "Elenmith",
        "Elensiwen", "Elindë", "Elweneth", "Elwingiel", "Emelindë",
        "Emelweniel", "Erynel", "Faenethiel", "Faenwiel", "Faelweniel",
        "Finduilaswen", "Galathwen", "Idriwen", "Ilmarië", "Ivrinë",
        "Laurelieth", "Lindeloth", "Lindeiel", "Luthieneth", "Maethwen",
        "Meldiel", "Melianeth", "Mirielwen", "Morweneth", "Nimlothwen",
        "Nimrodelwen", "Ornel", "Silmarweneth", "Taurielwen", "Uiloswen",
        "Vardien", "Yavannien",
    ],
    "human": [
        "Alice", "Amelia", "Angelica", "Anna", "Arabella", "Audrey",
        "Aurora", "Beatrice", "Bella", "Bernadette", "Blanche", "Bridget",
        "Camilla", "Caroline", "Catherine", "Cecilia", "Celestine",
        "Charlotte", "Clara", "Clarissa", "Clementine", "Constance",
        "Cora", "Cordelia", "Cynthia", "Daphne", "Deborah", "Delia",
        "Diana", "Dorothy", "Edith", "Edwina", "Elaine", "Eleanor",
        "Elizabeth", "Ella", "Emily", "Emma", "Esme", "Estella",
        "Esther", "Evelyn", "Felicity", "Fiona", "Flora", "Florence",
        "Genevieve", "Georgia", "Gertrude", "Grace", "Gwendolyn",
        "Harriet", "Hazel", "Helen", "Henrietta", "Hildegard", "Imogen",
        "Irene", "Isabel", "Isabella", "Ivy", "Jacqueline", "Jane",
        "Janet", "Jasmine", "Jessica", "Joanna", "Josephine", "Joy",
        "Judith", "Julia", "Juliet", "June", "Kate", "Katherine",
        "Kathleen", "Laura", "Lavinia", "Leah", "Lillian", "Lily",
        "Lucy", "Lydia", "Mabel", "Madeleine", "Madeline", "Margaret",
        "Margery", "Maria", "Marian", "Marjorie", "Martha", "Mary",
        "Matilda", "Maud", "Mavis", "Mildred", "Millicent", "Miranda",
        "Molly", "Muriel", "Nancy", "Naomi", "Natalie", "Nell", "Nora",
        "Octavia", "Olive", "Olivia", "Ophelia", "Pamela", "Patricia",
        "Pauline", "Pearl", "Penelope", "Phoebe", "Phyllis", "Priscilla",
        "Rachel", "Rebecca", "Rosalind", "Rose", "Rosemary", "Ruby",
        "Ruth", "Sabrina", "Sally", "Samantha", "Sarah", "Scarlett",
        "Selina", "Serena", "Sophie", "Stella", "Susan", "Sylvia",
        "Tessa", "Theodora", "Theresa", "Ursula", "Valerie", "Vanessa",
        "Vera", "Victoria", "Violet", "Virginia", "Vivian", "Winifred",
        "Yvonne", "Zoe", "Adelaide", "Agatha", "Agnes", "Agnella",
        "Alicia", "Allison", "Amanda", "Amber", "Amy", "Andrea",
        "Angela", "Anita", "Ann", "Annabel", "Anne", "Annette",
        "Anthea", "Antonia", "April", "Arlene", "Astrid", "Athena",
        "Augusta", "Ava", "Avis", "Barbara", "Beatrix", "Belinda",
        "Belle", "Bertha", "Beryl", "Bethany", "Beverly", "Bonnie",
        "Brenda", "Brianna", "Bronwen", "Caitlin", "Candace", "Cara",
        "Carla", "Carmen", "Carol", "Catherine", "Cathy", "Celia",
        "Charity", "Cheryl", "Chloe", "Christine", "Christina",
        "Clare", "Claudia", "Colette", "Colleen", "Connie", "Coral",
        "Crystal", "Daisy", "Dana", "Danielle", "Daphne", "Darlene",
        "Dawn", "Deanna", "Debra", "Denise", "Diana", "Dolly",
        "Dolores", "Donna", "Dora", "Doreen", "Doris", "Dorothea",
        "Edna", "Eileen", "Elisa", "Elise", "Ellen", "Eloise",
        "Elsie", "Emilia", "Erica", "Erika", "Erin", "Estelle",
        "Ethel", "Eunice", "Eva", "Evangeline", "Eve", "Faith",
        "Fannie", "Fay", "Faye", "Francine", "Freda", "Gabriella",
        "Gail", "Gayle", "Gemma", "Gina", "Ginger", "Gladys",
        "Glenda", "Gloria", "Goldie", "Greta", "Gwendoline", "Gwyneth",
        "Hailey", "Hannah", "Harriet", "Hattie", "Hazel", "Heather",
        "Heidi", "Helena", "Hester", "Hilda", "Holly", "Hope",
        "Ida", "Ingrid", "Irene", "Iris", "Isla", "Ivy", "Jade",
        "Jamie", "Jasmine", "Jean", "Jeanne", "Jenny", "Jessie",
        "Jill", "Jocelyn", "Jodie", "Joelle", "Johanna", "Jolene",
        "Josephine", "Josie", "Joyce", "Juanita", "Judy", "Julia",
        "Juliana", "Julie", "Justine", "Karen", "Kari", "Kate",
        "Kathleen", "Kathryn", "Kathy", "Katie", "Kay", "Kayla",
        "Keira", "Kelly", "Kendra", "Kerry", "Kiera", "Kimberly",
        "Kirsten", "Kitty", "Kristen", "Kristin", "Kristina", "Lacey",
    ],
    "orcish": [
        "Azga", "Bolga", "Gorbaga", "Gothmaga", "Grisha", "Grubasha",
        "Grukhala", "Kragga", "Kragna", "Krugla", "Lugba", "Lugdusha",
        "Lugna", "Lurza", "Magluba", "Magra", "Mauha", "Mazga",
        "Mogdusha", "Mogra", "Mogru", "Muzga", "Nazga", "Nekra",
        "Ogdu", "Ogru", "Ola", "Orcoba", "Radba", "Ragla", "Ratba",
        "Ratgu", "Shagra", "Shagru", "Sharkua", "Sharlu", "Shra",
        "Snagua", "Thraga", "Thraka", "Thraklu", "Thrara", "Thruga",
        "Thrugka", "Thruglu", "Ugba", "Uglu", "Ugmu", "Ugna",
        "Ugra", "Ugru", "Urma", "Urukdu", "Urukgu", "Urukmo",
        "Urukra", "Urukru", "Uruksha", "Uruktha", "Azgara", "Bolgra",
        "Gorbagra", "Gothmagra", "Grishana", "Grubashka", "Grukhalina",
        "Kraggina", "Kragnara", "Kruglash", "Lugbara", "Lugdushka",
        "Lugnara", "Lurzana", "Maglubina", "Magrasha", "Mauhana",
        "Mazgana", "Mogdushka", "Mograna", "Mogruna", "Muzgana",
        "Nazgana", "Nekrana", "Ogdura", "Ogriana", "Olasha",
        "Orcobina", "Radbasha", "Raglana", "Ratbasha", "Ratguna",
        "Shagrana", "Shagruna", "Sharkuana", "Sharluna", "Shrana",
        "Snaguana", "Thragana", "Thrakasha", "Thrakluna", "Thrarana",
        "Thrugasha", "Thrugluna", "Ugasha", "Ugluna", "Ugmuna",
        "Ugnasha", "Ugrasha", "Ugruna", "Urmasha", "Urukduna",
        "Urukguna", "Urukmosha", "Urukrana", "Urukruna", "Urukshа",
        "Uruktha", "Kharza", "Kharzana", "Krazga", "Krazgana",
        "Morgasha", "Morgaza", "Morgula", "Dargasha", "Dargula",
        "Murgasha", "Murgula", "Bargasha", "Bargula", "Urgasha",
        "Urgula", "Snagratha", "Snagruna", "Grizna", "Griznatha",
        "Shagratha", "Thragatha", "Mogdatha", "Ogmatha", "Uzgatha",
        "Uzgula", "Kharzatha", "Krazgatha", "Morgatha", "Urukkha",
        "Urukka", "Krazka", "Morgka", "Shagrа", "Thrugna",
    ],
    "northern": [
        "Astrid", "Aud", "Bergljot", "Bjorg", "Borghild", "Dagny",
        "Elsa", "Freya", "Freydis", "Frida", "Gerda", "Groa", "Gudrid",
        "Gudrun", "Gunnhild", "Halldis", "Hallfrid", "Halldora", "Hedda",
        "Helga", "Hervor", "Hilda", "Hildigunn", "Hjordis", "Inga",
        "Inger", "Ingeborg", "Ingibjorg", "Ingrid", "Ingunn", "Jorunn",
        "Kari", "Karin", "Katla", "Kirsten", "Kjersti", "Liv",
        "Magnhild", "Maren", "Margit", "Marit", "Mette", "Oda", "Olga",
        "Ragnhild", "Randi", "Rigmor", "Runa", "Signe", "Signy",
        "Sigrid", "Sigrun", "Solveig", "Steinunn", "Svanhild", "Thea",
        "Thora", "Thordis", "Tora", "Unn", "Aase", "Agnete", "Annika",
        "Anja", "Astri", "Benedikte", "Berit", "Birgit", "Birgitta",
        "Bodil", "Brita", "Brynhild", "Cecilie", "Dagmar", "Dorte",
        "Eldbjorg", "Eli", "Ellen", "Elin", "Embla", "Erika",
        "Frøydis", "Gerd", "Greta", "Gudbjorg", "Gunhild", "Gunn",
        "Guri", "Gyda", "Hanna", "Hanne", "Hege", "Heidi", "Helle",
        "Hilde", "Hjørdis", "Ida", "Ingvild", "Jorid", "Kaja",
        "Kristin", "Kristina", "Laila", "Lene", "Linda", "Linn",
        "Lisbet", "Malin", "Mari", "Martha", "Mia", "Mona", "Møyfrid",
        "Nora", "Oddny", "Olaug", "Oline", "Ragna", "Reidun", "Rita",
        "Ruth", "Sanna", "Silje", "Siri", "Sissel", "Stine", "Sunniva",
        "Susanne", "Synnøve", "Tiril", "Tone", "Tonje", "Torhild",
        "Tordis", "Tove", "Trine", "Tuva", "Unni", "Vigdis", "Wenche",
        "Ylva", "Åshild", "Åse", "Øydis", "Agnes", "Albjorg",
        "Aldis", "Alfhild", "Alva", "Arna", "Arnthora", "Aslaug",
        "Asrun", "Asta", "Audhild", "Audrun", "Aulie", "Ava",
        "Baldfrid", "Bera", "Bera", "Bergdis", "Bergthora", "Betty",
        "Birk", "Bjarnhild", "Borgny", "Borghild", "Botilda", "Bryndis",
        "Dagbjartur", "Dagfrid", "Dagmar", "Dalla", "Disa", "Dorte",
        "Dyveke", "Edda", "Edla", "Eldrid", "Elinborg", "Ellisif",
        "Elmfrid", "Elna", "Elvy", "Embla", "Erika", "Erla",
        "Erna", "Eydis", "Eyrun", "Fastny", "Filip", "Fjola",
        "Fredrika", "Freygerd", "Frøya", "Frøydis", "Gærd", "Gerd",
        "Gerda", "Gisla", "Gjerd", "Grimhild", "Groa", "Gudbjorg",
        "Gudfrid", "Gudlaug", "Gudny", "Gudrun", "Gulli", "Gunnhild",
        "Gunnlaug", "Gunnvor", "Gyða", "Haldis", "Halldora", "Hallgerd",
        "Harpa", "Hekla", "Helga", "Hella", "Herdís", "Hilda",
        "Hildigunn", "Hjaltfrid", "Hjördis", "Hlíf", "Hrafnhild",
        "Hrefna", "Hulda", "Hyrna", "Inga", "Ingebjorg", "Ingibjorg",
        "Ingifrid", "Ingigerd", "Ingileif", "Ingirid", "Ingunn",
        "Isold", "Jartrud", "Jofrid", "Jorid", "Jorunn", "Jorunn",
        "Judit", "Jytte", "Kaja", "Karin", "Katla", "Katrin",
        "Kirsten", "Kjersti", "Kolfinna", "Kristbjorg", "Kristin",
    ],
    "southern": [
        "Alessandra", "Alessia", "Alfonsina", "Amedea", "Angelica",
        "Angelina", "Antonella", "Antonia", "Arianna", "Aurora",
        "Beatrice", "Bella", "Benedetta", "Bianca", "Brunella",
        "Camilla", "Carina", "Carlotta", "Carmela", "Caterina",
        "Cecilia", "Celestina", "Chiara", "Cinzia", "Clara",
        "Clarissa", "Concetta", "Consolata", "Cristina", "Diana",
        "Donatella", "Elena", "Elisa", "Elisabetta", "Emilia", "Emma",
        "Emanuela", "Ernestina", "Eva", "Fabiana", "Federica",
        "Filomena", "Fiorella", "Flaminia", "Francesca", "Gabriella",
        "Gemma", "Gianna", "Ginevra", "Giorgia", "Giovanna", "Giulia",
        "Grazia", "Ilaria", "Imelda", "Isabella", "Livia", "Luisa",
        "Maddalena", "Marcella", "Marisa", "Marta", "Martina",
        "Matilde", "Melissa", "Michela", "Milena", "Monica", "Nadia",
        "Natalia", "Nicoletta", "Noemi", "Ornella", "Paola",
        "Patrizia", "Perla", "Rachele", "Raffaella", "Rebecca",
        "Renata", "Rita", "Roberta", "Romina", "Rosa", "Rosalia",
        "Rossana", "Sabrina", "Sara", "Selena", "Serena", "Silvia",
        "Simona", "Sofia", "Sonia", "Stefania", "Susanna", "Tania",
        "Teresa", "Tina", "Valentina", "Valeria", "Vanessa",
        "Veronica", "Viola", "Virginia", "Viviana", "Zita", "Adele",
        "Adriana", "Agnese", "Alberta", "Alda", "Alida", "Alina",
        "Amalia", "Ambra", "Anastasia", "Anita", "Annamaria",
        "Annarita", "Annunziata", "Antonietta", "Apollonia",
        "Assunta", "Aurelia", "Barbara", "Bernardina", "Betta",
        "Bice", "Bionda", "Brigida", "Bruna", "Calogera", "Candida",
        "Carla", "Carmina", "Celeste", "Cesira", "Clarice",
        "Clementina", "Clotilde", "Colombina", "Costanza",
        "Cristiana", "Daniela", "Daria", "Debora", "Diletta", "Dina",
        "Dora", "Dorotea", "Edvige", "Eleonora", "Elettra", "Eliana",
        "Eloisa", "Enrica", "Erika", "Ermelinda", "Ester", "Eugenia",
        "Evelina", "Fabiola", "Felicita", "Fernanda", "Fiamma",
        "Fiammetta", "Flora", "Flavia", "Fortunata", "Franca",
        "Gaetana", "Genoveffa", "Geraldina", "Germana", "Giacinta",
        "Gilda", "Gina", "Gioia", "Gisella", "Giuditta", "Giuliana",
        "Giuseppina", "Giusta", "Gloria", "Graziella", "Guglielma",
        "Ida", "Ilda", "Immacolata", "Ines", "Irene", "Irma",
        "Italia", "Laila", "Lara", "Laura", "Lea", "Lelia", "Lena",
        "Leonarda", "Letizia", "Lia", "Liana", "Lidia", "Liliana",
        "Lina", "Linda", "Lisa", "Loredana", "Lorella", "Lorena",
        "Luana", "Lucia", "Luciana", "Lucrezia", "Ludovica",
        "Luigia", "Luminosa", "Mafalda", "Malvina", "Manuela",
        "Mara", "Marcellina", "Margherita", "Maria", "Mariana",
        "Marilena", "Marina", "Mariolina", "Marzia", "Melania",
        "Mercedes", "Michaela", "Mimma", "Mirella", "Miriam",
        "Morena", "Nadja", "Natascia", "Nilde", "Nina", "Norma",
        "Novella", "Nunzia", "Olimpia", "Ombretta", "Ortensia",
        "Ottavia", "Petra", "Pia", "Piera", "Pina", "Pinuccia",
        "Priscilla", "Ramona", "Regina", "Rina", "Rosangela",
        "Rosanna", "Rosaria", "Rossella", "Sabina", "Samanta",
        "Sandra", "Savina", "Serafina", "Silvana", "Simonetta",
        "Stella", "Svetlana", "Tamara", "Tatiana", "Tiziana",
        "Tommasina", "Vanesa", "Vera", "Vilma", "Vita", "Vittoria",
        "Wanda", "Wilma", "Zaira", "Zoe",
    ],
}


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    path = root / "data" / "seed_names.json"

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Индекс (name_lower, style) → entry
    by_key = {(e["name"].lower(), e["style"]): e for e in data}

    overwritten = 0
    added = 0
    skipped = 0

    for style, names in FEMALE_NAMES.items():
        for latin in names:
            try:
                cyr = transliterate(latin)
                clean = normalize(cyr)
            except ValueError:
                skipped += 1
                continue
            key = (clean.lower(), style)
            if key in by_key:
                entry = by_key[key]
                if entry.get("gender") != "female":
                    entry["gender"] = "female"
                    overwritten += 1
                else:
                    skipped += 1
            else:
                new = {
                    "name": clean, "style": style,
                    "gender": "female", "source": "female_fix",
                }
                data.append(new)
                by_key[key] = new
                added += 1

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    c = Counter(e.get("gender", "unisex") for e in data)
    print(f"Перезаписано male→female: {overwritten}")
    print(f"Добавлено новых female: {added}")
    print(f"Пропущено (уже female или невалидно): {skipped}")
    print(f"Итого записей: {len(data)}")
    print(f"Распределение: {dict(c)}")


if __name__ == "__main__":
    main()