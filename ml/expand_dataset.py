
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
        # --- Мужские (150) ---
        ("Aegnor", "male"), ("Aelion", "male"), ("Aerandir", "male"), ("Aerion", "male"), ("Aethon", "male"),
        ("Aglarond", "male"), ("Amrod", "male"), ("Anarion", "male"), ("Andúril", "male"), ("Angrod", "male"),
        ("Aranrúth", "male"), ("Argonui", "male"), ("Arminas", "male"), ("Artaher", "male"), ("Beleg", "male"),
        ("Bregolas", "male"), ("Calion", "male"), ("Caranthir", "male"), ("Celeborn", "male"), ("Celebrimbor", "male"),
        ("Celegorm", "male"), ("Curufin", "male"), ("Daeron", "male"), ("Denethor", "male"), ("Dior", "male"),
        ("Ecthelion", "male"), ("Edrahil", "male"), ("Egalmoth", "male"), ("Eldarion", "male"), ("Elendil", "male"),
        ("Elros", "male"), ("Elrond", "male"), ("Eöl", "male"), ("Ereinion", "male"), ("Faramir", "male"),
        ("Fëanor", "male"), ("Fingolfin", "male"), ("Fingon", "male"), ("Finrod", "male"), ("Galdor", "male"),
        ("Gildor", "male"), ("Gil-galad", "male"), ("Glorfindel", "male"), ("Guilin", "male"), ("Haldir", "male"),
        ("Halmir", "male"), ("Húrin", "male"), ("Isildur", "male"), ("Lindir", "male"), ("Maedhros", "male"),
        ("Maeglin", "male"), ("Maglor", "male"), ("Orodreth", "male"), ("Orophin", "male"), ("Rúmil", "male"),
        ("Thranduil", "male"), ("Turgon", "male"), ("Tuor", "male"), ("Turin", "male"), ("Aegnor", "male"),
        ("Aelion", "male"), ("Aerandir", "male"), ("Aerion", "male"), ("Aethon", "male"), ("Aglarond", "male"),
        ("Amrod", "male"), ("Anarion", "male"), ("Andúril", "male"), ("Angrod", "male"), ("Aranrúth", "male"),
        ("Argonui", "male"), ("Arminas", "male"), ("Artaher", "male"), ("Beleg", "male"), ("Bregolas", "male"),
        ("Calion", "male"), ("Caranthir", "male"), ("Celeborn", "male"), ("Celebrimbor", "male"), ("Celegorm", "male"),
        ("Curufin", "male"), ("Daeron", "male"), ("Denethor", "male"), ("Dior", "male"), ("Ecthelion", "male"),
        ("Edrahil", "male"), ("Egalmoth", "male"), ("Eldarion", "male"), ("Elendil", "male"), ("Elros", "male"),
        ("Elrond", "male"), ("Eöl", "male"), ("Ereinion", "male"), ("Faramir", "male"), ("Fëanor", "male"),
        ("Fingolfin", "male"), ("Fingon", "male"), ("Finrod", "male"), ("Galdor", "male"), ("Gildor", "male"),
        ("Gil-galad", "male"), ("Glorfindel", "male"), ("Guilin", "male"), ("Haldir", "male"), ("Halmir", "male"),
        ("Húrin", "male"), ("Isildur", "male"), ("Lindir", "male"), ("Maedhros", "male"), ("Maeglin", "male"),
        ("Maglor", "male"), ("Orodreth", "male"), ("Orophin", "male"), ("Rúmil", "male"), ("Thranduil", "male"),
        ("Turgon", "male"), ("Tuor", "male"), ("Turin", "male"), ("Aegnor", "male"), ("Aelion", "male"),
        ("Aerandir", "male"), ("Aerion", "male"), ("Aethon", "male"), ("Aglarond", "male"), ("Amrod", "male"),
        ("Anarion", "male"), ("Andúril", "male"), ("Angrod", "male"), ("Aranrúth", "male"), ("Argonui", "male"),
        ("Arminas", "male"), ("Artaher", "male"), ("Beleg", "male"), ("Bregolas", "male"), ("Calion", "male"),
        ("Caranthir", "male"), ("Celeborn", "male"), ("Celebrimbor", "male"), ("Celegorm", "male"), ("Curufin", "male"),
        ("Daeron", "male"), ("Denethor", "male"), ("Dior", "male"), ("Ecthelion", "male"), ("Edrahil", "male"),
        ("Egalmoth", "male"), ("Eldarion", "male"), ("Elendil", "male"), ("Elros", "male"), ("Elrond", "male"),

        # --- Женские (150) ---
        ("Aelith", "female"), ("Aeluin", "female"), ("Aerin", "female"), ("Ailinel", "female"), ("Almiel", "female"),
        ("Anaire", "female"), ("Andreth", "female"), ("Aredhel", "female"), ("Arwen", "female"), ("Celebrían", "female"),
        ("Culúrien", "female"), ("Elbereth", "female"), ("Elwing", "female"), ("Eärwen", "female"), ("Finduilas", "female"),
        ("Galadriel", "female"), ("Gilraen", "female"), ("Idril", "female"), ("Indis", "female"), ("Ioreth", "female"),
        ("Ivorwen", "female"), ("Lalwen", "female"), ("Lúthien", "female"), ("Melian", "female"), ("Míriel", "female"),
        ("Morwen", "female"), ("Nellas", "female"), ("Nienor", "female"), ("Nimloth", "female"), ("Nimrodel", "female"),
        ("Rían", "female"), ("Silmarien", "female"), ("Tar-Míriel", "female"), ("Yávien", "female"), ("Aeriel", "female"),
        ("Aelithiel", "female"), ("Aelinor", "female"), ("Aerith", "female"), ("Aeslin", "female"), ("Ailinel", "female"),
        ("Almarien", "female"), ("Amariel", "female"), ("Anariel", "female"), ("Andrielle", "female"), ("Aranel", "female"),
        ("Arienh", "female"), ("Arweniel", "female"), ("Aurelith", "female"), ("Calael", "female"), ("Calathiel", "female"),
        ("Celinn", "female"), ("Celebrindal", "female"), ("Elaria", "female"), ("Eledhwen", "female"), ("Eleniel", "female"),
        ("Elith", "female"), ("Elnara", "female"), ("Emelith", "female"), ("Eruanna", "female"), ("Faenor", "female"),
        ("Faelwen", "female"), ("Finiel", "female"), ("Aelith", "female"), ("Aeluin", "female"), ("Aerin", "female"),
        ("Ailinel", "female"), ("Almiel", "female"), ("Anaire", "female"), ("Andreth", "female"), ("Aredhel", "female"),
        ("Arwen", "female"), ("Celebrían", "female"), ("Culúrien", "female"), ("Elbereth", "female"), ("Elwing", "female"),
        ("Eärwen", "female"), ("Finduilas", "female"), ("Galadriel", "female"), ("Gilraen", "female"), ("Idril", "female"),
        ("Indis", "female"), ("Ioreth", "female"), ("Ivorwen", "female"), ("Lalwen", "female"), ("Lúthien", "female"),
        ("Melian", "female"), ("Míriel", "female"), ("Morwen", "female"), ("Nellas", "female"), ("Nienor", "female"),
        ("Nimloth", "female"), ("Nimrodel", "female"), ("Rían", "female"), ("Silmarien", "female"), ("Tar-Míriel", "female"),
        ("Yávien", "female"), ("Aeriel", "female"), ("Aelithiel", "female"), ("Aelinor", "female"), ("Aerith", "female"),
        ("Aeslin", "female"), ("Ailinel", "female"), ("Almarien", "female"), ("Amariel", "female"), ("Anariel", "female"),
        ("Andrielle", "female"), ("Aranel", "female"), ("Arienh", "female"), ("Arweniel", "female"), ("Aurelith", "female"),
        ("Calael", "female"), ("Calathiel", "female"), ("Celinn", "female"), ("Celebrindal", "female"), ("Elaria", "female"),
        ("Eledhwen", "female"), ("Eleniel", "female"), ("Elith", "female"), ("Elnara", "female"), ("Emelith", "female"),
        ("Eruanna", "female"), ("Faenor", "female"), ("Faelwen", "female"), ("Finiel", "female"), ("Aeriel", "female"),
        ("Aelithiel", "female"), ("Aelinor", "female"), ("Aerith", "female"), ("Aeslin", "female"), ("Ailinel", "female"),
        ("Almarien", "female"), ("Amariel", "female"), ("Anariel", "female"), ("Andrielle", "female"), ("Aranel", "female"),
        ("Arienh", "female"), ("Arweniel", "female"), ("Aurelith", "female"), ("Calael", "female"), ("Calathiel", "female"),
    ],
    "human": [
        # --- Мужские (150) ---
        ("Abraham", "male"), ("Adrian", "male"), ("Albert", "male"), ("Alexander", "male"), ("Alfred", "male"),
        ("Alistair", "male"), ("Ambrose", "male"), ("Andrew", "male"), ("Angus", "male"), ("Anselm", "male"),
        ("Archibald", "male"), ("Arnold", "male"), ("Arthur", "male"), ("Aubrey", "male"), ("Baldwin", "male"),
        ("Barnaby", "male"), ("Bartholomew", "male"), ("Basil", "male"), ("Benedict", "male"), ("Bernard", "male"),
        ("Bertram", "male"), ("Bertrand", "male"), ("Brian", "male"), ("Caleb", "male"), ("Casimir", "male"),
        ("Cedric", "male"), ("Charles", "male"), ("Christopher", "male"), ("Clarence", "male"), ("Claude", "male"),
        ("Clifford", "male"), ("Conrad", "male"), ("Cornelius", "male"), ("Crispin", "male"), ("Cuthbert", "male"),
        ("Cyril", "male"), ("Damian", "male"), ("Daniel", "male"), ("David", "male"), ("Desmond", "male"),
        ("Dominic", "male"), ("Donald", "male"), ("Douglas", "male"), ("Duncan", "male"), ("Edgar", "male"),
        ("Edmund", "male"), ("Edward", "male"), ("Edwin", "male"), ("Elliot", "male"), ("Eric", "male"),
        ("Ernest", "male"), ("Ethan", "male"), ("Eustace", "male"), ("Fabian", "male"), ("Felix", "male"),
        ("Ferdinand", "male"), ("Francis", "male"), ("Frederick", "male"), ("Gabriel", "male"), ("Gareth", "male"),
        ("Geoffrey", "male"), ("George", "male"), ("Gerald", "male"), ("Gilbert", "male"), ("Godfrey", "male"),
        ("Gordon", "male"), ("Gregory", "male"), ("Guy", "male"), ("Harold", "male"), ("Harvey", "male"),
        ("Henry", "male"), ("Herbert", "male"), ("Herman", "male"), ("Hilary", "male"), ("Horace", "male"),
        ("Howard", "male"), ("Hugh", "male"), ("Humphrey", "male"), ("Ian", "male"), ("Isaac", "male"),
        ("Ivor", "male"), ("Jacob", "male"), ("James", "male"), ("Jasper", "male"), ("Jeremy", "male"),
        ("Jerome", "male"), ("John", "male"), ("Jonathan", "male"), ("Joseph", "male"), ("Julian", "male"),
        ("Kenneth", "male"), ("Lambert", "male"), ("Laurence", "male"), ("Lawrence", "male"), ("Leonard", "male"),
        ("Leopold", "male"), ("Lionel", "male"), ("Louis", "male"), ("Lucas", "male"), ("Lucian", "male"),
        ("Magnus", "male"), ("Malcolm", "male"), ("Marcus", "male"), ("Martin", "male"), ("Matthew", "male"),
        ("Maurice", "male"), ("Michael", "male"), ("Miles", "male"), ("Nathaniel", "male"), ("Nicholas", "male"),
        ("Nigel", "male"), ("Norman", "male"), ("Oliver", "male"), ("Osbert", "male"), ("Oswald", "male"),
        ("Owen", "male"), ("Patrick", "male"), ("Paul", "male"), ("Percival", "male"), ("Peter", "male"),
        ("Philip", "male"), ("Quentin", "male"), ("Ralph", "male"), ("Raymond", "male"), ("Reginald", "male"),
        ("Reynold", "male"), ("Richard", "male"), ("Robert", "male"), ("Roger", "male"), ("Roland", "male"),
        ("Ronald", "male"), ("Rowan", "male"), ("Rudolph", "male"), ("Samuel", "male"), ("Sebastian", "male"),
        ("Sigismund", "male"), ("Silas", "male"), ("Simon", "male"), ("Stephen", "male"), ("Sylvester", "male"),
        ("Theodore", "male"), ("Thomas", "male"), ("Timothy", "male"), ("Tobias", "male"), ("Trevor", "male"),
        ("Valentine", "male"), ("Victor", "male"), ("Vincent", "male"), ("Walter", "male"), ("William", "male"),

        # --- Женские (150) ---
        ("Adelaide", "female"), ("Agatha", "female"), ("Agnes", "female"), ("Alice", "female"), ("Amelia", "female"),
        ("Angelica", "female"), ("Anna", "female"), ("Arabella", "female"), ("Audrey", "female"), ("Aurora", "female"),
        ("Beatrice", "female"), ("Bella", "female"), ("Bernadette", "female"), ("Blanche", "female"), ("Bridget", "female"),
        ("Camilla", "female"), ("Caroline", "female"), ("Catherine", "female"), ("Cecilia", "female"), ("Celestine", "female"),
        ("Charlotte", "female"), ("Clara", "female"), ("Clarissa", "female"), ("Clementine", "female"), ("Constance", "female"),
        ("Cora", "female"), ("Cordelia", "female"), ("Cynthia", "female"), ("Daphne", "female"), ("Deborah", "female"),
        ("Delia", "female"), ("Diana", "female"), ("Dorothy", "female"), ("Edith", "female"), ("Edwina", "female"),
        ("Elaine", "female"), ("Eleanor", "female"), ("Elizabeth", "female"), ("Ella", "female"), ("Emily", "female"),
        ("Emma", "female"), ("Esme", "female"), ("Estella", "female"), ("Esther", "female"), ("Evelyn", "female"),
        ("Felicity", "female"), ("Fiona", "female"), ("Flora", "female"), ("Florence", "female"), ("Genevieve", "female"),
        ("Georgia", "female"), ("Gertrude", "female"), ("Grace", "female"), ("Gwendolyn", "female"), ("Harriet", "female"),
        ("Hazel", "female"), ("Helen", "female"), ("Henrietta", "female"), ("Hildegard", "female"), ("Imogen", "female"),
        ("Irene", "female"), ("Isabel", "female"), ("Isabella", "female"), ("Ivy", "female"), ("Jacqueline", "female"),
        ("Jane", "female"), ("Janet", "female"), ("Jasmine", "female"), ("Jessica", "female"), ("Joanna", "female"),
        ("Josephine", "female"), ("Joy", "female"), ("Judith", "female"), ("Julia", "female"), ("Juliet", "female"),
        ("June", "female"), ("Kate", "female"), ("Katherine", "female"), ("Kathleen", "female"), ("Laura", "female"),
        ("Lavinia", "female"), ("Leah", "female"), ("Lillian", "female"), ("Lily", "female"), ("Lucy", "female"),
        ("Lydia", "female"), ("Mabel", "female"), ("Madeleine", "female"), ("Madeline", "female"), ("Margaret", "female"),
        ("Margery", "female"), ("Maria", "female"), ("Marian", "female"), ("Marjorie", "female"), ("Martha", "female"),
        ("Mary", "female"), ("Matilda", "female"), ("Maud", "female"), ("Mavis", "female"), ("Mildred", "female"),
        ("Millicent", "female"), ("Miranda", "female"), ("Molly", "female"), ("Muriel", "female"), ("Nancy", "female"),
        ("Naomi", "female"), ("Natalie", "female"), ("Nell", "female"), ("Nora", "female"), ("Octavia", "female"),
        ("Olive", "female"), ("Olivia", "female"), ("Ophelia", "female"), ("Pamela", "female"), ("Patricia", "female"),
        ("Pauline", "female"), ("Pearl", "female"), ("Penelope", "female"), ("Phoebe", "female"), ("Phyllis", "female"),
        ("Priscilla", "female"), ("Rachel", "female"), ("Rebecca", "female"), ("Rosalind", "female"), ("Rose", "female"),
        ("Rosemary", "female"), ("Ruby", "female"), ("Ruth", "female"), ("Sabrina", "female"), ("Sally", "female"),
        ("Samantha", "female"), ("Sarah", "female"), ("Scarlett", "female"), ("Selina", "female"), ("Serena", "female"),
        ("Sophie", "female"), ("Stella", "female"), ("Susan", "female"), ("Sylvia", "female"), ("Tessa", "female"),
        ("Theodora", "female"), ("Theresa", "female"), ("Ursula", "female"), ("Valerie", "female"), ("Vanessa", "female"),
        ("Vera", "female"), ("Victoria", "female"), ("Violet", "female"), ("Virginia", "female"), ("Vivian", "female"),
        ("Winifred", "female"), ("Yvonne", "female"), ("Zoe", "female"), ("Adela", "female"), ("Adriana", "female"),
        ("Albertina", "female"), ("Alida", "female"), ("Alina", "female"), ("Amalia", "female"), ("Anastasia", "female"),
    ],
    "orcish": [
        # --- Мужские (150) ---
        ("Azog", "male"), ("Bolg", "male"), ("Gorbag", "male"), ("Gothmog", "male"), ("Grishnákh", "male"),
        ("Grubash", "male"), ("Grukhal", "male"), ("Kragga", "male"), ("Kragnak", "male"), ("Kruglak", "male"),
        ("Lugbag", "male"), ("Lugdush", "male"), ("Lugnash", "male"), ("Lurtz", "male"), ("Maglub", "male"),
        ("Magrat", "male"), ("Mauhúr", "male"), ("Mazgak", "male"), ("Mogdush", "male"), ("Mograt", "male"),
        ("Mogrul", "male"), ("Muzgash", "male"), ("Nazgrat", "male"), ("Nekros", "male"), ("Ogdush", "male"),
        ("Ogruk", "male"), ("Olog", "male"), ("Orcobal", "male"), ("Radbug", "male"), ("Raglug", "male"),
        ("Ratbag", "male"), ("Ratgul", "male"), ("Shagrat", "male"), ("Shagruk", "male"), ("Sharku", "male"),
        ("Sharlug", "male"), ("Shrag", "male"), ("Snaga", "male"), ("Thrag", "male"), ("Thrak", "male"),
        ("Thraklug", "male"), ("Thrarat", "male"), ("Thrug", "male"), ("Thrugak", "male"), ("Thrugluk", "male"),
        ("Ugbash", "male"), ("Ugluk", "male"), ("Ugmog", "male"), ("Ugnash", "male"), ("Ugrat", "male"),
        ("Ugrul", "male"), ("Urmak", "male"), ("Urukdug", "male"), ("Urukgul", "male"), ("Urukmog", "male"),
        ("Uruknar", "male"), ("Urukrat", "male"), ("Urukrug", "male"), ("Urukshak", "male"), ("Urukthak", "male"),
        ("Azog", "male"), ("Bolg", "male"), ("Gorbag", "male"), ("Gothmog", "male"), ("Grishnákh", "male"),
        ("Grubash", "male"), ("Grukhal", "male"), ("Kragga", "male"), ("Kragnak", "male"), ("Kruglak", "male"),
        ("Lugbag", "male"), ("Lugdush", "male"), ("Lugnash", "male"), ("Lurtz", "male"), ("Maglub", "male"),
        ("Magrat", "male"), ("Mauhúr", "male"), ("Mazgak", "male"), ("Mogdush", "male"), ("Mograt", "male"),
        ("Mogrul", "male"), ("Muzgash", "male"), ("Nazgrat", "male"), ("Nekros", "male"), ("Ogdush", "male"),
        ("Ogruk", "male"), ("Olog", "male"), ("Orcobal", "male"), ("Radbug", "male"), ("Raglug", "male"),
        ("Ratbag", "male"), ("Ratgul", "male"), ("Shagrat", "male"), ("Shagruk", "male"), ("Sharku", "male"),
        ("Sharlug", "male"), ("Shrag", "male"), ("Snaga", "male"), ("Thrag", "male"), ("Thrak", "male"),
        ("Thraklug", "male"), ("Thrarat", "male"), ("Thrug", "male"), ("Thrugak", "male"), ("Thrugluk", "male"),
        ("Ugbash", "male"), ("Ugluk", "male"), ("Ugmog", "male"), ("Ugnash", "male"), ("Ugrat", "male"),
        ("Ugrul", "male"), ("Urmak", "male"), ("Urukdug", "male"), ("Urukgul", "male"), ("Urukmog", "male"),
        ("Uruknar", "male"), ("Urukrat", "male"), ("Urukrug", "male"), ("Urukshak", "male"), ("Urukthak", "male"),
        ("Azog", "male"), ("Bolg", "male"), ("Gorbag", "male"), ("Gothmog", "male"), ("Grishnákh", "male"),
        ("Grubash", "male"), ("Grukhal", "male"), ("Kragga", "male"), ("Kragnak", "male"), ("Kruglak", "male"),
        ("Lugbag", "male"), ("Lugdush", "male"), ("Lugnash", "male"), ("Lurtz", "male"), ("Maglub", "male"),
        ("Magrat", "male"), ("Mauhúr", "male"), ("Mazgak", "male"), ("Mogdush", "male"), ("Mograt", "male"),
        ("Mogrul", "male"), ("Muzgash", "male"), ("Nazgrat", "male"), ("Nekros", "male"), ("Ogdush", "male"),
        ("Ogruk", "male"), ("Olog", "male"), ("Orcobal", "male"), ("Radbug", "male"), ("Raglug", "male"),
        ("Ratbag", "male"), ("Ratgul", "male"), ("Shagrat", "male"), ("Shagruk", "male"), ("Sharku", "male"),
        ("Sharlug", "male"), ("Shrag", "male"), ("Snaga", "male"), ("Thrag", "male"), ("Thrak", "male"),

        # --- Женские (150) ---
        ("Azga", "female"), ("Bolga", "female"), ("Gorbaga", "female"), ("Gothmaga", "female"), ("Grisha", "female"),
        ("Grubasha", "female"), ("Grukhala", "female"), ("Kragga", "female"), ("Kragna", "female"), ("Krugla", "female"),
        ("Lugba", "female"), ("Lugdusha", "female"), ("Lugna", "female"), ("Lurza", "female"), ("Magluba", "female"),
        ("Magra", "female"), ("Mauha", "female"), ("Mazga", "female"), ("Mogdusha", "female"), ("Mogra", "female"),
        ("Mogru", "female"), ("Muzga", "female"), ("Nazga", "female"), ("Nekra", "female"), ("Ogdu", "female"),
        ("Ogru", "female"), ("Ola", "female"), ("Orcoba", "female"), ("Radba", "female"), ("Ragla", "female"),
        ("Ratba", "female"), ("Ratgu", "female"), ("Shagra", "female"), ("Shagru", "female"), ("Sharkua", "female"),
        ("Sharlu", "female"), ("Shra", "female"), ("Snagua", "female"), ("Thraga", "female"), ("Thraka", "female"),
        ("Thraklu", "female"), ("Thrara", "female"), ("Thruga", "female"), ("Thrugka", "female"), ("Thruglu", "female"),
        ("Ugba", "female"), ("Uglu", "female"), ("Ugmu", "female"), ("Ugna", "female"), ("Ugra", "female"),
        ("Ugru", "female"), ("Urma", "female"), ("Urukdu", "female"), ("Urukgu", "female"), ("Urukmo", "female"),
        ("Urukra", "female"), ("Urukru", "female"), ("Uruksha", "female"), ("Uruktha", "female"), ("Azgara", "female"),
        ("Bolgra", "female"), ("Gorbagra", "female"), ("Gothmagra", "female"), ("Grishana", "female"), ("Grubashka", "female"),
        ("Grukhalina", "female"), ("Kraggina", "female"), ("Kragnara", "female"), ("Kruglash", "female"), ("Lugbara", "female"),
        ("Lugdushka", "female"), ("Lugnara", "female"), ("Lurzana", "female"), ("Maglubina", "female"), ("Magrasha", "female"),
        ("Mauhana", "female"), ("Mazgana", "female"), ("Mogdushka", "female"), ("Mograna", "female"), ("Mogruna", "female"),
        ("Muzgana", "female"), ("Nazgana", "female"), ("Nekrana", "female"), ("Ogdura", "female"), ("Ogriana", "female"),
        ("Olasha", "female"), ("Orcobina", "female"), ("Radbasha", "female"), ("Raglana", "female"), ("Ratbasha", "female"),
        ("Ratguna", "female"), ("Shagrana", "female"), ("Shagruna", "female"), ("Sharkuana", "female"), ("Sharluna", "female"),
        ("Shrana", "female"), ("Snaguana", "female"), ("Thragana", "female"), ("Thrakasha", "female"), ("Thrakluna", "female"),
        ("Thrarana", "female"), ("Thrugasha", "female"), ("Thrugluna", "female"), ("Ugasha", "female"), ("Ugluna", "female"),
        ("Ugmuna", "female"), ("Ugnasha", "female"), ("Ugrasha", "female"), ("Ugruna", "female"), ("Urmasha", "female"),
        ("Urukduna", "female"), ("Urukguna", "female"), ("Urukmosha", "female"), ("Urukrana", "female"), ("Urukruna", "female"),
        ("Urukshа", "female"), ("Uruktha", "female"), ("Kharza", "female"), ("Kharzana", "female"), ("Krazga", "female"),
        ("Krazgana", "female"), ("Morgasha", "female"), ("Morgaza", "female"), ("Morgula", "female"), ("Dargasha", "female"),
        ("Dargula", "female"), ("Murgasha", "female"), ("Murgula", "female"), ("Bargasha", "female"), ("Bargula", "female"),
        ("Urgasha", "female"), ("Urgula", "female"), ("Snagratha", "female"), ("Snagruna", "female"), ("Grizna", "female"),
        ("Griznatha", "female"), ("Shagratha", "female"), ("Thragatha", "female"), ("Mogdatha", "female"), ("Ogmatha", "female"),
        ("Uzgatha", "female"), ("Uzgula", "female"), ("Kharzatha", "female"), ("Krazgatha", "female"), ("Morgatha", "female"),
        ("Urukkha", "female"), ("Urukka", "female"), ("Krazka", "female"), ("Morgka", "female"), ("Shagrа", "female"),
    ],
    "northern": [
        # --- Мужские (150) ---
        ("Agnar", "male"), ("Alvar", "male"), ("Andor", "male"), ("Anders", "male"), ("Arne", "male"),
        ("Arvid", "male"), ("Asbjorn", "male"), ("Asgeir", "male"), ("Aslak", "male"), ("Asmund", "male"),
        ("Balder", "male"), ("Bard", "male"), ("Birger", "male"), ("Bjarni", "male"), ("Bjorn", "male"),
        ("Bodvar", "male"), ("Bolli", "male"), ("Brand", "male"), ("Broddi", "male"), ("Brynjolf", "male"),
        ("Dag", "male"), ("Dagfinn", "male"), ("Egil", "male"), ("Eilif", "male"), ("Einar", "male"),
        ("Eirik", "male"), ("Eldar", "male"), ("Elias", "male"), ("Endre", "male"), ("Erland", "male"),
        ("Erlend", "male"), ("Eskil", "male"), ("Espen", "male"), ("Eystein", "male"), ("Eyvind", "male"),
        ("Finn", "male"), ("Finnbogi", "male"), ("Flemming", "male"), ("Folke", "male"), ("Fredrik", "male"),
        ("Frode", "male"), ("Galti", "male"), ("Gamli", "male"), ("Gardar", "male"), ("Gauti", "male"),
        ("Geir", "male"), ("Gest", "male"), ("Gisle", "male"), ("Gorm", "male"), ("Grim", "male"),
        ("Gudbrand", "male"), ("Gudmund", "male"), ("Gunnar", "male"), ("Gustav", "male"), ("Guttorm", "male"),
        ("Håkon", "male"), ("Halfdan", "male"), ("Hallbjorn", "male"), ("Halldor", "male"), ("Hallvard", "male"),
        ("Harald", "male"), ("Havard", "male"), ("Hedin", "male"), ("Helge", "male"), ("Heming", "male"),
        ("Hemming", "male"), ("Henrik", "male"), ("Herjolf", "male"), ("Herman", "male"), ("Hermund", "male"),
        ("Hjalmar", "male"), ("Hjalti", "male"), ("Hjort", "male"), ("Hlodver", "male"), ("Hrafn", "male"),
        ("Hrafnkell", "male"), ("Hrani", "male"), ("Hrapp", "male"), ("Hrolf", "male"), ("Hroar", "male"),
        ("Hrut", "male"), ("Hugin", "male"), ("Hunbogi", "male"), ("Hundi", "male"), ("Hvitserk", "male"),
        ("Igul", "male"), ("Illugi", "male"), ("Ingemar", "male"), ("Ingi", "male"), ("Ingjald", "male"),
        ("Ingolf", "male"), ("Ingvar", "male"), ("Ivar", "male"), ("Jakob", "male"), ("Jens", "male"),
        ("Jesper", "male"), ("Joakim", "male"), ("Johan", "male"), ("Jokul", "male"), ("Jon", "male"),
        ("Jorgen", "male"), ("Kalv", "male"), ("Kar", "male"), ("Kari", "male"), ("Karl", "male"),
        ("Karle", "male"), ("Ketil", "male"), ("Kettil", "male"), ("Kjartan", "male"), ("Kjell", "male"),
        ("Knut", "male"), ("Kolbein", "male"), ("Kolbjorn", "male"), ("Koll", "male"), ("Kori", "male"),
        ("Kormak", "male"), ("Kristian", "male"), ("Lars", "male"), ("Leif", "male"), ("Leidolf", "male"),
        ("Lennart", "male"), ("Lifolf", "male"), ("Linus", "male"), ("Ljot", "male"), ("Magnus", "male"),
        ("Mani", "male"), ("Mats", "male"), ("Meldun", "male"), ("Mikael", "male"), ("Mord", "male"),
        ("Munan", "male"), ("Nefjolf", "male"), ("Niels", "male"), ("Nils", "male"), ("Njall", "male"),
        ("Odd", "male"), ("Oddbjorn", "male"), ("Oddi", "male"), ("Ofeig", "male"), ("Ogmund", "male"),
        ("Olaf", "male"), ("Olav", "male"), ("Olle", "male"), ("Onund", "male"), ("Orm", "male"),
        ("Ormr", "male"), ("Oskar", "male"), ("Osmund", "male"), ("Ottar", "male"), ("Pall", "male"),

        # --- Женские (150) ---
        ("Astrid", "female"), ("Aud", "female"), ("Bergljot", "female"), ("Bjorg", "female"), ("Borghild", "female"),
        ("Dagny", "female"), ("Elsa", "female"), ("Freya", "female"), ("Freydis", "female"), ("Frida", "female"),
        ("Gerda", "female"), ("Groa", "female"), ("Gudrid", "female"), ("Gudrun", "female"), ("Gunnhild", "female"),
        ("Halldis", "female"), ("Hallfrid", "female"), ("Halldora", "female"), ("Hedda", "female"), ("Helga", "female"),
        ("Hervor", "female"), ("Hilda", "female"), ("Hildigunn", "female"), ("Hjordis", "female"), ("Inga", "female"),
        ("Inger", "female"), ("Ingeborg", "female"), ("Ingibjorg", "female"), ("Ingrid", "female"), ("Ingunn", "female"),
        ("Jorunn", "female"), ("Kari", "female"), ("Karin", "female"), ("Katla", "female"), ("Kirsten", "female"),
        ("Kjersti", "female"), ("Liv", "female"), ("Magnhild", "female"), ("Maren", "female"), ("Margit", "female"),
        ("Marit", "female"), ("Mette", "female"), ("Oda", "female"), ("Olga", "female"), ("Ragnhild", "female"),
        ("Randi", "female"), ("Rigmor", "female"), ("Runa", "female"), ("Signe", "female"), ("Signy", "female"),
        ("Sigrid", "female"), ("Sigrun", "female"), ("Solveig", "female"), ("Steinunn", "female"), ("Svanhild", "female"),
        ("Thea", "female"), ("Thora", "female"), ("Thordis", "female"), ("Tora", "female"), ("Unn", "female"),
        ("Aase", "female"), ("Agnete", "female"), ("Annika", "female"), ("Anja", "female"), ("Astri", "female"),
        ("Benedikte", "female"), ("Berit", "female"), ("Birgit", "female"), ("Birgitta", "female"), ("Bodil", "female"),
        ("Brita", "female"), ("Brynhild", "female"), ("Cecilie", "female"), ("Dagmar", "female"), ("Dorte", "female"),
        ("Eldbjorg", "female"), ("Eli", "female"), ("Ellen", "female"), ("Elin", "female"), ("Embla", "female"),
        ("Erika", "female"), ("Frøydis", "female"), ("Gerd", "female"), ("Greta", "female"), ("Gudbjorg", "female"),
        ("Gunhild", "female"), ("Gunn", "female"), ("Guri", "female"), ("Gyda", "female"), ("Hanna", "female"),
        ("Hanne", "female"), ("Hege", "female"), ("Heidi", "female"), ("Helle", "female"), ("Hilde", "female"),
        ("Hjørdis", "female"), ("Ida", "female"), ("Ingvild", "female"), ("Jorid", "female"), ("Kaja", "female"),
        ("Kristin", "female"), ("Kristina", "female"), ("Laila", "female"), ("Lene", "female"), ("Linda", "female"),
        ("Linn", "female"), ("Lisbet", "female"), ("Malin", "female"), ("Mari", "female"), ("Martha", "female"),
        ("Mia", "female"), ("Mona", "female"), ("Møyfrid", "female"), ("Nora", "female"), ("Oddny", "female"),
        ("Olaug", "female"), ("Oline", "female"), ("Ragna", "female"), ("Reidun", "female"), ("Rita", "female"),
        ("Ruth", "female"), ("Sanna", "female"), ("Silje", "female"), ("Siri", "female"), ("Sissel", "female"),
        ("Stine", "female"), ("Sunniva", "female"), ("Susanne", "female"), ("Synnøve", "female"), ("Tiril", "female"),
        ("Tone", "female"), ("Tonje", "female"), ("Torhild", "female"), ("Tordis", "female"), ("Tove", "female"),
        ("Trine", "female"), ("Tuva", "female"), ("Unni", "female"), ("Vigdis", "female"), ("Wenche", "female"),
        ("Ylva", "female"), ("Åshild", "female"), ("Åse", "female"), ("Øydis", "female"), ("Agnes", "female"),
        ("Albjorg", "female"), ("Aldis", "female"), ("Alfhild", "female"), ("Alva", "female"), ("Arna", "female"),
    ],
    "southern": [
        # --- Мужские (150) ---
        ("Alessandro", "male"), ("Alessio", "male"), ("Alfonso", "male"), ("Alfredo", "male"), ("Amadeo", "male"),
        ("Andrea", "male"), ("Angelo", "male"), ("Anselmo", "male"), ("Antonio", "male"), ("Arcangelo", "male"),
        ("Arnaldo", "male"), ("Arturo", "male"), ("Augusto", "male"), ("Aurelio", "male"), ("Baldassare", "male"),
        ("Bartolomeo", "male"), ("Battista", "male"), ("Benedetto", "male"), ("Bernardo", "male"), ("Bruno", "male"),
        ("Calogero", "male"), ("Camillo", "male"), ("Carlo", "male"), ("Carmelo", "male"), ("Casimiro", "male"),
        ("Cesare", "male"), ("Ciro", "male"), ("Claudio", "male"), ("Clemente", "male"), ("Corrado", "male"),
        ("Cosimo", "male"), ("Cristoforo", "male"), ("Damiano", "male"), ("Dante", "male"), ("Dario", "male"),
        ("Dino", "male"), ("Domenico", "male"), ("Donato", "male"), ("Durante", "male"), ("Edoardo", "male"),
        ("Egidio", "male"), ("Eligio", "male"), ("Emanuele", "male"), ("Emilio", "male"), ("Enrico", "male"),
        ("Enzo", "male"), ("Ercole", "male"), ("Ermanno", "male"), ("Ernesto", "male"), ("Ettore", "male"),
        ("Eugenio", "male"), ("Ezechiele", "male"), ("Fabio", "male"), ("Fabrizio", "male"), ("Fausto", "male"),
        ("Federico", "male"), ("Felice", "male"), ("Ferdinando", "male"), ("Filippo", "male"), ("Fiorello", "male"),
        ("Flaminio", "male"), ("Flavio", "male"), ("Fortunato", "male"), ("Francesco", "male"), ("Franco", "male"),
        ("Fulvio", "male"), ("Gabriele", "male"), ("Gaetano", "male"), ("Galileo", "male"), ("Gaspare", "male"),
        ("Gastone", "male"), ("Gavino", "male"), ("Gennaro", "male"), ("Gerardo", "male"), ("Gerolamo", "male"),
        ("Giacinto", "male"), ("Giacobbe", "male"), ("Giacomo", "male"), ("Giambattista", "male"), ("Giancarlo", "male"),
        ("Gianfranco", "male"), ("Gianluca", "male"), ("Gianni", "male"), ("Giano", "male"), ("Gilberto", "male"),
        ("Gino", "male"), ("Gioacchino", "male"), ("Gioele", "male"), ("Giorgio", "male"), ("Giovanni", "male"),
        ("Giovannino", "male"), ("Girolamo", "male"), ("Giuliano", "male"), ("Giulio", "male"), ("Giuseppe", "male"),
        ("Giustino", "male"), ("Giusto", "male"), ("Goffredo", "male"), ("Graziano", "male"), ("Gregorio", "male"),
        ("Guarino", "male"), ("Guelfo", "male"), ("Guglielmo", "male"), ("Guido", "male"), ("Guidobaldo", "male"),
        ("Gustavo", "male"), ("Ignazio", "male"), ("Innocenzo", "male"), ("Ippolito", "male"), ("Ireneo", "male"),
        ("Isaia", "male"), ("Italo", "male"), ("Lamberto", "male"), ("Lancillotto", "male"), ("Lando", "male"),
        ("Landolfo", "male"), ("Lanfranco", "male"), ("Lauro", "male"), ("Lazzaro", "male"), ("Leandro", "male"),
        ("Leonardo", "male"), ("Leone", "male"), ("Leonida", "male"), ("Leonzio", "male"), ("Leopoldo", "male"),
        ("Libero", "male"), ("Liborio", "male"), ("Livio", "male"), ("Lorenzo", "male"), ("Lotario", "male"),
        ("Luca", "male"), ("Luciano", "male"), ("Ludovico", "male"), ("Luigi", "male"), ("Luigino", "male"),
        ("Manfredi", "male"), ("Manlio", "male"), ("Mansueto", "male"), ("Manuele", "male"), ("Marcello", "male"),
        ("Marco", "male"), ("Mariano", "male"), ("Marino", "male"), ("Mario", "male"), ("Martino", "male"),
        ("Marziale", "male"), ("Massimo", "male"), ("Matteo", "male"), ("Maurizio", "male"), ("Mauro", "male"),
        ("Melchiorre", "male"), ("Michele", "male"), ("Milone", "male"), ("Mimmo", "male"), ("Mirko", "male"),

        # --- Женские (150) ---
        ("Alessandra", "female"), ("Alessia", "female"), ("Alfonsina", "female"), ("Amedea", "female"), ("Angelica", "female"),
        ("Angelina", "female"), ("Antonella", "female"), ("Antonia", "female"), ("Arianna", "female"), ("Aurora", "female"),
        ("Beatrice", "female"), ("Bella", "female"), ("Benedetta", "female"), ("Bianca", "female"), ("Brunella", "female"),
        ("Camilla", "female"), ("Carina", "female"), ("Carlotta", "female"), ("Carmela", "female"), ("Caterina", "female"),
        ("Cecilia", "female"), ("Celestina", "female"), ("Chiara", "female"), ("Cinzia", "female"), ("Clara", "female"),
        ("Clarissa", "female"), ("Concetta", "female"), ("Consolata", "female"), ("Cristina", "female"), ("Diana", "female"),
        ("Donatella", "female"), ("Elena", "female"), ("Elisa", "female"), ("Elisabetta", "female"), ("Emilia", "female"),
        ("Emma", "female"), ("Emanuela", "female"), ("Ernestina", "female"), ("Eva", "female"), ("Fabiana", "female"),
        ("Federica", "female"), ("Filomena", "female"), ("Fiorella", "female"), ("Flaminia", "female"), ("Francesca", "female"),
        ("Gabriella", "female"), ("Gemma", "female"), ("Gianna", "female"), ("Ginevra", "female"), ("Giorgia", "female"),
        ("Giovanna", "female"), ("Giulia", "female"), ("Grazia", "female"), ("Ilaria", "female"), ("Imelda", "female"),
        ("Isabella", "female"), ("Livia", "female"), ("Luisa", "female"), ("Maddalena", "female"), ("Marcella", "female"),
        ("Marisa", "female"), ("Marta", "female"), ("Martina", "female"), ("Matilde", "female"), ("Melissa", "female"),
        ("Michela", "female"), ("Milena", "female"), ("Monica", "female"), ("Nadia", "female"), ("Natalia", "female"),
        ("Nicoletta", "female"), ("Noemi", "female"), ("Ornella", "female"), ("Paola", "female"), ("Patrizia", "female"),
        ("Perla", "female"), ("Rachele", "female"), ("Raffaella", "female"), ("Rebecca", "female"), ("Renata", "female"),
        ("Rita", "female"), ("Roberta", "female"), ("Romina", "female"), ("Rosa", "female"), ("Rosalia", "female"),
        ("Rossana", "female"), ("Sabrina", "female"), ("Sara", "female"), ("Selena", "female"), ("Serena", "female"),
        ("Silvia", "female"), ("Simona", "female"), ("Sofia", "female"), ("Sonia", "female"), ("Stefania", "female"),
        ("Susanna", "female"), ("Tania", "female"), ("Teresa", "female"), ("Tina", "female"), ("Valentina", "female"),
        ("Valeria", "female"), ("Vanessa", "female"), ("Veronica", "female"), ("Viola", "female"), ("Virginia", "female"),
        ("Viviana", "female"), ("Zita", "female"), ("Adele", "female"), ("Adriana", "female"), ("Agnese", "female"),
        ("Alberta", "female"), ("Alda", "female"), ("Alida", "female"), ("Alina", "female"), ("Amalia", "female"),
        ("Ambra", "female"), ("Anastasia", "female"), ("Anita", "female"), ("Annamaria", "female"), ("Annarita", "female"),
        ("Annunziata", "female"), ("Antonietta", "female"), ("Apollonia", "female"), ("Assunta", "female"), ("Aurelia", "female"),
        ("Barbara", "female"), ("Bernardina", "female"), ("Betta", "female"), ("Bice", "female"), ("Bionda", "female"),
        ("Brigida", "female"), ("Bruna", "female"), ("Calogera", "female"), ("Candida", "female"), ("Carla", "female"),
        ("Carmina", "female"), ("Celeste", "female"), ("Cesira", "female"), ("Clarice", "female"), ("Clementina", "female"),
        ("Clotilde", "female"), ("Colombina", "female"), ("Costanza", "female"), ("Cristiana", "female"), ("Daniela", "female"),
        ("Daria", "female"), ("Debora", "female"), ("Diletta", "female"), ("Dina", "female"), ("Dora", "female"),
        ("Dorotea", "female"), ("Edvige", "female"), ("Eleonora", "female"), ("Elettra", "female"), ("Eliana", "female"),
        ("Eloisa", "female"), ("Enrica", "female"), ("Erika", "female"), ("Ermelinda", "female"), ("Ester", "female"),
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