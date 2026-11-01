
"""Расширение seed-датасета новыми именами (вторая волна).

Идемпотентен: повторный запуск не создаёт дубликатов.
Запуск:
    python -m ml.expand_dataset
"""
import json
from pathlib import Path

from ml.preprocess import normalize
from ml.transliterate import transliterate

# ВТОРАЯ ВОЛНА: ~600 имён, ориентированных на разнообразие и качество.
NEW_NAMES = {
    "elven": [
        "Aerandir", "Aerion", "Aerith", "Aethon", "Aglarond", "Ainar",
        "Airendil", "Alassë", "Alcarin", "Aldarion", "Aldaron", "Amaranth",
        "Amdir", "Amras", "Anarion", "Ancalimon", "Andúril", "Anguirel",
        "Annael", "Aranel", "Aranrúth", "Arantar", "Aratan", "Arciryas",
        "Ardamin", "Aredhel", "Argon", "Argonui", "Arien", "Arminas",
        "Artaher", "Artamir", "Artanis", "Arvedui", "Arvegil", "Arveleg",
        "Arwen", "Atanalcar", "Atanatar", "Aulendil", "Avranc", "Axantur",
        "Beleth", "Bregalad", "Bregolas", "Bregor", "Calion", "Calmacil",
        "Caranthir", "Castamir", "Celeborn", "Celebrían", "Celebrimbor",
        "Celebrindor", "Celegorm", "Cemendur", "Círdan", "Ciryon",
        "Culúrien", "Curufin", "Daeron", "Dagmor", "Dagnir", "Dairuin",
        "Damrod", "Denethor", "Dior", "Dírhael", "Dorlas", "Draugluin",
        "Duilin", "Duinhir", "Eärendil", "Eärendur", "Eäron", "Ecthelion",
        "Edrahil", "Egalmoth", "Eldacar", "Eldalondë", "Eldarion",
        "Eldarion", "Elemmakil", "Elemmírë", "Elendil", "Elendur",
        "Elensar", "Elentirmo", "Elentir", "Elerína", "Elros", "Elrond",
        "Elured", "Elurín", "Elwing", "Enerdhil", "Eöl", "Eönwë",
        "Ereinion", "Erestor", "Erien", "Eruadan", "Eruestan", "Estel",
        "Esteldor", "Ettelion", "Faelivrin", "Falathar", "Falathrim",
        "Faramir", "Fëanor", "Fëanáro", "Findis", "Finduilas", "Finglas",
        "Fingolfin", "Fingon", "Finrod", "Finwë", "Forlong", "Fuinur",
        "Galdor", "Galador", "Galathil", "Galathilion", "Galdor",
        "Galion", "Gandalf", "Gelmir", "Gildor", "Gil-galad", "Gilraen",
        "Glirhuin", "Glorfindel", "Guilin", "Gwindor", "Hador", "Halbarad",
        "Haldad", "Haldan", "Haldar", "Haldir", "Hallacar", "Hallatan",
        "Halmir", "Handir", "Hardang", "Hareth", "Hathaldir", "Hathol",
        "Herion", "Hiril", "Hirluin", "Hundar", "Húrin", "Huor",
        "Idril", "Ilmarë", "Ilmarien", "Imrahil", "Indis", "Ingwë",
        "Ioreth", "Irmo", "Isildur", "Ivorwen", "Ivrin", "Ivriniel",
    ],
    "human": [
        "Abraham", "Adrian", "Alan", "Albert", "Aldrich", "Alexander",
        "Alfred", "Alistair", "Alwin", "Ambrose", "Andrew", "Angus",
        "Anselm", "Anthony", "Archibald", "Arnold", "Arthur", "Aubrey",
        "Augustine", "Austin", "Baldwin", "Barnaby", "Bartholomew",
        "Basil", "Benedict", "Benjamin", "Bernard", "Bertram", "Bertrand",
        "Boniface", "Brian", "Bruce", "Caleb", "Calvin", "Carlisle",
        "Casimir", "Cedric", "Charles", "Chester", "Christopher",
        "Clarence", "Claude", "Clement", "Clifford", "Clive", "Conrad",
        "Constantine", "Cornelius", "Crispin", "Cuthbert", "Cyril",
        "Damian", "Daniel", "Darius", "David", "Dennis", "Desmond",
        "Dominic", "Donald", "Dorian", "Douglas", "Duncan", "Edgar",
        "Edmund", "Edward", "Edwin", "Egbert", "Elias", "Elliot", "Eric",
        "Ernest", "Ethan", "Eugene", "Eustace", "Everard", "Ezra",
        "Fabian", "Felix", "Ferdinand", "Francis", "Frank", "Franklin",
        "Frederick", "Gabriel", "Gareth", "Geoffrey", "George", "Gerald",
        "Gerard", "Gilbert", "Giles", "Godfrey", "Gordon", "Graham",
        "Gregory", "Guy", "Harold", "Harvey", "Henry", "Herbert",
        "Herman", "Hilary", "Horace", "Howard", "Hugh", "Humphrey",
        "Ian", "Isaac", "Ivor", "Jacob", "James", "Jasper", "Jeremy",
        "Jerome", "John", "Jonathan", "Joseph", "Julian", "Kenneth",
        "Lambert", "Laurence", "Lawrence", "Leonard", "Leopold",
        "Lionel", "Louis", "Lucas", "Lucian", "Magnus", "Malcolm",
        "Marcus", "Martin", "Matthew", "Maurice", "Michael", "Miles",
        "Nathaniel", "Nicholas", "Nigel", "Norman", "Oliver", "Osbert",
        "Oswald", "Owen", "Patrick", "Paul", "Percival", "Peter",
        "Philip", "Quentin", "Ralph", "Raymond", "Reginald", "Reynold",
        "Richard", "Robert", "Roger", "Roland", "Ronald", "Rowan",
        "Rudolph", "Samuel", "Sebastian", "Sigismund", "Silas", "Simon",
        "Stephen", "Sylvester", "Theodore", "Thomas", "Timothy",
        "Tobias", "Trevor", "Valentine", "Victor", "Vincent", "Walter",
        "William", "Winfred",
    ],
    "orcish": [
        "Azog", "Bagronk", "Balcmeg", "Bolg", "Golfimbul", "Gorbag",
        "Gorgoroth", "Gothmog", "Grishnákh", "Grisnakh", "Grokh",
        "Gromgul", "Grubash", "Grukhal", "Gundabad", "Kragga", "Kragnak",
        "Kruglak", "Krugman", "Krugsh", "Lugbag", "Lugdush", "Luglur",
        "Lugnash", "Lurtz", "Magg", "Maglub", "Magrat", "Mauhúr",
        "Mazgak", "Mogdush", "Mograt", "Mogrul", "Muzgash", "Nazgul",
        "Nazgrat", "Nekros", "Ogdush", "Ogg", "Ogruk", "Ogrukh", "Olog",
        "Orcobal", "Othrod", "Radbug", "Raglug", "Ratbag", "Ratgul",
        "Ratlug", "Ratt", "Shagrat", "Shagruk", "Sharku", "Sharlug",
        "Shrag", "Shragrat", "Shraklug", "Shuk", "Snaga", "Snagrat",
        "Snagrat", "Snalg", "Snaulug", "Snaur", "Snaurg", "Snaurlug",
        "Snaurrat", "Snav", "Snavrat", "Snul", "Snurg", "Snurrat",
        "Thrag", "Thrak", "Thraklug", "Thrarat", "Thrat", "Thratlug",
        "Thratrat", "Thratrug", "Throt", "Throtlug", "Throtrug",
        "Thrug", "Thrugak", "Thrugluk", "Thrugram", "Thrugrat",
        "Thrugruk", "Thruklug", "Thrukrat", "Thrul", "Thrulug",
        "Thrurat", "Thrurlug", "Ufthak", "Ugbash", "Ugluk", "Ugmog",
        "Ugnash", "Ugrat", "Ugrul", "Ulk", "Ulog", "Uluk", "Uluksh",
        "Urmak", "Uruk", "Urukdug", "Urukful", "Urukgul", "Urukmog",
        "Uruknar", "Urukrat", "Urukrug", "Urukshak", "Urukthak",
        "Urukthar", "Urukthrak", "Urukthrat", "Uruktur",
    ],
    "northern": [
        "Agnar", "Aki", "Alf", "Algot", "Alvar", "Andor", "Anders",
        "Andreas", "Anund", "Arne", "Arnfast", "Arvid", "Asbjorn",
        "Asgeir", "Aslak", "Asmund", "Asvald", "Balder", "Bard", "Beinir",
        "Bendik", "Bengt", "Birger", "Bjarne", "Bjarni", "Bjorn",
        "Bodvar", "Bolli", "Brand", "Brandr", "Broddi", "Brynjolf",
        "Bui", "Dag", "Dagfinn", "Dagny", "Dufniall", "Egil", "Eilif",
        "Einar", "Eirik", "Eldar", "Eli", "Elias", "Endre", "Enok",
        "Erland", "Erlend", "Eskil", "Espen", "Eyjolf", "Eystein",
        "Eyvind", "Farman", "Filip", "Finn", "Finnbogi", "Flemming",
        "Folke", "Fredrik", "Frode", "Frodi", "Galti", "Gamli", "Gardar",
        "Gauti", "Geir", "Geirmund", "Gest", "Gisle", "Gnupa", "Gorm",
        "Grim", "Grimkel", "Gudbrand", "Gudmund", "Gunnar", "Gunnbjorn",
        "Gunvald", "Gustav", "Guttorm", "Håkon", "Halfdan", "Hallbjorn",
        "Halldor", "Hallgrim", "Hallvard", "Halvar", "Hamund", "Hans",
        "Hårek", "Harald", "Havard", "Hedin", "Helge", "Heming",
        "Hemming", "Henrik", "Herjolf", "Herman", "Hermund", "Hjalmar",
        "Hjalti", "Hjort", "Hlodver", "Hrafn", "Hrafnkell", "Hrani",
        "Hrapp", "Hrolf", "Hroar", "Hrut", "Hugin", "Hunbogi", "Hundi",
        "Hvitserk", "Igul", "Illugi", "Ingemar", "Ingi", "Ingjald",
        "Ingolf", "Ingvar", "Ivar", "Jakob", "Jens", "Jesper", "Joakim",
        "Johan", "Jokul", "Jon", "Jorgen", "Kalv", "Kar", "Kari",
        "Karl", "Karle", "Ketil", "Kettil", "Kjartan", "Kjell",
        "Knut", "Kolbein", "Kolbjorn", "Koll", "Kori", "Kormak",
        "Kristian", "Lars", "Leif", "Leidolf", "Lennart", "Lifolf",
        "Linus", "Ljot", "Magnus", "Mani", "Mats", "Meldun", "Mikael",
        "Mord", "Munan", "Nefjolf", "Niels", "Nils", "Njall", "Odd",
        "Oddbjorn", "Oddi", "Ofeig", "Ogmund", "Olaf", "Olav", "Olle",
        "Onund", "Orm", "Ormr", "Oskar", "Osmund", "Ottar", "Pall",
    ],
    "southern": [
        "Abelardo", "Adalberto", "Adolfo", "Adriano", "Agostino",
        "Alberto", "Alessandro", "Alessio", "Alfonso", "Alfredo",
        "Alonso", "Alonzo", "Álvaro", "Amadeo", "Amedeo", "Anacleto",
        "Anastasio", "Andrea", "Andrés", "Angelino", "Angelo", "Anselmo",
        "Antonino", "Antonio", "Arcangelo", "Arduino", "Ariel", "Armando",
        "Arnaldo", "Arnoldo", "Arturo", "Attilio", "Augusto", "Aurelio",
        "Baldassare", "Baldovino", "Battista", "Beppe", "Bernardo",
        "Bernardino", "Bettino", "Bindo", "Bonaventura", "Bonifacio",
        "Bruno", "Cacciaguida", "Calogero", "Calvino", "Camillo",
        "Carlo", "Carmelo", "Casimiro", "Cassio", "Cecco", "Celestino",
        "Cesare", "Cipriano", "Ciro", "Claudio", "Clemente", "Colombano",
        "Corrado", "Cosimo", "Costantino", "Cristoforo", "Dalmazio",
        "Damiano", "Danilo", "Dante", "Dario", "Delfino", "Dino",
        "Domenico", "Donato", "Doroteo", "Durante", "Edoardo", "Egidio",
        "Eligio", "Elio", "Emanuele", "Emiliano", "Emilio", "Enea",
        "Enrico", "Enzo", "Ercole", "Ermanno", "Ernesto", "Ettore",
        "Eugenio", "Eusebio", "Ezechiele", "Fabio", "Fabrizio",
        "Faustino", "Fausto", "Federico", "Felice", "Ferdinando",
        "Ferruccio", "Filiberto", "Filippo", "Fiorello", "Fiorenzo",
        "Flaminio", "Flavio", "Fortunato", "Francesco", "Franco",
        "Fulgenzio", "Fulvio", "Gabriele", "Gaetano", "Galileo",
        "Gaspare", "Gastone", "Gavino", "Gennaro", "Gerardo", "Gerolamo",
        "Giacinto", "Giacobbe", "Giacomo", "Giambattista", "Giancarlo",
        "Gianfranco", "Gianluca", "Gianni", "Giano", "Gilberto",
        "Gino", "Gioacchino", "Gioele", "Giorgio", "Giovanni", "Giovannino",
        "Girolamo", "Giuliano", "Giulio", "Giuseppe", "Giustino",
        "Giusto", "Goffredo", "Graziano", "Gregorio", "Guarino", "Guelfo",
        "Guglielmo", "Guido", "Guidobaldo", "Gustavo", "Ignazio",
        "Innocenzo", "Ippolito", "Ireneo", "Isaia", "Italo", "Lamberto",
        "Lancillotto", "Lando", "Landolfo", "Lanfranco", "Lauro",
        "Lazzaro", "Leandro", "Leonardo", "Leone", "Leonida",
        "Leonzio", "Leopoldo", "Libero", "Liborio", "Livio", "Lorenzo",
        "Lotario", "Luca", "Luciano", "Ludovico", "Luigi", "Luigino",
        "Manfredi", "Manlio", "Mansueto", "Manuele", "Marcello",
        "Marcello", "Marco", "Mariano", "Marino", "Mario", "Martino",
        "Marziale", "Massimo", "Matteo", "Maurizio", "Mauro", "Melchiorre",
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

    for style, names in NEW_NAMES.items():
        for latin in names:
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
            result.append({"name": clean, "style": style, "source": "expanded_v2"})
            added += 1

    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Добавлено: {added}, пропущено: {skipped}, всего: {len(result)}")


if __name__ == "__main__":
    expand()