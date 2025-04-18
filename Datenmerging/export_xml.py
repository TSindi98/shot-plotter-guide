import sqlite3
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from xml.dom import minidom  # Für bessere Formatierung

# 🎯 Spielname & Speicherorte
spiel_name = "bvb_vs_bayern"
output_folder = f"/Users/mac1kooperationsprojektrub-bvb/Desktop/PycharmProjects 1/Eventdaten whoscored/data/{spiel_name}/"

# 📂 Datenbankpfad
db_path = "/Users/mac1kooperationsprojektrub-bvb/Desktop/PycharmProjects 1/Eventdaten whoscored/fussball_events.db"

# 🎯 Gewünschte Spiel-ID
spiel_id = 5  # ⚠️ Hier die `spielId` des gewünschten Spiels setzen

# 🏟️ Verbindung zur SQLite-Datenbank
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 🗂️ Mapping für satisfiedEventsTypes (aus HTML-Referenz)
satisfied_events_mapping = {
    0: "shotSixYardBox",
    1: "shotPenaltyArea",
    2: "shotOboxTotal",
    3: "shotOpenPlay",
    4: "shotCounter",
    5: "shotSetPiece",
    6: "shotDirectCorner",
    7: "shotOffTarget",
    8: "shotOnPost",
    9: "shotOnTarget",
    10: "shotsTotal",
    11: "shotBlocked",
    12: "shotRightFoot",
    13: "shotLeftFoot",
    14: "shotHead",
    15: "shotObp",
    16: "goalSixYardBox",
    17: "goalPenaltyArea",
    18: "goalObox",
    19: "goalOpenPlay",
    20: "goalCounter",
    21: "goalSetPiece",
    22: "penaltyScored",
    23: "goalOwn",
    24: "goalNormal",
    25: "goalRightFoot",
    26: "goalLeftFoot",
    27: "goalHead",
    28: "goalObp",
    29: "shortPassInaccurate",
    30: "shortPassAccurate",
    31: "passCorner",
    32: "passCornerAccurate",
    33: "passCornerInaccurate",
    34: "passFreekick",
    35: "passBack",
    36: "passForward",
    37: "passLeft",
    38: "passRight",
    39: "keyPassLong",
    40: "keyPassShort",
    41: "keyPassCross",
    42: "keyPassCorner",
    43: "keyPassThroughball",
    44: "keyPassFreekick",
    45: "keyPassThrowin",
    46: "keyPassOther",
    47: "assistCross",
    48: "assistCorner",
    49: "assistThroughball",
    50: "assistFreekick",
    51: "assistThrowin",
    52: "assistOther",
    53: "dribbleLost",
    54: "dribbleWon",
    55: "challengeLost",
    56: "interceptionWon",
    57: "clearanceHead",
    58: "outfielderBlock",
    59: "passCrossBlockedDefensive",
    60: "outfielderBlockedPass",
    61: "offsideGiven",
    62: "offsideProvoked",
    63: "foulGiven",
    64: "foulCommitted",
    65: "yellowCard",
    66: "voidYellowCard",
    67: "secondYellow",
    68: "redCard",
    69: "turnover",
    70: "dispossessed",
    71: "saveLowLeft",
    72: "saveHighLeft",
    73: "saveLowCentre",
    74: "saveHighCentre",
    75: "saveLowRight",
    76: "saveHighRight",
    77: "saveHands",
    78: "saveFeet",
    79: "saveObp",
    80: "saveSixYardBox",
    81: "savePenaltyArea",
    82: "saveObox",
    83: "keeperDivingSave",
    84: "standingSave",
    85: "closeMissHigh",
    86: "closeMissHighLeft",
    87: "closeMissHighRight",
    88: "closeMissLeft",
    89: "closeMissRight",
    90: "shotOffTargetInsideBox",
    91: "touches",
    92: "assist",
    93: "ballRecovery",
    94: "clearanceEffective",
    95: "clearanceTotal",
    96: "clearanceOffTheLine",
    97: "dribbleLastman",
    98: "errorLeadsToGoal",
    99: "errorLeadsToShot",
    100: "intentionalAssist",
    101: "interceptionAll",
    102: "interceptionInTheBox",
    103: "keeperClaimHighLost",
    104: "keeperClaimHighWon",
    105: "keeperClaimLost",
    106: "keeperClaimWon",
    107: "keeperOneToOneWon",
    108: "parriedDanger",
    109: "parriedSafe",
    110: "collected",
    111: "keeperPenaltySaved",
    112: "keeperSaveInTheBox",
    113: "keeperSaveTotal",
    114: "keeperSmother",
    115: "keeperSweeperLost",
    116: "keeperMissed",
    117: "passAccurate",
    118: "passBackZoneInaccurate",
    119: "passForwardZoneAccurate",
    120: "passInaccurate",
    121: "passAccuracy",
    122: "cornerAwarded",
    123: "passKey",
    124: "passChipped",
    125: "passCrossAccurate",
    126: "passCrossInaccurate",
    127: "passLongBallAccurate",
    128: "passLongBallInaccurate",
    129: "passThroughBallAccurate",
    130: "passThroughBallInaccurate",
    131: "passThroughBallInacurate",
    132: "passFreekickAccurate",
    133: "passFreekickInaccurate",
    134: "penaltyConceded",
    135: "penaltyMissed",
    136: "penaltyWon",
    137: "passRightFoot",
    138: "passLeftFoot",
    139: "passHead",
    140: "sixYardBlock",
    141: "tackleLastMan",
    142: "tackleLost",
    143: "tackleWon",
    144: "cleanSheetGK",
    145: "cleanSheetDL",
    146: "cleanSheetDC",
    147: "cleanSheetDR",
    148: "cleanSheetDML",
    149: "cleanSheetDMC",
    150: "cleanSheetDMR",
    151: "cleanSheetML",
    152: "cleanSheetMC",
    153: "cleanSheetMR",
    154: "cleanSheetAML",
    155: "cleanSheetAMC",
    156: "cleanSheetAMR",
    157: "cleanSheetFWL",
    158: "cleanSheetFW",
    159: "cleanSheetFWR",
    160: "cleanSheetSub",
    161: "goalConcededByTeamGK",
    162: "goalConcededByTeamDL",
    163: "goalConcededByTeamDC",
    164: "goalConcededByTeamDR",
    165: "goalConcededByTeamDML",
    166: "goalConcededByTeamDMC",
    167: "goalConcededByTeamDMR",
    168: "goalConcededByTeamML",
    169: "goalConcededByTeamMC",
    170: "goalConcededByTeamMR",
    171: "goalConcededByTeamAML",
    172: "goalConcededByTeamAMC",
    173: "goalConcededByTeamAMR",
    174: "goalConcededByTeamFWL",
    175: "goalConcededByTeamFW",
    176: "goalConcededByTeamFWR",
    177: "goalConcededByTeamSub",
    178: "goalConcededOutsideBoxGoalkeeper",
    179: "goalScoredByTeamGK",
    180: "goalScoredByTeamDL",
    181: "goalScoredByTeamDC",
    182: "goalScoredByTeamDR",
    183: "goalScoredByTeamDML",
    184: "goalScoredByTeamDMC",
    185: "goalScoredByTeamDMR",
    186: "goalScoredByTeamML",
    187: "goalScoredByTeamMC",
    188: "goalScoredByTeamMR",
    189: "goalScoredByTeamAML",
    190: "goalScoredByTeamAMC",
    191: "goalScoredByTeamAMR",
    192: "goalScoredByTeamFWL",
    193: "goalScoredByTeamFW",
    194: "goalScoredByTeamFWR",
    195: "goalScoredByTeamSub",
    196: "aerialSuccess",
    197: "duelAerialWon",
    198: "duelAerialLost",
    199: "offensiveDuel",
    200: "defensiveDuel",
    201: "bigChanceMissed",
    202: "bigChanceScored",
    203: "bigChanceCreated",
    204: "overrun",
    205: "successfulFinalThirdPasses",
    206: "punches",
    207: "penaltyShootoutScored",
    208: "penaltyShootoutMissedOffTarget",
    209: "penaltyShootoutSaved",
    210: "penaltyShootoutSavedGK",
    211: "penaltyShootoutConcededGK",
    212: "throwIn",
    213: "subOn",
    214: "subOff",
    215: "defensiveThird",
    216: "midThird",
    217: "finalThird",
    218: "pos",
}

# 📜 Funktion zur Erstellung einer XML-Datei für eine Halbzeit
def create_xml(period, filename):
    root = ET.Element("file")

    # 📌 Sitzungs-Info
    session_info = ET.SubElement(root, "SESSION_INFO")
    start_time = ET.SubElement(session_info, "start_time")
    start_time.text = datetime.now().strftime("%Y-%m-%d %H:%M:%S.000000+0200")

    # 📂 Alle Instanzen (Events)
    all_instances = ET.SubElement(root, "ALL_INSTANCES")

    print(f"Debug: Abfrage mit spiel_id = {spiel_id} und period = {period}")

    # 📊 SQL-Abfrage: Events abrufen (inkl. `satisfiedEventsTypes`)
    cursor.execute("""
        SELECT 
            S.spielName, 
            E.period, 
            E.minute, 
            COALESCE(E.second, 0),  
            E.teamId, 
            COALESCE(P.name, 'Unbekannter Spieler'),  
            E.type, 
            E.outcome, 
            E.x, E.y, E.endX, E.endY,
            E.satisfiedEventsTypes
        FROM Events E
        JOIN Spiele S ON E.spielId = S.spielId
        LEFT JOIN Players P ON E.playerId = P.playerId  
        WHERE S.spielId = ? AND E.period = ?
        ORDER BY S.spielName, E.period, E.minute, E.second;
    """, (spiel_id, period))

    events = cursor.fetchall()

    print(f"Debug: Anzahl gefundener Events = {len(events)}")

    for event in events:
        instance = ET.SubElement(all_instances, "instance")

        # 🆕 ID
        ET.SubElement(instance, "ID").text = str(event[0])

        # Zeitstempel
        start_sec = event[2] * 60 + event[3] - 5  # -5s Sicherheit
        end_sec = start_sec + 10  # +10s Länge

        ET.SubElement(instance, "start").text = str(start_sec)
        ET.SubElement(instance, "end").text = str(end_sec)

        # 🆕 Spielername als Code
        ET.SubElement(instance, "code").text = event[5]  # Spielername

        # 🆕 Haupt-Event als Label
        main_label = ET.SubElement(instance, "label")
        ET.SubElement(main_label, "text").text = f"{event[6]} ({event[7]})"  # Event-Typ + Outcome

        # 🆕 satisfiedEventsTypes als eigene Labels hinzufügen
        if event[12]:
            for event_id in event[12].split(","):
                event_id = int(event_id.strip())
                event_name = satisfied_events_mapping.get(event_id, f"Unknown ({event_id})")
                event_label = ET.SubElement(instance, "label")  # Jedes Label separat speichern
                ET.SubElement(event_label, "text").text = event_name

        # 🆕 Positionsdaten als eigenes Label hinzufügen
        position_label = ET.SubElement(instance, "label")
        ET.SubElement(position_label, "text").text = f"Pos: ({event[8]}, {event[9]}) → ({event[10]}, {event[11]})"

    # 🆕 XML hübsch formatieren & speichern
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")

    # 📂 Sicherstellen, dass der Speicherordner existiert
    os.makedirs(output_folder, exist_ok=True)

    with open(os.path.join(output_folder, filename), "w", encoding="utf-8") as f:
        f.write(xml_str)


# 🎯 Zwei separate XML-Dateien für die Halbzeiten erstellen
create_xml(1, "export_synergy_first_half.xml")  # 1. Halbzeit
create_xml(2, "export_synergy_second_half.xml")  # 2. Halbzeit

print(f"✅ XML-Dateien für 1. und 2. Halbzeit wurden in {output_folder} gespeichert!")

# 💾 Verbindung schließen
conn.close()