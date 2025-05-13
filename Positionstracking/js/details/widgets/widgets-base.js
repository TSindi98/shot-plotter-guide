import { cfgDetails } from "../config-details.js";

// Globale Map für Shortcuts
const shortcutToDropdown = new Map();

// Hilfsfunktion zum Normalisieren von Shortcuts
function normalizeShortcut(shortcut) {
    return shortcut ? shortcut.toLowerCase().trim() : '';
}

// Hilfsfunktion zum Validieren von Shortcuts
function validateShortcut(shortcut, existingShortcuts = new Set()) {
    if (!shortcut) return true; // Leere Shortcuts sind erlaubt
    const normalized = normalizeShortcut(shortcut);
    return normalized.length > 0 && !existingShortcuts.has(normalized);
}

function createRadioButtons(selectId, { id, title, options }) {
    d3.select(selectId)
        .append("div")
        .attr("class", cfgDetails.detailClass)
        .attr("id", id)
        .append("h3")
        .text(title)
        .attr("class", "center");

    // Sammle existierende Shortcuts
    const existingShortcuts = new Set();
    options.forEach(option => {
        if (option.shortcut) {
            existingShortcuts.add(normalizeShortcut(option.shortcut));
        }
    });

    for (let option of options) {
        let div = d3
            .select("#" + id)
            .append("div")
            .attr("class", "form-check vertical");

        // Erstelle das Radio-Input-Element
        let radioInput = div.append("input")
            .attr("class", "form-check-input")
            .attr("type", "radio")
            .attr("name", id)
            .attr("id", option.value)
            .attr("value", option.value)
            .attr("checked", option.checked);

        // Erstelle das Label mit Shortcut-Anzeige
        let labelText = option.value;
        if (option.shortcut) {
            labelText += ` (${option.shortcut})`;
            // Füge den Shortcut zur Map hinzu
            shortcutToDropdown.set(normalizeShortcut(option.shortcut), {
                selectId: id,
                value: option.value,
                type: "radio"
            });
        }

        // Erstelle das Label-Element
        let label = div.append("label")
            .attr("class", "form-check-label")
            .attr("for", option.value)
            .text(labelText);

        // Füge Event-Listener für Änderungen hinzu
        radioInput.on("change", function() {
            // Aktualisiere die Shortcut-Map wenn sich die Auswahl ändert
            if (option.shortcut) {
                shortcutToDropdown.set(normalizeShortcut(option.shortcut), {
                    selectId: id,
                    value: option.value,
                    type: "radio"
                });
            }
        });
    }
}

function createTextField(
    selectId,
    { id, title, defaultValue, dataTableEditable }
) {
    let div = d3
        .select(selectId)
        .append("div")
        .attr("class", cfgDetails.detailClass + " " + "even-width")
        .attr("id", id);
    div.append("h3").text(title).attr("class", "center");
    div.append("div")
        .attr("class", "form-group")
        .append("input")
        .attr("type", "text")
        .attr("class", "form-control")
        .attr("value", defaultValue);
}

function createDropdown(selectId, { id, title, options }) {
    let div = d3
        .select(selectId)
        .append("div")
        .attr("class", cfgDetails.detailClass + " even-width")
        .attr("id", id);

    div.append("h3").text(title).attr("class", "center");

    let select = div
        .append("div")
        .append("select")
        .attr("id", id + "-select")
        .attr("class", "select2");

    // Sammle existierende Shortcuts
    const existingShortcuts = new Set();
    options.forEach(option => {
        if (option.shortcut) {
            existingShortcuts.add(normalizeShortcut(option.shortcut));
        }
    });

    options.forEach((option) => {
        const optionText = option.shortcut
            ? `${option.value} (${option.shortcut})`
            : option.value;

        select
            .append("option")
            .text(optionText)
            .attr("value", option.value)
            .attr("selected", option.selected);

        if (option.shortcut) {
            shortcutToDropdown.set(normalizeShortcut(option.shortcut), {
                selectId: `${id}-select`,
                value: option.value,
                type: "dropdown"
            });
        }
    });
}

// Modifiziere den Event-Listener für Tastaturbefehle
document.addEventListener("keydown", (event) => {
    // Normalisiere den Tastendruck für bessere Browser-Kompatibilität
    const key = event.key.toLowerCase();
    
    // Ignoriere Tastendrücke wenn ein Eingabefeld fokussiert ist
    if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA') {
        return;
    }
    
    for (const [shortcut, { selectId, value, type }] of shortcutToDropdown.entries()) {
        if (key === shortcut) {
            event.preventDefault(); // Verhindere Standard-Browser-Verhalten
            
            if (type === "radio") {
                // Für Radio Buttons
                const radioInput = d3.select(`#${selectId}`)
                    .select(`input[value="${value}"]`);
                
                if (!radioInput.empty()) {
                    radioInput
                        .property("checked", true)
                        .dispatch("change", { bubbles: true });
                }
            } else {
                // Für Dropdowns
                const dropdown = d3.select(`#${selectId}`);
                if (!dropdown.empty()) {
                    dropdown
                        .property("value", value)
                        .dispatch("change", { bubbles: true });
                }
            }
        }
    }
});

function createTimeWidget(selectId, { id, title, defaultTime, countdown }) {
    let div = d3
        .select(selectId)
        .append("div")
        .attr("class", cfgDetails.detailClass + " even-width")
        .attr("id", id);
    div.append("h3").text(title).attr("class", "center");

    // Standardwert für defaultTime mit Hundertstel
    if (defaultTime && !defaultTime.includes('.')) {
        defaultTime = defaultTime + '.00';
    } else if (!defaultTime) {
        defaultTime = '00:00.00';
    }
    
    // Füge ein Eingabefeld hinzu, um die Zeit anzuzeigen
    div.append("input")
        .attr("type", "text")
        .attr("readonly", true) // Nur Lesezugriff
        .attr("value", "00:00.00"); // Standardwert mit Hundertstel

    const videoElement = document.getElementById('gameVideo'); // Das Video-Element

    // Event-Listener für die Zeitaktualisierung
    videoElement.addEventListener('timeupdate', function() {
        const currentTime = videoElement.currentTime; // Aktueller Zeitstempel des Videos
        const formattedTime = formatTime(currentTime); // Zeit formatieren
        d3.select("#" + id).select("input").property("value", formattedTime); // Aktualisiere das Widget
    });

    // Hilfsfunktion zum Formatieren der Zeit mit Hundertstel
    function formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        const hundredths = Math.floor((seconds % 1) * 100);
        return `${minutes}:${secs < 10 ? '0' : ''}${secs}.${hundredths < 10 ? '0' : ''}${hundredths}`; // Format: mm:ss.hh
    }
}

export {
    createRadioButtons,
    createTextField,
    createDropdown,
    createTimeWidget,
    shortcutToDropdown,
    normalizeShortcut,
    validateShortcut
};

