import { cfgDetails } from "../config-details.js";

function createRadioButtons(selectId, { id, title, options }) {
    d3.select(selectId)
        .append("div")
        .attr("class", cfgDetails.detailClass)
        .attr("id", id)
        .append("h3")
        .text(title)
        .attr("class", "center");

    for (let option of options) {
        let div = d3
            .select("#" + id)
            .append("div")
            .attr("class", "form-check vertical");

        div.append("input")
            .attr("class", "form-check-input")
            .attr("type", "radio")
            .attr("name", id)
            .attr("id", option.value) // sanitize, make sure no duplicate values
            .attr("value", option.value)
            .attr("checked", option.checked);
        div.append("label")
            .attr("class", "form-check-label")
            .attr("for", option.value)
            .text(option.value);
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

    options.forEach((option, index) => {
        const optionText = option.shortcut
            ? `${option.value} (${option.shortcut})`
            : option.value;

        select
            .append("option")
            .text(optionText)
            .attr("value", option.value)
            .attr("selected", option.selected);

        console.log(`Option ${index}:`, option);

        if (option.shortcut) {
            console.log(`  → Registering shortcut: ${option.shortcut} for value: ${option.value}`);
            shortcutToDropdown.set(option.shortcut, {
                selectId: `${id}-select`,
                value: option.value,
            });
        } else {
            console.warn(`  ⚠️ Option ${index} has no shortcut`);
        }
    });
}

// Event-Listener für Tastaturbefehle
document.addEventListener('keydown', function(event) {
    options.forEach((option) => {
        // Überprüfen, ob der Shortcut gedrückt wurde
        if (event.key === option.shortcut) { // Verwende event.key
            const dropdownId = `#${id}-select`;
            d3.select(dropdownId).property('value', option.value); // Setze den Wert basierend auf dem Shortcut
            console.log(`Setting dropdown value to: ${option.value}`); // Log für Debugging
            
            // Optional: Führe eine Funktion aus, die auf die Auswahl reagiert
            // updateBasedOnSelection(option.value);
        }
    });
});


function createTimeWidget(selectId, { id, title, defaultTime, countdown }) {
    let div = d3
        .select(selectId)
        .append("div")
        .attr("class", cfgDetails.detailClass + " even-width")
        .attr("id", id);
    div.append("h3").text(title).attr("class", "center");

    
    // Füge ein Eingabefeld hinzu, um die Zeit anzuzeigen
    div.append("input")
        .attr("type", "text")
        .attr("readonly", true) // Nur Lesezugriff
        .attr("value", "00:00"); // Standardwert

    const videoElement = document.getElementById('gameVideo'); // Das Video-Element

    // Event-Listener für die Zeitaktualisierung
    videoElement.addEventListener('timeupdate', function() {
        const currentTime = videoElement.currentTime; // Aktueller Zeitstempel des Videos
        const formattedTime = formatTime(currentTime); // Zeit formatieren
        d3.select("#" + id).select("input").property("value", formattedTime); // Aktualisiere das Widget
    });

    // Hilfsfunktion zum Formatieren der Zeit
    function formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${minutes}:${secs < 10 ? '0' : ''}${secs}`; // Format: mm:ss

    let text = div.append("div").attr("class", "time-widget position-relative");
    text.append("input")
        .attr("type", "text")
        .attr("class", "form-control time-box")
        .attr("value", defaultTime);
    text.append("div")
        .attr("class", "invalid-tooltip")
        .text("Times must be in the form 'MM:SS' or 'M:SS'.");
    text.append("div")
        .attr("class", "white-btn time-btn")
        .on("click", function () {
            if (
                d3.select(this).select("i").attr("class") === "bi bi-stop-fill"
            ) {
                timer.stop();
                d3.select(this).select("i").remove();
                d3.select(this).append("i").attr("class", "bi bi-play-fill");
                d3.select("#" + id)
                    .select("input")
                    .attr("disabled", null);
            } else {
                let time = d3
                    .select("#" + id)
                    .select("input")
                    .property("value");
                if (/^\d{1,2}:\d\d$/.test(time)) {
                    d3.select("#" + id)
                        .select("input")
                        .attr("disabled", true)
                        .attr("class", "form-control time-box");
                    d3.select(this).select("i").remove();
                    d3.select(this)
                        .append("i")
                        .attr("class", "bi bi-stop-fill");
                    timer.start(time);
                } else {
                    d3.select("#" + id)
                        .select("input")
                        .attr("class", "form-control time-box is-invalid");
                }
            }
        })
        .append("i")
        .attr("class", "bi bi-play-fill");
    }
}


const shortcutToDropdown = new Map();

document.addEventListener("keydown", (event) => {
    for (const [shortcut, { selectId, value }] of shortcutToDropdown.entries()) {
        if (event.key === shortcut) {
            d3.select(`#${selectId}`).property("value", value).dispatch("change");
        }
    }
});


export {
    createRadioButtons,
    createTextField,
    createDropdown,
    createTimeWidget,
    shortcutToDropdown,
};

